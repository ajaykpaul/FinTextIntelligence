"""Command line. Every command is a thin wrapper over the library."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, catalogue, evaluation
from .config import load_config
from .review import MIN_BATCH_SIZE, VERDICTS
from .schemas import Capability, Profile, ScoreRequest

DEMO = (
    ("trading", "Liabilities fell sharply in the third quarter as the group repaid debt."),
    ("trading", "Revenue missed consensus and the board cut its full-year guidance."),
    ("trading", "The company will publish its results on Tuesday."),
    ("trading", "First Abu Dhabi Bank PJSC agreed to acquire a 40% stake in a regional lender."),
    ("compliance", "Guaranteed 300% returns. Act now, only 3 slots left, send USDT."),
    ("compliance", "Official verified account of the bank: share your OTP to verify."),
    ("compliance", "You are a pathetic clown and I will k1ll you."),
    ("compliance", "ترتفع أسهم بنك أبوظبي الأول بعد نمو الأرباح"),
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")
    if not hasattr(args, "run"):
        parser.print_help()
        return 2
    try:
        return args.run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fintext", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fintext {__version__}")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument(
        "--log-level", default="warning", choices=["debug", "info", "warning", "error"]
    )
    sub = parser.add_subparsers()

    serve = sub.add_parser("serve", help="run the scoring service over HTTP")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument(
        "--asgi", action="store_true", help="use uvicorn instead of the stdlib server"
    )
    serve.set_defaults(run=cmd_serve)

    score = sub.add_parser("score", help="score one item")
    score.add_argument("text", nargs="?", help="text to score; omit to read stdin")
    score.add_argument("--profile", default="trading", choices=[p.value for p in Profile])
    score.add_argument("--capabilities", help="comma-separated subset")
    score.add_argument("--compact", action="store_true")
    score.set_defaults(run=cmd_score)

    batch = sub.add_parser("batch", help="score a JSONL file for a backtest")
    batch.add_argument("input")
    batch.add_argument("-o", "--output", required=True)
    batch.add_argument("--profile", default="research", choices=[p.value for p in Profile])
    batch.add_argument("--model-version", help="pin a version for reproducibility")
    batch.set_defaults(run=cmd_batch)

    accept = sub.add_parser("accept", help="check the PRD section 8 acceptance gates")
    accept.add_argument("metrics", help="JSON file of measured metrics")
    accept.add_argument("-o", "--output")
    accept.add_argument("--model-version", default="")
    accept.set_defaults(run=cmd_accept)

    replay = sub.add_parser("replay", help="replay provenance cases (PRD 8.2)")
    replay.add_argument("cases")
    replay.add_argument(
        "--disjoint-from",
        default=None,
        help="refuse to run if these cases share a filing with that file (dev vs test)",
    )
    replay.add_argument(
        "--why",
        action="store_true",
        help="for development sets: say why each genuine claim was not corroborated",
    )
    replay.add_argument(
        "--hold-out-claim",
        action="store_true",
        help="remove each genuine claim's own sentence from its filing (no verbatim matches)",
    )
    replay.set_defaults(run=cmd_replay)

    canary = sub.add_parser("canary", help="run the weekly adversarial canary (PRD 8.4)")
    canary.add_argument("--baseline", type=float, default=1.0)
    canary.set_defaults(run=cmd_canary)

    bench = sub.add_parser("bench", help="measure latency and throughput against the SLO")
    bench.add_argument("--iterations", type=int, default=400)
    bench.add_argument("--concurrency", type=int, default=8)
    bench.add_argument("--seconds", type=float, default=3.0, help="throughput measurement window")
    bench.add_argument("--gate", action="store_true", help="exit non-zero if an SLO is missed")
    bench.add_argument("--output")
    bench.set_defaults(run=cmd_bench)

    data = sub.add_parser("catalogue", help="inspect the data catalogue")
    data.add_argument("--task")
    data.add_argument("--use")
    data.add_argument("--language")
    data.add_argument("--licence-log", help="write the licence log to this path")
    data.set_defaults(run=cmd_catalogue)

    registry = sub.add_parser("registry", help="inspect and promote artefacts")
    registry.add_argument("action", choices=["list", "show", "promote", "shadow"])
    registry.add_argument("version", nargs="?")
    registry.add_argument("--force", action="store_true", help="bypass the latency gate (recorded)")
    registry.set_defaults(run=cmd_registry)

    review = sub.add_parser("review", help="the weekly human review of flagged items (PRD 8.4)")
    review.add_argument("action", choices=["sample", "list", "decide", "precision", "export"])
    review.add_argument("alert_id", nargs="?", help="for `decide`")
    review.add_argument("verdict", nargs="?", choices=VERDICTS, help="for `decide`")
    review.add_argument("--days", type=float, default=7, help="window for `sample`")
    review.add_argument("--size", type=int, default=MIN_BATCH_SIZE)
    review.add_argument("--seed", type=int, help="override the reproducible default")
    review.add_argument("--reviewer", default="")
    review.add_argument("--note", default="")
    review.add_argument("-o", "--output", help="for `export`")
    review.add_argument(
        "--metrics",
        help="for `precision`: merge the measured value into this metrics file, "
        "so `fintext accept` reads a number that was actually reviewed",
    )
    review.set_defaults(run=cmd_review)

    trainer = sub.add_parser("train", help="fine-tune the shared encoder on the declared sources")
    trainer.add_argument("--epochs", type=int, default=3)
    trainer.add_argument("--batch-size", type=int, default=32)
    trainer.add_argument("--max-length", type=int, default=256)
    trainer.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap rows per source; the toxicity corpora are far larger than the rest",
    )
    trainer.add_argument("--sources", nargs="*", default=None)
    trainer.add_argument("--run-name", default=None)
    trainer.set_defaults(run=cmd_train)

    import_cmd = sub.add_parser(
        "import-model", help="bring a run archive from the training notebook into the project"
    )
    import_cmd.add_argument("bundle", help="the <run name>.zip the notebook produced")
    import_cmd.add_argument("--models-dir", default="artifacts/models")
    import_cmd.set_defaults(run=cmd_import_model)

    evaluate_cmd = sub.add_parser(
        "evaluate", help="score a trained model: thresholds tuned on dev, reported on test"
    )
    evaluate_cmd.add_argument("model_dir", help="a run directory under artifacts/models")
    evaluate_cmd.add_argument(
        "--intervals",
        type=int,
        default=0,
        metavar="N",
        help="also bootstrap 95%% intervals over N resamples of the test set (e.g. 1000)",
    )
    evaluate_cmd.set_defaults(run=cmd_evaluate)

    compare_cmd = sub.add_parser(
        "compare",
        help="paired bootstrap: is run B better than run A on the same test rows?",
    )
    compare_cmd.add_argument("baseline", help="run A's directory (evaluated)")
    compare_cmd.add_argument("candidate", help="run B's directory (evaluated)")
    compare_cmd.add_argument("--resamples", type=int, default=1000)
    compare_cmd.add_argument(
        "--common-rows",
        action="store_true",
        help="for runs with different splits: compare on the rows both held out for test",
    )
    compare_cmd.add_argument("--out", default="", help="also write the result as JSON here")
    compare_cmd.set_defaults(run=cmd_compare)

    groups_cmd = sub.add_parser(
        "compare-groups",
        help="is setup B better than setup A, given several seeds of each?",
    )
    groups_cmd.add_argument("--a", nargs="+", required=True, help="setup A's run directories")
    groups_cmd.add_argument("--b", nargs="+", required=True, help="setup B's run directories")
    groups_cmd.add_argument("--resamples", type=int, default=1000)
    groups_cmd.add_argument("--out", default="", help="also write the result as JSON here")
    groups_cmd.set_defaults(run=cmd_compare_groups)

    leak_cmd = sub.add_parser(
        "leakage-check",
        help="near-copies of test rows in a run's train-only sources, and their effect",
    )
    leak_cmd.add_argument("model_dir", help="an evaluated run directory")
    leak_cmd.add_argument(
        "--source", action="append", help="train-only source to check (default: all native ones)"
    )
    leak_cmd.add_argument("--threshold", type=float, default=0.5, help="cosine that flags a row")
    leak_cmd.add_argument("--show", type=int, default=5, help="print this many closest pairs")
    leak_cmd.add_argument("--out", default="", help="write scores and keys (no text) here")
    leak_cmd.set_defaults(run=cmd_leakage_check)

    parity_cmd = sub.add_parser(
        "serving-parity",
        help="served labels and scores against a run's offline per-row scores",
    )
    parity_cmd.add_argument("model_dir", help="the evaluated run the versions were exported from")
    parity_cmd.add_argument("--versions", nargs="+", required=True)
    parity_cmd.add_argument("--sentiment-rows", type=int, default=400)
    parity_cmd.add_argument("--score-rows", type=int, default=300)
    parity_cmd.add_argument("--tolerance", type=float, default=1e-4)
    parity_cmd.add_argument("--out", default="")
    parity_cmd.set_defaults(run=cmd_serving_parity)

    edgar_cmd = sub.add_parser(
        "edgar", help="build the provenance index from SEC EDGAR filings (PRD 8.2)"
    )
    edgar_cmd.add_argument("--out", default="artifacts/provenance/primary_index.jsonl")
    edgar_cmd.add_argument("--per-query", type=int, default=100)
    edgar_cmd.add_argument("--start", default="", help="YYYY-MM-DD")
    edgar_cmd.add_argument("--end", default="", help="YYYY-MM-DD")
    edgar_cmd.add_argument(
        "--with-text",
        action="store_true",
        help="fetch each filing's own text; one extra request per filing",
    )
    edgar_cmd.add_argument(
        "--contact",
        default="",
        help="contact address for the User-Agent, as SEC's fair-access policy asks",
    )
    edgar_cmd.set_defaults(run=cmd_edgar)

    data_cmd = sub.add_parser(
        "data", help="write each declared source to its own folder under data/"
    )
    data_cmd.add_argument("--out", default="data/sources")
    data_cmd.add_argument("--limit", type=int, default=None, help="cap rows per source")
    data_cmd.add_argument(
        "--full",
        action="store_true",
        help="ignore the default per-source caps and write everything",
    )
    data_cmd.add_argument(
        "--raw-cache",
        action="store_true",
        help="download originals into data/raw instead of the shared HuggingFace cache",
    )
    data_cmd.set_defaults(run=cmd_data)

    corpus_cmd = sub.add_parser(
        "corpus", help="build the fine-tuning corpus from captured items and human labels"
    )
    corpus_cmd.add_argument("--out", default="reports/corpus.jsonl")
    corpus_cmd.add_argument(
        "--summary-only", action="store_true", help="report what is there without writing it"
    )
    corpus_cmd.set_defaults(run=cmd_corpus)

    replay_set_cmd = sub.add_parser(
        "replay-set", help="build provenance replay cases from real 8-K press releases"
    )
    replay_set_cmd.add_argument("--out", default="data/samples/edgar_replay.jsonl")
    replay_set_cmd.add_argument("--per-category", type=int, default=40)
    replay_set_cmd.add_argument("--start", default="2024-07-01")
    replay_set_cmd.add_argument("--end", default="2025-12-31")
    replay_set_cmd.add_argument(
        "--refresh",
        action="store_true",
        help="re-clean and re-fabricate the existing file, offline",
    )
    replay_set_cmd.add_argument(
        "--add-text",
        metavar="CASES",
        default="",
        help="copy CASES to --out with each filing's text, as `edgar --with-text` indexes it",
    )
    replay_set_cmd.set_defaults(run=cmd_replay_set)

    offsets_cmd = sub.add_parser(
        "sentiment-offsets",
        help="tune per-language sentiment class offsets on dev; register them as a new version",
    )
    offsets_cmd.add_argument("model_dir", help="the evaluated run the artefact was exported from")
    offsets_cmd.add_argument("--language", default="ar")
    offsets_cmd.add_argument("--from-version", help="registered artefact to derive from")
    offsets_cmd.add_argument("--as-version", help="name for the derived version, e.g. v2.1")
    offsets_cmd.set_defaults(run=cmd_sentiment_offsets)

    export_cmd = sub.add_parser(
        "export", help="trained run -> int8 ONNX, parity-checked, benchmarked and registered"
    )
    export_cmd.add_argument("model_dir", help="a run directory under artifacts/models")
    export_cmd.add_argument("--version", required=True, help="registry name, e.g. shared-xlmr.v2")
    export_cmd.add_argument(
        "--promote", action="store_true", help="make it active, if it passes the latency gate"
    )
    export_cmd.add_argument(
        "--fp32",
        action="store_true",
        help="serve the float model; for when int8 fails parity or is slower on this CPU",
    )
    export_cmd.set_defaults(run=cmd_export)

    audit_cmd = sub.add_parser(
        "label-audit", help="relabel a blind sample to measure a dataset's label noise"
    )
    audit_sub = audit_cmd.add_subparsers(dest="audit_step", required=True)
    draw = audit_sub.add_parser("draw", help="write a blank sheet and a hidden key")
    draw.add_argument("--corpus", default="osact5")
    draw.add_argument("--per-class", type=int, default=100)
    draw.add_argument("--out", default="data/interim/label_audit")
    draw.set_defaults(run=cmd_label_audit_draw)
    score = audit_sub.add_parser("score", help="compare a filled-in sheet with the key")
    score.add_argument("sheet")
    score.add_argument("--key", default=None, help="defaults to key.json beside the sheet")
    score.set_defaults(run=cmd_label_audit_score)

    demo = sub.add_parser("demo", help="score a built-in set of examples")
    demo.set_defaults(run=cmd_demo)
    return parser


def _app(args):
    from .server import build_app

    return build_app(load_config(args.config))


def compact(body: dict[str, Any]) -> str:
    parts = [f"[{body['language']}] {body['latency_ms']:.1f} ms"]
    if sentiment := body.get("sentiment"):
        parts.append(f"sentiment={sentiment['label']} ({sentiment['confidence']:.2f})")
    if toxicity := body.get("toxicity"):
        override = "+lexicon" if toxicity["lexicon_override"] else ""
        state = "TOXIC" if toxicity["toxic"] else "clean"
        parts.append(f"toxicity={state}{override} ({toxicity['score']:.2f})")
    if misleading := body.get("misleading"):
        state = "FLAG" if misleading["likely_misleading"] else "clean"
        parts.append(
            f"misleading={state} ({misleading['score']:.2f}, {misleading['review_priority']})"
        )
    if provenance := body.get("provenance"):
        parts.append(f"provenance={provenance['state']}")
    return " | ".join(parts)


def cmd_serve(args) -> int:
    from . import server

    config = load_config(args.config)
    if args.host or args.port:
        config = config.replace(host=args.host or config.host, port=args.port or config.port)
    if args.asgi:
        import uvicorn

        uvicorn.run(server.asgi_app(config), host=config.host, port=config.port)
    else:
        server.serve(config)
    return 0


def cmd_score(args) -> int:
    app = _app(args)
    payload: dict[str, Any] = {
        "text": args.text if args.text is not None else sys.stdin.read(),
        "profile": args.profile,
    }
    if args.capabilities:
        payload["capabilities"] = [
            Capability(c.strip()).value for c in args.capabilities.split(",")
        ]
    status, body = app.handle("POST", "/v1/score", json.dumps(payload))
    app.shutdown()
    if status != 200:
        print(json.dumps(body, indent=2), file=sys.stderr)
        return 1
    print(compact(body) if args.compact else json.dumps(body, indent=2, ensure_ascii=False))
    return 0


def cmd_batch(args) -> int:
    app = _app(args)
    requests = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            payload.setdefault("profile", args.profile)
            if args.model_version:
                payload["model_version"] = args.model_version
            requests.append(ScoreRequest.from_dict(payload))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for response in app.pipeline.score_many(requests):
            handle.write(json.dumps(response.to_dict(), ensure_ascii=False) + "\n")
    app.shutdown()
    print(f"scored {len(requests)} items -> {output}")
    return 0


def cmd_accept(args) -> int:
    measured = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    report = evaluation.evaluate_acceptance(measured, model_version=args.model_version)
    print(report.to_markdown())
    if args.output:
        Path(args.output).write_text(report.to_markdown(), encoding="utf-8")
    # Non-zero on a failing gate, so CI can gate on it directly.
    return 0 if report.accepted else 1


def cmd_replay(args) -> int:
    from .provenance import Document
    from .schemas import ClaimType

    config = load_config(args.config)
    if args.disjoint_from:
        from .replay_set import shared_filings

        shared = shared_filings(args.cases, args.disjoint_from)
        if shared:
            print(
                f"refusing: {len(shared)} filing(s) appear in both sets, e.g. {sorted(shared)[:3]}"
            )
            return 1
    if args.why:
        from .replay_set import why_not_corroborated

        for reason, examples in why_not_corroborated(args.cases, config.min_match_score).items():
            print(f"\n{reason}: {len(examples)}")
            for example in examples[:8]:
                print(f"  {example}")
        return 0
    cases = []
    held = missed = 0
    for line in Path(args.cases).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if args.hold_out_claim and payload["is_genuine"]:
            from .replay_set import hold_out_claim

            if hold_out_claim(payload):
                held += 1
            else:
                missed += 1
        cases.append(
            evaluation.ReplayCase(
                item_id=payload["item_id"],
                text=payload["text"],
                observed_at=float(payload["observed_at"]),
                is_genuine=bool(payload["is_genuine"]),
                expected_claim_type=(
                    ClaimType(payload["expected_claim_type"])
                    if payload.get("expected_claim_type")
                    else None
                ),
                documents=tuple(Document.from_dict(d) for d in payload.get("documents", ())),
            )
        )
    result = evaluation.replay_provenance(
        cases, windows=config.windows, min_match_score=config.min_match_score
    )
    report = result.to_dict()
    if args.hold_out_claim:
        report["claim_held_out"] = {"removed": held, "not_found_in_filing": missed}
    print(json.dumps(report, indent=2))
    return 0


def cmd_canary(args) -> int:
    app = _app(args)

    def is_toxic(text: str, language) -> bool:
        response = app.pipeline.score(
            ScoreRequest(
                text=text,
                language=language,
                capabilities=(Capability.TOXICITY,),
                profile=Profile.COMPLIANCE,
                explain=False,
            )
        )
        return bool(response.toxicity and response.toxicity.toxic)

    result = evaluation.run_canary(is_toxic, baseline=args.baseline)
    app.shutdown()
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.triggers_retraining_review else 0


def cmd_bench(args) -> int:
    """Measure the two numbers the SLO is stated in: p95 latency at batch size one, and sustained
    requests per second per replica.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    config = load_config(args.config)
    app = _app(args)
    probes = [_pad(text, config.max_tokens) for _, text in DEMO[:5]]

    for index in range(50):  # warm up
        app.pipeline.score(ScoreRequest(text=probes[index % len(probes)]))

    samples = []
    for index in range(args.iterations):
        started = time.perf_counter()
        app.pipeline.score(ScoreRequest(text=probes[index % len(probes)]))
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()

    def percentile(q: float) -> float:
        return samples[max(0, min(len(samples) - 1, int(q * len(samples)) - 1))]

    # Throughput is a capacity statement, so it is measured with concurrent clients rather than
    # derived from the latency figure.
    deadline = time.perf_counter() + args.seconds

    def worker(worker_id: int) -> int:
        served = 0
        while time.perf_counter() < deadline:
            app.pipeline.score(ScoreRequest(text=probes[(worker_id + served) % len(probes)]))
            served += 1
        return served

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        total = sum(pool.map(worker, range(args.concurrency)))
    rps = total / (time.perf_counter() - started)
    app.shutdown()

    result = {
        "backend": app.pipeline.backend.version,
        "config_digest": config.digest(),
        "tokens": config.max_tokens,
        "latency_ms": {
            "iterations": args.iterations,
            "mean": round(sum(samples) / len(samples), 3),
            "p50": round(percentile(0.50), 3),
            "p95": round(percentile(0.95), 3),
            "p99": round(percentile(0.99), 3),
            "max": round(samples[-1], 3),
        },
        "throughput_rps": round(rps, 1),
        "concurrency": args.concurrency,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.gate:
        return 0
    failures = []
    if percentile(0.95) > config.p95_target_ms:
        failures.append(f"p95 {percentile(0.95):.2f} ms exceeds {config.p95_target_ms:.0f} ms")
    if rps < 100:
        failures.append(f"throughput {rps:.1f} rps is below 100 rps per replica")
    for failure in failures:
        print(f"SLO FAILURE: {failure}", file=sys.stderr)
    return 1 if failures else 0


def _pad(text: str, tokens: int) -> str:
    from .preprocess import count_tokens

    words = text.split()
    padded: list[str] = []
    while count_tokens(" ".join(padded)) < tokens:
        padded.extend(words)
    return " ".join(padded)


def cmd_import_model(args) -> int:
    """Unpack a Colab run into artifacts/models and show where it stands."""
    from .registry import RegistryError, import_run

    try:
        run = import_run(args.bundle, load_config(args.config).path(args.models_dir))
    except (RegistryError, OSError) as exc:
        print(f"not imported: {exc}")
        return 1

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    metrics = manifest.get("metrics", {})
    notes_path = run / "run_notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    size = sum(f.stat().st_size for f in run.iterdir()) / 1e9
    print(f"imported {manifest['run_name']} -> {run} ({size:.2f} GB)")
    if notes:
        print(f"  trained on {notes.get('gpu')} in {notes.get('wall_minutes')} min")
    totals = manifest.get("corpus", {}).get("totals", {})
    if totals:
        split = ", ".join(
            f"{k} {totals[k]}" for k in ("train", "dev", "test", "eval") if k in totals
        )
        print(f"  rows: {split}")

    protocol = metrics.get("protocol", {})
    if protocol.get("metrics_reported_on") != "test":
        print(
            "\nnote: this run predates the dev/test split, so its thresholded numbers were "
            f"tuned on the split they report. Re-score it with:\n  fintext evaluate {run}"
        )

    report = evaluation.evaluate_acceptance(metrics, model_version=manifest["run_name"])
    print(f"\n{'metric':34} {'test':>8} {'target':>9}  status")
    print("-" * 64)
    for result in report.results:
        measured = "--" if result.measured is None else f"{result.measured:.4f}"
        target = "" if result.target is None else f"{result.comparison} {result.target}"
        print(f"{result.name:34} {measured:>8} {target:>9}  {result.status}")
    return 0


def _rebuild_split(run: Path):
    """Rebuild a run's train/dev/test exactly as its manifest records them."""
    from . import corpora
    from .training import TrainingConfig

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest.get("corpus", {})
    fields = {
        k: v for k, v in manifest["config"].items() if k in TrainingConfig.__dataclass_fields__
    }
    config = TrainingConfig(**{**fields, "sources": tuple(fields.get("sources", ()))})

    rows, provenance = corpora.load_rows(
        config.sources,
        eval_sources=tuple(recorded.get("eval_only", corpora.EVAL_ONLY_LOADERS)),
        # Exactly the caps that run used. Today's defaults would trim sources it read in full,
        # and the rebuilt train set would no longer match.
        limits=dict.fromkeys((*config.sources, *recorded.get("eval_only", ())), None)
        | recorded.get("limits", {}),
        substitutions=tuple(recorded.get("substitutions", ())),
        train_only=tuple(recorded.get("train_only", ())),
        seed=config.seed,
    )

    was, now = recorded.get("totals", {}), provenance["totals"]
    if was.get("train") != now["train"]:
        return f"trained on {was.get('train')} rows; the rebuild has {now['train']}"
    if "train_fingerprint" in was and was["train_fingerprint"] != now["train_fingerprint"]:
        return "the rebuilt train set's fingerprint does not match the manifest"
    if "heldout_fingerprint" in was and was["heldout_fingerprint"] != now["heldout_fingerprint"]:
        return "the rebuilt dev and test rows' fingerprint does not match the manifest"
    if "train_fingerprint" not in was:
        print(
            "note: this manifest predates train fingerprints, so the train set is matched "
            "on row count and seed only\n"
        )
    if "eval" in was and was["eval"] != now["dev"] + now["test"]:
        return f"held-out rows were {was['eval']}, dev + test is {now['dev'] + now['test']}"
    return manifest, config, rows, provenance


def cmd_evaluate(args) -> int:
    """Re-score a trained model under the frozen-test protocol."""
    import logging

    import torch
    from transformers import AutoTokenizer

    from .training import (
        aggregate,
        build_model,
        compute_metrics,
        pick_device,
        score_each,
        write_scored,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = Path(args.model_dir)
    rebuilt = _rebuild_split(run)
    if isinstance(rebuilt, str):
        print(f"refusing: {rebuilt}")
        return 1
    manifest, config, rows, provenance = rebuilt
    now = provenance["totals"]

    device = pick_device()
    model = build_model(config.encoder)
    model.load_state_dict(
        torch.load(run / "pytorch_model.bin", map_location="cpu", weights_only=False)
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(run))

    # Scored once, per row: the point metrics and the bootstrap read the same scores.
    dev = score_each(model, tokenizer, rows["dev"], config, device, desc="scoring dev")
    test = score_each(model, tokenizer, rows["test"], config, device, desc="scoring test")
    metrics = compute_metrics(aggregate(dev), aggregate(test), config)
    metrics["protocol"] |= {"dev_rows": len(rows["dev"]), "test_rows": len(rows["test"])}
    metrics["protocol"] |= {
        "train_fingerprint": now["train_fingerprint"],
        "device": str(device),
        "evaluated_at": time.time(),
    }
    out = run / "metrics_frozen_test.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Per-row test scores (no text), for `fintext compare` against another run.
    write_scored(run / "scored_test.jsonl", manifest["run_name"], metrics["thresholds"], test)
    write_scored(run / "scored_dev.jsonl", manifest["run_name"], metrics["thresholds"], dev)

    report = evaluation.evaluate_acceptance(metrics, model_version=manifest["run_name"])
    print(f"\n{'metric':34} {'test':>8} {'target':>9}  status")
    print("-" * 64)
    for result in report.results:
        measured = "--" if result.measured is None else f"{result.measured:.4f}"
        target = "" if result.target is None else f"{result.comparison} {result.target}"
        print(f"{result.name:34} {measured:>8} {target:>9}  {result.status}")
    print(f"\nwrote {out}")

    if args.intervals:
        from .intervals import bootstrap, verdict

        found = bootstrap(test, metrics["thresholds"], resamples=args.intervals)
        print(f"\n95% intervals, {args.intervals} resamples of the {len(test)} test rows")
        print(f"{'metric':28} {'point':>7} {'low':>7} {'high':>7} {'target':>9}  verdict")
        print("-" * 78)
        verdicts: dict[str, str] = {}
        for name, interval in found.items():
            comparison, target_value, _ = evaluation.GATES.get(name, ("", None, ""))
            gate = f"{comparison} {target_value}" if target_value is not None else "reported"
            call = (
                verdict(name, interval, comparison, target_value)
                if target_value is not None
                else ""
            )
            verdicts[name] = call
            print(
                f"{name:28} {interval['point']:>7.4f} {interval['low']:>7.4f} "
                f"{interval['high']:>7.4f} {gate:>9}  {call}"
            )
        where = run / "intervals.json"
        where.write_text(
            json.dumps(
                {
                    "resamples": args.intervals,
                    "test_rows": len(test),
                    "thresholds": metrics["thresholds"],
                    "held_fixed": "thresholds, as tuned on dev",
                    "metrics": {n: {**v, "verdict": verdicts[n]} for n, v in found.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {where}")
    return 0


def cmd_compare(args) -> int:
    """Paired comparison of two evaluated runs on the same frozen test rows."""
    from .intervals import compare, load_scored

    runs = [Path(args.baseline), Path(args.candidate)]
    loaded = []
    for run in runs:
        where = run / "scored_test.jsonl"
        if not where.exists():
            print(f"{where} is missing: run `fintext evaluate {run}` first")
            return 1
        loaded.append(load_scored(where))
    (a_header, a_rows), (b_header, b_rows) = loaded
    if args.common_rows:
        from .intervals import common_rows

        total = (len(a_rows), len(b_rows))
        a_rows, b_rows = common_rows(a_rows, b_rows)
        print(f"common rows: {len(a_rows)} of A's {total[0]} and B's {total[1]} test rows")
        if not a_rows:
            print("refusing: the runs share no test rows")
            return 1
    try:
        found = compare(
            a_rows,
            a_header["thresholds"],
            b_rows,
            b_header["thresholds"],
            resamples=args.resamples,
        )
    except ValueError as problem:
        print(f"refusing: {problem}")
        return 1

    print(f"A = {a_header['run']}\nB = {b_header['run']}")
    print(
        f"paired over {len(a_rows)} test rows, {args.resamples} resamples; each run at its own"
        " dev-tuned thresholds\n"
    )
    print(f"{'metric':28} {'A':>7} {'B':>7} {'B-A':>8} {'low':>8} {'high':>8}  call")
    print("-" * 86)
    for name, row in found.items():
        print(
            f"{name:28} {row['a']:>7.4f} {row['b']:>7.4f} {row['diff']:>+8.4f} "
            f"{row['low']:>+8.4f} {row['high']:>+8.4f}  {row['call']}"
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "baseline": a_header["run"],
                    "candidate": b_header["run"],
                    "test_rows": len(a_rows),
                    "resamples": args.resamples,
                    "metrics": found,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


def cmd_compare_groups(args) -> int:
    """Is setup B better than setup A, given several training runs of each?"""
    from .intervals import compare_groups, load_scored

    groups = []
    for runs in (args.a, args.b):
        loaded = []
        for run in map(Path, runs):
            where = run / "scored_test.jsonl"
            if not where.exists():
                print(f"{where} is missing: run `fintext evaluate {run}` first")
                return 1
            header, rows = load_scored(where)
            loaded.append((header["run"], rows, header["thresholds"]))
        groups.append(loaded)
    a, b = groups
    try:
        found = compare_groups(
            [(rows, t) for _, rows, t in a],
            [(rows, t) for _, rows, t in b],
            resamples=args.resamples,
        )
    except ValueError as problem:
        print(f"refusing: {problem}")
        return 1
    print(f"A = {', '.join(n for n, _, _ in a)}\nB = {', '.join(n for n, _, _ in b)}")
    print(f"test rows and runs resampled together, {args.resamples} resamples\n")
    print(f"{'metric':28} {'A mean':>7} {'B mean':>7} {'B-A':>8} {'low':>8} {'high':>8}  call")
    print("-" * 86)
    for name, row in found.items():
        print(
            f"{name:28} {row['a_mean']:>7.4f} {row['b_mean']:>7.4f} {row['diff']:>+8.4f} "
            f"{row['low']:>+8.4f} {row['high']:>+8.4f}  {row['call']}"
        )
    if args.out:
        result = {
            "a": [n for n, _, _ in a],
            "b": [n for n, _, _ in b],
            "resamples": args.resamples,
            "metrics": found,
        }
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def cmd_leakage_check(args) -> int:
    """Near-copies of test rows in a run's train-only sources, and whether they matter."""
    from . import corpora
    from .intervals import load_scored
    from .release_checks import metrics_without, near_duplicates, summarise
    from .training import row_key

    run = Path(args.model_dir)
    rebuilt = _rebuild_split(run)
    if isinstance(rebuilt, str):
        print(f"refusing: {rebuilt}")
        return 1
    manifest, _, rows, _ = rebuilt
    header, scored = load_scored(run / "scored_test.jsonl")
    sources = args.source or list(manifest["corpus"].get("train_only", ()))
    native = [s for s in sources if s in corpora.TRAIN_ONLY_LOADERS]
    report: dict[str, Any] = {"run": header["run"], "threshold": args.threshold, "sources": {}}
    for source in native:
        # The rows that actually went into training, after the split's own exact-key filter, not
        # the source as published.
        source_rows = [r for r in rows["train"] if r.get("corpus") == source]
        if source_rows and "source_text" in source_rows[0]:
            print(f"{source}: translations of training rows, filtered by origin; skipped")
            continue
        if not source_rows:
            print(f"{source}: no rows of it in this run's training set; skipped")
            continue
        languages = {r["language"] for r in source_rows}
        held = [r for r in rows["test"] if r["language"].split("-")[0] in languages]
        best, which = near_duplicates([r["text"] for r in held], [r["text"] for r in source_rows])
        flagged = {row_key(r) for r, s in zip(held, best, strict=True) if s >= args.threshold}
        effect = metrics_without(scored, header["thresholds"], flagged)
        changed = {
            name: round(effect["without_flagged"][name] - value, 4)
            for name, value in effect["all"].items()
            if name in effect["without_flagged"]
        }
        report["sources"][source] = {
            "source_rows": len(source_rows),
            "test_rows_compared": len(held),
            "similarity": summarise(best),
            "flagged": len(flagged),
            "metric_change_without_flagged": changed,
        }
        print(f"\n{source}: {len(held)} test rows against {len(source_rows)} source rows")
        print(json.dumps(report["sources"][source]["similarity"], indent=2))
        print(f"flagged at >= {args.threshold}: {len(flagged)}")
        print(
            "largest metric change with them removed:", max(map(abs, changed.values()), default=0)
        )
        for i in sorted(range(len(held)), key=lambda i: -best[i])[: args.show]:
            closest = source_rows[which[i]]["text"][:70]
            print(f"  {best[i]:.3f}  {held[i]['text'][:70]!r}\n         {closest!r}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def cmd_serving_parity(args) -> int:
    """Score held-out rows through each served version and compare with offline scores."""
    from dataclasses import replace

    from .intervals import load_scored
    from .preprocess import model_text
    from .registry import Registry
    from .release_checks import serving_parity
    from .schemas import Language
    from .scoring import load_backend
    from .training import row_key

    run = Path(args.model_dir)
    rebuilt = _rebuild_split(run)
    if isinstance(rebuilt, str):
        print(f"refusing: {rebuilt}")
        return 1
    _, _, rows, _ = rebuilt
    _, scored = load_scored(run / "scored_test.jsonl")
    by_key = {r["key"]: r for r in scored}
    config = load_config(args.config)
    registry = Registry(config.path(config.registry_path), p95_target_ms=config.p95_target_ms)
    report: dict[str, Any] = {}
    failed = False
    for version in args.versions:
        entry = registry.resolve(version)
        backend = load_backend(replace(config, backend="onnx", model_version=version), registry)

        def predict(text, language, backend=backend):
            return backend.predict(model_text(text), Language(language.split("-")[0]))

        found = serving_parity(
            predict,
            rows["test"],
            by_key,
            row_key,
            entry.sentiment_offsets,
            sentiment_sample=args.sentiment_rows,
            score_sample=args.score_rows,
        )
        report[version] = found
        ok = (
            found["sentiment_label_matches"] == found["sentiment_rows"]
            and found["max_score_delta"] <= args.tolerance
        )
        failed |= not ok
        print(
            f"{version}: sentiment {found['sentiment_label_matches']}/{found['sentiment_rows']} "
            f"labels match, served {found['served_sentiment']}; max score delta "
            f"{found['max_score_delta']:.2e} on {found['score_rows']} rows "
            f"({'ok' if ok else 'MISMATCH'})"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if failed else 0


def cmd_replay_set(args) -> int:
    """Genuine claims from press-release exhibits, plus a fabricated copy of each."""
    import logging

    from .edgar import Fetcher
    from .replay_set import collect_genuine, fabricate, write_cases

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.refresh:
        from .replay_set import refresh

        kept, made = refresh(args.out)
        print(f"{kept} genuine and {made} fabricated cases -> {args.out}")
        return 0
    if not os.environ.get("FINTEXT_SEC_CONTACT"):
        print(
            "SEC's fair-access policy asks automated clients for a contact address.\n"
            "Set FINTEXT_SEC_CONTACT in your own shell first; it is sent in the request\n"
            "header only and never written to any file."
        )
        return 2
    if args.add_text:
        from .replay_set import add_text

        if Path(args.add_text).resolve() == Path(args.out).resolve():
            print("refusing: --out must be a new file; the source case set stays as built")
            return 2
        fetched, failed = add_text(args.add_text, args.out, Fetcher())
        if failed:
            print(
                f"{failed} filing(s) failed, so nothing was written. A 403 for every one "
                "usually means FINTEXT_SEC_CONTACT isn't a real address."
            )
            return 1
        print(f"text for {fetched} filing(s) -> {args.out}")
        return 0
    genuine = collect_genuine(
        Fetcher(), per_category=args.per_category, start=args.start, end=args.end
    )
    fabricated = fabricate(genuine)
    path = write_cases(genuine + fabricated, args.out)
    print(f"{len(genuine)} genuine and {len(fabricated)} fabricated cases -> {path}")
    return 0


def cmd_sentiment_offsets(args) -> int:
    """Tune sentiment offsets on a run's dev scores, show their effect on test, and optionally
    register them with an existing artefact as a new version.
    """
    from dataclasses import replace

    from . import sentiment_offsets as so
    from .intervals import compare, load_scored
    from .registry import Registry, RegistryError

    run = Path(args.model_dir)
    if not (run / "scored_dev.jsonl").exists():
        print(f"{run}/scored_dev.jsonl is missing: run `fintext evaluate {run}` first")
        return 1
    _, dev = load_scored(run / "scored_dev.jsonl")
    header, test = load_scored(run / "scored_test.jsonl")
    try:
        tuned = so.tune(dev, args.language)
    except ValueError as problem:
        print(f"refusing: {problem}")
        return 1
    offsets = tuned["offsets"]
    effect = compare(
        test,
        header["thresholds"],
        so.apply(test, offsets, args.language),
        header["thresholds"],
        resamples=1000,
    )[f"sentiment_macro_f1_{args.language}"]
    print(f"offsets ({args.language}, tuned on {tuned['dev_rows']} dev rows): {offsets}")
    print(f"dev macro-F1  {tuned['dev_macro_f1_without']:.4f} -> {tuned['dev_macro_f1']:.4f}")
    print(
        f"test macro-F1 {effect['a']:.4f} -> {effect['b']:.4f}  "
        f"({effect['low']:+.4f} to {effect['high']:+.4f}, {effect['call']})"
    )
    if not (args.from_version and args.as_version):
        return 0
    if effect["call"] != "B better":
        print("not registering: the offsets don't clearly help on test")
        return 1

    config = load_config(args.config)
    registry = Registry(config.path(config.registry_path), p95_target_ms=config.p95_target_ms)
    try:
        base = registry.resolve(args.from_version)
    except RegistryError as problem:
        print(f"refusing: {problem}")
        return 1
    if args.as_version in registry.entries:
        print(f"refusing: {args.as_version} already exists; versions are immutable")
        return 1
    metrics = dict(base.metrics)
    metrics[f"sentiment_macro_f1_{args.language}"] = effect["b"]
    entry = replace(
        base,
        version=args.as_version,
        created_at=time.time(),
        promoted=False,
        promoted_at=None,
        parent_version=base.version,
        metrics=metrics,
        sentiment_offsets={**base.sentiment_offsets, args.language: offsets},
        notes=(
            f"{base.notes}; {args.language} sentiment offsets {offsets} tuned on dev "
            f"(test macro-F1 {effect['a']:.3f} -> {effect['b']:.3f})"
        ).lstrip("; "),
    )
    registry.register(entry)
    print(f"registered {entry.version} (from {base.version}); not promoted")
    return 0


def cmd_export(args) -> int:
    """Export a trained run for serving."""
    import torch
    from transformers import AutoTokenizer

    from .registry import Registry
    from .training import build_model, export_and_register

    config = load_config(args.config)
    run = Path(args.model_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    model = build_model(manifest["config"]["encoder"])
    model.load_state_dict(
        torch.load(run / "pytorch_model.bin", map_location="cpu", weights_only=False)
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(run))
    registry = Registry(config.path(config.registry_path), p95_target_ms=config.p95_target_ms)
    metrics = {k: v for k, v in manifest["metrics"].items() if not isinstance(v, dict)}
    metrics["thresholds"] = manifest["metrics"].get("thresholds", {})
    entry = export_and_register(
        model,
        tokenizer,
        version=args.version,
        registry=registry,
        tokenizer_dir=str(run),
        metrics=metrics,
        promote=False,
        int8=not args.fp32,
        threads=config.onnx_threads,
    )
    print(f"registered {entry.version}: {entry.artifact_path}")
    print(f"  {entry.notes}")
    print(f"  p95 {entry.latency_p95_ms:.1f} ms on this machine (target {config.p95_target_ms} ms)")
    if args.promote:
        try:
            registry.promote(args.version)
            print("  promoted: active")
        except Exception as exc:
            print(f"  not promoted: {exc}")
            return 1
    return 0


def cmd_label_audit_draw(args) -> int:
    """Draw the held-out sample exactly as training split it, then blind it."""
    from . import corpora, label_audit

    rows, _ = corpora.load_rows(
        tuple(corpora.LOADERS),
        eval_sources=tuple(corpora.EVAL_ONLY_LOADERS),
        train_only=tuple(corpora.TRAIN_ONLY_LOADERS),
    )
    sample = label_audit.draw_sample(rows, args.corpus, per_class=args.per_class)
    folder = Path(args.out) / args.corpus
    sheet, key = label_audit.write_audit(sample, folder)
    print(f"{len(sample)} rows, half labelled offensive, in random order")
    print(f"  give the annotator: {sheet}")
    print(f"  keep to yourself:   {key}")
    return 0


def cmd_label_audit_score(args) -> int:
    from . import label_audit

    sheet = Path(args.sheet)
    key = Path(args.key) if args.key else sheet.with_name("key.json")
    print(label_audit.score_audit(sheet, key).summary())
    return 0


def cmd_edgar(args) -> int:
    """Fetch real filings so the provenance gates can be measured at all."""
    import logging

    from .edgar import EdgarError, build_index

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config)

    if not (args.contact or os.environ.get("FINTEXT_SEC_CONTACT")):
        print(
            "note: SEC's fair-access policy asks for a contact address in the "
            "User-Agent.\n      Set FINTEXT_SEC_CONTACT or pass --contact.\n"
        )
    try:
        summary = build_index(
            config.path(args.out),
            per_query=args.per_query,
            start=args.start,
            end=args.end,
            with_text=args.with_text,
            contact=args.contact,
        )
    except EdgarError as exc:
        print(f"EDGAR: {exc}")
        return 1

    print(f"\n{summary['documents']} documents -> {summary['path']}")
    print(f"  {summary['with_ticker']} carry a ticker")
    print(f"  filed between {summary['earliest']} and {summary['latest']}")
    print(
        "\nThe index supplies the entity vocabulary the claim extractor uses, so a "
        "larger index widens what can be corroborated as well as what it matches "
        "against."
    )
    return 0


def cmd_data(args) -> int:
    """Materialise every declared source into data/ so it can be inspected."""
    import logging
    import os

    from . import corpora

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.raw_cache:
        # Keep the downloaded originals with the project rather than in the shared cache in the
        # home directory.
        raw = Path(load_config(args.config).path("data/raw"))
        raw.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(raw)
        print(f"downloading originals into {raw}")

    limits: dict[str, int | None] | None = None
    if args.limit:
        limits = dict.fromkeys(list(corpora.LOADERS) + list(corpora.EVAL_ONLY_LOADERS), args.limit)
    elif args.full:
        limits = dict.fromkeys(corpora.DEFAULT_LIMITS, 10**9)

    written = corpora.materialise(
        tuple(corpora.LOADERS),
        eval_sources=tuple(corpora.EVAL_ONLY_LOADERS),
        out_dir=args.out,
        limits=limits,
    )

    print(f"\n{'source':24} {'rows':>8}  {'licence':38} use")
    print("-" * 88)
    for key, manifest in written.items():
        use = "EVAL ONLY" if manifest["eval_only"] else ", ".join(manifest["uses"])
        print(f"{key:24} {manifest['rows']:>8}  {manifest['licence'][:38]:38} {use}")
    print(f"\n{sum(m['rows'] for m in written.values())} rows written to {args.out}")
    print(
        "\nNot redistributable. This folder is git-ignored and excluded from any "
        "archive of the project; the licences permit you holding the data, not "
        "handing it on."
    )
    return 0


def cmd_train(args) -> int:
    """Fine-tune on the PRD's declared sources and write the run manifest."""
    import logging

    from . import corpora
    from .training import TrainingConfig, train

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sources = tuple(args.sources) if args.sources else tuple(corpora.LOADERS)
    eval_sources = tuple(corpora.EVAL_ONLY_LOADERS)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        sources=sources,
    )
    if args.run_name:
        config.run_name = args.run_name

    # An explicit --limit caps every source; otherwise the per-source defaults in
    # corpora.DEFAULT_LIMITS apply.
    limits = {s: args.limit for s in sources} if args.limit else None
    rows, provenance = corpora.load_rows(
        sources,
        eval_sources=eval_sources,
        train_only=tuple(corpora.TRAIN_ONLY_LOADERS),
        limits=limits,
        seed=config.seed,
    )

    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    if provenance["substitutions"]:
        print(
            f"\nNote: {', '.join(provenance['substitutions'])} could not be loaded as declared "
            "and a substitute was used. See the manifest.\n"
        )

    manifest = train(config, rows, provenance)
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"\nwrote {manifest['artifact_dir']} on {manifest['device']}")
    return 0


def cmd_corpus(args) -> int:
    """Join what the service captured to what people decided about it."""
    from . import corpus
    from .approvals import ApprovalQueue
    from .catalogue import LicenceError
    from .review import ReviewStore

    config = load_config(args.config)
    review_dir = config.path(config.review_dir)
    examples = corpus.build_corpus(
        corpus.read_log(config.path(config.decision_log or "reports/decision_log.jsonl")),
        review_decisions=list(ReviewStore(review_dir / "decisions.jsonl").latest().values()),
        resolutions=ApprovalQueue(review_dir / "approvals.jsonl").resolutions(),
    )
    summary = corpus.summarise(examples)
    print(json.dumps(summary, indent=2))

    if args.summary_only:
        return 0
    if not summary["gold"]:
        print("\nnothing to train on: no item has a human verdict on it yet.")
        return 1
    try:
        path = corpus.write_corpus(examples, config.path(args.out))
    except LicenceError as exc:
        print(f"\n{exc}")
        print(
            "\nThe production corpus is other people's text. Clear it with legal, then set "
            f"{corpus.PRODUCTION_SOURCE!r} to CLEARED in catalogue.py."
        )
        return 2
    print(f"\nwrote {summary['total']} examples to {path}")
    return 0


def cmd_catalogue(args) -> int:
    if args.licence_log:
        print(f"wrote licence log to {catalogue.write_licence_log(args.licence_log)}")
        return 0
    for dataset in catalogue.select(task=args.task, use=args.use, language=args.language):
        print(
            f"{dataset.key:22} {dataset.status:24} [{', '.join(dataset.uses)}]  {dataset.licence}"
        )
    for dataset in catalogue.unresolved():
        print(f"awaiting legal review: {dataset.key}", file=sys.stderr)
    return 0


def cmd_registry(args) -> int:
    from dataclasses import asdict

    from .registry import Registry

    config = load_config(args.config)
    registry = Registry(config.path(config.registry_path), p95_target_ms=config.p95_target_ms)

    if args.action == "list":
        for version in registry.versions():
            entry = registry.resolve(version)
            markers = " ".join(
                m
                for m, v in (
                    ("active", registry.active_version),
                    ("shadow", registry.shadow_version),
                )
                if v == version
            )
            p95 = f"{entry.latency_p95_ms:.1f} ms" if entry.latency_p95_ms else "unbenchmarked"
            print(f"{version:34} {p95:16} {markers}")
        return 0

    if not args.version:
        print("this action needs a version", file=sys.stderr)
        return 2
    if args.action == "show":
        print(json.dumps(asdict(registry.resolve(args.version)), indent=2))
    elif args.action == "promote":
        entry = registry.promote(args.version, force=args.force)
        print(f"promoted {entry.version} (p95 {entry.latency_p95_ms} ms)")
    else:
        registry.set_shadow(args.version)
        print(f"shadow traffic now mirrors {args.version}")
    return 0


def cmd_review(args) -> int:
    app = _app(args)
    try:
        if args.action == "sample":
            payload = {"days": args.days, "size": args.size}
            if args.seed is not None:
                payload["seed"] = args.seed
            _, batch = app.handle("POST", "/v1/review/batch", json.dumps(payload))
            print(
                f"batch {batch['batch_id']}: {batch['size']} alerts "
                f"(seed {batch['seed']}, redraw with --seed {batch['seed']})"
            )
            if batch["short_of_minimum"]:
                print(
                    f"note: below the {batch['minimum']}-alert weekly minimum (PRD 8.4)",
                    file=sys.stderr,
                )
            return 0

        if args.action == "list":
            status, batch = app.handle("GET", "/v1/review/batch")
            if status != 200:
                print(batch["detail"], file=sys.stderr)
                return 1
            for alert in batch["alerts"]:
                decision = batch["decisions"].get(alert["alert_id"])
                mark = decision["verdict"] if decision else "-"
                print(
                    f"{alert['alert_id']}  {alert['kind']:11} {alert['score']:.2f}  "
                    f"{mark:9} {alert['text'][:56]}"
                )
            print(f"\n{batch['outstanding']} of {batch['size']} still to review")
            return 0

        if args.action == "decide":
            if not args.alert_id or not args.verdict:
                print(
                    "usage: fintext review decide <alert_id> <correct|incorrect|unsure>",
                    file=sys.stderr,
                )
                return 2
            status, body = app.handle(
                "POST",
                "/v1/review/decision",
                json.dumps(
                    {
                        "alert_id": args.alert_id,
                        "verdict": args.verdict,
                        "reviewer": args.reviewer or "cli",
                        "note": args.note,
                    }
                ),
            )
            if status != 200:
                print(body.get("detail", body), file=sys.stderr)
                return 1
            print(f"recorded {args.verdict} for {args.alert_id}")
            return 0

        if args.action == "export":
            if not args.output:
                print("`export` needs -o/--output", file=sys.stderr)
                return 2
            print(f"wrote {app.export_review_csv(args.output)}")
            return 0

        _, report = app.handle("GET", "/v1/review/precision")
        print(json.dumps(report, indent=2))

        if args.metrics:
            # Without this the acceptance gate reads a hand-typed number.
            path = Path(args.metrics)
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            existing.update(report["measured_metrics"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            print(f"merged {sorted(report['measured_metrics'])} into {path}", file=sys.stderr)

        if report["spans_multiple_lineages"]:
            print(
                "warning: this batch spans more than one model/threshold version; "
                "precision across a release is not one number",
                file=sys.stderr,
            )
        english = report["by_language"].get("en")
        if english and english["judged"] and not english["passes"]:
            print(
                f"human-review precision {english['precision']} is below the "
                f"{report['gate']} gate (PRD 8.4)",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        app.shutdown()


def cmd_demo(args) -> int:
    app = _app(args)
    for profile, text in DEMO:
        status, body = app.handle(
            "POST", "/v1/score", json.dumps({"text": text, "profile": profile})
        )
        print(f"{profile:10} {text[:58]:60} {compact(body) if status == 200 else body}")
    app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
