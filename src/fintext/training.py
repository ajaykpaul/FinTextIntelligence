"""Training the shared encoder and exporting the served artefact (PRD 6.2, 6.6)."""

from __future__ import annotations

import json
import logging
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import catalogue
from .credentials import use_huggingface_token
from .evaluation import auroc, binary_scores, choose_threshold, macro_f1
from .preprocess import model_text
from .progress import enabled as progress_enabled
from .progress import note, progress

logger = logging.getLogger("fintext.training")

ENCODER = "xlm-roberta-base"
SENTIMENT_LABELS = ("positive", "negative", "neutral")
SUBTYPE_LABELS = ("hate", "harassment", "profanity", "threat")

# name, size, loss weight, languages. The misleading head is English-only (PRD 3.1), and the
# mask is derived from this table rather than hard-coded.
HEADS = (
    ("sentiment", 3, 1.0, ("en", "ar")),
    ("toxicity", 1, 1.0, ("en", "ar")),
    ("toxicity_subtypes", 4, 0.5, ("en", "ar")),
    ("misleading", 1, 0.7, ("en",)),
)

FINBERT_MARGIN = 0.02
PROBE_TEXTS = (
    "Liabilities fell sharply in the third quarter as the group repaid debt.",
    "Revenue missed consensus and the board cut full-year guidance.",
    "The company will publish results on Tuesday.",
    "Guaranteed 300% returns, act now, send USDT to our wallet.",
    "You are a pathetic clown and everyone knows it.",
)


@dataclass
class TrainingConfig:
    output_dir: str = "artifacts/models"
    run_name: str = field(default_factory=lambda: f"shared-xlmr-{int(time.time())}")
    encoder: str = ENCODER
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    encoder_learning_rate: float = 1e-5  # the encoder tolerates a smaller step than the heads
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_length: int = 256
    seed: int = 20240101
    training_seed: int | None = None
    resume: bool = True
    checkpoint_dir: str | None = None
    sentiment_weight: float = 2.0
    arabic_toxicity_weight: float = 1.5
    train_on_model_text: bool = True
    mixed_precision: bool = True
    sources: tuple[str, ...] = (
        "financial_phrasebank",
        "fiqa_2018",
        "jigsaw_toxic",
        "wikipedia_talk",
        "hatexplain",
        "osact4",
        "osact5",
        "isot_fake_news",
        "liar",
    )
    # Recall-first, bounded by the human-review precision floor (PRD 8.1, 8.4).
    recall_floor: float = 0.90
    precision_floor: float = 0.90


def build_model(encoder_name: str = ENCODER, dropout: float = 0.1):
    """One encoder, one linear head per task."""
    import torch
    from transformers import AutoConfig, AutoModel

    hidden = AutoConfig.from_pretrained(encoder_name).hidden_size

    class MultiTask(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.dropout = torch.nn.Dropout(dropout)
            self.heads = torch.nn.ModuleDict(
                {name: torch.nn.Linear(hidden, size) for name, size, _, _ in HEADS}
            )

        def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **_):
            encoded = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
            )
            pooled = self.dropout(encoded.last_hidden_state[:, 0])
            return {name: head(pooled) for name, head in self.heads.items()}

        def get_input_embeddings(self):
            """Exposed for integrated-gradients attribution."""
            return self.encoder.get_input_embeddings()

    return MultiTask()


def multitask_loss(
    outputs: dict, targets: dict, masks: dict, weights: dict[str, float] | None = None
) -> tuple[Any, dict[str, float]]:
    """Masked, weighted sum of per-task losses."""
    import torch
    from torch.nn import functional

    total = None
    parts: dict[str, float] = {}
    for name, size, weight, _ in HEADS:
        weight = (weights or {}).get(name, weight)
        if name not in outputs or name not in targets:
            continue
        mask = masks.get(name)
        logits, target = outputs[name], targets[name]
        elements = masks.get(f"{name}_elements")
        rows = masks.get(f"{name}_row_weight")
        if mask is not None:
            if not mask.any():
                parts[name] = 0.0
                continue
            logits, target = logits[mask], target[mask]
            if elements is not None:
                elements = elements[mask]
            if rows is not None:
                rows = rows[mask]

        if elements is not None:
            # Subtypes: average only over the labels that were actually given.
            if not elements.any():
                parts[name] = 0.0
                continue
            loss = functional.binary_cross_entropy_with_logits(
                logits[elements], target[elements].float()
            )
        elif name == "sentiment":
            loss = functional.cross_entropy(logits, target.long(), label_smoothing=0.05)
        else:
            loss = functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1) if size == 1 else logits,
                target.float(),
                # Per-row weights (Arabic toxicity, see TrainingConfig), as a weighted mean so
                # the head's overall scale doesn't drift.
                weight=None if rows is None else rows / rows.mean(),
                # Recall-first: a missed slur in a bank channel costs more than a false positive
                # a reviewer discards.
                pos_weight=torch.tensor(3.0, device=logits.device) if name == "toxicity" else None,
            )
        parts[name] = float(loss.detach())
        total = weight * loss if total is None else total + weight * loss

    if total is None:
        raise ValueError("no task in this batch carried a label")
    return total, parts


def pick_device():
    """CUDA, then Apple's MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism costs throughput and buys reproducible acceptance numbers, which is the right
    # trade for a gated artefact.
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(
    config: TrainingConfig,
    rows: dict[str, list[dict[str, Any]]],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune and write the run manifest."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    # Gate first. Nothing below runs on a source that cannot be trained on.
    catalogue.check_training_sources(config.sources)
    set_seed(config.seed if config.training_seed is None else config.training_seed)

    use_huggingface_token()  # the encoder download is the largest one
    tokenizer = AutoTokenizer.from_pretrained(config.encoder)
    model = build_model(config.encoder)
    device = pick_device()
    model.to(device)

    collate = _collator(
        tokenizer,
        config.max_length,
        device,
        arabic_toxicity_weight=config.arabic_toxicity_weight,
        on_model_text=config.train_on_model_text,
    )
    loader: Any = DataLoader(
        rows["train"],  # type: ignore[arg-type]  # a list of rows plus an explicit collate_fn
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    optimiser = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
            {"params": model.heads.parameters(), "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    steps = max(1, len(loader) * config.epochs)
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * config.warmup_ratio), steps)

    output = Path(config.output_dir) / config.run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(config.checkpoint_dir) / config.run_name if config.checkpoint_dir else output
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints / "checkpoint.pt"
    head_weights = {"sentiment": config.sentiment_weight}

    fp16 = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    if fp16:
        logger.info("mixed precision (fp16) on")

    total_steps = len(loader) * config.epochs
    step = 0
    first_epoch = 0
    started = time.time()

    # Resume if a checkpoint from this run name is sitting there.
    if config.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"])
        first_epoch = state["epoch"] + 1
        step = state["step"]
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        logger.info("resumed %s at epoch %d/%d", config.run_name, first_epoch + 1, config.epochs)

    bars = progress_enabled()
    for epoch in range(first_epoch, config.epochs):
        model.train()
        batches = progress(
            loader, desc=f"epoch {epoch + 1}/{config.epochs}", total=len(loader), unit="batch"
        )
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                outputs = model(batch["input_ids"], batch["attention_mask"])
            # The loss is taken in fp32: small per-head losses lose digits in fp16.
            outputs = {name: value.float() for name, value in outputs.items()}
            loss, parts = multitask_loss(outputs, batch["targets"], batch["masks"], head_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)  # clip the real gradients, not the scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            # A smaller scale afterwards means fp16 overflowed and the scaler skipped this step;
            # the schedule waits for the optimiser.
            if scaler.get_scale() >= scale:
                schedule.step()
            optimiser.zero_grad(set_to_none=True)

            # A run this long needs to be observable. With tqdm the bar shows it; without (a log
            # file, a bare install) a line every 25 steps.
            step += 1
            if bars and (step % 25 == 0 or step == total_steps):
                note(batches, loss=f"{loss.item():.4f}")
            elif step % 25 == 0 or step == total_steps:
                done = time.time() - started
                remaining = done / step * (total_steps - step)
                logger.info(
                    "step %d/%d  loss %.4f  %.1fs elapsed, ~%.0fm left",
                    step,
                    total_steps,
                    loss.item(),
                    done,
                    remaining / 60,
                )
        logger.info(
            "epoch %d/%d done; last batch losses: %s",
            epoch + 1,
            config.epochs,
            {name: round(float(value), 4) for name, value in parts.items()},
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "schedule": schedule.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "config": asdict(config),
            },
            checkpoint_path,
        )
        logger.info("checkpoint written: %s", checkpoint_path)

    metrics = evaluate(
        model,
        tokenizer,
        rows["dev"],
        rows["test"],
        config,
        device,
        scored_out=output / "scored_test.jsonl",
    )
    torch.save(model.state_dict(), output / "pytorch_model.bin")
    tokenizer.save_pretrained(output)
    manifest = {
        "run_name": config.run_name,
        "created_at": time.time(),
        "config": asdict(config),
        "metrics": metrics,
        "corpus": provenance or {},
        "device": str(device),
        "artifact_dir": str(output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def row_key(row: dict[str, Any]) -> str:
    """A short digest of a row's language, source and text."""
    import hashlib

    raw = "\x1f".join((row.get("language", "en"), str(row.get("corpus")), row["text"]))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def score_each(
    model, tokenizer, rows, config: TrainingConfig, device=None, *, desc: str = "scoring"
) -> list[dict[str, Any]]:
    """The model's output for every row, one record per row, in order."""
    import torch

    device = device or pick_device()
    model.eval()
    scored: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in progress(
            range(0, len(rows), 64), desc=desc, total=-(-len(rows) // 64), unit="batch"
        ):
            chunk = rows[start : start + 64]
            encoded = tokenizer(
                [model_text(r["text"]) for r in chunk],
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
                padding=True,
            ).to(device)
            outputs = model(**encoded)
            sentiment_probs = torch.softmax(outputs["sentiment"].float(), -1)
            sentiment = sentiment_probs.argmax(-1).tolist()
            toxicity = torch.sigmoid(outputs["toxicity"].squeeze(-1)).tolist()
            misleading = torch.sigmoid(outputs["misleading"].squeeze(-1)).tolist()
            probs = sentiment_probs.tolist()
            for row, s, ps, t, m in zip(chunk, sentiment, probs, toxicity, misleading, strict=True):
                labels = row.get("labels", {})
                record: dict[str, Any] = {
                    # Identifies the row without storing its (licensed) text, so two runs'
                    # scores can be paired row by row.
                    "key": row_key(row),
                    "language": row.get("language", "en"),
                    "corpus": row.get("corpus"),
                }
                if "sentiment" in labels:
                    record["sentiment_true"] = str(labels["sentiment"])
                    record["sentiment_pred"] = SENTIMENT_LABELS[s]
                    # In SENTIMENT_LABELS order, for re-deciding offline (e.g. per-language
                    # class offsets) without scoring again.
                    record["sentiment_probs"] = [round(x, 5) for x in ps]
                if labels.get("toxicity") is not None:
                    record["toxicity_true"] = int(labels["toxicity"])
                    record["toxicity_score"] = float(t)
                if "misleading" in labels:
                    record["misleading_true"] = int(labels["misleading"])
                    record["misleading_score"] = float(m)
                scored.append(record)
    return scored


def aggregate(scored: list[dict[str, Any]]) -> dict[str, list]:
    """Per-row records into the per-task, per-language lists the metrics read."""
    collected: dict[str, list] = {}

    def collect(key: str, value) -> None:
        collected.setdefault(key, []).append(value)

    for record in scored:
        language = record["language"]
        if "sentiment_true" in record:
            collect(f"sentiment_true_{language}", record["sentiment_true"])
            collect(f"sentiment_pred_{language}", record["sentiment_pred"])
        if "toxicity_true" in record:
            collect(f"toxicity_true_{language}", record["toxicity_true"])
            collect(f"toxicity_score_{language}", record["toxicity_score"])
        if "misleading_true" in record and language == "en":
            collect("misleading_true_en", record["misleading_true"])
            collect("misleading_score_en", record["misleading_score"])
    return collected


def score_rows(
    model, tokenizer, rows, config: TrainingConfig, device=None, *, desc: str = "scoring"
) -> dict[str, list]:
    return aggregate(score_each(model, tokenizer, rows, config, device, desc=desc))


# Floors for the misleading head.
MISLEADING_FLOORS = {"min_recall": 0.75, "min_precision": 0.60}


def compute_metrics(
    dev: dict[str, list], test: dict[str, list], config: TrainingConfig
) -> dict[str, Any]:
    """Thresholds from dev, every reported number from test."""
    metrics: dict[str, Any] = {
        "thresholds": {},
        "threshold_detail": {},
        "protocol": {"thresholds_tuned_on": "dev", "metrics_reported_on": "test"},
    }

    def thresholded(task: str, language: str, floors: dict[str, float]) -> None:
        key = f"{task}_{language}"
        dev_truth = dev.get(f"{task}_true_{language}")
        test_truth = test.get(f"{task}_true_{language}")
        if not dev_truth or not test_truth:
            return
        chosen = choose_threshold(dev_truth, dev[f"{task}_score_{language}"], **floors)
        precision, recall, f1 = binary_scores(
            test_truth, test[f"{task}_score_{language}"], threshold=chosen.threshold
        )
        metrics["thresholds"][key] = chosen.threshold
        metrics[f"{task}_f1_{language}"] = f1
        metrics[f"{task}_precision_{language}"] = precision
        metrics[f"{task}_recall_{language}"] = recall
        metrics[f"{task}_feasible_{language}"] = chosen.feasible
        metrics["threshold_detail"][key] = {
            "threshold": chosen.threshold,
            "feasible": chosen.feasible,
            "basis": chosen.basis,
            "dev_precision": chosen.precision,
            "dev_recall": chosen.recall,
            "dev_f1": chosen.f1,
            "dev_ceiling_f1": chosen.ceiling_f1,
            **floors,
        }

    toxicity_floors = {"min_recall": config.recall_floor, "min_precision": config.precision_floor}
    for language in ("en", "ar"):
        sentiment_truth = test.get(f"sentiment_true_{language}")
        if sentiment_truth:
            metrics[f"sentiment_macro_f1_{language}"] = macro_f1(
                sentiment_truth, test[f"sentiment_pred_{language}"], SENTIMENT_LABELS
            )
        toxicity_truth = test.get(f"toxicity_true_{language}")
        if toxicity_truth:
            metrics[f"toxicity_auroc_{language}"] = auroc(
                toxicity_truth, test[f"toxicity_score_{language}"]
            )
        thresholded("toxicity", language, toxicity_floors)
    thresholded("misleading", "en", MISLEADING_FLOORS)

    # Dialect slices (PRD 8.1: reported, not gated).
    for key in sorted(test):
        if not key.startswith("toxicity_true_"):
            continue
        language = key.removeprefix("toxicity_true_")
        if language in ("en", "ar"):
            continue
        truth, scores = test[key], test[f"toxicity_score_{language}"]
        metrics[f"toxicity_auroc_{language}"] = auroc(truth, scores)
        if "toxicity_ar" in metrics["thresholds"]:
            _, recall, f1 = binary_scores(
                truth, scores, threshold=metrics["thresholds"]["toxicity_ar"]
            )
            metrics[f"toxicity_f1_{language}"] = f1
            metrics[f"toxicity_recall_{language}"] = recall

    if "toxicity_f1_en" in metrics and "toxicity_f1_ar" in metrics:
        # PRD 8.1 hard cap, on test, at the thresholds dev chose.
        metrics["cross_language_f1_gap"] = abs(
            metrics["toxicity_f1_en"] - metrics["toxicity_f1_ar"]
        )
    return metrics


def evaluate(
    model,
    tokenizer,
    dev_rows,
    test_rows,
    config: TrainingConfig,
    device=None,
    *,
    scored_out: Path | None = None,
) -> dict[str, Any]:
    """Tune thresholds on dev, report on test; optionally keep the per-row test scores."""
    dev = score_each(model, tokenizer, dev_rows, config, device, desc="scoring dev")
    test = score_each(model, tokenizer, test_rows, config, device, desc="scoring test")
    metrics = compute_metrics(aggregate(dev), aggregate(test), config)
    metrics["protocol"] |= {"dev_rows": len(dev_rows), "test_rows": len(test_rows)}
    if scored_out is not None:
        write_scored(scored_out, config.run_name, metrics["thresholds"], test)
        write_scored(
            scored_out.with_name("scored_dev.jsonl"), config.run_name, metrics["thresholds"], dev
        )
    return metrics


def write_scored(
    path: Path, run_name: str, thresholds: dict[str, float], records: list[dict[str, Any]]
) -> None:
    """Per-row test scores (no text) for `fintext compare`: a header line, then one row each."""
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"thresholds": thresholds, "run": run_name}) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")


def compare_finbert(
    shared_f1: float, finbert_f1: float, margin: float = FINBERT_MARGIN
) -> dict[str, Any]:
    """The week-three challenger rule (PRD 6.2)."""
    delta = finbert_f1 - shared_f1
    replace = delta > margin
    return {
        "shared_macro_f1": shared_f1,
        "finbert_macro_f1": finbert_f1,
        "delta": delta,
        "replace_shared_head": replace,
        "decision": (
            "FinBERT takes English financial sentiment; all other tasks stay on the shared encoder."
            if replace
            else f"Shared encoder keeps English sentiment; a gain of {delta:+.3f} F1 does not "
            f"clear the {margin:.2f} margin and would add a permanent second artefact."
        ),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(model, tokenizer, output_path: str | Path, max_length: int = 256) -> Path:
    """Export with one named output per head."""
    import torch

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    names = [name for name, _, _, _ in HEADS]
    encoded = tokenizer(PROBE_TEXTS[0], return_tensors="pt", truncation=True, max_length=max_length)

    class Tuple(torch.nn.Module):
        """Flattens the dict output into a fixed, declared order."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            heads = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            return tuple(heads[name] for name in names)

    torch.onnx.export(
        Tuple(model),
        (encoded["input_ids"], encoded["attention_mask"]),
        str(target),
        input_names=["input_ids", "attention_mask"],
        output_names=names,
        dynamic_axes={"input_ids": {1: "sequence"}, "attention_mask": {1: "sequence"}},
        opset_version=17,
        do_constant_folding=True,
        # torch 2.6+ defaults to the dynamo exporter, whose graph the ONNX Runtime quantiser
        # cannot shape-infer here: it reads a head's weight as (768,) where the graph declares
        # (3,) and aborts.
        dynamo=False,
    )
    return target


def quantise(model_path: str | Path, output_path: str | Path | None = None) -> Path:
    """int8 dynamic quantisation."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(model_path)
    target = Path(output_path) if output_path else source.with_suffix(".int8.onnx")
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
    return target


def check_parity(float_backend, int8_backend, tolerance: float = 0.05) -> dict[str, Any]:
    """Compare float and quantised scores on the probe set."""
    from .schemas import Language

    deltas = {"toxicity": 0.0, "misleading": 0.0, "sentiment": 0.0}
    for text in PROBE_TEXTS:
        left, right = (
            float_backend.predict(text, Language.EN),
            int8_backend.predict(text, Language.EN),
        )
        deltas["toxicity"] = max(deltas["toxicity"], abs(left.toxicity - right.toxicity))
        deltas["misleading"] = max(deltas["misleading"], abs(left.misleading - right.misleading))
        deltas["sentiment"] = max(
            deltas["sentiment"],
            max(abs(left.sentiment[k] - right.sentiment[k]) for k in left.sentiment),
        )
    return {"max_delta": deltas, "tolerance": tolerance, "ok": max(deltas.values()) <= tolerance}


def measure_latency(backend, *, iterations: int = 300, warmup: int = 20) -> dict[str, float]:
    """Latency for 256-token inputs, which is what the SLO is stated for."""
    from .schemas import Language

    probes = [(text + " filler") * 40 for text in PROBE_TEXTS]
    for _ in range(warmup):
        backend.predict(probes[0][:2000], Language.EN)

    samples = []
    for index in progress(range(iterations), desc="timing", total=iterations, unit="call"):
        started = time.perf_counter()
        backend.predict(probes[index % len(probes)][:2000], Language.EN)
        samples.append((time.perf_counter() - started) * 1000.0)

    samples.sort()
    return {
        "iterations": float(iterations),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[max(0, int(0.95 * len(samples)) - 1)],
        "p99_ms": samples[max(0, int(0.99 * len(samples)) - 1)],
        "max_ms": samples[-1],
    }


def export_and_register(
    model,
    tokenizer,
    *,
    version: str,
    registry,
    tokenizer_dir: str,
    output_dir: str = "artifacts/onnx",
    metrics: dict | None = None,
    parent_version: str | None = None,
    promote: bool = True,
    int8: bool = True,
    threads: int = 1,
):
    """Export, quantise, verify, benchmark, register and optionally promote."""
    from .registry import Entry, file_digest
    from .scoring import Onnx

    float_path = export_onnx(model, tokenizer, Path(output_dir) / f"{version}.onnx")
    served_path = float_path
    note = f"fp32; {threads} thread(s)"
    if int8:
        served_path = quantise(float_path)
        parity = check_parity(
            Onnx(float_path, tokenizer, version=f"{version}-fp32"),
            Onnx(served_path, tokenizer, version=version),
        )
        if not parity["ok"]:
            raise ValueError(
                f"int8 quantisation shifted scores beyond tolerance: {parity['max_delta']}; "
                "thresholds tuned on the float model would not transfer"
            )
        note = f"int8 dynamic; parity {parity['max_delta']}; {threads} thread(s)"

    latency = measure_latency(Onnx(served_path, tokenizer, version=version, threads=threads))
    entry = registry.register(
        Entry(
            version=version,
            artifact_path=str(served_path),
            artifact_digest=file_digest(served_path),
            created_at=time.time(),
            tokenizer_path=tokenizer_dir,
            metrics={**(metrics or {}), **latency},
            latency_p95_ms=latency["p95_ms"],
            parent_version=parent_version,
            notes=note,
        )
    )
    return registry.promote(version) if promote else entry


def _collator(
    tokenizer,
    max_length: int,
    device,
    *,
    arabic_toxicity_weight: float = 1.0,
    on_model_text: bool = True,
):
    import torch

    label_index = {label: i for i, label in enumerate(SENTIMENT_LABELS)}

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [model_text(row["text"]) if on_model_text else row["text"] for row in batch],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        targets, masks = {}, {}
        for name, size, _, _ in HEADS:
            present = [name in row.get("labels", {}) for row in batch]
            masks[name] = torch.tensor(present, dtype=torch.bool, device=device)
            if name == "sentiment":
                values = [
                    label_index.get(row.get("labels", {}).get(name, "neutral"), 2) for row in batch
                ]
                targets[name] = torch.tensor(values, dtype=torch.long, device=device)
            elif size > 1:
                # A source may label some subtypes and not others: OSACT5 has no harassment
                # label, HateXplain only labels hate.
                vectors = [row.get("labels", {}).get(name) or [None] * size for row in batch]
                targets[name] = torch.tensor(
                    [[0.0 if v is None else float(v) for v in vector] for vector in vectors],
                    dtype=torch.float,
                    device=device,
                )
                masks[f"{name}_elements"] = torch.tensor(
                    [[v is not None for v in vector] for vector in vectors],
                    dtype=torch.bool,
                    device=device,
                )
            else:
                values = [row.get("labels", {}).get(name, 0.0) for row in batch]
                targets[name] = torch.tensor(values, dtype=torch.float, device=device)

        masks["toxicity_row_weight"] = torch.tensor(
            [
                arabic_toxicity_weight if row.get("language", "en").startswith("ar") else 1.0
                for row in batch
            ],
            dtype=torch.float,
            device=device,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "targets": targets,
            "masks": masks,
        }

    return collate
