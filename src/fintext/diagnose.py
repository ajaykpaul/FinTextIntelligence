"""Where the toxicity head goes wrong, broken down by source."""

from __future__ import annotations

import random
from typing import Any

from .evaluation import auroc, binary_scores, choose_threshold


def _toxicity_rows(rows: list[dict[str, Any]], languages: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("labels", {}).get("toxicity") is not None and r.get("language") in languages
    ]


def score_toxicity(model, tokenizer, rows, config, device=None, *, desc="scoring") -> list[float]:
    """The toxicity probability for each row, in the order given."""
    from .training import score_rows

    # score_rows groups by language; giving every row the same one keeps them in a single list,
    # in order.
    tagged = [{"text": r["text"], "language": "x", "labels": {"toxicity": 0}} for r in rows]
    return score_rows(model, tokenizer, tagged, config, device, desc=desc).get(
        "toxicity_score_x", []
    )


def toxicity_by_source(
    model,
    tokenizer,
    rows: dict[str, list[dict[str, Any]]],
    config,
    device=None,
    *,
    languages: tuple[str, ...] = ("en", "ar", "ar-lev"),
    train_sample: int = 3000,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], float]]]:
    """Per-source toxicity numbers, plus every held-out row with its score."""
    held = _toxicity_rows(rows["dev"] + rows["test"], languages)
    train = _toxicity_rows(rows["train"], languages)
    random.Random(seed).shuffle(train)
    train = train[:train_sample]

    held_scores = score_toxicity(model, tokenizer, held, config, device, desc="held-out rows")
    train_scores = score_toxicity(model, tokenizer, train, config, device, desc="train sample")

    # The deployed threshold per language: tuned on dev, recall-first. Dialect slices use the
    # Arabic one, as the service would.
    dev_texts = {r["text"] for r in rows["dev"]}
    thresholds: dict[str, float] = {}
    for language in ("en", "ar"):
        pairs = [
            (int(r["labels"]["toxicity"]), s)
            for r, s in zip(held, held_scores, strict=True)
            if r["language"] == language and r["text"] in dev_texts
        ]
        if pairs:
            truth, scores = zip(*pairs, strict=True)
            thresholds[language] = choose_threshold(
                list(truth),
                list(scores),
                min_recall=config.recall_floor,
                min_precision=config.precision_floor,
            ).threshold

    def group(subset, scores):
        out: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
        for row, score in zip(subset, scores, strict=True):
            key = (row["language"], row.get("corpus", "?"))
            truth, values = out.setdefault(key, ([], []))
            truth.append(int(row["labels"]["toxicity"]))
            values.append(score)
        return out

    held_groups, train_groups = group(held, held_scores), group(train, train_scores)
    table = []
    for (language, corpus), (truth, scores) in sorted(held_groups.items()):
        threshold = thresholds.get("en" if language == "en" else "ar", 0.5)
        precision, recall, f1 = binary_scores(truth, scores, threshold=threshold)
        best = choose_threshold(truth, scores, min_recall=config.recall_floor)
        train_truth, train_scores_ = train_groups.get((language, corpus), ([], []))
        table.append(
            {
                "language": language,
                "corpus": corpus,
                "rows": len(truth),
                "positive_rate": sum(truth) / len(truth),
                "auroc_train": auroc(train_truth, train_scores_) if train_truth else None,
                "auroc_held": auroc(truth, scores),
                "ceiling_f1": best.ceiling_f1,
                "f1_at_floor_own_threshold": best.f1,
                "f1_deployed": f1,
                "precision_deployed": precision,
                "recall_deployed": recall,
                "threshold": threshold,
            }
        )
    return table, list(zip(held, held_scores, strict=True))


def worst_errors(
    scored: list[tuple[dict[str, Any], float]],
    *,
    language: str,
    count: int = 10,
) -> dict[str, list[tuple[float, str, str]]]:
    """The most confident false positives and false negatives for one language."""
    pool = [(r, s) for r, s in scored if r["language"] == language]
    negatives = sorted(
        ((s, r.get("corpus", "?"), r["text"]) for r, s in pool if not r["labels"]["toxicity"]),
        reverse=True,
    )
    positives = sorted(
        (s, r.get("corpus", "?"), r["text"]) for r, s in pool if r["labels"]["toxicity"]
    )
    return {"false_positives": negatives[:count], "false_negatives": positives[:count]}


def print_table(table: list[dict[str, Any]]) -> None:
    header = (
        f"{'lang':7} {'corpus':16} {'rows':>6} {'pos%':>5} {'AUC tr':>7} {'AUC held':>8} "
        f"{'ceilF1':>7} {'F1@0.9R':>8} {'F1 dep':>7} {'P dep':>6} {'R dep':>6}"
    )
    print(header)
    print("-" * len(header))
    for t in table:
        train = "--" if t["auroc_train"] is None else f"{t['auroc_train']:.3f}"
        print(
            f"{t['language']:7} {t['corpus']:16} {t['rows']:>6} {100 * t['positive_rate']:>4.0f}% "
            f"{train:>7} {t['auroc_held']:>8.3f} {t['ceiling_f1']:>7.3f} "
            f"{t['f1_at_floor_own_threshold']:>8.3f} {t['f1_deployed']:>7.3f} "
            f"{t['precision_deployed']:>6.2f} {t['recall_deployed']:>6.2f}"
        )
