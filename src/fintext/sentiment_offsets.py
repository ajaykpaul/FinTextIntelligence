"""Per-language class offsets for the sentiment head, tuned on dev."""

from __future__ import annotations

from typing import Any

LABELS = ("positive", "negative", "neutral")  # training.SENTIMENT_LABELS order


def _rows(scored: list[dict[str, Any]], language: str):
    import numpy as np

    rows = [
        r
        for r in scored
        if r["language"] == language and "sentiment_true" in r and "sentiment_probs" in r
    ]
    if not rows:
        raise ValueError(
            f"no {language} sentiment rows with probabilities; re-run `fintext evaluate`"
        )
    probs = np.array([r["sentiment_probs"] for r in rows], dtype=float)
    truth = np.array([LABELS.index(r["sentiment_true"]) for r in rows])
    return probs, truth


def decide(probs, offsets) -> Any:
    """Argmax of log-probability plus a per-class offset."""
    import numpy as np

    return np.argmax(np.log(np.clip(probs, 1e-9, 1.0)) + np.asarray(offsets), axis=1)


def macro_f1(truth, predicted) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0))


def neutral_ranking(scored: list[dict[str, Any]], language: str = "ar") -> dict[str, float]:
    """AUROC of each class's probability, one class against the rest."""
    from sklearn.metrics import roc_auc_score

    probs, truth = _rows(scored, language)
    return {label: float(roc_auc_score(truth == i, probs[:, i])) for i, label in enumerate(LABELS)}


def tune(
    dev: list[dict[str, Any]],
    language: str = "ar",
    *,
    grid: tuple[float, float, float] = (-3.0, 3.0, 0.1),
) -> dict[str, Any]:
    """Offsets for positive and neutral (negative fixed at 0) that maximise dev macro-F1."""
    import numpy as np

    probs, truth = _rows(dev, language)
    low, high, step = grid
    values = np.round(np.arange(low, high + step / 2, step), 4)
    best = (macro_f1(truth, decide(probs, (0.0, 0.0, 0.0))), 0.0, 0.0)
    for positive in values:
        for neutral in values:
            score = macro_f1(truth, decide(probs, (positive, 0.0, neutral)))
            # Ties go to the smaller adjustment, so offsets never drift for nothing.
            if score > best[0] + 1e-12 or (
                abs(score - best[0]) <= 1e-12
                and abs(positive) + abs(neutral) < abs(best[1]) + abs(best[2])
            ):
                best = (score, float(positive), float(neutral))
    return {
        "offsets": {"positive": best[1], "negative": 0.0, "neutral": best[2]},
        "dev_macro_f1": best[0],
        "dev_macro_f1_without": macro_f1(truth, decide(probs, (0.0, 0.0, 0.0))),
        "dev_rows": int(len(truth)),
    }


def apply(
    scored: list[dict[str, Any]], offsets: dict[str, float], language: str = "ar"
) -> list[dict[str, Any]]:
    """Copies of `scored` with that language's sentiment re-decided under `offsets`."""
    vector = [offsets[label] for label in LABELS]
    out = []
    for record in scored:
        if record["language"] == language and "sentiment_probs" in record:
            chosen = int(decide([record["sentiment_probs"]], vector)[0])
            record = {**record, "sentiment_pred": LABELS[chosen]}
        out.append(record)
    return out
