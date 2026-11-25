"""Checks a new run goes through before it is written up or registered."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from typing import Any

from .corpora import heldout_key


def near_duplicates(
    held_out: list[str],
    source: list[str],
    *,
    ngrams: tuple[int, int] = (3, 5),
    chunk: int = 1000,
) -> tuple[list[float], list[int]]:
    """For each held-out text, its highest cosine against any source text, and which."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not held_out or not source:
        return [0.0] * len(held_out), [-1] * len(held_out)
    a = [heldout_key(t) or "_" for t in held_out]
    b = [heldout_key(t) or "_" for t in source]
    vectoriser = TfidfVectorizer(analyzer="char", ngram_range=ngrams, sublinear_tf=True)
    vectoriser.fit(a + b)
    left, right = vectoriser.transform(a), vectoriser.transform(b)
    best = np.zeros(len(a))
    which = np.zeros(len(a), dtype=int)
    for start in range(0, len(a), chunk):
        sims = (left[start : start + chunk] @ right.T).toarray()
        best[start : start + chunk] = sims.max(axis=1)
        which[start : start + chunk] = sims.argmax(axis=1)
    return [float(x) for x in best], [int(x) for x in which]


def summarise(similarities: Iterable[float], cuts=(0.5, 0.6, 0.7, 0.8, 0.9)) -> dict[str, Any]:
    values = list(similarities)
    return {
        "rows": len(values),
        "max": max(values, default=0.0),
        "at_or_above": {str(c): sum(v >= c for v in values) for c in cuts},
    }


def metrics_without(
    scored: list[dict[str, Any]], thresholds: dict[str, float], drop_keys: set[str]
) -> dict[str, dict[str, float]]:
    """A run's metrics on all its test rows, and on the rows not in `drop_keys`."""
    import numpy as np

    from .intervals import _arrays, metrics_on

    arrays = _arrays(scored)
    everything = np.arange(len(scored))
    kept = np.array([i for i, r in enumerate(scored) if r["key"] not in drop_keys], dtype=int)
    return {
        "all": metrics_on(arrays, everything, thresholds),
        "without_flagged": metrics_on(arrays, kept, thresholds),
    }


def expected_label(record: dict[str, Any], offsets: dict[str, dict[str, float]]) -> str:
    """The offline sentiment decision for one scored row under a version's offsets."""
    from . import sentiment_offsets as so

    language = record["language"]
    if language in offsets and "sentiment_probs" in record:
        return str(so.apply([record], offsets[language], language)[0]["sentiment_pred"])
    return str(record["sentiment_pred"])


def serving_parity(
    predict: Callable[[str, str], Any],
    rows: list[dict[str, Any]],
    scored_by_key: dict[str, dict[str, Any]],
    key_of: Callable[[dict[str, Any]], str],
    offsets: dict[str, dict[str, float]],
    *,
    sentiment_sample: int = 400,
    score_sample: int = 300,
    seed: int = 20261119,
    sentiment_language: str = "ar",
) -> dict[str, Any]:
    """Served labels and scores against the offline per-row scores."""
    rng = random.Random(seed)
    sentiment_rows = [
        r
        for r in rows
        if r["language"] == sentiment_language
        and key_of(r) in scored_by_key
        and "sentiment_true" in scored_by_key[key_of(r)]
    ]
    taken = {id(r) for r in sentiment_rows}
    other_rows = [r for r in rows if id(r) not in taken and key_of(r) in scored_by_key]
    sentiment_rows = rng.sample(sentiment_rows, min(sentiment_sample, len(sentiment_rows)))
    other_rows = rng.sample(other_rows, min(score_sample, len(other_rows)))

    matches, answers = 0, {"positive": 0, "negative": 0, "neutral": 0}
    for row in sentiment_rows:
        served = predict(row["text"], row["language"]).top_sentiment()[0].value
        answers[served] += 1
        matches += served == expected_label(scored_by_key[key_of(row)], offsets)

    largest = 0.0
    for row in other_rows:
        scores = predict(row["text"], row["language"])
        record = scored_by_key[key_of(row)]
        for name, value in (
            ("toxicity_score", scores.toxicity),
            ("misleading_score", scores.misleading),
        ):
            if name in record:
                largest = max(largest, abs(record[name] - value))
    return {
        "sentiment_rows": len(sentiment_rows),
        "sentiment_label_matches": matches,
        "served_sentiment": answers,
        "score_rows": len(other_rows),
        "max_score_delta": largest,
    }


def quantisation_parity(
    reference: Callable[[str, str], Any],
    candidate: Callable[[str, str], Any],
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    offsets: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """How far a quantised model's scores and decisions move from the float model's."""
    import numpy as np

    deltas: dict[str, list[float]] = {"toxicity": [], "misleading": [], "sentiment": []}
    flips = {"toxicity": 0, "misleading": 0, "sentiment": 0}
    by_language: dict[str, int] = {}
    for row in rows:
        language = row["language"].split("-")[0]
        by_language[language] = by_language.get(language, 0) + 1
        a, b = reference(row["text"], language), candidate(row["text"], language)
        deltas["toxicity"].append(abs(a.toxicity - b.toxicity))
        deltas["misleading"].append(abs(a.misleading - b.misleading))
        deltas["sentiment"].append(max(abs(a.sentiment[k] - b.sentiment[k]) for k in a.sentiment))
        cut = thresholds[f"toxicity_{language}"]
        flips["toxicity"] += (a.toxicity >= cut) != (b.toxicity >= cut)
        if language == "en" and "misleading_en" in thresholds:
            cut = thresholds["misleading_en"]
            flips["misleading"] += (a.misleading >= cut) != (b.misleading >= cut)
        flips["sentiment"] += a.top_sentiment()[0] != b.top_sentiment()[0]
    return {
        "rows": len(rows),
        "by_language": by_language,
        "max_delta": {k: float(max(v, default=0.0)) for k, v in deltas.items()},
        "p99_delta": {k: float(np.percentile(v, 99)) if v else 0.0 for k, v in deltas.items()},
        "mean_delta": {k: float(np.mean(v)) if v else 0.0 for k, v in deltas.items()},
        "decision_flips": flips,
        "offsets": offsets or {},
    }
