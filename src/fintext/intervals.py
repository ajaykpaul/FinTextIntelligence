"""Bootstrap confidence intervals for the model metrics."""

from __future__ import annotations

from typing import Any

SENTIMENT = ("positive", "negative", "neutral")

# Metrics where a smaller number is the better one.
LOWER_IS_BETTER = frozenset({"cross_language_f1_gap"})


def _arrays(scored: list[dict[str, Any]]):
    import numpy as np

    def column(name, default=np.nan, dtype=float):
        return np.array([r.get(name, default) for r in scored], dtype=dtype)

    language = np.array([r["language"] for r in scored])
    has_sentiment = np.array(["sentiment_true" in r for r in scored])
    sentiment_true = np.array([r.get("sentiment_true", "") for r in scored])
    sentiment_pred = np.array([r.get("sentiment_pred", "") for r in scored])
    has_toxicity = np.array(["toxicity_true" in r for r in scored])
    has_misleading = np.array(["misleading_true" in r for r in scored])
    return {
        "language": language,
        "has_sentiment": has_sentiment,
        "sentiment_true": sentiment_true,
        "sentiment_pred": sentiment_pred,
        "has_toxicity": has_toxicity,
        "toxicity_true": column("toxicity_true", 0, int),
        "toxicity_score": column("toxicity_score"),
        "has_misleading": has_misleading,
        "misleading_true": column("misleading_true", 0, int),
        "misleading_score": column("misleading_score"),
    }


def metrics_on(a: dict[str, Any], index, thresholds: dict[str, float]) -> dict[str, float]:
    """Every gated model metric on the rows at `index` (with repeats)."""
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    out: dict[str, float] = {}
    language = a["language"][index]

    for lang in ("en", "ar"):
        pick = index[a["has_sentiment"][index] & (language == lang)]
        if pick.size:
            out[f"sentiment_macro_f1_{lang}"] = float(
                f1_score(
                    a["sentiment_true"][pick],
                    a["sentiment_pred"][pick],
                    labels=list(SENTIMENT),
                    average="macro",
                    zero_division=0,
                )
            )

    # Dialect slices are scored at the Arabic threshold, as the service would.
    for lang in ("en", "ar", "ar-lev"):
        pick = index[a["has_toxicity"][index] & (language == lang)]
        truth = a["toxicity_true"][pick]
        if pick.size and 0 < truth.sum() < truth.size:
            scores = a["toxicity_score"][pick]
            out[f"toxicity_auroc_{lang}"] = float(roc_auc_score(truth, scores))
            threshold = thresholds.get(f"toxicity_{'en' if lang == 'en' else 'ar'}")
            if threshold is not None:
                out[f"toxicity_f1_{lang}"] = float(
                    f1_score(truth, scores >= threshold, zero_division=0)
                )

    pick = index[a["has_misleading"][index] & (language == "en")]
    if pick.size and "misleading_en" in thresholds:
        out["misleading_f1_en"] = float(
            f1_score(
                a["misleading_true"][pick],
                a["misleading_score"][pick] >= thresholds["misleading_en"],
                zero_division=0,
            )
        )

    if "toxicity_f1_en" in out and "toxicity_f1_ar" in out:
        out["cross_language_f1_gap"] = float(np.abs(out["toxicity_f1_en"] - out["toxicity_f1_ar"]))
    return out


def bootstrap(
    scored: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    resamples: int = 1000,
    seed: int = 20261119,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Point estimate and percentile interval for each metric."""
    import numpy as np

    from .progress import progress

    a = _arrays(scored)
    everything = np.arange(len(scored))
    point = metrics_on(a, everything, thresholds)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in progress(range(resamples), desc="bootstrap", total=resamples, unit="resample"):
        index = rng.integers(0, len(scored), len(scored))
        for name, value in metrics_on(a, index, thresholds).items():
            draws.setdefault(name, []).append(value)

    tail = (1 - level) / 2 * 100
    return {
        name: {
            "point": value,
            "low": float(np.percentile(draws[name], tail)),
            "high": float(np.percentile(draws[name], 100 - tail)),
            "resamples": len(draws[name]),
        }
        for name, value in point.items()
    }


def verdict(name: str, interval: dict[str, float], comparison: str, target: float) -> str:
    """Whether the gate passes with room, fails clearly, or sits inside the noise."""
    low, high = interval["low"], interval["high"]
    if comparison == ">=":
        if low >= target:
            return "pass"
        return "fail" if high < target else "too close to call"
    if high <= target:
        return "pass"
    return "fail" if low > target else "too close to call"


def load_scored(path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header line (run name, thresholds) and per-row records `evaluate` saved."""
    import json
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    return header, [json.loads(line) for line in lines[1:] if line.strip()]


def common_rows(
    a_scored: list[dict[str, Any]], b_scored: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows both runs were tested on, aligned, for runs whose splits differ."""
    from collections import Counter

    def unique(rows):
        counts = Counter(r["key"] for r in rows)
        return {r["key"]: r for r in rows if counts[r["key"]] == 1}

    a, b = unique(a_scored), unique(b_scored)
    shared = sorted(a.keys() & b.keys())
    return [a[k] for k in shared], [b[k] for k in shared]


def compare(
    a_scored: list[dict[str, Any]],
    a_thresholds: dict[str, float],
    b_scored: list[dict[str, Any]],
    b_thresholds: dict[str, float],
    *,
    resamples: int = 1000,
    seed: int = 20261119,
    level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Paired bootstrap of B − A for every metric both runs report."""
    import numpy as np

    from .progress import progress

    if len(a_scored) != len(b_scored):
        raise ValueError(f"the runs were scored on {len(a_scored)} and {len(b_scored)} test rows")
    for i, (x, y) in enumerate(zip(a_scored, b_scored, strict=True)):
        if "key" not in x or "key" not in y:
            raise ValueError("scores predate row keys; re-run `fintext evaluate` on both runs")
        same_labels = all(
            x.get(f) == y.get(f) for f in ("toxicity_true", "sentiment_true", "misleading_true")
        )
        if x["key"] != y["key"] or not same_labels:
            raise ValueError(f"test row {i} differs between the runs: not the same frozen split")

    a, b = _arrays(a_scored), _arrays(b_scored)
    everything = np.arange(len(a_scored))
    point_a = metrics_on(a, everything, a_thresholds)
    point_b = metrics_on(b, everything, b_thresholds)
    names = [n for n in point_a if n in point_b]

    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {n: [] for n in names}
    for _ in progress(range(resamples), desc="paired bootstrap", total=resamples, unit="resample"):
        index = rng.integers(0, len(a_scored), len(a_scored))
        on_a, on_b = metrics_on(a, index, a_thresholds), metrics_on(b, index, b_thresholds)
        for n in names:
            if n in on_a and n in on_b:
                diffs[n].append(on_b[n] - on_a[n])

    tail = (1 - level) / 2 * 100
    out: dict[str, dict[str, Any]] = {}
    for n in names:
        low = float(np.percentile(diffs[n], tail))
        high = float(np.percentile(diffs[n], 100 - tail))
        out[n] = {
            "a": point_a[n],
            "b": point_b[n],
            "diff": point_b[n] - point_a[n],
            "low": low,
            "high": high,
            "resamples": len(diffs[n]),
            "call": difference_call(n, low, high),
        }
    return out


def difference_call(name: str, low: float, high: float) -> str:
    """Which run is better on `name`, or neither, from the interval on B − A."""
    if low <= 0 <= high:
        return "no detectable difference"
    b_higher = low > 0
    if name in LOWER_IS_BETTER:
        b_higher = not b_higher
    return "B better" if b_higher else "A better"


def compare_groups(
    a_runs: list[tuple[list[dict[str, Any]], dict[str, float]]],
    b_runs: list[tuple[list[dict[str, Any]], dict[str, float]]],
    *,
    resamples: int = 1000,
    seed: int = 20261119,
    level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Is configuration B better than A, given several training runs of each?"""
    import numpy as np

    from .progress import progress

    runs = [*a_runs, *b_runs]
    if len(a_runs) < 2 or len(b_runs) < 2:
        raise ValueError("each configuration needs at least two runs")
    first = runs[0][0]
    for scored, _ in runs[1:]:
        if len(scored) != len(first) or any(
            x.get("key") != y.get("key") for x, y in zip(first, scored, strict=True)
        ):
            raise ValueError("the runs were not scored on the same test rows in the same order")

    arrays = [(_arrays(scored), thresholds) for scored, thresholds in runs]
    everything = np.arange(len(first))
    points = [metrics_on(a, everything, t) for a, t in arrays]
    names = [n for n in points[0] if all(n in p for p in points)]
    na = len(a_runs)

    def means(values: list[dict[str, float]], picks_a, picks_b) -> dict[str, float]:
        return {
            n: float(
                np.mean([values[i][n] for i in picks_b]) - np.mean([values[i][n] for i in picks_a])
            )
            for n in names
        }

    a_idx, b_idx = list(range(na)), list(range(na, len(runs)))
    point = means(points, a_idx, b_idx)
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {n: [] for n in names}
    for _ in progress(range(resamples), desc="group bootstrap", total=resamples, unit="resample"):
        rows = rng.integers(0, len(first), len(first))
        picks_a = list(rng.choice(a_idx, len(a_idx)))
        picks_b = list(rng.choice(b_idx, len(b_idx)))
        needed = set(picks_a) | set(picks_b)
        on = {i: metrics_on(arrays[i][0], rows, arrays[i][1]) for i in needed}
        for n in names:
            if all(n in on[i] for i in needed):
                diffs[n].append(
                    float(
                        np.mean([on[i][n] for i in picks_b]) - np.mean([on[i][n] for i in picks_a])
                    )
                )

    tail = (1 - level) / 2 * 100
    out: dict[str, dict[str, Any]] = {}
    for n in names:
        low = float(np.percentile(diffs[n], tail))
        high = float(np.percentile(diffs[n], 100 - tail))
        out[n] = {
            "a_mean": float(np.mean([points[i][n] for i in a_idx])),
            "b_mean": float(np.mean([points[i][n] for i in b_idx])),
            "a_runs": [points[i][n] for i in a_idx],
            "b_runs": [points[i][n] for i in b_idx],
            "diff": point[n],
            "low": low,
            "high": high,
            "call": difference_call(n, low, high),
        }
    return out
