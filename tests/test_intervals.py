"""Bootstrap intervals agree with the point metrics and behave like intervals."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from fintext import intervals  # noqa: E402
from fintext.evaluation import auroc, binary_scores, macro_f1  # noqa: E402


@pytest.fixture
def scored():
    rng = random.Random(3)
    rows = []
    for i in range(600):
        language = "en" if i % 2 else "ar"
        toxic = int(rng.random() < 0.3)
        # Informative but noisy scores; English a little better than Arabic.
        noise = 0.25 if language == "en" else 0.35
        score = min(1.0, max(0.0, 0.3 + 0.4 * toxic + rng.gauss(0, noise)))
        truth = rng.choice(intervals.SENTIMENT)
        pred = truth if rng.random() < 0.8 else rng.choice(intervals.SENTIMENT)
        rows.append(
            {
                "language": language,
                "toxicity_true": toxic,
                "toxicity_score": score,
                "sentiment_true": truth,
                "sentiment_pred": pred,
            }
        )
    return rows


THRESHOLDS = {"toxicity_en": 0.5, "toxicity_ar": 0.5}


def test_point_estimates_match_the_harness(scored):
    result = intervals.bootstrap(scored, THRESHOLDS, resamples=50)
    en = [r for r in scored if r["language"] == "en"]
    truth = [r["toxicity_true"] for r in en]
    scores = [r["toxicity_score"] for r in en]
    assert result["toxicity_auroc_en"]["point"] == pytest.approx(auroc(truth, scores))
    assert result["toxicity_f1_en"]["point"] == pytest.approx(
        binary_scores(truth, scores, threshold=0.5)[2]
    )
    assert result["sentiment_macro_f1_en"]["point"] == pytest.approx(
        macro_f1(
            [r["sentiment_true"] for r in en],
            [r["sentiment_pred"] for r in en],
            intervals.SENTIMENT,
        )
    )


def test_intervals_contain_the_point_and_are_reproducible(scored):
    one = intervals.bootstrap(scored, THRESHOLDS, resamples=200)
    two = intervals.bootstrap(scored, THRESHOLDS, resamples=200)
    assert one == two
    for value in one.values():
        assert value["low"] <= value["point"] <= value["high"]
        assert value["high"] - value["low"] > 0


def test_more_rows_give_a_narrower_interval(scored):
    small = intervals.bootstrap(scored[:150], THRESHOLDS, resamples=300)
    large = intervals.bootstrap(scored, THRESHOLDS, resamples=300)
    width = lambda r: r["toxicity_auroc_en"]["high"] - r["toxicity_auroc_en"]["low"]  # noqa: E731
    assert width(large) < width(small)


def test_the_gap_is_paired(scored):
    result = intervals.bootstrap(scored, THRESHOLDS, resamples=100)
    gap = result["cross_language_f1_gap"]
    assert gap["point"] == pytest.approx(
        abs(result["toxicity_f1_en"]["point"] - result["toxicity_f1_ar"]["point"])
    )


@pytest.mark.parametrize(
    "low,high,comparison,target,expected",
    [
        (0.96, 0.98, ">=", 0.95, "pass"),
        (0.94, 0.96, ">=", 0.95, "too close to call"),
        (0.90, 0.93, ">=", 0.95, "fail"),
        (0.03, 0.06, "<=", 0.07, "pass"),
        (0.09, 0.14, "<=", 0.07, "fail"),
    ],
)
def test_verdict(low, high, comparison, target, expected):
    assert intervals.verdict("x", {"low": low, "high": high}, comparison, target) == expected


# --- paired comparison ---------------------------------------------------------


def _with_keys(rows):
    return [{**r, "key": f"k{i}"} for i, r in enumerate(rows)]


def _sharpened(rows, by):
    out = []
    for r in rows:
        shift = by if r["toxicity_true"] else -by
        out.append({**r, "toxicity_score": min(1.0, max(0.0, r["toxicity_score"] + shift))})
    return out


def test_a_run_against_itself_shows_no_difference(scored):
    rows = _with_keys(scored)
    found = intervals.compare(rows, THRESHOLDS, rows, THRESHOLDS, resamples=100)
    for row in found.values():
        assert row["diff"] == 0
        assert row["low"] == row["high"] == 0
        assert row["call"] == "no detectable difference"


def test_a_clearly_better_run_is_called_better(scored):
    a = _with_keys(scored)
    b = _sharpened(a, 0.15)
    found = intervals.compare(a, THRESHOLDS, b, THRESHOLDS, resamples=200)
    assert found["toxicity_auroc_en"]["call"] == "B better"
    assert found["toxicity_auroc_ar"]["call"] == "B better"
    reverse = intervals.compare(b, THRESHOLDS, a, THRESHOLDS, resamples=200)
    assert reverse["toxicity_auroc_en"]["call"] == "A better"


def test_pairing_detects_a_change_the_separate_intervals_cannot(scored):
    a = _with_keys(scored)
    b = _sharpened(a, 0.02)
    paired = intervals.compare(a, THRESHOLDS, b, THRESHOLDS, resamples=300)["toxicity_auroc_en"]
    alone_a = intervals.bootstrap(a, THRESHOLDS, resamples=300)["toxicity_auroc_en"]
    alone_b = intervals.bootstrap(b, THRESHOLDS, resamples=300)["toxicity_auroc_en"]
    # The separate intervals overlap, so unpaired they'd say "can't tell" ...
    assert alone_a["high"] > alone_b["low"]
    # ... but the paired interval on the difference is narrow and above zero.
    assert paired["call"] == "B better"
    assert paired["high"] - paired["low"] < (alone_a["high"] - alone_a["low"])


def test_a_lower_gap_counts_as_better():
    assert intervals.difference_call("cross_language_f1_gap", -0.05, -0.01) == "B better"
    assert intervals.difference_call("toxicity_f1_en", -0.05, -0.01) == "A better"


def test_refuses_to_pair_different_splits(scored):
    a = _with_keys(scored)
    with pytest.raises(ValueError, match="test rows"):
        intervals.compare(a, THRESHOLDS, a[:-1], THRESHOLDS, resamples=10)
    shuffled = [a[1], a[0], *a[2:]]
    with pytest.raises(ValueError, match="differs"):
        intervals.compare(a, THRESHOLDS, shuffled, THRESHOLDS, resamples=10)
    with pytest.raises(ValueError, match="predate"):
        intervals.compare(scored, THRESHOLDS, scored, THRESHOLDS, resamples=10)


def test_common_rows_aligns_the_overlap_of_two_splits(scored):
    rows = _with_keys(scored)
    a = rows[:400]
    b = list(reversed(rows[200:]))  # different split, different order
    b.append({**rows[0], "key": "dup"})
    b.append({**rows[1], "key": "dup"})
    a2, b2 = intervals.common_rows(a, b)
    assert len(a2) == len(b2) == 200
    assert [r["key"] for r in a2] == [r["key"] for r in b2]
    found = intervals.compare(a2, THRESHOLDS, b2, THRESHOLDS, resamples=20)
    assert found["toxicity_auroc_en"]["diff"] == 0


# --- configurations with several runs each --------------------------------------


def test_groups_see_seed_noise_that_a_single_pair_does_not(scored):
    rng = random.Random(9)
    base = _with_keys(scored)

    def run(shift):
        # A run: the same rows, scores nudged by its own seed noise.
        return (
            [
                {
                    **r,
                    "toxicity_score": min(
                        1.0,
                        max(
                            0.0,
                            r["toxicity_score"]
                            + shift * (1 if r["toxicity_true"] else -1)
                            + rng.gauss(0, 0.06),
                        ),
                    ),
                }
                for r in base
            ],
            THRESHOLDS,
        )

    same_a = [run(0.0) for _ in range(4)]
    same_b = [run(0.0) for _ in range(4)]
    found = intervals.compare_groups(same_a, same_b, resamples=200)
    assert found["toxicity_auroc_en"]["call"] == "no detectable difference"
    better = [run(0.12) for _ in range(4)]
    found = intervals.compare_groups(same_a, better, resamples=200)
    assert found["toxicity_auroc_en"]["call"] == "B better"
    assert len(found["toxicity_auroc_en"]["b_runs"]) == 4


def test_groups_need_two_runs_and_the_same_rows(scored):
    rows = _with_keys(scored)
    with pytest.raises(ValueError, match="two runs"):
        intervals.compare_groups([(rows, THRESHOLDS)], [(rows, THRESHOLDS)] * 2, resamples=5)
    with pytest.raises(ValueError, match="same test rows"):
        intervals.compare_groups(
            [(rows, THRESHOLDS)] * 2, [(rows[:-1], THRESHOLDS)] * 2, resamples=5
        )
