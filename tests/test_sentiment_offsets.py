"""Class offsets recover a class the model ranks well but under-predicts."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from fintext import sentiment_offsets as so  # noqa: E402


def _scored(n=900, neutral_scale=0.25, seed=5):
    """Neutral rows get the highest neutral probability, but it's always too low to win."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        truth = so.LABELS[i % 3]
        p = [0.2, 0.2, 0.2]
        p[so.LABELS.index(truth)] += 0.5 + rng.random() * 0.2
        p[2] *= neutral_scale  # the shift: neutral is ranked right, scaled down
        if truth == "neutral":
            p[rng.randrange(2)] += 0.05  # so without offsets neutral rarely wins
        total = sum(p)
        rows.append(
            {
                "language": "ar",
                "sentiment_true": truth,
                "sentiment_probs": [x / total for x in p],
                "sentiment_pred": so.LABELS[max(range(3), key=lambda k: p[k])],
            }
        )
    return rows


def test_ranking_survives_the_shift():
    ranking = so.neutral_ranking(_scored())
    assert ranking["neutral"] > 0.95


def test_offsets_tuned_on_dev_fix_test():
    dev, test = _scored(seed=1), _scored(seed=2)
    tuned = so.tune(dev)
    assert tuned["offsets"]["neutral"] > 0
    assert tuned["dev_macro_f1"] > tuned["dev_macro_f1_without"] + 0.2
    fixed = so.apply(test, tuned["offsets"])
    truth = [so.LABELS.index(r["sentiment_true"]) for r in test]
    before = so.macro_f1(truth, [so.LABELS.index(r["sentiment_pred"]) for r in test])
    after = so.macro_f1(truth, [so.LABELS.index(r["sentiment_pred"]) for r in fixed])
    assert after > before + 0.2


def test_apply_leaves_other_languages_and_the_input_alone():
    rows = _scored(30) + [
        {
            "language": "en",
            "sentiment_true": "neutral",
            "sentiment_probs": [0.5, 0.4, 0.1],
            "sentiment_pred": "positive",
        }
    ]
    before = [dict(r) for r in rows]
    fixed = so.apply(rows, {"positive": 0.0, "negative": 0.0, "neutral": 5.0})
    assert rows == before
    assert fixed[-1]["sentiment_pred"] == "positive"
    assert all(r["sentiment_pred"] == "neutral" for r in fixed[:-1])


def test_needs_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        so.tune([{"language": "ar", "sentiment_true": "neutral", "sentiment_pred": "neutral"}])
