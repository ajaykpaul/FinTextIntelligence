"""The checks every run goes through before it is written up or registered."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from fintext import cli, release_checks  # noqa: E402
from fintext.intervals import SENTIMENT  # noqa: E402


def test_a_copy_hiding_behind_a_url_and_elongation_is_found():
    held = ["السهم ارتفع كثيييير اليوم https://t.co/abc", "مباراة كرة القدم انتهت بالتعادل"]
    source = ["@user السهم ارتفع كثير اليوم https://t.co/xyz", "الطقس جميل في الربيع"]
    best, which = release_checks.near_duplicates(held, source)
    assert best[0] > 0.95 and which[0] == 0
    assert best[1] < 0.5


def test_empty_sides_flag_nothing():
    assert release_checks.near_duplicates(["a"], []) == ([0.0], [-1])
    assert release_checks.near_duplicates([], ["a"]) == ([], [])


def test_summary_counts_each_cut():
    found = release_checks.summarise([0.1, 0.55, 0.65, 0.95])
    assert found["max"] == 0.95
    assert found["at_or_above"]["0.5"] == 3
    assert found["at_or_above"]["0.9"] == 1


def _scored(n=200, seed=1):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        truth = rng.choice(SENTIMENT)
        rows.append(
            {
                "key": f"k{i}",
                "language": "ar",
                "sentiment_true": truth,
                "sentiment_pred": truth if rng.random() < 0.7 else rng.choice(SENTIMENT),
                "sentiment_probs": [0.5, 0.3, 0.2],
            }
        )
    return rows


def test_metrics_without_drops_exactly_the_flagged_rows():
    rows = _scored()
    for row in rows[:20]:  # make the flagged rows all wrong
        row["sentiment_pred"] = next(s for s in SENTIMENT if s != row["sentiment_true"])
    effect = release_checks.metrics_without(rows, {}, {r["key"] for r in rows[:20]})
    assert (
        effect["without_flagged"]["sentiment_macro_f1_ar"] > effect["all"]["sentiment_macro_f1_ar"]
    )
    assert (
        release_checks.metrics_without(rows, {}, set())["all"]
        == (release_checks.metrics_without(rows, {}, set())["without_flagged"])
    )


def test_expected_label_applies_only_that_languages_offsets():
    record = {"language": "ar", "sentiment_pred": "positive", "sentiment_probs": [0.5, 0.3, 0.2]}
    assert release_checks.expected_label(record, {}) == "positive"
    lifted = {"ar": {"positive": 0.0, "negative": 0.0, "neutral": 2.0}}
    assert release_checks.expected_label(record, lifted) == "neutral"
    assert release_checks.expected_label({**record, "language": "en"}, lifted) == "positive"


def test_serving_parity_counts_matches_and_score_drift():
    rows = [{"text": f"t{i}", "language": "ar"} for i in range(10)]
    rows += [{"text": f"e{i}", "language": "en"} for i in range(5)]
    scored = {}
    for row in rows:
        record = {"language": row["language"], "sentiment_pred": "negative", "toxicity_score": 0.25}
        if row["language"] == "ar":
            record["sentiment_true"] = "negative"
        scored[row["text"]] = record

    def predict(text, language):
        label = "positive" if text == "t3" else "negative"
        return SimpleNamespace(
            top_sentiment=lambda: (SimpleNamespace(value=label), 0.9),
            toxicity=0.25 + (0.01 if text == "e2" else 0.0),
            misleading=0.0,
        )

    found = release_checks.serving_parity(
        predict, rows, scored, lambda r: r["text"], {}, sentiment_sample=10, score_sample=5
    )
    assert found["sentiment_rows"] == 10
    assert found["sentiment_label_matches"] == 9
    assert found["served_sentiment"] == {"positive": 1, "negative": 9, "neutral": 0}
    assert found["score_rows"] == 5
    assert found["max_score_delta"] == pytest.approx(0.01)


def _write_run(directory, name, rows):
    directory.mkdir()
    lines = [json.dumps({"run": name, "thresholds": {}})] + [json.dumps(r) for r in rows]
    (directory / "scored_test.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_compare_groups_command_writes_its_result(tmp_path, capsys):
    paths = {}
    for group, accuracy in (("a", 0.5), ("b", 0.9)):
        for seed in (1, 2):
            rng = random.Random(seed)
            rows = []
            for i in range(300):
                truth = SENTIMENT[i % 3]
                right = rng.random() < accuracy
                rows.append(
                    {
                        "key": f"k{i}",
                        "language": "ar",
                        "sentiment_true": truth,
                        "sentiment_pred": truth if right else SENTIMENT[(i + 1) % 3],
                    }
                )
            path = tmp_path / f"{group}{seed}"
            _write_run(path, f"{group}{seed}", rows)
            paths.setdefault(group, []).append(str(path))
    out = tmp_path / "groups.json"
    code = cli.main(
        [
            "compare-groups",
            "--a",
            *paths["a"],
            "--b",
            *paths["b"],
            "--resamples",
            "50",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    result = json.loads(out.read_text())
    assert result["metrics"]["sentiment_macro_f1_ar"]["call"] == "B better"
    assert "B better" in capsys.readouterr().out


def test_compare_groups_refuses_a_missing_score_file(tmp_path, capsys):
    (tmp_path / "x").mkdir()
    assert cli.main(["compare-groups", "--a", str(tmp_path / "x"), "--b", str(tmp_path / "x")]) == 1
    assert "missing" in capsys.readouterr().out


def _scores(toxicity, misleading, label):
    probs = {"positive": 0.1, "negative": 0.1, "neutral": 0.1, label: 0.8}
    return SimpleNamespace(
        toxicity=toxicity,
        misleading=misleading,
        sentiment=probs,
        top_sentiment=lambda: (label, 0.8),
    )


def test_quantisation_parity_counts_decision_flips_at_the_thresholds():
    rows = [
        {"text": "a", "language": "en"},
        {"text": "b", "language": "ar"},
        {"text": "c", "language": "ar-lev"},
    ]
    reference = {
        "a": _scores(0.50, 0.40, "neutral"),
        "b": _scores(0.20, 0.0, "negative"),
        "c": _scores(0.90, 0.0, "positive"),
    }
    # a: toxicity crosses the EN cut; b: sentiment flips; c: moves but stays flagged.
    candidate = {
        "a": _scores(0.60, 0.41, "neutral"),
        "b": _scores(0.21, 0.0, "neutral"),
        "c": _scores(0.80, 0.0, "positive"),
    }
    found = release_checks.quantisation_parity(
        lambda t, _l: reference[t],
        lambda t, _l: candidate[t],
        rows,
        {"toxicity_en": 0.55, "toxicity_ar": 0.25, "misleading_en": 0.5},
    )
    assert found["rows"] == 3 and found["by_language"] == {"en": 1, "ar": 2}
    assert found["decision_flips"] == {"toxicity": 1, "misleading": 0, "sentiment": 1}
    assert found["max_delta"]["toxicity"] == pytest.approx(0.10)
    assert found["max_delta"]["sentiment"] == pytest.approx(0.7)
