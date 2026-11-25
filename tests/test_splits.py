"""Train / dev / test, and thresholds that are never tuned on what they report."""

from __future__ import annotations

import random

import pytest

from fintext import corpora
from fintext.evaluation import GATES, binary_scores, choose_threshold, threshold_sweep
from fintext.training import TrainingConfig, compute_metrics


class TestTheSweep:
    def test_it_agrees_with_scoring_each_threshold_directly(self):
        rng = random.Random(7)
        truth = [rng.randint(0, 1) for _ in range(300)]
        scores = [round(rng.random(), 2) for _ in range(300)]  # rounded, so ties exist
        for threshold, precision, recall, f1 in threshold_sweep(truth, scores):
            assert (precision, recall, f1) == pytest.approx(
                binary_scores(truth, scores, threshold=threshold)
            )

    def test_tied_scores_are_one_threshold(self):
        sweep = threshold_sweep([1, 0, 1], [0.8, 0.8, 0.3])
        assert [row[0] for row in sweep] == [0.8, 0.3]

    def test_no_positives_means_nothing_to_tune(self):
        assert threshold_sweep([0, 0, 0], [0.1, 0.5, 0.9]) == []


class TestChoosingAThreshold:
    def test_when_both_floors_can_be_met_it_takes_the_best_f1_among_them(self):
        truth = [1, 1, 1, 1, 0, 0, 0, 0]
        scores = [0.9, 0.85, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1]
        chosen = choose_threshold(truth, scores, min_recall=0.9, min_precision=0.9)
        assert chosen.feasible
        assert chosen.f1 == pytest.approx(1.0)

    def test_when_they_cannot_recall_wins_and_it_says_so(self):
        truth = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        scores = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        chosen = choose_threshold(truth, scores, min_recall=1.0, min_precision=0.9)
        assert not chosen.feasible
        assert chosen.recall == 1.0
        assert chosen.f1 > 0
        assert "recall-first" in chosen.basis

    def test_the_ceiling_ignores_the_floors(self):
        truth = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        scores = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        chosen = choose_threshold(truth, scores, min_recall=1.0, min_precision=0.9)
        best_anywhere = max(row[3] for row in threshold_sweep(truth, scores))
        assert chosen.ceiling_f1 == pytest.approx(best_anywhere)
        assert chosen.ceiling_f1 >= chosen.f1


def collected(truth: list[int], scores: list[float], language: str = "en") -> dict[str, list]:
    return {f"toxicity_true_{language}": truth, f"toxicity_score_{language}": scores}


class TestTuneOnDevReportOnTest:
    def test_the_reported_number_comes_from_test_at_the_dev_threshold(self):
        """Dev and test disagree about where the threshold should sit."""
        dev = collected([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.2])
        test = collected([1, 1, 0, 0], [0.6, 0.5, 0.7, 0.1])
        config = TrainingConfig(recall_floor=1.0, precision_floor=1.0)

        metrics = compute_metrics(dev, test, config)
        threshold = metrics["thresholds"]["toxicity_en"]
        assert threshold == 0.8  # chosen on dev
        assert metrics["toxicity_f1_en"] == pytest.approx(
            binary_scores([1, 1, 0, 0], [0.6, 0.5, 0.7, 0.1], threshold=0.8)[2]
        )
        assert metrics["protocol"]["thresholds_tuned_on"] == "dev"
        assert metrics["protocol"]["metrics_reported_on"] == "test"

    def test_auroc_and_sentiment_come_from_test(self):
        dev = {
            **collected([1, 0], [0.9, 0.1]),
            "sentiment_true_en": ["positive"],
            "sentiment_pred_en": ["negative"],
        }
        test = {
            **collected([1, 0], [0.1, 0.9]),
            "sentiment_true_en": ["positive"],
            "sentiment_pred_en": ["positive"],
        }
        metrics = compute_metrics(dev, test, TrainingConfig())
        assert metrics["toxicity_auroc_en"] == pytest.approx(0.0)  # test's, not dev's 1.0
        assert metrics["sentiment_macro_f1_en"] > 0  # test got it right; dev did not

    def test_infeasibility_is_reported_not_hidden(self):
        dev = collected(
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        )
        metrics = compute_metrics(dev, dev, TrainingConfig(recall_floor=1.0, precision_floor=0.9))
        assert metrics["toxicity_feasible_en"] is False
        assert metrics["toxicity_f1_en"] > 0
        assert "recall-first" in metrics["threshold_detail"]["toxicity_en"]["basis"]

    def test_the_gap_uses_the_name_the_gate_reads(self):
        both = {**collected([1, 0], [0.9, 0.1], "en"), **collected([1, 0], [0.9, 0.1], "ar")}
        metrics = compute_metrics(both, both, TrainingConfig(recall_floor=0.5, precision_floor=0.5))
        assert "cross_language_f1_gap" in metrics
        assert "cross_language_f1_gap" in GATES


@pytest.fixture
def fake_sources(monkeypatch):
    """Deterministic in-memory loaders, so the split is tested without network."""

    def rows(prefix: str, n: int, language: str = "en", task: str = "sentiment"):
        return lambda limit: [
            {"text": f"{prefix} {i}", "language": language, "labels": {task: "neutral"}}
            for i in range(n)
        ]

    monkeypatch.setitem(corpora.LOADERS, "financial_phrasebank", rows("pb", 120))
    monkeypatch.setitem(corpora.LOADERS, "liar", rows("liar", 80, task="misleading"))
    monkeypatch.setitem(corpora.EVAL_ONLY_LOADERS, "arsas", rows("ar", 41, "ar"))
    return ("financial_phrasebank", "liar")


class TestTheSplit:
    def test_three_splits_and_no_overlap(self, fake_sources):
        split, _ = corpora.load_rows(fake_sources, eval_sources=("arsas",))
        texts = {name: {r["text"] for r in rows} for name, rows in split.items()}
        assert set(split) == {"train", "dev", "test"}
        assert not texts["train"] & texts["dev"]
        assert not texts["train"] & texts["test"]
        assert not texts["dev"] & texts["test"]

    def test_the_train_set_is_what_it_always_was(self, fake_sources):
        split, _ = corpora.load_rows(fake_sources, eval_sources=("arsas",), seed=11)
        everything = corpora.LOADERS["financial_phrasebank"](None) + corpora.LOADERS["liar"](None)
        random.Random(11).shuffle(everything)
        expected = everything[: int(len(everything) * 0.85)]
        assert [r["text"] for r in split["train"]] == [r["text"] for r in expected]

    def test_every_row_says_which_corpus_it_came_from(self, fake_sources):
        split, _ = corpora.load_rows(fake_sources, eval_sources=("arsas",))
        corpora_seen = {r["corpus"] for rows in split.values() for r in rows}
        assert corpora_seen == {"financial_phrasebank", "liar", "arsas"}

    def test_eval_only_rows_never_reach_training_and_are_halved(self, fake_sources):
        split, _ = corpora.load_rows(fake_sources, eval_sources=("arsas",))
        arabic = {name: sum(r["language"] == "ar" for r in rows) for name, rows in split.items()}
        assert arabic["train"] == 0
        assert arabic["dev"] + arabic["test"] == 41
        assert abs(arabic["dev"] - arabic["test"]) <= 1

    def test_the_split_is_reproducible_and_fingerprinted(self, fake_sources):
        first, a = corpora.load_rows(fake_sources, eval_sources=("arsas",))
        second, b = corpora.load_rows(fake_sources, eval_sources=("arsas",))
        assert first == second
        assert a["totals"]["train_fingerprint"] == b["totals"]["train_fingerprint"]
        assert {"train", "dev", "test"} <= set(a["totals"])
