"""Loading the PRD's declared sources, and keeping the eval-only ones out."""

from __future__ import annotations

import pytest

from fintext import catalogue, corpora


class TestTheCatalogueAndTheLoaders:
    def test_every_loader_is_a_declared_source(self):
        for key in [*corpora.LOADERS, *corpora.EVAL_ONLY_LOADERS, *corpora.TRAIN_ONLY_LOADERS]:
            assert catalogue.get(key)

    def test_every_training_loader_is_cleared_to_train(self):
        catalogue.check_training_sources(list(corpora.LOADERS))

    def test_eval_only_loaders_are_not_cleared_to_train(self):
        for key in corpora.EVAL_ONLY_LOADERS:
            assert catalogue.TRAIN not in catalogue.get(key).uses

    def test_every_loader_has_a_mirror_recorded(self):
        for key in list(corpora.LOADERS) + list(corpora.EVAL_ONLY_LOADERS):
            assert corpora.MIRRORS[key].path


class TestKeepingEvalOnlySourcesOut:
    def test_a_source_cannot_be_both(self):
        with pytest.raises(corpora.CorpusError, match="cannot also be a training source"):
            corpora.load_rows(("financial_phrasebank",), eval_sources=("financial_phrasebank",))

    def test_an_unknown_eval_source_is_refused(self):
        with pytest.raises(catalogue.LicenceError, match="unknown dataset"):
            corpora.load_rows(("financial_phrasebank",), eval_sources=("nonesuch",))


class TestLabelMapping:
    @pytest.mark.parametrize(
        "score,expected",
        [(0.9, "positive"), (-0.374, "negative"), (0.05, "neutral"), (-0.05, "neutral")],
    )
    def test_fiqa_scores_cut_into_three_labels(self, score, expected):
        band = corpora.FIQA_DEAD_BAND
        got = "positive" if score > band else "negative" if score < -band else "neutral"
        assert got == expected

    def test_arsas_mixed_is_dropped_not_folded_into_neutral(self):
        assert set(corpora.ARSAS_LABELS) == {0, 1, 2}
        assert corpora.ARSAS_LABELS[1] == "neutral"

    def test_liar_splits_a_six_point_scale_in_half(self):
        assert sorted(corpora.LIAR_MISLEADING) == [0, 1, 2]


class TestTheArabicSentimentGap:
    def test_arsas_is_the_sanity_check_and_never_trained_on(self):
        arsas = catalogue.get("arsas")
        assert arsas.uses == ("eval",)
        assert "arsas" in corpora.EVAL_ONLY_LOADERS
        assert "arsas" not in corpora.LOADERS

    def test_arabic_sentiment_trains_only_on_train_only_sources(self):
        """The PRD trains Arabic sentiment on machine-translated copies of the English sets."""
        trainers = [
            key
            for key in [*corpora.LOADERS, *corpora.TRAIN_ONLY_LOADERS]
            if "sentiment" in catalogue.get(key).tasks and "ar" in catalogue.get(key).languages
        ]
        assert sorted(trainers) == ["arsarcasm_v2", "translated_sentiment_ar"]
        assert all(key in corpora.TRAIN_ONLY_LOADERS for key in trainers)
        assert all(catalogue.get(key).uses == (catalogue.TRAIN,) for key in trainers)
        assert "arsas" not in trainers
