"""PRD conformance: one test per product commitment."""

from __future__ import annotations

import pytest
from conftest import T0, make_document

from fintext import catalogue
from fintext.schemas import Capability, Language, Profile, ProvenanceState, ScoreRequest, Sentiment


class TestTargets:
    def test_english_sentiment_is_gated_and_arabic_is_reported(self, pipeline):
        english = pipeline.score(ScoreRequest(text="Profit rose sharply"))
        arabic = pipeline.score(ScoreRequest(text="ارتفاع أرباح البنك بشكل كبير"))
        assert english.sentiment.acceptance_gated
        assert not arabic.sentiment.acceptance_gated

    @pytest.mark.parametrize(
        ("text", "language"),
        [("i will kill you", Language.EN), ("سهم البنك __ar_slur_placeholder_1__", Language.AR)],
    )
    def test_toxicity_is_delivered_in_both_languages(self, pipeline, text, language):
        response = pipeline.score(ScoreRequest(text=text))
        assert response.language is language and response.toxicity.toxic


class TestScope:
    def test_the_service_emits_scores_and_takes_no_action(self, pipeline):
        payload = pipeline.score(ScoreRequest(text="Guaranteed 300% returns, act now")).to_dict()
        assert not any(k in payload for k in ("action", "decision", "block", "suppress", "order"))

    def test_no_private_data_source_is_declared(self):
        external = [d for d in catalogue.CATALOGUE if not d.internal]
        assert all(d.url.startswith("http") for d in external)

        internal = [d for d in catalogue.CATALOGUE if d.internal]
        assert [d.key for d in internal] == ["production_capture"]
        assert all(not d.redistributable and d.status == catalogue.PENDING for d in internal)
        reddit = catalogue.get("reddit")
        assert reddit.uses == ("backtest",) and not reddit.redistributable

    def test_misleading_content_is_english_only(self, pipeline):
        arabic = pipeline.score(ScoreRequest(text="ارتفاع أرباح البنك بشكل كبير"))
        assert arabic.misleading is None and "misleading" in arabic.skipped


class TestCapabilities:
    def test_neutral_suppresses_rather_than_defaults(self, pipeline):
        response = pipeline.score(ScoreRequest(text="The annual general meeting is on Tuesday"))
        assert response.sentiment.label is Sentiment.NEUTRAL
        assert response.sentiment.scores["neutral"] > 0.5

    def test_lexicon_match_overrides_the_model(self, pipeline):
        response = pipeline.score(ScoreRequest(text="great quarter, anyway k.y.s"))
        assert response.toxicity.toxic and response.toxicity.lexicon_override

    def test_subtype_tags_accompany_the_binary_flag(self, pipeline):
        response = pipeline.score(ScoreRequest(text="i will kill you"))
        assert set(response.toxicity.subtypes) == {"hate", "harassment", "profanity", "threat"}
        assert response.toxicity.subtypes["threat"] > 0.5

    def test_misleading_output_is_always_needs_review(self, pipeline):
        response = pipeline.score(ScoreRequest(text="Guaranteed risk-free returns of 40% a month"))
        assert response.misleading.disposition == "likely_misleading_needs_review"

    def test_provenance_returns_one_of_four_states_and_no_truth_score(self, pipeline):
        response = pipeline.score(ScoreRequest(text="Markets were quiet"))
        assert response.provenance.state in set(ProvenanceState)
        assert response.provenance.asserts_falsity is False


class TestUsers:
    def test_profiles_get_different_budgets(self, pipeline):
        assert pipeline.config.budget_ms("trading") == 50.0
        assert pipeline.config.budget_ms("compliance") == 500.0

    def test_every_flag_carries_a_plain_explanation(self, pipeline):
        response = pipeline.score(ScoreRequest(text="i will kill you", profile=Profile.COMPLIANCE))
        assert response.toxicity.explanation.summary and response.toxicity.explanation.spans

    def test_a_pinned_version_returns_identical_scores(self, pipeline):
        request = ScoreRequest(
            text="Costs fell and margins improved",
            profile=Profile.RESEARCH,
            model_version="pinned.v1",
        )
        first, second = pipeline.score(request), pipeline.score(request)
        assert first.sentiment == second.sentiment and first.lineage == second.lineage


class TestTechnicalApproach:
    def test_one_backend_serves_every_head(self, pipeline):
        response = pipeline.score(
            ScoreRequest(text="FAB agreed to acquire a lender; guaranteed returns")
        )
        assert response.sentiment and response.toxicity and response.misleading
        assert response.lineage.model_version == pipeline.backend.version

    def test_pending_is_the_normal_condition_of_breaking_news(self, pipeline):
        response = pipeline.score(
            ScoreRequest(text="FAB raises its full-year guidance", observed_at=T0)
        )
        assert response.provenance.state is ProvenanceState.PENDING
        assert "no matching primary document found yet" in response.provenance.reason

    def test_uncorroborated_is_the_state_that_routes_to_review(self, pipeline, clock):
        pipeline.score(ScoreRequest(text="SEC charges Acme Corp", observed_at=T0))
        clock.advance(pipeline.config.window("regulatory") + 60)
        pipeline.tracker.recheck_due(now=clock.now)
        assert len(pipeline.tracker.review_queue()) == 1

    def test_corroboration_needs_a_primary_source(self, pipeline):
        pipeline.index.add(make_document(doc_id="wire-1", source_type="news_wire"))
        response = pipeline.score(
            ScoreRequest(
                text="First Abu Dhabi Bank PJSC agreed to acquire a lender", observed_at=T0
            )
        )
        assert response.provenance.state is ProvenanceState.PENDING

    def test_lexicons_are_versioned_like_models(self, pipeline):
        versions = pipeline.lineage("m").lexicon_versions
        assert versions["en"] and versions["ar"] and "rows" in versions["en"]

    def test_the_token_cap_is_enforced(self, pipeline):
        assert pipeline.score(ScoreRequest(text="word " * 2000)).truncated
        assert pipeline.config.max_tokens == 256


class TestAcceptance:
    def test_latency_stays_inside_the_budget(self, pipeline):
        for _ in range(20):
            pipeline.score(ScoreRequest(text="word " * 250))
        assert pipeline.monitor.latency.quantile(0.95) < 50.0

    def test_the_two_degradation_policies_are_distinct(self, pipeline):
        config = pipeline.config
        assert "trading" in config.fail_open and "compliance" in config.fail_closed
        assert not set(config.fail_open) & set(config.fail_closed)

    def test_thresholds_are_recall_first(self, pipeline):
        assert pipeline.config.toxicity_en < 0.5 and pipeline.config.toxicity_ar < 0.5

    @pytest.mark.parametrize(
        "text", ["You are a worthless piece of shit", "kill yourself", "i will k1ll you"]
    )
    def test_clear_abuse_is_flagged(self, pipeline, text):
        assert pipeline.score(ScoreRequest(text=text)).toxicity.toxic

    @pytest.mark.parametrize(
        "text",
        [
            "First Abu Dhabi Bank reported a 12% rise in first-quarter net profit",
            "The board approved a dividend of 0.35 per share",
            "Analysts remain divided on the outlook for the sector",
            "The stock was hammered after the profit warning",
        ],
    )
    def test_ordinary_financial_text_is_not_flagged(self, pipeline, text):
        response = pipeline.score(ScoreRequest(text=text))
        assert not response.toxicity.toxic and not response.misleading.likely_misleading


class TestArchitecture:
    def test_two_replicas_agree(self, config, index):
        from fintext.pipeline import Pipeline

        first, second = Pipeline(config, index=index), Pipeline(config, index=index)
        request = ScoreRequest(text="Revenue rose sharply", id="fixed")
        assert first.score(request).sentiment == second.score(request).sentiment
        first.close()
        second.close()

    def test_any_flag_traces_to_what_produced_it(self, pipeline):
        """9: the immutable registry supports audit."""
        lineage = pipeline.score(
            ScoreRequest(text="i will kill you", profile=Profile.COMPLIANCE)
        ).lineage
        assert all(
            [
                lineage.model_version,
                lineage.lexicon_versions,
                lineage.threshold_version,
                lineage.rules_version,
                lineage.config_digest,
            ]
        )

    def test_capabilities_are_independently_requestable(self, pipeline):
        response = pipeline.score(
            ScoreRequest(
                text="FAB raises its full-year guidance",
                capabilities=(Capability.SENTIMENT, Capability.PROVENANCE),
            )
        )
        assert response.sentiment and response.provenance
        assert response.toxicity is None and response.misleading is None
