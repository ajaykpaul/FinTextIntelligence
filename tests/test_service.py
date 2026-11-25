"""Pipeline behaviour and the HTTP contract."""

from __future__ import annotations

import json

import pytest
from conftest import T0, make_document

from fintext.pipeline import Budget, Pipeline, ScoringFailed
from fintext.schemas import Capability, Language, Profile, ProvenanceState, ScoreRequest, Sentiment
from fintext.scoring import Heuristic, Scores
from fintext.server import App, parse_json


class Broken(Heuristic):
    version = "broken.v0"

    def predict(self, text, language):
        raise RuntimeError("backend unavailable")


class NoSignal(Heuristic):
    """A model that has lost all signal, to prove the rules stand alone."""

    version = "zero.v0"

    def predict(self, text, language) -> Scores:
        scores = super().predict(text, language)
        return Scores(scores.sentiment, 0.0, dict.fromkeys(scores.subtypes, 0.0), 0.0, self.version)


def post(app: App, path: str, payload: dict):
    return app.handle("POST", path, json.dumps(payload))


class TestScoring:
    def test_scores_every_requested_capability(self, pipeline):
        response = pipeline.score(
            ScoreRequest(text="First Abu Dhabi Bank PJSC agreed to acquire a lender")
        )
        assert all(
            [response.sentiment, response.toxicity, response.misleading, response.provenance]
        )
        assert response.degraded == ()

    def test_capability_selection_is_honoured(self, pipeline):
        response = pipeline.score(
            ScoreRequest(text="Revenue rose sharply", capabilities=(Capability.SENTIMENT,))
        )
        assert response.sentiment and response.toxicity is None
        assert response.skipped["toxicity"] == "not requested"

    def test_finance_specific_sentiment(self, pipeline):
        # The PRD's own example: a general-purpose model gets this wrong.
        good = pipeline.score(ScoreRequest(text="Liabilities fell sharply this quarter"))
        bad = pipeline.score(ScoreRequest(text="Revenue fell sharply this quarter"))
        assert good.sentiment.label is Sentiment.POSITIVE
        assert bad.sentiment.label is Sentiment.NEGATIVE

    def test_neutral_is_a_substantive_output(self, pipeline):
        response = pipeline.score(ScoreRequest(text="The company will publish results on Tuesday"))
        assert response.sentiment.label is Sentiment.NEUTRAL

    def test_arabic_sentiment_is_reported_but_not_gated(self, pipeline):
        response = pipeline.score(ScoreRequest(text="ترتفع أسهم البنك بعد نمو الأرباح"))
        assert response.language is Language.AR
        assert response.sentiment and response.sentiment.acceptance_gated is False

    def test_english_sentiment_is_gated(self, pipeline):
        assert pipeline.score(ScoreRequest(text="Revenue rose sharply")).sentiment.acceptance_gated

    def test_misleading_is_skipped_for_arabic_not_failed(self, pipeline):
        # English-only scope must not look like a partial outage.
        response = pipeline.score(ScoreRequest(text="ترتفع أسهم البنك بعد نمو الأرباح"))
        assert response.misleading is None
        assert "English-only" in response.skipped["misleading"]
        assert "misleading" not in response.degraded

    def test_lexicon_floor_overrides_a_low_model_score(self, pipeline):
        response = pipeline.score(ScoreRequest(text="nice quarter, now k.y.s"))
        assert response.toxicity.toxic and response.toxicity.lexicon_override
        assert response.toxicity.explanation.spans

    def test_scam_rules_flag_without_the_model(self, config, index):
        # If the head regresses or is cut to rules-only, this coverage survives.
        pipeline = Pipeline(config, backend=NoSignal(), index=index)
        response = pipeline.score(
            ScoreRequest(text="Guaranteed 300% returns. Hurry, minimum deposit 500 USDT.")
        )
        assert response.misleading.likely_misleading and response.misleading.rule_matches
        pipeline.close()

    def test_flagged_output_never_asserts_falsity(self, pipeline):
        response = pipeline.score(ScoreRequest(text="Guaranteed returns, risk-free investment"))
        assert response.misleading.disposition == "likely_misleading_needs_review"
        assert "not an assessment of truth" in response.misleading.explanation.summary

    def test_provenance_states(self, pipeline):
        assert (
            pipeline.score(ScoreRequest(text="Markets were quiet on Tuesday")).provenance.state
            is ProvenanceState.NOT_APPLICABLE
        )
        assert (
            pipeline.score(
                ScoreRequest(text="FAB raises its full-year guidance", observed_at=T0)
            ).provenance.state
            is ProvenanceState.PENDING
        )

    def test_corroborates_once_the_filing_is_indexed(self, pipeline):
        pipeline.index.add(make_document())
        response = pipeline.score(
            ScoreRequest(
                text="First Abu Dhabi Bank PJSC agreed to acquire a lender", observed_at=T0
            )
        )
        assert response.provenance.state is ProvenanceState.CORROBORATED
        assert response.provenance.evidence

    def test_truncates_at_the_token_cap(self, pipeline):
        assert pipeline.score(ScoreRequest(text="word " * 1000)).truncated

    def test_records_per_stage_latency(self, pipeline):
        response = pipeline.score(ScoreRequest(text="Revenue rose sharply"))
        assert {"preprocess", "inference", "sentiment"} <= set(response.stage_latency_ms)


class TestLineage:
    def test_every_response_carries_full_lineage(self, pipeline):
        lineage = pipeline.score(ScoreRequest(text="Revenue rose")).lineage
        assert lineage.model_version and lineage.threshold_version and lineage.rules_version
        assert lineage.config_digest.startswith("sha256:")
        assert set(lineage.lexicon_versions) == {"en", "ar"}

    def test_a_config_change_changes_the_digest(self, config, index):
        first = Pipeline(config, index=index)
        second = Pipeline(config.replace(toxicity_en=0.9), index=index)
        assert first.lineage("v").config_digest != second.lineage("v").config_digest
        first.close()
        second.close()

    def test_a_pinned_version_is_echoed(self, pipeline):
        response = pipeline.score(ScoreRequest(text="Revenue rose", model_version="pinned.v3"))
        assert response.lineage.model_version == "pinned.v3"

    def test_batch_scoring_is_reproducible(self, pipeline):
        requests = [
            ScoreRequest(text=text, id=f"i{n}", profile=Profile.RESEARCH)
            for n, text in enumerate(["Revenue rose", "Costs fell", "Results on Tuesday"])
        ]
        first = pipeline.score_many(requests)
        second = pipeline.score_many(requests)
        assert [r.sentiment for r in first] == [r.sentiment for r in second]


class TestDegradation:
    def test_trading_fails_open(self, config, index):
        # An outage removes the gate rather than halting trading.
        pipeline = Pipeline(config, backend=Broken(), index=index)
        response = pipeline.score(ScoreRequest(text="Revenue rose", profile=Profile.TRADING))
        assert response.sentiment is None
        assert set(response.degraded) == {"sentiment", "toxicity", "misleading", "provenance"}
        pipeline.close()

    def test_compliance_fails_closed(self, config, index):
        # The queue backs up rather than passing unscreened content.
        pipeline = Pipeline(config, backend=Broken(), index=index)
        with pytest.raises(ScoringFailed, match="fail-closed"):
            pipeline.score(ScoreRequest(text="abuse", profile=Profile.COMPLIANCE))
        pipeline.close()

    def test_a_stage_failure_degrades_only_that_stage(self, pipeline, monkeypatch):
        def boom(_text):
            raise RuntimeError("claim extraction failed")

        monkeypatch.setattr(pipeline.claims, "extract", boom)
        response = pipeline.score(ScoreRequest(text="FAB raises its full-year guidance"))
        assert response.degraded == ("provenance",)
        assert response.sentiment and response.toxicity

    def test_an_exhausted_budget_drops_the_last_stage(self, config, index):
        pipeline = Pipeline(config.replace(trading_budget_ms=0.0), index=index)
        response = pipeline.score(ScoreRequest(text="FAB raises its full-year guidance"))
        assert "provenance" in response.degraded
        pipeline.close()

    def test_budget_tracks_time(self):
        assert not Budget(50.0).exhausted()
        assert Budget(0.0).exhausted()


class TestDecisionLog:
    def _records(self, config, kind="score"):
        from pathlib import Path

        lines = Path(config.decision_log).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if json.loads(line)["kind"] == kind]

    def test_compliance_flags_carry_the_triggering_text(self, pipeline, config):
        pipeline.score(ScoreRequest(text="i will kill you", profile=Profile.COMPLIANCE))
        pipeline.log.flush()
        assert self._records(config)[-1]["text"] == "i will kill you"

    def test_flags_carry_their_text_on_every_profile(self, pipeline, config):
        # The weekly review sample draws from this log, and "was this flag correct" cannot be
        # answered without the item, so a trading-path flag is retained too, not only a
        # compliance one.
        pipeline.score(ScoreRequest(text="i will kill you", profile=Profile.TRADING))
        pipeline.log.flush()
        assert self._records(config)[-1]["text"] == "i will kill you"

    def test_unflagged_items_carry_no_text_when_capture_is_off(self, config, index):
        from fintext.pipeline import Pipeline

        minimal = Pipeline(config.replace(retain_text_for_training=False), index=index)
        minimal.score(ScoreRequest(text="The board approved a dividend", profile=Profile.TRADING))
        minimal.log.flush()
        assert "text" not in self._records(config)[-1]
        minimal.close()

    def test_capture_retains_everything_so_there_is_a_corpus_to_train_on(self, pipeline, config):
        assert config.retain_text_for_training
        pipeline.score(ScoreRequest(text="The board approved a dividend", profile=Profile.TRADING))
        pipeline.log.flush()
        assert self._records(config)[-1]["text"] == "The board approved a dividend"

    def test_logged_text_is_already_scrubbed(self, pipeline, config):
        pipeline.score(
            ScoreRequest(
                text="i will kill you, reach me at a.b@example.com", profile=Profile.COMPLIANCE
            )
        )
        pipeline.log.flush()
        record = self._records(config)[-1]
        assert "a.b@example.com" not in record["text"] and "[EMAIL]" in record["text"]


class TestHttp:
    def test_the_ui_bundle_is_served(self, app):
        content_type, body = app.static_file("index.html")
        assert content_type.startswith("text/html")
        assert body.lstrip().startswith(b"<!doctype html>")

    def test_the_ui_bundle_loads_only_local_assets(self, app):
        # No CDN: the UI has to work on a laptop with no network, same as the rest of the
        # service.
        page = app.static_file("index.html")[1].decode()
        assert "//" not in page.replace("<!doctype", "").replace("https://www.w3.org", "")

    def test_ui_assets_are_readable(self, app):
        from fintext.server import STATIC_DIR

        for asset in (STATIC_DIR / "assets").iterdir():
            content_type, body = app.static_file(f"assets/{asset.name}")
            assert body and content_type.startswith(("text/", "image/", "font/"))

    def test_static_file_refuses_path_traversal(self, app):
        assert app.static_file("../config.py") is None
        assert app.static_file("assets/../../schemas.py") is None

    def test_static_file_returns_none_for_a_missing_file(self, app):
        assert app.static_file("assets/nope.js") is None

    def test_root_lists_the_endpoints(self, app):
        status, body = app.handle("GET", "/")
        assert status == 200 and body["service"] == "fintext"
        assert "POST /v1/score" in body["endpoints"]

    def test_every_advertised_get_route_resolves(self, app):
        # Keeps the advertised list from drifting away from the router. Some routes need state
        # to exist, so it is created first.
        from fintext.server import ROUTES

        post(app, "/v1/score", {"text": "i will kill you", "profile": "compliance"})
        app.pipeline.log.flush()
        post(app, "/v1/review/batch", {"days": 7})

        for route in (r for r, _ in ROUTES if r.startswith("GET ")):
            path = route.removeprefix("GET ")
            if "{" in path:
                continue  # needs a live claim id; covered by test_claim_lookup
            if path in ("/metrics", "/ui"):
                # Served by the transport, not the router: these are text/plain and text/html,
                # and handle() returns JSON.
                continue
            assert app.handle("GET", path)[0] == 200, path

    def test_health_does_not_depend_on_the_model(self, pipeline):
        assert App(pipeline).handle("GET", "/healthz")[0] == 200

    def test_readiness_waits_for_warmup(self, pipeline):
        app = App(pipeline)
        assert app.handle("GET", "/readyz")[0] == 503
        app.startup()
        assert app.handle("GET", "/readyz")[0] == 200

    def test_a_failed_warmup_comes_up_degraded(self, config, index):
        # Holding every replica out of rotation turns a degraded service into a dead one; the
        # per-request policy should decide.
        app = App(Pipeline(config, backend=Broken(), index=index))
        app.startup()
        status, body = app.handle("GET", "/readyz")
        assert status == 200 and body["status"] == "ready_degraded"
        app.shutdown()

    def test_scores_an_item(self, app):
        status, body = post(app, "/v1/score", {"text": "Revenue rose sharply"})
        assert status == 200 and body["sentiment"]["label"] == "positive"

    @pytest.mark.parametrize(
        ("payload", "detail"),
        [({"text": ""}, "non-empty"), ({"text": "hi", "threshold": 0.9}, "unknown fields")],
    )
    def test_rejects_bad_requests(self, app, payload, detail):
        status, body = post(app, "/v1/score", payload)
        assert status == 400 and detail in body["detail"]

    def test_rejects_malformed_json(self, app):
        assert app.handle("POST", "/v1/score", b"{not json")[0] == 400

    def test_routing_edges(self, app):
        assert app.handle("GET", "/healthz/")[0] == 200  # trailing slash
        assert app.handle("GET", "/v1/nope")[0] == 404
        assert app.handle("DELETE", "/v1/score")[0] == 405

    def test_batch(self, app):
        status, body = post(
            app,
            "/v1/score:batch",
            {"profile": "research", "items": [{"text": "a rise"}, {"text": "a fall"}]},
        )
        assert status == 200 and body["count"] == 2

    def test_per_item_fields_override_batch_defaults(self, app):
        _, body = post(
            app,
            "/v1/score:batch",
            {"profile": "research", "items": [{"text": "hi", "profile": "trading"}]},
        )
        assert body["results"][0]["profile"] == "trading"

    def test_batch_needs_an_items_array(self, app):
        assert post(app, "/v1/score:batch", {"text": "hi"})[0] == 400

    def test_batch_size_limit(self, app):
        status, body = post(
            app, "/v1/score:batch", {"items": [{"text": "hi"}] * (app.config.max_batch + 1)}
        )
        assert status == 400 and "exceeds the limit" in body["detail"]

    def test_trading_gets_200_with_everything_degraded(self, config, index):
        app = App(Pipeline(config, backend=Broken(), index=index))
        app.startup()
        status, body = post(app, "/v1/score", {"text": "hi", "profile": "trading"})
        assert status == 200 and body["degraded"] and body["sentiment"] is None
        app.shutdown()

    def test_compliance_gets_503(self, config, index):
        app = App(Pipeline(config, backend=Broken(), index=index))
        app.startup()
        status, body = post(app, "/v1/score", {"text": "hi", "profile": "compliance"})
        assert status == 503 and body["error"] == "scoring_unavailable"
        app.shutdown()

    def test_metrics(self, app):
        post(app, "/v1/score", {"text": "Revenue rose"})
        assert app.handle("GET", "/v1/metrics")[1]["latency_ms"]["count"] == 1
        text = app.metrics_text()
        assert "fintext_latency_ms_bucket" in text
        assert 'fintext_requests_total{profile="trading"} 1' in text

    def test_model_and_config_endpoints(self, app):
        assert app.handle("GET", "/v1/model")[1]["lineage"]["config_digest"].startswith("sha256:")
        body = app.handle("GET", "/v1/config")[1]
        assert body["threshold_version"] and body["windows"]["regulatory"] == 900

    def test_claim_lookup(self, app):
        _, body = post(app, "/v1/score", {"text": "FAB raises its full-year guidance"})
        status, record = app.handle("GET", f"/v1/provenance/{body['provenance']['claim_id']}")
        assert status == 200 and record["state"] == "pending"
        assert app.handle("GET", "/v1/provenance/" + "0" * 32)[0] == 404

    def test_compliance_queue_excludes_pending(self, app):
        post(app, "/v1/score", {"text": "FAB raises its full-year guidance"})
        status, body = app.handle("GET", "/v1/queue/compliance")
        assert status == 200 and body["count"] == 0
        assert "not an assertion of falsity" in body["note"]

    def test_recheck_endpoint(self, app, clock):
        post(app, "/v1/score", {"text": "SEC charges Acme Corp"})
        clock.advance(1000)
        status, body = app.handle("POST", "/v1/admin/recheck", "{}")
        assert status == 200 and [t["to"] for t in body["transitions"]] == ["uncorroborated"]

    def test_body_parsing(self):
        from fintext.schemas import ValidationError

        assert parse_json(b'{"a": 1}') == parse_json('{"a": 1}') == {"a": 1}
        with pytest.raises(ValidationError, match="JSON object"):
            parse_json("[1, 2, 3]")
