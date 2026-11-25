"""The trading gate: one decision per item, and the reason for it (PRD 5.2, 6.4)."""

from __future__ import annotations

import json

import pytest
from conftest import T0, make_document

from fintext.config import load_config
from fintext.schemas import (
    Explanation,
    Language,
    Lineage,
    MisleadingResult,
    Profile,
    ProvenanceResult,
    ProvenanceState,
    ScoreResponse,
    Sentiment,
    SentimentResult,
)
from fintext.signal import Direction, decide


@pytest.fixture
def gate_config():
    return load_config()


def scored(
    label=Sentiment.POSITIVE, confidence=0.9, *, state=None, misleading=False, degraded=()
) -> ScoreResponse:
    """A response shaped the way the pipeline emits one."""
    return ScoreResponse(
        id="i1",
        language=Language.EN,
        profile=Profile.TRADING,
        lineage=Lineage("m", {"en": "l"}, "t", "r", "1.5.0", "sha256:x"),
        scored_at=T0,
        latency_ms=1.0,
        sentiment=None
        if "sentiment" in degraded
        else SentimentResult(
            label,
            {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            confidence,
            Language.EN,
            acceptance_gated=True,
        ),
        misleading=MisleadingResult(
            misleading,
            0.9 if misleading else 0.1,
            0.5,
            ("guaranteed_returns",) if misleading else (),
            "high",
            Explanation("s"),
        ),
        provenance=None if state is None else ProvenanceResult(state=state, reason="r"),
        degraded=degraded,
    )


class TestDirection:
    def test_clear_good_news_says_buy(self, gate_config):
        signal = decide(scored(Sentiment.POSITIVE, 0.9), gate_config)
        assert signal.direction is Direction.BUY
        assert signal.conviction == pytest.approx(0.9)
        assert signal.gated and not signal.priority

    def test_clear_bad_news_says_sell(self, gate_config):
        assert decide(scored(Sentiment.NEGATIVE, 0.9), gate_config).direction is Direction.SELL

    def test_nothing_to_trade_is_a_routine_review(self, gate_config):
        # A clear answer, not an ambiguous one, so nobody has to act on it.
        signal = decide(scored(Sentiment.NEUTRAL, 0.9), gate_config)
        assert signal.direction is Direction.REVIEW and signal.priority == "routine"
        assert "nothing directional" in signal.reason

    def test_a_borderline_lean_needs_a_person_now(self, gate_config):
        # Too close to call is genuine ambiguity, unlike a routine notice.
        signal = decide(scored(Sentiment.POSITIVE, 0.4), gate_config)
        assert signal.direction is Direction.REVIEW and signal.priority == "now"
        assert "too close to call" in signal.reason

    def test_the_bar_is_configurable(self, gate_config):
        weak = scored(Sentiment.POSITIVE, 0.5)
        assert decide(weak, gate_config).direction is Direction.REVIEW
        relaxed = gate_config.replace(signal_min_confidence=0.4)
        assert decide(weak, relaxed).direction is Direction.BUY

    def test_every_input_produces_exactly_one_direction(self, gate_config):
        """Only buy, sell or review."""
        cases = [
            scored(Sentiment.POSITIVE, 0.9),
            scored(Sentiment.NEGATIVE, 0.3),
            scored(Sentiment.NEUTRAL, 0.8),
            scored(state=ProvenanceState.PENDING),
            scored(state=ProvenanceState.UNCORROBORATED),
            scored(misleading=True),
            scored(degraded=("sentiment",)),
        ]
        for response in cases:
            signal = decide(response, gate_config)
            assert signal.direction in set(Direction) or signal.gated is False
            assert signal.reason, "every decision explains itself"


class TestProvenance:
    def test_pending_trims_conviction_but_never_suppresses(self, gate_config):
        signal = decide(scored(state=ProvenanceState.PENDING), gate_config)
        assert signal.direction is Direction.BUY
        assert 0 < signal.conviction < 0.9

    def test_no_hold_by_default(self, gate_config):
        # On a 20 ms path a hold of any real length is suppression renamed.
        assert gate_config.pending_hold_ms == 0
        assert decide(scored(state=ProvenanceState.PENDING), gate_config).hold_ms == 0

    def test_a_hold_can_be_configured_for_slower_consumers(self, gate_config):
        patient = gate_config.replace(pending_hold_ms=500)
        signal = decide(scored(state=ProvenanceState.PENDING), patient)
        assert signal.hold_ms == 500 and signal.direction is Direction.BUY

    def test_corroborated_trades_at_full_conviction(self, gate_config):
        signal = decide(scored(state=ProvenanceState.CORROBORATED), gate_config)
        assert signal.conviction == pytest.approx(0.9) and signal.hold_ms == 0

    def test_uncorroborated_goes_to_a_person_now(self, gate_config):
        # The state that carries signal routes to review, not into the book.
        signal = decide(scored(state=ProvenanceState.UNCORROBORATED), gate_config)
        assert signal.direction is Direction.REVIEW and signal.priority == "now"

    def test_not_applicable_does_not_penalise(self, gate_config):
        signal = decide(scored(state=ProvenanceState.NOT_APPLICABLE), gate_config)
        assert signal.direction is Direction.BUY and signal.conviction == pytest.approx(0.9)


class TestEscalation:
    def test_a_misleading_flag_goes_to_a_person_now(self, gate_config):
        signal = decide(scored(misleading=True), gate_config)
        assert signal.direction is Direction.REVIEW and signal.priority == "now"

    def test_misleading_outranks_a_strong_signal(self, gate_config):
        # Conviction does not buy its way past a scam flag.
        signal = decide(scored(Sentiment.POSITIVE, 0.99, misleading=True), gate_config)
        assert signal.direction is Direction.REVIEW and signal.priority == "now"

    def test_routine_items_never_demand_attention(self, gate_config):
        for response in (scored(Sentiment.POSITIVE, 0.9), scored(Sentiment.NEUTRAL, 0.9)):
            assert decide(response, gate_config).priority != "now"


class TestFailOpen:
    def test_a_failed_score_makes_no_call_at_all(self, gate_config):
        signal = decide(scored(degraded=("sentiment",)), gate_config)
        assert signal.direction is None
        assert signal.gated is False
        assert signal.priority == ""
        assert "as though the gate were not deployed" in signal.reason

    def test_a_real_call_is_marked_gated(self, gate_config):
        assert decide(scored(Sentiment.NEUTRAL, 0.9), gate_config).gated is True


class TestOverHttp:
    def test_signal_endpoint_returns_a_decision_and_the_score(self, app):
        status, body = app.handle(
            "POST",
            "/v1/signal",
            json.dumps({"text": "Record profit, strong margins and an upgraded outlook."}),
        )
        assert status == 200
        assert body["signal"]["direction"] == "buy"
        assert body["score"]["sentiment"]["label"] == "positive"

    def test_the_score_reply_still_cannot_carry_an_action(self, app):
        # The gate lives on its own route so this stays true.
        _, body = app.handle("POST", "/v1/score", json.dumps({"text": "Profit rose sharply"}))
        assert not {"signal", "direction", "action", "order"} & set(body)

    def test_a_confirmed_claim_trades_at_full_conviction(self, app):
        app.pipeline.index.add(make_document())
        _, body = app.handle(
            "POST",
            "/v1/signal",
            json.dumps(
                {
                    "text": (
                        "First Abu Dhabi Bank PJSC agreed to acquire a lender, "
                        "lifting profit sharply"
                    ),
                    "observed_at": T0,
                }
            ),
        )
        assert body["score"]["provenance"]["state"] == "corroborated"
        assert body["signal"]["hold_ms"] == 0

    def test_bad_input_is_rejected(self, app):
        assert app.handle("POST", "/v1/signal", json.dumps({"text": ""}))[0] == 400


class TestReviewVolume:
    """The split exists to keep the review queue usable (PRD 2.3)."""

    STREAM = [
        # Routine notices, which are most of any real feed.
        *["The company will publish its results on Tuesday."] * 5,
        *["The board approved a dividend of 0.35 per share."] * 5,
        # Clearly directional.
        "Record profit, strong margins and an upgraded outlook.",
        "Revenue missed consensus and the board cut its full-year guidance.",
        # Genuinely ambiguous.
        "Guaranteed 300% returns. Act now, send USDT.",
    ]

    def test_routine_notices_do_not_reach_a_reviewer(self, pipeline, gate_config):
        from fintext.schemas import ScoreRequest

        calls = [decide(pipeline.score(ScoreRequest(text=t)), gate_config) for t in self.STREAM]
        urgent = [s for s in calls if s.priority == "now"]
        reviewed = [s for s in calls if s.direction is Direction.REVIEW]

        # Most of the stream is a review call, but almost none of it is urgent.
        assert len(reviewed) > len(urgent) * 3
        assert len(urgent) / len(calls) < 0.25
