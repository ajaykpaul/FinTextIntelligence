"""The trading gate: scores in, one decision out (PRD 5.2, 6.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config
from .schemas import ProvenanceState, ScoreResponse, Sentiment, Str

VERSION = "gate.v1"


class Direction(Str):
    """Three outcomes."""

    BUY = "buy"
    SELL = "sell"
    REVIEW = "review"


@dataclass(frozen=True)
class Signal:
    """One call, and why."""

    direction: Direction | None
    conviction: float
    gated: bool
    priority: str = ""
    hold_ms: int = 0
    reason: str = ""
    gate_version: str = VERSION
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = str(self.direction) if self.direction else None
        payload["conviction"] = round(self.conviction, 4)
        return payload


def decide(response: ScoreResponse, config: Config) -> Signal:
    """Turn a scored item into a trading decision."""
    sentiment = response.sentiment
    inputs = {
        "id": response.id,
        "sentiment": str(sentiment.label) if sentiment else None,
        "confidence": round(sentiment.confidence, 4) if sentiment else None,
        "provenance": str(response.provenance.state) if response.provenance else None,
        "misleading": bool(response.misleading and response.misleading.likely_misleading),
        "degraded": list(response.degraded),
    }

    # 1. Scoring failed: no view, no call.
    if sentiment is None or "sentiment" in response.degraded:
        return Signal(
            None,
            0.0,
            gated=False,
            reason="scoring unavailable; no call made, trade as though the gate were not deployed",
            inputs=inputs,
        )

    # 2. Looks like promotion or a scam. Not tradeable, and a person should see it.
    if response.misleading and response.misleading.likely_misleading:
        return Signal(
            Direction.REVIEW,
            0.0,
            gated=True,
            priority="now",
            reason="flagged as likely misleading; a person decides before this informs anything",
            inputs=inputs,
        )

    state = response.provenance.state if response.provenance else None

    # 3. Window closed with no confirmation: send to a person.
    if state is ProvenanceState.UNCORROBORATED:
        return Signal(
            Direction.REVIEW,
            0.0,
            gated=True,
            priority="now",
            reason="no confirmation arrived within the window for this kind of claim",
            inputs=inputs,
        )

    # 4. Nothing directional to trade on (PRD 4.1, 5.2).
    if sentiment.label is Sentiment.NEUTRAL:
        return Signal(
            Direction.REVIEW,
            0.0,
            gated=True,
            priority="routine",
            reason="nothing directional to trade on",
            inputs=inputs,
        )
    if sentiment.confidence < config.signal_min_confidence:
        return Signal(
            Direction.REVIEW,
            0.0,
            gated=True,
            priority="now",
            reason=f"leans {sentiment.label} but only at {sentiment.confidence:.2f}, "
            f"under the {config.signal_min_confidence:.2f} bar, too close to call",
            inputs=inputs,
        )

    direction = Direction.BUY if sentiment.label is Sentiment.POSITIVE else Direction.SELL
    conviction = sentiment.confidence
    hold_ms = 0
    reason = f"clearly {sentiment.label} at {sentiment.confidence:.2f}"

    # 5. Still pending: hold briefly and cut conviction, but don't suppress (PRD 6.4).
    if state is ProvenanceState.PENDING:
        conviction *= config.pending_conviction
        hold_ms = config.pending_hold_ms
        reason += ", unconfirmed so far, so a short hold and reduced conviction"
    elif state is ProvenanceState.CORROBORATED:
        reason += ", confirmed by a primary source"

    return Signal(
        direction,
        conviction,
        gated=True,
        hold_ms=hold_ms,
        reason=reason,
        inputs=inputs,
    )
