"""Request and response types. These are the API."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

MAX_TOKENS = 256


class ValidationError(Exception):
    """A request failed contract validation."""


class Str(str, Enum):
    """Enum that is also a string, so json.dumps writes the plain value."""

    def __str__(self) -> str:
        return str(self.value)


class Language(Str):
    EN = "en"
    AR = "ar"
    UNKNOWN = "unknown"


class Sentiment(Str):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Subtype(Str):
    HATE = "hate"
    HARASSMENT = "harassment"
    PROFANITY = "profanity"
    THREAT = "threat"


class ProvenanceState(Str):
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    CORROBORATED = "corroborated"
    UNCORROBORATED = "uncorroborated"


class ClaimType(Str):
    MERGER = "merger"
    GUIDANCE = "guidance"
    REGULATORY = "regulatory"
    CONTRACT = "contract"


class Profile(Str):
    """Picks the latency budget and the failure policy."""

    TRADING = "trading"
    COMPLIANCE = "compliance"
    RESEARCH = "research"


class Capability(Str):
    SENTIMENT = "sentiment"
    TOXICITY = "toxicity"
    MISLEADING = "misleading"
    PROVENANCE = "provenance"


ALL_CAPABILITIES = tuple(Capability)


@dataclass(frozen=True)
class Span:
    """A character range in the scored text, with the substring inlined."""

    start: int
    end: int
    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class Explanation:
    summary: str
    method: str = "lexicon"
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class Lineage:
    """Everything needed to reproduce a decision."""

    model_version: str
    lexicon_versions: dict[str, str]
    threshold_version: str
    rules_version: str
    service_version: str
    config_digest: str


@dataclass(frozen=True)
class SentimentResult:
    label: Sentiment
    scores: dict[str, float]
    confidence: float
    language: Language
    # False for Arabic: reported, never an acceptance gate (PRD 1.1).
    acceptance_gated: bool = False


@dataclass(frozen=True)
class ToxicityResult:
    toxic: bool
    score: float
    threshold: float
    subtypes: dict[str, float]
    lexicon_override: bool
    explanation: Explanation


@dataclass(frozen=True)
class MisleadingResult:
    """Triage only."""

    likely_misleading: bool
    score: float
    threshold: float
    rule_matches: tuple[str, ...]
    review_priority: str
    explanation: Explanation

    @property
    def disposition(self) -> str:
        return "likely_misleading_needs_review" if self.likely_misleading else "no_flag"


@dataclass(frozen=True)
class Claim:
    claim_type: ClaimType
    entity: str
    entity_key: str
    span: Span
    confidence: float
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProvenanceResult:
    state: ProvenanceState
    reason: str
    claim: Claim | None = None
    claim_id: str | None = None
    window_seconds: int | None = None
    expires_at: float | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    # Stated in the payload so no consumer reads a truth value into a state.
    asserts_falsity: bool = False


@dataclass(frozen=True)
class ScoreRequest:
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    language: Language | None = None
    profile: Profile = Profile.TRADING
    capabilities: tuple[Capability, ...] = ALL_CAPABILITIES
    model_version: str | None = None
    source_url: str | None = None
    observed_at: float | None = None
    explain: bool = True

    FIELDS = frozenset(
        [
            "text",
            "id",
            "language",
            "profile",
            "capabilities",
            "model_version",
            "source_url",
            "observed_at",
            "explain",
        ]
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreRequest:
        if not isinstance(payload, dict):
            raise ValidationError("request body must be a JSON object")
        unknown = set(payload) - cls.FIELDS
        if unknown:
            # Ignoring an unknown field lets a caller believe an option took effect.
            raise ValidationError(f"unknown fields: {sorted(unknown)}")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("text must be a non-empty string")
        if len(text) > 100_000:
            raise ValidationError("text exceeds 100000 characters")
        try:
            capabilities = payload.get("capabilities")
            return cls(
                text=text,
                id=str(payload.get("id") or uuid.uuid4().hex),
                language=Language(payload["language"]) if payload.get("language") else None,
                profile=Profile(payload.get("profile", "trading")),
                capabilities=(
                    tuple(Capability(c) for c in capabilities) if capabilities else ALL_CAPABILITIES
                ),
                model_version=payload.get("model_version"),
                source_url=payload.get("source_url"),
                observed_at=payload.get("observed_at"),
                explain=bool(payload.get("explain", True)),
            )
        except (ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc


@dataclass(frozen=True)
class ScoreResponse:
    id: str
    language: Language
    profile: Profile
    lineage: Lineage
    scored_at: float
    latency_ms: float
    sentiment: SentimentResult | None = None
    toxicity: ToxicityResult | None = None
    misleading: MisleadingResult | None = None
    provenance: ProvenanceResult | None = None
    scored_text: str | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    # Should have run and did not (failure or budget).
    degraded: tuple[str, ...] = ()
    # Was never going to run, and why (out of scope, or not requested).
    skipped: dict[str, str] = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = round_floats(asdict(self))
        if self.misleading is not None:
            payload["misleading"]["disposition"] = self.misleading.disposition
        return payload


def round_floats(value: Any, digits: int = 4) -> Any:
    """Round every float in a nested structure, so payloads read cleanly."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [round_floats(v, digits) for v in value]
    return value
