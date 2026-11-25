"""Provenance checking: rules and retrieval, no model (PRD 4.4, 6.3, 6.4)."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .schemas import Claim, ClaimType, ProvenanceResult, ProvenanceState, Span

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def _patterns(*sources: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(s, re.IGNORECASE) for s in sources)


# Checked in this order: shortest disclosure clock first, so a text carrying more than one claim
# resolves against the fastest-confirming source.
CLAIM_PATTERNS: dict[ClaimType, tuple[re.Pattern[str], ...]] = {
    ClaimType.REGULATORY: _patterns(
        r"\b(?:fined|fines|penalis(?:ed|es)|penaliz(?:ed|es)|sanction(?:ed|s))\b",
        # Exchange compliance notices: the listing-rule letters issuers must disclose.
        r"\b(?:deficiency\s+(?:letter|notice)|delisting\s+(?:determination|notice|letter)"
        r"|notice\s+of\s+non-?compliance|listing\s+qualifications)\b",
        r"\b(?:investigation|probe|enforcement\s+action|consent\s+order)\b",
        # Named regulators plus the generic nouns a wire uses when it does not name one.
        r"\b(?:sec|cbuae|fca|esma|dfsa|adgm|central\s+bank|regulator|watchdog|authority"
        r"|prosecutors?)\b.{0,40}\b(?:charges?|charged|orders?|suspends?|halts?|approves?"
        r"|licen[cs]es?|opens?\s+an?\s+investigation)\b",
        r"\btrading\s+halt(?:ed)?\b",
        r"\b(?:delisting|delisted)\b",
    ),
    ClaimType.MERGER: _patterns(
        r"\b(?:to\s+)?acquires?\b",
        r"\bacquisition\s+of\b",
        r"\bmerger\s+(?:with|of|talks)\b",
        r"\bto\s+merge\s+with\b",
        r"\b(?:takeover|buyout)\s+(?:bid|offer|approach)\b",
        r"\bagreed\s+to\s+(?:buy|acquire)\b",
        r"\bstake\s+(?:sale|purchase)\s+in\b",
    ),
    ClaimType.GUIDANCE: _patterns(
        r"\b(?:raises?|raised|lowers?|lowered|cuts?|cut|reaffirms?|withdraws?)\s+(?:its\s+)?"
        r"(?:full[\s-]year\s+|fy\d{2,4}\s+|q[1-4]\s+)?(?:guidance|outlook|forecast)\b",
        # The same verbs with the words releases put in between: "raised total company
        # guidance", "increases 2021 EPS guidance", "raising our full-year revenue and earnings
        # outlook".
        r"\b(?:rais(?:es|ed|ing)|increas(?:es|ed|ing)|lower(?:s|ed|ing)|reaffirm(?:s|ed|ing)"
        r"|reiterat(?:es|ed|ing)|narrow(?:s|ed|ing)|updat(?:es|ed|ing)|withdr(?:aws|ew|awing))"
        r"\s+(?:[\w$%&'’-]+\s+){0,6}?(?:guidance|outlook)\b",
        r"\bprofit\s+(?:warning|alert)\b",
        r"\bexpects?\s+(?:full[\s-]year\s+)?(?:revenue|earnings|profit)\s+(?:of|to)\b",
        r"\b(?:beats?|misses?)\s+(?:consensus|estimates?|expectations)\b",
    ),
    ClaimType.CONTRACT: _patterns(
        # Up to three intervening tokens, so "wins a $2bn five-year contract" matches without
        # reaching across a clause boundary.
        r"\b(?:wins?|won|awarded|secures?|signs?|signed)\s+"
        r"(?:(?:a|an|the|\$?[\d.,]+\s*(?:bn|billion|m|million)|[\w-]+)\s+){0,5}?"
        r"(?:contract|deal|mandate|order|agreement)\b",
        r"\bcontract\s+(?:worth|valued\s+at)\b",
        r"\bframework\s+agreement\s+with\b",
    ),
}

TICKER = re.compile(
    r"(?:\$([A-Z]{1,5})\b|\(\s*(?:NYSE|NASDAQ|LSE|ADX|DFM)\s*:\s*([A-Z.]{1,6})\s*\))"
)
ORG = re.compile(
    # The comma before a suffix and the all-capitals house style are both ordinary in real
    # filings, "Techpoint, Inc." and "REVVITY, INC." are how EDGAR writes them, and neither
    # matched until a real index arrived.
    r"\b((?:[A-Z][\w&.\-]*(?:\s+(?:of|and|the|for)\s+)?\s*){1,5}"
    r",?\s*(?i:Inc|Corp|Corporation|Ltd|Limited|LLC|PLC|PJSC|SA|AG|NV|"
    r"Group|Holdings|Bank|Company|Co)\.?)(?!\w)"
)
CAPITALISED = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")

# Sentence-initial capitals and headline words produce false entities.
STOP_ENTITIES = frozenset(
    [
        "the",
        "a",
        "an",
        "breaking",
        "exclusive",
        "update",
        "reuters",
        "bloomberg",
        "market",
        "markets",
        "shares",
        "stocks",
        "trading",
        "investors",
        "analysts",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "sec",
        "fca",
        "esma",
    ]
)

# The monitored issuer universe.
ALIASES = {
    "FAB": "first abu dhabi bank",
    "First Abu Dhabi Bank": "first abu dhabi bank",
    "ADCB": "abu dhabi commercial bank",
    "Emirates NBD": "emirates nbd",
    "ADNOC": "abu dhabi national oil company",
    "Aldar": "aldar properties",
    "Etisalat": "e and",
}

_SUFFIXES = re.compile(
    r"\b(?:inc|corp|corporation|ltd|limited|llc|plc|pjsc|sa|ag|nv|group|holdings|co|company)\b"
)


def entity_key(name: str) -> str:
    """Canonical key for entity matching."""
    cleaned = _SUFFIXES.sub(" ", re.sub(r"[^\w\s]", " ", name.lower()))
    return re.sub(r"\s+", " ", cleaned).strip()


class ClaimExtractor:
    """Finds (claim type, entity) pairs."""

    def __init__(self, aliases: dict[str, str] | None = None, *, min_confidence: float = 0.5):
        source = aliases or ALIASES
        self.aliases = {entity_key(k): v for k, v in source.items()}
        self.min_confidence = min_confidence
        # How each alias is looked for in text.
        self._finders: list[re.Pattern[str]] = []
        seen: set[str] = set()
        for raw in source:
            if raw.isupper() and len(raw.replace(".", "")) <= 6:
                if len(raw) < 2 or raw in seen:
                    continue
                self._finders.append(re.compile(rf"(?<![\w$]){re.escape(raw)}(?![\w])"))
                seen.add(raw)
            else:
                key = entity_key(raw)
                if len(key) < 3 or key in seen or key in STOP_ENTITIES:
                    continue
                self._finders.append(re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE))
                seen.add(key)

    def extract(self, text: str) -> Claim | None:
        """Return the highest-priority checkable claim, or None."""
        entity = self._best_entity(text)
        for claim_type, patterns in CLAIM_PATTERNS.items():
            match = next(filter(None, (p.search(text) for p in patterns)), None)
            if match is None:
                continue
            span = Span(match.start(), match.end(), match.group(0))
            if entity is None:
                # Reported at low confidence: the state machine holds it pending rather than
                # searching for a document it cannot address.
                return Claim(claim_type, "", "", span, 0.30)
            name, name_confidence = entity
            key = self.aliases.get(entity_key(name), entity_key(name))
            return Claim(
                claim_type=claim_type,
                entity=name,
                entity_key=key,
                span=span,
                confidence=min(0.99, 0.55 + 0.4 * name_confidence),
                details=claim_details(text, name, key),
            )
        return None

    def _best_entity(self, text: str) -> tuple[str, float] | None:
        best = max(self._candidates(text), key=lambda c: c[1], default=None)
        return best if best and best[1] >= self.min_confidence else None

    def _candidates(self, text: str) -> Iterator[tuple[str, float]]:
        for match in TICKER.finditer(text):
            yield match.group(1) or match.group(2), 1.0
        for match in ORG.finditer(text):
            name = match.group(1).strip()
            if entity_key(name) and entity_key(name) not in STOP_ENTITIES:
                yield name, 0.9
        for match in CAPITALISED.finditer(text):
            name = match.group(1).strip()
            key = entity_key(name)
            if key and key.split()[0] not in STOP_ENTITIES:
                # A known alias promotes a bare capitalised run to a confident entity; an
                # unknown one stays below the threshold.
                yield name, 0.95 if key in self.aliases else 0.45
        for finder in self._finders:
            for found in finder.finditer(text):
                if found.group(0)[0].isupper():
                    yield found.group(0), 0.95
                    break


# ---------------------------------------------------------------------------
# Primary-source index
# ---------------------------------------------------------------------------

# Only these corroborate. Counting news would corroborate exactly the item the check exists to
# catch: a fabricated headline surrounded by copies of itself.
PRIMARY_TYPES = frozenset({"edgar", "issuer_ir", "exchange_notice", "regulator"})

# How far from a claim a document may sit and still be evidence about it.
EVIDENCE_HORIZON = 7 * 24 * 3600

# What a primary document says for each claim type. Terse: filings are formulaic, and a long
# list buys recall at the cost of gated precision.
CLAIM_VOCABULARY = {
    ClaimType.MERGER: (
        "merger",
        "acquisition",
        "acquire",
        "business combination",
        "stake",
        "tender offer",
        "definitive agreement",
        "8-k",
        "item 1.01",
    ),
    ClaimType.GUIDANCE: (
        "guidance",
        "outlook",
        "forecast",
        "results",
        "earnings",
        "financial statements",
        "quarterly report",
        "10-q",
        "trading update",
    ),
    ClaimType.REGULATORY: (
        "enforcement",
        "order",
        "charges",
        "settlement",
        "penalty",
        "suspension",
        "halt",
        "consent",
        "administrative proceeding",
        "notice",
        "delisting",
        "listing rule",
    ),
    ClaimType.CONTRACT: (
        "contract",
        "award",
        "agreement",
        "mandate",
        "order intake",
        "material contract",
        "item 1.01",
    ),
}


# Words that only a filing about this kind of event would use.
SPECIFIC_VOCABULARY = {
    ClaimType.MERGER: (
        "merger",
        "acquisition",
        "acquire",
        "business combination",
        "stake",
        "tender offer",
    ),
    ClaimType.GUIDANCE: (
        "guidance",
        "outlook",
        "forecast",
        "results",
        "earnings",
        "quarterly report",
        "10-q",
        "trading update",
    ),
    ClaimType.REGULATORY: (
        "enforcement",
        "charges",
        "settlement",
        "penalty",
        "suspension",
        "halt",
        "consent",
        "administrative proceeding",
        "delisting",
        "listing rule",
    ),
    ClaimType.CONTRACT: ("contract", "award", "mandate", "order intake"),
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_type: str
    entity: str
    title: str = ""
    url: str = ""
    published_at: float = 0.0
    body: str = ""
    tickers: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return entity_key(self.entity)

    @property
    def is_primary(self) -> bool:
        return self.source_type in PRIMARY_TYPES

    def summary(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Document:
        return cls(
            doc_id=str(payload["doc_id"]),
            source_type=str(payload["source_type"]),
            entity=str(payload["entity"]),
            title=str(payload.get("title", "")),
            url=str(payload.get("url", "")),
            published_at=float(payload.get("published_at", 0.0)),
            body=str(payload.get("body", "")),
            tickers=tuple(payload.get("tickers") or ()),
        )


# Capitalised words that say nothing specific about a claim: titles, exchanges, months,
# corporate suffixes and the event words themselves.
DETAIL_STOP = frozenset(
    """
    the a an and or of for to in on at by with from as this that these those we our
    us it its their they he she his her you your i is are was were will be has have
    had today announced announces announcing said says reported reports report
    company companies corporation corp inc incorporated ltd limited plc llc lp co
    holdings holding group technologies technology international global systems
    ceo cfo coo president chief executive officer officers chairman chairwoman chair
    founder co-founder director directors board management team vice senior general
    nasdaq nyse amex exchange stock stocks common shares share securities sec
    january february march april may june july august september october november
    december monday tuesday wednesday thursday friday saturday sunday
    first second third fourth quarter quarters fiscal year years full annual
    q1 q2 q3 q4 fy guidance outlook revenue revenues earnings ebitda net income
    agreement agreements merger acquisition acquire acquires acquired definitive
    contract contracts award awarded order orders deficiency notice listing
    million billion thousand percent usd dollars approximately about
    north south east west america american united states u.s new
    """.split()  # noqa: SIM905 -- easier to read and extend as a block
)
_DETAIL_NAME = re.compile(r"(?<![\w])[A-Z][A-Za-z&'\-]{2,}")
_DETAIL_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORDS = re.compile(r"[a-z0-9][a-z0-9&'\-]*")


def claim_details(text: str, entity: str, key: str = "") -> tuple[str, ...]:
    """The particulars a claim states besides its company: names and amounts."""
    own = set(_WORDS.findall(f"{entity} {key}".lower()))
    found: set[str] = set()
    for match in _DETAIL_NAME.finditer(text):
        word = match.group(0).lower().strip("'-")
        if match.start() == 0 or word in own or word in DETAIL_STOP or len(word) < 3:
            continue
        found.add(word)
    for match in _DETAIL_AMOUNT.finditer(text):
        number = match.group(0).replace(",", "").rstrip(".")
        digits = number.replace(".", "")
        if "." not in number and (len(digits) < 3 or 1900 <= int(digits) <= 2100):
            continue
        found.add(number)
    return tuple(sorted(found))


def has_text(document: Document) -> bool:
    """Whether a filing carries text beyond its form and 8-K item titles."""
    from .edgar import SEC_ITEMS

    body = document.body.lower()
    for number, title in SEC_ITEMS.items():
        body = body.replace(f"item {number} {title.lower()}", " ")
    return len(_WORDS.findall(body.replace("8-k", " "))) >= 25


def mentions_detail(details: tuple[str, ...], document: Document) -> bool:
    haystack = f"{document.title} {document.body}".lower()
    words = set(_WORDS.findall(haystack))
    flat = haystack.replace(",", "")
    return any(d in words if d[0].isalpha() else d in flat for d in details)


def score_match(
    claim: Claim, document: Document, *, observed_at: float, window: int
) -> tuple[float, str]:
    """Score a document against a claim in [0, 1]."""
    if not document.is_primary:
        return 0.0, "not a primary source"
    if not claim.entity_key:
        return 0.0, "claim has no resolvable entity"

    # 1. Entity identity. Substring matches only when the claim entity is the longer string, so
    # "Bank" matches nothing.
    if claim.entity_key == document.key:
        entity_score, reason = 1.0, "exact entity match"
    elif claim.entity.upper() in document.tickers:
        entity_score, reason = 1.0, "ticker match"
    elif document.key and document.key in claim.entity_key and len(document.key.split()) >= 2:
        entity_score, reason = 0.85, "entity contained in the claim entity"
    else:
        return 0.0, "entity does not match"

    # 2. Claim-type vocabulary.
    haystack = f"{document.title} {document.body}".lower()
    hits = sum(1 for word in CLAIM_VOCABULARY[claim.claim_type] if word in haystack)
    if hits == 0:
        return 0.0, "document does not use the claim-type vocabulary"
    if not any(word in haystack for word in SPECIFIC_VOCABULARY[claim.claim_type]):
        # "8-k", "item 1.01" and "agreement" appear in filings about any kind of agreement.
        return 0.0, "document has only generic filing words for this claim type"
    if claim.details and has_text(document) and not mentions_detail(claim.details, document):
        # With a filing's full text, "guidance", "acquire" and "contract" turn up in releases
        # about other events.
        return 0.0, "document has text but none of the claim's particulars"

    # 3. Time plausibility. A document may precede the headline (the headline reports the
    # filing) or follow it (the filing confirms the headline).
    age = observed_at - document.published_at
    if abs(age) > EVIDENCE_HORIZON:
        return 0.0, "document is outside the evidence horizon"
    recency = 1.0 if abs(age) <= window else max(0.5, 1.0 - abs(age) / EVIDENCE_HORIZON)

    keyword_score = min(1.0, 0.6 + 0.2 * hits)
    return (
        entity_score * keyword_score * recency,
        f"{reason}, {hits} keyword(s), {age / 60:+.0f} min",
    )


class PrimaryIndex:
    """In-memory index over primary documents."""

    def __init__(self, documents: Iterable[Document] = ()) -> None:
        self.documents: list[Document] = []
        self.by_entity: dict[str, list[Document]] = {}
        self.last_updated = 0.0
        self.extend(documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def add(self, document: Document) -> None:
        self.documents.append(document)
        self.by_entity.setdefault(document.key, []).append(document)
        for ticker in document.tickers:
            self.by_entity.setdefault(entity_key(ticker), []).append(document)
        # Freshness is measured in the stream's own time, never wall time, so a replay and a
        # live replica reach the same staleness verdict.
        self.last_updated = max(self.last_updated, document.published_at)

    def extend(self, documents: Iterable[Document]) -> int:
        before = len(self.documents)
        for document in documents:
            self.add(document)
        return len(self.documents) - before

    def aliases(self) -> dict[str, str]:
        """Entity vocabulary derived from the index itself."""
        found: dict[str, str] = {}
        for document in self.documents:
            key = document.key
            if not key:
                continue
            found[document.entity] = key
            for ticker in document.tickers:
                found[ticker] = key
        return found

    def mark_polled(self, at: float) -> None:
        """Record a successful poll that found nothing new."""
        self.last_updated = max(self.last_updated, at)

    def search(
        self, claim: Claim, *, observed_at: float, window: int
    ) -> list[tuple[Document, float, str]]:
        if not claim.entity_key:
            return []
        # The entity bucket, plus any key contained in the claim key, which covers suffix
        # variation without scanning the whole index.
        candidates = {d.doc_id: d for d in self.by_entity.get(claim.entity_key, ())}
        for key, documents in self.by_entity.items():
            if key and key != claim.entity_key and key in claim.entity_key:
                candidates.update({d.doc_id: d for d in documents})

        scored = []
        for document in candidates.values():
            score, reason = score_match(claim, document, observed_at=observed_at, window=window)
            if score > 0:
                scored.append((document, score, reason))
        return sorted(scored, key=lambda row: -row[1])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PrimaryIndex:
        file = Path(path)
        if not file.exists():
            return cls()
        return cls(
            Document.from_dict(json.loads(line))
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

LEGAL_TRANSITIONS = {
    ProvenanceState.NOT_APPLICABLE: frozenset(),
    ProvenanceState.PENDING: frozenset(
        {ProvenanceState.CORROBORATED, ProvenanceState.UNCORROBORATED}
    ),
    ProvenanceState.UNCORROBORATED: frozenset({ProvenanceState.CORROBORATED}),
    ProvenanceState.CORROBORATED: frozenset(),
}

NOT_APPLICABLE = ProvenanceResult(
    state=ProvenanceState.NOT_APPLICABLE,
    reason="no checkable market-moving claim in the text",
)


class IllegalTransition(Exception):
    pass


@dataclass(frozen=True)
class Transition:
    from_state: ProvenanceState
    to_state: ProvenanceState
    at: float
    reason: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": str(self.from_state),
            "to": str(self.to_state),
            "at": self.at,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass
class ClaimRecord:
    claim_id: str
    claim: Claim
    item_id: str
    observed_at: float
    window: int
    state: ProvenanceState = ProvenanceState.PENDING
    transitions: list[Transition] = field(default_factory=list)
    evidence: list[tuple[Document, float, str]] = field(default_factory=list)
    last_checked_at: float = 0.0
    reason: str = "claim registered, awaiting corroboration"

    @property
    def expires_at(self) -> float:
        return self.observed_at + self.window

    @property
    def time_to_corroboration(self) -> float | None:
        for transition in self.transitions:
            if transition.to_state is ProvenanceState.CORROBORATED:
                return transition.at - self.observed_at
        return None

    def move_to(
        self, state: ProvenanceState, *, at: float, reason: str, evidence: tuple[str, ...] = ()
    ) -> Transition:
        if state not in LEGAL_TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.state} -> {state} is not legal")
        transition = Transition(self.state, state, at, reason, evidence)
        self.transitions.append(transition)
        self.state = state
        self.reason = reason
        return transition

    def to_result(self) -> ProvenanceResult:
        return ProvenanceResult(
            state=self.state,
            reason=self.reason,
            claim=self.claim,
            claim_id=self.claim_id,
            window_seconds=self.window,
            expires_at=self.expires_at,
            evidence=tuple(
                {**d.summary(), "match_score": round(s, 4), "reason": r}
                for d, s, r in self.evidence[:3]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_result()
        return {
            "claim_id": self.claim_id,
            "item_id": self.item_id,
            "state": str(self.state),
            "observed_at": self.observed_at,
            "window_seconds": self.window,
            "expires_at": self.expires_at,
            "reason": self.reason,
            "transitions": [t.to_dict() for t in self.transitions],
            "evidence": list(payload.evidence),
        }


class Tracker:
    """Registers claims, resolves them against the index and ages them out."""

    def __init__(
        self,
        index: PrimaryIndex,
        *,
        windows: dict[str, int],
        min_match_score: float = 0.82,
        stale_after: int = 3600,
        recheck_seconds: int = 60,
        capacity: int = 100_000,
        clock: Callable[[], float] = time.time,
        on_transition: Callable[[ClaimRecord, Transition], None] | None = None,
    ) -> None:
        self.index = index
        self.windows = windows
        self.min_match_score = min_match_score
        self.stale_after = stale_after
        self.recheck_seconds = recheck_seconds
        self.capacity = capacity
        self.clock = clock
        self.on_transition = on_transition
        self.records: OrderedDict[str, ClaimRecord] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ClaimRecord]:
        return iter(self.records.values())

    def get(self, claim_id: str) -> ClaimRecord | None:
        return self.records.get(claim_id)

    def check(
        self,
        claim: Claim | None,
        *,
        item_id: str,
        observed_at: float | None = None,
        now: float | None = None,
    ) -> ProvenanceResult:
        """Register a claim and resolve it once."""
        if claim is None:
            return NOT_APPLICABLE

        now = self.clock() if now is None else now
        if str(claim.claim_type) not in self.windows:
            raise ValueError(
                f"no corroboration window configured for {claim.claim_type}; windows come "
                "from observed confirmation times, never from a runtime default (PRD 6.3)"
            )
        record = ClaimRecord(
            claim_id=uuid.uuid4().hex,
            claim=claim,
            item_id=item_id,
            observed_at=now if observed_at is None else observed_at,
            window=self.windows[str(claim.claim_type)],
        )
        self.records[record.claim_id] = record
        while len(self.records) > self.capacity:
            self.records.popitem(last=False)

        self._resolve(record, now)
        return record.to_result()

    def recheck_due(self, *, now: float | None = None) -> list[Transition]:
        """Re-check open claims whose interval has elapsed."""
        now = self.clock() if now is None else now
        transitions = []
        for record in list(self.records.values()):
            if record.state is ProvenanceState.CORROBORATED:
                continue
            if now - record.last_checked_at < self.recheck_seconds:
                continue
            if transition := self._resolve(record, now):
                transitions.append(transition)
        return transitions

    def on_documents(
        self, documents: Iterable[Document], *, now: float | None = None
    ) -> list[Transition]:
        """Re-check immediately when new primary documents land."""
        now = self.clock() if now is None else now
        added = self.index.extend(documents)
        # Documents arriving proves ingestion is alive, so this clears staleness too, otherwise
        # a backfill of older filings would read as an outage.
        self.index.mark_polled(now)
        if not added:
            return []
        transitions = []
        for record in list(self.records.values()):
            if record.state is ProvenanceState.CORROBORATED:
                continue
            if transition := self._resolve(record, now):
                transitions.append(transition)
        return transitions

    def _resolve(self, record: ClaimRecord, now: float) -> Transition | None:
        record.last_checked_at = now
        if record.state is ProvenanceState.CORROBORATED:
            return None

        if record.claim.confidence < 0.5 or not record.claim.entity_key:
            # A wrong entity risks corroborating against someone else's filing, so the check
            # waits rather than resolving.
            record.reason = "entity is ambiguous; holding pending rather than resolving"
            return None

        if self.index.last_updated and now - self.index.last_updated > self.stale_after:
            # Never uncorroborated: an ingestion outage must not manufacture the state that
            # carries signal.
            record.reason = "primary index is stale; holding pending"
            return None

        matches = self.index.search(
            record.claim, observed_at=record.observed_at, window=record.window
        )
        if matches:
            record.evidence = matches[:3]
        best = matches[0] if matches else None

        if best and best[1] >= self.min_match_score:
            document, score, _ = best
            return self._emit(
                record.move_to(
                    ProvenanceState.CORROBORATED,
                    at=now,
                    reason=f"matched primary document {document.doc_id} "
                    f"({document.source_type}, score {score:.2f})",
                    evidence=tuple(d.doc_id for d, _, _ in record.evidence),
                ),
                record,
            )

        record.reason = (
            f"best candidate {best[0].doc_id} scored {best[1]:.2f}, below the "
            f"{self.min_match_score:.2f} corroboration threshold"
            if best
            else "no matching primary document found yet"
        )
        if record.state is ProvenanceState.PENDING and now >= record.expires_at:
            return self._emit(
                record.move_to(
                    ProvenanceState.UNCORROBORATED,
                    at=now,
                    reason=f"no supporting primary document arrived within the {record.window}s "
                    f"window calibrated for {record.claim.claim_type}",
                ),
                record,
            )
        return None

    def _emit(self, transition: Transition, record: ClaimRecord) -> Transition:
        if self.on_transition:
            self.on_transition(record, transition)
        return transition

    def open_claims(self) -> list[ClaimRecord]:
        return [r for r in self.records.values() if r.state is ProvenanceState.PENDING]

    def review_queue(self) -> list[ClaimRecord]:
        """Expired-window items: the state that carries signal (PRD 6.4)."""
        return [r for r in self.records.values() if r.state is ProvenanceState.UNCORROBORATED]

    def statistics(self) -> dict[str, Any]:
        counts = dict.fromkeys((s.value for s in ProvenanceState), 0)
        times: dict[str, list[float]] = {}
        in_time = settled = 0
        for record in self.records.values():
            counts[record.state.value] += 1
            ttc = record.time_to_corroboration
            if ttc is not None:
                times.setdefault(str(record.claim.claim_type), []).append(ttc)
                settled += 1
                in_time += ttc <= record.window
            elif record.state is ProvenanceState.UNCORROBORATED:
                settled += 1
        return {
            "tracked": len(self.records),
            "counts": counts,
            "median_time_to_corroboration_seconds": {
                k: median(v) for k, v in sorted(times.items())
            },
            "pending_resolves_before_expiry": in_time / settled if settled else None,
        }
