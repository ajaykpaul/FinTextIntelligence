"""Weekly human review of flagged items (PRD 5.3, 8.4)."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MIN_BATCH_SIZE = 50
"""PRD 8.4. A smaller batch is allowed but the report says it fell short."""

ENGLISH_PRECISION_GATE = 0.90

CORRECT, INCORRECT, UNSURE = "correct", "incorrect", "unsure"
VERDICTS = (CORRECT, INCORRECT, UNSURE)


class ReviewError(Exception):
    pass


@dataclass(frozen=True)
class Alert:
    """One flagged item, pulled from the decision log for review."""

    alert_id: str
    item_id: str
    at: float
    language: str
    profile: str
    kind: str  # "toxicity" or "misleading"
    score: float
    threshold: float
    text: str
    summary: str
    spans: tuple[dict[str, Any], ...] = ()
    lineage: dict[str, Any] = field(default_factory=dict)

    @property
    def mt_assisted(self) -> bool:
        """Arabic is reviewed through machine translation."""
        return self.language == "ar"

    @property
    def lineage_key(self) -> str:
        """What has to be constant for a set of decisions to share one number."""
        versions = self.lineage.get("lexicon_versions", {})
        return "|".join(
            [
                str(self.lineage.get("model_version", "?")),
                str(self.lineage.get("threshold_version", "?")),
                str(self.lineage.get("rules_version", "?")),
                *(f"{k}={v}" for k, v in sorted(versions.items())),
            ]
        )


@dataclass(frozen=True)
class Decision:
    """A reviewer's verdict on one flag."""

    alert_id: str
    verdict: str
    reviewer: str
    at: float
    language: str
    kind: str
    lineage_key: str
    mt_assisted: bool
    note: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ReviewError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")


@dataclass(frozen=True)
class Batch:
    """A week's sample, recorded so it can be redrawn."""

    batch_id: str
    created_at: float
    window_start: float
    window_end: float
    seed: int
    alerts: tuple[Alert, ...]

    @property
    def short(self) -> bool:
        return len(self.alerts) < MIN_BATCH_SIZE

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Batch:
        return cls(
            batch_id=payload["batch_id"],
            created_at=payload["created_at"],
            window_start=payload["window_start"],
            window_end=payload["window_end"],
            seed=payload["seed"],
            alerts=tuple(
                Alert(**{**a, "spans": tuple(a.get("spans", ()))}) for a in payload["alerts"]
            ),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), "utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Batch | None:
        file = Path(path)
        if not file.exists():
            return None
        return cls.from_dict(json.loads(file.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "seed": self.seed,
            "size": len(self.alerts),
            "short_of_minimum": self.short,
            "minimum": MIN_BATCH_SIZE,
            "alerts": [asdict(a) for a in self.alerts],
        }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def read_alerts(log_path: str | Path) -> Iterator[Alert]:
    """Every flagged item in the decision log, newest last."""
    path = Path(log_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partially written line at the tail of an open log
        if record.get("kind") != "score":
            continue
        yield from _alerts_in(record)


def _alerts_in(record: dict[str, Any]) -> Iterator[Alert]:
    text = record.get("text")
    if not text:
        # Unflagged items carry no text by design, so there is nothing to review.
        return
    content_hash = record.get("content_hash", record.get("id", ""))

    for kind, flagged_key in (("toxicity", "toxic"), ("misleading", "likely_misleading")):
        payload = record.get(kind)
        if not payload or not payload.get(flagged_key):
            continue
        yield Alert(
            alert_id=hashlib.sha256(f"{content_hash}:{kind}".encode()).hexdigest()[:16],
            item_id=str(record.get("id", "")),
            at=float(record.get("logged_at", 0.0)),
            language=str(record.get("language", "unknown")),
            profile=str(record.get("profile", "unknown")),
            kind=kind,
            score=float(payload.get("score", 0.0)),
            threshold=float(payload.get("threshold", 0.0)),
            text=str(text),
            summary=str(payload.get("explanation", {}).get("summary", "")),
            spans=tuple(payload.get("explanation", {}).get("spans", ())),
            lineage=dict(record.get("lineage", {})),
        )


def sample_batch(
    log_path: str | Path,
    *,
    window_start: float,
    window_end: float,
    size: int = MIN_BATCH_SIZE,
    seed: int | None = None,
    exclude: Iterable[str] = (),
) -> Batch:
    """Draw a reproducible sample of alerts raised inside a window."""
    if window_end <= window_start:
        raise ReviewError("window_end must be after window_start")

    already_seen = set(exclude)
    unique: dict[str, Alert] = {}
    for alert in read_alerts(log_path):
        if not window_start <= alert.at < window_end:
            continue
        if alert.alert_id in already_seen:
            continue
        unique.setdefault(alert.alert_id, alert)

    population = sorted(unique.values(), key=lambda a: (a.at, a.alert_id))
    chosen_seed = int(window_start) if seed is None else seed
    if len(population) > size:
        population = random.Random(chosen_seed).sample(population, size)
        population.sort(key=lambda a: (a.at, a.alert_id))

    return Batch(
        batch_id=uuid.uuid4().hex[:12],
        created_at=time.time(),
        window_start=window_start,
        window_end=window_end,
        seed=chosen_seed,
        alerts=tuple(population),
    )


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class ReviewStore:
    """Append-only JSONL of reviewer decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self, alert: Alert, *, verdict: str, reviewer: str, note: str = "", at: float | None = None
    ) -> Decision:
        decision = Decision(
            alert_id=alert.alert_id,
            verdict=verdict,
            reviewer=reviewer,
            at=time.time() if at is None else at,
            language=alert.language,
            kind=alert.kind,
            lineage_key=alert.lineage_key,
            mt_assisted=alert.mt_assisted,
            note=note,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
        return decision

    def all(self) -> list[Decision]:
        if not self.path.exists():
            return []
        decisions = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                decisions.append(Decision(**json.loads(line)))
        return decisions

    def latest(self) -> dict[str, Decision]:
        """The current verdict per alert: last write wins."""
        current: dict[str, Decision] = {}
        for decision in sorted(self.all(), key=lambda d: d.at):
            current[decision.alert_id] = decision
        return current


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


@dataclass
class LanguageResult:
    language: str
    correct: int
    incorrect: int
    unsure: int
    gated: bool
    mt_assisted: bool

    @property
    def judged(self) -> int:
        """Unsure items are excluded: forcing a binary would corrupt the metric, and an ambiguous
        item is signal for the native-speaker hours.
        """
        return self.correct + self.incorrect

    @property
    def precision(self) -> float | None:
        return self.correct / self.judged if self.judged else None

    @property
    def passes(self) -> bool | None:
        if not self.gated:
            return None
        if self.precision is None:
            return False
        return self.precision >= ENGLISH_PRECISION_GATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "unsure": self.unsure,
            "judged": self.judged,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "gated": self.gated,
            "passes": self.passes,
            "mt_assisted": self.mt_assisted,
            "qualification": (
                "Reviewed via machine translation; reported, not gated (PRD 8.4)."
                if self.mt_assisted
                else ""
            ),
        }


@dataclass
class ReviewReport:
    by_language: dict[str, LanguageResult]
    lineages: list[str]
    reviewed: int
    outstanding: int = 0

    @property
    def spans_multiple_lineages(self) -> bool:
        """Precision averaged across a threshold change is not one number."""
        return len(self.lineages) > 1

    @property
    def passes(self) -> bool:
        english = self.by_language.get("en")
        return bool(english and english.passes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewed": self.reviewed,
            "outstanding": self.outstanding,
            "gate": ENGLISH_PRECISION_GATE,
            "passes": self.passes,
            "spans_multiple_lineages": self.spans_multiple_lineages,
            "lineages": self.lineages,
            "by_language": {k: v.to_dict() for k, v in sorted(self.by_language.items())},
            "measured_metrics": self.as_measured(),
        }

    def as_measured(self) -> dict[str, float]:
        """Shaped for `fintext accept`, so the gate stops being self-certified."""
        out = {}
        for language, result in self.by_language.items():
            if result.precision is not None:
                out[f"human_review_precision_{language}"] = round(result.precision, 4)
        return out


def build_report(decisions: Sequence[Decision], *, outstanding: int = 0) -> ReviewReport:
    """Precision per language from the current verdicts."""
    by_language: dict[str, LanguageResult] = {}
    for decision in decisions:
        result = by_language.setdefault(
            decision.language,
            LanguageResult(
                language=decision.language,
                correct=0,
                incorrect=0,
                unsure=0,
                # English is the only gated language (PRD 8.4).
                gated=decision.language == "en",
                mt_assisted=decision.mt_assisted,
            ),
        )
        if decision.verdict == CORRECT:
            result.correct += 1
        elif decision.verdict == INCORRECT:
            result.incorrect += 1
        else:
            result.unsure += 1

    return ReviewReport(
        by_language=by_language,
        lineages=sorted({d.lineage_key for d in decisions}),
        reviewed=len(decisions),
        outstanding=outstanding,
    )


def to_csv(batch: Batch, decisions: dict[str, Decision], path: str | Path) -> Path:
    """Spreadsheet export, which the PRD says is sufficient for the MVP (5.3)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alert_id",
                "item_id",
                "flagged_at",
                "language",
                "kind",
                "score",
                "threshold",
                "text",
                "why_flagged",
                "verdict",
                "reviewer",
                "note",
                "mt_assisted",
                "lineage",
            ]
        )
        for alert in batch.alerts:
            decision = decisions.get(alert.alert_id)
            writer.writerow(
                [
                    alert.alert_id,
                    alert.item_id,
                    alert.at,
                    alert.language,
                    alert.kind,
                    f"{alert.score:.4f}",
                    f"{alert.threshold:.2f}",
                    alert.text,
                    alert.summary,
                    decision.verdict if decision else "",
                    decision.reviewer if decision else "",
                    decision.note if decision else "",
                    "yes" if alert.mt_assisted else "no",
                    alert.lineage_key,
                ]
            )
    return target
