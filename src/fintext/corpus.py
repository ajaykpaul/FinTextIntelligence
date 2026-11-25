"""Turning what the service sees in production into training data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import catalogue
from .approvals import ACCEPT, Resolution
from .review import CORRECT, INCORRECT, Decision

GOLD, UNLABELLED = "gold", "unlabelled"

TRAIN, HELD_OUT = "train", "held_out"

HOLDOUT_SHARE = 0.2
"""One item in five is held back for evaluation. Split on the content hash so a
rebuild puts the same item in the same place -- a random split re-rolled every
time leaks the eval set into training within a couple of runs."""

PRODUCTION_SOURCE = "production_capture"
"""The catalogue key this corpus is declared under. Declared `pending`, so
`catalogue.check_training_sources` blocks a fine-tune on it until someone has
actually cleared it."""


class CorpusError(Exception):
    """The corpus cannot be built as asked."""


@dataclass(frozen=True)
class Example:
    """One training example, with enough lineage to explain where it came from."""

    item_id: str
    text: str
    language: str
    kind: str
    split: str = TRAIN
    labels: dict[str, Any] = field(default_factory=dict)
    label_source: str = ""
    reviewer: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)
    captured_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(text: str, holdout_share: float = HOLDOUT_SHARE) -> str:
    """Deterministic split."""
    bucket = int(content_hash(text)[:8], 16) / 0xFFFFFFFF
    return HELD_OUT if bucket < holdout_share else TRAIN


# --------------------------------------------------------------------------
# Labels from the two human loops
# --------------------------------------------------------------------------


def labels_from_review(decision: Decision) -> dict[str, Any]:
    """A weekly verdict says whether the flag was right, which is a label on the flag rather than
    on the text.
    """
    if decision.verdict == CORRECT:
        return {"flag_correct": True}
    if decision.verdict == INCORRECT:
        return {"flag_correct": False}
    return {}


def labels_from_approval(resolution: Resolution) -> dict[str, Any]:
    """An accepted call names a direction, which is the strongest label the service ever gets: a
    person looked at an ambiguous item and committed to a side.
    """
    if resolution.action == ACCEPT:
        return {"tradeable": True, "direction": resolution.direction}
    return {"tradeable": False}


# --------------------------------------------------------------------------
# Building it
# --------------------------------------------------------------------------


def build_corpus(
    records: Iterable[dict[str, Any]],
    *,
    review_decisions: Sequence[Decision] = (),
    resolutions: Sequence[Resolution] = (),
    holdout_share: float = HOLDOUT_SHARE,
) -> list[Example]:
    """Join captured items to whatever human labels exist for them."""
    by_review = {d.alert_id: d for d in review_decisions}
    by_approval = {r.item_id: r for r in resolutions}

    examples: list[Example] = []
    for record in records:
        text = record.get("text") or ""
        item_id = record.get("id") or ""
        if not text or not item_id:
            continue

        labels: dict[str, Any] = {}
        source = ""
        reviewer = ""
        # An approval is a decision about this item; a weekly verdict is a decision about a flag
        # on it. Both can exist, and both are kept.
        if resolution := by_approval.get(item_id):
            labels |= labels_from_approval(resolution)
            source, reviewer = "approval", resolution.reviewer
        if decision := by_review.get(item_id):
            labels |= labels_from_review(decision)
            source = f"{source}+weekly_review" if source else "weekly_review"
            reviewer = reviewer or decision.reviewer

        examples.append(
            Example(
                item_id=item_id,
                text=text,
                language=str(record.get("language", "")),
                kind=GOLD if labels else UNLABELLED,
                split=split_for(text, holdout_share),
                labels=labels,
                label_source=source,
                reviewer=reviewer,
                lineage=record.get("lineage", {}),
                captured_at=float(record.get("scored_at", 0.0)),
            )
        )
    return examples


def write_corpus(examples: Sequence[Example], path: str | Path) -> Path:
    """Write the corpus as JSONL, after checking the licence gate."""
    catalogue.check_training_sources([PRODUCTION_SOURCE])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return destination


def summarise(examples: Sequence[Example]) -> dict[str, Any]:
    """What the corpus actually contains, which is the number that decides whether a fine-tune is
    worth running at all.
    """
    gold = [e for e in examples if e.kind == GOLD]
    directions = [e.labels.get("direction") for e in gold if e.labels.get("direction")]
    return {
        "total": len(examples),
        "gold": len(gold),
        "unlabelled": len(examples) - len(gold),
        "held_out": sum(1 for e in gold if e.split == HELD_OUT),
        "by_language": _counts(str(e.language) for e in gold),
        "by_source": _counts(e.label_source for e in gold if e.label_source),
        "directions": _counts(str(d) for d in directions),
        # main number; the rest is bookkeeping
        "trainable": sum(1 for e in gold if e.split == TRAIN),
    }


def _counts(values: Iterator[str] | Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def read_log(path: str | Path, kind: str = "score") -> Iterator[dict[str, Any]]:
    """Stream `score` records out of the decision log."""
    source = Path(path)
    if not source.exists():
        raise CorpusError(f"no decision log at {source}")
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind", kind) == kind or "sentiment" in record:
                yield record
