"""The queue a gated call waits in until a person accepts or rejects it."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ACCEPT, REJECT = "accept", "reject"
ACTIONS = (ACCEPT, REJECT)

BUY, SELL = "buy", "sell"
DIRECTIONS = (BUY, SELL)


class ApprovalError(Exception):
    """A resolution that does not make sense as a decision."""


@dataclass(frozen=True)
class Waiting:
    """One gated item, waiting for a person."""

    item_id: str
    text: str
    language: str
    priority: str
    reason: str
    lean: str = ""
    conviction: float = 0.0
    queued_at: float = field(default_factory=time.time)
    lineage: dict[str, Any] = field(default_factory=dict)

    def age_ms(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.queued_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["age_ms"] = round(self.age_ms(), 1)
        return payload


@dataclass(frozen=True)
class Resolution:
    """What the person decided, and how long it took them."""

    item_id: str
    action: str
    reviewer: str
    direction: str = ""
    note: str = ""
    waited_ms: float = 0.0
    decided_at: float = field(default_factory=time.time)
    resolution_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ApprovalError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if not self.reviewer.strip():
            raise ApprovalError("a resolution has to record who made it")
        if self.action == ACCEPT and self.direction not in DIRECTIONS:
            # no direction stored: the gate didn't have one
            raise ApprovalError(f"accepting requires a direction, one of {DIRECTIONS}")
        if self.action == REJECT and self.direction:
            raise ApprovalError("a rejected item does not carry a direction")

    @property
    def outcome(self) -> str:
        """What the desk acts on: buy, sell, or nothing at all."""
        return self.direction if self.action == ACCEPT else "no trade"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome
        return payload


class ApprovalQueue:
    """Gated items in, resolutions out, both on disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._waiting: dict[str, Waiting] = {}
        self._resolved: dict[str, Resolution] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self._replay()

    # -- writing ---------------------------------------------------------

    def add(self, item: Waiting) -> Waiting:
        """Put a gated item in the queue."""
        with self._lock:
            if item.item_id in self._waiting or item.item_id in self._resolved:
                return self._waiting.get(item.item_id, item)
            self._waiting[item.item_id] = item
        self._append("waiting", item.to_dict())
        return item

    def resolve(
        self, item_id: str, *, action: str, reviewer: str, direction: str = "", note: str = ""
    ) -> Resolution:
        with self._lock:
            item = self._waiting.get(item_id)
            if item is None:
                if item_id in self._resolved:
                    raise ApprovalError(f"{item_id} has already been resolved")
                raise ApprovalError(f"{item_id} is not in the queue")
            resolution = Resolution(
                item_id=item_id,
                action=action,
                reviewer=reviewer,
                direction=direction,
                note=note,
                waited_ms=round(item.age_ms(), 1),
            )
            del self._waiting[item_id]
            self._resolved[item_id] = resolution
        self._append("resolved", resolution.to_dict())
        return resolution

    # -- reading ---------------------------------------------------------

    def waiting(self) -> list[Waiting]:
        """Oldest first, and urgent ahead of routine."""
        with self._lock:
            items = list(self._waiting.values())
        return sorted(items, key=lambda i: (i.priority != "now", i.queued_at))

    def resolution(self, item_id: str) -> Resolution | None:
        return self._resolved.get(item_id)

    def resolutions(self) -> list[Resolution]:
        return sorted(self._resolved.values(), key=lambda r: r.decided_at)

    def summary(self) -> dict[str, Any]:
        waiting = self.waiting()
        resolved = self.resolutions()
        waits = [r.waited_ms for r in resolved]
        return {
            "waiting": len(waiting),
            "urgent_waiting": sum(1 for i in waiting if i.priority == "now"),
            "oldest_age_ms": round(waiting[0].age_ms(), 1) if waiting else 0.0,
            "resolved": len(resolved),
            "accepted": sum(1 for r in resolved if r.action == ACCEPT),
            "rejected": sum(1 for r in resolved if r.action == REJECT),
            # Median rather than mean: one reviewer going to lunch should not look like a
            # service-wide slowdown.
            "median_wait_ms": round(sorted(waits)[len(waits) // 2], 1) if waits else 0.0,
        }

    # -- disk ------------------------------------------------------------

    def _append(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")

    def _replay(self) -> None:
        """Rebuild the queue from the file."""
        assert self.path is not None
        for record in _records(self.path):
            kind = record.pop("kind", "")
            record.pop("age_ms", None)
            record.pop("outcome", None)
            if kind == "waiting":
                self._waiting[record["item_id"]] = Waiting(**record)
            elif kind == "resolved":
                self._resolved[record["item_id"]] = Resolution(**record)
                self._waiting.pop(record["item_id"], None)


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A half-written last line after a hard kill. Skip it rather than refuse to
                    # start.
                    continue
