"""The decision log: one JSONL record per scored item and per state change."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any


class DecisionLog:
    def __init__(self, path: str | Path | None, *, max_queue: int = 10_000) -> None:
        self.path = Path(path) if path else None
        self.written = 0
        self.dropped = 0
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._drain, daemon=True, name="fintext-log")
            self._thread.start()

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.path:
            return
        try:
            self.queue.put_nowait({"kind": kind, "logged_at": time.time(), **payload})
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                self.written += 1

    def flush(self, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self, timeout: float = 2.0) -> None:
        self.flush(timeout)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
