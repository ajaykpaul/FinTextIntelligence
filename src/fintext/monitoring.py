"""Per-replica counters, latency and drift."""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from collections import Counter, deque
from collections.abc import Sequence

BUCKETS_MS = (1, 2, 5, 10, 20, 35, 50, 75, 100, 250, 500, 1000)


class Latency:
    """Bucket counts for export, plus a bounded sample for live quantiles."""

    def __init__(self, buckets: Sequence[float] = BUCKETS_MS, sample_size: int = 4096) -> None:
        self.buckets = tuple(buckets)
        self.counts = [0] * len(self.buckets)
        self.overflow = 0
        self.count = 0
        self.total_ms = 0.0
        self.samples: deque[float] = deque(maxlen=sample_size)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total_ms += value
        self.samples.append(value)
        index = bisect_left(self.buckets, value)
        if index >= len(self.buckets):
            self.overflow += 1
        else:
            self.counts[index] += 1

    def quantile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    def cumulative(self) -> list[tuple[float, int]]:
        running = 0
        out = []
        for bound, count in zip(self.buckets, self.counts, strict=True):
            running += count
            out.append((bound, running))
        return out


def psi(baseline: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """Population stability index: <0.1 stable, 0.1-0.2 watch, >0.2 alert."""
    if not baseline or not current:
        return 0.0
    ordered = sorted(baseline)
    edges = [ordered[min(len(ordered) - 1, len(ordered) * i // bins)] for i in range(1, bins)]

    def share(values: Sequence[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            counts[bisect_left(edges, value)] += 1
        # Floor at epsilon: one empty bin should not make the whole statistic infinite.
        return [max(c / len(values), 1e-6) for c in counts]

    return sum(
        (c - b) * math.log(c / b) for b, c in zip(share(ordered), share(current), strict=True)
    )


class Monitor:
    def __init__(self, *, drift_window: int = 2000, drift_alert: float = 0.20) -> None:
        self.started_at = time.time()
        self.drift_alert = drift_alert
        self.drift_window = drift_window
        self.latency = Latency()
        self.stages: dict[str, Latency] = {}
        self.requests: Counter[str] = Counter()
        self.outcomes: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.degradations: Counter[str] = Counter()
        self.languages: Counter[str] = Counter()
        self.scores: dict[str, deque[float]] = {}
        self.baselines: dict[str, list[float]] = {}

    def request(
        self,
        *,
        profile: str,
        language: str,
        latency_ms: float,
        stages: dict[str, float] | None = None,
        degraded: Sequence[str] = (),
    ) -> None:
        self.latency.observe(latency_ms)
        self.requests[profile] += 1
        self.languages[language] += 1
        for stage, value in (stages or {}).items():
            self.stages.setdefault(stage, Latency()).observe(value)
        for capability in degraded:
            self.degradations[capability] += 1

    def score(self, task: str, value: float) -> None:
        self.scores.setdefault(task, deque(maxlen=self.drift_window)).append(value)

    def outcome(self, name: str) -> None:
        self.outcomes[name] += 1

    def error(self, kind: str) -> None:
        self.errors[kind] += 1

    def set_baseline(self, task: str, values: Sequence[float]) -> None:
        """Freeze the reference distribution, normally the frozen test split."""
        self.baselines[task] = list(values)

    def drift(self) -> dict[str, float]:
        return {
            task: psi(baseline, list(self.scores.get(task, ())))
            for task, baseline in self.baselines.items()
        }

    def drift_alerts(self) -> list[str]:
        return [task for task, value in self.drift().items() if value > self.drift_alert]

    def snapshot(self) -> dict[str, object]:
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests": dict(self.requests),
            "outcomes": dict(self.outcomes),
            "errors": dict(self.errors),
            "degradations": dict(self.degradations),
            "languages": dict(self.languages),
            "latency_ms": {
                "count": self.latency.count,
                "mean": round(self.latency.total_ms / self.latency.count, 3)
                if self.latency.count
                else 0.0,
                "p50": round(self.latency.quantile(0.50), 3),
                "p95": round(self.latency.quantile(0.95), 3),
                "p99": round(self.latency.quantile(0.99), 3),
            },
            "stage_p95_ms": {k: round(v.quantile(0.95), 3) for k, v in sorted(self.stages.items())},
            "drift_psi": {k: round(v, 4) for k, v in self.drift().items()},
            "drift_alerts": self.drift_alerts(),
        }

    def prometheus(self) -> str:
        lines = [
            "# HELP fintext_latency_ms Request latency in milliseconds.",
            "# TYPE fintext_latency_ms histogram",
            *(
                f'fintext_latency_ms_bucket{{le="{bound}"}} {count}'
                for bound, count in self.latency.cumulative()
            ),
            f'fintext_latency_ms_bucket{{le="+Inf"}} {self.latency.count}',
            f"fintext_latency_ms_sum {self.latency.total_ms:.3f}",
            f"fintext_latency_ms_count {self.latency.count}",
        ]
        for name, counter, help_text in (
            ("fintext_requests_total", self.requests, "Requests by consumer profile."),
            ("fintext_degradations_total", self.degradations, "Capabilities dropped or failed."),
            ("fintext_errors_total", self.errors, "Errors by kind."),
        ):
            label = {
                "fintext_requests_total": "profile",
                "fintext_degradations_total": "capability",
            }.get(name, "kind")
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
            lines += [
                f'{name}{{{label}="{key}"}} {value}' for key, value in sorted(counter.items())
            ]

        lines += [
            "# HELP fintext_drift_psi Prediction drift per task.",
            "# TYPE fintext_drift_psi gauge",
        ]
        lines += [
            f'fintext_drift_psi{{task="{task}"}} {value:.6f}'
            for task, value in sorted(self.drift().items())
        ]
        return "\n".join(lines) + "\n"
