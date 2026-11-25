"""Metrics, acceptance gates, provenance replay and the weekly canary (PRD 8)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .provenance import ALIASES, ClaimExtractor, Document, PrimaryIndex, Tracker
from .schemas import ClaimType, Language, ProvenanceState

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float:
    """Mean per-label F1 over every declared label."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        raise ValueError("cannot evaluate an empty split")
    pairs = [(str(a), str(p)) for a, p in zip(y_true, y_pred, strict=True)]
    scores = []
    for label in map(str, labels):
        tp = sum(a == label and p == label for a, p in pairs)
        fp = sum(a != label and p == label for a, p in pairs)
        fn = sum(a == label and p != label for a, p in pairs)
        scores.append(prf(tp, fp, fn)[2])
    return sum(scores) / len(scores)


def confusion(
    y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]
) -> dict[str, dict[str, int]]:
    names = [str(x) for x in labels]
    matrix = {a: dict.fromkeys(names, 0) for a in names}
    for actual, predicted in zip(y_true, y_pred, strict=True):
        if str(actual) in matrix and str(predicted) in matrix[str(actual)]:
            matrix[str(actual)][str(predicted)] += 1
    return matrix


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC with tie correction."""
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if not positives or not negatives:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1

    positive_ranks = sum(r for r, t in zip(ranks, y_true, strict=True) if t == 1)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def binary_scores(
    y_true: Sequence[int], scores: Sequence[float], *, threshold: float
) -> tuple[float, float, float]:
    """(precision, recall, f1) at a specific threshold."""
    predicted = [s >= threshold for s in scores]
    tp = sum(t == 1 and p for t, p in zip(y_true, predicted, strict=True))
    fp = sum(t == 0 and p for t, p in zip(y_true, predicted, strict=True))
    fn = sum(t == 1 and not p for t, p in zip(y_true, predicted, strict=True))
    return prf(tp, fp, fn)


def tune_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    min_recall: float,
    min_precision: float = 0.0,
    steps: int = 200,
) -> tuple[float, dict[str, float]]:
    """Best-F1 threshold subject to a recall floor and a precision floor."""
    best_threshold, best = 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for step in range(steps + 1):
        threshold = step / steps
        precision, recall, f1 = binary_scores(y_true, scores, threshold=threshold)
        if recall >= min_recall and precision >= min_precision and f1 > best["f1"]:
            best_threshold = threshold
            best = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best


def threshold_sweep(
    y_true: Sequence[int], scores: Sequence[float]
) -> list[tuple[float, float, float, float]]:
    """(threshold, precision, recall, f1) at every distinct score, exactly."""
    positives = sum(1 for t in y_true if t == 1)
    if positives == 0:
        return []
    ranked = sorted(zip(scores, y_true, strict=True), key=lambda pair: -pair[0])
    sweep: list[tuple[float, float, float, float]] = []
    tp = fp = 0
    for index, (score, truth) in enumerate(ranked):
        if truth == 1:
            tp += 1
        else:
            fp += 1
        # Only record once every item tied at this score is counted, because a threshold cannot
        # separate items with the same score.
        if index + 1 < len(ranked) and ranked[index + 1][0] == score:
            continue
        precision = tp / (tp + fp)
        recall = tp / positives
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.append((score, precision, recall, f1))
    return sweep


@dataclass(frozen=True)
class ChosenThreshold:
    """The threshold that gets deployed, and how honestly it was arrived at."""

    threshold: float
    feasible: bool
    basis: str
    precision: float
    recall: float
    f1: float
    ceiling_f1: float


def choose_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    min_recall: float,
    min_precision: float = 0.0,
) -> ChosenThreshold:
    """Pick a deployable threshold, recall-first, and say whether it is feasible."""
    sweep = threshold_sweep(y_true, scores)
    if not sweep:
        return ChosenThreshold(0.5, False, "no positive examples to tune on", 0, 0, 0, 0)

    ceiling = max(row[3] for row in sweep)
    both = [row for row in sweep if row[2] >= min_recall and row[1] >= min_precision]
    if both:
        pick, feasible, basis = max(both, key=lambda r: r[3]), True, "best F1 within both floors"
    else:
        at_recall = [row for row in sweep if row[2] >= min_recall]
        if at_recall:
            pick = max(at_recall, key=lambda r: r[3])
            basis = "recall-first: no threshold also meets the precision floor"
        else:
            pick = max(sweep, key=lambda r: r[2])
            basis = "no threshold reaches the recall floor"
        feasible = False
    threshold, precision, recall, f1 = pick
    return ChosenThreshold(threshold, feasible, basis, precision, recall, f1, ceiling)


def expected_calibration_error(
    y_true: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    """Reported rather than gated, but it is what a threshold decision rests on."""
    if not y_true:
        return 0.0
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for truth, score in zip(y_true, scores, strict=True):
        buckets[min(bins - 1, int(score * bins))].append((truth, score))
    return sum(
        len(b) / len(y_true) * abs(sum(t for t, _ in b) / len(b) - sum(s for _, s in b) / len(b))
        for b in buckets
        if b
    )


# ---------------------------------------------------------------------------
# Acceptance gates (PRD 8)
# ---------------------------------------------------------------------------

# name -> (comparison, target, note). These are PRD constants, not deployment settings, which is
# why they live in code rather than in configs/default.toml.
GATES: dict[str, tuple[str, float, str]] = {
    "sentiment_macro_f1_en": (">=", 0.80, "English financial sentiment, frozen split."),
    "toxicity_auroc_en": (">=", 0.95, "Frozen Jigsaw split."),
    "toxicity_auroc_ar": (">=", 0.92, "Frozen OSACT split; the Arabic anchor."),
    "toxicity_f1_en": (">=", 0.85, "At the deployed recall-first threshold."),
    "toxicity_f1_ar": (">=", 0.80, "At the deployed recall-first threshold."),
    "misleading_f1_en": (">=", 0.80, "General-domain benchmark; triage only."),
    "cross_language_f1_gap": ("<=", 0.07, "Hard cap on the toxicity gap."),
    "corroboration_precision": (">=", 0.95, "Replay against known filings."),
    "claim_extraction_recall": (">=", 0.85, "Share of market-moving claims found."),
    "pending_resolves_before_expiry": (">=", 0.90, "Genuine breaking items corroborated in time."),
    "human_review_precision_en": (">=", 0.90, "Weekly sample of at least 50 alerts."),
    "latency_p95_ms_cpu": ("<=", 50.0, "256-token inputs on the reference instance."),
    "throughput_rps_per_replica": (">=", 100.0, "Sustained during the pilot."),
    "availability": (">=", 0.995, "Below the 99.9% norm by design (PRD 8.3)."),
}

# Measured and published, never gating.
REPORTED_ONLY = (
    "sentiment_macro_f1_ar",
    "dialect_slices",
    "human_review_precision_ar",
    "median_time_to_corroboration",
)

WEEK_THREE_ARABIC_GATE = 0.70


@dataclass(frozen=True)
class GateResult:
    name: str
    measured: float | None
    target: float | None
    comparison: str
    passed: bool
    gated: bool
    note: str = ""

    @property
    def status(self) -> str:
        if not self.gated:
            return "reported"
        if self.measured is None:
            return "MISSING"
        return "pass" if self.passed else "FAIL"


@dataclass
class AcceptanceReport:
    results: list[GateResult]
    notes: list[str] = field(default_factory=list)
    model_version: str = ""
    generated_at: float = field(default_factory=time.time)

    @property
    def accepted(self) -> bool:
        """Reported-only metrics never decide this."""
        return all(r.passed for r in self.results if r.gated)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if r.gated and not r.passed]

    def to_markdown(self) -> str:
        lines = [
            "# Acceptance report",
            "",
            f"Model version: `{self.model_version or 'unspecified'}`",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(self.generated_at))}",
            "",
            f"**Result: {'ACCEPTED' if self.accepted else 'NOT ACCEPTED'}**",
            "",
            "| Criterion | Measured | Target | Status |",
            "| --- | --- | --- | --- |",
        ]
        for result in self.results:
            measured = "-" if result.measured is None else f"{result.measured:.4f}"
            target = "-" if result.target is None else f"{result.comparison} {result.target:.4f}"
            lines.append(f"| {result.name} | {measured} | {target} | {result.status} |")
        if self.failures:
            lines += ["", "## Failing gates", ""]
            lines += [
                f"* **{r.name}**: {r.measured} vs {r.comparison} {r.target}. {r.note}"
                for r in self.failures
            ]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"* {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def evaluate_acceptance(
    measured: dict[str, float | None], *, model_version: str = ""
) -> AcceptanceReport:
    """Check every PRD section 8 criterion."""
    results = [
        GateResult(
            name=name,
            measured=measured.get(name),
            target=target,
            comparison=comparison,
            passed=_compare(measured.get(name), target, comparison),
            gated=name not in REPORTED_ONLY,
            note=note,
        )
        for name, (comparison, target, note) in GATES.items()
    ]
    results += [
        GateResult(
            name,
            measured.get(name),
            None,
            "",
            True,
            False,
            "Measured and published; never gates acceptance.",
        )
        for name in REPORTED_ONLY
    ]

    notes = []
    arabic = measured.get("week_three_arabic_toxicity_f1")
    if arabic is not None:
        # A scope decision, not a quality gate: failing it does not fail the build, it reduces
        # Arabic toxicity to lexicon filtering (PRD 10).
        if arabic < WEEK_THREE_ARABIC_GATE:
            notes.append(
                f"Week-three gate: Arabic toxicity F1 {arabic:.3f} is below "
                f"{WEEK_THREE_ARABIC_GATE:.2f} on the OSACT development split. Per PRD 10, reduce "
                "Arabic toxicity to published-lexicon filtering for the MVP, move the released "
                "capacity to the English models and the service, and record the decision in a "
                "written memo. This does not fail the build."
            )
        else:
            notes.append(
                f"Week-three gate: Arabic toxicity F1 {arabic:.3f} meets "
                f"{WEEK_THREE_ARABIC_GATE:.2f}; the firm Arabic commitment stands as "
                "model-plus-lexicon."
            )
    return AcceptanceReport(results, notes, model_version)


def _compare(value: float | None, target: float, comparison: str) -> bool:
    if value is None:
        return False
    return value >= target if comparison == ">=" else value <= target


# ---------------------------------------------------------------------------
# Provenance replay (PRD 8.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayCase:
    item_id: str
    text: str
    observed_at: float
    is_genuine: bool
    expected_claim_type: ClaimType | None = None
    # Documents arrive at their true publication times, so time-to-corroboration reflects real
    # confirmation delays.
    documents: tuple[Document, ...] = ()


@dataclass
class ReplayResult:
    corroboration_precision: float | None
    claim_extraction_recall: float | None
    pending_resolves_before_expiry: float | None
    median_time_to_corroboration: dict[str, float] = field(default_factory=dict)
    false_corroborations: list[str] = field(default_factory=list)
    missed_claims: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corroboration_precision": self.corroboration_precision,
            "claim_extraction_recall": self.claim_extraction_recall,
            "pending_resolves_before_expiry": self.pending_resolves_before_expiry,
            "median_time_to_corroboration_seconds": self.median_time_to_corroboration,
            "false_corroborations": self.false_corroborations,
            "missed_claims": self.missed_claims,
            "counts": self.counts,
        }


class _Clock:
    """A clock the replay drives, instead of wall time."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def replay_provenance(
    cases: Sequence[ReplayCase],
    *,
    windows: dict[str, int],
    min_match_score: float = 0.82,
    tick_seconds: int = 60,
) -> ReplayResult:
    """Replay cases on a virtual clock against a real tracker."""
    # The service builds its extractor from the index's own vocabulary (pipeline.py), so the
    # replay does too; with the seven-name alias table alone it measured a weaker extractor than
    # the one deployed.
    known = PrimaryIndex(d for c in cases for d in c.documents)
    extractor = ClaimExtractor({**ALIASES, **known.aliases()})
    clock = _Clock(min((c.observed_at for c in cases), default=0.0))
    tracker = Tracker(
        PrimaryIndex(),
        windows=windows,
        min_match_score=min_match_score,
        recheck_seconds=tick_seconds,
        # The replay controls document arrival explicitly, so index freshness is not what is
        # under test here.
        stale_after=10**9,
        clock=clock,
    )

    truth_by_claim: dict[str, ReplayCase] = {}
    expected = found = 0
    missed: list[str] = []

    # One timeline.
    ordered = sorted(cases, key=lambda c: c.observed_at)
    arrivals = sorted(
        ((d.published_at, d) for c in cases for d in c.documents), key=lambda pair: pair[0]
    )
    longest = max(windows.values())
    next_arrival = 0
    cursor = min([c.observed_at for c in ordered[:1]] + [t for t, _ in arrivals[:1]], default=0.0)
    last_claim = float("-inf")

    def deliver(now: float) -> None:
        nonlocal next_arrival
        due = []
        while next_arrival < len(arrivals) and arrivals[next_arrival][0] <= now:
            due.append(arrivals[next_arrival][1])
            next_arrival += 1
        if due:
            tracker.on_documents(due, now=now)

    def advance(to: float) -> None:
        nonlocal cursor
        while cursor < to:
            clock.now = cursor
            deliver(cursor)
            tracker.recheck_due(now=cursor)
            if cursor - last_claim > longest + tick_seconds:
                # Every window has closed: nothing can change until the next document or claim,
                # so skip the idle stretch in one step.
                upcoming = arrivals[next_arrival][0] if next_arrival < len(arrivals) else to
                cursor = max(cursor + tick_seconds, min(to, upcoming))
            else:
                cursor += tick_seconds
        clock.now = to
        deliver(to)

    for case in ordered:
        advance(case.observed_at)
        claim = extractor.extract(case.text)
        if case.expected_claim_type is not None:
            expected += 1
            if claim and claim.claim_type is case.expected_claim_type:
                found += 1
            else:
                missed.append(case.item_id)
        result = tracker.check(claim, item_id=case.item_id, observed_at=case.observed_at)
        if result.claim_id:
            truth_by_claim[result.claim_id] = case
        last_claim = case.observed_at

    advance(max((c.observed_at for c in cases), default=cursor) + longest + tick_seconds)
    tracker.recheck_due(now=clock.now)

    corroborated = true_positives = genuine = in_time = 0
    false_corroborations: list[str] = []
    times: dict[str, list[float]] = {}
    counts = dict.fromkeys((s.value for s in ProvenanceState), 0)

    for record in tracker:
        counts[record.state.value] += 1
        truth = truth_by_claim.get(record.claim_id)
        if truth is None:
            continue
        genuine += truth.is_genuine
        if record.state is ProvenanceState.CORROBORATED:
            corroborated += 1
            ttc = record.time_to_corroboration or 0.0
            times.setdefault(str(record.claim.claim_type), []).append(ttc)
            if truth.is_genuine:
                true_positives += 1
                in_time += ttc <= record.window
            else:
                # fabricated item corroborated by someone else's filing
                false_corroborations.append(truth.item_id)

    return ReplayResult(
        corroboration_precision=true_positives / corroborated if corroborated else None,
        claim_extraction_recall=found / expected if expected else None,
        pending_resolves_before_expiry=in_time / genuine if genuine else None,
        median_time_to_corroboration={k: median(v) for k, v in sorted(times.items())},
        false_corroborations=false_corroborations,
        missed_claims=missed,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Weekly canary (PRD 8.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryCase:
    text: str
    language: Language
    should_flag: bool
    technique: str


# Obfuscations the deployed configuration has to survive, plus negative controls, a canary that
# measures recall alone rewards a model that flags everything.
CANARY: tuple[CanaryCase, ...] = (
    CanaryCase("i will kill you", Language.EN, True, "plain"),
    CanaryCase("i will k1ll you", Language.EN, True, "leetspeak"),
    CanaryCase("i w i l l  k i l l  y o u", Language.EN, True, "spacing"),
    CanaryCase("k.y.s", Language.EN, True, "substitution"),
    CanaryCase("kys", Language.EN, True, "plain"),
    CanaryCase("you should die", Language.EN, True, "plain"),
    CanaryCase("f*u*c*k you", Language.EN, True, "substitution"),
    CanaryCase("fuuuck you", Language.EN, True, "repetition"),
    CanaryCase("you are a pathetic moron", Language.EN, True, "plain"),
    # Arabic-script context: language is identified from script, so a bare Latin placeholder
    # would be screened against the English list.
    CanaryCase("سهم البنك __ar_slur_placeholder_1__", Language.AR, True, "plain"),
    CanaryCase("quarterly results beat expectations", Language.EN, False, "plain"),
    CanaryCase("the killing of the deal cost shareholders", Language.EN, False, "plain"),
    CanaryCase("this stock is a bloodbath today", Language.EN, False, "figurative"),
)

CANARY_ALERT_DROP = 0.05


@dataclass
class CanaryResult:
    recall: float
    baseline: float
    total: int
    missed: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    by_technique: dict[str, float] = field(default_factory=dict)

    @property
    def drop(self) -> float:
        return self.baseline - self.recall

    @property
    def triggers_retraining_review(self) -> bool:
        return self.drop > CANARY_ALERT_DROP

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": round(self.recall, 4),
            "baseline": self.baseline,
            "drop": round(self.drop, 4),
            "total": self.total,
            "missed": self.missed,
            "false_positives": self.false_positives,
            "by_technique": {k: round(v, 4) for k, v in self.by_technique.items()},
            "triggers_retraining_review": self.triggers_retraining_review,
        }


def run_canary(
    is_toxic: Callable[[str, Language], bool],
    *,
    cases: Sequence[CanaryCase] = CANARY,
    baseline: float = 1.0,
) -> CanaryResult:
    """Score the canary set."""
    positives = [c for c in cases if c.should_flag]
    if not positives:
        raise ValueError("canary set contains no positive cases")

    missed = [c.text for c in positives if not is_toxic(c.text, c.language)]
    by_technique: dict[str, float] = {}
    for technique in sorted({c.technique for c in positives}):
        group = [c for c in positives if c.technique == technique]
        by_technique[technique] = sum(is_toxic(c.text, c.language) for c in group) / len(group)

    return CanaryResult(
        recall=1.0 - len(missed) / len(positives),
        baseline=baseline,
        total=len(positives),
        missed=missed,
        false_positives=[
            c.text for c in cases if not c.should_flag and is_toxic(c.text, c.language)
        ],
        by_technique=by_technique,
    )
