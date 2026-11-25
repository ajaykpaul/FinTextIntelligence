"""Scoring orchestration: the request path."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager

from . import __version__, explain, preprocess, scam_rules
from .audit import DecisionLog
from .config import Config
from .lexicon import Lexicons
from .monitoring import Monitor
from .provenance import ALIASES, NOT_APPLICABLE, ClaimExtractor, PrimaryIndex, Tracker
from .schemas import (
    Capability,
    Explanation,
    Language,
    Lineage,
    MisleadingResult,
    ScoreRequest,
    ScoreResponse,
    Sentiment,
    SentimentResult,
    ToxicityResult,
)
from .scoring import Heuristic, Scores

# Below this many milliseconds left, starting another stage risks blowing the budget rather than
# degrading cleanly.
STAGE_FLOOR_MS = 1.5


class ScoringFailed(Exception):
    """Raised for a fail-closed profile; the transport turns it into a 503."""


class Budget:
    """A monotonic deadline shared across stages."""

    def __init__(self, total_ms: float) -> None:
        self.total_ms = total_ms
        self.start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0

    def exhausted(self, floor_ms: float = 0.0) -> bool:
        return self.total_ms - self.elapsed_ms <= floor_ms


class Pipeline:
    def __init__(
        self,
        config: Config,
        *,
        backend=None,
        lexicons: Lexicons | None = None,
        index: PrimaryIndex | None = None,
        monitor: Monitor | None = None,
        log: DecisionLog | None = None,
        shadow_backend=None,
        clock=time.time,
    ) -> None:
        self.config = config
        self.clock = clock
        self.backend = backend or Heuristic()
        self.shadow_backend = shadow_backend
        self.lexicons = lexicons if lexicons is not None else Lexicons.load(config)
        self.monitor = monitor or Monitor()
        self.log = log or DecisionLog(
            config.path(config.decision_log) if config.decision_log else None
        )
        self.index = (
            index if index is not None else PrimaryIndex.from_jsonl(config.path(config.index_path))
        )
        # The index supplies the entity vocabulary.
        self.claims = ClaimExtractor({**ALIASES, **self.index.aliases()})
        self.tracker = Tracker(
            self.index,
            windows=config.windows,
            min_match_score=config.min_match_score,
            stale_after=config.index_stale_seconds,
            recheck_seconds=config.recheck_seconds,
            clock=clock,
            on_transition=self._log_transition,
        )
        self._requests_seen = 0

    # -- public API --------------------------------------------------------

    def warmup(self) -> None:
        self.backend.warmup()

    def close(self) -> None:
        self.log.close()

    def lineage(self, model_version: str) -> Lineage:
        return Lineage(
            model_version=model_version,
            lexicon_versions=self.lexicons.versions,
            threshold_version=self.config.threshold_version,
            rules_version=scam_rules.VERSION,
            service_version=__version__,
            config_digest=self.config.digest(),
        )

    def score(self, request: ScoreRequest) -> ScoreResponse:
        budget = Budget(self.config.budget_ms(str(request.profile)))
        stages: dict[str, float] = {}
        degraded: list[str] = []
        skipped: dict[str, str] = {}

        try:
            with self._timed(stages, "preprocess"):
                item = preprocess.prepare(
                    request.text,
                    max_tokens=self.config.max_tokens,
                    language_hint=request.language,
                )
            with self._timed(stages, "inference"):
                scores = self.backend.predict(item.text, item.language)
        except Exception as exc:
            self.monitor.error("inference")
            return self._degrade(request, exc, budget, stages)

        self._maybe_shadow(item.text, item.language)
        wanted = set(request.capabilities)
        sentiment = toxicity = misleading = provenance = None

        if Capability.SENTIMENT in wanted:
            with self._timed(stages, "sentiment"):
                sentiment = self._sentiment(scores, item.language)
        else:
            skipped["sentiment"] = "not requested"

        if Capability.TOXICITY in wanted:
            with self._timed(stages, "toxicity"):
                toxicity = self._toxicity(scores, item, request, budget)
        else:
            skipped["toxicity"] = "not requested"

        if Capability.MISLEADING not in wanted:
            skipped["misleading"] = "not requested"
        elif item.language is Language.AR:
            # English-only by scope (PRD 3.1). Reported as skipped, not null, so Arabic traffic
            # does not look like a continuous partial outage.
            skipped["misleading"] = "English-only capability; item is Arabic"
        else:
            with self._timed(stages, "misleading"):
                misleading = self._misleading(scores, item.text)

        if Capability.PROVENANCE not in wanted:
            skipped["provenance"] = "not requested"
        elif budget.exhausted(STAGE_FLOOR_MS):
            degraded.append("provenance")
        else:
            try:
                with self._timed(stages, "provenance"):
                    provenance = self._provenance(item.text, request)
            except Exception as exc:
                self.monitor.error("provenance")
                self.log.write(
                    "stage_error", {"id": request.id, "stage": "provenance", "error": str(exc)}
                )
                degraded.append("provenance")

        response = ScoreResponse(
            id=request.id,
            language=item.language,
            profile=request.profile,
            lineage=self.lineage(request.model_version or scores.version),
            scored_at=self.clock(),
            latency_ms=budget.elapsed_ms,
            sentiment=sentiment,
            toxicity=toxicity,
            misleading=misleading,
            provenance=provenance,
            scored_text=item.text if request.explain else None,
            stage_latency_ms=stages,
            degraded=tuple(degraded),
            skipped=skipped,
            truncated=item.truncated,
        )
        self._record(request, response, item)
        return response

    def score_many(self, requests: Sequence[ScoreRequest]) -> list[ScoreResponse]:
        """Batch scoring for backtests."""
        return [self.score(r) for r in requests]

    # -- stages ------------------------------------------------------------

    def _sentiment(self, scores: Scores, language: Language) -> SentimentResult:
        label, confidence = scores.top_sentiment()
        if confidence < self.config.min_sentiment_confidence:
            # Low confidence reads as neutral, which is the substantive answer for a trading
            # gate: a directionless item should suppress a signal.
            label = Sentiment.NEUTRAL
        self.monitor.score("sentiment_confidence", confidence)
        return SentimentResult(
            label=label,
            scores=scores.sentiment,
            confidence=confidence,
            language=language,
            acceptance_gated=language is Language.EN,
        )

    def _toxicity(
        self, scores: Scores, item: preprocess.Item, request: ScoreRequest, budget: Budget
    ) -> ToxicityResult:
        threshold = self.config.toxicity_threshold(str(item.language))
        forced, matches = self.lexicons.floor(item.text, item.language)
        toxic = forced or scores.toxicity >= threshold
        self.monitor.score("toxicity", scores.toxicity)
        if toxic:
            self.monitor.outcome("toxic_flag")

        subtypes = dict(scores.subtypes)
        for match in matches:
            # A lexicon hit is direct evidence for its subtype; take the stronger of the two
            # rather than letting a low model score bury it.
            key = str(match.entry.subtype)
            subtypes[key] = max(subtypes.get(key, 0.0), 0.95 if match.forces_toxic else 0.6)

        return ToxicityResult(
            toxic=toxic,
            score=max(scores.toxicity, 0.99) if forced else scores.toxicity,
            threshold=threshold,
            subtypes=subtypes,
            lexicon_override=forced,
            explanation=self._toxicity_explanation(
                item, request, budget, matches, forced, scores.toxicity, threshold, toxic
            ),
        )

    def _toxicity_explanation(
        self, item, request, budget, matches, forced, score, threshold, toxic
    ) -> Explanation:
        if not toxic:
            return Explanation(
                f"Not toxic: model score {score:.2f} is below the {threshold:.2f} "
                "recall-first threshold and no lexicon entry matched.",
                method="threshold",
            )
        if matches:
            note = (
                "Raised by the high-severity lexicon floor regardless of model score."
                if forced
                else f"Model score {score:.2f} against a {threshold:.2f} threshold."
            )
            return explain.explain(
                [m.span for m in matches], subject="Toxicity flag", method="lexicon", note=note
            )
        # Attribution costs a forward pass per token, so it runs only when the budget allows:
        # compliance has 500 ms, trading does not.
        if request.explain and not budget.exhausted(10.0):
            spans = explain.attribute(
                item.text, lambda text: self.backend.predict(text, item.language).toxicity
            )
            if spans:
                return explain.explain(
                    spans,
                    subject="Toxicity flag",
                    method="occlusion",
                    note=f"Model score {score:.2f}, threshold {threshold:.2f}.",
                )
        return Explanation(
            f"Toxicity flag from the model score {score:.2f} against a {threshold:.2f} "
            "threshold; attribution skipped to hold the latency budget.",
            method="score_only",
        )

    def _misleading(self, scores: Scores, text: str) -> MisleadingResult:
        threshold = self.config.misleading
        rules = scam_rules.evaluate(text)
        # Rules are independent of the model, so a model regression, or the week-three cut to
        # rules-only, leaves this coverage intact.
        flagged = rules.matched or scores.misleading >= threshold
        score = max(scores.misleading, rules.score)
        self.monitor.score("misleading", scores.misleading)
        if flagged:
            self.monitor.outcome("misleading_flag")

        if not flagged:
            explanation = Explanation(
                f"No misleading-content flag: model score {scores.misleading:.2f} is below the "
                f"{threshold:.2f} threshold and no scam pattern matched.",
                method="threshold",
            )
            priority = "normal"
        else:
            priority = (
                "high"
                if rules.priority == "high" or score >= self.config.misleading_high_priority
                else "review"
            )
            triage_note = "This is a triage signal, not an assessment of truth."
            if rules.matched:
                explanation = explain.explain(
                    rules.spans,
                    subject="Likely-misleading triage flag",
                    method="scam_rules",
                    note=f"Matched {'; '.join(rules.descriptions)} {triage_note}",
                )
            else:
                explanation = Explanation(
                    f"Likely-misleading triage flag from the model score {score:.2f} against a "
                    f"{threshold:.2f} threshold. {triage_note}",
                    method="model",
                )

        return MisleadingResult(
            likely_misleading=flagged,
            score=score,
            threshold=threshold,
            rule_matches=rules.rule_ids,
            review_priority=priority,
            explanation=explanation,
        )

    def _provenance(self, text: str, request: ScoreRequest):
        claim = self.claims.extract(text)
        if claim is None:
            return NOT_APPLICABLE
        result = self.tracker.check(
            claim, item_id=request.id, observed_at=request.observed_at, now=self.clock()
        )
        self.monitor.outcome(f"provenance_{result.state}")
        return result

    # -- failure, shadow and logging ---------------------------------------

    def _degrade(
        self, request: ScoreRequest, exc: Exception, budget: Budget, stages: dict[str, float]
    ) -> ScoreResponse:
        """Fail open or fail closed by profile (PRD 8.3)."""
        profile = str(request.profile)
        if profile in self.config.fail_closed:
            self.log.write("fail_closed", {"id": request.id, "profile": profile, "error": str(exc)})
            raise ScoringFailed(f"scoring failed for a fail-closed profile ({profile}): {exc}")

        self.log.write("fail_open", {"id": request.id, "profile": profile, "error": str(exc)})
        self.monitor.outcome("fail_open")
        language = request.language or Language.UNKNOWN
        response = ScoreResponse(
            id=request.id,
            language=language,
            profile=request.profile,
            lineage=self.lineage(self.backend.version),
            scored_at=self.clock(),
            latency_ms=budget.elapsed_ms,
            stage_latency_ms=stages,
            degraded=tuple(str(c) for c in request.capabilities),
            skipped={"_reason": str(exc)},
        )
        self.monitor.request(
            profile=profile,
            language=str(language),
            latency_ms=response.latency_ms,
            stages=stages,
            degraded=response.degraded,
        )
        return response

    def _maybe_shadow(self, text: str, language: Language) -> None:
        every = self.config.shadow_every
        if not self.shadow_backend or every <= 0:
            return
        self._requests_seen += 1
        if self._requests_seen % every:
            return
        threading.Thread(target=self._run_shadow, args=(text, language), daemon=True).start()

    def _run_shadow(self, text: str, language: Language) -> None:
        try:
            started = time.perf_counter()
            scores = self.shadow_backend.predict(text, language)
            self.log.write(
                "shadow",
                {
                    "version": scores.version,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "sentiment": scores.top_sentiment()[0].value,
                    "toxicity": round(scores.toxicity, 4),
                },
            )
        except Exception as exc:
            self.monitor.error("shadow")
            self.log.write("shadow_error", {"error": str(exc)})

    def _record(
        self, request: ScoreRequest, response: ScoreResponse, item: preprocess.Item
    ) -> None:
        self.monitor.request(
            profile=str(request.profile),
            language=str(item.language),
            latency_ms=response.latency_ms,
            stages=response.stage_latency_ms,
            degraded=response.degraded,
        )
        payload = response.to_dict()
        payload["content_hash"] = item.content_hash
        flagged = (response.toxicity and response.toxicity.toxic) or (
            response.misleading and response.misleading.likely_misleading
        )
        if flagged or self.config.retain_text_for_training:
            # Every flag carries its text, on any profile.
            payload["text"] = item.text
            payload["source_url"] = request.source_url
        self.log.write("score", payload)

    def _log_transition(self, record, transition) -> None:
        self.log.write(
            "provenance_transition",
            {"claim_id": record.claim_id, "item_id": record.item_id, **transition.to_dict()},
        )

    @staticmethod
    @contextmanager
    def _timed(stages: dict[str, float], name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            stages[name] = (time.perf_counter() - started) * 1000.0
