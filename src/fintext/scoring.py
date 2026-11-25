"""Model backends."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .preprocess import fold_for_model
from .schemas import Language, Sentiment, Subtype


class BackendError(Exception):
    pass


@dataclass(frozen=True)
class Scores:
    """Raw model output for one item, before thresholds and rules."""

    sentiment: dict[str, float]
    toxicity: float
    subtypes: dict[str, float]
    misleading: float
    version: str

    def top_sentiment(self) -> tuple[Sentiment, float]:
        label, score = max(self.sentiment.items(), key=lambda kv: kv[1])
        return Sentiment(label), score


# --------------------------------------------------------------------------
# Heuristic backend
# --------------------------------------------------------------------------

# Finance-specific sentiment.
UP = frozenset(
    [
        "rise",
        "rises",
        "rose",
        "rising",
        "gain",
        "gains",
        "gained",
        "increase",
        "increased",
        "increases",
        "climb",
        "climbed",
        "surge",
        "surged",
        "jump",
        "jumped",
        "up",
        "higher",
        "grew",
        "grow",
        "growth",
        "expand",
        "expanded",
        "beat",
        "beats",
        "exceeded",
        "topped",
        "improved",
        "improves",
        "strengthened",
        "advanced",
        "soared",
    ]
)
DOWN = frozenset(
    [
        "fall",
        "falls",
        "fell",
        "falling",
        "drop",
        "dropped",
        "drops",
        "decline",
        "declined",
        "declines",
        "decrease",
        "decreased",
        "slump",
        "slumped",
        "plunge",
        "plunged",
        "sank",
        "down",
        "lower",
        "shrank",
        "narrowed",
        "cut",
        "cuts",
        "reduced",
        "weakened",
        "missed",
        "miss",
        "misses",
        "slid",
        "tumbled",
        "eased",
    ]
)
GOOD_SUBJECT = frozenset(
    [
        "profit",
        "profits",
        "revenue",
        "revenues",
        "earnings",
        "margin",
        "margins",
        "income",
        "sales",
        "cash",
        "dividend",
        "dividends",
        "guidance",
        "orders",
        "backlog",
        "output",
        "productivity",
        "eps",
        "assets",
        "capital",
        "deposits",
        "shares",
        "stock",
        "price",
    ]
)
BAD_SUBJECT = frozenset(
    [
        "loss",
        "losses",
        "liabilities",
        "liability",
        "debt",
        "debts",
        "costs",
        "cost",
        "expenses",
        "impairment",
        "impairments",
        "provisions",
        "npl",
        "defaults",
        "arrears",
        "leverage",
        "risk",
        "exposure",
        "writedown",
        "overheads",
    ]
)
POSITIVE = frozenset(
    [
        "record",
        "strong",
        "robust",
        "outperform",
        "upgrade",
        "upgraded",
        "buyback",
        "approval",
        "approved",
        "wins",
        "won",
        "awarded",
        "profitable",
        "recovery",
        "resilient",
        "expansion",
    ]
)
NEGATIVE = frozenset(
    [
        "bankruptcy",
        "insolvency",
        "fraud",
        "probe",
        "investigation",
        "downgrade",
        "downgraded",
        "lawsuit",
        "fine",
        "fined",
        "penalty",
        "recall",
        "resign",
        "resigned",
        "halt",
        "halted",
        "suspension",
        "default",
        "restructuring",
        "warning",
        "warned",
        "delisting",
    ]
)
NEUTRAL_CUES = frozenset(
    [
        "said",
        "says",
        "report",
        "reports",
        "reported",
        "announced",
        "announces",
        "scheduled",
        "will",
        "plans",
        "meeting",
        "agenda",
        "publish",
        "published",
        "statement",
        "conference",
    ]
)
NEGATORS = frozenset({"not", "no", "never", "without", "fails", "failed", "unable"})

# Arabic cues are minimal and carry no accuracy claim: Arabic sentiment is reported, never gated
# (PRD 1.1). They keep the path exercised.
AR_POSITIVE = frozenset({"ارتفاع", "ارتفعت", "نمو", "أرباح", "مكاسب", "تحسن", "صعود"})
AR_NEGATIVE = frozenset({"انخفاض", "انخفضت", "خسائر", "تراجع", "هبوط", "أزمة", "عجز"})

THREAT = re.compile(
    r"\b(?:kill|shoot|stab|hunt (?:you|him|her|them) down|come for you|find you|hurt you"
    r"|destroy you)\b",
    re.I,
)
HARASSMENT = re.compile(
    r"\b(?:you(?:'re| are)? (?:a |an )?(?:idiot|moron|clown|joke|pathetic|worthless|stupid)"
    r"|shut up|nobody (?:likes|wants) you|get lost|loser)\b",
    re.I,
)
HATE = re.compile(
    r"\b(?:all (?:of )?(?:them|those people|these people) (?:are|should)"
    r"|go back to your country|your kind|deport (?:them|all))\b",
    re.I,
)
PROFANITY = re.compile(r"\b(?:damn|hell|crap|bloody|bastard|shit|fuck\w*)\b", re.I)
SECOND_PERSON = re.compile(r"\b(?:you|your|u|ur)\b", re.I)

SENSATIONAL = re.compile(
    r"(?:!{2,}|\?{2,}|\b(?:shocking|unbelievable|secret|exposed|breaking|urgent"
    r"|they don't want you to know|exclusive leak|insider)\b)",
    re.I,
)
ABSOLUTE = re.compile(r"\b(?:confirmed|100%|definitely|proven|guaranteed)\b", re.I)
ATTRIBUTED = re.compile(
    r"\b(?:according to|filed with|reuters|bloomberg|press release|sec filing|8-k|10-q|10-k"
    r"|exchange notice|regulatory filing)\b",
    re.I,
)

_WORD = re.compile(r"[\w']+", re.UNICODE)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class Heuristic:
    """Every score is a logistic or softmax transform of readable features, so a surprising output
    traces to a feature rather than to an opaque weight.
    """

    version = "shared-xlmr.v0-heuristic"

    def warmup(self) -> None:
        self.predict("warmup text for the shared encoder", Language.EN)

    def predict(self, text: str, language: Language) -> Scores:
        words = [w.lower() for w in _WORD.findall(text)]
        toxicity, subtypes = self._toxicity(text, words)
        return Scores(
            sentiment=(
                self._arabic_sentiment(words)
                if language is Language.AR
                else self._english_sentiment(words)
            ),
            toxicity=toxicity,
            subtypes=subtypes,
            misleading=self._misleading(text),
            version=self.version,
        )

    def _english_sentiment(self, words: list[str]) -> dict[str, float]:
        polarity = 0.0
        evidence = 0
        for index, word in enumerate(words):
            direction = 1 if word in UP else -1 if word in DOWN else 0
            if direction:
                subject = self._subject_near(words, index)
                # The finance-specific step: direction times subject.
                polarity += direction * (subject if subject else 0.5)
                evidence += 1
            if word in POSITIVE:
                polarity += 1.0
                evidence += 1
            elif word in NEGATIVE:
                polarity -= 1.2
                evidence += 1
            if word in NEGATORS and index + 1 < len(words):
                following = words[index + 1]
                if following in POSITIVE or following in UP:
                    polarity -= 1.0
                    evidence += 1

        neutral_cues = sum(1 for w in words if w in NEUTRAL_CUES)
        return self._distribution(polarity, evidence, neutral_cues, len(words))

    @staticmethod
    def _subject_near(words: list[str], index: int) -> int:
        window = words[max(0, index - 3) : index] + words[index + 1 : index + 4]
        for word in window:
            if word in GOOD_SUBJECT:
                return 1
            if word in BAD_SUBJECT:
                return -1
        return 0

    def _arabic_sentiment(self, words: list[str]) -> dict[str, float]:
        polarity = sum(
            1.0 if w in AR_POSITIVE else -1.0 if w in AR_NEGATIVE else 0.0 for w in words
        )
        evidence = sum(1 for w in words if w in AR_POSITIVE or w in AR_NEGATIVE)
        return self._distribution(polarity, evidence, 0, len(words))

    @staticmethod
    def _distribution(
        polarity: float, evidence: int, neutral_cues: int, length: int
    ) -> dict[str, float]:
        neutral_logit = 1.2 + 0.35 * neutral_cues - 0.55 * evidence + (0.4 if length > 40 else 0.0)
        positive, negative, neutral = softmax([1.6 * polarity, -1.6 * polarity, neutral_logit])
        return {
            Sentiment.POSITIVE.value: positive,
            Sentiment.NEGATIVE.value: negative,
            Sentiment.NEUTRAL.value: neutral,
        }

    @staticmethod
    def _toxicity(text: str, words: list[str]) -> tuple[float, dict[str, float]]:
        threat = len(THREAT.findall(text))
        harassment = len(HARASSMENT.findall(text))
        hate = len(HATE.findall(text))
        profanity = len(PROFANITY.findall(text))
        directed = 1.0 if SECOND_PERSON.search(text) else 0.0
        letters = [c for c in text if c.isalpha()]
        shouting = sum(c.isupper() for c in letters) / len(letters) if len(letters) > 8 else 0.0

        score = sigmoid(
            -3.1
            + 3.0 * min(threat, 2)
            + 2.2 * min(harassment, 2)
            + 2.6 * min(hate, 2)
            + 1.1 * min(profanity, 3)
            + 0.9 * directed * min(profanity + harassment, 2)
            + 1.4 * shouting
            - 0.25 * math.log1p(max(len(words), 1))
        )
        subtypes = {
            Subtype.THREAT.value: sigmoid(-3.4 + 3.4 * threat + 0.6 * directed),
            Subtype.HARASSMENT.value: sigmoid(
                -3.2 + 2.6 * harassment + 1.1 * directed * min(profanity, 2)
            ),
            Subtype.HATE.value: sigmoid(-3.6 + 3.2 * hate),
            Subtype.PROFANITY.value: sigmoid(-2.8 + 1.9 * min(profanity, 3)),
        }
        return score, subtypes

    @staticmethod
    def _misleading(text: str) -> float:
        return sigmoid(
            -2.4
            + 1.3 * min(len(SENSATIONAL.findall(text)), 3)
            + 1.1 * min(len(ABSOLUTE.findall(text)), 3)
            + 0.35 * min(text.count("!"), 4)
            - 1.6 * min(len(ATTRIBUTED.findall(text)), 2)
        )


# --------------------------------------------------------------------------
# ONNX backend
# --------------------------------------------------------------------------


class Onnx:
    """ONNX Runtime on CPU (PRD 6.6), int8 or fp32."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer: Any,
        *,
        version: str,
        max_length: int = 256,
        threads: int = 1,
        sentiment_offsets: dict[str, dict[str, float]] | None = None,
    ) -> None:
        try:
            import numpy
            import onnxruntime
        except ImportError as exc:
            raise BackendError(
                "the onnx backend needs onnxruntime and numpy: pip install '.[onnx]'"
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise BackendError(f"ONNX artefact not found: {path}")

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.np = numpy
        self.session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = tokenizer
        self.version = version
        self.max_length = max_length
        # In head order (positive, negative, neutral), per language.
        order = (Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL)
        self.sentiment_offsets = {
            language: numpy.array([float(offsets.get(s.value, 0.0)) for s in order])
            for language, offsets in (sentiment_offsets or {}).items()
        }
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]

    def warmup(self) -> None:
        """So the first real request does not pay graph-allocation cost."""
        self.predict("warmup " * 32, Language.EN)

    def predict(self, text: str, language: Language) -> Scores:
        # The pipeline hands over prepared text; the fold is the last step the model was trained
        # with (preprocess.model_text).
        encoded = self.tokenizer(
            fold_for_model(text),
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        feeds = {
            name: value.astype(self.np.int64)
            for name, value in encoded.items()
            if name in self.inputs
        }
        try:
            raw = dict(zip(self.outputs, self.session.run(self.outputs, feeds), strict=True))
        except Exception as exc:
            raise BackendError(f"ONNX inference failed: {exc}") from exc

        def head(name: str):
            if name not in raw:
                raise BackendError(f"artefact is missing the {name!r} head")
            return self.np.asarray(raw[name]).reshape(-1)

        sentiment = head("sentiment")
        offsets = self.sentiment_offsets.get(getattr(language, "value", str(language)))
        if offsets is not None:
            # Adding to the logits is adding to the log-probabilities; the softmax below
            # renormalises, so the confidence floor still reads a probability.
            sentiment = sentiment + offsets
        exps = self.np.exp(sentiment - sentiment.max())
        probabilities = exps / exps.sum()
        subtypes = 1.0 / (1.0 + self.np.exp(-head("toxicity_subtypes")))
        return Scores(
            sentiment={
                Sentiment.POSITIVE.value: float(probabilities[0]),
                Sentiment.NEGATIVE.value: float(probabilities[1]),
                Sentiment.NEUTRAL.value: float(probabilities[2]),
            },
            toxicity=float(sigmoid(float(head("toxicity")[0]))),
            subtypes={s.value: float(subtypes[i]) for i, s in enumerate(Subtype)},
            misleading=float(sigmoid(float(head("misleading")[0]))),
            version=self.version,
        )


def load_backend(config: Config, registry=None):
    """Build the configured backend."""
    if config.backend == "heuristic":
        return Heuristic()
    if registry is None:
        raise BackendError(f"backend {config.backend!r} needs a model registry")
    entry = registry.resolve(config.model_version)
    return Onnx(
        entry.artifact_path,
        entry.load_tokenizer(),
        version=entry.version,
        max_length=config.max_tokens,
        threads=config.onnx_threads,
        sentiment_offsets=entry.sentiment_offsets,
    )
