"""The service applies registered sentiment offsets exactly as evaluation did."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from fintext import sentiment_offsets as so  # noqa: E402
from fintext.registry import Entry, Registry  # noqa: E402
from fintext.schemas import Language  # noqa: E402
from fintext.scoring import Onnx  # noqa: E402

LOGITS = [1.0, 0.8, 0.2]  # positive, negative, neutral: positive wins untouched
OFFSETS = {"ar": {"positive": -0.6, "negative": 0.0, "neutral": 1.2}}


class _Session:
    def run(self, outputs, feeds):
        heads = {
            "sentiment": np.array([LOGITS]),
            "toxicity": np.array([[0.0]]),
            "toxicity_subtypes": np.zeros((1, 4)),
            "misleading": np.array([[0.0]]),
        }
        return [heads[name] for name in outputs]


def _backend(offsets):
    backend = Onnx.__new__(Onnx)  # skip loading a real artefact
    backend.np = np
    backend.session = _Session()
    backend.tokenizer = lambda text, **_: {"input_ids": np.array([[1, 2, 3]])}
    backend.version = "test"
    backend.max_length = 32
    backend.inputs = {"input_ids"}
    backend.outputs = ["sentiment", "toxicity", "toxicity_subtypes", "misleading"]
    order = ("positive", "negative", "neutral")
    backend.sentiment_offsets = {
        lang: np.array([o.get(k, 0.0) for k in order]) for lang, o in (offsets or {}).items()
    }
    return backend


def test_offsets_change_only_their_language():
    backend = _backend(OFFSETS)
    arabic = backend.predict("نص", Language.AR)
    english = backend.predict("text", Language.EN)
    assert arabic.top_sentiment()[0].value == "neutral"
    assert english.top_sentiment()[0].value == "positive"
    assert math.isclose(sum(arabic.sentiment.values()), 1.0, rel_tol=1e-9)


def test_serving_decides_as_evaluation_does():
    backend = _backend(OFFSETS)
    probs = np.exp(LOGITS) / np.exp(LOGITS).sum()
    offline = so.LABELS[int(so.decide([probs], [OFFSETS["ar"][k] for k in so.LABELS])[0])]
    assert backend.predict("نص", Language.AR).top_sentiment()[0].value == offline


def test_no_offsets_is_the_old_behaviour():
    assert _backend(None).predict("نص", Language.AR).top_sentiment()[0].value == "positive"


def test_registry_keeps_offsets_and_old_entries_load(tmp_path):
    registry = Registry(tmp_path)
    registry.register(
        Entry(
            version="v",
            artifact_path="x.onnx",
            artifact_digest="sha256:0",
            created_at=0.0,
            sentiment_offsets=OFFSETS,
        ),
        verify_digest=False,
    )
    assert Registry(tmp_path).resolve("v").sentiment_offsets == OFFSETS
    old = Entry(version="w", artifact_path="x.onnx", artifact_digest="sha256:0", created_at=0.0)
    assert old.sentiment_offsets == {}
