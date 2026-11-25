"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fintext.config import load_config  # noqa: E402
from fintext.pipeline import Pipeline  # noqa: E402
from fintext.provenance import Document, PrimaryIndex  # noqa: E402
from fintext.server import App  # noqa: E402

T0 = 1_700_000_000.0
WINDOWS = {"regulatory": 900, "guidance": 3600, "merger": 7200, "contract": 10800}


class Clock:
    """Advanced by hand, so window expiry is tested without waiting for it."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def config(tmp_path):
    """Default config with writable paths redirected into the test's tmpdir."""
    return load_config().replace(
        decision_log=str(tmp_path / "decisions.jsonl"),
        index_path=str(tmp_path / "index.jsonl"),
        review_dir=str(tmp_path / "review"),
    )


@pytest.fixture
def index() -> PrimaryIndex:
    return PrimaryIndex()


@pytest.fixture
def pipeline(config, clock, index):
    built = Pipeline(config, index=index, clock=clock)
    yield built
    built.close()


@pytest.fixture
def app(pipeline) -> App:
    built = App(pipeline)
    built.startup()
    return built


def make_document(
    doc_id="edgar-1",
    *,
    entity="First Abu Dhabi Bank PJSC",
    source_type="edgar",
    published_at=T0,
    title="Definitive agreement for the acquisition of a stake",
    body="business combination definitive agreement item 1.01 8-K",
    tickers=(),
) -> Document:
    return Document(
        doc_id=doc_id,
        source_type=source_type,
        entity=entity,
        title=title,
        url=f"https://www.sec.gov/{doc_id}",
        published_at=published_at,
        body=body,
        tickers=tickers,
    )
