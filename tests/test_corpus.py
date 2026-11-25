"""Turning captured production items into a fine-tuning corpus."""

from __future__ import annotations

import pytest

from fintext import catalogue, corpus
from fintext.approvals import Resolution
from fintext.review import Decision


def record(item_id="i1", text="Profit rose sharply", language="en", **kwargs):
    return {"id": item_id, "text": text, "language": language, "scored_at": 1.0, **kwargs}


def approval(item_id="i1", action="accept", direction="buy"):
    return Resolution(item_id=item_id, action=action, direction=direction, reviewer="ajay")


def verdict(alert_id="i1", verdict_value="correct"):
    return Decision(
        alert_id=alert_id,
        verdict=verdict_value,
        reviewer="ajay",
        at=1.0,
        language="en",
        kind="toxicity",
        lineage_key="lex.v1",
        mt_assisted=False,
    )


class TestWhatCountsAsALabel:
    def test_a_model_output_is_not_a_label(self):
        [example] = corpus.build_corpus([record(sentiment={"label": "positive"})])
        assert example.kind == corpus.UNLABELLED
        assert example.labels == {}

    def test_an_approval_is_a_label(self):
        [example] = corpus.build_corpus([record()], resolutions=[approval()])
        assert example.kind == corpus.GOLD
        assert example.labels == {"tradeable": True, "direction": "buy"}
        assert example.label_source == "approval"

    def test_a_rejection_is_a_label_too(self):
        [example] = corpus.build_corpus(
            [record()], resolutions=[approval(action="reject", direction="")]
        )
        assert example.labels == {"tradeable": False}

    def test_a_weekly_verdict_labels_the_flag(self):
        [example] = corpus.build_corpus([record()], review_decisions=[verdict()])
        assert example.labels == {"flag_correct": True}
        assert example.label_source == "weekly_review"

    def test_unsure_produces_nothing(self):
        [example] = corpus.build_corpus([record()], review_decisions=[verdict("i1", "unsure")])
        assert example.kind == corpus.UNLABELLED

    def test_both_loops_can_label_the_same_item(self):
        [example] = corpus.build_corpus(
            [record()], review_decisions=[verdict()], resolutions=[approval()]
        )
        assert example.labels == {"tradeable": True, "direction": "buy", "flag_correct": True}
        assert example.label_source == "approval+weekly_review"


class TestCapture:
    def test_an_item_with_no_retained_text_is_skipped(self):
        assert corpus.build_corpus([{"id": "i1", "language": "en"}]) == []

    def test_the_split_is_stable_across_rebuilds(self):
        first = corpus.split_for("Profit rose sharply")
        assert all(corpus.split_for("Profit rose sharply") == first for _ in range(5))

    def test_the_split_holds_back_roughly_the_share_asked_for(self):
        texts = [f"item number {n} about the market" for n in range(400)]
        held = sum(1 for t in texts if corpus.split_for(t) == corpus.HELD_OUT)
        assert 0.12 < held / len(texts) < 0.28


class TestTheLicenceGate:
    def test_writing_is_blocked_until_the_licence_clears(self, tmp_path):
        with pytest.raises(catalogue.LicenceError, match="production_capture"):
            corpus.write_corpus([], tmp_path / "corpus.jsonl")

    def test_the_source_is_declared_in_the_catalogue(self):
        entry = catalogue.get(corpus.PRODUCTION_SOURCE)
        assert entry.status == catalogue.PENDING
        assert not entry.redistributable

    def test_writing_works_once_it_clears(self, tmp_path, monkeypatch):
        """What clearing it looks like: one word in the catalogue, after a lawyer has said so."""
        import dataclasses

        entry = catalogue.get(corpus.PRODUCTION_SOURCE)
        monkeypatch.setitem(
            catalogue.BY_KEY,
            corpus.PRODUCTION_SOURCE,
            dataclasses.replace(entry, status=catalogue.CLEARED),
        )
        examples = corpus.build_corpus([record()], resolutions=[approval()])
        path = corpus.write_corpus(examples, tmp_path / "corpus.jsonl")
        assert path.read_text(encoding="utf-8").count("\n") == 1


class TestSummary:
    def test_it_reports_what_is_actually_trainable(self):
        examples = corpus.build_corpus(
            [record("i1"), record("i2", text="Shares fell on the news today")],
            resolutions=[approval("i1")],
        )
        summary = corpus.summarise(examples)
        assert summary["total"] == 2
        assert summary["gold"] == 1 and summary["unlabelled"] == 1
        assert summary["trainable"] + summary["held_out"] == summary["gold"]
