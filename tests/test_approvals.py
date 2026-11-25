"""The queue a gated call waits in until someone accepts or rejects it."""

from __future__ import annotations

import json

import pytest

from fintext.approvals import ApprovalError, ApprovalQueue, Resolution, Waiting


def waiting(item_id="a1", priority="now", **kwargs):
    return Waiting(
        item_id=item_id,
        text=kwargs.pop("text", "Acme may be exploring a merger, sources say"),
        language="en",
        priority=priority,
        reason=kwargs.pop("reason", "uncorroborated"),
        **kwargs,
    )


class TestResolving:
    def test_accepting_names_a_side(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting())
        with pytest.raises(ApprovalError, match="accepting requires a direction"):
            queue.resolve("a1", action="accept", reviewer="ajay")

    def test_rejecting_does_not(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting())
        resolution = queue.resolve("a1", action="reject", reviewer="ajay")
        assert resolution.outcome == "no trade"

    def test_a_rejection_cannot_carry_a_direction(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting())
        with pytest.raises(ApprovalError, match="does not carry a direction"):
            queue.resolve("a1", action="reject", direction="buy", reviewer="ajay")

    def test_every_decision_has_a_name_on_it(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting())
        with pytest.raises(ApprovalError, match="who made it"):
            queue.resolve("a1", action="accept", direction="buy", reviewer="   ")

    def test_an_item_is_resolved_once(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting())
        queue.resolve("a1", action="accept", direction="buy", reviewer="ajay")
        with pytest.raises(ApprovalError, match="already been resolved"):
            queue.resolve("a1", action="reject", reviewer="sam")

    def test_the_reviewer_may_go_against_the_lean(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting(lean="buy", conviction=0.55))
        assert queue.resolve("a1", action="accept", direction="sell", reviewer="ajay").outcome == (
            "sell"
        )


class TestOrdering:
    def test_urgent_comes_before_routine(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting("old-routine", priority="routine"))
        queue.add(waiting("new-urgent", priority="now"))
        assert [i.item_id for i in queue.waiting()] == ["new-urgent", "old-routine"]

    def test_within_a_priority_the_oldest_goes_first(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting("first", priority="now", queued_at=100.0))
        queue.add(waiting("second", priority="now", queued_at=200.0))
        assert [i.item_id for i in queue.waiting()] == ["first", "second"]

    def test_scoring_the_same_item_twice_does_not_queue_it_twice(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting(queued_at=100.0))
        queue.add(waiting(queued_at=500.0))
        assert len(queue.waiting()) == 1
        # The first wait is the one that counts, so the clock is not reset.
        assert queue.waiting()[0].queued_at == 100.0


class TestDurability:
    def test_the_queue_survives_a_restart(self, tmp_path):
        path = tmp_path / "q.jsonl"
        first = ApprovalQueue(path)
        first.add(waiting("a1"))
        first.add(waiting("a2", priority="routine"))
        first.resolve("a1", action="accept", direction="buy", reviewer="ajay")

        second = ApprovalQueue(path)
        assert [i.item_id for i in second.waiting()] == ["a2"]
        assert second.resolution("a1").outcome == "buy"

    def test_a_half_written_line_does_not_stop_it_starting(self, tmp_path):
        path = tmp_path / "q.jsonl"
        queue = ApprovalQueue(path)
        queue.add(waiting("a1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "waiting", "item_id": "trunc')
        assert [i.item_id for i in ApprovalQueue(path).waiting()] == ["a1"]


class TestTheNumbersADeskCaresAbout:
    def test_the_wait_is_recorded_on_every_resolution(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        queue.add(waiting(queued_at=0.0))
        assert queue.resolve("a1", action="reject", reviewer="ajay").waited_ms > 0

    def test_the_summary_counts_what_matters(self, tmp_path):
        queue = ApprovalQueue(tmp_path / "q.jsonl")
        for n in range(3):
            queue.add(waiting(f"a{n}", priority="now" if n == 0 else "routine"))
        queue.resolve("a0", action="accept", direction="buy", reviewer="ajay")
        queue.resolve("a1", action="reject", reviewer="ajay")
        summary = queue.summary()
        assert summary["waiting"] == 1
        assert summary["accepted"] == 1 and summary["rejected"] == 1
        assert summary["resolved"] == 2


class TestOverHttp:
    def test_a_gated_call_lands_in_the_queue(self, app):
        status, body = app.handle(
            "POST", "/v1/signal", json.dumps({"text": "Guaranteed 300% returns, DM me to join"})
        )
        assert status == 200
        assert body["signal"]["direction"] == "review"
        assert body["queued"]["item_id"] == body["score"]["id"]

    def test_a_decisive_call_does_not(self, app):
        status, body = app.handle(
            "POST",
            "/v1/signal",
            json.dumps({"text": "Profit rose 40% and the board raised full-year guidance"}),
        )
        assert body["signal"]["direction"] in ("buy", "sell")
        assert "queued" not in body

    def test_accept_and_reject_over_the_wire(self, app):
        _, scored = app.handle(
            "POST", "/v1/signal", json.dumps({"text": "Guaranteed 300% returns, DM me to join"})
        )
        item_id = scored["queued"]["item_id"]
        status, body = app.handle(
            "POST",
            "/v1/approvals/resolve",
            json.dumps(
                {"item_id": item_id, "action": "accept", "direction": "sell", "reviewer": "ajay"}
            ),
        )
        assert status == 200
        assert body["resolution"]["outcome"] == "sell"
        assert app.handle("GET", "/v1/approvals")[1]["waiting"] == 0

    def test_a_decision_that_makes_no_sense_is_a_409(self, app):
        status, body = app.handle(
            "POST",
            "/v1/approvals/resolve",
            json.dumps({"item_id": "nope", "action": "reject", "reviewer": "ajay"}),
        )
        assert status == 409
        assert body["error"] == "cannot_resolve"


class TestRecordShape:
    def test_the_outcome_is_what_the_desk_acts_on(self):
        accept = Resolution(item_id="a", action="accept", direction="buy", reviewer="ajay")
        reject = Resolution(item_id="a", action="reject", reviewer="ajay")
        assert (accept.outcome, reject.outcome) == ("buy", "no trade")
