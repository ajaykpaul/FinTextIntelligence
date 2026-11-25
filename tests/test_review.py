"""The weekly human review: sampling, verdicts, precision (PRD 5.3, 8.4)."""

from __future__ import annotations

import json

import pytest

from fintext.review import (
    ENGLISH_PRECISION_GATE,
    MIN_BATCH_SIZE,
    Alert,
    Batch,
    Decision,
    ReviewError,
    ReviewStore,
    build_report,
    read_alerts,
    sample_batch,
    to_csv,
)
from fintext.schemas import Profile, ScoreRequest

T0 = 1_700_000_000.0

FLAGGED = [
    "You are a pathetic clown and I will k1ll you.",
    "Guaranteed 300% returns. Act now, send USDT.",
    "kill yourself",
    "Official verified account of FAB: share your OTP to verify.",
]


def make_alert(alert_id="a1", *, language="en", kind="toxicity", lineage=None) -> Alert:
    return Alert(
        alert_id=alert_id,
        item_id="i1",
        at=T0,
        language=language,
        profile="trading",
        kind=kind,
        score=0.9,
        threshold=0.35,
        text="i will kill you",
        summary="lexicon floor",
        lineage=lineage or {"model_version": "m1", "threshold_version": "t1"},
    )


def decisions(
    correct: int, incorrect: int = 0, unsure: int = 0, *, language: str = "en"
) -> list[Decision]:
    out = []
    verdicts = ["correct"] * correct + ["incorrect"] * incorrect + ["unsure"] * unsure
    for index, verdict in enumerate(verdicts):
        out.append(
            Decision(
                f"a{index}",
                verdict,
                "ana",
                T0 + index,
                language,
                "toxicity",
                "m1|t1",
                language == "ar",
            )
        )
    return out


class TestReadingTheLog:
    def test_reads_flagged_items_only(self, tmp_path, config, index):
        from fintext.pipeline import Pipeline

        pipeline = Pipeline(config, index=index)
        for text in [*FLAGGED, "The board approved a dividend of 0.35 per share"]:
            pipeline.score(ScoreRequest(text=text, profile=Profile.TRADING))
        pipeline.log.flush()
        pipeline.close()

        alerts = list(read_alerts(config.decision_log))
        assert len(alerts) == len(FLAGGED)
        assert all(alert.text for alert in alerts)

    def test_one_item_can_raise_two_alerts(self, tmp_path):
        # A text that is both toxic and misleading is two separate judgements.
        log = tmp_path / "log.jsonl"
        log.write_text(
            json.dumps(
                {
                    "kind": "score",
                    "id": "i1",
                    "logged_at": T0,
                    "language": "en",
                    "profile": "trading",
                    "content_hash": "h1",
                    "text": "abuse and a scam",
                    "toxicity": {"toxic": True, "score": 0.9, "threshold": 0.35, "explanation": {}},
                    "misleading": {
                        "likely_misleading": True,
                        "score": 0.8,
                        "threshold": 0.5,
                        "explanation": {},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        kinds = {alert.kind for alert in read_alerts(log)}
        assert kinds == {"toxicity", "misleading"}

    def test_survives_a_truncated_tail(self, tmp_path):
        # The log is written by a live service; the last line may be partial.
        log = tmp_path / "log.jsonl"
        log.write_text('{"kind": "score", "id": "a", "text": "x", "toxi', encoding="utf-8")
        assert list(read_alerts(log)) == []

    def test_missing_log_is_empty(self, tmp_path):
        assert list(read_alerts(tmp_path / "absent.jsonl")) == []


class TestSampling:
    def write_log(self, path, count: int) -> None:
        lines = [
            json.dumps(
                {
                    "kind": "score",
                    "id": f"i{n}",
                    "logged_at": T0 + n,
                    "language": "en",
                    "profile": "trading",
                    "content_hash": f"h{n}",
                    "text": f"abusive item {n}",
                    "toxicity": {"toxic": True, "score": 0.9, "threshold": 0.35, "explanation": {}},
                }
            )
            for n in range(count)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_draws_the_requested_size(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 200)
        batch = sample_batch(log, window_start=T0, window_end=T0 + 1000, size=50)
        assert len(batch.alerts) == 50 and not batch.short

    def test_is_reproducible(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 200)
        first = sample_batch(log, window_start=T0, window_end=T0 + 1000, size=20)
        second = sample_batch(log, window_start=T0, window_end=T0 + 1000, size=20)
        assert [a.alert_id for a in first.alerts] == [a.alert_id for a in second.alerts]

    def test_a_different_seed_draws_differently(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 200)
        first = sample_batch(log, window_start=T0, window_end=T0 + 1000, size=20, seed=1)
        second = sample_batch(log, window_start=T0, window_end=T0 + 1000, size=20, seed=2)
        assert [a.alert_id for a in first.alerts] != [a.alert_id for a in second.alerts]

    def test_respects_the_window(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 100)
        batch = sample_batch(log, window_start=T0 + 10, window_end=T0 + 20, size=50)
        assert all(T0 + 10 <= alert.at < T0 + 20 for alert in batch.alerts)
        assert len(batch.alerts) == 10

    def test_collapses_duplicate_text(self, tmp_path):
        # The same text twice is not two pieces of evidence.
        log = tmp_path / "log.jsonl"
        record = {
            "kind": "score",
            "id": "i1",
            "logged_at": T0,
            "language": "en",
            "profile": "trading",
            "content_hash": "same",
            "text": "identical",
            "toxicity": {"toxic": True, "score": 0.9, "threshold": 0.35, "explanation": {}},
        }
        log.write_text("\n".join(json.dumps(record) for _ in range(5)), encoding="utf-8")
        assert len(sample_batch(log, window_start=T0 - 1, window_end=T0 + 1).alerts) == 1

    def test_a_short_batch_is_flagged_not_hidden(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 5)
        batch = sample_batch(log, window_start=T0, window_end=T0 + 1000)
        assert batch.short and batch.to_dict()["minimum"] == MIN_BATCH_SIZE

    def test_can_exclude_already_reviewed_alerts(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 20)
        first = sample_batch(log, window_start=T0, window_end=T0 + 1000, size=5)
        second = sample_batch(
            log,
            window_start=T0,
            window_end=T0 + 1000,
            size=5,
            exclude=[a.alert_id for a in first.alerts],
        )
        assert not {a.alert_id for a in first.alerts} & {a.alert_id for a in second.alerts}

    def test_rejects_an_inverted_window(self, tmp_path):
        with pytest.raises(ReviewError, match="after"):
            sample_batch(tmp_path / "log.jsonl", window_start=T0, window_end=T0 - 1)

    def test_round_trips_through_json(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self.write_log(log, 10)
        batch = sample_batch(log, window_start=T0, window_end=T0 + 1000)
        path = batch.save(tmp_path / "batch.json")
        reloaded = Batch.load(path)
        assert reloaded.batch_id == batch.batch_id
        assert [a.alert_id for a in reloaded.alerts] == [a.alert_id for a in batch.alerts]


class TestDecisions:
    def test_records_and_reads_back(self, tmp_path):
        store = ReviewStore(tmp_path / "decisions.jsonl")
        store.record(make_alert(), verdict="correct", reviewer="ana", note="clear threat")
        assert [d.verdict for d in store.all()] == ["correct"]

    def test_a_changed_verdict_keeps_the_history(self, tmp_path):
        store = ReviewStore(tmp_path / "decisions.jsonl")
        alert = make_alert()
        store.record(alert, verdict="correct", reviewer="ana", at=T0)
        store.record(alert, verdict="incorrect", reviewer="ana", at=T0 + 60)
        assert len(store.all()) == 2
        assert store.latest()[alert.alert_id].verdict == "incorrect"

    def test_rejects_an_unknown_verdict(self, tmp_path):
        with pytest.raises(ReviewError, match="verdict must be"):
            ReviewStore(tmp_path / "d.jsonl").record(make_alert(), verdict="maybe", reviewer="ana")

    def test_arabic_decisions_are_marked_machine_translated(self, tmp_path):
        store = ReviewStore(tmp_path / "decisions.jsonl")
        decision = store.record(make_alert(language="ar"), verdict="correct", reviewer="ana")
        assert decision.mt_assisted

    def test_the_decision_carries_the_lineage_that_produced_the_flag(self, tmp_path):
        store = ReviewStore(tmp_path / "decisions.jsonl")
        decision = store.record(make_alert(), verdict="correct", reviewer="ana")
        assert "m1" in decision.lineage_key and "t1" in decision.lineage_key


class TestPrecision:
    def test_precision_is_correct_over_judged(self):
        report = build_report(decisions(correct=9, incorrect=1))
        assert report.by_language["en"].precision == pytest.approx(0.9)

    def test_unsure_is_excluded_from_the_denominator(self):
        # Forcing a binary would corrupt the metric; ambiguity is its own signal.
        report = build_report(decisions(correct=9, incorrect=1, unsure=10))
        english = report.by_language["en"]
        assert english.judged == 10 and english.unsure == 10
        assert english.precision == pytest.approx(0.9)

    def test_the_english_gate_is_enforced(self):
        assert build_report(decisions(correct=9, incorrect=1)).passes
        assert not build_report(decisions(correct=8, incorrect=2)).passes
        assert ENGLISH_PRECISION_GATE == 0.90

    def test_arabic_is_reported_not_gated(self):
        report = build_report(decisions(correct=1, incorrect=9, language="ar"))
        arabic = report.by_language["ar"]
        assert not arabic.gated and arabic.passes is None
        assert "machine translation" in arabic.to_dict()["qualification"]

    def test_arabic_alone_cannot_pass_the_gate(self):
        # The gate is on English; an all-Arabic batch has not satisfied it.
        assert not build_report(decisions(correct=10, language="ar")).passes

    def test_no_judgements_means_the_gate_is_unmet(self):
        # "Nobody reviewed anything" is not a pass.
        assert not build_report([]).passes

    def test_a_batch_spanning_two_releases_is_called_out(self):
        mixed = [
            Decision("a1", "correct", "ana", T0, "en", "toxicity", "m1|t1", False),
            Decision("a2", "correct", "ana", T0, "en", "toxicity", "m2|t2", False),
        ]
        assert build_report(mixed).spans_multiple_lineages

    def test_report_feeds_the_acceptance_harness(self):
        """The gate stops being self-certified once this number is measured."""
        from fintext.evaluation import evaluate_acceptance

        measured = build_report(decisions(correct=9, incorrect=1)).as_measured()
        assert measured == {"human_review_precision_en": 0.9}

        result = next(
            r
            for r in evaluate_acceptance(measured).results
            if r.name == "human_review_precision_en"
        )
        assert result.measured == 0.9 and result.passed


class TestExport:
    def test_csv_carries_the_item_and_the_verdict(self, tmp_path):
        """PRD 5.3: spreadsheet export is sufficient for the MVP."""
        alert = make_alert()
        batch = Batch("b1", T0, T0, T0 + 10, 1, (alert,))
        store = ReviewStore(tmp_path / "d.jsonl")
        store.record(alert, verdict="incorrect", reviewer="ana", note="figurative")

        path = to_csv(batch, store.latest(), tmp_path / "review.csv")
        content = path.read_text(encoding="utf-8")
        assert "i will kill you" in content
        assert "incorrect" in content and "figurative" in content
        assert "lineage" in content.splitlines()[0]


class TestOverHttp:
    def flag_some(self, app):
        for text in FLAGGED:
            app.handle("POST", "/v1/score", json.dumps({"text": text, "profile": "trading"}))
        app.pipeline.log.flush()

    def test_full_loop(self, app):
        self.flag_some(app)

        status, batch = app.handle("POST", "/v1/review/batch", json.dumps({"days": 7}))
        assert status == 200 and batch["size"] == len(FLAGGED)

        for alert, verdict in zip(
            batch["alerts"], ["correct", "correct", "correct", "incorrect"], strict=True
        ):
            status, _ = app.handle(
                "POST",
                "/v1/review/decision",
                json.dumps({"alert_id": alert["alert_id"], "verdict": verdict, "reviewer": "ana"}),
            )
            assert status == 200

        status, report = app.handle("GET", "/v1/review/precision")
        assert status == 200
        assert report["by_language"]["en"]["precision"] == 0.75
        assert report["passes"] is False
        assert report["measured_metrics"] == {"human_review_precision_en": 0.75}

    def test_outstanding_count_tracks_progress(self, app):
        self.flag_some(app)
        _, batch = app.handle("POST", "/v1/review/batch", "{}")
        app.handle(
            "POST",
            "/v1/review/decision",
            json.dumps(
                {
                    "alert_id": batch["alerts"][0]["alert_id"],
                    "verdict": "correct",
                    "reviewer": "ana",
                }
            ),
        )
        _, current = app.handle("GET", "/v1/review/batch")
        assert current["outstanding"] == len(FLAGGED) - 1

    def test_no_batch_yet_is_404_not_an_error(self, app):
        status, body = app.handle("GET", "/v1/review/batch")
        assert status == 404 and "POST /v1/review/batch" in body["detail"]

    def test_a_verdict_needs_a_batch(self, app):
        status, _ = app.handle(
            "POST", "/v1/review/decision", json.dumps({"alert_id": "nope", "verdict": "correct"})
        )
        assert status == 400

    def test_an_unknown_alert_is_rejected(self, app):
        self.flag_some(app)
        app.handle("POST", "/v1/review/batch", "{}")
        status, _ = app.handle(
            "POST", "/v1/review/decision", json.dumps({"alert_id": "0" * 16, "verdict": "correct"})
        )
        assert status == 404

    def test_an_unknown_verdict_is_rejected(self, app):
        self.flag_some(app)
        _, batch = app.handle("POST", "/v1/review/batch", "{}")
        status, body = app.handle(
            "POST",
            "/v1/review/decision",
            json.dumps({"alert_id": batch["alerts"][0]["alert_id"], "verdict": "block it"}),
        )
        assert status == 400 and "verdict must be" in body["detail"]

    def test_export_needs_a_batch(self, app, tmp_path):
        from fintext.schemas import ValidationError

        with pytest.raises(ValidationError, match="no review batch"):
            app.export_review_csv(tmp_path / "x.csv")
