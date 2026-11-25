"""Claim extraction, source matching and the state machine (PRD 4.4, 6.3, 6.4)."""

from __future__ import annotations

import pytest
from conftest import T0, WINDOWS, Clock, make_document

from fintext.provenance import (
    ClaimExtractor,
    Document,
    IllegalTransition,
    PrimaryIndex,
    Tracker,
    entity_key,
    score_match,
)
from fintext.schemas import ClaimType, ProvenanceState


@pytest.fixture
def extractor() -> ClaimExtractor:
    return ClaimExtractor()


def make_tracker(index=None, clock=None, **kwargs) -> Tracker:
    return Tracker(index or PrimaryIndex(), windows=WINDOWS, clock=clock or Clock(), **kwargs)


class TestClaimExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("First Abu Dhabi Bank PJSC agreed to acquire a lender", ClaimType.MERGER),
            ("FAB raises its full-year guidance", ClaimType.GUIDANCE),
            ("SEC charges Acme Corp; trading halted", ClaimType.REGULATORY),
            ("Aldar wins a $2bn contract with the government", ClaimType.CONTRACT),
        ],
    )
    def test_identifies_claim_types(self, extractor, text, expected):
        assert extractor.extract(text).claim_type is expected

    def test_no_claim_in_ordinary_text(self, extractor):
        assert extractor.extract("Analysts expect a quiet session on Tuesday") is None

    @pytest.mark.parametrize(
        "text", ["Acme Corp charges customers a monthly fee", "The fund charges 1.5% in fees"]
    )
    def test_fee_language_is_not_a_regulatory_claim(self, extractor, text):
        assert extractor.extract(text) is None

    def test_resolves_a_known_alias(self, extractor):
        assert extractor.extract("FAB raises its guidance").entity_key == "first abu dhabi bank"

    def test_entity_key_ignores_legal_suffixes(self):
        # A genuine filing must corroborate its own headline.
        assert entity_key("First Abu Dhabi Bank PJSC") == entity_key("FIRST ABU DHABI BANK")

    def test_unresolvable_entity_gives_low_confidence(self, extractor):
        # A wrong entity risks corroborating against someone else's filing.
        claim = extractor.extract("An unnamed lender agreed to acquire a rival")
        assert claim.confidence < 0.5 and claim.entity_key == ""

    def test_regulatory_claims_are_checked_first(self, extractor):
        # Shortest disclosure clock first.
        claim = extractor.extract(
            "SEC charges Acme Corp after it agreed to acquire a rival and raised guidance"
        )
        assert claim.claim_type is ClaimType.REGULATORY

    def test_span_locates_the_claim(self, extractor):
        text = "First Abu Dhabi Bank PJSC agreed to acquire a lender"
        claim = extractor.extract(text)
        assert text[claim.span.start : claim.span.end] == claim.span.text


class TestSourceMatching:
    @pytest.fixture
    def claim(self, extractor):
        return extractor.extract("First Abu Dhabi Bank PJSC agreed to acquire a lender")

    def test_matches_a_primary_document(self, claim):
        score, _ = score_match(claim, make_document(), observed_at=T0, window=7200)
        assert score >= 0.82

    def test_rejects_a_secondary_source(self, claim):
        # A fabricated headline is surrounded by copies of itself.
        score, reason = score_match(
            claim, make_document(source_type="news_wire"), observed_at=T0, window=7200
        )
        assert score == 0.0 and "not a primary source" in reason

    def test_rejects_a_different_entity(self, claim):
        score, _ = score_match(
            claim, make_document(entity="Abu Dhabi Commercial Bank"), observed_at=T0, window=7200
        )
        assert score == 0.0

    def test_rejects_the_wrong_kind_of_event(self, claim):
        score, reason = score_match(
            claim,
            make_document(title="Annual sustainability report", body="sustainability disclosures"),
            observed_at=T0,
            window=7200,
        )
        assert score == 0.0 and "vocabulary" in reason

    def test_rejects_a_document_outside_the_evidence_horizon(self, claim):
        score, _ = score_match(
            claim, make_document(published_at=T0 - 30 * 24 * 3600), observed_at=T0, window=7200
        )
        assert score == 0.0

    def test_matches_on_ticker(self, extractor):
        claim = extractor.extract("$ACME agreed to acquire a rival")
        # The filer's legal name shares nothing with the ticker.
        score, reason = score_match(
            claim,
            make_document(entity="Pinnacle Industrial Holdings", tickers=("ACME",)),
            observed_at=T0,
            window=7200,
        )
        assert score > 0 and "ticker match" in reason


class TestStateMachine:
    def test_no_claim_is_not_applicable(self):
        result = make_tracker().check(None, item_id="i1")
        assert result.state is ProvenanceState.NOT_APPLICABLE and result.window_seconds is None

    def test_a_new_claim_starts_pending(self, extractor):
        result = make_tracker().check(
            extractor.extract("FAB raises its full-year guidance"), item_id="i1", observed_at=T0
        )
        assert result.state is ProvenanceState.PENDING

    def test_window_depends_on_the_claim_type(self, extractor):
        # Disclosure clocks differ, so windows are per claim type.
        tracker = make_tracker()
        regulatory = tracker.check(
            extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0
        )
        merger = tracker.check(
            extractor.extract("Acme Corp agreed to acquire a rival"), item_id="i2", observed_at=T0
        )
        assert regulatory.window_seconds < merger.window_seconds

    def test_an_arriving_document_corroborates(self, extractor):
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(
            extractor.extract("First Abu Dhabi Bank PJSC agreed to acquire a lender"),
            item_id="i1",
            observed_at=T0,
        )
        clock.advance(180)
        transitions = tracker.on_documents([make_document(published_at=T0 + 120)], now=clock.now)
        assert [t.to_state for t in transitions] == [ProvenanceState.CORROBORATED]
        assert next(iter(tracker)).time_to_corroboration == 180

    def test_expiry_moves_to_uncorroborated(self, extractor):
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)
        clock.advance(WINDOWS["regulatory"] + 60)
        assert [t.to_state for t in tracker.recheck_due(now=clock.now)] == [
            ProvenanceState.UNCORROBORATED
        ]

    def test_pending_persists_inside_the_window(self, extractor):
        # Pending is the normal condition of breaking news.
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)
        clock.advance(300)
        assert tracker.recheck_due(now=clock.now) == []
        assert next(iter(tracker)).state is ProvenanceState.PENDING

    def test_a_late_document_still_corroborates(self, extractor):
        # The window is calibrated to normal confirmation times, not a deadline after which
        # evidence stops counting.
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)
        clock.advance(WINDOWS["regulatory"] + 60)
        tracker.recheck_due(now=clock.now)
        assert next(iter(tracker)).state is ProvenanceState.UNCORROBORATED

        clock.advance(600)
        tracker.on_documents(
            [
                make_document(
                    entity="Acme Corp",
                    published_at=clock.now - 60,
                    title="Administrative proceeding order",
                    body="enforcement order administrative proceeding charges",
                )
            ],
            now=clock.now,
        )
        assert next(iter(tracker)).state is ProvenanceState.CORROBORATED

    def test_corroborated_is_terminal(self, extractor):
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(
            extractor.extract("First Abu Dhabi Bank PJSC agreed to acquire a lender"),
            item_id="i1",
            observed_at=T0,
        )
        tracker.on_documents([make_document()], now=T0 + 10)
        clock.advance(WINDOWS["merger"] * 2)
        tracker.recheck_due(now=clock.now)
        assert next(iter(tracker)).state is ProvenanceState.CORROBORATED

    def test_illegal_transitions_are_refused(self, extractor):
        tracker = make_tracker()
        tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)
        record = next(iter(tracker))
        record.move_to(ProvenanceState.CORROBORATED, at=T0, reason="matched")
        with pytest.raises(IllegalTransition):
            record.move_to(ProvenanceState.PENDING, at=T0, reason="reopen")

    def test_a_weak_match_leaves_the_claim_pending(self, extractor):
        # Precision on corroborated takes priority over recall.
        tracker = make_tracker(min_match_score=0.99)
        tracker.check(
            extractor.extract("First Abu Dhabi Bank PJSC agreed to acquire a lender"),
            item_id="i1",
            observed_at=T0,
        )
        # Right entity, right event, but only one keyword of support.
        tracker.on_documents(
            [make_document(title="Notice to shareholders", body="stake")], now=T0 + 10
        )
        record = next(iter(tracker))
        assert record.state is ProvenanceState.PENDING and "below the" in record.reason

    def test_an_ambiguous_entity_never_resolves_either_way(self, extractor):
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(
            extractor.extract("An unnamed lender agreed to acquire a rival"),
            item_id="i1",
            observed_at=T0,
        )
        clock.advance(WINDOWS["merger"] * 2)
        tracker.recheck_due(now=clock.now)
        record = next(iter(tracker))
        # Neither corroborated (a false positive) nor uncorroborated (manufacturing the signal-
        # bearing state out of an ambiguity).
        assert record.state is ProvenanceState.PENDING and "ambiguous" in record.reason

    def test_a_stale_index_degrades_to_pending(self, extractor):
        # An ingestion outage must not manufacture the state that carries signal.
        clock = Clock()
        tracker = make_tracker(
            index=PrimaryIndex([make_document(published_at=T0 - 10_000)]),
            clock=clock,
            stale_after=60,
        )
        tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)
        clock.advance(WINDOWS["regulatory"] * 3)
        tracker.recheck_due(now=clock.now)
        record = next(iter(tracker))
        assert record.state is ProvenanceState.PENDING and "stale" in record.reason

    def test_transitions_are_timestamped_with_evidence(self, extractor):
        tracker = make_tracker()
        tracker.check(
            extractor.extract("First Abu Dhabi Bank PJSC agreed to acquire a lender"),
            item_id="i1",
            observed_at=T0,
        )
        tracker.on_documents([make_document()], now=T0 + 42)
        transition = next(iter(tracker)).transitions[0]
        assert transition.at == T0 + 42 and transition.evidence

    def test_review_queue_holds_only_expired_items(self, extractor):
        # Pending items wait; expired-window items are findings.
        clock = Clock()
        tracker = make_tracker(clock=clock)
        tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)
        tracker.check(extractor.extract("Aldar wins a $2bn contract"), item_id="i2", observed_at=T0)
        clock.advance(WINDOWS["regulatory"] + 60)
        tracker.recheck_due(now=clock.now)
        assert len(tracker.review_queue()) == 1 and len(tracker.open_claims()) == 1

    def test_unconfigured_claim_type_raises(self, extractor):
        # Windows come from observed confirmation times, never a default.
        tracker = Tracker(PrimaryIndex(), windows={"merger": 100})
        with pytest.raises(ValueError, match="no corroboration window"):
            tracker.check(extractor.extract("SEC charges Acme Corp"), item_id="i1", observed_at=T0)

    def test_store_is_bounded(self, extractor):
        tracker = make_tracker(capacity=5)
        for index in range(20):
            tracker.check(
                extractor.extract(f"Acme{index} Corp agreed to acquire a rival"),
                item_id=f"i{index}",
                observed_at=T0,
            )
        assert len(tracker) <= 5

    def test_statistics_report_time_to_corroboration(self, extractor):
        tracker = make_tracker()
        tracker.check(
            extractor.extract("First Abu Dhabi Bank PJSC agreed to acquire a lender"),
            item_id="i1",
            observed_at=T0,
        )
        tracker.on_documents([make_document()], now=T0 + 120)
        stats = tracker.statistics()
        assert stats["median_time_to_corroboration_seconds"]["merger"] == 120
        assert stats["counts"]["corroborated"] == 1


class TestIndex:
    def test_reads_jsonl(self, tmp_path):
        path = tmp_path / "index.jsonl"
        path.write_text(
            '{"doc_id":"d1","source_type":"edgar","entity":"Acme Corp"}\n', encoding="utf-8"
        )
        index = PrimaryIndex.from_jsonl(path)
        assert len(index) == 1 and next(iter(index)).is_primary

    def test_missing_file_is_an_empty_index(self, tmp_path):
        assert len(PrimaryIndex.from_jsonl(tmp_path / "absent.jsonl")) == 0

    def test_freshness_uses_stream_time_not_wall_time(self):
        index = PrimaryIndex([Document("d", "edgar", "Acme Corp", published_at=T0)])
        assert index.last_updated == T0

    def test_polling_marks_the_index_fresh(self):
        index = PrimaryIndex()
        index.mark_polled(T0)
        assert index.last_updated == T0
