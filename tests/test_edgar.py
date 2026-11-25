"""Building the provenance index from real SEC filings."""

from __future__ import annotations

import pytest

from fintext.edgar import SEC_ITEMS, _parse

HIT = {
    "_id": "0001193125-25-006663:d879011d8k.htm",
    "_source": {
        "ciks": ["0001556898"],
        "display_names": ["Techpoint, Inc.  (THPTF)  (CIK 0001556898)"],
        "adsh": "0001193125-25-006663",
        "form": "8-K",
        "file_date": "2025-01-15",
        "items": ["1.01", "8.01", "9.01"],
    },
}


class TestParsingASearchHit:
    def test_the_filer_and_ticker_come_out_of_the_display_name(self):
        filing = _parse(HIT)
        assert filing.entity == "Techpoint, Inc."
        assert filing.ticker == "THPTF"

    def test_the_url_points_at_the_actual_filing(self):
        assert _parse(HIT).url == (
            "https://www.sec.gov/Archives/edgar/data/1556898/000119312525006663/d879011d8k.htm"
        )

    def test_a_hit_with_no_filer_is_skipped_rather_than_guessed_at(self):
        assert _parse({"_id": "x:y.htm", "_source": {"ciks": [], "display_names": []}}) is None

    def test_a_filer_with_no_ticker_still_indexes(self):
        hit = {**HIT, "_source": {**HIT["_source"], "display_names": ["Some Trust (CIK 0001)"]}}
        filing = _parse(hit)
        assert filing.ticker == ""
        assert filing.to_document()["tickers"] == []


class TestTheDocumentRecord:
    def test_the_body_is_the_sec_item_taxonomy(self):
        body = _parse(HIT).to_document()["body"]
        assert "item 1.01 entry into a material definitive agreement" in body
        for item in HIT["_source"]["items"]:
            assert SEC_ITEMS[item].lower() in body

    def test_the_body_carries_the_vocabulary_matching_looks_for(self):
        from fintext.provenance import CLAIM_VOCABULARY, ClaimType

        body = _parse(HIT).to_document()["body"]
        assert any(word in body for word in CLAIM_VOCABULARY[ClaimType.MERGER])

    def test_it_is_indexed_as_a_primary_source(self):
        from fintext.provenance import Document

        assert Document.from_dict(_parse(HIT).to_document()).is_primary

    def test_the_title_names_the_first_recognised_item(self):
        assert _parse(HIT).title == "Form 8-K: Entry into a Material Definitive Agreement"

    def test_an_unrecognised_item_code_does_not_break_the_record(self):
        hit = {**HIT, "_source": {**HIT["_source"], "items": ["99.99"]}}
        assert _parse(hit).title == "Form 8-K"


class TestDates:
    def test_a_filing_date_becomes_midnight(self):
        import time

        filed = _parse(HIT).published_at
        assert time.strftime("%Y-%m-%d", time.localtime(filed)) == "2025-01-15"


class TestTheIndexSuppliesTheVocabulary:
    def test_aliases_come_from_the_documents_themselves(self):
        from fintext.provenance import Document, PrimaryIndex

        index = PrimaryIndex([Document.from_dict(_parse(HIT).to_document())])
        aliases = index.aliases()
        assert aliases["Techpoint, Inc."] == "techpoint"
        assert aliases["THPTF"] == "techpoint"

    def test_a_real_filer_is_recognised_end_to_end(self):
        from fintext.provenance import ALIASES, ClaimExtractor, Document, PrimaryIndex

        document = Document.from_dict(_parse(HIT).to_document())
        index = PrimaryIndex([document])
        extractor = ClaimExtractor({**ALIASES, **index.aliases()})

        claim = extractor.extract("Techpoint, Inc. agreed to acquire a rival in a merger")
        assert claim.entity_key == "techpoint"
        # The filing's items are 1.01, 8.01 and 9.01: "a material definitive agreement" says
        # nothing a contract or a loan agreement wouldn't, so on its own it no longer confirms a
        # merger.
        assert not index.search(claim, observed_at=document.published_at + 3600, window=86400)

        # The same filing with its text fetched (`fintext edgar --with-text`) names the merger,
        # and confirms the claim.
        with_text = Document.from_dict(
            _parse(HIT).to_document(body="agreement and plan of merger with indie semiconductor")
        )
        assert PrimaryIndex([with_text]).search(
            claim, observed_at=document.published_at + 3600, window=86400
        )


class TestEntityPatterns:
    @pytest.mark.parametrize(
        "headline,expected",
        [
            # The comma and the all-capitals house style are how EDGAR writes filers, and
            # neither matched until a real index arrived.
            ("Techpoint, Inc. agreed to acquire a rival in a merger", "Techpoint, Inc."),
            ("REVVITY, INC. agreed to acquire a rival in a merger", "REVVITY, INC."),
            ("First Abu Dhabi Bank PJSC agreed to acquire a lender", "First Abu Dhabi Bank"),
        ],
    )
    def test_real_filer_names_resolve(self, headline, expected):
        from fintext.provenance import ClaimExtractor

        assert ClaimExtractor().extract(headline).entity == expected
