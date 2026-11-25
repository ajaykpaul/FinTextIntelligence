"""Claim extraction against an index's own vocabulary (developed on the 2022-24 dev set)."""

from __future__ import annotations

import pytest

from fintext.provenance import ALIASES, ClaimExtractor
from fintext.schemas import ClaimType

INDEX = {
    "AGILENT TECHNOLOGIES INC": "agilent technologies",
    "A": "agilent technologies",
    "LEIDOS HOLDINGS, INC.": "leidos",
    "LDOS": "leidos",
    "CACI INTERNATIONAL INC": "caci international",
    "CACI": "caci international",
    "TARGET CORP": "target",
    "TGT": "target",
    "ON SEMICONDUCTOR CORP": "on semiconductor",
    "ON": "on semiconductor",
}


@pytest.fixture
def extractor():
    return ClaimExtractor({**ALIASES, **INDEX})


class TestTickersAreMatchedAsWritten:
    def test_the_article_a_is_not_agilent(self, extractor):
        claim = extractor.extract("Leidos was awarded a task order by the Air Force.")
        assert claim.entity_key == "leidos"

    def test_on_and_it_are_words_not_tickers(self, extractor):
        claim = extractor.extract("Analysts said the deal is on track; it was awarded a contract.")
        assert claim is None or claim.entity_key not in ("on semiconductor", "agilent technologies")

    def test_an_actual_ticker_still_resolves(self, extractor):
        claim = extractor.extract("CACI was awarded a $174 million contract by the Navy.")
        assert claim.entity_key == "caci international"

    def test_a_name_must_be_capitalised_where_it_appears(self, extractor):
        claim = extractor.extract("The analyst raised her price target and guidance looks safe.")
        assert claim is None or claim.entity_key != "target"


class TestClaimPhrasing:
    @pytest.mark.parametrize(
        "text",
        [
            "Leidos raised total company guidance for the full fiscal year.",
            "Leidos increases 2021 EPS guidance and raises 2022 guidance.",
            "We are once again raising our full-year revenue and earnings outlook, Leidos said.",
            "Leidos reiterated its previously announced guidance for fiscal 2024.",
        ],
    )
    def test_guidance_with_words_in_between(self, extractor, text):
        assert extractor.extract(text).claim_type is ClaimType.GUIDANCE

    @pytest.mark.parametrize(
        "text",
        [
            "CACI received a deficiency letter from Nasdaq.",
            "CACI received a Staff Delisting Determination from the exchange.",
            "CACI received a notice of noncompliance from NYSE American.",
        ],
    )
    def test_exchange_compliance_notices(self, extractor, text):
        assert extractor.extract(text).claim_type is ClaimType.REGULATORY

    def test_contract_with_a_long_description_in_between(self, extractor):
        text = "Leidos was awarded a formulation development and manufacturing contract."
        assert extractor.extract(text).claim_type is ClaimType.CONTRACT

    def test_guidance_does_not_reach_across_sentences(self, extractor):
        text = "Leidos raised prices. The outlook is uncertain."
        claim = extractor.extract(text)
        assert claim is None or claim.claim_type is not ClaimType.GUIDANCE


class TestGenericFilingWordsAreNotEnough:
    def _document(self, title, body):
        from fintext.provenance import Document

        return Document(
            doc_id="edgar-x",
            source_type="edgar",
            entity="EMCORE CORP",
            title=title,
            url="https://www.sec.gov/x",
            published_at=1_650_000_000.0,
            body=body,
            tickers=("EMKR",),
        )

    def test_an_agreement_filing_does_not_confirm_a_contract_award(self, extractor):
        from fintext.provenance import score_match

        claim = ClaimExtractor({"EMCORE CORP": "emcore", "EMKR": "emcore"}).extract(
            "Emcore was awarded a task order by the Air Combat Command."
        )
        agreement = self._document(
            "Form 8-K: Entry into a Material Definitive Agreement",
            "8-k item 1.01 entry into a material definitive agreement",
        )
        score, why = score_match(claim, agreement, observed_at=1_650_000_000.0, window=10800)
        assert score == 0.0 and "generic" in why

    def test_a_filing_that_names_the_contract_does(self, extractor):
        from fintext.provenance import score_match

        claim = ClaimExtractor({"EMCORE CORP": "emcore", "EMKR": "emcore"}).extract(
            "Emcore was awarded a task order by the Air Combat Command."
        )
        award = self._document(
            "Form 8-K: Entry into a Material Definitive Agreement",
            "8-k item 1.01 material definitive agreement contract award air force",
        )
        score, _ = score_match(claim, award, observed_at=1_650_000_000.0, window=10800)
        assert score >= 0.82


# --- claim particulars against filings with text --------------------------------


def _filing(body, entity="Azenta, Inc.", tickers=("AZTA",)):
    from fintext.provenance import Document

    return Document(
        doc_id="d",
        source_type="edgar",
        entity=entity,
        title="Form 8-K",
        url="",
        published_at=0.0,
        body=body,
        tickers=tickers,
    )


_CLAIM = (
    "Azenta, Inc. (NASDAQ: AZTA) today announced that it has entered into a definitive "
    "agreement to acquire Evertel Technologies for $215 million."
)
_TITLES = "8-k item 1.01 entry into a material definitive agreement item 8.01 other events"
_OTHER_DEAL = (
    _TITLES + " azenta announced today that it has completed the acquisition of b medical "
    "systems, a provider of cold chain solutions. the company expects the transaction to "
    "add to revenue in fiscal 2023, and management will discuss the merger on its next call."
)


def test_claim_details_are_the_particulars_not_the_company():
    from fintext.provenance import claim_details

    details = claim_details(_CLAIM, "AZTA", "azenta")
    assert "evertel" in details and "215" in details
    assert not {"azenta", "nasdaq", "azta", "inc", "technologies"} & set(details)


def test_a_filing_with_text_must_mention_a_particular():
    from fintext.provenance import ClaimExtractor, score_match

    claim = ClaimExtractor({"Azenta": "azenta", "AZTA": "azenta"}).extract(_CLAIM)
    assert claim is not None and claim.details
    score, why = score_match(claim, _filing(_OTHER_DEAL), observed_at=0.0, window=3600)
    assert score == 0.0 and "particulars" in why
    right = _OTHER_DEAL.replace("b medical systems", "evertel technologies")
    assert score_match(claim, _filing(right), observed_at=0.0, window=3600)[0] > 0


def test_a_filing_with_only_item_titles_is_judged_as_before():
    from fintext.provenance import ClaimExtractor, has_text, score_match

    claim = ClaimExtractor({"Azenta": "azenta", "AZTA": "azenta"}).extract(_CLAIM)
    titles_only = _TITLES.replace(
        "item 8.01 other events", "item 2.01 completion of acquisition or disposition of assets"
    )
    assert not has_text(_filing(titles_only))
    assert score_match(claim, _filing(titles_only), observed_at=0.0, window=3600)[0] > 0
