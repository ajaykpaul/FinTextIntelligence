"""Building provenance replay cases from press-release exhibits."""

from __future__ import annotations

from fintext import replay_set
from fintext.provenance import EVIDENCE_HORIZON

MERGER, GUIDANCE, CONTRACT, REGULATORY = (c.states for c in replay_set.CATEGORIES)

CONCENTRA = (
    "EX-99.1 EXHIBIT 99.1 NEWS RELEASE FOR IMMEDIATE RELEASE Concentra Announces Signing of "
    "Definitive Agreement to Acquire Pivot Onsite Innovations ADDISON, TEXAS — Concentra Group "
    "Holdings Parent, Inc. (“Concentra,” “the Company,” “we,” “us,” or “our”) (NYSE: CON), the "
    "nation’s largest provider of occupational health services, today announced the signing of "
    "a definitive agreement to acquire Pivot Onsite Innovations from Athletico Physical Therapy. "
    "Pivot Onsite Innovations is a leader in the onsite health industry."
)
CSW = (
    "Transaction economics allow CSWI to maintain a strong balance sheet. Dallas, Texas – March "
    "18, 2025 – CSW Industrials, Inc. (Nasdaq: CSWI) and Aspen Manufacturing announced today the "
    "execution of a definitive agreement under which CSWI will acquire Aspen Manufacturing for "
    "approximately $313.5 million in cash, subject to customary post-closing adjustments. The "
    "proposed purchase price is approximately 11x."
)
LABCORP = (
    "BURLINGTON, N.C. – Labcorp (NYSE: LH), a global leader of laboratory services, today "
    "announced results for the second quarter ended June 30, 2025. Second quarter revenue grew "
    "9.5%. Labcorp raised its full-year guidance for revenue and adjusted EPS. Other news follows."
)


class TestClaimSentence:
    def test_dateline_and_quoted_aliases_are_dropped(self):
        found = replay_set.claim_sentence(
            CONCENTRA, states=MERGER, entity="Concentra Group Holdings Parent, Inc.", ticker="CON"
        )
        assert found.startswith("Concentra Group Holdings Parent, Inc. (NYSE: CON)")
        assert "“the Company,”" not in found
        assert found.endswith("from Athletico Physical Therapy.")

    def test_inc_does_not_end_the_sentence(self):
        found = replay_set.claim_sentence(
            CSW, states=MERGER, entity="CSW INDUSTRIALS, INC.", ticker="CSWI"
        )
        assert found.startswith("CSW Industrials, Inc. (Nasdaq: CSWI)")
        assert found.endswith("post-closing adjustments.")

    def test_the_sentence_that_states_the_claim_wins_over_the_first(self):
        found = replay_set.claim_sentence(LABCORP, states=GUIDANCE, entity="LABCORP", ticker="LH")
        assert found == "Labcorp raised its full-year guidance for revenue and adjusted EPS."

    def test_no_statement_of_the_claim_no_case(self):
        assert (
            replay_set.claim_sentence(LABCORP, states=CONTRACT, entity="LABCORP", ticker="LH")
            is None
        )

    def test_a_sentence_about_someone_else_is_not_used(self):
        other = "Rival Corp (NYSE: RIV) was awarded a contract by the Navy. More text here."
        assert (
            replay_set.claim_sentence(other, states=CONTRACT, entity="LABCORP", ticker="LH") is None
        )


class TestNames:
    def test_edgar_state_codes_and_capitals(self):
        assert replay_set.display_name("AMETEK INC/") == "Ametek Inc"
        assert replay_set.display_name("CSW INDUSTRIALS, INC.") == "Csw Industrials, Inc."
        assert replay_set.display_name("Revvity, Inc./DE/") == "Revvity, Inc."


class TestSwap:
    def test_every_mention_is_replaced_and_the_suffix_kept(self):
        found = replay_set.claim_sentence(
            CSW, states=MERGER, entity="CSW Industrials, Inc.", ticker="CSWI"
        )
        swapped = replay_set.swap_company(
            found,
            old_entity="CSW Industrials, Inc.",
            old_ticker="CSWI",
            new_entity="HEALTH CATALYST, INC.",
            new_ticker="HCAT",
        )
        assert swapped.startswith("Health Catalyst, Inc. (Nasdaq: HCAT)")
        assert "will acquire Aspen" in swapped and "HCAT will acquire" in swapped
        assert "CSW" not in swapped

    def test_gives_up_rather_than_leave_the_real_company_in(self):
        # "First" is the first word of the name and also an ordinary word.
        text = "First Watch, Inc. (NASDAQ: FWRG) raised guidance for the first quarter."
        assert (
            replay_set.swap_company(
                text,
                old_entity="First Watch, Inc.",
                old_ticker="FWRG",
                new_entity="Labcorp",
                new_ticker="LH",
            )
            is None
        )


def _case(entity, ticker, kind, when):
    return {
        "item_id": f"genuine-{kind}-{entity}",
        "is_genuine": True,
        "observed_at": when,
        "text": f"{entity}, Inc. (NYSE: {ticker}) today announced something about {kind}.",
        "expected_claim_type": kind,
        "expected_entity": f"{entity}, Inc.",
        "ticker": ticker,
        "documents": [],
    }


class TestFabricate:
    def test_a_partner_never_has_a_filing_that_could_support_the_claim(self):
        day = 86400.0
        genuine = [
            _case("Alpha", "ALP", "merger", 0.0),
            _case("Beta", "BET", "merger", day),  # same kind, nearby: never a partner for Alpha
            _case("Gamma", "GAM", "guidance", 2 * day),  # nearby, other kind
            _case("Delta", "DEL", "contract", 60 * day),  # far away
            _case("Echo", "ECH", "merger", 61 * day),
        ]
        fabricated = replay_set.fabricate(genuine)
        assert fabricated
        assert all(not f["is_genuine"] and not f["documents"] for f in fabricated)
        filed = {}
        for g in genuine:
            filed.setdefault(g["expected_entity"], []).append(
                (g["observed_at"], g["expected_claim_type"])
            )
        for case in fabricated:
            near = [
                kind
                for when, kind in filed[case["expected_entity"]]
                if abs(when - case["observed_at"]) <= EVIDENCE_HORIZON
            ]
            assert case["expected_claim_type"] not in near
            assert bool(near) == ("-nearby-" in case["item_id"])


def test_exhibit_headers_are_stripped():
    raw = (
        "EX-99.1 3 ea027027701ex99-1_realpha.htm PRESS RELEASE, DATED DECEMBER 22, 2025 "
        "Exhibit 99.1 reAlpha (Nasdaq: AIRE) Signs Definitive Agreement"
    )
    assert replay_set.strip_headers(raw) == "reAlpha (Nasdaq: AIRE) Signs Definitive Agreement"
    assert replay_set.strip_headers("EX-99.1 Document PayPal Reports") == "PayPal Reports"


def test_the_listing_ticker_is_replaced_even_when_edgar_disagrees():
    swapped = replay_set.swap_company(
        "DeFi Development Corp. (Nasdaq: DFDV) agreed to acquire a validator business.",
        old_entity="DeFi Development Corp.",
        old_ticker="JFU",
        new_entity="Palatin Technologies Inc",
        new_ticker="PTN",
    )
    assert (
        swapped == "Palatin Technologies Inc (Nasdaq: PTN) agreed to acquire a validator business."
    )


def test_filers_with_several_listed_classes():
    assert replay_set.split_display_name(
        "Banzai International, Inc.  (BNZI, BNZIW)  (CIK 0001826011)"
    ) == (
        "Banzai International, Inc.",
        "BNZI",
    )
    assert replay_set.split_display_name("QXO, Inc.  (QXO)  (CIK 0001236275)") == (
        "QXO, Inc.",
        "QXO",
    )


def test_edgar_index_parses_several_classes_too():
    from fintext.edgar import _parse

    filing = _parse(
        {
            "_id": "0001-24-000001:ex99.htm",
            "_source": {
                "ciks": ["0001826011"],
                "display_names": ["Banzai International, Inc.  (BNZI, BNZIW)  (CIK 0001826011)"],
                "adsh": "0001-24-000001",
                "form": "8-K",
                "file_date": "2024-12-10",
            },
        }
    )
    assert (filing.entity, filing.ticker) == ("Banzai International, Inc.", "BNZI")


def _write(path, cases):
    import json

    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


def _with_doc(case, doc_id, entity="Alpha, Inc.", tickers=("ALP",)):
    return case | {
        "documents": [
            {
                "doc_id": doc_id,
                "source_type": "edgar",
                "entity": entity,
                "title": "Form 8-K: Entry into a Material Definitive Agreement",
                "url": "https://www.sec.gov/x",
                "published_at": case["observed_at"],
                "body": "8-k item 2.01 completion of acquisition or disposition of assets",
                "tickers": list(tickers),
            }
        ]
    }


def test_dev_and_test_sets_must_not_share_filings(tmp_path):
    one = _with_doc(_case("Alpha", "ALP", "merger", 0.0), "edgar-1")
    two = _with_doc(_case("Beta", "BET", "merger", 0.0), "edgar-2")
    a = _write(tmp_path / "a.jsonl", [one])
    b = _write(tmp_path / "b.jsonl", [two])
    c = _write(tmp_path / "c.jsonl", [one, two])
    assert replay_set.shared_filings(a, b) == set()
    assert replay_set.shared_filings(a, c) == {"edgar-1"}


def test_why_names_the_failing_step(tmp_path):
    good = _with_doc(
        _case("Alpha", "ALP", "merger", 1.7e9)
        | {"text": "Alpha, Inc. (NYSE: ALP) agreed to acquire Gamma Labs for $2 billion."},
        "edgar-1",
    )
    vague = _with_doc(
        _case("Alpha", "ALP", "merger", 1.7e9) | {"text": "Markets were quiet."}, "edgar-2"
    )
    reasons = replay_set.why_not_corroborated(_write(tmp_path / "d.jsonl", [good, vague]), 0.82)
    assert len(reasons["corroborates"]) == 1
    assert len(reasons["no claim recognised"]) == 1


def test_a_fabrication_never_names_a_company_that_really_filed_nearby():
    day = 86400.0
    acquirer = _case("Onewater", "ONEW", "merger", 0.0)
    target = _case("Oceanbio", "OBCI", "merger", 0.0) | {
        "text": "Onewater, Inc. (NASDAQ: ONEW) signed a definitive agreement to acquire "
        "Oceanbio, Inc. (NYSE: OBCI) for $120 million."
    }
    others = [_case(n, t, "contract", 60 * day) for n, t in (("Delta", "DEL"), ("Echo", "ECH"))]
    fabricated = replay_set.fabricate([acquirer, target, *others])
    from_target = [f for f in fabricated if f["fabricated_from"] == target["item_id"]]
    # Swapping Oceanbio out still leaves Onewater, which filed a merger the same day: that would
    # be a true claim, so it must not be generated.
    assert all("Onewater" not in f["text"] for f in from_target)


# --- filing text for the replay ------------------------------------------------


class _FakeFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url):
        from fintext.edgar import EdgarError

        self.calls.append(url)
        if url not in self.pages:
            raise EdgarError("404")
        return self.pages[url].encode()


def _text_case(item_id, text, url, body="8-k item 1.01 entry into a material definitive agreement"):
    return {
        "item_id": item_id,
        "is_genuine": True,
        "observed_at": 1.0,
        "text": text,
        "documents": [
            {
                "doc_id": item_id,
                "source_type": "edgar",
                "entity": "Acme Corp",
                "url": url,
                "published_at": 1.0,
                "body": body,
                "tickers": [],
            }
        ],
    }


def test_add_text_appends_the_index_text_and_changes_nothing_else(tmp_path):
    import json

    from fintext.replay_set import add_text

    source = tmp_path / "cases.jsonl"
    cases = [
        _text_case("a", "Acme Corp signed a supply agreement with Widget Inc.", "u1"),
        _text_case("b", "Acme Corp raised guidance.", "u2"),
    ]
    source.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    before = source.read_text()
    out = tmp_path / "with_text.jsonl"
    partial = _FakeFetcher({"u1": "<p>Acme Corp signed a <b>supply</b> agreement</p>"})
    assert add_text(source, out, partial) == (1, 1)
    assert not out.exists()  # a partial set is never written
    fetcher = _FakeFetcher(
        {"u1": "<p>Acme Corp signed a <b>supply</b> agreement</p>", "u2": "<p>Guidance up</p>"}
    )
    assert add_text(source, out, fetcher) == (2, 0)
    assert source.read_text() == before  # the source set is untouched
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["text"] for r in rows] == [c["text"] for c in cases]
    assert rows[0]["documents"][0]["body"] == (
        "8-k item 1.01 entry into a material definitive agreement "
        "acme corp signed a supply agreement"
    )
    assert rows[1]["documents"][0]["body"].endswith("guidance up")


def test_hold_out_claim_removes_only_the_claim_sentence():
    from fintext.replay_set import hold_out_claim

    case = _text_case(
        "a",
        "Acme Corp’s board approved the merger with Widget Inc.",
        "u1",
        body="8-k item 2.01 news acme corp&#8217;s board approved the merger with widget inc. "
        "the combined company will be called acme widget.",
    )
    assert hold_out_claim(case)
    body = case["documents"][0]["body"]
    assert "approved the merger" not in body
    assert "combined company will be called acme widget" in body
    assert body.startswith("8 k item 2 01 news")


def test_hold_out_claim_reports_a_sentence_it_could_not_find():
    from fintext.replay_set import hold_out_claim

    case = _text_case("a", "Acme Corp won a contract.", "u1", body="8-k item 8.01 other events")
    assert not hold_out_claim(case)


def test_hold_out_claim_survives_the_claims_own_cleaning():
    from fintext.replay_set import hold_out_claim

    case = _text_case(
        "a",
        "Acme Corp agreed to acquire Kappa Bioscience, a leading maker of vitamin K2.",
        "u1",
        body="8-k item 2.01 news acme corp (nasdaq: acme) agreed to acquire kappa bioscience "
        "(&#8220;kappa&#8221;), a leading maker of vitamin k2. kappa was founded in oslo.",
    )
    assert hold_out_claim(case)
    body = case["documents"][0]["body"]
    assert "vitamin" not in body and "agreed" not in body
    assert body.endswith("kappa was founded in oslo")
