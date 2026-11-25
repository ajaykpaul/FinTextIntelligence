"""Preprocessing, lexicons, scam rules, explanations and the catalogue."""

from __future__ import annotations

import pytest

from fintext import catalogue, scam_rules
from fintext.config import ConfigError, load_config
from fintext.explain import attribute, rationale_agreement
from fintext.lexicon import Lexicon, Lexicons, fold
from fintext.preprocess import (
    count_tokens,
    detect_arabizi,
    detect_language,
    normalise,
    prepare,
    scrub_pii,
)
from fintext.schemas import Language, Span, Subtype


class TestNormalisation:
    def test_removes_invisible_characters(self):
        # Bidi overrides and zero-width joiners hide terms from a naive matcher. They become
        # spaces rather than vanishing, so two words never merge.
        cleaned = normalise("kill​you‮")
        assert cleaned == "kill you"
        assert not any(ord(c) in (0x200B, 0x202E) for c in cleaned)

    def test_collapses_whitespace(self):
        assert normalise("a\n\n  b\t c") == "a b c"

    def test_strips_arabic_diacritics_and_tatweel(self):
        assert normalise("أَسـهم") == "أسهم"


class TestLanguage:
    def test_english(self):
        language, confidence, _ = detect_language("Revenue rose sharply this quarter")
        assert language is Language.EN and confidence > 0.9

    def test_arabic(self):
        assert detect_language("ترتفع أسهم البنك")[0] is Language.AR

    def test_arabic_with_a_latin_ticker_stays_arabic(self):
        # Missing Arabic loses the coverage the deliverable exists for.
        assert detect_language("أسهم FAB ترتفع اليوم")[0] is Language.AR

    def test_digits_only_is_unknown(self):
        assert detect_language("123 456")[0] is Language.UNKNOWN

    def test_flags_arabizi_without_claiming_arabic(self):
        assert detect_arabizi("ya3ni wallah 7abibi shu 3am tsawi")
        assert not detect_arabizi("the results beat expectations")

    def test_hint_only_breaks_ties(self):
        item = prepare("Revenue rose sharply", language_hint=Language.AR)
        assert item.language is Language.EN


class TestPII:
    def test_redacts_email_and_phone(self):
        text, redactions = scrub_pii("mail a.b@example.com or call +971 50 123 4567")
        assert "[EMAIL]" in text and "[PHONE]" in text
        assert {kind for kind, _, _ in redactions} == {"EMAIL", "PHONE"}

    def test_offsets_index_into_the_scrubbed_text(self):
        text, redactions = scrub_pii("contact a.b@example.com now")
        _, start, end = redactions[0]
        assert text[start:end] == "[EMAIL]"

    def test_luhn_check_spares_ordinary_long_numbers(self):
        text, _ = scrub_pii("order reference 1234567890123456 confirmed")
        assert "[CARD]" not in text

    def test_redacts_a_valid_card(self):
        assert "[CARD]" in scrub_pii("card 4111 1111 1111 1111 on file")[0]

    def test_scrubbed_before_the_text_is_stored(self):
        assert "@example.com" not in prepare("write to a.b@example.com about the merger").text


class TestTokenCap:
    def test_enforces_the_cap(self):
        item = prepare("word " * 500)
        assert item.token_count == 256 and item.truncated

    def test_cuts_on_a_word_boundary(self):
        item = prepare("alpha beta gamma delta epsilon zeta", max_tokens=3)
        assert item.text in ("alpha beta gamma", "alpha beta")

    def test_drops_redactions_the_cut_removed(self):
        item = prepare("alpha beta gamma a.b@example.com", max_tokens=3)
        assert all(end <= len(item.text) for _, _, end in item.redactions)

    def test_short_text_is_not_truncated(self):
        assert not prepare("short headline").truncated

    def test_long_words_cost_more_than_one_token(self):
        assert count_tokens("internationalisation") > 1

    def test_preprocessing_is_deterministic(self):
        # Batch reproducibility rests on this.
        text = "FAB profit rose 12%, contact a.b@example.com"
        assert prepare(text).content_hash == prepare(text).content_hash


class TestLexicon:
    @pytest.fixture
    def lexicons(self) -> Lexicons:
        return Lexicons.load(load_config())

    def test_version_carries_row_count_and_digest(self, lexicons):
        # A lexicon change is a release, so the version tracks content.
        assert "rows" in lexicons.versions["en"]

    def test_high_severity_forces_the_flag(self, lexicons):
        forced, matches = lexicons.floor("i will kill you tomorrow", Language.EN)
        assert forced and matches[0].entry.subtype is Subtype.THREAT

    def test_medium_severity_does_not_force(self, lexicons):
        forced, matches = lexicons.floor("this is shit", Language.EN)
        assert matches and not forced

    def test_clean_financial_text_does_not_match(self, lexicons):
        assert lexicons.floor("The bank reported a rise in profit", Language.EN) == (False, [])

    @pytest.mark.parametrize(
        "text",
        [
            "i will k1ll you",
            "I WILL KILL YOU",
            "i will  kill   you",
            "i w-i-l-l k.i.l.l y.o.u",
            "i will kiiill you",
        ],
    )
    def test_matches_through_obfuscation(self, lexicons, text):
        assert lexicons.floor(text, Language.EN)[0]

    def test_spans_point_at_the_original_text(self, lexicons):
        text = "honestly i will k1ll you"
        span = lexicons.floor(text, Language.EN)[1][0].span
        assert text[span.start : span.end] == span.text == "i will k1ll you"

    def test_longest_match_wins(self, lexicons):
        # "fuck you" (harassment) must beat "fuck" (profanity).
        matches = lexicons.floor("fuck you", Language.EN)[1]
        assert len(matches) == 1 and matches[0].entry.subtype is Subtype.HARASSMENT

    def test_phrase_requires_a_separator(self, lexicons):
        assert not lexicons.floor("killyou now", Language.EN)[0]

    def test_unknown_language_is_screened_against_every_list(self, lexicons):
        # A missed language identification must not disable the floor.
        assert lexicons.floor("__ar_slur_placeholder_1__", Language.UNKNOWN)[0]

    def test_can_be_disabled(self):
        assert Lexicons({}, enabled=False).floor("i will kill you", Language.EN) == (False, [])

    def test_folding_preserves_length(self):
        assert len(fold("F4CK y0u")) == len("F4CK y0u")

    def test_rejects_a_bad_subtype(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("term\tsarcasm\thigh\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown subtype"):
            Lexicon.load(path)

    def test_rejects_an_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("# comments only\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="empty"):
            Lexicon.load(path)


class TestScamRules:
    def test_guaranteed_returns(self):
        result = scam_rules.evaluate("Guaranteed 300% returns every month, no risk")
        assert "guaranteed_returns" in result.rule_ids and result.priority == "high"

    def test_urgency_alone_does_not_fire(self):
        # Neither half is suspicious alone; the conjunction is the signal.
        assert not scam_rules.evaluate("Act now to read our quarterly report").matched

    def test_payment_alone_does_not_fire(self):
        assert not scam_rules.evaluate("You can transfer AED to your savings account").matched

    def test_urgency_plus_payment_fires(self):
        result = scam_rules.evaluate("Hurry, only 2 slots left. Minimum deposit of 500 USDT.")
        assert "urgency_plus_payment" in result.rule_ids

    def test_bank_impersonation(self):
        result = scam_rules.evaluate("Official verified account of FAB. Share your OTP to verify.")
        assert "bank_impersonation" in result.rule_ids

    @pytest.mark.parametrize(
        "text",
        [
            "First Abu Dhabi Bank reported a 12% rise in first-quarter net profit.",
            "Our new savings account offers a competitive rate; terms apply.",
            "The fund charges 1.5% in management fees.",
        ],
    )
    def test_legitimate_text_does_not_fire(self, text):
        assert not scam_rules.evaluate(text).matched

    def test_score_stays_in_range_with_several_matches(self):
        result = scam_rules.evaluate(
            "Guaranteed returns! Insider tip: next 100x. Hurry, minimum deposit 500 USDT, no KYC."
        )
        assert len(result.rule_ids) >= 3 and 0.0 <= result.score <= 1.0

    def test_reports_spans_for_the_reviewer(self):
        result = scam_rules.evaluate("Guaranteed returns of 40% monthly")
        assert result.spans and all(s.text for s in result.spans)


class TestExplanations:
    def test_attributes_the_driving_token(self):
        spans = attribute(
            "the results were terrible today", lambda text: 0.9 if "terrible" in text else 0.1
        )
        assert [s.text for s in spans] == ["terrible"]

    def test_no_attribution_when_nothing_drives_the_score(self):
        assert attribute("neutral text", lambda _: 0.5) == ()

    def test_rationale_agreement(self):
        # Explanations are validated against HateXplain rationales.
        assert rationale_agreement([Span(4, 12, "pathetic")], [(4, 12)], 30)["iou"] == 1.0
        assert rationale_agreement([Span(0, 3, "the")], [(10, 18)], 30)["iou"] == 0.0


class TestCatalogue:
    def test_every_source_has_a_licence_and_a_link(self):
        for dataset in catalogue.CATALOGUE:
            assert dataset.licence
            # The internal capture has no link because there is nowhere to link.
            assert dataset.url.startswith("http") or dataset.internal

    def test_the_only_unreviewed_source_is_the_one_holding_our_own_traffic(self):
        assert [d.key for d in catalogue.unresolved()] == ["production_capture"]

    def test_named_fallbacks_resolve(self):
        for dataset in catalogue.CATALOGUE:
            if dataset.fallback:
                assert catalogue.get(dataset.fallback)

    def test_allows_declared_training_sources(self):
        catalogue.check_training_sources(["jigsaw_toxic", "osact4", "financial_phrasebank"])

    def test_blocks_reddit_from_training(self):
        with pytest.raises(catalogue.LicenceError, match="training not permitted"):
            catalogue.check_training_sources(["reddit"])

    def test_blocks_evaluation_only_dialect_probes(self):
        with pytest.raises(catalogue.LicenceError, match="training not permitted"):
            catalogue.check_training_sources(["lhsab"])

    def test_blocks_unknown_sources(self):
        with pytest.raises(catalogue.LicenceError, match="unknown dataset"):
            catalogue.check_training_sources(["some_scraped_corpus"])

    def test_reports_every_problem_at_once(self):
        with pytest.raises(catalogue.LicenceError) as error:
            catalogue.check_training_sources(["reddit", "lhsab"])
        assert "reddit" in str(error.value) and "lhsab" in str(error.value)

    def test_the_arabic_anchor_is_trainable(self):
        keys = {d.key for d in catalogue.select(task="toxicity", use="train", language="ar")}
        assert "osact4" in keys

    def test_writes_a_licence_log(self, tmp_path):
        content = catalogue.write_licence_log(tmp_path / "log.txt").read_text(encoding="utf-8")
        assert "Redistribute:" in content and "never used in training" in content
