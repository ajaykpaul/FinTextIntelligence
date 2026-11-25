"""The v2 scam rules, one per scam type in the SEC investor alerts."""

from __future__ import annotations

import pytest

from fintext import scam_rules


def fired(text: str) -> tuple[str, ...]:
    return scam_rules.evaluate(text).rule_ids


SCAMS = [
    # "fraudsters may impersonate the SEC, or SEC officials or employees, on social media or in
    # text messages to solicit you for scams" (Sep 2025)
    (
        "regulator_impersonation",
        "This is an SEC official. We can help with the recovery of your lost funds; "
        "a processing fee is required first.",
    ),
    (
        "regulator_impersonation",
        "Message from the Securities and Exchange Commission: verify your account "
        "on WhatsApp to release your assets.",
    ),
    # The SEC does not approve investments; ERA and Form 4 alerts describe filings and
    # registration used "to create a false impression of legitimacy".
    ("registration_as_endorsement", "Our fund is SEC-approved, so your money is safe."),
    (
        "registration_as_endorsement",
        "Every trade is approved by the SEC before we place it.",
    ),
    (
        "registration_cited_to_sell",
        "We are registered with the SEC. Join now, limited spots for new investors.",
    ),
    (
        "registration_cited_to_sell",
        "Check our ERA filing, then DM me to start with a minimum deposit of $500.",
    ),
    # "Fraudsters often seek to use disasters and other events, such as floods, wildfires, and
    # hurricanes to lure victims into investment scams."
    (
        "disaster_pitch",
        "Wildfire rebuilding is booming. This investment opportunity "
        "offers returns of 20% a month.",
    ),
    (
        "disaster_pitch",
        "Our company has a coronavirus cure. Get in early before the news breaks.",
    ),
    # Relationship scams: contact "online or on social media platforms", building trust "through
    # friendship, romance, or an offer to help achieve financial goals".
    (
        "relationship_lure",
        "I can teach you how to trade crypto, I made $40k last month on this platform.",
    ),
    (
        "relationship_lure",
        "Sorry, wrong number! By the way, are you into investing? My returns are amazing.",
    ),
    (
        "relationship_lure",
        "My uncle runs a trading platform that made me $12,000 in a week.",
    ),
    # "many pre-IPO offerings targeted at the general public may be illegal"
    (
        "pre_ipo_offer",
        "Secure your pre-IPO shares in the AI startup today, limited allocation.",
    ),
    ("pre_ipo_offer", "You can buy shares before it goes public through our fund."),
    # A group may "promote an AI trading node or trading algorithm that generates stock picks".
    (
        "ai_trading_pitch",
        "Our AI trading bot earns 3% daily in passive income.",
    ),
    ("ai_trading_pitch", "Our AI generates winning stock picks every morning."),
    # "The hallmark of a high-yield investment program (HYIP) scam is the promise of incredible
    # returns with little or no risk."
    ("high_yield_program", "Join our high-yield investment program."),
    ("high_yield_program", "Exclusive prime bank instruments for qualified investors."),
    ("high_yield_program", "Bank guarantee monetization program, returns paid weekly."),
    # Spoofed sites offering phony CDs, "falsely stated that the deposits were FDIC-insured".
    ("bogus_cd", "Certificates of deposit paying 9.5% APY, fully FDIC-insured."),
    ("bogus_cd", "Lock in 12% on our 1-year CDs, guaranteed."),
    # Faith-based affinity fraud: the fraudster "may even be (or pretend to be) part of the very
    # group they are trying to cheat".
    (
        "affinity_appeal",
        "Fellow believers, God has blessed this fund. Join us, minimum investment $1,000.",
    ),
]


@pytest.mark.parametrize("rule,text", SCAMS)
def test_each_scam_type_in_the_alerts_is_caught(rule, text):
    assert rule in fired(text)


NEWS = [
    # Every one of these names the same topic as a rule, the way a news story or a filing would,
    # and none of them is asking anyone for money.
    "The SEC staff charged the adviser with fraud on Tuesday, according to the complaint.",
    "The fund said it is registered with the SEC under the Investment Advisers Act.",
    "The company filed a Form 4 with the SEC disclosing the director's share purchase.",
    "Insurers fell after Hurricane Ida losses exceeded estimates.",
    "Wildfire cleanup costs weighed on the utility's quarterly results.",
    "The European Bank for Reconstruction and Development extended a 21.6 million euro loan.",
    "The startup raised money in a pre-IPO funding round led by two venture firms.",
    "Analysts said AI trading algorithms now account for a large share of daily volume.",
    "Average one-year CD rates rose to 4.2%, FDIC data showed.",
    "Banks are offering CDs at 5.1%, FDIC-insured up to the usual limit.",
    "The church sold its building to a developer for $3 million.",
    "The pandemic hurt demand, and the company is exploring investment opportunities in Asia.",
]


@pytest.mark.parametrize("text", NEWS)
def test_news_on_the_same_topics_does_not_fire(text):
    new_rules = {rule.id for rule in scam_rules.RULES[5:]}
    assert not new_rules & set(fired(text))


class TestTheRelease:
    def test_a_rule_change_moves_the_version(self):
        assert scam_rules.VERSION == "scam_rules.v2"

    def test_rule_ids_are_unique(self):
        ids = [rule.id for rule in scam_rules.RULES]
        assert len(ids) == len(set(ids))

    def test_the_impersonation_and_hyip_rules_are_high_priority(self):
        high = {rule.id for rule in scam_rules.RULES if rule.priority == "high"}
        assert {"regulator_impersonation", "high_yield_program"} <= high

    def test_a_topic_alone_never_fires(self):
        for text in (
            "Hurricane season starts in June.",
            "We are registered with the SEC.",
            "Our congregation meets on Sundays.",
        ):
            new_rules = {rule.id for rule in scam_rules.RULES[5:]}
            assert not new_rules & set(fired(text)), text
