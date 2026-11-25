"""Scam-pattern rules for misleading financial content (PRD 4.3, 6.5)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import Span

VERSION = "scam_rules.v2"
"""v2 adds the scam types in the SEC investor alerts saved under
data/raw/sec_investor_alerts (40 alerts, 2019-2026). A rule change is a release
(PRD 6.5), so it moves the version every decision is stamped with."""


def _patterns(*sources: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(s, re.IGNORECASE) for s in sources)


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    weight: float  # 1.0 flags on its own
    patterns: tuple[re.Pattern[str], ...]
    # A composite rule fires only when a pattern AND every requirement match.
    requires: tuple[re.Pattern[str], ...] = ()
    priority: str = "normal"

    def find(self, text: str) -> list[Span]:
        hits = [m for pattern in self.patterns for m in pattern.finditer(text)]
        if not hits:
            return []
        for requirement in self.requires:
            found = list(requirement.finditer(text))
            if not found:
                return []
            hits += found
        return sorted(
            (Span(m.start(), m.end(), m.group(0), self.weight) for m in hits),
            key=lambda s: s.start,
        )


# Someone is being asked to put money in. Reused by the topic rules so that a topic on its own,
# a disaster, an IPO, an SEC filing, never fires.
_SOLICITATION = (
    r"\b(?:invest(?:ing)?\s+(?:now|today|with\s+us|in\s+(?:this|our))"
    # "investment opportunities" alone is ordinary news prose; offered to the
    # reader ("this", "our", "an exclusive") it is a pitch.
    r"|(?:this|our|an?|exclusive|unique|rare|once[\s-]in[\s-]a[\s-]lifetime)\s+"
    r"investment\s+opportunit(?:y|ies)|opportunity\s+to\s+invest"
    r"|(?:join|get\s+in)\s+(?:now|early|today|us)|limited\s+(?:spots?|slots?|allocation)"
    r"|(?:minimum|min\.?)\s+(?:deposit|investment)|returns?\s+of\s+\d"
    r"|guaranteed|risk[\s-]?free|(?:dm|message|contact)\s+(?:me|us)"
    r"|whatsapp|telegram|send\s+(?:your|the)\s+(?:funds|payment|deposit))\b"
)

RULES = (
    Rule(
        id="guaranteed_returns",
        description="Promises a guaranteed, risk-free or fixed high return.",
        weight=1.0,
        priority="high",
        patterns=_patterns(
            r"\bguarantee(?:d|s|ing)?\s+(?:[\w%.,$]+\s+){0,3}?"
            r"(?:returns?|profits?|gains?|income|payouts?)\b",
            r"\b(?:risk[\s-]?free|no[\s-]?risk|zero[\s-]?risk)\s+"
            r"(?:investment|returns?|profits?|trade)",
            r"\b\d{2,4}\s?%\s*(?:\w+\s+){0,2}?"
            r"(?:guaranteed|per\s+(?:day|week|month)|monthly|weekly|daily)\b",
            r"\bcan(?:'|’)?t\s+lose\b",
            r"\bdouble\s+your\s+(?:money|investment|capital)\b",
            r"\bassured\s+(?:returns?|profits?)\b",
        ),
    ),
    Rule(
        id="urgency_plus_payment",
        # Neither half is suspicious alone; the conjunction is the signal, which is why this is
        # one composite rule and not two keyword lists.
        description="Time pressure combined with a request to transfer funds.",
        weight=1.0,
        priority="high",
        patterns=_patterns(
            r"\b(?:act|invest|deposit|transfer|send)\s+"
            r"(?:now|today|immediately|fast|within\s+\d+\s*\w+)\b",
            r"\b(?:offer|window|slots?|spots?)\s+(?:closes?|expires?|ends?)\s+"
            r"(?:today|tonight|in\s+\d+\s*\w+)\b",
            r"\b(?:last|final)\s+(?:chance|call)\b",
            r"\bonly\s+\d+\s+(?:slots?|spots?|places?)\s+(?:left|remaining)\b",
            r"\bhurry\b",
        ),
        requires=_patterns(
            r"\b(?:send|transfer|deposit|wire|pay)\b.{0,40}?"
            r"\b(?:usd|aed|eur|gbp|btc|eth|usdt|crypto|bitcoin|wallet|iban|account)\b"
            r"|\b(?:wallet\s+address|bank\s+details|western\s+union|gift\s+cards?)\b"
            r"|\b(?:minimum|min\.?)\s+(?:deposit|investment)\b",
        ),
    ),
    Rule(
        id="bank_impersonation",
        description="Institutional identity plus an off-brand channel or credential request.",
        weight=1.0,
        priority="high",
        patterns=_patterns(
            r"\b(?:official|verified|authoris?ed)\s+(?:account|channel|page|group)\s+of\b",
            r"\bon\s+behalf\s+of\s+(?:the\s+)?bank\b",
            r"\b(?:first\s+abu\s+dhabi\s+bank|FAB|central\s+bank)\b",
        ),
        requires=_patterns(
            r"\b(?:whatsapp|telegram|dm\s+(?:me|us)|direct\s+message|click\s+(?:the\s+)?link"
            r"|verify\s+your\s+(?:account|identity)|share\s+your\s+(?:otp|pin|password)"
            r"|update\s+your\s+(?:details|kyc))\b",
        ),
    ),
    Rule(
        id="stock_promotion",
        description="Pump-and-dump style promotion of a specific instrument.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:next|the)\s+(?:10|100|1000)\s?x\b",
            r"\bto\s+the\s+moon\b",
            r"\b(?:load|loading)\s+up\s+before\s+(?:the\s+)?(?:announcement|news|halt)\b",
            r"\b(?:buy|accumulate)\s+\$?[A-Z]{2,5}\s+(?:now|before)\b",
            r"\bmassive\s+(?:pump|breakout)\s+(?:incoming|coming|today)\b",
            r"\binsider\s+(?:tip|info(?:rmation)?)\b",
        ),
    ),
    Rule(
        id="unregistered_scheme",
        description="Solicitation carrying markers of an unregistered investment scheme.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:no\s+(?:kyc|paperwork|questions)|without\s+(?:kyc|registration))\b",
            r"\b(?:refer|referral)\s+(?:a\s+friend|and\s+earn)\b.{0,40}\b(?:commission|bonus|%)\b",
            r"\b(?:forex|crypto|binary\s+options?)\s+(?:signals?|robot|bot)\s+(?:group|vip|club)\b",
            r"\bprivate\s+(?:fund|pool)\b.{0,40}\b(?:invite\s+only|closed\s+group)\b",
        ),
    ),
    Rule(
        id="regulator_impersonation",
        # SEC alert, Sep 2025: fraudsters "may impersonate the SEC, or SEC officials or
        # employees, on social media or in text messages to solicit you for scams".
        description="Claims to speak for the SEC or another regulator, then asks for something.",
        weight=1.0,
        priority="high",
        patterns=_patterns(
            r"\b(?:sec|finra|securities\s+and\s+exchange\s+commission)\s+"
            r"(?:official|officer|agent|investigator|employee|staff|representative|department)\b",
            r"\b(?:on\s+behalf\s+of|from|with)\s+the\s+"
            r"(?:sec|finra|securities\s+and\s+exchange\s+commission)\b",
        ),
        requires=_patterns(
            r"\b(?:recover(?:y)?\s+(?:your|of\s+your|lost)?\s*(?:funds|money|losses|assets)"
            r"|(?:processing|release|clearance|unlock(?:ing)?)\s+fee|pay\s+(?:a|the)\s+fee"
            r"|verify\s+your\s+(?:account|identity)|share\s+your\s+(?:otp|pin|password|ssn)"
            r"|whatsapp|telegram|click\s+(?:the\s+)?link|wire\s+(?:the|your)\s+(?:funds|money)"
            r"|gift\s+cards?)\b",
        ),
    ),
    Rule(
        id="registration_as_endorsement",
        # SEC alerts, Aug 2026 (ERA filings) and the updated Form 4 alert: a filing or a
        # registration is cited as proof the offer is legitimate.
        description="Presents SEC approval, registration or a filing as proof an offer is safe.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:approved|endorsed|backed|guaranteed|certified)\s+by\s+the\s+"
            r"(?:sec|securities\s+and\s+exchange\s+commission)\b",
            r"\bsec[\s-](?:approved|endorsed|backed|certified|guaranteed)\b",
        ),
    ),
    Rule(
        id="registration_cited_to_sell",
        description="Cites SEC registration or a filing while soliciting an investment.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:registered|filed)\s+with\s+the\s+(?:sec|securities\s+and\s+exchange\s+commission)\b",
            r"\bsec[\s-]registered\b",
            r"\b(?:check|see|view|verify)\s+our\s+(?:sec\s+)?(?:era|form\s+(?:adv|4|d))\s*(?:filing)?\b",
        ),
        requires=_patterns(_SOLICITATION),
    ),
    Rule(
        id="disaster_pitch",
        # SEC alerts on natural disasters, the Texas floods, the California wildfires, Hurricane
        # Ida and COVID-19: "Fraudsters often seek to use disasters and other events ... to lure
        # victims into investment scams."
        description="An investment pitch built on a disaster, recovery effort or health crisis.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:hurricane|wildfires?|floods?|flooding|earthquakes?|disasters?|"
            # Not "reconstruction": it names institutions (the European Bank
            # for Reconstruction and Development) far more often than pitches.
            r"rebuilding|cleanup|clean-up|relief\s+efforts?|"
            r"pandemic|covid(?:-19)?|coronavirus|vaccine|cure)\b",
        ),
        requires=_patterns(_SOLICITATION),
    ),
    Rule(
        id="relationship_lure",
        # SEC alert on relationship investment scams: contact "online or on social media
        # platforms", then "the long process of building their trust, be it through friendship,
        # romance, or an offer to help achieve financial goals".
        description="A personal contact offering to teach or show the way to trading profits.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:i|we)\s+(?:can|will|could)\s+(?:teach|show|help|guide)\s+you\s+"
            r"(?:how\s+)?(?:to\s+)?(?:invest|trade|earn|make\s+money|grow\s+your)",
            r"\bmy\s+(?:uncle|aunt|mentor|cousin|teacher|friend)\b.{0,50}\b"
            r"(?:trading|crypto|platform|signals?|made\s+(?:me\s+)?\$?\d)",
            r"\bsorry,?\s+(?:wrong\s+number|i\s+(?:texted|messaged)\s+the\s+wrong)\b.{0,80}"
            r"\b(?:invest|trad|crypto|profit)",
        ),
    ),
    Rule(
        id="pre_ipo_offer",
        # SEC alert on pre-IPO investment scams: "many pre-IPO offerings targeted at the general
        # public may be illegal."
        description="Offers the public pre-IPO shares or a way in before a listing.",
        weight=0.7,
        patterns=_patterns(
            r"\bpre[\s-]?ipo\s+(?:shares?|stock|fund|offering|allocation|access|units?)\b",
            r"\b(?:buy|get|own|secure)\s+(?:in|shares?|stock)\s+before\s+"
            r"(?:the\s+ipo|it\s+goes\s+public|(?:the\s+)?listing)\b",
        ),
        requires=_patterns(_SOLICITATION + r"|\b(?:you|your)\b"),
    ),
    Rule(
        id="ai_trading_pitch",
        # SEC alerts on group chats and on AI and investment fraud: a group may "promote an AI
        # trading node or trading algorithm that generates stock picks".
        description="An AI bot, node or algorithm sold on the returns it will make.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:ai|a\.i\.|artificial\s+intelligence)[\s-](?:powered\s+)?"
            r"(?:trading\s+)?(?:bot|robot|node|algorithm|system|software)\b.{0,80}"
            r"\b(?:returns?|profits?|earn(?:s|ings)?|%|guarantee[ds]?|daily|passive\s+income)\b",
            r"\b(?:our|my|this)\s+(?:ai|algorithm)\s+(?:generates?|picks?|finds?)\s+"
            r"(?:winning|stock|the\s+best)\b",
        ),
    ),
    Rule(
        id="high_yield_program",
        # SEC alert: "The hallmark of a high-yield investment program (HYIP) scam is the promise
        # of incredible returns with little or no risk." Prime-bank and bank-guarantee programs
        # do not exist as investments, so the names alone are enough.
        description="High-yield investment program, prime-bank or bank-guarantee scheme.",
        weight=1.0,
        priority="high",
        patterns=_patterns(
            r"\bhigh[\s-]yield\s+investment\s+programs?\b",
            r"\bhyips?\b",
            r"\bprime[\s-]bank\s+(?:notes?|instruments?|programs?|trading|guarantees?)\b",
            r"\b(?:bank\s+guarantee|standby\s+letter\s+of\s+credit)\s+"
            r"(?:trading|programs?|platform|monetization|monetisation)\b",
        ),
    ),
    Rule(
        id="bogus_cd",
        # SEC alert on spoofed websites offering phony certificates of deposit: CDs advertised
        # at high fixed rates and falsely described as insured.
        description="Certificates of deposit at an implausibly high rate, presented as insured.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:cds?|certificates?\s+of\s+deposit)\b.{0,60}?"
            r"\b(?:[6-9]|[1-9]\d)(?:\.\d+)?\s?%",
            r"\b(?:[6-9]|[1-9]\d)(?:\.\d+)?\s?%.{0,40}?\b(?:cds?|certificates?\s+of\s+deposit)\b",
        ),
        requires=_patterns(r"\b(?:fdic[\s-]?insured|fully\s+insured|guaranteed|risk[\s-]?free)\b"),
    ),
    Rule(
        id="affinity_appeal",
        # SEC alert on fraud in faith-based communities: the fraudster "may even be (or pretend
        # to be) part of the very group they are trying to cheat".
        description="An investment offer framed as being for members of a faith or community.",
        weight=0.7,
        patterns=_patterns(
            r"\b(?:fellow\s+(?:believers|brothers|sisters|members|worshippers)"
            r"|our\s+(?:church|congregation|parish|mosque|temple|synagogue|faith\s+community)"
            r"|god\s+(?:wants|has\s+blessed|will\s+bless)|blessed\s+(?:returns|investment))\b",
        ),
        requires=_patterns(_SOLICITATION),
    ),
)


@dataclass(frozen=True)
class RuleResult:
    matched: bool = False
    score: float = 0.0
    priority: str = "normal"
    rule_ids: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()
    spans: tuple[Span, ...] = ()


def evaluate(text: str, rules: tuple[Rule, ...] = RULES) -> RuleResult:
    """Apply the scam rules."""
    hits = [(rule, spans) for rule in rules if (spans := rule.find(text))]
    if not hits:
        return RuleResult()

    # Combine with diminishing returns rather than a sum: three weak rules should not outrank
    # one decisive one, and the score has to stay in [0, 1] to be comparable with the model
    # score it sits beside.
    miss = 1.0
    for rule, _ in hits:
        miss *= 1.0 - rule.weight

    return RuleResult(
        matched=True,
        score=1.0 - miss,
        priority="high" if any(r.priority == "high" for r, _ in hits) else "normal",
        rule_ids=tuple(r.id for r, _ in hits),
        descriptions=tuple(r.description for r, _ in hits),
        spans=tuple(sorted((s for _, spans in hits for s in spans), key=lambda s: s.start)),
    )
