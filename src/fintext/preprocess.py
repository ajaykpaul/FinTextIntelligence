"""Normalise, identify language, scrub PII and enforce the token cap."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from .schemas import Language

# Arabic script blocks, used for language identification.
_ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFEFF))


def _char_class(*ranges: tuple[int, int]) -> re.Pattern[str]:
    chars = "".join(chr(c) for lo, hi in ranges for c in range(lo, hi + 1))
    return re.compile("[" + re.escape(chars) + "]")


# Controls, zero-width characters and bidi overrides. Bidi matters: it is a known way to hide a
# term from a naive matcher.
INVISIBLE = _char_class(
    (0x0000, 0x0008),
    (0x000B, 0x000C),
    (0x000E, 0x001F),
    (0x007F, 0x009F),
    (0x200B, 0x200F),
    (0x2028, 0x202E),
    (0x2060, 0x2064),
    (0xFEFF, 0xFEFF),
)
# Harakat and tatweel, which Arabic text carries inconsistently.
DIACRITICS = _char_class(
    (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED), (0x0640, 0x0640)
)

# Most specific first: CARD and PHONE overlap and the first match wins.
PII_PATTERNS = [
    ("URL_USERINFO", re.compile(r"\bhttps?://[^\s/@]+:[^\s/@]+@\S+")),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("EMIRATES_ID", re.compile(r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    (
        "CARD",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    ),  # ends on a digit, so the match keeps no trailing space
    ("PHONE", re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?){2,4}\d{2,4}(?!\w)")),
]

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Item:
    """The preprocessed view of one input."""

    text: str
    language: Language
    language_confidence: float
    token_count: int
    truncated: bool
    content_hash: str
    redactions: tuple[tuple[str, int, int], ...] = ()
    script_mix: dict[str, float] = field(default_factory=dict)
    arabizi_suspected: bool = False


def normalise(text: str) -> str:
    """NFKC, drop invisible characters, strip diacritics, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE.sub(" ", text)
    text = DIACRITICS.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


# Hamza-carrying and wasla alef forms. Tweets use them interchangeably with bare alef, so one
# word arrives in three spellings the encoder sees as different.
_ALEF = str.maketrans(
    {"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0671": "\u0627"}
)
# Three or more of the same letter: "هههههه", "soooo".
_ELONGATED = re.compile(r"([^\W\d_])\1{2,}")


def fold_for_model(text: str) -> str:
    """Spelling variation the encoder shouldn't have to learn twice."""
    return _ELONGATED.sub(r"\1\1", text.translate(_ALEF))


def model_text(text: str) -> str:
    """Raw text to exactly what the deployed model reads."""
    clean, _ = scrub_pii(normalise(text))
    return fold_for_model(clean)


def detect_language(text: str) -> tuple[Language, float, dict[str, float]]:
    """English and MSA Arabic use disjoint scripts, so a script ratio is enough."""
    arabic = latin = other = 0
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS):
            arabic += 1
        elif char.isascii():
            latin += 1
        else:
            other += 1

    total = arabic + latin + other
    if total == 0:
        return Language.UNKNOWN, 0.0, {"arabic": 0.0, "latin": 0.0, "other": 0.0}

    mix = {"arabic": arabic / total, "latin": latin / total, "other": other / total}
    # Low on purpose so Arabic text that quotes Latin tickers is still treated as Arabic.
    if mix["arabic"] >= 0.20:
        return Language.AR, min(1.0, 0.5 + mix["arabic"] / 2), mix
    if mix["latin"] >= 0.60:
        return Language.EN, min(1.0, 0.5 + mix["latin"] / 2), mix
    return Language.UNKNOWN, 1.0 - max(mix.values()), mix


_ARABIZI_WORDS = frozenset({"wallah", "yalla", "habibi", "khalas", "inshallah", "shu", "akhi"})
_ARABIZI_DIGIT = re.compile(r"^[a-z']*[2379][a-z0-9']*$")


def detect_arabizi(text: str) -> bool:
    """Latin-script Arabic: digits standing in for letters, plus common words."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return False
    digit_ratio = sum(1 for w in words if _ARABIZI_DIGIT.match(w)) / len(words)
    known = any(w in _ARABIZI_WORDS for w in words)
    return digit_ratio > 0.15 or (known and digit_ratio > 0.05)


def scrub_pii(text: str) -> tuple[str, tuple[tuple[str, int, int], ...]]:
    """Replace PII with typed placeholders; returns (text, [(kind, start, end)])."""
    order = {kind: i for i, (kind, _) in enumerate(PII_PATTERNS)}
    matches = [
        (m.start(), m.end(), kind)
        for kind, pattern in PII_PATTERNS
        for m in pattern.finditer(text)
        if kind != "CARD" or _passes_luhn(m.group(0))
    ]
    matches.sort(key=lambda m: (m[0], order[m[2]], -(m[1] - m[0])))

    out: list[str] = []
    redactions: list[tuple[str, int, int]] = []
    cursor = position = 0
    for start, end, kind in matches:
        if start < cursor:  # overlaps one we already took
            continue
        out.append(text[cursor:start])
        position += start - cursor
        placeholder = f"[{kind}]"
        out.append(placeholder)
        redactions.append((kind, position, position + len(placeholder)))
        position += len(placeholder)
        cursor = end
    out.append(text[cursor:])
    return "".join(out), tuple(redactions)


def _passes_luhn(candidate: str) -> bool:
    """So ordinary long numbers are not redacted as card numbers."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def count_tokens(text: str) -> int:
    """Approximate subword count: long words split every four characters."""
    return sum(max(1, -(-len(w) // 4)) for w in _TOKEN.findall(text))


def prepare(text: str, *, max_tokens: int = 256, language_hint: Language | None = None) -> Item:
    """Full preprocessing."""
    clean, redactions = scrub_pii(normalise(text))
    language, confidence, mix = detect_language(clean)

    # A caller hint only breaks ties: ingestion metadata is often wrong about language, and the
    # script evidence is not.
    weak = language is Language.UNKNOWN or confidence < 0.6
    if weak and language_hint and language_hint is not Language.UNKNOWN:
        language, confidence = language_hint, 0.6

    tokens = count_tokens(clean)
    truncated = tokens > max_tokens
    if truncated:
        clean = _truncate(clean, max_tokens)
        tokens = max_tokens
        # Drop spans the cut removed, so every recorded offset stays valid.
        redactions = tuple(r for r in redactions if r[2] <= len(clean))

    return Item(
        text=clean,
        language=language,
        language_confidence=confidence,
        token_count=tokens,
        truncated=truncated,
        content_hash="sha256:" + hashlib.sha256(clean.encode()).hexdigest(),
        redactions=redactions,
        script_mix=mix,
        arabizi_suspected=language is not Language.AR and detect_arabizi(clean),
    )


def _truncate(text: str, max_tokens: int) -> str:
    """Cut at a word boundary at or before the cap, so no span ends mid-word."""
    kept: list[str] = []
    used = 0
    for word in text.split(" "):
        cost = count_tokens(word)
        if used + cost > max_tokens:
            break
        kept.append(word)
        used += cost
    return " ".join(kept) if kept else text[: max_tokens * 4]
