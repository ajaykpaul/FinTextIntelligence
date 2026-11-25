"""The toxicity lexicon floor (PRD 4.2, 6.5)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ConfigError
from .schemas import Language, Span, Subtype

# Leetspeak and lookalike folding, applied to both the text and the terms.
LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "9": "g",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "i",
    }
)
SEP = r"[\s._\-*+~^]"


def fold(text: str) -> str:
    """Case- and leet-fold, preserving length exactly."""
    lowered = "".join(c.lower() if len(c.lower()) == 1 else c for c in text)
    return lowered.translate(LEET)


@dataclass(frozen=True)
class Entry:
    term: str
    subtype: Subtype
    severity: str  # "high" forces the flag, "medium" only informs the subtype

    @property
    def forces_toxic(self) -> bool:
        return self.severity == "high"


@dataclass(frozen=True)
class Match:
    entry: Entry
    span: Span
    lexicon_version: str

    @property
    def forces_toxic(self) -> bool:
        return self.entry.forces_toxic


class Lexicon:
    """One versioned list for one language."""

    def __init__(self, entries: list[Entry], *, source: str) -> None:
        self.entries = entries
        digest = hashlib.sha256(
            "\n".join(sorted(f"{e.term}\t{e.subtype}\t{e.severity}" for e in entries)).encode()
        ).hexdigest()
        self.version = f"{Path(source).stem}+{len(entries)}rows+{digest[:8]}"
        self._compiled = [(e, re.compile(_pattern(e.term))) for e in entries]

    def __len__(self) -> int:
        return len(self.entries)

    def match(self, text: str) -> list[Match]:
        folded = fold(text)
        found = [
            Match(entry, Span(m.start(), m.end(), text[m.start() : m.end()]), self.version)
            for entry, pattern in self._compiled
            for m in pattern.finditer(folded)
        ]
        return _longest_wins(found)

    @classmethod
    def load(cls, path: Path) -> Lexicon:
        if not path.exists():
            raise ConfigError(f"lexicon not found: {path}")
        entries = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [f.strip() for f in line.split("\t") if f.strip()]
            if len(fields) < 3:
                raise ConfigError(f"{path}:{number}: expected term<TAB>subtype<TAB>severity")
            term, subtype, severity = fields[:3]
            if severity not in ("high", "medium"):
                raise ConfigError(f"{path}:{number}: unknown severity {severity!r}")
            try:
                entries.append(Entry(term, Subtype(subtype), severity))
            except ValueError as exc:
                raise ConfigError(f"{path}:{number}: unknown subtype {subtype!r}") from exc
        if not entries:
            raise ConfigError(f"{path}: lexicon is empty")
        return cls(entries, source=str(path))


def _pattern(term: str) -> str:
    words = [(SEP + "*").join(re.escape(c) + "+" for c in word) for word in fold(term).split()]
    body = (SEP + "+").join(words)
    return rf"(?<!\w){body}(?!\w)"


def _longest_wins(matches: list[Match]) -> list[Match]:
    matches.sort(key=lambda m: (m.span.start, -m.span.end))
    kept: list[Match] = []
    end = -1
    for match in matches:
        if match.span.start >= end:
            kept.append(match)
            end = match.span.end
    return kept


class Lexicons:
    """The per-language lists the service has loaded."""

    def __init__(self, by_language: dict[Language, Lexicon], *, enabled: bool = True) -> None:
        self.by_language = by_language
        self.enabled = enabled

    @property
    def versions(self) -> dict[str, str]:
        return {str(lang): lex.version for lang, lex in self.by_language.items()}

    def floor(self, text: str, language: Language) -> tuple[bool, list[Match]]:
        """Returns (forces_toxic, matches)."""
        if not self.enabled:
            return False, []
        lexicon = self.by_language.get(language)
        if lexicon is None:
            # A missed language identification must not disable the floor, so an unknown-
            # language item is screened against every list.
            matches = _longest_wins(
                [m for lex in self.by_language.values() for m in lex.match(text)]
            )
        else:
            matches = lexicon.match(text)
        return any(m.forces_toxic for m in matches), matches

    @classmethod
    def load(cls, config: Config) -> Lexicons:
        directory = config.path(config.lexicon_dir)
        return cls(
            {
                Language.EN: Lexicon.load(directory / config.lexicon_en),
                Language.AR: Lexicon.load(directory / config.lexicon_ar),
            },
            enabled=config.lexicons_enabled,
        )
