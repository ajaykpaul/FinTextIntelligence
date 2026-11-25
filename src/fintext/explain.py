"""Attribution for flags (PRD 5.3, 6.5)."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .schemas import Explanation, Span

_TOKEN = re.compile(r"\S+")


def attribute(
    text: str, score: Callable[[str], float], *, top_k: int = 5, min_drop: float = 0.01
) -> tuple[Span, ...]:
    """Return the spans that pushed the score up, strongest first by weight."""
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)][:64]
    if not tokens:
        return ()

    baseline = score(text)
    drops = []
    for index, (_, start, end) in enumerate(tokens):
        without = " ".join(t for i, (t, _, _) in enumerate(tokens) if i != index)
        drop = baseline - score(without)
        if drop > min_drop:
            drops.append((drop, start, end))

    drops.sort(key=lambda d: -d[0])
    top = drops[:top_k]
    if not top:
        return ()
    largest = top[0][0]
    return tuple(
        Span(start, end, text[start:end], drop / largest)
        for drop, start, end in sorted(top, key=lambda d: d[1])
    )


def explain(spans: Sequence[Span], *, subject: str, method: str, note: str = "") -> Explanation:
    """Build the plain summary a reviewer reads first."""
    if spans:
        quoted = ", ".join(repr(s.text) for s in sorted(spans, key=lambda s: -s.weight)[:3])
        summary = f"{subject} driven mainly by {quoted}."
    else:
        summary = f"{subject} raised by the overall model score; no single span dominated."
    return Explanation(f"{summary} {note}".strip(), method, tuple(spans))


def rationale_agreement(
    predicted: Sequence[Span], human: Sequence[tuple[int, int]], length: int
) -> dict[str, float]:
    """Character-level agreement with human rationales, for HateXplain validation."""
    ours = {i for s in predicted for i in range(s.start, min(s.end, length))}
    theirs = {i for start, end in human for i in range(start, min(end, length))}
    if not ours and not theirs:
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0}
    overlap = len(ours & theirs)
    return {
        "iou": overlap / len(ours | theirs) if ours | theirs else 0.0,
        "precision": overlap / len(ours) if ours else 0.0,
        "recall": overlap / len(theirs) if theirs else 0.0,
    }
