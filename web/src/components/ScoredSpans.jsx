import { Fragment } from "react";

// Highlights spans in the text. Offsets are into scored_text (after
// normalisation/PII scrub/truncation), not the raw input.
export default function ScoredSpans({ text, spans, kind }) {
  const ordered = [...(spans ?? [])].sort((a, b) => a.start - b.start || b.end - a.end);
  const pieces = [];
  let cursor = 0;

  for (const span of ordered) {
    if (span.start < cursor) continue; // overlaps one already taken; first wins
    pieces.push(<Fragment key={`gap-${cursor}`}>{text.slice(cursor, span.start)}</Fragment>);
    pieces.push(
      <mark className={span.kind ?? kind} key={`mark-${span.start}`}>
        {text.slice(span.start, span.end)}
      </mark>,
    );
    cursor = span.end;
  }
  pieces.push(<Fragment key="tail">{text.slice(cursor)}</Fragment>);

  return pieces;
}
