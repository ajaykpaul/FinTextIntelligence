import ScoredSpans from "./ScoredSpans.jsx";

export default function ScoredText({ result }) {
  if (!result.scored_text) return null;

  const spans = [
    ...(result.toxicity?.explanation.spans ?? []).map((span) => ({ ...span, kind: "tox" })),
    ...(result.misleading?.explanation.spans ?? []).map((span) => ({ ...span, kind: "mis" })),
    ...(result.provenance?.claim ? [{ ...result.provenance.claim.span, kind: "claim" }] : []),
  ];

  return (
    <div className="card">
      <h2>Scored text</h2>
      <p className="excerpt" dir="auto">
        <ScoredSpans text={result.scored_text} spans={spans} />
      </p>
      <div className="legend">
        <span>
          <i className="swatch tox" />
          toxicity evidence
        </span>
        <span>
          <i className="swatch mis" />
          scam pattern
        </span>
        <span>
          <i className="swatch claim" />
          checkable claim
        </span>
      </div>
    </div>
  );
}
