// plain-English meaning per state (uncorroborated != false)
const MEANING = {
  not_applicable: ["neu", "No checkable market-moving claim in the text."],
  pending: [
    "info",
    "Normal for breaking news. Trading may hold briefly but won't block on it.",
  ],
  corroborated: [
    "good",
    "Matched a filing, issuer release or exchange notice.",
  ],
  uncorroborated: [
    "flag",
    "No supporting document within the expected window. Sent for review; this doesn't mean the claim is false.",
  ],
};

export default function ProvenanceCard({ provenance }) {
  if (!provenance) return null;

  const [tone, meaning] = MEANING[provenance.state] ?? ["neu", ""];

  return (
    <div className="card">
      <h2>Provenance</h2>
      <p className={`verdict ${tone}`}>{provenance.state.replace(/_/g, " ")}</p>

      {provenance.claim && (
        <p className="sub">
          {provenance.claim.claim_type} claim about{" "}
          <strong>{provenance.claim.entity || "an unresolved entity"}</strong>
        </p>
      )}
      {provenance.window_seconds && (
        <p className="sub">corroboration window {Math.round(provenance.window_seconds / 60)} min</p>
      )}

      <div className="chips">
        {provenance.evidence.map((document) => (
          <span className="chip info" key={document.doc_id}>
            {document.doc_id}, {document.source_type}, {Math.round(document.match_score * 100)}%
          </span>
        ))}
      </div>

      <p className="why">{meaning}</p>
      {/* skip the reason if it just repeats the line above */}
      {!sameSentence(provenance.reason, meaning) && (
        <p className="why plain">{provenance.reason}</p>
      )}
    </div>
  );
}

function sameSentence(a, b) {
  const letters = (text) => text.toLowerCase().replace(/[^a-z]/g, "");
  return letters(a) === letters(b);
}
