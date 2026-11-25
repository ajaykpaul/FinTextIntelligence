import Bars from "./Bars.jsx";

export default function ToxicityCard({ toxicity }) {
  if (!toxicity) return null;

  const subtypes = Object.entries(toxicity.subtypes).sort((a, b) => b[1] - a[1]);

  return (
    <div className="card">
      <h2>Toxicity</h2>
      <p className={`verdict ${toxicity.toxic ? "bad" : "good"}`}>
        {toxicity.toxic ? "toxic" : "not toxic"}
      </p>
      <p className="sub">
        score {toxicity.score.toFixed(2)}, threshold {toxicity.threshold.toFixed(2)}
      </p>

      {/* lexicon match overrode the model */}
      {toxicity.lexicon_override && (
        <div className="chips">
          <span className="chip">lexicon override</span>
        </div>
      )}

      <Bars rows={subtypes} />

      <p className="why">{toxicity.explanation.summary}</p>
    </div>
  );
}
