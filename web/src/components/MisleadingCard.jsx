export default function MisleadingCard({ misleading, skipped }) {
  // English only; show why it was skipped
  if (!misleading) {
    if (!skipped.misleading) return null;
    return (
      <div className="card">
        <h2>Misleading content</h2>
        <p className="verdict neu">not run</p>
        <p className="sub">{skipped.misleading}</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Misleading content</h2>
      <p className={`verdict ${misleading.likely_misleading ? "flag" : "good"}`}>
        {misleading.likely_misleading ? "needs review" : "no flag"}
      </p>
      <p className="sub">
        score {misleading.score.toFixed(2)}, threshold {misleading.threshold.toFixed(2)}, priority{" "}
        {misleading.review_priority}
      </p>

      <div className="chips">
        {misleading.rule_matches.map((rule) => (
          <span className="chip" key={rule}>
            {rule}
          </span>
        ))}
      </div>

      <p className="why">{misleading.explanation.summary}</p>
    </div>
  );
}
