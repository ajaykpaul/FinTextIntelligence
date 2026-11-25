export default function StatusStrip({ result, fetched }) {
  const facts = [
    `language ${result.language}`,
    `${result.latency_ms.toFixed(1)} ms`,
    `profile ${result.profile}`,
  ];

  // degraded = should have run but didn't; skipped = not applicable
  const warnings = [
    ...(result.truncated ? ["truncated at the 256-token cap"] : []),
    ...result.degraded.map((capability) => `degraded: ${capability}`),
    ...Object.entries(result.skipped).map(([name, why]) => `skipped ${name}: ${why}`),
  ];

  return (
    <>
      {fetched && (
        <div className="card fetched">
          <h2>Fetched</h2>
          <p className="verdict small">{fetched.title || "Untitled"}</p>
          <p className="sub">
            <a href={fetched.url} target="_blank" rel="noopener noreferrer">
              {fetched.url}
            </a>
          </p>
          <p className="why">
            {fetched.headline_only
              ? "Headline only; no article body found on the page."
              : `${fetched.characters.toLocaleString()} characters, cut to 256 tokens.`}{" "}
            Fetch time isn't counted in the latency.
          </p>
        </div>
      )}
    <div className="strip">
      {facts.map((fact) => (
        <span className="tag" key={fact}>
          {fact}
        </span>
      ))}
      {warnings.map((warning) => (
        <span className="tag warn" key={warning}>
          {warning}
        </span>
      ))}
    </div>
    </>
  );
}
