// Gate output: buy, sell or review. Shown first since it's what downstream uses.
export default function CallCard({ signal }) {
  if (!signal) return null;

  // scoring failed, so there's no signal at all
  if (!signal.gated) {
    return (
      <div className="card call">
        <h2>Signal</h2>
        <p className="verdict neu">no call</p>
        <p className="sub">scoring failed</p>
        <p className="why">{signal.reason}</p>
      </div>
    );
  }

  const routine = signal.priority === "routine";
  const tone = { buy: "good", sell: "bad", review: routine ? "neu" : "flag" }[signal.direction];

  return (
    <div className={`card call ${tone}${routine ? " quiet" : ""}`}>
      <h2>Signal</h2>
      <p className={`verdict ${tone}`}>{signal.direction}</p>

      <p className="sub">
        {signal.direction === "review"
          ? routine
            ? "routine, no action needed"
            : "needs a person now"
          : `conviction ${Math.round(signal.conviction * 100)}%`}
        {signal.hold_ms > 0 && `, hold ${signal.hold_ms} ms`}
      </p>

      <p className="why">{signal.reason}</p>
    </div>
  );
}
