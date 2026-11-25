import Bars, { percent } from "./Bars.jsx";

const TONE = { positive: "pos", negative: "neg", neutral: "neu" };

export default function SentimentCard({ sentiment }) {
  if (!sentiment) return null;

  return (
    <div className="card">
      <h2>Sentiment</h2>
      <p className={`verdict ${TONE[sentiment.label]}`}>{sentiment.label}</p>
      <p className="sub">confidence {percent(sentiment.confidence)}</p>

      <Bars rows={["positive", "negative", "neutral"].map((k) => [k, sentiment.scores[k]])} />

      <p className="why">
        {sentiment.acceptance_gated
          ? "English: acceptance-gated."
          : "Not acceptance-gated (no verified Arabic labels)."}
      </p>
    </div>
  );
}
