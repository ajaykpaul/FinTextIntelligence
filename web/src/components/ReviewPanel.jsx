import { useEffect, useState } from "react";

import { decide, drawBatch, getBatch } from "../api.js";
import ScoredSpans from "./ScoredSpans.jsx";

// Weekly review (PRD 8.4). The reviewer marks whether the flag was right, not
// whether the content should be blocked.
export default function ReviewPanel({ reviewer, onReviewerChange }) {
  const [batch, setBatch] = useState(null);
  const [precision, setPrecision] = useState(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    getBatch().then(({ ok, body }) => setBatch(ok ? body : null));
  }, []);

  const pending = batch?.alerts.filter((a) => !batch.decisions[a.alert_id]) ?? [];
  const current = pending[0];

  async function draw() {
    setBusy(true);
    const { ok, body } = await drawBatch({ days: 7, size: 50 });
    setBatch(ok ? { ...body, decisions: {}, outstanding: body.size } : null);
    setMessage(ok && body.size === 0 ? "No flags in the last seven days to review." : null);
    setBusy(false);
  }

  async function submit(verdict) {
    setBusy(true);
    const { ok, body } = await decide({
      alertId: current.alert_id,
      verdict,
      reviewer: reviewer.trim() || "unattributed",
      note,
    });
    if (ok) {
      setPrecision(body.precision);
      setNote("");
      const { body: refreshed } = await getBatch();
      setBatch(refreshed);
    }
    setBusy(false);
  }

  if (!batch) {
    return (
      <div className="card">
        <h2>Weekly review</h2>
        <p className="empty">
          No sample drawn yet. Target: 50+ flagged items a week, English precision 90% or more.
        </p>
        <button className="go" type="button" onClick={draw} disabled={busy}>
          {busy ? "Drawing..." : "Draw this week's sample"}
        </button>
      </div>
    );
  }

  return (
    <>
      <div className="strip">
        <span>
          batch {batch.batch_id}, seed {batch.seed}
        </span>
        <span>
          {batch.size - pending.length} of {batch.size} reviewed
        </span>
        {batch.short_of_minimum && (
          <span className="warn">below the {batch.minimum}-item weekly minimum</span>
        )}
      </div>

      {current ? (
        <div className="card">
          <h2>
            {current.kind === "toxicity" ? "Toxicity flag" : "Misleading-content flag"},{" "}
            {current.language}
          </h2>

          <p className="excerpt" dir="auto">
            <ScoredSpans text={current.text} spans={current.spans} kind={kindOf(current)} />
          </p>

          <p className="sub">
            score {current.score.toFixed(2)}, threshold {current.threshold.toFixed(2)}
          </p>
          <p className="why">{current.summary}</p>

          {current.language === "ar" && (
            <p className="why plain">
              Arabic alerts are reviewed via machine translation, so their precision is reported but
              not gated. Mark unclear ones Unsure.
            </p>
          )}

          <label className="note">
            <span>Note (optional)</span>
            <input
              type="text"
              value={note}
              placeholder="why this call"
              onChange={(event) => setNote(event.target.value)}
            />
          </label>

              <p className="ask">Was this flag correct?</p>
          <div className="verdicts">
            <button type="button" className="go" disabled={busy} onClick={() => submit("correct")}>
              Correct
            </button>
            <button
              type="button"
              className="go secondary"
              disabled={busy}
              onClick={() => submit("incorrect")}
            >
              Not correct
            </button>
            <button
              type="button"
              className="go ghost"
              disabled={busy}
              onClick={() => submit("unsure")}
            >
              Unsure
            </button>
          </div>
          <p className="hint">Unsure items don't count toward precision.</p>
        </div>
      ) : (
        <div className="card">
          <h2>Batch complete</h2>
          <p className="empty">{message ?? "Every item in this sample has a verdict."}</p>
          <button className="go" type="button" onClick={draw} disabled={busy}>
            Draw another sample
          </button>
        </div>
      )}

      <Precision report={precision} />

      <label className="note reviewer">
        <span>Reviewer</span>
        <input
          type="text"
          value={reviewer}
          placeholder="your name"
          onChange={(event) => onReviewerChange(event.target.value)}
        />
      </label>
    </>
  );
}

function kindOf(alert) {
  return alert.kind === "toxicity" ? "tox" : "mis";
}

function Precision({ report }) {
  if (!report) return null;

  return (
    <div className="card">
      <h2>Measured precision</h2>
      {Object.values(report.by_language).map((result) => (
        <div className="measure" key={result.language}>
          <p className={`verdict ${verdictTone(result)}`}>
            {result.precision === null ? "n/a" : `${Math.round(result.precision * 100)}%`}
          </p>
          <p className="sub">
            {result.language}, {result.correct} correct, {result.incorrect} not correct
            {result.unsure ? `, ${result.unsure} unsure` : ""}
          </p>
          <p className="why plain">
            {result.gated
              ? `Gated at ${Math.round(report.gate * 100)}%. ${
                  result.passes ? "Meets the gate." : "Below the gate."
                }`
              : result.qualification}
          </p>
        </div>
      ))}
      {report.spans_multiple_lineages && (
        <p className="why">
          This batch mixes model or threshold versions, so read precision per release.
        </p>
      )}
    </div>
  );
}

function verdictTone(result) {
  if (!result.gated || result.precision === null) return "neu";
  return result.passes ? "good" : "bad";
}
