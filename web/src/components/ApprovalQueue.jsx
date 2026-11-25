import { useCallback, useEffect, useState } from "react";

import { getApprovals, resolveApproval } from "../api.js";

// Items the gate sent for review. Accept needs a side (buy/sell); reject drops it.
export default function ApprovalQueue({ reviewer, onReviewerChange }) {
  const [queue, setQueue] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [note, setNote] = useState("");
  const [last, setLast] = useState(null);

  const refresh = useCallback(async () => {
    const { ok, body } = await getApprovals();
    if (ok) setQueue(body);
    else setError(body.detail ?? "could not read the queue");
  }, []);

  useEffect(() => {
    refresh();
    // poll every 5s
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  async function resolve(item, action, direction = "") {
    if (!reviewer.trim()) {
      setError("Enter your name first.");
      return;
    }
    setBusy(true);
    setError(null);
    const { ok, body } = await resolveApproval({
      itemId: item.item_id,
      action,
      direction,
      reviewer,
      note,
    });
    if (ok) {
      setLast(body.resolution);
      setNote("");
      setQueue({ ...body, items: (queue?.items ?? []).filter((i) => i.item_id !== item.item_id) });
    } else {
      setError(body.detail ?? "that decision was not recorded");
    }
    setBusy(false);
  }

  const items = queue?.items ?? [];
  const item = items[0];

  return (
    <div className="card">
      <h2>Waiting on a person</h2>

      {queue && (
        <p className="sub">
          {queue.waiting === 0
            ? "Nothing waiting."
            : `${queue.waiting} waiting, ${queue.urgent_waiting} of them urgent. Oldest has been here ${formatAge(queue.oldest_age_ms)}.`}
        </p>
      )}

      <label className="note reviewer">
        <span>Reviewer</span>
        <input
          value={reviewer}
          placeholder="your name"
          onChange={(event) => onReviewerChange(event.target.value)}
        />
      </label>

      {error && <p className="error inline">{error}</p>}

      {item ? (
        <div className="pending">
          <div className="pending-head">
            <span className={`tag ${item.priority === "now" ? "urgent" : ""}`}>
              {item.priority === "now" ? "urgent" : "routine"}
            </span>
            <span className="age">waiting {formatAge(item.age_ms)}</span>
          </div>

          <p className="item-text" dir="auto">
            {item.text}
          </p>
          <p className="why">{item.reason}</p>

          <p className="lean">
            {item.lean
              ? `Model leaned ${item.lean} (${(item.conviction * 100).toFixed(0)}% conviction).`
              : "No lean from the model."}
          </p>

          <label className="note">
            <span>Note (optional)</span>
            <input
              value={note}
              placeholder="why this call"
              onChange={(event) => setNote(event.target.value)}
            />
          </label>

          <p className="ask">Accept this as a trade, or reject it?</p>
          <div className="verdicts">
            <button
              type="button"
              className="go buy"
              disabled={busy}
              onClick={() => resolve(item, "accept", "buy")}
            >
              Accept: buy
            </button>
            <button
              type="button"
              className="go sell"
              disabled={busy}
              onClick={() => resolve(item, "accept", "sell")}
            >
              Accept: sell
            </button>
            <button
              type="button"
              className="go ghost"
              disabled={busy}
              onClick={() => resolve(item, "reject")}
            >
              Reject
            </button>
          </div>
        </div>
      ) : (
        <p className="empty">Queue is empty.</p>
      )}

      {last && (
        <p className="recorded">
          Saved: <strong>{last.outcome}</strong> on {last.item_id.slice(0, 8)} by {last.reviewer},
          after {formatAge(last.waited_ms)}.
        </p>
      )}

      {queue && queue.resolved > 0 && (
        <p className="why">
          {queue.resolved} resolved ({queue.accepted} accepted, {queue.rejected} rejected), median
          wait {formatAge(queue.median_wait_ms)}.
        </p>
      )}
    </div>
  );
}

function formatAge(ms) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 90) return `${seconds.toFixed(0)}s`;
  return `${(seconds / 60).toFixed(0)} min`;
}
