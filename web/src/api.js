// The only module that talks to the Python service.

/** Score one item and get the trading call with it.
 *
 *  Resolves with the response body for any HTTP status the service produces,
 *  because a 503 is a meaningful answer here rather than a transport failure: a
 *  compliance request that cannot be scored fails closed by design, and the UI
 *  should say so. */
export async function score({ text, profile, capabilities }) {
  const response = await fetch("/v1/signal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, profile, capabilities }),
  });
  const body = await response.json();
  return { ok: response.ok, status: response.status, body };
}

/** Service version for the header badge. Returns null when the service is not
 *  reachable, which the header renders as "offline" rather than throwing. */
export async function health() {
  try {
    const response = await fetch("/healthz");
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

/** Gated calls waiting for a person, urgent first. */
export async function getApprovals() {
  const response = await fetch("/v1/approvals");
  return { ok: response.ok, body: await response.json() };
}

/** Accept a gated call as a buy or a sell, or reject it outright. */
export async function resolveApproval({ itemId, action, direction, reviewer, note }) {
  const response = await fetch("/v1/approvals/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item_id: itemId, action, direction, reviewer, note }),
  });
  return { ok: response.ok, body: await response.json() };
}

/** Draw this week's sample of flagged items to review (PRD 8.4). */
export async function drawBatch({ days = 7, size = 50 } = {}) {
  const response = await fetch("/v1/review/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ days, size }),
  });
  return { ok: response.ok, body: await response.json() };
}

/** The current batch with the verdicts recorded so far. */
export async function getBatch() {
  const response = await fetch("/v1/review/batch");
  return { ok: response.ok, body: await response.json() };
}

/** Record a reviewer's verdict on one flag. */
export async function decide({ alertId, verdict, reviewer, note }) {
  const response = await fetch("/v1/review/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alert_id: alertId, verdict, reviewer, note }),
  });
  return { ok: response.ok, body: await response.json() };
}
