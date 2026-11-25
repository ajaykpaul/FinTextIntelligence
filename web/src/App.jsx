import { useEffect, useState } from "react";

import { health, score } from "./api.js";
import ApprovalQueue from "./components/ApprovalQueue.jsx";
import CallCard from "./components/CallCard.jsx";
import Lineage from "./components/Lineage.jsx";
import MisleadingCard from "./components/MisleadingCard.jsx";
import ProvenanceCard from "./components/ProvenanceCard.jsx";
import ReviewPanel from "./components/ReviewPanel.jsx";
import ScoreForm from "./components/ScoreForm.jsx";
import ScoredText from "./components/ScoredText.jsx";
import SentimentCard from "./components/SentimentCard.jsx";
import StatusStrip from "./components/StatusStrip.jsx";
import ToxicityCard from "./components/ToxicityCard.jsx";

const LABELS = { score: "Score", approvals: "Waiting on a person", review: "Weekly review" };

export default function App() {
  const [version, setVersion] = useState(null);
  const [view, setView] = useState("score");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  // headline offered by the error panel when a link can't be fetched
  const [seed, setSeed] = useState(null);
  // lives here so it survives switching tabs
  const [reviewer, setReviewer] = useState("");

  useEffect(() => {
    health().then((body) => setVersion(body ? `v${body.version}` : "offline"));
  }, []);

  async function handleSubmit(request) {
    setBusy(true);
    try {
      const { ok, status, body } = await score(request);
      setResult(ok ? body : null);
      setError(ok ? null : { status, ...body });
    } catch (caught) {
      setResult(null);
      setError({ status: 0, error: "unreachable", detail: String(caught) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <header>
        <h1>FinTextIntelligence</h1>
        <p>
          Score a headline or post for sentiment, toxicity, scam patterns and provenance.
        </p>
        <nav className="views">
          {["score", "approvals", "review"].map((name) => (
            <button
              key={name}
              type="button"
              className={view === name ? "active" : ""}
              onClick={() => setView(name)}
            >
              {LABELS[name]}
            </button>
          ))}
        </nav>
        <span className="pill">{version ?? "connecting..."}</span>
      </header>

      {view === "review" ? (
        <main className="single">
          <ReviewPanel reviewer={reviewer} onReviewerChange={setReviewer} />
        </main>
      ) : view === "approvals" ? (
        <main className="single">
          <ApprovalQueue reviewer={reviewer} onReviewerChange={setReviewer} />
        </main>
      ) : (
      <main>
        <ScoreForm busy={busy} seed={seed} onSubmit={handleSubmit} />
        <div>
          {error && (
            <ErrorPanel
              {...error}
              onUseSuggestion={(text) => setSeed({ text, at: Date.now() })}
            />
          )}
          {!error && !result && (
            <div className="card">
              <p className="empty">Nothing scored yet.</p>
            </div>
          )}
          {result && <Results result={result} />}
        </div>
      </main>
      )}

      <footer>
        Signals give a direction only, not size or timing. Nothing is removed automatically;
        flagged items go to a person.
      </footer>
    </>
  );
}

function Results({ result }) {
  const { signal, score, fetched } = result;
  return (
    <>
      <StatusStrip result={score} fetched={fetched} />
      <CallCard signal={signal} />
      <ScoredText result={score} />
      <div className="grid">
        <SentimentCard sentiment={score.sentiment} />
        <ToxicityCard toxicity={score.toxicity} />
        <MisleadingCard misleading={score.misleading} skipped={score.skipped} />
        <ProvenanceCard provenance={score.provenance} />
      </div>
      <Lineage result={score} />
    </>
  );
}

const TITLES = {
  link_unreachable: "That link would not open",
  invalid_request: "The service could not use that",
  scoring_unavailable: "Nothing was scored",
  unreachable: "The service is not answering",
};

function ErrorPanel({ status, error, detail, suggestion, onUseSuggestion }) {
  return (
    <div className="card">
      <div className="error">
        <strong>{TITLES[error] ?? `${status || "network"} ${error}`}</strong>
        <p>{detail}</p>
        {error === "link_unreachable" && (
          <>
            <p>
              Nothing was scored. Paste the headline or article text instead.
            </p>
            {suggestion && (
              <div className="offer">
                <p className="quote">{suggestion}</p>
                <button type="button" onClick={() => onUseSuggestion(suggestion)}>
                  Score this headline
                </button>
                <p className="caveat">
                  Taken from the URL, not the page, so expect a less confident result.
                </p>
              </div>
            )}
          </>
        )}
        {status === 503 && (
          <p>
            Compliance requests fail closed: if scoring is down, items wait instead of passing.
          </p>
        )}
      </div>
    </div>
  );
}
