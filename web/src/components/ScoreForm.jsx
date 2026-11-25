import { useEffect, useState } from "react";

import { SAMPLES } from "../samples.js";

const PROFILES = ["trading", "compliance", "research"];
const CAPABILITIES = ["sentiment", "toxicity", "misleading", "provenance"];

export default function ScoreForm({ busy, seed, onSubmit }) {
  const [text, setText] = useState("");
  const [profile, setProfile] = useState("trading");
  const [capabilities, setCapabilities] = useState(CAPABILITIES);

  function submit(value = text) {
    const trimmed = value.trim();
    if (trimmed && !busy) onSubmit({ text: trimmed, profile, capabilities });
  }

  // seeded text from the error panel: show it, then score it
  useEffect(() => {
    if (!seed) return;
    setText(seed.text);
    submit(seed.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed]);

  function useSample(sample) {
    setText(sample.text);
    submit(sample.text);
  }

  function toggle(capability) {
    setCapabilities((current) =>
      current.includes(capability)
        ? current.filter((c) => c !== capability)
        : // keep CAPABILITIES order
          CAPABILITIES.filter((c) => current.includes(c) || c === capability),
    );
  }

  return (
    <form
      className="card"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <h2>Input</h2>

      <textarea
        // dir=auto handles Arabic RTL
        dir="auto"
        autoFocus
        value={text}
        placeholder="Paste a headline, or a link to an article."
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submit();
        }}
      />

      <div className="samples">
        {SAMPLES.map((sample) => (
          <button key={sample.label} type="button" onClick={() => useSample(sample)}>
            {sample.label}
          </button>
        ))}
      </div>

      <fieldset>
        <legend>Profile (sets latency budget and failure policy)</legend>
        <div className="segmented">
          {PROFILES.map((value) => (
            <label key={value}>
              <input
                type="radio"
                name="profile"
                value={value}
                checked={profile === value}
                onChange={() => setProfile(value)}
              />
              <span>{value}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>Capabilities</legend>
        <div className="checks">
          {CAPABILITIES.map((capability) => (
            <label key={capability}>
              <input
                type="checkbox"
                checked={capabilities.includes(capability)}
                onChange={() => toggle(capability)}
              />
              {capability}
            </label>
          ))}
        </div>
      </fieldset>

      <button className="go" type="submit" disabled={busy || !text.trim()}>
        {busy ? "Scoring..." : "Score"}
      </button>
      <p className="hint">Cmd/Ctrl+Enter to score. A URL on its own is fetched first.</p>
    </form>
  );
}
