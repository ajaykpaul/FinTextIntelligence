Sample data
===========

Small fixtures so every command in the README runs with no downloads. They are
illustrative, not evaluation data: measured_metrics.json carries plausible numbers so
fintext accept can be demonstrated, and the replay set is hand-built to exercise each
provenance path.

  File                      Used by
  ------------------------- --------------------------
  score_items.jsonl         fintext batch
  provenance_replay.jsonl   fintext replay (PRD 8.2)
  measured_metrics.json     fintext accept (PRD 8)

Replace measured_metrics.json with real harness output before a release. A gated
criterion that is absent fails, so a partial file cannot pass by omission.
