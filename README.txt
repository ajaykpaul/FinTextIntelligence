FinTextIntelligence
===================

Sentiment, toxicity, misleading-content triage and provenance signals for trading and
compliance, in English and Modern Standard Arabic, behind one low-latency service.
Implements PRD v1.5.

    make demo

    trading    Revenue missed consensus and the board cut its full-year g   [en] 0.6 ms | sentiment=negative (0.90) | toxicity=clean (0.02) | misleading=clean (0.08, normal) | provenance=pending
    compliance Guaranteed 300% returns. Act now, only 3 slots left, send    [en] 0.5 ms | sentiment=neutral (0.62) | toxicity=clean (0.03) | misleading=FLAG (1.00, high) | provenance=not_applicable
    compliance You are a pathetic clown and I will k1ll you.                [en] 0.5 ms | sentiment=neutral (0.70) | toxicity=TOXIC+lexicon (0.99) | misleading=clean (0.08, normal) | provenance=not_applicable

That's the rule-based reference backend: standard library only, no model download, no
GPU, no network. The trained model gives different scores; see "Running the trained
model" below.

----------------------------------------------------------------------------------------

Project summary
---------------

Problem. Trading desks and compliance teams need four signals on financial text, fast
enough for the trading hot path and honest about their limits: sentiment, toxicity,
likely-misleading content, and whether a claim is corroborated by a primary source. The
text is in English and Arabic.

Approach. One XLM-R base encoder with four heads, trained only on public data whose
licences are checked in code. Rules and retrieval sit around the model: a high-severity
lexicon floor, scam-pattern rules, and an SEC EDGAR provenance check. The policy layer
decides; the service only reports.

Result: the current release is shared-xlmr.v4.1, which is not accepted. Of the 14 gates,
5 pass, 6 fail and 3 need a pilot. Every number below is on a frozen test split, with
thresholds tuned on dev.

  Gate                           v4.1            Target    
  ------------------------------ --------------- --------- ------
  English sentiment F1           0.807           ≥ 0.80    pass
  English toxicity AUROC         0.953           ≥ 0.95    pass
  English misleading F1          0.807           ≥ 0.80    pass
  Corroboration precision        1.000 (72/72)   ≥ 0.95    pass
  Claim-extraction recall        0.934           ≥ 0.85    pass
  Arabic toxicity AUROC          0.909           ≥ 0.92    fail
  Arabic toxicity F1             0.710           ≥ 0.80    fail
  English toxicity F1            0.793           ≥ 0.85    fail
  EN–AR toxicity F1 gap          0.083           ≤ 0.07    fail
  p95 latency, CPU, 256 tokens   51–61 ms        ≤ 50 ms   fail
  Throughput per replica         27–30 rps       ≥ 100     fail

Arabic sentiment is reported, not gated. It rose from 0.490 to 0.739 when native Arabic
tweets were added to training, and a check found no leakage into the test set.

Three findings shaped the work more than any single fix. Retraining with a different
seed moved the numbers as much as the first round of fixes did, so setups are compared
over several seeds, with intervals covering both seed and test sample. The Arabic F1
target may be capped by noisy labels rather than by the model; a label audit is designed
to settle that. And int8 quantisation changed 4–9% of decisions on both ARM and x86, so
the service runs fp32. The evaluation report has the measurements, and its Future work
section covers what's left: a smaller encoder for latency, Arabic pre-training and the
label audit for Arabic toxicity, an in-domain test set, and three measurements that need
a pilot.

Running the trained model

Full steps for a laptop, a remote server and retraining are in docs/INSTALL.txt.

The weights ship separately, as a privately shared model pack, for size and licence
reasons (docs/MODEL_PACK.txt).

    unzip FinTextIntelligence-model-v4.1.zip -d .     # from the project folder
    pip install -e '.[dev,onnx]'
    fintext --config configs/release_v4.1.toml demo
    fintext --config configs/release_v4.1.toml serve  # http://127.0.0.1:8080/ui

Without the pack, everything below still runs on the heuristic reference backend.

----------------------------------------------------------------------------------------

What it does
------------

  Capability            Output                                                 Languages    Acceptance-gated
  --------------------- ------------------------------------------------------ ------------ --------------------------
  Financial sentiment   positive / negative / neutral                          EN, MSA AR   English only
  Toxicity              binary flag + hate / harassment / profanity / threat   EN, MSA AR   Both
  Misleading content    triage flag + scam-pattern rules                       EN           English
  Provenance            one of four corroboration states                       EN           Precision ≥ 0.95
  Trading gate          buy / sell / review, and how urgently                  EN, AR       —
  Approvals             accept a gated call as buy or sell, or reject it       EN, AR       —
  Human review          reviewer verdict on each sampled flag                  EN, AR       English precision ≥ 0.90

Three properties are in the types rather than in the docs, because a consumer reads the
schema:

- Nothing asserts falsity. MisleadingResult.disposition has two values: no_flag and
  likely_misleading_needs_review. The provenance payload carries an explicit
  asserts_falsity: false.
- pending and uncorroborated are separate states. "Not yet confirmed" is the normal
  condition of breaking news and cannot be collapsed into "unsupported" by a consumer
  checking one negative state.
- The service reports; the policy layer decides. No score and no state blocks content,
  suppresses a signal or takes any action.

Quick start
-----------

    pip install -e '.[dev]'
    make test          # 625 tests, no network, no artefacts
    make demo          # score a built-in set of examples in the terminal
    make serve         # http://127.0.0.1:8080, then open /ui

http://127.0.0.1:8080/ui is the demonstration surface: paste a headline, pick a consumer
profile, and see the scores with the triggering spans highlighted in the text the model
actually saw.

Two views: Score, and Weekly review: the sampled batch of flagged items where a person
marks each flag correct, not correct or unsure. That verdict is what produces the
human-review precision the acceptance gate reads, so the gate measures something rather
than accepting a hand-typed number.

It is a React app under web/, built into the Python package and served as static assets.
React is the controller only: it holds form state, calls POST /v1/score, and renders
what comes back. It has no scoring logic and no privileged access, so it cannot show
anything the API does not return. The built bundle is committed, so make serve works
with no Node installed.

    make ui-install    # once
    make ui-dev        # http://localhost:5173, hot reload, API proxied to :8080
    make ui-build      # rebuild into src/fintext/static/, then commit

    curl -s localhost:8080/v1/score -d '{
      "text": "First Abu Dhabi Bank PJSC agreed to acquire a 40% stake in a regional lender.",
      "profile": "trading"
    }' | python -m json.tool

The two constraints that shape everything
-----------------------------------------

Latency. Scoring completes inside a 50 ms p95 on CPU or trading systems will not call
the service. So: batch size one, single-threaded int8 ONNX Runtime, a hard 256-token
cap, a deadline check before each stage, and a CI benchmark that refuses to promote an
artefact that misses the objective.

Language coverage. The developer does not read Arabic, so Arabic commitments are made
only where native-speaker-labelled public benchmarks supply the evidence. Arabic
toxicity is a firm deliverable anchored on OSACT; Arabic sentiment is measured and
reported but is not an acceptance gate, and the response says so in a field
(acceptance_gated) rather than a footnote. The Arabic lexicon is a published research
list, versioned and integrated but not curated in-house.

Layout
------

One module per concern, no sub-packages.

    src/fintext/
      schemas.py      request and response types (the API)
      config.py       one dataclass, loaded from TOML, hashed onto every response
      preprocess.py   normalise, language ID, PII scrub, token cap
      fetch.py        a pasted link, turned into the article behind it
      catalogue.py    data sources and the licence gate that blocks training
      scoring.py      model backends: heuristic (no deps) and int8 ONNX
      lexicon.py      versioned high-severity lexicon floor
      scam_rules.py   SEC-derived scam patterns, independent of the model
      explain.py      occlusion attribution and HateXplain rationale agreement
      provenance.py   claim extraction, primary-source index, state machine
      pipeline.py     orchestration: budgets, degradation, lineage
      server.py       HTTP routing, stdlib transport, FastAPI adapter
      static/         the built UI bundle, served at /ui
      registry.py     immutable registry with the latency promotion gate
      monitoring.py   latency histogram, PSI drift, Prometheus
      audit.py        the decision log
      signal.py       the trading gate: scores in, one decision out
      approvals.py    the queue a gated call waits in for a person
      corpus.py       captured traffic + human labels -> fine-tuning data
      corpora.py      the PRD's declared sources, loaded and split
      edgar.py        SEC filings -> the provenance index
      training.py     shared-encoder fine-tune, checkpointed, ONNX export
      evaluation.py   metrics, PRD 8 gates, provenance replay, canary
      review.py       weekly sampling, reviewer verdicts, measured precision
      cli.py          fintext <command>

    web/              the UI, a Vite + React app
      index.html      the HTML shell
      src/api.js      the only module that talks to the Python service
      src/App.jsx     the controller: request state, response state, errors
      src/components/ one presentational component per result card
      src/styles.css  plain CSS, no framework, no CSS-in-JS

Pasting a link
--------------

Paste a bare link and the service fetches it, pulls out the headline and body, and
scores that instead of the URL string. Anything else (a link inside a sentence, a quoted
post) is scored as the text it is.

The fetch runs before the pipeline and outside its budget. The 50 ms clock covers
scoring; a remote site answers on its own schedule.

Two things worth knowing:

- The address is checked before we connect. http and https only, at most three
  redirects, and the resolved address has to be a public one. A link pointing at
  127.0.0.1 or 10.0.0.5 is refused with a 400; that one is the caller's to fix.

- Plenty of publishers will not open their articles to us. Reuters, the FT and most
  paywalled sites answer a non-browser with a 401 or 403. That comes back as
  502 link_unreachable, not a 400, because the request was fine and the refusal was at
  the other end. Where the URL carries a readable headline in its path, the response
  offers it as a suggestion and the UI puts a button on it, so a blocked link ends
  somewhere rather than nowhere. A headline without its article is thinner evidence, and
  the call is less sure for it.

- A front page is not an article. news.google.com/home and its like leave a two-word
  page title and nothing else, either because they are an index of many stories or
  because they draw their text in the browser. That is refused rather than scored: a
  confident-looking answer about thirteen characters is worse than no answer. A page
  that leaves a real headline but no body is scored, and the response says headline_only
  so nobody mistakes it for the whole story.

Set fetch_urls = false to turn the whole thing off; links then score as strings.

The trading gate
----------------

POST /v1/signal scores an item and returns one call with it: buy, sell, or review,
meaning a person decides. Every item gets one of the three and the reason behind it.

    curl -s localhost:8080/v1/signal -d '{"text":"Record profit, strong margins and an upgraded outlook."}'

    {"direction": "buy", "conviction": 0.99, "gated": true,
     "priority": "", "hold_ms": 0, "reason": "clearly positive at 0.99"}

It costs about 1.3 µs on top of the score. Three things about it are easy to get wrong:

- Review is not one queue. A scam flag or a borderline reading comes back
  priority: "now". A routine notice with nothing to trade comes back
  priority: "routine": a clear answer, not an ambiguous one, and nobody has to work
  through it. Without that split about 70% of a realistic stream lands on a reviewer;
  with it, 9%. That pile is the thing the product exists to shrink (PRD 2.3).
- A failed score makes no call at all. direction: null with gated: false means scoring
  broke, so the strategy carries on as though the gate were not deployed. Turning an
  outage into a review call would bury the reviewers at the worst possible moment.
- Unconfirmed never blocks the trade. A pending claim costs conviction, not the call
  (PRD 6.4). The hold is off by default: against a 20 ms budget, holding for anything
  meaningful is suppression by another name.

The gate is a separate module and a separate route on purpose. PRD 3.2 keeps trading
strategy out of scope, and it still is. The gate says which way, not how much, what
instrument, or when to execute. Nothing in a /v1/score reply can carry an action.

The provenance index
--------------------

fintext edgar builds it from SEC EDGAR full-text search: free, no key, 2001 onward. Each
8-K becomes one document, and the body is the SEC's own item taxonomy ("item 1.01 entry
into a material definitive agreement") rather than synthesised text. That matters in an
index whose entire job is corroboration: every word of the evidence came from the SEC.
--with-text fetches the filings themselves at one extra request each, and is what the
shipped index uses.

With text, a filing has to mention something particular to the claim (another name or an
amount), not only the company and an event word: a long release mentions "guidance" or
"acquire" whatever it is about, and on the development replay that let unrelated filings
confirm fabricated claims. Filings with only item titles are judged on company, event
and date, as before.

Requests are throttled to five a second against a published limit of ten, and
FINTEXT_SEC_CONTACT sets the contact address SEC's fair-access policy asks for.

The index supplies the entity vocabulary

This is the part worth knowing. The claim extractor scores an unrecognised capitalised
name at 0.45, below its own threshold, so it only resolves entities it has seen before.
That list was seven hand-written UAE names, and against real filings it recognised
nothing at all.

The index already knows every filer it holds and their tickers, so it supplies that
vocabulary now. 874 documents yield 1,093 aliases. A claim about a company with no
primary document stays unrecognised, which is correct: there would be nothing to
corroborate it against.

The two human loops
-------------------

They answer different questions, so they are different surfaces.

Waiting on a person (approvals.py, /v1/approvals) is the live one. The gate returns
review when it will not make the call itself; the item lands here and stays until
somebody resolves it. The clock is running, so the queue is ordered urgent-first and
every resolution records how long the item sat there.

Accepting means naming a side. The gate withheld a direction because it did not have
one, so an "accept" that did not say buy or sell would just pass the ambiguity along.
The queue shows which way the model leaned where it leaned at all, clearly marked as
what the model saw rather than a recommendation. A reviewer who can only agree with the
model is not a control.

    curl -s localhost:8080/v1/approvals | jq '.waiting, .urgent_waiting, .oldest_age_ms'

Weekly review (review.py, /v1/review/*) is the audit. It samples flagged items, asks
"was this flag correct", and produces the precision number PRD 8.4 gates on. It never
asks whether to block anything; the service reports and the policy layer decides.

Feeding it back
---------------

Everything scored is captured; corpus.py turns it into training data. The module hangs
on one distinction:

  A human verdict is a label. A model output is not.

Training on the model's own scores is a loop. It learns to reproduce its own answers,
its mistakes get reinforced rather than corrected, and the eval numbers keep looking
fine because the eval set drifts with it. So model outputs are captured as unlabelled
and never used as targets. They earn their keep as the pool the next review batch is
sampled from, which is where labels actually come from.

That leaves three kinds of record:

  Kind         Where the label came from                       Used as a target
  ------------ ----------------------------------------------- -----------------------------
  gold         a person accepted, rejected, or judged a flag   yes
  unlabelled   scored, nobody has looked at it                 no; sampling and drift only
  held out     gold, reserved for evaluation                   no

The eval split is taken from the content hash, so an item lands in the same split every
rebuild. A split re-rolled each time leaks the eval set into training within a couple of
runs.

    fintext corpus --summary-only   # what is there
    fintext corpus                  # write it, if the licence has cleared

Two gates sit in front of any of it:

- Licence. Production text arrives with no licence: somebody pasted a news article. It
  is declared in the catalogue as production_capture, and declared pending, so
  check_training_sources refuses to build the file. Whether you may fine-tune on
  articles your users pasted is a question for a lawyer, and clearing it is one word in
  catalogue.py once they have answered.
- PII. Captured text has already been through preprocess.scrub.

retain_text_for_training (on by default) is what makes the corpus possible: it keeps the
text of every scored item rather than only flagged ones. That is a retention decision
before it is an ML one. Turn it off and the service holds only what it flagged, which is
what a data-minimisation review will ask for.

Training
--------

    pip install -e '.[train]'
    fintext train                       # all declared sources, 3 epochs
    fintext train --limit 2000 --epochs 1   # a quick pass to check the wiring

corpora.py loads the sources the catalogue declares, and the licence gate runs on the
declared names before anything downloads. Each source resolves to a concrete Hub
artefact recorded in MIRRORS and printed with every run, because several of the
canonical repositories ship as loading scripts that datasets 5.0 no longer executes. The
declared source is what the licence log governs; the mirror is what actually arrived.

HuggingFace token

Optional; downloads work without it, just slower and with tighter rate limits. It is
never written into the project, because the project gets zipped and synced.
fintext/credentials.py looks for it in:

- Mac or any local machine: run hf auth login once (saved in
  ~/.cache/huggingface/token), or set HF_TOKEN in your shell profile
- Colab: the 🔑 Secrets panel, as fintext (HF_TOKEN also works), with notebook access on

Train, dev, test

Held-out rows are split in half. Thresholds are tuned on dev; every number is reported
on test, which nothing is ever tuned on. A threshold tuned on the split it is reported
on makes every thresholded figure optimistic, and PRD 8.1 asks for frozen test sets for
exactly that reason.

When no threshold meets both toxicity floors (recall 0.90 for PRD 8.1, precision 0.90
for PRD 8.4's review), recall wins and the run says so: toxicity_feasible_en is False,
the reported F1 is the real F1 at that threshold, and threshold_detail records the dev
ceiling, the best F1 any threshold could have reached. Before this, an infeasible
threshold reported F1 as 0.0, which failed the gate for the wrong reason and hid the
real number.

    fintext import-model ~/Downloads/<run>.zip   # bring a notebook run back into artifacts/models
    fintext evaluate artifacts/models/<run>      # re-score a trained model under this protocol

evaluate rebuilds the run's splits from its manifest and refuses to continue if the
train set does not match the one the model was fitted on.

A checkpoint is written after every epoch and a re-run picks up from the last one. Long
fine-tunes get interrupted, and losing an hour to a dropped session is a choice rather
than a fact.

notebooks/train_colab.ipynb runs the same thing on a free T4. On Apple silicon the
trainer uses MPS, but MPS has no deterministic implementation for one op in the backward
pass, and CUDA's memory-efficient attention backward is not deterministic either, so no
training run here is bit-reproducible. Reruns with the same seed land close, not
identical; the served ONNX model is deterministic.

Arabic sentiment: translated training data

The PRD's plan for Arabic sentiment is translate-train: machine-translated Arabic copies
of the English sentiment sets (PRD 7), with ArSAS as "a sanity check for the
cross-lingual head".

corpora.translate_sentiment() builds that set. It translates Financial PhraseBank and
FiQA with Helsinki-NLP/opus-mt-en-ar, and each Arabic sentence keeps the label the
English annotators gave its original, so every label is still a person's; only the
wording is generated. The English original is kept beside each translation. The result
is cached in data/raw/translated_sentiment_ar/ and is resumable if interrupted.

Two rules keep it honest:

- Training only. Arabic sentiment is always scored on ArSAS, which is native Arabic. A
  model tested on translations would only show it agrees with the translator.
- Only alongside its original. A translation joins training only when its English
  sentence is in the training split. Otherwise the shared encoder would have seen an
  Arabic copy of an English dev or test item, and the English score would rise for the
  wrong reason.

The first trained model had no Arabic sentiment in training at all, so its Arabic score
(0.595) is zero-shot. That is the baseline: the gap between it and a translate-train run
is what translation buys. Translations are literal in places ("cut its guidance" comes
out with the everyday word for "cut"), and nobody on the project reads Arabic, so report
the result as translate-train, never as native Arabic data.

Reference backend
-----------------

scoring.Heuristic is a dependency-free lexical model implementing the full contract. It
is not the deliverable model and its scores are not the PRD's accuracy numbers. It
exists so the API, rules, provenance, monitoring and acceptance machinery could be built
and tested before a model finished training, and so a fail-open response is still
explainable.

It handles the PRD's own motivating example, which general-purpose sentiment models get
wrong:

    Liabilities fell sharply in the third quarter.  → positive
    Revenue fell sharply in the third quarter.      → negative

Polarity is composed from a direction and a subject rather than read off a word list.
Switch to a trained artefact with backend = "onnx" in configs/default.toml; nothing else
changes.

Commands
--------

    fintext score "Revenue missed consensus" --compact
    fintext batch data/samples/score_items.jsonl -o reports/scored.jsonl
    fintext replay data/samples/provenance_replay.jsonl     # PRD 8.2
    fintext canary                                          # PRD 8.4
    fintext accept data/samples/measured_metrics.json       # PRD 8, exits non-zero on failure
    fintext bench --gate                                    # PRD 6.6 / 8.3 SLO gate
    fintext catalogue --licence-log docs/LICENCE_LOG.txt     # PRD 7
    fintext review sample --days 7 --size 50                # PRD 8.4, draw the weekly batch
    fintext review precision --metrics reports/metrics.json # feed the gate a measured number
    fintext review export -o reports/review.csv             # spreadsheet, per PRD 5.3
    fintext corpus --summary-only                           # what the corpus would contain
    fintext corpus -o reports/corpus.jsonl                  # write it, licence permitting
    fintext data                                            # write every source to data/sources
    fintext edgar --per-query 250                           # rebuild the provenance index from SEC
    fintext train --epochs 3                                # fine-tune on the declared sources
    fintext registry list | promote <v> | shadow <v>
    fintext serve [--asgi]

Configuration
-------------

configs/default.toml is one flat file, one field per line, overridable from the
environment:

    FINTEXT_TOXICITY_EN=0.45 fintext serve

The config is hashed at start-up and the digest travels on every response as
lineage.config_digest, so a flag traces to the exact thresholds that produced it,
including one set by an environment override during an incident.

Acceptance targets are deliberately not here: they are PRD constants, not per-deployment
settings, and live in evaluation.py. Validation refuses configurations that contradict
the PRD: a token cap above 256, a trading budget above the timeout, a profile that is
both fail-open and fail-closed, a non-positive corroboration window.

Degradation
-----------

  Profile             On failure                                  Why
  ------------------- ------------------------------------------- -----------------------------------------------------------
  trading, research   200 with every capability marked degraded   An outage removes the gate rather than halting trading
  compliance          503                                         The queue backs up rather than passing unscreened content

Chosen from config, never from the caller. A failed warm-up does not hold a replica out
of rotation: removing every replica turns a degraded service into a dead one. The
replica comes up, reports ready_degraded, and lets the per-request policy decide.

Testing
-------

    make test    # 625 tests
    make check   # lint, typecheck, tests, SLO gate

tests/test_prd.py is the executable summary of what the service promises: one test per
PRD commitment that is a product decision rather than an implementation detail. The rest
runs against the reference backend on a virtual clock, so window expiry across hours of
simulated time is asserted in milliseconds.

Not built
---------

- gRPC. The PRD's week-four cut list names it as the first thing to drop if the week
  slips (PRD 10), and HTTP covers every consumer in the pilot.
- Ingestion schedulers. The EDGAR/GDELT/IR pullers that materialise the primary index.
  The index format and reader are here; the pullers are not.

Docs
----

- ARCHITECTURE.txt: request path, state model, why the stage order is what it is
- MODEL_CARD.txt: intended use, limitations, what is gated and what is only reported
- RUNBOOK.txt: alerts, promotion, rollback, lexicon and threshold changes, auditing a
  flag
- LICENCE_LOG.txt: generated from the catalogue, checked in CI so it cannot drift
- EVALUATION_REPORT.txt: every gate with its interval, why each failure happened, and
  future work
- ARABIC_TOXICITY.txt: what limits Arabic toxicity
- INTEGRATING_A_RUN.txt: the checks a new Colab run goes through before it is registered
- MODEL_PACK.txt: the trained weights: contents, checksums, install, licence
- PROJECT_REPORT.txt: the project from start to finish, in order, including what went
  wrong
- INSTALL.txt: the packages, and installing, serving and retraining on a laptop or
  server

Status
------

MVP, complete at shared-xlmr.v4.1: trained, evaluated against every PRD gate with
intervals, checked for leakage and serving parity, and documented. It is not accepted (5
of 14 gates pass), registered but not promoted, and no output is approved for unattended
action. The remaining work is listed in the evaluation report's Future work section.
