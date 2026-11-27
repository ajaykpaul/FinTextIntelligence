FinTextIntelligence
===================

Sentiment, toxicity, misleading-content triage and provenance signals for trading and
compliance, in English and Modern Standard Arabic, behind one low-latency service.
Implements PRD v1.5.

Project summary
---------------

Problem. Trading desks and compliance teams need four signals on financial text, fast
enough for the trading hot path: sentiment, toxicity, likely-misleading content, and
whether a claim is corroborated by a primary source. The text is in English and Arabic.

Approach. One XLM-R base encoder with four heads, trained only on public data whose
licences are checked in code. Around the model: a high-severity lexicon floor,
SEC-derived scam-pattern rules, and an SEC EDGAR provenance check. A trading gate turns
the scores into buy / sell / review. The service reports; the policy layer decides.

Result. The current release, shared-xlmr.v4.1, is not accepted: of 14 gates, 5 pass, 6
fail and 3 need a pilot. All numbers are on a frozen test split, thresholds tuned on dev.

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

Arabic sentiment is reported, not gated: it rose from 0.490 to 0.739 once native Arabic
tweets joined training, with no leakage into test. Key findings: a different seed moves the numbers as
much as the first fixes did, so setups are compared over several seeds; the Arabic F1
target may be capped by noisy labels (a label audit is designed); and int8 changed 4–9%
of decisions on ARM and x86, so the service runs fp32. What's left is in the Future work
section of docs/EVALUATION_REPORT.txt.

Quick start
-----------

New to the project? QUICKSTART.txt has step-by-step install instructions. In short:

    pip install -e '.[dev]'
    make test          # 625 tests, no network, no model needed
    make demo          # score built-in examples with the rule-based reference backend
    make serve         # http://127.0.0.1:8080/ui

With the trained model: download FinTextIntelligence-model-v4.1.zip (2.2 GB,
non-commercial use only, please don't pass it on; see docs/MODEL_PACK.txt) from
https://drive.google.com/drive/folders/1LZ11Yt3gAboU3hi0or40qj-Xq79aTPsz?usp=sharing

    unzip FinTextIntelligence-model-v4.1.zip -d .      # from the project folder
    pip install -e '.[dev,onnx]'
    fintext --config configs/release_v4.1.toml serve   # always use the release config

Without the pack, everything runs on the heuristic reference backend (standard library
only). It is not the deliverable model, but it gets the PRD's example right: "Liabilities
fell sharply" is positive, "Revenue fell sharply" is negative.

What it does
------------

  Capability            Output                                           Languages    Gated
  --------------------- ------------------------------------------------ ------------ ----------------
  Financial sentiment   positive / negative / neutral                    EN, MSA AR   English only
  Toxicity              flag + hate / harassment / profanity / threat    EN, MSA AR   Both
  Misleading content    triage flag + scam-pattern rules                 EN           English
  Provenance            pending / corroborated / uncorroborated / n/a    EN           Precision ≥ 0.95
  Trading gate          buy / sell / review, and how urgently            EN, AR       -
  Approvals, review     a person accepts, rejects or judges each flag    EN, AR       Precision ≥ 0.90

Guarantees built into the types: nothing asserts falsity (misleading items are only
"needs review", provenance carries asserts_falsity: false); "pending" and
"uncorroborated" are separate states; and no score or state blocks content or acts.

Web UI (/ui). Paste a headline or a bare link (the article is fetched; private
addresses are refused; paywalled sites return 502 with the URL's headline offered
instead). Tabs: Score, Waiting on a person, Weekly review. React app in web/, built into
src/fintext/static/ (make ui-install, ui-dev, ui-build); no Node needed to serve it.

Trading gate (POST /v1/signal). Direction only, never size or timing. Review comes as
"now" or "routine", which cuts the human pile from ~70% to ~9% of a stream. A failed score
makes no call (direction null), and an unconfirmed claim lowers conviction but never
blocks. Known gap: the gate ignores toxicity, so abusive negative posts can give SELL.

Human loops. Waiting on a person (/v1/approvals) holds gated calls; accepting means
naming buy or sell. Weekly review (/v1/review/*) samples flags and produces the measured
precision gate. Human verdicts become training labels; model outputs never do. Training
on captured text stays blocked by a licence gate in catalogue.py; PII is scrubbed first.

Provenance. fintext edgar builds the index from SEC EDGAR (free, no key; set
FINTEXT_SEC_CONTACT for the SEC's contact rule). A filing must mention something specific
to the claim (a name or amount), and the index supplies the company vocabulary.

Degradation. Trading and research fail open (200, marked degraded); compliance fails
closed (503). Set from config, never by the caller.

Training
--------

    pip install -e '.[train]'
    fintext train --epochs 3                    # or notebooks/train_colab.ipynb on Colab
    fintext import-model ~/Downloads/<run>.zip  # bring a Colab run back
    fintext evaluate artifacts/models/<run>     # re-score under the frozen protocol

Splits are fixed and fingerprinted: thresholds tuned on dev, numbers reported on test.
Arabic sentiment trains on machine-translated PhraseBank/FiQA (only where the English
original is in train) plus ArSarcasm-v2, and is tested on native ArSAS. A HuggingFace
token is optional and never stored in the project (hf auth login, or Colab Secrets).
Steps for a new run: docs/INTEGRATING_A_RUN.txt.

Commands and configuration
--------------------------

    fintext score "text" | batch FILE | demo | serve [--asgi]
    fintext replay | canary | accept FILE | bench --gate
    fintext review sample | precision | export     fintext corpus [--summary-only]
    fintext data | edgar | train | registry list | promote <v> | shadow <v>
    fintext catalogue --licence-log docs/LICENCE_LOG.txt

configs/default.toml is the heuristic backend's config; configs/release_v4.1.toml holds
the trained model's thresholds. Any field can be overridden from the environment
(FINTEXT_TOXICITY_EN=0.45). The config digest is on every response for traceability.

Layout and testing
------------------

src/fintext/ has one module per concern (preprocess, scoring, lexicon, scam_rules,
provenance, pipeline, signal, approvals, review, corpus, training, evaluation, server,
cli). web/ is the UI. tests/test_prd.py has one test per PRD commitment. Not built:
gRPC (first on the PRD's cut list) and scheduled ingestion pullers.

    make test    # 625 tests        make check   # lint, typecheck, tests, SLO gate

Docs
----

In the project root: QUICKSTART.txt (install and run), USE_AND_TEST_CASES.txt (use
cases, test cases, last results, known issues). In docs/: INSTALL, ARCHITECTURE,
MODEL_CARD, EVALUATION_REPORT (every gate and future work), PROJECT_REPORT (the story,
including what went wrong), ARABIC_TOXICITY, LABEL_AUDIT, INTEGRATING_A_RUN, MODEL_PACK,
RUNBOOK, LICENCE_LOG (all .txt), and paper/FinTextIntelligence.pdf (the project paper,
ACL format).
