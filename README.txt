FinTextIntelligence
===================

Scores financial text in English and Arabic for sentiment, toxicity, misleading content
and provenance (is the claim backed by an SEC filing?), and turns the scores into a
buy / sell / review signal. One XLM-R model with four heads, plus rule layers. Runs on CPU.

Results (shared-xlmr.v4.1, frozen test set)
-------------------------------------------

  Gate                          v4.1        Target
  ----------------------------  ----------  --------  ----
  English sentiment F1          0.807       0.80      pass
  English toxicity AUROC        0.953       0.95      pass
  English misleading F1         0.807       0.80      pass
  Provenance precision          72/72       0.95      pass
  Claim-extraction recall       0.934       0.85      pass
  Arabic toxicity AUROC         0.909       0.92      fail
  Arabic toxicity F1            0.710       0.80      fail
  English toxicity F1           0.793       0.85      fail
  EN-AR toxicity F1 gap         0.083       0.07      fail
  p95 latency (CPU)             51-61 ms    50 ms     fail
  Throughput per replica        27-30 rps   100       fail

5 pass, 6 fail, 3 need a live pilot, so the release is not accepted. Arabic sentiment
(not gated) is 0.739. Details and future work: docs/EVALUATION_REPORT.txt.

Quick start
-----------

    pip install -e '.[dev]'
    make test     # 625 tests
    make serve    # http://127.0.0.1:8080/ui

This uses a simple rule-based backend. For the trained model, download
FinTextIntelligence-model-v4.1.zip (2.2 GB, non-commercial use only, please don't pass
it on) from
https://drive.google.com/drive/folders/1LZ11Yt3gAboU3hi0or40qj-Xq79aTPsz?usp=sharing
and run from the project folder:

    unzip FinTextIntelligence-model-v4.1.zip -d .
    pip install -e '.[dev,onnx]'
    fintext --config configs/release_v4.1.toml serve

Always use configs/release_v4.1.toml with the trained model. Step-by-step instructions
are in QUICKSTART.txt.

Good to know
------------

- The service only reports. Nothing is blocked automatically; flagged items go to a
  person ("Waiting on a person" tab), and their decisions become training labels.
- The signal gives a direction only, not size or timing.
- Known issues: ordinary company news is sometimes flagged as misleading, and abusive
  posts can produce SELL because the gate ignores toxicity.
- Training runs on Colab (notebooks/train_colab.ipynb); see docs/INSTALL.txt.

Docs
----

  QUICKSTART.txt              install and run
  USE_AND_TEST_CASES.txt      use cases, test cases and results
  docs/INSTALL.txt            laptop, server and retraining setup
  docs/ARCHITECTURE.txt       how the service works
  docs/MODEL_CARD.txt         intended use and limitations
  docs/EVALUATION_REPORT.txt  all results and future work
  docs/PROJECT_REPORT.txt     how the project went
  docs/paper/                 project paper (PDF)
