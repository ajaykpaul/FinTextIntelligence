"""The per-source toxicity breakdown, with a stand-in for the model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fintext import diagnose


def _row(text, language, corpus, toxic):
    return {"text": text, "language": language, "corpus": corpus, "labels": {"toxicity": toxic}}


@pytest.fixture
def rows():
    # osact4: perfectly separable. osact5: the model gets it backwards.
    def block(prefix, language, corpus, n):
        return [_row(f"{prefix}{i}", language, corpus, float(i % 2)) for i in range(n)]

    return {
        "train": block("tr4-", "ar", "osact4", 40) + block("tr5-", "ar", "osact5", 40),
        "dev": block("d4-", "ar", "osact4", 20) + block("d5-", "ar", "osact5", 20),
        "test": block("t4-", "ar", "osact4", 20) + block("t5-", "ar", "osact5", 20),
    }


@pytest.fixture
def fake_scores(monkeypatch):
    def score(model, tokenizer, rows, config, device=None, *, desc=""):
        out = []
        for r in rows:
            toxic = int(r["text"].rsplit("-", 1)[1]) % 2
            backwards = "5-" in r["text"]
            out.append(0.2 if toxic == backwards else 0.8)
        return out

    monkeypatch.setattr(diagnose, "score_toxicity", score)


CONFIG = SimpleNamespace(recall_floor=0.9, precision_floor=0.9)


def test_each_source_gets_its_own_line(rows, fake_scores):
    table, _ = diagnose.toxicity_by_source(None, None, rows, CONFIG)
    by_corpus = {t["corpus"]: t for t in table}
    assert set(by_corpus) == {"osact4", "osact5"}
    assert by_corpus["osact4"]["rows"] == 40
    assert by_corpus["osact4"]["auroc_held"] == 1.0
    assert by_corpus["osact5"]["auroc_held"] == 0.0
    assert by_corpus["osact4"]["auroc_train"] == 1.0


def test_worst_errors_come_from_the_source_that_is_wrong(rows, fake_scores):
    _, scored = diagnose.toxicity_by_source(None, None, rows, CONFIG)
    errors = diagnose.worst_errors(scored, language="ar", count=5)
    assert all(corpus == "osact5" for _, corpus, _ in errors["false_positives"])
    assert all(corpus == "osact5" for _, corpus, _ in errors["false_negatives"])
    assert errors["false_positives"][0][0] == 0.8


def test_the_table_prints(rows, fake_scores, capsys):
    table, _ = diagnose.toxicity_by_source(None, None, rows, CONFIG)
    diagnose.print_table(table)
    assert "osact5" in capsys.readouterr().out
