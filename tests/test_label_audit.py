"""The label-noise audit: sampling, blinding and scoring."""

from __future__ import annotations

import csv
import json

import pytest

from fintext import label_audit


@pytest.fixture
def rows():
    def block(prefix, n, rate):
        return [
            {
                "text": f"{prefix}{i}",
                "language": "ar",
                "corpus": "osact5",
                "labels": {"toxicity": float(i < n * rate)},
            }
            for i in range(n)
        ]

    other = [
        {"text": "en", "language": "en", "corpus": "jigsaw_toxic", "labels": {"toxicity": 1.0}}
    ]
    return {
        "train": block("train-", 500, 0.4),
        "dev": block("d", 300, 0.4),
        "test": block("t", 300, 0.4) + other,
    }


def test_sample_is_held_out_balanced_and_from_one_corpus(rows):
    sample = label_audit.draw_sample(rows, "osact5", per_class=50)
    assert len(sample) == 100
    assert sum(s["original"] for s in sample) == 50
    assert not any(s["text"].startswith("train-") for s in sample)
    assert {s["corpus"] for s in sample} == {"osact5"}
    assert sample[0]["base_rate"] == pytest.approx(0.4)


def test_the_sample_is_reproducible(rows):
    a = label_audit.draw_sample(rows, "osact5", per_class=20)
    b = label_audit.draw_sample(rows, "osact5", per_class=20)
    assert a == b


def test_the_sheet_carries_no_labels(rows, tmp_path):
    sample = label_audit.draw_sample(rows, "osact5", per_class=10)
    sheet, key = label_audit.write_audit(sample, tmp_path)
    content = sheet.read_text(encoding="utf-8-sig")
    assert "original" not in content and "base_rate" not in content
    with sheet.open(encoding="utf-8-sig") as handle:
        assert all(not r["offensive"] for r in csv.DictReader(handle))
    assert len(json.loads(key.read_text())["original"]) == 20


def _fill(sheet, answers):
    with sheet.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r.update(answers(r["item"]))
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_audit.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_perfect_agreement_is_zero_noise(rows, tmp_path):
    sample = label_audit.draw_sample(rows, "osact5", per_class=25)
    sheet, key = label_audit.write_audit(sample, tmp_path)
    truth = {s["item"]: s["original"] for s in sample}
    _fill(sheet, lambda item: {"offensive": str(truth[item])})
    result = label_audit.score_audit(sheet, key)
    assert result.agreement == 1.0
    assert result.kappa == pytest.approx(1.0)
    assert result.noise_rate == 0.0


def test_noise_is_reweighted_to_the_base_rate(rows, tmp_path):
    sample = label_audit.draw_sample(rows, "osact5", per_class=50)
    sheet, key = label_audit.write_audit(sample, tmp_path)
    truth = {s["item"]: s["original"] for s in sample}
    offensive = sorted(i for i, v in truth.items() if v == 1)
    flip = set(offensive[:10])  # 20% of offensive rows read as clean; clean rows all agree
    _fill(sheet, lambda item: {"offensive": str(0 if item in flip else truth[item])})
    result = label_audit.score_audit(sheet, key)
    assert result.flipped_offensive == (10, 50)
    assert result.flipped_clean == (0, 50)
    # 40% of the corpus is offensive, so 20% of 40% = 8% overall, not 10%.
    assert result.noise_rate == pytest.approx(0.08)
    assert "8.0%" in result.summary()


def test_unsure_and_blank_rows_are_counted_but_not_scored(rows, tmp_path):
    sample = label_audit.draw_sample(rows, "osact5", per_class=10)
    sheet, key = label_audit.write_audit(sample, tmp_path)
    truth = {s["item"]: s["original"] for s in sample}
    items = sorted(truth)
    _fill(
        sheet,
        lambda item: {"unsure": "1"}
        if item == items[0]
        else {}
        if item == items[1]
        else {"offensive": str(truth[item])},
    )
    result = label_audit.score_audit(sheet, key)
    assert (result.annotated, result.unsure, result.unanswered) == (18, 1, 1)


def test_wilson_interval_is_sane():
    lo, hi = label_audit.wilson(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = label_audit.wilson(50, 100)
    assert lo < 0.5 < hi


def test_label_ceiling():
    assert label_audit.label_ceiling_f1(0.4, 0.0, 0.0) == 1.0
    # 20% of offensive labels wrong, 10% of clean labels wrong, 36% offensive: tp .288, fn .072,
    # fp .064 -> 2*.288 / (2*.288 + .136)
    assert label_audit.label_ceiling_f1(0.36, 0.2, 0.1) == pytest.approx(0.576 / 0.712)
