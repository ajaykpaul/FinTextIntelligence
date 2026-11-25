"""Measuring how noisy a toxicity dataset's labels are, by relabelling a sample."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHEET_FIELDS = ("item", "text", "offensive", "unsure", "note")
"""`offensive` takes 1 or 0. `unsure` is 1 when the annotator can't decide; those
rows are reported but left out of the rate, as the shared task did with ties."""


def draw_sample(
    rows: dict[str, list[dict[str, Any]]],
    corpus: str,
    *,
    per_class: int = 100,
    seed: int = 20261118,
) -> list[dict[str, Any]]:
    """`per_class` held-out rows of each label from one corpus, shuffled together."""
    held = [
        r
        for r in rows["dev"] + rows["test"]
        if r.get("corpus") == corpus and r.get("labels", {}).get("toxicity") is not None
    ]
    if not held:
        raise ValueError(f"no held-out toxicity rows from {corpus!r}")
    rng = random.Random(f"{seed}:{corpus}")
    chosen: list[dict[str, Any]] = []
    for label in (1, 0):
        pool = [r for r in held if int(r["labels"]["toxicity"]) == label]
        if len(pool) < per_class:
            raise ValueError(f"{corpus} has only {len(pool)} held-out rows labelled {label}")
        chosen += rng.sample(pool, per_class)
    rng.shuffle(chosen)
    base_rate = sum(int(r["labels"]["toxicity"]) for r in held) / len(held)
    return [
        {
            "item": f"{corpus}-{i:03d}",
            "text": r["text"],
            "original": int(r["labels"]["toxicity"]),
            "corpus": corpus,
            "base_rate": base_rate,
        }
        for i, r in enumerate(chosen, 1)
    ]


def write_audit(sample: list[dict[str, Any]], folder: str | Path) -> tuple[Path, Path]:
    """The blank sheet for the annotator, and the key, as two separate files."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    sheet, key = out / "sheet.csv", out / "key.json"
    # utf-8-sig so Excel and Google Sheets open the Arabic correctly.
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for s in sample:
            writer.writerow({"item": s["item"], "text": s["text"]})
    key.write_text(
        json.dumps(
            {
                "corpus": sample[0]["corpus"],
                "base_rate": sample[0]["base_rate"],
                "original": {s["item"]: s["original"] for s in sample},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return sheet, key


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; sane at small n and near 0 or 1."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def label_ceiling_f1(base_rate: float, flip_offensive: float, flip_clean: float) -> float:
    """F1 against noisy labels for a model that is right about every row."""
    tp = base_rate * (1 - flip_offensive)
    fn = base_rate * flip_offensive  # labelled offensive, model (rightly) says clean
    fp = (1 - base_rate) * flip_clean  # labelled clean, model (rightly) flags it
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class AuditResult:
    corpus: str
    annotated: int
    unsure: int
    unanswered: int
    agreement: float
    kappa: float
    flipped_offensive: tuple[int, int]
    flipped_clean: tuple[int, int]
    base_rate: float
    noise_rate: float
    noise_interval: tuple[float, float]
    label_ceiling_f1: float

    def summary(self) -> str:
        off_n, off_d = self.flipped_offensive
        cl_n, cl_d = self.flipped_clean
        lo_off, hi_off = wilson(off_n, off_d)
        lo_cl, hi_cl = wilson(cl_n, cl_d)
        lo, hi = self.noise_interval
        return "\n".join(
            [
                f"{self.corpus}: {self.annotated} relabelled, {self.unsure} unsure, "
                f"{self.unanswered} left blank",
                f"  raw agreement with the original labels  {self.agreement:.1%}",
                f"  Cohen's kappa                           {self.kappa:.2f}",
                f"  labelled offensive, reads as not        {off_n}/{off_d} "
                f"= {off_n / max(off_d, 1):.1%}  (95% CI {lo_off:.1%}-{hi_off:.1%})",
                f"  labelled not, reads as offensive        {cl_n}/{cl_d} "
                f"= {cl_n / max(cl_d, 1):.1%}  (95% CI {lo_cl:.1%}-{hi_cl:.1%})",
                f"  estimated label noise in the corpus     {self.noise_rate:.1%}  "
                f"(95% CI {lo:.1%}-{hi:.1%}; base rate {self.base_rate:.0%} offensive)",
                f"  F1 a perfect model scores on these labels {self.label_ceiling_f1:.3f}",
            ]
        )


def score_audit(sheet: str | Path, key: str | Path) -> AuditResult:
    """Compare a filled-in sheet with the hidden original labels."""
    recorded = json.loads(Path(key).read_text(encoding="utf-8"))
    original: dict[str, int] = recorded["original"]
    base_rate = float(recorded["base_rate"])

    pairs: list[tuple[int, int]] = []
    unsure = unanswered = 0
    with Path(sheet).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = (row.get("item") or "").strip()
            if item not in original:
                continue
            if (row.get("unsure") or "").strip() in ("1", "y", "yes", "true"):
                unsure += 1
                continue
            answer = (row.get("offensive") or "").strip().lower()
            if answer in ("1", "y", "yes", "true", "off"):
                pairs.append((original[item], 1))
            elif answer in ("0", "n", "no", "false", "not"):
                pairs.append((original[item], 0))
            else:
                unanswered += 1
    if not pairs:
        raise ValueError("no answered rows in the sheet")

    was, now = [p[0] for p in pairs], [p[1] for p in pairs]
    off = [(w, n) for w, n in pairs if w == 1]
    clean = [(w, n) for w, n in pairs if w == 0]
    off_flip = sum(n == 0 for _, n in off)
    clean_flip = sum(n == 1 for _, n in clean)
    rate_off = off_flip / len(off) if off else 0.0
    rate_clean = clean_flip / len(clean) if clean else 0.0
    noise = base_rate * rate_off + (1 - base_rate) * rate_clean

    # Interval for the reweighted rate: the two strata are independent, so the variances add
    # (normal approximation; fine at a hundred rows a stratum).
    variance = (base_rate**2) * rate_off * (1 - rate_off) / max(len(off), 1) + (
        (1 - base_rate) ** 2
    ) * rate_clean * (1 - rate_clean) / max(len(clean), 1)
    half = 1.96 * math.sqrt(variance)

    return AuditResult(
        label_ceiling_f1=label_ceiling_f1(base_rate, rate_off, rate_clean),
        corpus=recorded.get("corpus", "?"),
        annotated=len(pairs),
        unsure=unsure,
        unanswered=unanswered,
        agreement=sum(w == n for w, n in pairs) / len(pairs),
        kappa=cohen_kappa(was, now),
        flipped_offensive=(off_flip, len(off)),
        flipped_clean=(clean_flip, len(clean)),
        base_rate=base_rate,
        noise_rate=noise,
        noise_interval=(max(0.0, noise - half), min(1.0, noise + half)),
    )
