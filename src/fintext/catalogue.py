"""The data catalogue and licence gate (PRD 7)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRAIN, EVAL, BACKTEST, INDEX, RULES = "train", "eval", "backtest", "provenance_index", "rule_source"
CLEARED, CONDITIONAL, PENDING, REJECTED = (
    "cleared",
    "cleared_with_conditions",
    "pending",
    "rejected",
)


class LicenceError(Exception):
    pass


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    url: str
    languages: tuple[str, ...]
    tasks: tuple[str, ...]
    uses: tuple[str, ...]
    licence: str
    status: str
    redistributable: bool = False
    internal: bool = False
    conditions: tuple[str, ...] = ()
    # A named fallback, so a late licence withdrawal has a rehearsed answer.
    fallback: str | None = None
    notes: str = ""


CATALOGUE: tuple[Dataset, ...] = (
    Dataset(
        "production_capture",
        "Production capture (items scored by this service)",
        "internal",
        ("en", "ar"),
        ("sentiment", "toxicity", "misleading"),
        (TRAIN, EVAL),
        "none: third-party text submitted by callers",
        PENDING,
        redistributable=False,
        internal=True,
        conditions=(
            "PII-scrubbed at preprocessing before it is ever written",
            "only human-adjudicated items are used as training targets",
            "clear with legal before any fine-tune: the text is other people's",
        ),
        notes=(
            "Built by corpus.py from the decision log plus the two human review "
            "loops. Deliberately left pending: whether you may train on articles "
            "your users pasted is a legal question, not an engineering one."
        ),
    ),
    Dataset(
        "financial_phrasebank",
        "Financial PhraseBank",
        "https://huggingface.co/datasets/takala/financial_phrasebank",
        ("en",),
        ("sentiment",),
        (TRAIN, EVAL),
        "CC BY-NC-SA 3.0",
        CONDITIONAL,
        conditions=("non-commercial research only; re-verify before any commercial pilot",),
        fallback="fiqa_2018",
        notes="EN sentiment anchor.",
    ),
    Dataset(
        "fiqa_2018",
        "FiQA 2018 Task 1",
        "https://sites.google.com/view/fiqa",
        ("en",),
        ("sentiment",),
        (TRAIN, EVAL),
        "Open shared-task data",
        CLEARED,
        fallback="sec_edgar",
        notes="EN sentiment augmentation.",
    ),
    Dataset(
        "sec_edgar",
        "SEC EDGAR full-text search",
        "https://www.sec.gov/edgar/search",
        ("en",),
        ("sentiment", "provenance"),
        (TRAIN, EVAL, INDEX),
        "US Government public domain",
        CLEARED,
        redistributable=True,
        conditions=("respect the published fair-access rate limits and user-agent policy",),
        notes="Filing excerpts, weak labels, and the primary-document index.",
    ),
    Dataset(
        "gdelt",
        "GDELT raw data files",
        "https://www.gdeltproject.org/data.html",
        ("en", "ar"),
        ("sentiment", "misleading"),
        (TRAIN, EVAL, BACKTEST),
        "Open, attribution requested",
        CLEARED,
        notes="EN and AR headlines; distribution realism and validated weak labels.",
    ),
    Dataset(
        "issuer_ir",
        "Issuer investor-relations pages",
        "https://www.bankfab.com/en-ae/investor-relations",
        ("en",),
        ("sentiment", "provenance"),
        (EVAL, INDEX),
        "Public web pages",
        CONDITIONAL,
        conditions=("robots.txt respected", "hand-labelled slice not redistributed"),
        notes="Hand-labelled EN evaluation set and provenance index.",
    ),
    Dataset(
        "jigsaw_toxic",
        "Jigsaw Toxic Comment Classification",
        "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge",
        ("en",),
        ("toxicity",),
        (TRAIN, EVAL),
        "Kaggle competition data-use terms",
        CONDITIONAL,
        conditions=("competition rules bind derived artefacts; recorded in the model card",),
        fallback="wikipedia_talk",
        notes="EN toxicity core training; the frozen AUROC split.",
    ),
    Dataset(
        "wikipedia_talk",
        "Wikipedia Talk Labels: Toxicity",
        "https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973",
        ("en",),
        ("toxicity",),
        (TRAIN, EVAL),
        "CC0 / open figshare release",
        CLEARED,
        redistributable=True,
        notes="EN toxicity augmentation.",
    ),
    Dataset(
        "hatexplain",
        "HateXplain",
        "https://github.com/hate-alert/HateXplain",
        ("en",),
        ("toxicity",),
        (TRAIN, EVAL),
        "Repository licence recorded in the log",
        CLEARED,
        notes="Subtype tags and the human rationales used to validate explanations.",
    ),
    Dataset(
        "osact4",
        "OSACT4 shared task (2020)",
        "https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/",
        ("ar",),
        ("toxicity",),
        (TRAIN, EVAL),
        "Shared-task research terms",
        CLEARED,
        notes="The Arabic anchor: native-speaker labels carry every firm Arabic commitment.",
    ),
    Dataset(
        "osact5",
        "OSACT5 shared task (2022)",
        "https://aclanthology.org/2022.osact-1.20/",
        ("ar",),
        ("toxicity",),
        (TRAIN, EVAL),
        "Shared-task research terms",
        CLEARED,
        notes="AR fine-grained hate speech subtypes.",
    ),
    Dataset(
        "arsas",
        "ArSAS",
        "https://homepages.inf.ed.ac.uk/wmagdy/resources.htm",
        ("ar",),
        ("sentiment",),
        (EVAL,),
        "Research terms",
        CLEARED,
        notes="AR sentiment sanity check. Reported, never gated (PRD 1.1).",
    ),
    Dataset(
        "translated_sentiment_ar",
        "Financial PhraseBank and FiQA, machine-translated to Arabic",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-ar",
        ("ar",),
        ("sentiment",),
        (TRAIN,),
        "Inherits CC BY-NC-SA 3.0 (PhraseBank) and the FiQA shared-task terms",
        CONDITIONAL,
        conditions=(
            "derived from Financial PhraseBank and FiQA; their terms apply",
            "machine-translated, not native Arabic; reported as translate-train",
            "training only: Arabic sentiment is evaluated on ArSAS",
        ),
        notes=(
            "PRD 7: machine-translated Arabic training copies are derived from "
            "the public English sets and introduce no additional source. Labels "
            "are the English annotators'; only the wording is generated."
        ),
    ),
    Dataset(
        "arsarcasm_v2",
        "ArSarcasm-v2 (WANLP 2021 shared task)",
        "https://github.com/iabufarha/ArSarcasm-v2",
        ("ar",),
        ("sentiment",),
        (TRAIN,),
        "MIT",
        CLEARED,
        conditions=(
            "tweets remain subject to the platform's terms; not redistributed here",
            "training only: Arabic sentiment stays evaluated on ArSAS, which shares no tweet",
        ),
        notes=(
            "Native Arabic sentiment (positive, negative, neutral) on 15,548 tweets, mostly "
            "MSA and Egyptian. Added because translated financial English left the head "
            "unable to tell a neutral tweet from an opinionated one (EVALUATION_REPORT)."
        ),
    ),
    Dataset(
        "lhsab",
        "L-HSAB (Levantine)",
        "https://github.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset",
        ("ar",),
        ("toxicity",),
        (EVAL,),
        "Research licence",
        CONDITIONAL,
        conditions=("evaluation only; dialect probe, not a training source",),
        notes="Dialect slice, reported and not gating.",
    ),
    Dataset(
        "isot_fake_news",
        "ISOT Fake News Dataset",
        "https://onlineacademiccommunity.uvic.ca/isot/",
        ("en",),
        ("misleading",),
        (TRAIN, EVAL),
        "Free research download",
        CLEARED,
        fallback="liar",
        notes="Misleading-content core training.",
    ),
    Dataset(
        "liar",
        "LIAR (PolitiFact statements)",
        "https://aclanthology.org/P17-2067/",
        ("en",),
        ("misleading",),
        (TRAIN, EVAL),
        "Research use",
        CLEARED,
        notes="Short-statement misleading-style training and eval.",
    ),
    Dataset(
        "fakenewsnet",
        "FakeNewsNet",
        "https://github.com/KaiDMML/FakeNewsNet",
        ("en",),
        ("misleading",),
        (TRAIN, EVAL),
        "Fetched via official scripts, per platform terms",
        CONDITIONAL,
        conditions=("hydrate with the official scripts only; never redistribute content",),
        notes="Misleading-content augmentation.",
    ),
    Dataset(
        "pheme",
        "PHEME rumours dataset",
        "https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619",
        ("en",),
        ("misleading", "provenance"),
        (TRAIN, EVAL),
        "Research licence (figshare)",
        CLEARED,
        notes="Rumour versus non-rumour signal.",
    ),
    Dataset(
        "sec_investor_alerts",
        "SEC investor alerts",
        "https://www.investor.gov/introduction-investing/general-resources/news-alerts",
        ("en",),
        ("misleading",),
        (RULES,),
        "US Government public domain",
        CLEARED,
        redistributable=True,
        notes="Source material for the scam-pattern rules.",
    ),
    Dataset(
        "reddit",
        "Reddit (official API)",
        "https://www.reddit.com/dev/api/",
        ("en",),
        ("sentiment", "toxicity"),
        (BACKTEST,),
        "Reddit API terms, research use",
        CONDITIONAL,
        conditions=("never used in training", "never redistributed", "backtest realism only"),
        notes="Excluded from every training run; the catalogue gate enforces it.",
    ),
)

BY_KEY = {dataset.key: dataset for dataset in CATALOGUE}


def get(key: str) -> Dataset:
    if key not in BY_KEY:
        raise LicenceError(f"unknown dataset {key!r}")
    return BY_KEY[key]


def select(
    *, task: str | None = None, use: str | None = None, language: str | None = None
) -> list[Dataset]:
    return [
        d
        for d in CATALOGUE
        if (task is None or task in d.tasks)
        and (use is None or use in d.uses)
        and (language is None or language in d.languages)
    ]


def check_training_sources(keys: list[str] | tuple[str, ...]) -> None:
    """Refuse a training run over sources that do not permit training."""
    problems = []
    for key in keys:
        dataset = get(key)
        if TRAIN not in dataset.uses:
            problems.append(f"{key}: training not permitted (allowed: {', '.join(dataset.uses)})")
        if dataset.status == REJECTED:
            problems.append(f"{key}: licence rejected in review")
        if dataset.status == PENDING:
            problems.append(f"{key}: licence review has not cleared")
    if problems:
        # Report every problem at once: one per run means one failed run each.
        raise LicenceError("training blocked by the licence log:\n  " + "\n  ".join(problems))


def unresolved() -> list[Dataset]:
    return [d for d in CATALOGUE if d.status == PENDING]


def to_json() -> str:
    return json.dumps([asdict(d) for d in CATALOGUE], indent=2, ensure_ascii=False)


def write_licence_log(path: str | Path) -> Path:
    """Write the licence log as plain text for the legal review packet."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Licence log",
        "===========",
        "",
        "Generated from fintext.catalogue. Every source used for training or evaluation",
        "appears here. No private, internal, customer or employee data is used anywhere in",
        "this project (PRD 3.2, 7).",
    ]
    for d in CATALOGUE:
        lines += [
            "",
            d.name,
            "-" * len(d.name),
            f"  URL:           {d.url}",
            f"  Languages:     {', '.join(d.languages)}",
            f"  Tasks:         {', '.join(d.tasks)}",
            f"  Permitted:     {', '.join(d.uses)}",
            f"  Licence:       {d.licence}",
            f"  Status:        {d.status}",
            f"  Redistribute:  {'yes' if d.redistributable else 'no'}",
            f"  Fallback:      {d.fallback or '-'}",
        ]
        if d.conditions:
            lines.append("  Conditions:")
            lines += [f"    - {c}" for c in d.conditions]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
