"""Loading the PRD's declared training sources into rows the trainer accepts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import catalogue
from .credentials import use_huggingface_token
from .progress import enabled as progress_enabled
from .progress import note, progress

logger = logging.getLogger("fintext.corpora")

SENTIMENT = ("positive", "negative", "neutral")

# Financial PhraseBank ships 0=negative, 1=neutral, 2=positive.
PHRASEBANK_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# LIAR2 grades six ways, from pants-on-fire (0) to true (5).
LIAR_MISLEADING = {0, 1, 2}

# FiQA scores sentiment on a continuous -1..1 scale.
FIQA_DEAD_BAND = 0.10

# ArSAS labels: 0 Negative, 1 Neutral, 2 Positive, 3 Mixed.
ARSAS_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

RAW_ROOT = Path(os.environ.get("FINTEXT_RAW_DIR", "data/raw"))
"""Where publisher downloads live. Git-ignored and kept out of every archive."""

OSACT4 = "https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/datasets"
GITHUB = "https://raw.githubusercontent.com"
FIGSHARE = "https://ndownloader.figshare.com/files"
TRANSLATOR = "Helsinki-NLP/opus-mt-en-ar"
"""English to Arabic translation model for the translated sentiment set."""


@dataclass(frozen=True)
class Mirror:
    """What a declared source actually resolves to on the Hub."""

    source: str
    path: str
    config: str | None = None
    split: str = "train"
    note: str = ""


MIRRORS: dict[str, Mirror] = {
    "financial_phrasebank": Mirror(
        "financial_phrasebank",
        "descartes100/enhanced-financial-phrasebank",
        note="parquet mirror; the canonical repo is a loading script",
    ),
    "jigsaw_toxic": Mirror(
        "jigsaw_toxic",
        "tasksource/jigsaw_toxicity",
        note="the Kaggle release, mirrored to parquet",
    ),
    "liar": Mirror(
        "liar",
        "chengxuphd/liar2",
        note="LIAR2, the extended re-release of the LIAR corpus",
    ),
    "fiqa_2018": Mirror(
        "fiqa_2018",
        "TheFinAI/fiqa-sentiment-classification",
        note="the 2018 shared-task data; continuous scores cut into three labels",
    ),
    "arsas": Mirror(
        "arsas",
        "arbml/ArSAS",
        note=(
            "EVAL ONLY: native Arabic, the PRD's sanity check for the "
            "cross-lingual head that the translated set trains"
        ),
    ),
    "arsarcasm_v2": Mirror(
        "arsarcasm_v2",
        "arbml/ArSarcasm_v2",
        split="train+test",
        note="TRAIN ONLY: native Arabic sentiment; both shared-task splits, none in ArSAS",
    ),
    "osact4": Mirror(
        "osact4",
        f"{OSACT4}/OSACT2020-sharedTask-train.txt (+ -dev.txt)",
        split="train+dev",
        note=(
            "the shared-task release from the organisers. The test tweets were "
            "never published, only their labels, so train and dev are all there is"
        ),
    ),
    "osact5": Mirror(
        "osact5",
        "arbml/osact5_hatespeech (parquet)",
        split="train+validation",
        note="offensive, hate (six types), vulgar and violence labels on Arabic tweets",
    ),
    "lhsab": Mirror(
        "lhsab",
        f"{GITHUB}/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset/master/Dataset/L-HSAB",
        note=(
            "EVAL ONLY: the PRD's Levantine dialect probe. Scored as its own "
            "slice, never mixed into the MSA Arabic numbers"
        ),
    ),
    "hatexplain": Mirror(
        "hatexplain",
        f"{GITHUB}/hate-alert/HateXplain/master/Data/dataset.json",
        note="majority of three annotators; only the hate subtype is labelled",
    ),
    "wikipedia_talk": Mirror(
        "wikipedia_talk",
        "figshare article 4563973 (toxicity_annotated_comments.tsv, toxicity_annotations.tsv)",
        note="toxic when most of the ten or so annotators said so",
    ),
    "isot_fake_news": Mirror(
        "isot_fake_news",
        "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip",
        note=(
            "every real article is Reuters, so the dateline is stripped and the "
            "word replaced; left in, the model learns the wire service, not the style"
        ),
    ),
    "fakenewsnet": Mirror(
        "fakenewsnet",
        f"{GITHUB}/KaiDMML/FakeNewsNet/master/dataset/*.csv",
        note="headlines only; the article text needs the platform scripts and keys",
    ),
    "pheme": Mirror(
        "pheme",
        "figshare article 4010619 (pheme-rnr-dataset.tar.bz2)",
        note=(
            "source tweets of each thread. A rumour is unverified, not false, so "
            "it trains the 'needs review' flag and nothing that claims falsity"
        ),
    ),
}

MIRRORS["translated_sentiment_ar"] = Mirror(
    "translated_sentiment_ar",
    f"{TRANSLATOR} over Financial PhraseBank and FiQA",
    split="train only",
    note=(
        "machine translation of human-labelled English sentences; only those whose "
        "English original is in the training split are used"
    ),
)

SUBSTITUTE_MIRRORS = {
    "osact4": Mirror(
        "osact4",
        "textdetox/multilingual_toxicity_dataset",
        split="ar",
        note=(
            "SUBSTITUTION used for the first trained model: a smaller, differently "
            "annotated Arabic toxicity set standing in for OSACT4"
        ),
    ),
}


DEFAULT_LIMITS: dict[str, int | None] = {
    # English toxicity: 24k across three sources instead of 54k.
    "jigsaw_toxic": 10_000,
    "wikipedia_talk": 6_000,
    "hatexplain": 8_000,
    # Misleading content: about 29k across four sources instead of 49k.
    "liar": 10_000,
    "isot_fake_news": 8_000,
    "fakenewsnet": 5_000,
    # Arabic toxicity (OSACT4, OSACT5) and all sentiment stay uncapped: Arabic toxicity is a
    # firm deliverable with little data, and sentiment is the head the trading gate reads.
}
"""Sentiment has about 6,000 labelled rows; the toxicity and misleading sources
together have well over 100,000. Uncapped, they decide what the shared encoder
learns and the sentiment head -- the one the trading gate reads -- trains on
the leftovers. These caps, with the extra loss weight on sentiment in
`TrainingConfig.sentiment_weight`, are the balance. Both are blunt, and both
are recorded in the manifest rather than hidden."""


class CorpusError(Exception):
    """A source could not be loaded as declared."""


def _raw(source: str, name: str, url: str) -> Path:
    """A publisher file, from `data/raw` if present, otherwise downloaded there."""
    path = RAW_ROOT / source / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("%s: downloading %s", source, url)
        request = urllib.request.Request(url, headers={"User-Agent": "FinTextIntelligence"})
        with urllib.request.urlopen(request, timeout=600) as response, path.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    return path


def _cap(rows: list[dict[str, Any]], limit: int | None, source: str) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    return random.Random(f"20240101:{source}").sample(rows, limit)


def _rows_financial_phrasebank(limit: int | None) -> list[dict[str, Any]]:
    data = _load("descartes100/enhanced-financial-phrasebank")
    rows = []
    for record in _take(data, limit):
        # This mirror nests the real record under a "train" key.
        inner = record.get("train", record)
        label = PHRASEBANK_LABELS.get(inner.get("label"))
        text = (inner.get("sentence") or "").strip()
        if label and text:
            rows.append({"text": text, "language": "en", "labels": {"sentiment": label}})
    return rows


def _rows_jigsaw(limit: int | None) -> list[dict[str, Any]]:
    data = _load("tasksource/jigsaw_toxicity")
    rows = []
    for record in _take(data, limit):
        text = (record.get("comment_text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "language": "en",
                "labels": {
                    "toxicity": float(record.get("toxic", 0)),
                    # Order matches SUBTYPE_LABELS: hate, harassment, profanity, threat.
                    "toxicity_subtypes": [
                        float(record.get("identity_hate", 0)),
                        float(record.get("insult", 0)),
                        float(record.get("obscene", 0)),
                        float(record.get("threat", 0)),
                    ],
                },
            }
        )
    return rows


def _rows_liar(limit: int | None) -> list[dict[str, Any]]:
    data = _load("chengxuphd/liar2")
    rows = []
    for record in _take(data, limit):
        text = (record.get("statement") or "").strip()
        label = record.get("label")
        if text and label is not None:
            rows.append(
                {
                    "text": text,
                    "language": "en",
                    "labels": {"misleading": 1.0 if label in LIAR_MISLEADING else 0.0},
                }
            )
    return rows


FIQA_REPO = "TheFinAI/fiqa-sentiment-classification"


def _rows_fiqa(limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test"):
        try:
            data = _load(FIQA_REPO, split=split)
        except Exception as exc:
            # This mirror became gated: a download is refused until you accept its terms on the
            # dataset page, with a token that may read gated repos.
            raise CorpusError(
                f"FiQA could not be loaded ({type(exc).__name__}). The dataset is gated: "
                f"sign in at https://huggingface.co/datasets/{FIQA_REPO}, accept its terms, "
                "and make sure your HuggingFace token can read gated repositories."
            ) from exc
        for record in _take(data, limit):
            text = (record.get("sentence") or "").strip()
            score = record.get("score")
            if not text or score is None:
                continue
            value = float(score)
            if value > FIQA_DEAD_BAND:
                label = "positive"
            elif value < -FIQA_DEAD_BAND:
                label = "negative"
            else:
                label = "neutral"
            rows.append({"text": text, "language": "en", "labels": {"sentiment": label}})
    return rows


def _rows_arsas(limit: int | None) -> list[dict[str, Any]]:
    data = _load("arbml/ArSAS")
    rows = []
    for record in _take(data, limit):
        text = (record.get("Tweet_text") or "").strip()
        label = ARSAS_LABELS.get(record.get("label"))
        if text and label:
            rows.append({"text": text, "language": "ar", "labels": {"sentiment": label}})
    return rows


def _rows_arabic_toxicity(limit: int | None) -> list[dict[str, Any]]:
    data = _load("textdetox/multilingual_toxicity_dataset", split="ar")
    rows = []
    for record in _take(data, limit):
        text = (record.get("text") or "").strip()
        if text:
            # Only the binary head: this set carries no subtype annotation, and inventing one
            # from a single flag would be fabricating labels.
            rows.append(
                {
                    "text": text,
                    "language": "ar",
                    "labels": {"toxicity": float(record.get("toxic", 0))},
                }
            )
    return rows


def _rows_osact4(limit: int | None) -> list[dict[str, Any]]:
    """OSACT4 shared task: Arabic tweets, offensive or not, and hate speech or not."""
    rows = []
    for part in ("train", "dev"):
        name = f"OSACT2020-sharedTask-{part}.txt"
        path = _raw("osact4", name, f"{OSACT4}/{name}")
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 3 or fields[1] not in ("OFF", "NOT_OFF"):
                continue
            text = fields[0].replace("<LF>", " ").strip()
            if text:
                rows.append(
                    {
                        "text": text,
                        "language": "ar",
                        "labels": {
                            "toxicity": float(fields[1] == "OFF"),
                            # Hate is labelled; the other three subtypes are not, and None
                            # leaves them out of the loss.
                            "toxicity_subtypes": [float(fields[2] == "HS"), None, None, None],
                        },
                    }
                )
    return _cap(rows, limit, "osact4")


def _rows_osact5(limit: int | None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    base = "https://huggingface.co/datasets/arbml/osact5_hatespeech/resolve/main/data"
    rows = []
    for name in ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"):
        for record in pq.read_table(_raw("osact5", name, f"{base}/{name}")).to_pylist():
            text = (record.get("tweet") or "").replace("<LF>", " ").strip()
            if text:
                rows.append(
                    {
                        "text": text,
                        "language": "ar",
                        "labels": {
                            "toxicity": float(record.get("off") == "OFF"),
                            # hate, harassment, profanity, threat.
                            "toxicity_subtypes": [
                                float(record.get("hs") != "NOT_HS"),
                                None,
                                float(record.get("vlg") == "VLG"),
                                float(record.get("vio") == "VIO"),
                            ],
                        },
                    }
                )
    return _cap(rows, limit, "osact5")


def _rows_lhsab(limit: int | None) -> list[dict[str, Any]]:
    """Levantine Arabic, for evaluation only, as its own dialect slice."""
    path = _raw("lhsab", "L-HSAB", MIRRORS["lhsab"].path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            text, label = (record.get("Tweet") or "").strip(), (record.get("Class") or "").strip()
            if text and label in ("normal", "abusive", "hate"):
                rows.append(
                    {
                        "text": text,
                        "language": "ar-lev",
                        "labels": {"toxicity": float(label != "normal")},
                    }
                )
    return _cap(rows, limit, "lhsab")


def _rows_hatexplain(limit: int | None) -> list[dict[str, Any]]:
    path = _raw("hatexplain", "dataset.json", MIRRORS["hatexplain"].path)
    rows = []
    for post in json.loads(path.read_text(encoding="utf-8")).values():
        votes = [a["label"] for a in post.get("annotators", [])]
        majority = max(set(votes), key=votes.count) if votes else None
        # Three annotators; a three-way split has no majority and no label.
        if majority is None or votes.count(majority) < 2:
            continue
        text = " ".join(post.get("post_tokens", [])).strip()
        if text:
            rows.append(
                {
                    "text": text,
                    "language": "en",
                    "labels": {
                        "toxicity": float(majority != "normal"),
                        "toxicity_subtypes": [float(majority == "hatespeech"), None, None, None],
                    },
                }
            )
    return _cap(rows, limit, "hatexplain")


def _rows_wikipedia_talk(limit: int | None) -> list[dict[str, Any]]:
    files = {
        "toxicity_annotated_comments.tsv": f"{FIGSHARE}/7394542",
        "toxicity_annotations.tsv": f"{FIGSHARE}/7394539",
    }
    paths = {name: _raw("wikipedia_talk", name, url) for name, url in files.items()}
    votes: dict[str, list[int]] = {}
    with paths["toxicity_annotations.tsv"].open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            votes.setdefault(record["rev_id"], []).append(int(float(record["toxicity"])))
    rows = []
    with paths["toxicity_annotated_comments.tsv"].open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            ballot = votes.get(record["rev_id"])
            text = (
                (record.get("comment") or "")
                .replace("NEWLINE_TOKEN", " ")
                .replace("TAB_TOKEN", " ")
                .strip()
            )
            if ballot and text:
                rows.append(
                    {
                        "text": text,
                        "language": "en",
                        "labels": {"toxicity": float(sum(ballot) / len(ballot) >= 0.5)},
                    }
                )
    return _cap(rows, limit, "wikipedia_talk")


# "WASHINGTON (Reuters) - " at the start of every real ISOT article.
_REUTERS = re.compile(r"^[^()]{0,80}\(Reuters\)\s*[-–]\s*|\(Reuters\)", re.MULTILINE)


def _rows_isot(limit: int | None) -> list[dict[str, Any]]:
    folder = RAW_ROOT / "isot_fake_news"
    if not (folder / "True.csv").exists():
        archive = _raw("isot_fake_news", "News-_dataset.zip", MIRRORS["isot_fake_news"].path)
        with zipfile.ZipFile(archive) as bundle:
            for member in ("Fake.csv", "True.csv"):
                (folder / member).write_bytes(bundle.read(member))
    csv.field_size_limit(10**9)
    rows = []
    for name, misleading in (("Fake.csv", 1.0), ("True.csv", 0.0)):
        with (folder / name).open(encoding="utf-8", errors="replace") as handle:
            for record in csv.DictReader(handle):
                body = _REUTERS.sub("", (record.get("text") or "")[:1500]).strip()
                # "told Reuters" survives the dateline strip in about one real article in seven
                # and almost no fake ones, which is the same shortcut in a different place.
                text = f"{(record.get('title') or '').strip()}. {body}".strip(". ")
                text = re.sub(r"\bReuters\b", "a news agency", text)
                if text:
                    rows.append(
                        {"text": text, "language": "en", "labels": {"misleading": misleading}}
                    )
    return _cap(rows, limit, "isot_fake_news")


def _rows_fakenewsnet(limit: int | None) -> list[dict[str, Any]]:
    base = f"{GITHUB}/KaiDMML/FakeNewsNet/master/dataset"
    rows = []
    for name in ("politifact_fake", "politifact_real", "gossipcop_fake", "gossipcop_real"):
        path = _raw("fakenewsnet", f"{name}.csv", f"{base}/{name}.csv")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for record in csv.DictReader(handle):
                title = (record.get("title") or "").strip()
                if title:
                    rows.append(
                        {
                            "text": title,
                            "language": "en",
                            "labels": {"misleading": float(name.endswith("_fake"))},
                        }
                    )
    return _cap(rows, limit, "fakenewsnet")


def _rows_pheme(limit: int | None) -> list[dict[str, Any]]:
    """Source tweets, read straight out of the archive."""
    archive = _raw("pheme", "pheme-rnr-dataset.tar.bz2", f"{FIGSHARE}/6453753")
    rows = []
    with tarfile.open(archive) as bundle:
        for member in bundle:
            parts = member.name.split("/")
            # .../<event>/<rumours|non-rumours>/<thread>/source-tweet/<id>.json
            if not member.isfile() or len(parts) < 5 or parts[-2] != "source-tweet":
                continue
            kind = parts[-4]
            if kind not in ("rumours", "non-rumours") or not parts[-1].endswith(".json"):
                continue
            handle = bundle.extractfile(member)
            if handle is None:
                continue
            text = (json.loads(handle.read().decode("utf-8")).get("text") or "").strip()
            if text:
                rows.append(
                    {
                        "text": text,
                        "language": "en",
                        "labels": {"misleading": float(kind == "rumours")},
                    }
                )
    # Sorted before sampling: directory order differs between machines.
    rows.sort(key=lambda r: str(r["text"]))
    return _cap(rows, limit, "pheme")


# --------------------------------------------------------------------------
# Arabic sentiment, translated (PRD 7)
# --------------------------------------------------------------------------

TRANSLATED_FROM = ("financial_phrasebank", "fiqa_2018")
TRANSLATED_FILE = "rows.jsonl"


def translate_sentiment(
    *, batch_size: int = 32, limit: int | None = None, translator: str = TRANSLATOR
) -> Path:
    """Machine-translate the English financial sentiment sets into Arabic."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    from .credentials import use_huggingface_token
    from .training import pick_device

    folder = RAW_ROOT / "translated_sentiment_ar"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / TRANSLATED_FILE
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            done = {json.loads(line)["source_text"] for line in handle if line.strip()}

    todo: list[tuple[str, dict[str, Any]]] = []
    for source in TRANSLATED_FROM:
        for row in LOADERS[source](None):
            if row["text"] not in done and "sentiment" in row["labels"]:
                todo.append((source, row))
                done.add(row["text"])  # the two sets share a few sentences
    if limit is not None:
        todo = todo[:limit]

    use_huggingface_token()
    device = pick_device()
    tokenizer = MarianTokenizer.from_pretrained(translator)
    model = MarianMTModel.from_pretrained(translator).to(device).eval()
    logger.info("translating %d sentences with %s on %s", len(todo), translator, device)

    with out.open("a", encoding="utf-8") as handle, torch.inference_mode():
        batches = progress(
            range(0, len(todo), batch_size),
            desc="translating",
            total=-(-len(todo) // batch_size),
            unit="batch",
        )
        for start in batches:
            chunk = todo[start : start + batch_size]
            batch = tokenizer(
                [row["text"] for _, row in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            generated = model.generate(**batch, num_beams=4, max_new_tokens=256)
            arabic = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (source, row), text in zip(chunk, arabic, strict=True):
                record = {
                    "text": text.strip(),
                    "language": "ar",
                    "labels": {"sentiment": row["labels"]["sentiment"]},
                    "source": source,
                    "source_text": row["text"],
                    "translator": translator,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if not progress_enabled():
                logger.info("translated %d / %d", min(start + batch_size, len(todo)), len(todo))

    (folder / "SOURCE.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d"),
                "translator": translator,
                "translated_from": list(TRANSLATED_FROM),
                "rows": sum(1 for _ in out.open(encoding="utf-8")),
                "note": "machine translation of human-labelled English sentences; "
                "labels are the English annotators'",
            },
            indent=2,
        )
    )
    return out


_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")
_STUTTER = re.compile(r"(\S+)(?:\s+\1){4,}")


def translation_problem(arabic: str, english: str) -> str | None:
    """Why a machine translation is not fit to train on, or None if it is."""
    if not arabic.strip():
        return "empty"
    if len(_LATIN.findall(arabic)) > len(_ARABIC.findall(arabic)):
        return "more Latin than Arabic"
    if _STUTTER.search(arabic):
        return "a word repeated five or more times"
    if len(arabic) < 0.3 * len(english):
        return "under 30% of the original's length"
    return None


ARSARCASM_SENTIMENT = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}


def _rows_arsarcasm_v2(limit: int | None) -> list[dict[str, Any]]:
    """Native Arabic sentiment tweets, for training only."""
    rows = []
    for split in ("train", "test"):
        for record in _load("arbml/ArSarcasm_v2", split=split):
            text = (record.get("tweet") or "").strip()
            label = ARSARCASM_SENTIMENT.get(record.get("sentiment"))
            if text and label:
                rows.append(
                    {
                        "text": text,
                        "language": "ar",
                        "labels": {"sentiment": label},
                        "dialect": record.get("dialect"),
                    }
                )
    return _cap(rows, limit, "arsarcasm_v2")


def heldout_key(text: str) -> str:
    """A tweet reduced to its letters, folded, without links or handles: the same tweet with a
    different @mention or elongation is one key.
    """
    from .preprocess import fold_for_model, normalise

    folded = fold_for_model(normalise(text)).lower()
    folded = re.sub(r"https?://\S+|@\w+|#", "", folded)
    letters = "".join(ch for ch in folded if ch.isalpha())
    # Every run of a letter to one: stricter than the model's fold, which keeps two, because a
    # near-duplicate should count as a duplicate here.
    return re.sub(r"(.)\1+", r"\1", letters)


def _rows_translated_sentiment(limit: int | None) -> list[dict[str, Any]]:
    path = RAW_ROOT / "translated_sentiment_ar" / TRANSLATED_FILE
    if not path.exists():
        translate_sentiment()  # only on a machine that has never translated
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    kept = [r for r in rows if not translation_problem(r.get("text", ""), r["source_text"])]
    if len(kept) < len(rows):
        logger.info(
            "translated_sentiment_ar: dropped %d broken translations", len(rows) - len(kept)
        )
    return _cap(kept, limit, "translated_sentiment_ar")


LOADERS = {
    "financial_phrasebank": _rows_financial_phrasebank,
    "fiqa_2018": _rows_fiqa,
    "jigsaw_toxic": _rows_jigsaw,
    "wikipedia_talk": _rows_wikipedia_talk,
    "hatexplain": _rows_hatexplain,
    "osact4": _rows_osact4,
    "osact5": _rows_osact5,
    "liar": _rows_liar,
    "isot_fake_news": _rows_isot,
    "fakenewsnet": _rows_fakenewsnet,
    "pheme": _rows_pheme,
}
"""Sources the encoder trains on. Every key here is declared `train` in the
catalogue, and `check_training_sources` enforces that before anything loads."""

EVAL_ONLY_LOADERS = {
    "arsas": _rows_arsas,
    "lhsab": _rows_lhsab,
}
"""Sources that may only ever be evaluated on. The catalogue declares these
`eval`, so putting one in `LOADERS` would be caught by the licence gate -- but
the gate catches a licence breach, not a leak. These rows go into the held-out
splits and nowhere else, so the Arabic score is always measured on native
Arabic the model never trained on."""

TRAIN_ONLY_LOADERS = {
    "translated_sentiment_ar": _rows_translated_sentiment,
    "arsarcasm_v2": _rows_arsarcasm_v2,
}
"""Derived sources that may only be trained on. A translated sentence enters
training only when its English original is in the training split: otherwise the
shared encoder would see a translation of an English dev or test item, and the
English numbers would quietly improve for the wrong reason. Arabic sentiment is
evaluated on ArSAS, which is native Arabic, never on these translations."""

SUBSTITUTE_LOADERS = {
    "osact4": _rows_arabic_toxicity,
}
"""Stand-ins a trained model's manifest may record. Used only when re-scoring
that model, so its rebuilt train set matches the one it was fitted on."""


def load_rows(
    sources: tuple[str, ...],
    *,
    eval_sources: tuple[str, ...] = (),
    limits: dict[str, int | None] | None = None,
    substitutions: tuple[str, ...] = (),
    train_only: tuple[str, ...] = (),
    eval_share: float = 0.15,
    seed: int = 20240101,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load the declared sources and split them three ways."""
    catalogue.check_training_sources(list(sources) + list(train_only))
    for source in eval_sources:
        if source in sources:
            raise CorpusError(f"{source!r} is eval-only and cannot also be a training source")
        if catalogue.TRAIN in catalogue.get(source).uses:
            logger.warning("%s is cleared for training but is being held out", source)

    limits = DEFAULT_LIMITS | (limits or {})
    provenance: dict[str, Any] = {"sources": {}, "substitutions": []}
    everything: list[dict[str, Any]] = []

    sources_bar = progress(sources, desc="loading sources", total=len(sources), unit="source")
    for source in sources_bar:
        note(sources_bar, now=source)
        loader = (SUBSTITUTE_LOADERS if source in substitutions else LOADERS).get(source)
        if loader is None:
            raise CorpusError(f"no loader for declared source {source!r}")
        rows = loader(limits.get(source))
        if not rows:
            # A source that loads nothing is a failure, not a smaller dataset: FiQA once
            # returned 0 rows silently and a model trained without it.
            raise CorpusError(f"{source} loaded 0 rows; refusing to train without it")
        for row in rows:
            # Which corpus a row came from, so errors can be traced to a source (see
            # diagnose.py). Not a feature: the model never sees it.
            row["corpus"] = source
        everything.extend(rows)

        mirror = (SUBSTITUTE_MIRRORS if source in substitutions else MIRRORS)[source]
        provenance["sources"][source] = {
            "hub_path": mirror.path,
            "split": mirror.split,
            "rows": len(rows),
            "note": mirror.note,
            "licence": catalogue.get(source).licence,
        }
        if mirror.note.startswith("SUBSTITUTION"):
            provenance["substitutions"].append(source)
        logger.info("%s: %d rows from %s", source, len(rows), mirror.path)

    # Shuffle once, with the run's seed, so the split is reproducible.
    random.Random(seed).shuffle(everything)
    cut = int(len(everything) * (1 - eval_share))
    held = everything[cut:]
    half = len(held) // 2
    split = {"train": everything[:cut], "dev": held[:half], "test": held[half:]}

    # Eval-only sources are appended to the eval side after the split, so no shuffle or share
    # calculation can ever put one of them in training.
    for source in eval_sources:
        loader = EVAL_ONLY_LOADERS.get(source) or LOADERS.get(source)
        if loader is None:
            raise CorpusError(f"no loader for eval-only source {source!r}")
        held = loader(limits.get(source))
        for row in held:
            row["corpus"] = source
        # Shuffled before halving: a loader's order is the publisher's order, often grouped by
        # topic or label, and halving that would hand dev and test different distributions.
        random.Random(f"{seed}:{source}").shuffle(held)
        half = len(held) // 2
        split["dev"].extend(held[:half])
        split["test"].extend(held[half:])
        mirror = MIRRORS[source]
        provenance["sources"][source] = {
            "hub_path": mirror.path,
            "split": mirror.split,
            "rows": len(held),
            "note": mirror.note,
            "licence": catalogue.get(source).licence,
            "held_out": True,
        }
        logger.info("%s: %d rows from %s, evaluation only", source, len(held), mirror.path)
    # Train-only sources join after the split. A translation joins only when its original is in
    # training; a native row only when nothing held out matches it.
    train_texts = {r["text"] for r in split["train"]}
    heldout_keys = {heldout_key(r["text"]) for r in (*split["dev"], *split["test"])}
    heldout_keys.discard("")
    for source in train_only:
        loader = TRAIN_ONLY_LOADERS.get(source)
        if loader is None:
            raise CorpusError(f"no loader for train-only source {source!r}")
        derived = loader(limits.get(source))
        kept = [
            r
            for r in derived
            if (
                r["source_text"] in train_texts
                if "source_text" in r
                else heldout_key(r["text"]) not in heldout_keys
            )
        ]
        for row in kept:
            row["corpus"] = source
        split["train"].extend(kept)
        mirror = MIRRORS[source]
        provenance["sources"][source] = {
            "hub_path": mirror.path,
            "split": mirror.split,
            "rows": len(kept),
            "dropped_matches_dev_or_test": len(derived) - len(kept),
            "note": mirror.note,
            "licence": catalogue.get(source).licence,
            "train_only": True,
        }
        logger.info("%s: %d rows kept, %d held back", source, len(kept), len(derived) - len(kept))
    # Shuffled again so translated rows don't all sit at the end of an epoch.
    if train_only:
        random.Random(f"{seed}:train").shuffle(split["train"])
    provenance["train_only"] = list(train_only)

    provenance["limits"] = {k: v for k, v in limits.items() if k in sources + eval_sources}
    provenance["eval_only"] = list(eval_sources)
    provenance["totals"] = {
        "train": len(split["train"]),
        "dev": len(split["dev"]),
        "test": len(split["test"]),
        # Lets a later evaluation prove it is looking at the same train set the model was fitted
        # on, rather than assuming it.
        "train_fingerprint": hashlib.sha256(
            "\n".join(r["text"] for r in split["train"]).encode("utf-8")
        ).hexdigest()[:16],
        # The same for dev and test together: runs that add training data can still prove they
        # were evaluated on exactly the same rows.
        "heldout_fingerprint": hashlib.sha256(
            "\n".join(r["text"] for r in (*split["dev"], *split["test"])).encode("utf-8")
        ).hexdigest()[:16],
        "by_language": _counts(r["language"] for r in everything),
        "by_task": _counts(task for r in everything for task in r["labels"]),
    }
    return split, provenance


def materialise(
    sources: tuple[str, ...],
    *,
    eval_sources: tuple[str, ...] = (),
    out_dir: str | Path = "data/sources",
    limits: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Write one folder per declared source, so the data can be looked at."""
    catalogue.check_training_sources(list(sources))
    limits = DEFAULT_LIMITS | (limits or {})
    root = Path(out_dir)
    written: dict[str, Any] = {}

    every = sources + eval_sources
    sources_bar = progress(every, desc="writing sources", total=len(every), unit="source")
    for source in sources_bar:
        note(sources_bar, now=source)
        loader = LOADERS.get(source) or EVAL_ONLY_LOADERS.get(source)
        if loader is None:
            raise CorpusError(f"no loader for {source!r}")
        rows = loader(limits.get(source))
        folder = root / source
        folder.mkdir(parents=True, exist_ok=True)

        with (folder / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        entry = catalogue.get(source)
        mirror = MIRRORS[source]
        manifest = {
            "declared_source": source,
            "hub_path": mirror.path,
            "split": mirror.split,
            "note": mirror.note,
            "licence": entry.licence,
            "licence_status": entry.status,
            "redistributable": entry.redistributable,
            "conditions": list(entry.conditions),
            "tasks": list(entry.tasks),
            "uses": list(entry.uses),
            "eval_only": source in eval_sources,
            "rows": len(rows),
            "limit_applied": limits.get(source),
            "languages": _counts(r["language"] for r in rows),
            "labels": _counts(task for r in rows for task in r["labels"]),
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written[source] = manifest
        logger.info("%s: %d rows -> %s", source, len(rows), folder)

    (root / "INDEX.json").write_text(
        json.dumps(
            {
                "generated_at": time.time(),
                "sources": written,
                "total_rows": sum(m["rows"] for m in written.values()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return written


def _load(path: str, split: str = "train"):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - the extras are optional
        raise CorpusError(
            "loading corpora needs the 'train' extras: pip install '.[train]'"
        ) from exc
    use_huggingface_token()
    return load_dataset(path, split=split)


def _take(data, limit: int | None):
    if limit is None or limit >= data.num_rows:
        return list(data)
    return list(data.shuffle(seed=20240101).select(range(limit)))


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
