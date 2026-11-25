"""A provenance replay set built from real SEC filings."""

from __future__ import annotations

import html
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .edgar import ARCHIVE_URL, SEARCH_URL, EdgarError, Fetcher
from .progress import note, progress
from .provenance import EVIDENCE_HORIZON

logger = logging.getLogger("fintext.replay_set")


@dataclass(frozen=True)
class Category:
    claim_type: str
    phrases: tuple[str, ...]
    items: frozenset[str]
    states: str = ""


CATEGORIES = (
    Category(
        "merger",
        (
            '"definitive agreement to acquire"',
            '"has completed the acquisition of"',
            '"merger agreement"',
        ),
        frozenset({"1.01", "2.01"}),
        r"\b(?:agreement\s+(?:to|under\s+which\s+\S+(?:\s+\S+){0,3}\s+will)\s+acquire|will\s+acquire"
        r"|completed\s+(?:the|its)\s+(?:previously\s+announced\s+)?acquisition|merger\s+agreement"
        r"|agreed\s+to\s+(?:acquire|merge|be\s+acquired))\b",
    ),
    Category(
        "guidance",
        (
            '"raises full-year guidance"',
            '"raises guidance"',
            '"lowers guidance"',
            '"reaffirms guidance"',
            '"raised its full-year"',
            '"updates full-year guidance"',
            '"increases full-year guidance"',
        ),
        frozenset({"2.02", "7.01"}),
        r"\b(?:rais|lower|reaffirm|updat|increas|narrow|maintain|withdr[ae]w|cut)\w*\b"
        r"[^.]{0,60}\b(?:guidance|outlook)\b",
    ),
    Category(
        "contract",
        ('"awarded a contract"', '"has been awarded a"', '"secured a contract"', '"was awarded a"'),
        frozenset({"1.01", "7.01", "8.01"}),
        r"\b(?:awarded|award\s+of|wins?|won|secured)\b[^.]{0,80}\b(?:contract|task\s+order)\b",
    ),
    Category(
        "regulatory",
        (
            '"delisting determination"',
            '"notice of delisting"',
            '"received a deficiency notice"',
            '"deficiency letter"',
            '"notice of noncompliance"',
        ),
        frozenset({"3.01"}),
        r"\b(?:delisting|deficiency\s+(?:notice|letter)|notice\s+of\s+non-?compliance)\b",
    ),
)

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
# The dateline ends at a dash or a wire tag: "DALLAS, Texas – March 18, 2025 –".
_DATELINE = re.compile(
    r"(?:\s[–—-]{1,2}\s"
    r"|/(?:PRNewswire|GLOBE NEWSWIRE|Business Wire)/\s*[–—-]*\s*)"
)
_ANNOUNCES = re.compile(
    r"\b(?:today\s+announced|announced\s+today|announced|announces|today\s+reported)\b", re.I
)
_EXCHANGE = re.compile(
    r"\(\s*(?:NYSE(?:\s+American)?|Nasdaq|NASDAQ|NYSE\s+MKT|OTCQX|OTCQB|TSX)\s*:\s*([A-Z.]{1,6})\s*\)"
)
# A sentence ends at a full stop before a capital, unless the word before it is one of the
# abbreviations company names and addresses are full of.
_ABBREVIATIONS = (
    "Inc",
    "Corp",
    "Co",
    "Ltd",
    "L.P",
    "N.A",
    "U.S",
    "S.A",
    "No",
    "Mr",
    "Ms",
    "Dr",
    "St",
    "Jr",
    "LLC",
    "plc",
    "N.V",
)
_QUOTED_ALIAS = re.compile(r"\(\s*(?:the\s+)?[“\"][^)]{0,200}\)")


def page_text(raw: str) -> str:
    return _SPACE.sub(" ", html.unescape(_TAGS.sub(" ", raw))).strip()


def sentences(text: str, *, limit: int = 30000) -> list[str]:
    """Split release text into sentences, minding the abbreviations in names."""
    out, start = [], 0
    body = text[:limit]
    while start < len(body):
        end = _sentence_end(body, start)
        if end is None:
            break
        out.append(body[start:end].strip())
        start = end
    return out


# What EDGAR and the wire put in front of a release's first words: the exhibit label and file
# name, "Exhibit 99.1", "PRESS RELEASE, DATED ...", "News Release".
_HEADER = re.compile(
    r"^(?:EX-99\.\d+\s+\d+\s+\S+\.html?\s*|EX-99\.\d+\s*(?:Document\s*)?|Exhibit\s+99\.\d+\s*"
    r"|PRESS\s+RELEASE(?:,\s+DATED\s+[A-Z]+\s+\d{1,2},\s+\d{4})?\s*|NEWS\s+RELEASE\s*"
    r"|FOR\s+IMMEDIATE\s+RELEASE\s*|Document\s+|\u200b\s*)+",
    re.I,
)


def strip_headers(sentence: str) -> str:
    return _HEADER.sub("", sentence).strip()


def _mentions(sentence: str, entity: str, ticker: str) -> bool:
    first = display_name(entity).split()[0]
    if ticker and re.search(rf"\b{re.escape(ticker)}\b", sentence):
        return True
    return len(first) > 2 and re.search(rf"\b{re.escape(first)}\b", sentence, re.I) is not None


def claim_sentence(
    text: str, *, states: str, entity: str, ticker: str, max_chars: int = 400
) -> str | None:
    """The first sentence that names the company and states the claim."""
    pattern = re.compile(states, re.I)
    for sentence in sentences(text):
        if not pattern.search(sentence) or not _mentions(sentence, entity, ticker):
            continue
        cuts = [m for m in _DATELINE.finditer(sentence) if m.end() <= 250]
        if cuts:
            sentence = sentence[cuts[-1].end() :]
        sentence = strip_headers(_SPACE.sub(" ", _QUOTED_ALIAS.sub("", sentence)).strip())
        sentence = re.sub(r"\s+([,.])", r"\1", sentence)
        if (
            40 <= len(sentence) <= max_chars
            and sentence[0].isupper()
            and pattern.search(sentence)
            and _mentions(sentence, entity, ticker)
        ):
            return sentence
    return None


def _sentence_end(text: str, after: int) -> int | None:
    for match in re.finditer(r"\.(?=\s+[A-Z“\"]|\s*$)", text[after : after + 1500]):
        position = after + match.start()
        word = text[max(0, position - 4) : position].split(" ")[-1]
        if word.rstrip(".") in _ABBREVIATIONS or len(word) == 1:
            continue
        return position + 1
    return None


_LISTED = re.compile(r"^(?P<name>.+?)\s+\((?P<ticker>[A-Z.\-]{1,10})(?:,\s*[A-Z.\-]{1,10})*\)")


def split_display_name(display: str) -> tuple[str, str]:
    """EDGAR's 'Name (TICK, TICKW) (CIK ...)' -> ('Name', 'TICK')."""
    match = _LISTED.match(display)
    if match:
        return match.group("name").strip(), match.group("ticker")
    return display.split("(CIK")[0].strip(), ""


def _source(hit: dict[str, Any]) -> dict[str, Any] | None:
    source = hit.get("_source", {})
    if not str(source.get("file_type", "")).upper().startswith("EX-99"):
        return None
    names = source.get("display_names") or []
    ciks = source.get("ciks") or []
    if not names or not ciks:
        return None
    filename = hit.get("_id", "").split(":")[-1]
    entity, ticker = split_display_name(names[0])
    return {
        "accession": source.get("adsh", ""),
        "cik": ciks[0],
        "entity": entity,
        "ticker": ticker,
        "filed": source.get("file_date", ""),
        "items": tuple(source.get("items") or ()),
        "filename": filename,
    }


def _filing_document(found: dict[str, Any]) -> dict[str, Any]:
    from .edgar import Filing

    filing = Filing(
        accession=found["accession"],
        cik=found["cik"],
        entity=found["entity"],
        ticker=found["ticker"],
        form="8-K",
        filed=found["filed"],
        items=found["items"],
        filename=found["filename"],
    )
    return filing.to_document()


def collect_genuine(
    fetcher: Fetcher,
    *,
    per_category: int = 40,
    start: str = "2024-07-01",
    end: str = "2025-12-31",
    max_pages: int = 15,
) -> list[dict[str, Any]]:
    """Genuine cases: one per filing, the press release's lead as the claim."""
    import urllib.parse

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        kept = 0
        bar = progress(category.phrases, desc=f"EDGAR {category.claim_type}", unit="query")
        for phrase in bar:
            for page in range(max_pages):
                if kept >= per_category:
                    break
                params = {
                    "q": phrase,
                    "forms": "8-K",
                    "startdt": start,
                    "enddt": end,
                    "from": str(page * 10),
                }
                try:
                    payload = json.loads(
                        fetcher.get(f"{SEARCH_URL}?{urllib.parse.urlencode(params)}")
                    )
                except EdgarError as exc:
                    logger.warning("%s: %s", phrase, exc)
                    break
                hits = payload.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    if kept >= per_category:
                        break
                    found = _source(hit)
                    if not found or found["accession"] in seen:
                        continue
                    matched = category.items & set(found["items"])
                    if not matched:
                        continue
                    url = (
                        f"{ARCHIVE_URL}/{int(found['cik'])}/"
                        f"{found['accession'].replace('-', '')}/{found['filename']}"
                    )
                    try:
                        text = page_text(fetcher.get(url).decode("utf-8", errors="replace"))
                    except EdgarError as exc:
                        logger.warning("%s: %s", found["accession"], exc)
                        continue
                    sentence = claim_sentence(
                        text,
                        states=category.states,
                        entity=found["entity"],
                        ticker=found["ticker"],
                    )
                    if not sentence:
                        continue
                    seen.add(found["accession"])
                    document = _filing_document(found)
                    cases.append(
                        {
                            "item_id": f"genuine-{category.claim_type}-{found['accession']}",
                            "is_genuine": True,
                            "observed_at": document["published_at"],
                            "text": sentence,
                            "expected_claim_type": category.claim_type,
                            "expected_entity": found["entity"],
                            "ticker": found["ticker"],
                            "selected_by": f"{phrase} + 8-K items {sorted(matched)}",
                            "source_url": url,
                            "documents": [document],
                        }
                    )
                    kept += 1
                    note(bar, kept=kept)
        logger.info("%s: %d genuine cases", category.claim_type, kept)
    return cases


_SUFFIX = r"(?:Inc|Corp|Corporation|Ltd|LLC|plc|Co|Company|Holdings|Group|N\.V|S\.A|L\.P)\.?"


def display_name(entity: str) -> str:
    """How a release writes a filer's name: 'AMETEK INC/' -> 'Ametek Inc'."""
    name = re.sub(r"\s*/[A-Z]{0,3}/?\s*$", "", entity.strip()).strip(" ,/")
    return name.title() if name.isupper() else name


def _short(entity: str) -> str:
    return re.sub(rf",?\s*{_SUFFIX}$", "", display_name(entity), flags=re.I).strip(" ,")


def swap_company(
    text: str, *, old_entity: str, old_ticker: str, new_entity: str, new_ticker: str
) -> str | None:
    """The sentence with every mention of one company replaced by another."""
    old_full, new_full = display_name(old_entity), display_name(new_entity)
    old_short, new_short = _short(old_entity), _short(new_entity)
    pairs = [(old_full, new_full), (old_short, new_short)]
    old_first, new_first = old_short.split()[0], new_short.split()[0]
    if len(old_first) > 2:
        pairs.append((old_first, new_first))
    if old_ticker:
        pairs.append((old_ticker, new_ticker))
    swapped = text
    for index, (old, new) in enumerate(pairs):
        # Full names match in any case (releases shout them in headlines); the first word and
        # ticker only as written, so "First Watch" doesn't take "the first quarter" with it.
        swapped = re.sub(
            rf"\b{re.escape(old)}(?!\w)",
            "\x00" + new + "\x01",
            swapped,
            flags=re.I if index < 2 else 0,
        )
    swapped = re.sub("\x00|\x01", "", swapped)
    # The exchange listing names the company too, and the ticker EDGAR records isn't always the
    # one the release prints ("(Nasdaq: DFDV)").
    swapped = _EXCHANGE.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)]
        + new_ticker
        + m.group(0)[m.end(1) - m.start(0) :],
        swapped,
    )
    if _mentions(swapped, old_entity, old_ticker) and not (len(old_first) <= 2 and not old_ticker):
        return None
    return swapped


def fabricate(genuine: list[dict[str, Any]], *, seed: int = 20261118) -> list[dict[str, Any]]:
    """One fabricated copy of each genuine case, with a company that can't support it."""
    rng = random.Random(seed)
    filings: dict[str, list[tuple[float, str]]] = {}
    tickers: dict[str, str] = {}
    for case in genuine:
        filings.setdefault(case["expected_entity"], []).append(
            (case["observed_at"], case["expected_claim_type"])
        )
        tickers[case["expected_entity"]] = case["ticker"]

    fabricated = []
    for index, case in enumerate(genuine):
        want_nearby = index % 2 == 1
        when, kind = case["observed_at"], case["expected_claim_type"]
        candidates = []
        for entity, dated in filings.items():
            if entity == case["expected_entity"] or not tickers[entity]:
                continue
            near = [k for t, k in dated if abs(t - when) <= EVIDENCE_HORIZON]
            if kind in near:
                continue  # could honestly support it: not a fabrication
            if want_nearby == bool(near):
                candidates.append(entity)
        if not candidates:
            continue
        partner = rng.choice(sorted(candidates))
        text = swap_company(
            case["text"],
            old_entity=case["expected_entity"],
            old_ticker=case["ticker"],
            new_entity=partner,
            new_ticker=tickers[partner],
        )
        # A fabrication has to name the new company, and has to differ from the original;
        # otherwise it is the genuine claim under another label.
        if not text or text == case["text"] or not _mentions(text, partner, tickers[partner]):
            continue
        # Nor may it still name a company that really did file about this kind of event nearby:
        # Also reject it if it still names a real acquirer that filed nearby (e.g. OneWater /
        # Ocean Bio-Chem), otherwise a true claim gets a fabricated label.
        if any(
            _mentions(text, other, tickers[other])
            for other, dated in filings.items()
            if other != partner
            and any(k == kind and abs(t - when) <= EVIDENCE_HORIZON for t, k in dated)
        ):
            continue
        fabricated.append(
            {
                "item_id": f"fabricated-{'nearby' if want_nearby else 'none'}-{kind}-{index:03d}",
                "is_genuine": False,
                "observed_at": when,
                "text": text,
                "expected_claim_type": kind,
                "expected_entity": partner,
                "fabricated_from": case["item_id"],
                "documents": [],
            }
        )
    return fabricated


def refresh(path: str | Path) -> tuple[int, int]:
    """Re-clean the genuine cases in a built file and fabricate again, offline."""
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    genuine = []
    for case in rows:
        if not case["is_genuine"]:
            continue
        case["text"] = strip_headers(case["text"])
        if "(" in case["expected_entity"]:
            entity, ticker = split_display_name(case["expected_entity"])
            case["expected_entity"], case["ticker"] = entity, ticker or case["ticker"]
            for document in case["documents"]:
                document["entity"] = entity
                document["tickers"] = [ticker] if ticker else document["tickers"]
        if len(case["text"]) >= 40 and _mentions(
            case["text"], case["expected_entity"], case["ticker"]
        ):
            genuine.append(case)
    genuine.sort(key=lambda c: (c["observed_at"], c["item_id"]))
    fabricated = fabricate(genuine)
    write_cases(genuine + fabricated, path)
    return len(genuine), len(fabricated)


def add_text(source: str | Path, out: str | Path, fetcher: Fetcher) -> tuple[int, int]:
    """Copy a built case file, giving every filing the text the live index stores."""
    from .edgar import _WS, html_text

    rows = [
        json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line
    ]
    fetched = failed = 0
    cache: dict[str, str] = {}
    documents = [d for case in rows for d in case.get("documents", ())]
    for document in progress(documents, desc="filing text", unit="filing"):
        url = document["url"]
        if url not in cache:
            try:
                cache[url] = html_text(fetcher.get(url).decode("utf-8", errors="replace"))
                fetched += 1
            except EdgarError as exc:
                logger.warning("%s: %s", url, exc)
                cache[url] = ""
                failed += 1
        if cache[url]:
            document["body"] = _WS.sub(" ", f"{document['body']} {cache[url]}").strip().lower()
    # A partly enriched set would be scored as if it were complete, so nothing is written unless
    # every filing got its text.
    if not failed:
        write_cases(rows, out)
    return fetched, failed


_WORD = re.compile(r"\w+")


def _matched(window: list[str], need) -> int:
    from collections import Counter

    return sum((Counter(window) & need).values())


def hold_out_claim(case: dict[str, Any], *, overlap: float = 0.8) -> bool:
    """Remove a genuine case's own claim sentence from its filing's text."""
    from collections import Counter

    claim = [w.lower() for w in _WORD.findall(html.unescape(case["text"]))]
    need = Counter(claim)
    span = max(1, round(len(claim) * 1.25))
    found = False
    for document in case.get("documents", ()):
        words = _WORD.findall(html.unescape(document["body"]).lower())
        drop: set[int] = set()
        if claim:
            for start in range(len(words)):
                window = words[start : start + span]
                shared = _matched(window, need)
                if shared >= overlap * len(claim):
                    # Trim each end while the word there isn't needed for the match, so a repeat
                    # of a claim word in the next sentence stays.
                    low, high = start, start + len(window)
                    while high - low > 1 and _matched(words[low : high - 1], need) == shared:
                        high -= 1
                    while high - low > 1 and _matched(words[low + 1 : high], need) == shared:
                        low += 1
                    drop.update(range(low, high))
        found = found or bool(drop)
        document["body"] = " ".join(w for i, w in enumerate(words) if i not in drop)
    return found


def write_cases(cases: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for case in sorted(cases, key=lambda c: (c["observed_at"], c["item_id"])):
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return target


def _accessions(path: str | Path) -> set[str]:
    found = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            found |= {d["doc_id"] for d in case.get("documents", ())}
    return found


def shared_filings(a: str | Path, b: str | Path) -> set[str]:
    """Filings present in both replay sets."""
    return _accessions(a) & _accessions(b)


def why_not_corroborated(path: str | Path, min_match_score: float) -> dict[str, list[str]]:
    """For each genuine case, the first step at which it fails, with examples."""
    from .provenance import ALIASES, ClaimExtractor, Document, PrimaryIndex, score_match

    cases = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    known = PrimaryIndex(Document.from_dict(d) for c in cases for d in c.get("documents", ()))
    extractor = ClaimExtractor({**ALIASES, **known.aliases()})
    reasons: dict[str, list[str]] = {}
    for case in cases:
        if not case["is_genuine"]:
            continue
        claim = extractor.extract(case["text"])
        text = case["text"][:150]
        if claim is None:
            reason = "no claim recognised"
        elif claim.claim_type.value != case["expected_claim_type"]:
            reason = f"claim type {claim.claim_type.value}, expected {case['expected_claim_type']}"
        elif not claim.entity_key:
            reason = "no company recognised"
        else:
            document = Document.from_dict(case["documents"][0])
            score, why = score_match(claim, document, observed_at=case["observed_at"], window=3600)
            if score >= min_match_score:
                reason = "corroborates"
                text = ""
            else:
                reason = f"no match: {why.split(',')[0]}"
                text = f"[{claim.entity} vs {document.entity}] {text}"
        reasons.setdefault(reason, []).append(text)
    return dict(sorted(reasons.items(), key=lambda kv: -len(kv[1])))
