"""Building the provenance index from SEC EDGAR."""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .progress import note, progress

logger = logging.getLogger("fintext.edgar")

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

PAGE_SIZE = 10
"""What the endpoint returns per call. Not configurable at the far end."""

MAX_FROM = 9_990
"""EDGAR refuses to page past ten thousand results. A query that hits this is
too broad to be a useful sample; narrow the date range instead."""

REQUEST_INTERVAL = 0.2
"""Five requests a second, against a published limit of ten. The margin is
deliberate: being throttled by the SEC costs more time than it saves."""

# The 8-K item taxonomy, as the SEC titles it.
SEC_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure of Directors or Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

# Queries chosen to populate each claim type the extractor recognises.
DEFAULT_QUERIES: tuple[tuple[str, str], ...] = (
    ('"material definitive agreement"', "8-K"),
    ('"merger agreement"', "8-K"),
    ('"definitive agreement to acquire"', "8-K"),
    ('"results of operations"', "8-K"),
    ('"full-year guidance"', "8-K"),
    ('"updates guidance"', "8-K"),
    ('"contract award"', "8-K"),
    ('"consent order"', "8-K"),
    ('"administrative proceeding"', "8-K"),
)

# "Banzai International, Inc. (BNZI, BNZIW) (CIK 0001826011)": a filer with several listed
# classes shows them all. The first is the common stock.
_DISPLAY = re.compile(
    r"^(?P<name>.+?)\s+\((?P<ticker>[A-Z.\-]{1,10})(?:,\s*[A-Z.\-]{1,10})*\)\s+\(CIK", re.IGNORECASE
)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


class EdgarError(Exception):
    """EDGAR could not be queried as asked."""


@dataclass(frozen=True)
class Filing:
    """One search hit, before it becomes an index document."""

    accession: str
    cik: str
    entity: str
    ticker: str
    form: str
    filed: str
    items: tuple[str, ...]
    filename: str

    @property
    def url(self) -> str:
        return f"{ARCHIVE_URL}/{int(self.cik)}/{self.accession.replace('-', '')}/{self.filename}"

    @property
    def published_at(self) -> float:
        """Filing date at midnight UTC."""
        return time.mktime(time.strptime(self.filed, "%Y-%m-%d"))

    @property
    def title(self) -> str:
        named = [SEC_ITEMS[i] for i in self.items if i in SEC_ITEMS]
        return f"Form {self.form}: {named[0]}" if named else f"Form {self.form}"

    def to_document(self, body: str = "") -> dict[str, Any]:
        """The record shape `PrimaryIndex` loads."""
        taxonomy = " ".join(
            f"item {i} {SEC_ITEMS[i]}" for i in self.items if i in SEC_ITEMS
        ).lower()
        return {
            "doc_id": f"edgar-{self.accession}",
            "source_type": "edgar",
            "entity": self.entity,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at,
            "body": _WS.sub(" ", f"{self.form} {taxonomy} {body}").strip().lower(),
            "tickers": [self.ticker] if self.ticker else [],
        }


@dataclass
class Fetcher:
    """A throttled EDGAR client."""

    contact: str = ""
    interval: float = REQUEST_INTERVAL
    _last: float = field(default=0.0, repr=False)

    @property
    def user_agent(self) -> str:
        contact = self.contact or os.environ.get("FINTEXT_SEC_CONTACT", "")
        # SEC's fair-access policy asks for a contact address. Without one the request still
        # works, but it is discourteous and they may block it.
        return f"FinTextIntelligence/1.0 research {contact}".strip()

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()

    def get(self, url: str) -> bytes:
        self._wait()
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise EdgarError("EDGAR is rate-limiting; slow down or try later") from exc
            raise EdgarError(f"EDGAR returned {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise EdgarError(f"could not reach EDGAR: {exc.reason}") from exc

    def search(
        self, query: str, *, forms: str = "8-K", start: str = "", end: str = "", limit: int = 100
    ) -> list[Filing]:
        """Page through full-text search until `limit` hits or the results run out."""
        found: list[Filing] = []
        offset = 0
        while len(found) < limit and offset <= MAX_FROM:
            params = {"q": query, "forms": forms, "from": str(offset)}
            if start and end:
                params |= {"startdt": start, "enddt": end}
            payload = json.loads(self.get(f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"))
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break
            found.extend(filing for hit in hits if (filing := _parse(hit)))
            offset += PAGE_SIZE
        return found[:limit]

    def document_text(self, filing: Filing, max_chars: int = 4000) -> str:
        """The filing's own text, for when the item taxonomy is not enough."""
        try:
            html = self.get(filing.url).decode("utf-8", errors="replace")
        except EdgarError as exc:
            logger.warning("%s: %s", filing.accession, exc)
            return ""
        return html_text(html, max_chars)


def html_text(html: str, max_chars: int = 4000) -> str:
    """A filing document's text as the index stores it: tags out, whitespace collapsed."""
    return _WS.sub(" ", _TAGS.sub(" ", html)).strip()[:max_chars]


def _parse(hit: dict[str, Any]) -> Filing | None:
    source = hit.get("_source", {})
    ciks = source.get("ciks") or []
    names = source.get("display_names") or []
    if not ciks or not names:
        return None

    match = _DISPLAY.match(names[0])
    entity = match.group("name").strip() if match else names[0].split("(CIK")[0].strip()
    ticker = match.group("ticker").upper() if match else ""
    # A filer with no ticker still indexes fine, entity matching does not need one, but the
    # ticker path in score_match is the strongest signal there is.

    filename = hit.get("_id", "").split(":")[-1]
    if not filename:
        return None

    return Filing(
        accession=source.get("adsh", ""),
        cik=ciks[0],
        entity=entity,
        ticker=ticker,
        form=source.get("form", source.get("file_type", "")),
        filed=source.get("file_date", ""),
        items=tuple(source.get("items", ())),
        filename=filename,
    )


def build_index(
    out_path: str | Path,
    *,
    queries: tuple[tuple[str, str], ...] = DEFAULT_QUERIES,
    per_query: int = 100,
    start: str = "",
    end: str = "",
    with_text: bool = False,
    contact: str = "",
) -> dict[str, Any]:
    """Fetch filings and write the provenance index."""
    fetcher = Fetcher(contact=contact)
    documents: dict[str, dict[str, Any]] = {}
    per_query_counts: dict[str, int] = {}

    queries_bar = progress(queries, desc="EDGAR queries", total=len(queries), unit="query")
    for query, forms in queries_bar:
        note(queries_bar, documents=len(documents))
        try:
            filings = fetcher.search(query, forms=forms, start=start, end=end, limit=per_query)
        except EdgarError as exc:
            logger.warning("%s: %s", query, exc)
            per_query_counts[query] = 0
            continue

        new = 0
        for filing in filings:
            if not filing.accession or filing.accession in documents:
                continue
            body = fetcher.document_text(filing) if with_text else ""
            documents[filing.accession] = filing.to_document(body)
            new += 1
        per_query_counts[query] = new
        logger.info("%-38s %3d filings, %3d new", query, len(filings), new)

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for document in sorted(documents.values(), key=lambda d: d["published_at"]):
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    dated = [d["published_at"] for d in documents.values()]
    return {
        "path": str(destination),
        "documents": len(documents),
        "with_ticker": sum(1 for d in documents.values() if d["tickers"]),
        "per_query": per_query_counts,
        "earliest": time.strftime("%Y-%m-%d", time.localtime(min(dated))) if dated else "",
        "latest": time.strftime("%Y-%m-%d", time.localtime(max(dated))) if dated else "",
        "full_text": with_text,
    }
