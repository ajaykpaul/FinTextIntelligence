"""Resolving a pasted link into text to score."""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse, urlunparse

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 3
# Shortest string we will accept as a headline rather than a page title.
HEADLINE_WORDS = 5
USER_AGENT = "FinTextIntelligence/1.5 (+scoring service; contact ops)"

# A bare link, possibly with surrounding whitespace, and nothing else.
URL_ONLY = re.compile(r"^\s*(https?://\S+)\s*$", re.IGNORECASE)


class FetchError(Exception):
    """The link could not be turned into text, with a reason a person can read."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.suggestion = suggestion


class UnsafeURL(FetchError):
    """The link itself is the problem: wrong scheme, or an address we will not open."""


def looks_like_url(text: str) -> str | None:
    """Return the URL if the input is just a link, otherwise None."""
    match = URL_ONLY.match(text)
    return match.group(1) if match else None


def check_address(url: str) -> str:
    """Reject anything that is not a public http(s) endpoint."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURL(f"only http and https links can be fetched, not {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURL("that link has no hostname")

    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURL(f"could not resolve {host}") from exc

    for *_, sockaddr in resolved:
        address = ipaddress.ip_address(sockaddr[0])
        # Everything non-public is refused, not just loopback: link-local covers the cloud
        # metadata endpoint, and private covers internal services.
        if not address.is_global or address.is_loopback or address.is_link_local:
            raise UnsafeURL(f"{host} resolves to a non-public address; refusing to fetch it")
    return host


@dataclass(frozen=True)
class Article:
    url: str
    title: str
    text: str

    @property
    def scoreable(self) -> str:
        """Headline first, then the body."""
        return f"{self.title}. {self.text}".strip(". ").strip() if self.title else self.text

    @property
    def headline_only(self) -> bool:
        """True when the page gave up a headline but no article body."""
        return not self.text


def fetch_article(url: str, *, timeout_s: float = 8.0, max_bytes: int = 2_000_000) -> Article:
    """Fetch a link and pull the readable text out of it."""
    import urllib.error
    import urllib.request

    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_address(current)
        request = urllib.request.Request(
            current,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain"},
        )

        class NoRedirects(urllib.request.HTTPRedirectHandler):
            """Redirects are followed by hand so each hop is re-checked."""

            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(NoRedirects)
        try:
            with opener.open(request, timeout=timeout_s) as response:
                kind = response.headers.get_content_type()
                if kind not in ("text/html", "text/plain"):
                    raise FetchError(
                        f"that link returns {kind}, which has no text to score",
                        headline_from_url(current),
                    )
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise FetchError("the page is larger than the fetch limit")
                charset = response.headers.get_content_charset() or "utf-8"
                html = body.decode(charset, errors="replace")
                try:
                    title, text = extract(html)
                except FetchError as exc:
                    # extract() sees the page, not the link, so the headline in the URL is
                    # offered here.
                    raise FetchError(str(exc), headline_from_url(current)) from exc
                return Article(current, title, text)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                if not location:
                    raise FetchError("the link redirected without saying where") from exc
                current = (
                    urlunparse(urlparse(current)._replace(path="", query="")) + location
                    if location.startswith("/")
                    else location
                )
                continue
            raise FetchError(explain_status(exc.code), headline_from_url(current)) from exc
        except urllib.error.URLError as exc:
            raise FetchError(f"could not reach that link: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FetchError("the site took too long to answer") from exc

    raise FetchError("too many redirects")


# --------------------------------------------------------------------------
# Turning HTML into something worth scoring
# --------------------------------------------------------------------------

_DROP = re.compile(
    r"<(script|style|noscript|nav|header|footer|aside|form|figure)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_MAIN = re.compile(r"<(article|main)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")


def extract(html: str) -> tuple[str, str]:
    """Pull a headline and body text out of a page."""
    cleaned = _DROP.sub(" ", html)

    heading = _H1.search(cleaned) or _TITLE.search(cleaned)
    title = _text(heading.group(1)) if heading else ""

    # Prefer the article body when the page marks one; otherwise take every paragraph and rely
    # on the token cap to stop at the useful part.
    main = _MAIN.search(cleaned)
    region = main.group(2) if main else cleaned
    paragraphs = [_text(p) for p in _PARA.findall(region)]
    # Drop nav crumbs, bylines and cookie notices, which are all short.
    body = " ".join(p for p in paragraphs if len(p.split()) >= 8)

    # A headline on its own is worth scoring.
    if not body and len(title.split()) < HEADLINE_WORDS:
        raise FetchError(
            "no article on that page; front pages, section indexes and sites "
            "that render their text in the browser leave nothing to read"
        )
    return title, body


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAGS.sub(" ", fragment))).strip()


# --------------------------------------------------------------------------
# When the page will not open
# --------------------------------------------------------------------------

_BLOCKED = {
    401: "that site requires a subscription or blocks automated readers",
    403: "that site blocks automated readers",
    402: "that article is behind a paywall",
    429: "that site is rate-limiting us; try again shortly",
    404: "that page was not found",
    410: "that page has been taken down",
}


def explain_status(code: int) -> str:
    """Say what actually happened, in words that suggest what to do next."""
    if code in _BLOCKED:
        return _BLOCKED[code]
    if 500 <= code < 600:
        return "that site is having trouble right now"
    return f"the site refused the request ({code})"


# Trailing slug noise: a date, or an id like -idUSKBN2A1234.
_SLUG_TAIL = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{6,}|id[a-z0-9]{6,})$", re.IGNORECASE)
_SLUG_STOP = frozenset({"amp", "html", "htm", "story", "article", "news", "index"})


def headline_from_url(url: str) -> str:
    """Recover a headline from the URL when the page itself is closed to us."""
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    slug = _SLUG_TAIL.sub("", slug)
    words = [w for w in re.split(r"[-_]+", slug) if w and w.lower() not in _SLUG_STOP]
    if len(words) < 4:
        return ""
    sentence = " ".join(words)
    return sentence[0].upper() + sentence[1:]
