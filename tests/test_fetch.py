"""Turning a pasted link into text, without opening an SSRF hole."""

from __future__ import annotations

import json

import pytest

from fintext.fetch import (
    Article,
    FetchError,
    check_address,
    explain_status,
    extract,
    headline_from_url,
    looks_like_url,
)

PAGE = """
<html><head><title>Bank reports higher profit | Wire</title></head>
<body>
  <nav><p>Home Markets Opinion Subscribe today for full access</p></nav>
  <script>var tracking = {id: 1};</script>
  <article>
    <h1>First Abu Dhabi Bank reports a rise in quarterly profit</h1>
    <p>Byline</p>
    <p>The lender said net profit rose twelve per cent in the first quarter, helped by
       stronger lending income and lower impairment charges across the corporate book.</p>
    <p>Analysts had expected a smaller increase, and the shares rose in early trading
       before giving back some of the gain later in the session.</p>
  </article>
  <footer><p>Copyright notice and a long list of other links for the reader</p></footer>
</body></html>
"""


class TestDetection:
    @pytest.mark.parametrize(
        "text", ["https://example.com/a", "  http://example.com/a  ", "HTTPS://EXAMPLE.COM/a"]
    )
    def test_a_bare_link_is_a_link(self, text):
        assert looks_like_url(text)

    @pytest.mark.parametrize(
        "text",
        [
            "see https://example.com/a for more",  # a post quoting an article is the post
            "Revenue missed consensus",
            "file:///etc/passwd",
            "ftp://example.com/a",
        ],
    )
    def test_everything_else_is_text(self, text):
        assert looks_like_url(text) is None


class TestAddressGuard:
    """The fetch reaches out to an address the caller chose."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/admin",
            "http://localhost:8080/v1/config",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://10.0.0.5/internal",
            "http://192.168.1.1/",
            "http://[::1]/",
        ],
    )
    def test_non_public_addresses_are_refused(self, url):
        with pytest.raises(FetchError, match="non-public|resolve"):
            check_address(url)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/"])
    def test_only_http_schemes_are_allowed(self, url):
        with pytest.raises(FetchError, match="only http"):
            check_address(url)

    def test_a_missing_hostname_is_refused(self):
        with pytest.raises(FetchError, match="no hostname"):
            check_address("http:///path")

    def test_a_name_that_does_not_resolve_is_refused(self):
        with pytest.raises(FetchError, match="could not resolve"):
            check_address("http://no-such-host.invalid/x")


class TestExtraction:
    def test_prefers_the_headline_and_body(self):
        title, body = extract(PAGE)
        assert "rise in quarterly profit" in title
        assert "net profit rose twelve per cent" in body

    def test_drops_scripts_navigation_and_footers(self):
        _, body = extract(PAGE)
        assert "tracking" not in body
        assert "Subscribe today" not in body
        assert "Copyright notice" not in body

    def test_drops_short_fragments_like_bylines(self):
        _, body = extract(PAGE)
        assert "Byline" not in body

    def test_falls_back_to_the_title_tag(self):
        title, _ = extract(
            "<html><head><title>Only a title here</title></head>"
            "<body><p>A paragraph long enough to survive the filter easily.</p>"
            "</body></html>"
        )
        assert title == "Only a title here"

    def test_a_page_with_nothing_readable_is_an_error(self):
        with pytest.raises(FetchError, match="no article on that page"):
            extract("<html><body><script>x=1</script></body></html>")

    def test_entities_are_decoded(self):
        _, body = extract("<p>Profit rose &amp; margins improved across the whole group today.</p>")
        assert "&amp;" not in body and "rose & margins" in body

    def test_the_headline_leads_the_scored_text(self):
        # Most of the signal in a news item is its headline, and the token cap cuts the tail, so
        # the headline has to come first.
        article = Article("https://x/a", "Profit rises", "Some body text follows here.")
        assert article.scoreable.startswith("Profit rises")


class TestOverHttp:
    def test_a_pasted_link_that_cannot_be_fetched_explains_itself(self, app):
        status, body = app.handle(
            "POST", "/v1/score", json.dumps({"text": "http://127.0.0.1:9/nothing"})
        )
        assert status == 400
        assert "non-public" in body["detail"]

    def test_a_publisher_refusing_us_is_not_a_bad_request(self, app, monkeypatch):
        """The caller sent a perfectly good link."""
        import fintext.server

        def refuse(url, **_):
            raise FetchError("that site requires a subscription", headline_from_url(url))

        monkeypatch.setattr(fintext.server, "fetch_article", refuse)
        status, body = app.handle(
            "POST",
            "/v1/score",
            json.dumps(
                {
                    "text": "https://www.reuters.com/markets/oil-prices-slide-on-supply-glut-2026-08-30/"
                }
            ),
        )
        assert status == 502
        assert body["error"] == "link_unreachable"
        assert body["detail"] == "that site requires a subscription"
        # ...and the headline off the link, so the person has somewhere to go.
        assert body["suggestion"] == "Oil prices slide on supply glut"

    def test_ordinary_text_is_untouched(self, app):
        status, body = app.handle(
            "POST",
            "/v1/score",
            json.dumps({"text": "Revenue missed consensus and the board cut guidance"}),
        )
        assert status == 200 and "fetched" not in body

    def test_fetching_can_be_turned_off(self, config, index):
        from fintext.pipeline import Pipeline
        from fintext.server import App

        app = App(Pipeline(config.replace(fetch_urls=False), index=index))
        app.startup()
        # With fetching off the link is just a string, and scoring it succeeds.
        status, body = app.handle("POST", "/v1/score", json.dumps({"text": "http://127.0.0.1/x"}))
        assert status == 200 and "fetched" not in body
        app.shutdown()


class TestBlockedPages:
    """Most publishers close their articles to anything that is not a browser."""

    @pytest.mark.parametrize(
        "code,phrase",
        [
            (401, "subscription"),
            (403, "blocks automated readers"),
            (402, "paywall"),
            (429, "rate-limiting"),
            (404, "not found"),
            (503, "having trouble"),
            (418, "refused the request (418)"),
        ],
    )
    def test_the_reason_reads_like_english(self, code, phrase):
        assert phrase in explain_status(code)

    @pytest.mark.parametrize(
        "url,expected",
        [
            (
                "https://www.reuters.com/business/energy/"
                "trump-says-us-will-refill-the-reserve-2026-08-30/",
                "Trump says us will refill the reserve",
            ),
            # A trailing wire id is noise, not part of the headline.
            (
                "https://x.com/a/bank-lifts-profit-guidance-idUSKBN2A1234",
                "Bank lifts profit guidance",
            ),
            # Opaque slugs carry no headline, so nothing is offered.
            ("https://www.bbc.com/news/articles/c5ymvdm5520o", ""),
            ("https://example.com/", ""),
            ("https://example.com/markets/oil", ""),
        ],
    )
    def test_a_headline_is_recovered_from_the_link_when_there_is_one(self, url, expected):
        assert headline_from_url(url) == expected

    def test_the_error_carries_the_headline_as_a_way_forward(self):
        error = FetchError(
            "that site blocks automated readers",
            headline_from_url(
                "https://www.reuters.com/markets/oil-prices-slide-on-supply-glut-2026-08-30/"
            ),
        )
        assert error.suggestion == "Oil prices slide on supply glut"

    def test_a_plain_failure_offers_nothing(self):
        assert FetchError("no readable text on that page").suggestion == ""


class TestPagesThatAreNotArticles:
    """A front page, a section index, or an app that draws its text in the browser all leave a two-
    word page title and nothing else.
    """

    FRONT_PAGE = "<html><head><title>Your briefing</title></head><body><div>x</div></body></html>"
    SECTION = "<html><head><title>Markets</title></head><body><p>Sign in</p></body></html>"

    @pytest.mark.parametrize("html", [FRONT_PAGE, SECTION])
    def test_a_page_title_is_not_a_headline(self, html):
        with pytest.raises(FetchError, match="no article on that page"):
            extract(html)

    def test_a_real_headline_survives_without_a_body(self):
        html = "<html><h1>First Abu Dhabi Bank cuts its full-year profit guidance</h1></html>"
        title, body = extract(html)
        assert body == ""
        assert Article("https://x.com/a", title, body).headline_only

    def test_a_whole_article_is_not_headline_only(self):
        title, body = extract(PAGE)
        assert not Article("https://x.com/a", title, body).headline_only
