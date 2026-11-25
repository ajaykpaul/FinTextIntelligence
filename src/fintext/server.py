"""HTTP interface."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .approvals import ApprovalError, ApprovalQueue, Waiting
from .config import Config, load_config
from .fetch import FetchError, UnsafeURL, fetch_article, looks_like_url
from .pipeline import Pipeline, ScoringFailed
from .registry import Registry, RegistryError
from .review import (
    MIN_BATCH_SIZE,
    VERDICTS,
    Batch,
    ReviewStore,
    build_report,
    sample_batch,
    to_csv,
)
from .schemas import ScoreRequest, ValidationError
from .scoring import load_backend
from .signal import Direction, decide

logger = logging.getLogger("fintext.server")

CLAIM_PATH = re.compile(r"^/v1/provenance/([0-9a-f]{8,64})$")
UI_ASSET = re.compile(r"^/ui/(assets/[\w.\-]+)$")

STATIC_DIR = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

# Advertised at GET /. A test asserts every GET route here actually resolves, so the list cannot
# drift from the router.
ROUTES: tuple[tuple[str, str], ...] = (
    ("GET /ui", "a page for pasting a news item and seeing it scored (text/html)"),
    ("GET /healthz", "liveness; does not depend on the model"),
    ("GET /readyz", "readiness: ready, ready_degraded or starting"),
    ("GET /metrics", "Prometheus exposition (text/plain, served by the transport)"),
    ("GET /v1/metrics", "JSON metrics snapshot, including per-stage p95 and drift"),
    ("GET /v1/model", "serving lineage and registry pointers"),
    ("GET /v1/config", "thresholds and windows a consumer needs to read a score"),
    ("GET /v1/queue/compliance", "claims whose corroboration window expired"),
    ("GET /v1/review/batch", "this week's sample of flagged items, with verdicts so far"),
    ("POST /v1/review/batch", "draw a new weekly sample: {days?, size?, seed?}"),
    ("POST /v1/review/decision", "record a verdict: {alert_id, verdict, reviewer, note?}"),
    ("GET /v1/review/precision", "measured review precision per language"),
    ("GET /v1/provenance/stats", "corroboration counts and time-to-corroboration"),
    ("GET /v1/provenance/{claim_id}", "one tracked claim and its transitions"),
    ("POST /v1/score", "score one item: {text, profile, capabilities?}; text may be a link"),
    ("POST /v1/score:batch", "score many: {items: [...]}"),
    ("POST /v1/signal", "score one item and return a trading decision with it"),
    ("GET /v1/approvals", "gated calls waiting for a person, urgent first"),
    ("POST /v1/approvals/resolve", "accept a gated call as buy or sell, or reject it"),
    ("POST /v1/admin/recheck", "re-check open claims; called by the scheduler"),
)
MAX_BODY_BYTES = 4 * 1024 * 1024


class App:
    def __init__(self, pipeline: Pipeline, registry: Registry | None = None) -> None:
        self.pipeline = pipeline
        self.registry = registry
        self.config = pipeline.config
        review_dir = self.config.path(self.config.review_dir)
        self.batch_path = review_dir / "batch.json"
        self.reviews = ReviewStore(review_dir / "decisions.jsonl")
        self.approvals = ApprovalQueue(review_dir / "approvals.jsonl")
        self.ready = False
        self.warmup_error: str | None = None

    def startup(self) -> None:
        """Warm the backend, then accept traffic."""
        try:
            self.pipeline.warmup()
        except Exception as exc:
            self.warmup_error = str(exc)
            self.pipeline.monitor.error("warmup")
        self.ready = True

    def shutdown(self) -> None:
        self.ready = False
        self.pipeline.close()

    def handle(
        self, method: str, path: str, body: bytes | str | None = None
    ) -> tuple[int, dict[str, Any]]:
        path = path.rstrip("/") or "/"
        try:
            if method.upper() == "GET":
                return self._get(path)
            if method.upper() == "POST":
                return self._post(path, body)
            return 405, {"error": "method_not_allowed", "detail": method}
        except ApprovalError as exc:
            return 409, {"error": "cannot_resolve", "detail": str(exc)}
        except UnsafeURL as exc:
            return 400, {"error": "invalid_request", "detail": str(exc)}
        except FetchError as exc:
            # The caller's request was fine; the far end would not hand over the article. 502
            # says the failure is upstream, not in what was sent.
            payload: dict[str, Any] = {"error": "link_unreachable", "detail": str(exc)}
            if exc.suggestion:
                payload["suggestion"] = exc.suggestion
            return 502, payload
        except ValidationError as exc:
            return 400, {"error": "invalid_request", "detail": str(exc)}
        except RegistryError as exc:
            return 404, {"error": "not_found", "detail": str(exc)}
        except ScoringFailed as exc:
            return 503, {"error": "scoring_unavailable", "detail": str(exc)}
        except Exception as exc:
            self.pipeline.monitor.error("unhandled")
            return 500, {"error": "internal_error", "detail": str(exc)}

    def _get(self, path: str) -> tuple[int, dict[str, Any]]:
        if path == "/":
            return 200, self._index()
        if path == "/healthz":
            return 200, {"status": "ok", "version": __version__}
        if path == "/readyz":
            if not self.ready:
                return 503, {"status": "starting"}
            if self.warmup_error:
                return 200, {"status": "ready_degraded", "warmup_error": self.warmup_error}
            return 200, {"status": "ready"}
        if path == "/v1/metrics":
            return 200, self.pipeline.monitor.snapshot()
        if path == "/v1/config":
            # Thresholds and windows are part of the contract a consumer needs to interpret a
            # score. No secrets live in this config.
            return 200, {"digest": self.config.digest(), **asdict(self.config)}
        if path == "/v1/model":
            return 200, self._model_info()
        if path == "/v1/provenance/stats":
            return 200, self.pipeline.tracker.statistics()
        if path == "/v1/queue/compliance":
            return 200, self._compliance_queue()
        if path == "/v1/review/batch":
            batch = Batch.load(self.batch_path)
            if batch is None:
                return 404, {"error": "no_batch", "detail": "draw one with POST /v1/review/batch"}
            decided = self.reviews.latest()
            return 200, {
                **batch.to_dict(),
                "decisions": {k: asdict(v) for k, v in decided.items()},
                "outstanding": sum(1 for a in batch.alerts if a.alert_id not in decided),
            }
        if path == "/v1/approvals":
            return 200, {
                "items": [item.to_dict() for item in self.approvals.waiting()],
                **self.approvals.summary(),
            }
        if path == "/v1/review/precision":
            return 200, self._precision().to_dict()
        if match := CLAIM_PATH.match(path):
            record = self.pipeline.tracker.get(match.group(1))
            if record is None:
                return 404, {"error": "not_found", "detail": "unknown claim id"}
            return 200, record.to_dict()
        return 404, {"error": "not_found", "detail": f"no route for GET {path}"}

    def _post(self, path: str, body: bytes | str | None) -> tuple[int, dict[str, Any]]:
        if path == "/v1/score":
            request, fetched = self._resolve(ScoreRequest.from_dict(parse_json(body)))
            payload = self.pipeline.score(request).to_dict()
            if fetched:
                payload["fetched"] = fetched
            return 200, payload

        if path == "/v1/score:batch":
            payload = parse_json(body)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValidationError("batch requests require an 'items' array")
            if len(items) > self.config.max_batch:
                raise ValidationError(
                    f"batch of {len(items)} exceeds the limit of {self.config.max_batch}"
                )
            defaults = {k: v for k, v in payload.items() if k != "items"}
            responses = self.pipeline.score_many(
                [ScoreRequest.from_dict({**defaults, **item}) for item in items]
            )
            return 200, {"count": len(responses), "results": [r.to_dict() for r in responses]}

        if path == "/v1/approvals/resolve":
            payload = parse_json(body)
            resolution = self.approvals.resolve(
                str(payload.get("item_id", "")),
                action=str(payload.get("action", "")),
                reviewer=str(payload.get("reviewer", "")),
                direction=str(payload.get("direction", "")),
                note=str(payload.get("note", "")),
            )
            self.pipeline.log.write("approval", resolution.to_dict())
            return 200, {"resolution": resolution.to_dict(), **self.approvals.summary()}

        if path == "/v1/review/batch":
            payload = parse_json(body) if body else {}
            days = float(payload.get("days", 7))
            now = time.time()
            drawn = sample_batch(
                self.config.path(self.config.decision_log) if self.config.decision_log else "",
                window_start=now - days * 86400,
                window_end=now,
                size=int(payload.get("size", MIN_BATCH_SIZE)),
                seed=payload.get("seed"),
            )
            drawn.save(self.batch_path)
            return 200, drawn.to_dict()

        if path == "/v1/review/decision":
            payload = parse_json(body)
            batch = Batch.load(self.batch_path)
            if batch is None:
                raise ValidationError("no review batch has been drawn")
            alert_id = payload.get("alert_id")
            alert = next((a for a in batch.alerts if a.alert_id == alert_id), None)
            if alert is None:
                return 404, {"error": "not_found", "detail": f"no alert {alert_id!r} in this batch"}
            verdict = str(payload.get("verdict", ""))
            if verdict not in VERDICTS:
                raise ValidationError(f"verdict must be one of {VERDICTS}")
            decision = self.reviews.record(
                alert,
                verdict=verdict,
                reviewer=str(payload.get("reviewer") or "unattributed"),
                note=str(payload.get("note", "")),
            )
            return 200, {"decision": asdict(decision), "precision": self._precision().to_dict()}

        if path == "/v1/signal":
            # separate route so /v1/score never carries an action
            request, fetched = self._resolve(ScoreRequest.from_dict(parse_json(body)))
            response = self.pipeline.score(request)
            signal = decide(response, self.config)
            payload = {"signal": signal.to_dict(), "score": response.to_dict()}
            # review calls go to the approval queue
            if signal.direction is Direction.REVIEW:
                payload["queued"] = self.approvals.add(
                    Waiting(
                        item_id=response.id,
                        text=request.text,
                        language=str(response.language),
                        priority=signal.priority,
                        reason=signal.reason,
                        lean=_lean(response),
                        conviction=signal.conviction,
                        lineage=asdict(response.lineage),
                    )
                ).to_dict()
            if fetched:
                payload["fetched"] = fetched
            return 200, payload

        if path == "/v1/admin/recheck":
            # Called by the scheduler, not by a consumer: pending claims resolve without anyone
            # re-submitting them.
            transitions = self.pipeline.tracker.recheck_due()
            return 200, {
                "transitions": [t.to_dict() for t in transitions],
                "open_claims": len(self.pipeline.tracker.open_claims()),
            }

        return 404, {"error": "not_found", "detail": f"no route for POST {path}"}

    def _index(self) -> dict[str, Any]:
        """What lives where."""
        return {
            "service": "fintext",
            "version": __version__,
            "description": (
                "Sentiment, toxicity, misleading-content triage and provenance signals. "
                "The service reports; the policy layer decides."
            ),
            "ui": "/ui",
            "endpoints": dict(ROUTES),
        }

    def _resolve(self, request: ScoreRequest) -> tuple[ScoreRequest, dict[str, Any] | None]:
        """If the caller pasted a bare link, fetch it and score the article."""
        from dataclasses import replace

        url = looks_like_url(request.text) if self.config.fetch_urls else None
        if not url:
            return request, None
        article = fetch_article(
            url,
            timeout_s=self.config.fetch_timeout_s,
            max_bytes=self.config.fetch_max_bytes,
        )
        return (
            replace(request, text=article.scoreable, source_url=article.url),
            {
                "url": article.url,
                "title": article.title,
                "characters": len(article.scoreable),
                "headline_only": article.headline_only,
            },
        )

    def _model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "service_version": __version__,
            "backend": self.config.backend,
            "active_version": self.pipeline.backend.version,
            "lineage": asdict(self.pipeline.lineage(self.pipeline.backend.version)),
        }
        if self.registry:
            active, shadow = self.registry.active(), self.registry.shadow()
            info["registry"] = {
                "versions": self.registry.versions(),
                "active": asdict(active) if active else None,
                "shadow": asdict(shadow) if shadow else None,
            }
        return info

    def _compliance_queue(self) -> dict[str, Any]:
        """Expired-window claims only."""
        records = self.pipeline.tracker.review_queue()
        return {
            "count": len(records),
            "items": [r.to_dict() for r in records],
            "note": (
                "Uncorroborated means no supporting primary document arrived within a window "
                "calibrated to normal confirmation times. It is not an assertion of falsity."
            ),
        }

    def _precision(self):
        batch = Batch.load(self.batch_path)
        decided = self.reviews.latest()
        if batch is None:
            return build_report(list(decided.values()))
        in_batch = {a.alert_id for a in batch.alerts}
        return build_report(
            [d for alert_id, d in decided.items() if alert_id in in_batch],
            outstanding=sum(1 for a in batch.alerts if a.alert_id not in decided),
        )

    def export_review_csv(self, path: str | Path) -> Path:
        batch = Batch.load(self.batch_path)
        if batch is None:
            raise ValidationError("no review batch has been drawn")
        return to_csv(batch, self.reviews.latest(), path)

    def metrics_text(self) -> str:
        return self.pipeline.monitor.prometheus()

    def static_file(self, name: str) -> tuple[str, bytes] | None:
        """Read one file from the built UI bundle."""
        target = (STATIC_DIR / name).resolve()
        # A request path must not escape the bundle directory.
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            return None
        return CONTENT_TYPES.get(target.suffix, "application/octet-stream"), target.read_bytes()

    @staticmethod
    def ui_missing_message() -> bytes:
        return (
            b"<!doctype html><meta charset=utf-8><title>UI not built</title>"
            b"<body style='font:15px system-ui;padding:40px;max-width:44em'>"
            b"<h1>The UI bundle is not built</h1>"
            b"<p>The React app under <code>web/</code> compiles into this package. "
            b"Build it with:</p>"
            b"<pre style='background:#f4f4f6;padding:12px;border-radius:6px'>"
            b"cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            b"<p>The API itself is unaffected; see <a href='/'>/</a> for the endpoints.</p>"
        )


def parse_json(body: bytes | str | None) -> dict[str, Any]:
    if not body:
        raise ValidationError("request body is empty")
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")
    return payload


def build_app(config: Config | None = None) -> App:
    """Assemble the service, falling back to the heuristic backend."""
    config = config or load_config()
    registry = backend = None
    if config.backend != "heuristic":
        try:
            registry = Registry(
                config.path(config.registry_path), p95_target_ms=config.p95_target_ms
            )
            backend = load_backend(config, registry)
        except Exception as exc:
            logger.warning("falling back to the heuristic backend: %s", exc)

    app = App(Pipeline(config, backend=backend), registry)
    app.startup()
    return app


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------


def serve(config: Config | None = None) -> None:
    """Run on the standard-library HTTP server."""
    config = config or load_config()
    app = build_app(config)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "fintext"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: dict[str, Any] | str, content_type: str) -> None:
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            self._send_bytes(status, text.encode("utf-8"), content_type)

        def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            # The UI and the metrics endpoint are not JSON, so the transport serves them rather
            # than going through the router.
            path = self.path.rstrip("/")
            if path == "/metrics":
                self._send(200, app.metrics_text(), "text/plain; version=0.0.4")
            elif path == "/ui":
                found = app.static_file("index.html")
                if found:
                    self._send_bytes(200, found[1], found[0])
                else:
                    self._send_bytes(503, app.ui_missing_message(), CONTENT_TYPES[".html"])
            elif match := UI_ASSET.match(path):
                found = app.static_file(match.group(1))
                if found:
                    self._send_bytes(200, found[1], found[0])
                else:
                    self._send(404, {"error": "not_found"}, "application/json")
            else:
                self._send(*app.handle("GET", self.path), "application/json")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "payload_too_large"}, "application/json")
                return
            status, payload = app.handle("POST", self.path, self.rfile.read(length))
            self._send(status, payload, "application/json")

    server = ThreadingHTTPServer((config.host, config.port), Handler)
    logger.info(
        "serving on http://%s:%s (backend=%s, config=%s)",
        config.host,
        config.port,
        config.backend,
        config.digest(),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        app.shutdown()


def asgi_app(config: Config | None = None):
    """FastAPI adapter."""
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    app = build_app(config)
    api = FastAPI(title="FinTextIntelligence", version=__version__)

    @api.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return app.metrics_text()

    @api.get("/ui", response_class=HTMLResponse)
    async def ui() -> Response:
        found = app.static_file("index.html")
        if not found:
            return HTMLResponse(app.ui_missing_message(), status_code=503)
        return HTMLResponse(found[1])

    @api.get("/ui/assets/{name}")
    async def ui_asset(name: str) -> Response:
        found = app.static_file(f"assets/{name}")
        if not found:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return Response(found[1], media_type=found[0])

    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def dispatch(request: Request, path: str) -> Response:
        status, payload = app.handle(request.method, "/" + path, await request.body())
        return JSONResponse(status_code=status, content=payload)

    return api


def _lean(response) -> str:
    sentiment = response.sentiment
    if sentiment is None or str(sentiment.label) == "neutral":
        return ""
    return "buy" if str(sentiment.label) == "positive" else "sell"
