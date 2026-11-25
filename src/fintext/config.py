"""Service configuration, loaded once at start-up and hashed."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    # Latency (PRD 8.3). 50 ms p95 objective, 75 ms hard timeout.
    max_tokens: int = 256
    timeout_ms: float = 75.0
    trading_budget_ms: float = 50.0
    compliance_budget_ms: float = 500.0
    research_budget_ms: float = 2000.0
    p95_target_ms: float = 50.0

    # Thresholds. Recall-first for toxicity, bounded by the 90% weekly review-precision floor
    # (PRD 8.1, 8.4).
    threshold_version: str = "thresholds.v1"
    toxicity_en: float = 0.35
    toxicity_ar: float = 0.30
    misleading: float = 0.50
    misleading_high_priority: float = 0.75
    min_sentiment_confidence: float = 0.45

    # Provenance. Windows come from observed confirmation times, never set at runtime (PRD 6.3).
    windows: dict[str, int] = field(
        default_factory=lambda: {
            "regulatory": 900,
            "guidance": 3600,
            "merger": 7200,
            "contract": 10800,
        }
    )
    min_match_score: float = 0.82
    recheck_seconds: int = 60
    index_stale_seconds: int = 3600
    index_path: str = "artifacts/provenance/primary_index.jsonl"

    # Lexicons and models.
    lexicon_dir: str = "artifacts/lexicons"
    lexicon_en: str = "en_high_severity.v1.txt"
    lexicon_ar: str = "ar_osact_published.v1.txt"
    lexicons_enabled: bool = True
    backend: str = "heuristic"
    registry_path: str = "artifacts/registry"
    model_version: str = "shared-xlmr.v0-heuristic"
    onnx_threads: int = 4

    # The trading gate (PRD 5.2, 6.4).
    signal_min_confidence: float = 0.60
    pending_conviction: float = 0.85
    pending_hold_ms: int = 0

    # Resolving a pasted link. Off the latency path: fetching takes as long as the far end
    # takes, so only a person pasting a link gets here.
    fetch_urls: bool = True
    fetch_timeout_s: float = 8.0
    fetch_max_bytes: int = 2_000_000

    # Serving.
    host: str = "127.0.0.1"
    port: int = 8080
    fail_open: tuple[str, ...] = ("trading", "research")
    fail_closed: tuple[str, ...] = ("compliance",)
    decision_log: str | None = "reports/decision_log.jsonl"
    review_dir: str = "reports/review"

    retain_text_for_training: bool = True
    max_batch: int = 512
    shadow_version: str | None = None
    shadow_every: int = 0  # shadow one request in N; 0 disables

    root: str = str(PROJECT_ROOT)

    def budget_ms(self, profile: str) -> float:
        return {
            "trading": self.trading_budget_ms,
            "compliance": self.compliance_budget_ms,
            "research": self.research_budget_ms,
        }.get(profile, self.timeout_ms)

    def toxicity_threshold(self, language: str) -> float:
        return self.toxicity_ar if language == "ar" else self.toxicity_en

    def window(self, claim_type: str) -> int:
        if claim_type not in self.windows:
            raise ConfigError(f"no corroboration window configured for {claim_type!r}")
        return self.windows[claim_type]

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else Path(self.root) / candidate

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def replace(self, **changes: Any) -> Config:
        updated = replace(self, **changes)
        updated.validate()
        return updated

    def validate(self) -> None:
        if self.max_tokens > 256:
            raise ConfigError("max_tokens exceeds the 256-token hot-path cap (PRD 6.6)")
        if self.trading_budget_ms > self.timeout_ms:
            raise ConfigError("trading budget cannot exceed the timeout budget")
        if not 1 <= self.onnx_threads <= 64:
            raise ConfigError("onnx_threads must be between 1 and 64")
        if self.backend not in ("heuristic", "onnx"):
            raise ConfigError(f"unknown backend {self.backend!r}")
        both = set(self.fail_open) & set(self.fail_closed)
        if both:
            raise ConfigError(f"profiles are both fail-open and fail-closed: {sorted(both)}")
        for claim_type, seconds in self.windows.items():
            if seconds <= 0:
                raise ConfigError(f"window for {claim_type!r} must be positive")


_TUPLES = {"fail_open", "fail_closed"}


def load_config(path: str | Path | None = None, *, use_env: bool = True) -> Config:
    """Load from TOML, then apply FINTEXT_ environment overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    elif path is not None:
        raise ConfigError(f"config file not found: {config_path}")

    if use_env:
        raw.update(_env_overrides())

    known = set(Config.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")

    values: dict[str, Any] = {k: tuple(v) if k in _TUPLES else v for k, v in raw.items()}
    # Keys are checked against the field names above; TOML supplies the types.
    config = Config(**values)
    config.validate()
    return config


def _env_overrides() -> dict[str, Any]:
    """FINTEXT_TOXICITY_EN=0.4 overrides one field, parsed as a TOML scalar."""
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("FINTEXT_"):
            continue
        name = key[len("FINTEXT_") :].lower()
        if name not in Config.__dataclass_fields__:
            continue
        try:
            overrides[name] = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            overrides[name] = value
    return overrides
