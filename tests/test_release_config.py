"""The release config serves the current release with that model's own thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fintext.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "configs" / "release_v4.1.toml"


def test_release_config_matches_the_registry_entry():
    config = load_config(RELEASE, use_env=False)
    entries = json.loads((ROOT / "artifacts/registry/index.json").read_text())["entries"]
    entry = entries[config.model_version]
    tuned = entry["metrics"]["thresholds"]
    assert config.backend == "onnx"
    assert config.toxicity_en == pytest.approx(tuned["toxicity_en"], abs=1e-6)
    assert config.toxicity_ar == pytest.approx(tuned["toxicity_ar"], abs=1e-6)
    assert config.misleading == pytest.approx(tuned["misleading_en"], abs=1e-6)
    # The high-priority bar must stay above the flag bar.
    assert config.misleading_high_priority > config.misleading


def test_release_differs_from_default_only_where_it_should():
    default = load_config(ROOT / "configs" / "default.toml", use_env=False)
    release = load_config(RELEASE, use_env=False)
    changed = {
        name
        for name in default.__dataclass_fields__
        if getattr(default, name) != getattr(release, name)
    }
    assert changed == {
        "backend",
        "model_version",
        "threshold_version",
        "toxicity_en",
        "toxicity_ar",
        "misleading",
    }
