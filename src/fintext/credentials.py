"""Finding the HuggingFace token without ever storing it in the project."""

from __future__ import annotations

import os
from pathlib import Path

ENV_NAME = "HF_TOKEN"
"""The variable huggingface_hub reads. Fixed by that library, so a Colab
secret under any other name gets copied into it."""

COLAB_SECRETS = ("fintext", "HF_TOKEN")
"""Colab secret names to try, in order. `fintext` is the one this project uses."""


def _from_colab() -> tuple[str, str] | None:
    """(secret name, token) for the first secret found, or None."""
    try:
        from google.colab import userdata  # only exists on Colab
    except ImportError:
        return None
    for name in COLAB_SECRETS:
        try:
            token = userdata.get(name)
        except Exception:
            # Not in Secrets, or this notebook hasn't been given access to it.
            continue
        if token:
            return name, token
    return None


def _saved_login() -> str | None:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "token"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def huggingface_token_source() -> str | None:
    """Where the token was found, for status messages."""
    if os.environ.get(ENV_NAME):
        return "HF_TOKEN environment variable"
    found = _from_colab()
    if found:
        return f"Colab secret {found[0]}"
    if _saved_login():
        return "saved login (hf auth login)"
    return None


def use_huggingface_token() -> str | None:
    """Make the token visible to huggingface_hub, and say where it came from."""
    if os.environ.get(ENV_NAME):
        return "HF_TOKEN environment variable"
    found = _from_colab()
    if found:
        os.environ[ENV_NAME] = found[1]
        return f"Colab secret {found[0]}"
    return huggingface_token_source()
