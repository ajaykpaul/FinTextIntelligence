"""Finding the HuggingFace token on a Mac, on Colab, or anywhere else."""

from __future__ import annotations

import sys
import types

import pytest

from fintext import credentials

SECRET = "hf_test_not_a_real_token"


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # no saved login unless a test writes one
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)


def fake_colab(monkeypatch, value, name="fintext"):
    def get(asked):
        if asked != name:
            raise KeyError(asked)  # what Colab does for a secret that isn't there
        return value

    userdata = types.SimpleNamespace(get=get)
    colab = types.ModuleType("google.colab")
    colab.userdata = userdata
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_no_token_anywhere_means_anonymous():
    assert credentials.use_huggingface_token() is None


def test_an_environment_variable_works_everywhere(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    assert credentials.use_huggingface_token() == "HF_TOKEN environment variable"


def test_a_colab_secret_is_copied_into_the_environment(monkeypatch):
    fake_colab(monkeypatch, SECRET)
    assert credentials.use_huggingface_token() is not None
    import os

    assert os.environ["HF_TOKEN"] == SECRET


def test_a_saved_login_on_a_mac_is_found(tmp_path):
    (tmp_path / "token").write_text(SECRET + "\n")
    assert credentials.use_huggingface_token() == "saved login (hf auth login)"


def test_the_environment_wins_over_colab(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    fake_colab(monkeypatch, SECRET)
    credentials.use_huggingface_token()
    import os

    assert os.environ["HF_TOKEN"] == "from-env"


def test_the_token_itself_is_never_reported(monkeypatch, capsys, caplog):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    source = credentials.use_huggingface_token()
    assert SECRET not in (source or "")
    assert SECRET not in capsys.readouterr().out + caplog.text


def test_the_project_never_holds_a_token():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert ".env" in (root / ".gitignore").read_text().split()
    assert '"*.env"' in (root / "make_archive.sh").read_text()


def test_the_fintext_secret_is_what_colab_uses(monkeypatch):
    fake_colab(monkeypatch, SECRET, name="fintext")
    assert credentials.use_huggingface_token() == "Colab secret fintext"
    import os

    assert os.environ["HF_TOKEN"] == SECRET  # where huggingface_hub reads it


def test_a_secret_named_hf_token_still_works(monkeypatch):
    fake_colab(monkeypatch, SECRET, name="HF_TOKEN")
    assert credentials.use_huggingface_token() == "Colab secret HF_TOKEN"
