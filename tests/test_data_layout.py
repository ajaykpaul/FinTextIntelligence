"""The data folder: what goes in it, and what must never leave in an archive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fintext import catalogue, corpora

ROOT = Path(__file__).resolve().parent.parent


class TestNotRedistributable:
    def test_the_materialised_folder_is_git_ignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
        assert "data/sources/" in ignored
        assert "data/raw/" in ignored

    def test_the_sources_that_need_that_protection_are_marked(self):
        for key in ("financial_phrasebank", "jigsaw_toxic"):
            assert not catalogue.get(key).redistributable

    def test_only_the_fixtures_are_committed(self):
        committed = {p.name for p in (ROOT / "data").iterdir() if p.is_dir()}
        assert "samples" in committed


class TestManifests:
    def test_a_manifest_records_what_a_reader_needs(self, tmp_path, monkeypatch):
        """A folder of rows with no provenance is an unattributed dataset."""
        monkeypatch.setitem(
            corpora.LOADERS,
            "liar",
            lambda limit: [{"text": "a claim", "language": "en", "labels": {"misleading": 1.0}}],
        )
        corpora.materialise(("liar",), out_dir=tmp_path)

        manifest = json.loads((tmp_path / "liar" / "manifest.json").read_text())
        assert manifest["declared_source"] == "liar"
        assert manifest["hub_path"] == corpora.MIRRORS["liar"].path
        assert manifest["licence"] == catalogue.get("liar").licence
        assert manifest["rows"] == 1
        assert (tmp_path / "INDEX.json").exists()

    def test_rows_are_written_post_mapping(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            corpora.LOADERS,
            "liar",
            lambda limit: [{"text": "a claim", "language": "en", "labels": {"misleading": 0.0}}],
        )
        corpora.materialise(("liar",), out_dir=tmp_path)
        row = json.loads((tmp_path / "liar" / "rows.jsonl").read_text().strip())
        assert row["labels"] == {"misleading": 0.0}

    def test_an_eval_only_source_says_so_on_the_tin(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            corpora.EVAL_ONLY_LOADERS,
            "arsas",
            lambda limit: [{"text": "جملة", "language": "ar", "labels": {"sentiment": "neutral"}}],
        )
        corpora.materialise((), eval_sources=("arsas",), out_dir=tmp_path)
        manifest = json.loads((tmp_path / "arsas" / "manifest.json").read_text())
        assert manifest["eval_only"] is True

    def test_an_undeclared_source_cannot_be_materialised(self, tmp_path):
        with pytest.raises(catalogue.LicenceError):
            corpora.materialise(("nonesuch",), out_dir=tmp_path)
