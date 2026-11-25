"""Bringing a trained run back from the notebook into artifacts/models."""

from __future__ import annotations

import json
import zipfile

import pytest

from fintext.registry import RUN_FILES, RegistryError, import_run


def bundle(tmp_path, *, prefix="run-1/", files=RUN_FILES, extra=(), run_name="run-1"):
    path = tmp_path / "run.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name in files:
            content = json.dumps({"run_name": run_name}) if name == "manifest.json" else "x"
            z.writestr(prefix + name, content)
        for name in extra:
            z.writestr(name, "x")
    return path


def test_a_complete_run_lands_under_its_own_name(tmp_path):
    run = import_run(bundle(tmp_path), tmp_path / "models")
    assert run == tmp_path / "models" / "run-1"
    assert {f.name for f in run.iterdir()} == set(RUN_FILES)


def test_an_archive_without_a_top_folder_also_works(tmp_path):
    run = import_run(bundle(tmp_path, prefix=""), tmp_path / "models")
    assert (run / "pytorch_model.bin").exists()


def test_the_resume_checkpoint_is_left_behind(tmp_path):
    path = bundle(tmp_path, files=RUN_FILES + ("checkpoint.pt",))
    run = import_run(path, tmp_path / "models")
    assert not (run / "checkpoint.pt").exists()


def test_unsafe_paths_are_refused_before_anything_is_written(tmp_path):
    path = bundle(tmp_path, extra=("../escape.txt",))
    with pytest.raises(RegistryError, match="unsafe"):
        import_run(path, tmp_path / "models")
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_an_incomplete_run_is_refused(tmp_path):
    path = bundle(tmp_path, files=("manifest.json", "tokenizer.json"))
    with pytest.raises(RegistryError, match="missing pytorch_model.bin"):
        import_run(path, tmp_path / "models")


def test_a_run_is_never_overwritten(tmp_path):
    import_run(bundle(tmp_path), tmp_path / "models")
    with pytest.raises(RegistryError, match="already exists"):
        import_run(bundle(tmp_path), tmp_path / "models")


def test_a_run_name_cannot_point_outside_the_models_folder(tmp_path):
    with pytest.raises(RegistryError, match="run_name"):
        import_run(bundle(tmp_path, run_name="../elsewhere"), tmp_path / "models")
