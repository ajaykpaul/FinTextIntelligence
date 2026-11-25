"""Progress bars: shown when tqdm is there, silent and harmless when it isn't."""

from __future__ import annotations

import subprocess
import sys

from fintext import progress


def test_the_items_come_through_unchanged():
    assert list(progress.progress(range(5), desc="x")) == [0, 1, 2, 3, 4]


def test_bars_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("FINTEXT_PROGRESS", "0")
    items = [1, 2, 3]
    assert progress.progress(items, desc="x") is items  # the plain iterable, no wrapper
    assert not progress.enabled()


def test_without_tqdm_everything_still_works(monkeypatch):
    monkeypatch.setitem(sys.modules, "tqdm", None)  # makes `import tqdm` fail
    assert not progress.enabled()
    items = [1, 2]
    bar = progress.progress(items, desc="x")
    assert bar is items
    progress.note(bar, loss="1.0")  # a no-op, not an error
    with progress.ByteBar(100, desc="copying") as copy:
        copy.update(50)


def test_the_service_does_not_import_tqdm():
    code = "import sys; import fintext.server, fintext.registry; print('tqdm' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": ""},
        check=True,
    )
    assert out.stdout.strip() == "False"
