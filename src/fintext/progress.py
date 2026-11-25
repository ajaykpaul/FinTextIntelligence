"""Progress bars for the long jobs: training, evaluation, translation, loading."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def enabled() -> bool:
    if os.environ.get("FINTEXT_PROGRESS", "1") == "0":
        return False
    try:
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    **options: Any,
) -> Iterable[T]:
    """`iterable`, with a progress bar when one can be shown."""
    if not enabled():
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True, **options)


def note(bar: Iterable[Any], **values: Any) -> None:
    """Show values (a loss, a count) at the end of a bar."""
    setter = getattr(bar, "set_postfix", None)
    if setter is not None:
        setter(values, refresh=False)


class ByteBar:
    """A bar counted in bytes, for copying or unpacking large files."""

    def __init__(self, total: int, desc: str) -> None:
        self._bar = None
        if enabled():
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=total,
                desc=desc,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            )

    def update(self, count: int) -> None:
        if self._bar is not None:
            self._bar.update(count)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> ByteBar:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
