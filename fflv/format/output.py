"""Publishing a finished file: every writer (pack, Writer, edits) writes to a hidden temporary file
next to the destination, validates it, and only then atomically renames it over the destination.
Readers of the destination therefore only ever see a complete, valid file (the old or the new one).
"""

from __future__ import annotations

import os
from pathlib import Path

from .validate import Report, validate


class InvalidOutput(RuntimeError):
    """The file just written failed validation; the destination was left untouched."""

    def __init__(self, dst: Path, report: Report):
        self.report = report
        issues = "; ".join(str(i) for i in report.errors[:5]) or "fatal structural error"
        super().__init__(f"{dst} was not written: the result failed validation: {issues}")


def temp_path_for(dst: str | os.PathLike) -> Path:
    """Hidden sibling of `dst` (same directory, so the final rename is atomic)."""
    dst = Path(dst)
    return dst.with_name(f".{dst.name}.fflv-tmp")


def publish(tmp: str | os.PathLike, dst: str | os.PathLike, *, check: bool = True) -> Report | None:
    """Validate `tmp` (unless check=False), then atomically replace `dst` with it.
    On a validation failure `tmp` is deleted, `dst` is untouched, and InvalidOutput is raised."""
    tmp, dst = Path(tmp), Path(dst)
    rep = None
    if check:
        rep = validate(str(tmp))
        if not rep.ok:
            tmp.unlink(missing_ok=True)
            raise InvalidOutput(dst, rep)
    os.replace(tmp, dst)
    return rep
