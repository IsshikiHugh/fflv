"""Validation, `pack`, `render` / `extract` and the viewer, from Python."""

from __future__ import annotations

import os
import signal
import threading
from typing import Callable, Iterable

from . import _fflv, _util
from .report import Report

Progress = Callable[[int, int], None]


def validate(path) -> Report:
    """Check a file's structure, metadata and the format invariants I1–I10."""
    return Report.from_json(_fflv.validate(os.fspath(path)))


def pack(project, output=None, *, threads: int | None = None, log: Callable[[str], None] | None = print) -> Report:
    """Build an .lvd from a project JSON (spec 8.1). Returns the validation report."""
    return Report.from_json(_fflv.pack(os.fspath(project), None if output is None else os.fspath(output), threads, log))


def render(path, output, *, layers: Iterable | None = None, hide: Iterable | None = None, start: int = 0,
           end: int | None = None, transparent: bool = False, crf: int = 18, progress: Progress | None = None) -> int:
    """Composite the chosen layers over [start, end) into `output`; returns the frames written.

    The output type follows the path: out.png / out.jpg (one frame), frames/%05d.png or dir/ (one
    image per frame, named by frame index), .mp4 / .mov / .webm / .mkv, or .npy (N×H×W×C uint8).
    """
    keys = None if layers is None else [_util.key(k) for k in layers]
    return _fflv.render_file(os.fspath(path), os.fspath(output), keys, [_util.key(k) for k in (hide or [])], int(start), _util.frame_arg(end, "end"),
                             bool(transparent), int(crf), progress)


def extract(path, layer, output, *, start: int | None = None, end: int | None = None, crf: int = 18,
            progress: Progress | None = None) -> int:
    """Write one layer's own pixels (RGBA, content size) for the frames where it is active."""
    return _fflv.extract_layer(os.fspath(path), _util.key(layer), os.fspath(output), _util.frame_arg(start, "start"),
                               _util.frame_arg(end, "end"), int(crf), progress)


def view(path, *, port: int = 0, host: str = "127.0.0.1", browser: str | None = None, open_page: bool = True) -> None:
    """Serve the interactive player for `path` and open it in Chrome / Edge; blocks until Ctrl+C."""
    argv = ["fflv", "view", os.fspath(path), "--port", str(int(port)), "--host", host]
    if browser:
        argv += ["--browser", browser]
    if not open_page:
        argv.append("--no-open")
    main_thread = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGINT, signal.SIG_DFL) if main_thread else None  # Ctrl+C ends the server
    try:
        _fflv.cli_main(argv)
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, old)
