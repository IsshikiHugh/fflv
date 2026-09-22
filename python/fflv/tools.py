"""Validation, `pack`, `render` / `extract` and the viewer, from Python."""

from __future__ import annotations

import os
from typing import Callable, Iterable

from . import _fflv, _util
from .report import Report

Progress = Callable[[int, int], None]


def validate(path) -> Report:
    """Check a file's structure, metadata and the format invariants I1–I10."""
    return Report.from_json(_fflv.validate(os.fspath(path)))


def pack(project, output=None, *, threads: int | None = None, log: Callable[[str], None] | None = print,
         progress: Progress | None = None) -> Report:
    """Build an .lvd from a project JSON (spec 8.1). Returns the validation report.

    `progress(done, total)` is called after every frame; an exception from it (or Ctrl+C) stops
    the pack and leaves any previous output as it was. An exception from `log` stops the pack at the
    next frame; raised by one of the last lines, it comes after the file was written."""
    threads = None if threads is None else _util.uint(threads, "threads")
    out = None if output is None else os.fspath(output)
    return Report.from_json(_fflv.pack(os.fspath(project), out, threads, log, progress))


def render(path, output, *, layers: Iterable | None = None, hide: Iterable | None = None, start: int = 0,
           end: int | None = None, transparent: bool = False, crf: int = 18, progress: Progress | None = None) -> int:
    """Composite the chosen layers over [start, end) into `output`; returns the frames written.

    The output type follows the path: out.png / out.jpg (one frame), frames/%05d.png or dir/ (one
    image per frame, named by frame index), .mp4 / .mov / .webm / .mkv, or .npy (N×H×W×C uint8).
    `progress(done, total)` is called after every frame; an exception from it (or Ctrl+C) stops the
    render, leaving the frames written so far (a video or .npy output is then truncated).
    """
    keys = None if layers is None else _util.keys(layers)
    hidden = [] if hide is None else _util.keys(hide, "hide")
    return _fflv.render_file(os.fspath(path), os.fspath(output), keys, hidden, _util.uint(start, "start"),
                             _util.frame_arg(end, "end"), bool(transparent), _util.uint(crf, "crf", 63), progress)


def extract(path, layer, output, *, start: int | None = None, end: int | None = None, crf: int = 18,
            progress: Progress | None = None) -> int:
    """Write one layer's own pixels (RGBA, content size) for the frames where it is active."""
    return _fflv.extract_layer(os.fspath(path), _util.key(layer), os.fspath(output), _util.frame_arg(start, "start"),
                               _util.frame_arg(end, "end"), _util.uint(crf, "crf", 63), progress)


def view(path, *, port: int = 0, host: str = "127.0.0.1", browser: str | None = None, open_page: bool = True,
         ready: Callable[[str], None] | None = None) -> None:
    """Serve the web player for `path` and open it in Chrome / Edge; blocks until Ctrl+C
    (KeyboardInterrupt). `ready(url)` is called once the server listens (default: print the URL).
    Raises ViewError when the file cannot be served."""
    if ready is None:
        def ready(url: str) -> None:
            print(f"serving {os.fspath(path)} at\n  {url}\n(the page follows changes to the file; Ctrl+C to stop)",
                  flush=True)
    _fflv.view(os.fspath(path), host, _util.uint(port, "port", 65535), browser, bool(open_page), ready)
