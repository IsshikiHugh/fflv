"""Editing .lvd files without re-encoding what is already there.

  add_layer    new video layer from a media file, a sequence/iterator of numpy images or a callable
  add_still    new still layer (PNG / image / array)
  remove_layers, set_audio (replace or remove), set_layer (id, name, z, rect, blend, opacity, visible)

Existing layers are never re-encoded: their packets are copied bit for bit into a rewritten file
(written beside the output, validated, then atomically renamed over it). Only a newly added video
layer is encoded, with key frames exactly on the file's existing random-access points, so random
access stays the same. `set_layer` rewrites the metadata in place — instant, whatever the file
size — when it fits the space reserved after the metadata (it always does for ordinary edits).
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Iterable

import numpy as np

from . import _fflv, _util
from ._fflv import MetaError
from .report import Report

EDITABLE_FIELDS = ("id", "name", "z", "rect", "blend", "opacity", "visible")


def _out(output) -> str | None:
    return None if output is None else os.fspath(output)


def add_layer(path, id: str, source, *, output=None, start: int = 0, end: int | None = None,
              alpha: bool | None = None, lossless: bool = False, rect=None, z: float | None = None,
              name: str | None = None, blend: str = "normal", opacity: float = 1.0, visible: bool = True,
              crf: int = 32, speed: str = "balanced", check: bool = True) -> Report | None:
    """Append a video layer.

    `source`: a media file (scaled to `rect`, default the whole canvas), a sequence or iterator of
    images (one array: a frame stack N×H×W or N×H×W×C), or a callable `frame -> image`. Images
    are H×W / H×W×3 / H×W×4 arrays (see `fflv.Writer.write`); with arrays, `rect` defaults to the
    first image's size at (0, 0).
    `alpha=None` picks alpha if the source has an alpha channel. Frames: [start, end), `end`
    defaults to the source length (sequences) or the end of the file.
    """
    media, images, length = None, None, None
    if isinstance(source, (str, os.PathLike)):
        media = os.fspath(source)
    elif callable(source):
        images = (_util.as_uint8(source(f), id) for f in itertools.count(_util.uint(start, "start")))
    else:
        if isinstance(source, np.ndarray):
            _check_frame_stack(source, id)
        length = len(source) if hasattr(source, "__len__") else None
        images = (_util.as_uint8(img, id) for img in source)
    rep = _fflv.add_layer(os.fspath(path), id, media, images, length, _out(output), _util.uint(start, "start"),
                          _util.frame_arg(end, "end"), alpha, bool(lossless), _util.rect(rect), _util.number(z, "z"),
                          name, blend, _util.number(opacity, "opacity"), bool(visible), _util.uint(crf, "crf", 63), speed,
                          bool(check))
    return Report.from_json(rep)


def _check_frame_stack(a: np.ndarray, id: str) -> None:
    """An array source is a stack of frames: N×H×W (gray) or N×H×W×C (C = 1, 3 or 4).

    N×H×W×C is unambiguous. With three dimensions, a last dimension of 1, 3 or 4 is taken for one
    H×W×C image (a stack of gray frames 1, 3 or 4 pixels wide is not a likely thing to pass), which
    would otherwise be read as H frames of W×C gray pixels.
    """
    one_image = MetaError(f"{id}: the source looks like one image (shape {a.shape}); give a list of images "
                          f"or a frame stack N×H×W / N×H×W×C (e.g. [image] or image[None] for one frame)")
    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (1, 3, 4)):
        raise one_image
    if not (a.ndim == 3 or (a.ndim == 4 and a.shape[3] in (1, 3, 4))):
        raise MetaError(f"{id}: a frame stack must be N×H×W or N×H×W×C (C = 1, 3 or 4), got shape {a.shape}")


def add_still(path, id: str, image, *, output=None, rect=None, start: int = 0, end: int | None = None,
              z: float | None = None, name: str | None = None, blend: str = "normal", opacity: float = 1.0,
              visible: bool = True, check: bool = True) -> Report | None:
    """Append a still layer shown in frames [start, end) (default: the whole file)."""
    rep = _fflv.add_still(os.fspath(path), id, _util.still_png(image), _out(output), _util.rect(rect),
                          _util.uint(start, "start"),
                          _util.frame_arg(end, "end"), _util.number(z, "z"), name, blend,
                          _util.number(opacity, "opacity"), bool(visible), bool(check))
    return Report.from_json(rep)


def remove_layers(path, keys: Iterable, *, output=None, check: bool = True) -> Report | None:
    keys = _util.keys(keys)
    return Report.from_json(_fflv.remove_layers(os.fspath(path), keys, _out(output), bool(check)))


def set_audio(path, source, *, output=None, bitrate: str = "128k", channels: int = 2,
              check: bool = True) -> Report | None:
    """Replace the audio track with `source` (any file FFmpeg reads), or remove it (None)."""
    src = None if source is None else os.fspath(source)
    return Report.from_json(_fflv.set_audio(os.fspath(path), src, _out(output), bitrate,
                                            _util.uint(channels, "channels"), bool(check)))


def set_layer(path, key, *, output=None, **fields) -> bool:
    """Change id / name / z / rect / blend / opacity / visible of one layer.
    Returns True when done in place (metadata only), False when the file had to be rewritten."""
    if not fields:
        raise MetaError(f"nothing to set; editable fields: {', '.join(EDITABLE_FIELDS)}")
    try:
        pairs = json.dumps([[k, _util.jsonable(v)] for k, v in fields.items()], allow_nan=False)
    except (ValueError, TypeError):
        raise MetaError(f"values must be JSON values with finite numbers, got {fields!r}") from None
    return _fflv.set_layer(os.fspath(path), _util.key(key), pairs, _out(output))
