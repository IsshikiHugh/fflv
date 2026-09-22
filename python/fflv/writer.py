"""Streaming writer: numpy images in, .lvd out.

    with fflv.Writer("debug.lvd", size=(1280, 720), fps=30) as w:
        w.add_layer("rgb")                                  # opaque, full canvas
        w.add_layer("mask", alpha=True, lossless=True)      # exact pixel values
        w.add_still("legend", "legend.png", rect=(1100, 20, 160, 80))
        for img, mask in data:
            w.write(rgb=img, mask=mask)

Semantics
  * Every `write()` appends one composite frame; all layers are encoded (in parallel, without the
    GIL) and the frame is written immediately, so memory use does not grow with the video.
  * A layer starts at the first frame it is given an image. If it is omitted later it keeps
    showing its last image ("sticky"), until `end_layer()` or the end of the file.
  * Key frames sit on the global grid (every `gop` frames) plus each layer's first frame, so every
    multiple of `gop` is a random-access point.
  * The file is written to a hidden temporary file beside `path`; `close()` validates it and then
    atomically renames it to `path` (spec B.11).
  * If encoding fails inside `write()`, the layers' encoders may be out of step with the file, so
    the writer refuses to continue: further `write()` / `close()` calls raise; use `abort()`.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

from . import _fflv, _util
from ._fflv import WriterError
from .report import Report


class Writer:
    def __init__(self, path: str | os.PathLike, size: tuple[int, int], fps=30, *, gop: int | None = None,
                 background: str = "#000000", crf: int = 32, speed: str = "balanced", check: bool = True,
                 threads: int | None = None):
        self.path = Path(path)
        self.width, self.height = int(size[0]), int(size[1])
        if self.width <= 0 or self.height <= 0:
            raise WriterError(f"size must be positive, got {size!r}")
        self.fps: Fraction = _util.fps(fps)
        self._w = _fflv._Writer(os.fspath(path), self.width, self.height, self.fps.numerator, self.fps.denominator,
                                None if gop is None else int(gop), background, int(crf), speed, bool(check),
                                threads)
        self.report: Report | None = None

    @property
    def gop(self) -> int:
        return self._w.gop

    @property
    def frame_count(self) -> int:
        return self._w.frame_count

    @property
    def layer_ids(self) -> list[str]:
        return self._w.layer_ids

    def add_layer(self, id: str, *, alpha: bool = False, lossless: bool = False, rect=None, z: float | None = None,
                  name: str | None = None, blend: str = "normal", opacity: float = 1.0, visible: bool = True,
                  crf: int | None = None, speed: str | None = None) -> None:
        """Declare a video layer. `rect` = (x, y, w, h) on the canvas (default: the whole canvas);
        images written to it must be w×h. `lossless` keeps pixel values exactly (bigger files)."""
        self._w.add_layer(id, bool(alpha), bool(lossless), _util.rect(rect), _util.number(z, "z"), name, blend,
                          _util.number(opacity, "opacity"), bool(visible), crf, speed)

    def add_still(self, id: str, image, *, rect=None, start: int = 0, end: int | None = None,
                  z: float | None = None, name: str | None = None, blend: str = "normal", opacity: float = 1.0,
                  visible: bool = True) -> None:
        """Declare a still layer shown in frames [start, end) (end None = to the end of the file).
        `image`: an image path, PNG bytes, or a uint8 array (H×W, H×W×3, H×W×4).
        Default rect: the image's own size at the top-left corner."""
        png = _util.still_png(image)
        if end is not None and end <= start or start < 0:
            raise WriterError(f"still {id!r}: bad frame range [{start}, {end})")
        self._w.add_still(id, png, _util.rect(rect), int(start), None if end is None else int(end),
                          _util.number(z, "z"), name, blend, _util.number(opacity, "opacity"), bool(visible))

    def set_audio(self, src: str | os.PathLike, *, bitrate: str = "128k", channels: int = 2) -> None:
        """Audio track from any file FFmpeg can read (cut to the video length)."""
        self._w.set_audio(os.fspath(src), bitrate, int(channels))

    def write(self, images: dict | None = None, /, **kw) -> int:
        """Append one composite frame. Pass images by layer id (keyword or dict). Returns the frame index."""
        imgs = dict(images or {})
        imgs.update(kw)
        return self._w.write([(k, _util.as_uint8(v, k)) for k, v in imgs.items()])

    def end_layer(self, id: str) -> None:
        """The layer's last frame was the previous write(); it is empty from now on."""
        self._w.end_layer(id)

    def close(self) -> Report | None:
        """Finish, validate and publish the file; returns the validation report."""
        if not self._w.closed:
            self.report = Report.from_json(self._w.close())
        return self.report

    def abort(self) -> None:
        """Discard everything written so far."""
        self._w.abort()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def __repr__(self) -> str:
        return (f"<fflv.Writer {self.path.name} {self.width}x{self.height} @ {self.fps} fps, "
                f"{self.frame_count} frames, layers {self.layer_ids}>")
