"""`fflv render` / `fflv extract`: decode selected layers to images, videos or numpy arrays.

Output type follows the path:
  out.png / out.jpg           a single frame
  frames/%05d.png, dir/        one image per frame (named by frame index)
  out.mp4 / .mov              H.264 (yuv420p); .mov with alpha → PNG-in-MOV (RGBA)
  out.webm                    VP9 (with alpha: yuva420p)
  out.mkv                     FFV1, lossless (RGB or RGBA)
  out.npy                     all frames stacked (N×H×W×C uint8) — mind the memory
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import numpy as np
from av.video.frame import VideoFrame

from .decode import Reader

IMAGE_EXT = {".png": "png", ".jpg": "mjpeg", ".jpeg": "mjpeg"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".webm", ".mkv"}


class OutputError(ValueError):
    pass


def _encode_image(img: np.ndarray, codec: str) -> bytes:
    ctx = av.CodecContext.create(codec, "w")
    ctx.height, ctx.width = img.shape[:2]
    if codec == "png":
        fmt = "rgba" if img.shape[2] == 4 else "rgb24"
    else:
        fmt, img = "yuvj420p", img[:, :, :3]
    ctx.pix_fmt = fmt
    ctx.time_base = Fraction(1, 1)
    src = VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgba" if img.shape[2] == 4 else "rgb24")
    if src.format.name != fmt:
        src = src.reformat(format=fmt)
    pk = ctx.encode(src) + ctx.encode(None)
    return b"".join(bytes(p) for p in pk)


class _ImageSink:
    def __init__(self, pattern: str | None, single: str | None, codec: str):
        self.pattern, self.single, self.codec, self.count = pattern, single, codec, 0

    def write(self, index: int, img: np.ndarray) -> None:
        if self.single:
            if self.count:
                raise OutputError(f"{self.single} holds one frame; use a pattern like frames/%05d.png or a directory")
            path = self.single
        else:
            path = self.pattern % index
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(_encode_image(img, self.codec))
        self.count += 1

    def close(self) -> None:
        pass


class _NpySink:
    def __init__(self, path: str):
        self.path, self.frames = path, []

    def write(self, index: int, img: np.ndarray) -> None:
        self.frames.append(img)

    def close(self) -> None:
        np.save(self.path, np.stack(self.frames) if self.frames else np.zeros((0,), np.uint8))


class _VideoSink:
    def __init__(self, path: str, fps: Fraction, alpha: bool, crf: int):
        ext = Path(path).suffix.lower()
        self.alpha = alpha
        self.out = av.open(path, "w")
        if ext == ".webm":
            codec, pix, opts = "libvpx-vp9", ("yuva420p" if alpha else "yuv420p"), {"crf": str(crf), "b": "0",
                                                                                   "row-mt": "1", "deadline": "good",
                                                                                   "cpu-used": "4"}
        elif ext == ".mkv":
            codec, pix, opts = "ffv1", ("bgra" if alpha else "bgr0"), {}
        elif alpha:  # .mov / .mp4 with alpha
            if ext != ".mov":
                raise OutputError("H.264 has no alpha channel: use .mov, .webm, .mkv or PNG frames for --transparent")
            codec, pix, opts = "png", "rgba", {}
        else:
            codec, pix, opts = "libx264", "yuv420p", {"crf": str(crf), "preset": "medium"}
        self.stream = self.out.add_stream(codec, rate=fps)
        self.stream.pix_fmt = pix
        self.stream.options = opts
        self.even = pix.startswith("yuv")  # 4:2:0 needs even dimensions
        self.started = False

    def write(self, index: int, img: np.ndarray) -> None:
        if self.even and (img.shape[0] & 1 or img.shape[1] & 1):
            img = np.pad(img, [(0, img.shape[0] & 1), (0, img.shape[1] & 1), (0, 0)], mode="edge")
        if not self.started:
            self.stream.height, self.stream.width = img.shape[:2]
            self.started = True
        fr = VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgba" if img.shape[2] == 4 else "rgb24")
        for p in self.stream.encode(fr):
            self.out.mux(p)

    def close(self) -> None:
        if self.started:
            for p in self.stream.encode(None):
                self.out.mux(p)
        self.out.close()


def open_sink(output: str | os.PathLike, fps: Fraction, alpha: bool, crf: int = 18):
    out = os.fspath(output)
    ext = Path(out).suffix.lower()
    if out.endswith(os.sep) or Path(out).is_dir():
        return _ImageSink(os.path.join(out, "%06d.png"), None, "png")
    if ext in IMAGE_EXT:
        if alpha and IMAGE_EXT[ext] != "png":
            raise OutputError("JPEG has no alpha channel; use .png")
        if "%" in out:
            return _ImageSink(out, None, IMAGE_EXT[ext])
        return _ImageSink(None, out, IMAGE_EXT[ext])
    if ext == ".npy":
        return _NpySink(out)
    if ext in VIDEO_EXT:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return _VideoSink(out, fps, alpha, crf)
    raise OutputError(f"don't know how to write {out!r}: use .png/.jpg (optionally with %d), a directory, "
                      f".mp4/.mov/.webm/.mkv or .npy")


Progress = Callable[[int, int], None]


def render(path, output, *, layers=None, hide=None, start: int = 0, end: int | None = None,
           transparent: bool = False, crf: int = 18, progress: Progress | None = None) -> int:
    """Composite the chosen layers over [start, end) into `output`. Returns frames written."""
    with Reader(path) as r:
        end = r.frame_count if end is None else end
        sink = open_sink(output, r.fps, transparent, crf)
        n = 0
        try:
            for f, img in r.frames(start, end, layers=layers, hide=hide, transparent=transparent):
                sink.write(f, img)
                n += 1
                if progress:
                    progress(n, end - start)
        finally:
            sink.close()
        return n


def extract(path, layer, output, *, start: int | None = None, end: int | None = None, crf: int = 18,
            progress: Progress | None = None) -> int:
    """Write one layer's own pixels (RGBA, content size) for the frames where it is active."""
    with Reader(path) as r:
        L = r.layer(layer)
        sink = open_sink(output, r.fps, True, crf)
        s, e = max(L.start, start or L.start), min(L.end, end or L.end)
        n = 0
        try:
            for f, rgba in r.layer_frames(L.index, start, end):
                sink.write(f, rgba)
                n += 1
                if progress:
                    progress(n, max(1, e - s))
        finally:
            sink.close()
        return n
