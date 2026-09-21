"""FFmpeg-based helpers for reading source media (video layers, still images)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from ..format.constants import PNG_SIGNATURE


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], what: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        res = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise MediaError(f"{what}: {cmd[0]} not found on PATH") from exc
    if res.returncode != 0:
        tail = res.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise MediaError(f"{what} failed ({cmd[0]} exit {res.returncode}):\n  " + "\n  ".join(tail)
                         + "\n  command: " + " ".join(cmd))
    return res


def pix_fmt_has_alpha(name: str) -> bool:
    try:
        fmt = av.VideoFormat(name)
    except (ValueError, av.FFmpegError):
        return False
    return any(c.is_alpha for c in fmt.components)


@dataclass
class VideoSource:
    path: Path
    codec: str
    pix_fmt: str
    decoder: str | None  # FFmpeg decoder to force (libvpx for WebM with alpha)
    has_alpha: bool


def open_video_source(src: str | Path) -> VideoSource:
    path = Path(src)
    if not path.exists():
        raise MediaError(f"source file not found: {path}")
    res = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)],
              f"ffprobe {path.name}")
    streams = json.loads(res.stdout).get("streams") or []
    if not streams:
        raise MediaError(f"{path} has no video stream")
    info = streams[0]
    tags = {k.lower(): v for k, v in (info.get("tags") or {}).items()}
    webm_alpha = str(tags.get("alpha_mode", "0")) == "1"
    # FFmpeg's native VP8/VP9 decoders drop the WebM alpha side channel; libvpx keeps it.
    decoder = {"vp9": "libvpx-vp9", "vp8": "libvpx"}.get(info.get("codec_name")) if webm_alpha else None
    pix = info.get("pix_fmt", "")
    return VideoSource(path, info.get("codec_name", "?"), pix, decoder, webm_alpha or pix_fmt_has_alpha(pix))


def layer_filter(fps: Fraction, w: int, h: int) -> list[str]:
    """Common filter chain: constant frame rate, scale to the layer rect, pad to even dimensions by
    repeating the last row/column (the padding is cropped again on playback)."""
    chain = [f"fps={fps.numerator}/{fps.denominator}", f"scale={w}:{h}:flags=bicubic", "setsar=1"]
    pw, ph = w & 1, h & 1
    if pw or ph:
        chain += [f"pad={w + pw}:{h + ph}:0:0",
                  f"fillborders=left=0:right={pw}:top=0:bottom={ph}:mode=smear"]
    return chain


def input_args(src: VideoSource) -> list[str]:
    return (["-c:v", src.decoder] if src.decoder else []) + ["-i", str(src.path)]


def raw_frames(src: VideoSource, fps: Fraction, w: int, h: int, count: int, alpha: bool) -> Iterator[np.ndarray]:
    """Decode exactly `count` frames at `fps`, scaled to w×h, as uint8 arrays (H×W×4 or H×W×3).
    Short sources are extended by repeating their last frame."""
    fmt, ch = ("rgba", 4) if alpha else ("rgb24", 3)
    vf = [f"fps={fps.numerator}/{fps.denominator}", f"scale={w}:{h}:flags=bicubic", "setsar=1",
          f"format={fmt}", "tpad=stop_mode=clone:stop=-1"]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *input_args(src),
           "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", ",".join(vf), "-frames:v", str(count),
           "-f", "rawvideo", "-pix_fmt", fmt, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = w * h * ch
    try:
        for i in range(count):
            buf = proc.stdout.read(size)
            if len(buf) != size:
                err = proc.stderr.read().decode(errors="replace").strip()
                raise MediaError(f"decoding {src.path.name} stopped after {i} of {count} frames: {err}")
            yield np.frombuffer(buf, np.uint8).reshape(h, w, ch)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
        proc.stderr.close()


def image_to_png(image) -> bytes:
    """Encode an RGB/RGBA/gray uint8 array as PNG."""
    a = np.asarray(image)
    if a.dtype != np.uint8:
        raise ValueError("still images must be uint8 arrays (or PNG files)")
    if a.ndim == 2:
        fmt = "gray"
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        fmt = "rgb24" if a.shape[2] == 3 else "rgba"
    else:
        raise ValueError(f"still image has unsupported shape {a.shape}")
    ctx = av.CodecContext.create("png", "w")
    ctx.width, ctx.height = a.shape[1], a.shape[0]
    ctx.pix_fmt = fmt
    ctx.time_base = Fraction(1, 1)
    packets = ctx.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(a), format=fmt))
    packets += ctx.encode(None)
    return b"".join(bytes(p) for p in packets)


def still_png(src) -> bytes:
    """PNG bytes for a still layer: a PNG file as is, any other image converted, arrays encoded."""
    if isinstance(src, (bytes, bytearray)):
        if bytes(src[:8]) != PNG_SIGNATURE:
            raise MediaError("still image bytes are not a PNG")
        return bytes(src)
    if isinstance(src, np.ndarray):
        return image_to_png(src)
    path = Path(src)
    if not path.exists():
        raise MediaError(f"still image not found: {path}")
    data = path.read_bytes()
    if data.startswith(PNG_SIGNATURE):
        return data
    res = run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
               "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"], f"convert {path.name} to PNG")
    return res.stdout


def png_size(png: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR chunk."""
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
