"""In-process VP9 encoding of numpy images (PyAV + libvpx-vp9) with exact key-frame control.

Every call to `encode` yields exactly one packet that displays exactly one frame, and the frame is
a key frame exactly when asked (libvpx's own key-frame interval is pushed out of reach, alt-ref and
look-ahead are off). Both properties are verified on every packet from the VP9 frame header.

Plane formats:
  color, lossy      profile 0, 8-bit 4:2:0, BT.709 limited range
  color, lossless   profile 1, 8-bit 4:4:4 RGB (gbrp), lossless — pixel values survive exactly
  alpha             profile 0 luma; limited range (Y = 16..235) when lossy, full range and
                    lossless (Y = alpha) when the layer is lossless
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
from av.video.frame import PictureType, VideoFrame
from av.video.reformatter import ColorRange, Colorspace, VideoReformatter

from ..format.vp9 import Vp9Error, inspect_packet

# speed preset → (libvpx deadline, cpu-used)
SPEED_PRESETS = {"fast": ("realtime", 8), "balanced": ("good", 4), "best": ("good", 1)}
_NEVER = 1 << 30  # key-frame interval libvpx will never reach on its own


class EncodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncodeOptions:
    crf: int = 32
    speed: str = "balanced"
    threads: int = 0  # 0: pick automatically

    def __post_init__(self):
        if self.speed not in SPEED_PRESETS:
            raise ValueError(f"speed must be one of {sorted(SPEED_PRESETS)}, got {self.speed!r}")
        if not 0 <= self.crf <= 63:
            raise ValueError(f"crf must be in 0..63, got {self.crf}")


def even(n: int) -> int:
    return n + (n & 1)


class PlaneEncoder:
    """One VP9 stream: the color or the alpha plane of one layer."""

    def __init__(self, kind: str, width: int, height: int, fps: Fraction, lossless: bool,
                 options: EncodeOptions = EncodeOptions()):
        if kind not in ("color", "alpha"):
            raise ValueError(kind)
        if width % 2 or height % 2:
            raise ValueError("coded size must be even")
        self.kind, self.width, self.height, self.lossless = kind, width, height, lossless
        c = av.CodecContext.create("libvpx-vp9", "w")
        c.width, c.height = width, height
        c.time_base = Fraction(fps.denominator, fps.numerator)
        c.framerate = fps
        c.gop_size = _NEVER
        c.thread_count = options.threads or min(8, os.cpu_count() or 2)
        deadline, cpu_used = SPEED_PRESETS[options.speed]
        opts = {"b": "0", "lag-in-frames": "0", "auto-alt-ref": "0", "row-mt": "1",
                "deadline": deadline, "cpu-used": str(cpu_used), "keyint_min": str(_NEVER)}
        if lossless:
            opts["lossless"] = "1"
        else:
            opts["crf"] = str(options.crf)
        if kind == "color" and lossless:
            c.pix_fmt = "gbrp"  # profile 1, RGB: bit-exact
            opts.update({"colorspace": "rgb", "color_primaries": "bt709", "color_trc": "iec61966-2-1",
                         "color_range": "pc"})
        else:
            c.pix_fmt = "yuv420p"
            full = kind == "alpha" and lossless
            opts.update({"colorspace": "bt709", "color_primaries": "bt709", "color_trc": "bt709",
                         "color_range": "pc" if full else "tv"})
        c.options = opts
        self.alpha_full_range = kind == "alpha" and lossless
        self._ctx = c
        self._pts = 0
        self._reformatter = VideoReformatter()

    # -- image → VideoFrame -----------------------------------------------------------------------
    def _frame(self, img: np.ndarray) -> VideoFrame:
        if self.kind == "color":
            if img.shape != (self.height, self.width, 3):
                raise ValueError(f"color image must be {self.height}x{self.width}x3, got {img.shape}")
            fr = VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            if self.lossless:
                return fr.reformat(format="gbrp")
            return self._reformatter.reformat(fr, format="yuv420p", dst_colorspace=Colorspace.ITU709,
                                              dst_color_range=ColorRange.MPEG)
        if img.shape != (self.height, self.width):
            raise ValueError(f"alpha image must be {self.height}x{self.width}, got {img.shape}")
        if self.alpha_full_range:
            y = img
        else:  # alpha 0..255 → limited-range luma 16..235
            y = (img.astype(np.uint16) * 219 + 127) // 255 + 16
        w, h = self.width, self.height
        chroma = np.full(w * h // 2, 128, np.uint8)
        packed = np.concatenate([y.astype(np.uint8).ravel(), chroma]).reshape(h * 3 // 2, w)
        return VideoFrame.from_ndarray(packed, format="yuv420p")

    def encode(self, img: np.ndarray, key: bool) -> bytes:
        fr = self._frame(img)
        fr.pts = self._pts
        self._pts += 1
        if key:
            fr.pict_type = PictureType.I
        packets = self._ctx.encode(fr)
        if len(packets) != 1:
            raise EncodeError(f"{self.kind} encoder returned {len(packets)} packets for one frame")
        data = bytes(packets[0])
        try:
            info = inspect_packet(data)
        except Vp9Error as exc:
            raise EncodeError(f"{self.kind} encoder produced an unparseable packet: {exc}") from exc
        if info.shown_count != 1:
            raise EncodeError(f"{self.kind} packet shows {info.shown_count} frames")
        if info.key_frame != key:
            raise EncodeError(f"{self.kind} packet is {'a key' if info.key_frame else 'an inter'} frame, "
                              f"expected {'key' if key else 'inter'}")
        return data

    def close(self) -> None:
        tail = self._ctx.encode(None)
        if tail:
            raise EncodeError(f"{self.kind} encoder held back {len(tail)} packets")


# --------------------------------------------------------------------------------------------------
# Layer images
# --------------------------------------------------------------------------------------------------
def to_rgba(image, width: int, height: int, *, name: str = "layer") -> tuple[np.ndarray, np.ndarray]:
    """Normalise an image for a width×height layer to (rgb uint8 H×W×3, alpha uint8 H×W).

    Accepts uint8 / bool / float (0..1) arrays shaped H×W (gray), H×W×1, H×W×3 (RGB) or H×W×4
    (RGBA). Images without alpha are opaque.
    """
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.shape[:2] != (height, width):
        raise ValueError(f"{name}: image is {a.shape[1] if a.ndim > 1 else '?'}x{a.shape[0]}, "
                         f"the layer is {width}x{height}")
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif np.issubdtype(a.dtype, np.floating):
        a = np.clip(np.nan_to_num(a) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        raise ValueError(f"{name}: dtype {a.dtype} is not supported (use uint8, bool or float in 0..1)")
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2), np.full(a.shape, 255, np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        return a, np.full(a.shape[:2], 255, np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        return a[:, :, :3], a[:, :, 3]
    raise ValueError(f"{name}: expected H×W, H×W×3 or H×W×4, got shape {a.shape}")


def pad_even(a: np.ndarray) -> np.ndarray:
    """Pad to even width/height by repeating the last column/row (cropped away again on playback)."""
    h, w = a.shape[:2]
    if not (h & 1 or w & 1):
        return a
    pad = [(0, h & 1), (0, w & 1)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, mode="edge")


class LayerEncoder:
    """Color (+ alpha) encoder pair for one video layer of size width×height (any parity)."""

    def __init__(self, width: int, height: int, fps: Fraction, *, alpha: bool, lossless: bool,
                 options: EncodeOptions = EncodeOptions(), name: str = "layer"):
        self.width, self.height, self.name = width, height, name
        cw, ch = even(width), even(height)
        self.color = PlaneEncoder("color", cw, ch, fps, lossless, options)
        self.alpha = PlaneEncoder("alpha", cw, ch, fps, lossless, options) if alpha else None

    def prepare(self, image) -> tuple[np.ndarray, np.ndarray | None]:
        rgb, a = to_rgba(image, self.width, self.height, name=self.name)
        return pad_even(rgb), (pad_even(a) if self.alpha else None)

    def encode_prepared(self, rgb: np.ndarray, a: np.ndarray | None, key: bool) -> tuple[bytes, bytes]:
        color = self.color.encode(rgb, key)
        alpha = self.alpha.encode(a, key) if self.alpha else b""
        return color, alpha

    def encode(self, image, key: bool) -> tuple[bytes, bytes]:
        rgb, a = self.prepare(image)
        return self.encode_prepared(rgb, a, key)

    def close(self) -> None:
        self.color.close()
        if self.alpha:
            self.alpha.close()
