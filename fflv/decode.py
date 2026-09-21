"""Decoding .lvd files in Python: selected layers only, composited like the player.

    with fflv.open("debug.lvd") as f:
        img = f.frame(120)                                  # composite RGB, file's default layers
        img = f.frame(120, layers=["rgb", "mask"])          # any subset
        for i, rgba in f.layer_frames("mask", 100, 200):    # a layer's own pixels (RGBA)
            ...

Only the layers you ask for are decoded — the packets of all other layers are skipped, so switching
between layer subsets costs nothing extra. Decoding starts at the nearest random-access point at or
before the first requested frame; the planes of a frame are decoded in parallel.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator

import av
import numpy as np
from av.video.frame import VideoFrame

from .format import LVFReader, pts_us, validate
from .format.constants import ENTRY_FRAME
from .meta import resolve_layer


class DecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayerInfo:
    index: int
    id: str
    name: str
    kind: str  # "video" | "still"
    z: float
    rect: tuple[int, int, int, int]
    start: int
    end: int
    blend: str
    opacity: float
    visible: bool
    has_alpha: bool = False
    lossless: bool = False
    alpha_range: str = "limited"
    coded_size: tuple[int, int] | None = None
    content_size: tuple[int, int] | None = None

    def active(self, frame: int) -> bool:
        return self.start <= frame < self.end


def _layer_info(i: int, L: dict) -> LayerInfo:
    r = L["rect"]
    common = dict(index=i, id=L["id"], name=L.get("name", L["id"]), kind=L["kind"], z=L["z"],
                  rect=(r["x"], r["y"], r["w"], r["h"]), start=L["start_frame"], end=L["end_frame"],
                  blend=L["blend"], opacity=L["opacity"], visible=L["visible"])
    if L["kind"] != "video":
        return LayerInfo(**common)
    coded = (L["coded_width"], L["coded_height"])
    return LayerInfo(**common, has_alpha=L["has_alpha"], lossless=bool(L.get("lossless", False)),
                     alpha_range=L.get("alpha_range", "limited"), coded_size=coded,
                     content_size=tuple(L.get("content_size") or coded))


# --------------------------------------------------------------------------------------------------
# YUV → RGB
# --------------------------------------------------------------------------------------------------
# Done here rather than by swscale: swscale's result depends on the frame width (its C fallback,
# used when the width is not a multiple of 16, maps Y=235 to 253 instead of 255) and on the CPU.
_KR_KB = {1: (0.2126, 0.0722), 5: (0.299, 0.114), 6: (0.299, 0.114), 9: (0.2627, 0.0593)}


def _plane(p) -> np.ndarray:
    return np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, :p.width]


def frame_to_rgb(frame: VideoFrame) -> np.ndarray:
    """uint8 RGB of a decoded frame: exact for gbrp (lossless layers), the exact BT.601/709/2020
    formula for 8-bit YUV (limited or full range, as the frame says), swscale otherwise."""
    fmt = frame.format.name
    if fmt == "gbrp":
        g, b, r = (_plane(p) for p in frame.planes)
        return np.dstack([r, g, b])
    if fmt not in ("yuv420p", "yuvj420p", "yuv444p", "yuvj444p"):
        return frame.to_ndarray(format="rgb24")
    kr, kb = _KR_KB.get(int(frame.colorspace or 1), _KR_KB[1])
    kg = 1 - kr - kb
    full = fmt.startswith("yuvj") or int(frame.color_range or 0) == 2
    y = _plane(frame.planes[0]).astype(np.float32)
    u = _plane(frame.planes[1]).astype(np.float32) - 128
    v = _plane(frame.planes[2]).astype(np.float32) - 128
    if full:
        cs = np.float32(1)
    else:
        y = (y - 16) * np.float32(255 / 219)
        cs = np.float32(255 / 224)
    h, w = y.shape
    terms = (v * np.float32(2 * (1 - kr) * cs),
             u * np.float32(-2 * kb * (1 - kb) / kg * cs) + v * np.float32(-2 * kr * (1 - kr) / kg * cs),
             u * np.float32(2 * (1 - kb) * cs))
    out = np.empty((h, w, 3), np.float32)
    for i, t in enumerate(terms):
        if t.shape != (h, w):  # 4:2:0 chroma: nearest-neighbour upsampling
            t = np.repeat(np.repeat(t, 2, 0), 2, 1)[:h, :w]
        np.add(y, t, out=out[:, :, i])
    out += 0.5
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------------------------------
# Compositing (straight alpha; the same formulas as the WebGL player, player/src/render)
# --------------------------------------------------------------------------------------------------
def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    if a.shape[1] == w and a.shape[0] == h:
        return a
    fmt = "gray" if a.ndim == 2 else "rgb24"
    return VideoFrame.from_ndarray(np.ascontiguousarray(a), format=fmt).reformat(width=w, height=h).to_ndarray()


class Canvas:
    """Premultiplied float canvas. Opaque unless created transparent."""

    def __init__(self, width: int, height: int, background: str | None):
        self.width, self.height = width, height
        self.color = np.zeros((height, width, 3), np.float32)
        self.alpha = np.zeros((height, width), np.float32)
        if background is not None:
            self.color[:] = [int(background[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            self.alpha[:] = 1

    def draw(self, rgb: np.ndarray, alpha: np.ndarray | None, rect: tuple[int, int, int, int], blend: str,
             opacity: float) -> None:
        x, y, w, h = rect
        rgb = _resize(rgb, w, h)
        alpha = _resize(alpha, w, h) if alpha is not None else None
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, self.width), min(y + h, self.height)
        if x0 >= x1 or y0 >= y1:
            return
        cs = rgb[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255
        a = (alpha[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255 if alpha is not None
             else np.ones(cs.shape[:2], np.float32)) * np.float32(opacity)
        P = self.color[y0:y1, x0:x1]
        A = self.alpha[y0:y1, x0:x1]
        a3 = a[:, :, None]
        if blend == "add":
            P[:] = np.minimum(1.0, P + a3 * cs)
        else:
            if blend == "normal":
                src = cs
            else:
                cb = P / np.maximum(A, 1e-6)[:, :, None]
                bf = cb * cs if blend == "multiply" else cb + cs - cb * cs  # multiply / screen
                ab = A[:, :, None]
                src = (1 - ab) * cs + ab * bf
            P[:] = a3 * src + P * (1 - a3)
        A[:] = a + A * (1 - a)

    def image(self, transparent: bool = False) -> np.ndarray:
        if not transparent:
            return np.clip(self.color * 255 + 0.5, 0, 255).astype(np.uint8)
        c = self.color / np.maximum(self.alpha, 1e-6)[:, :, None]
        rgba = np.dstack([c, self.alpha])
        return np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------------
class Reader:
    def __init__(self, path: str | os.PathLike, *, threads: int | None = None):
        self.path = os.fspath(path)
        self._r = LVFReader(self.path)
        m = self._r.meta
        self.meta = m
        self.fps = Fraction(m["fps"]["num"], m["fps"]["den"])
        self.frame_count: int = m["frame_count"]
        self.size = (m["canvas"]["width"], m["canvas"]["height"])
        self.background: str = m["canvas"]["background"]
        self.layers = [_layer_info(i, L) for i, L in enumerate(m["layers"])]
        _, _, entries = self._r.index()
        if len(entries) != self.frame_count:
            raise DecodeError("index does not match frame_count (run `fflv check`)")
        self._offsets = [e.cau_offset for e in entries]
        self._raps = [i for i, e in enumerate(entries) if e.is_rap]
        self._pool = ThreadPoolExecutor(max_workers=threads or min(16, os.cpu_count() or 4))
        self._stills: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def close(self) -> None:
        self._r.close()
        self._pool.shutdown(wait=False)

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<fflv.Reader {os.path.basename(self.path)} {self.size[0]}x{self.size[1]} @ {self.fps} fps, "
                f"{self.frame_count} frames, layers {[L.id for L in self.layers]}>")

    # -- helpers -----------------------------------------------------------------------------------
    def layer(self, key) -> LayerInfo:
        return self.layers[resolve_layer(self.meta, key)]

    def select(self, layers: Iterable | None = None, hide: Iterable | None = None) -> list[LayerInfo]:
        """Layers to show: `layers` (ids/indices) if given, else the file's visible layers; minus `hide`."""
        chosen = [self.layer(k) for k in layers] if layers is not None else [L for L in self.layers if L.visible]
        hidden = {self.layer(k).index for k in (hide or [])}
        return [L for L in chosen if L.index not in hidden]

    def rap_at_or_before(self, frame: int) -> int:
        lo, hi = 0, len(self._raps) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._raps[mid] <= frame:
                lo = mid
            else:
                hi = mid - 1
        return self._raps[lo]

    def pts_us(self, frame: int) -> int:
        return pts_us(frame, self.fps.numerator, self.fps.denominator)

    def _range(self, start: int, end: int | None) -> tuple[int, int]:
        end = self.frame_count if end is None else end
        if not 0 <= start < end <= self.frame_count:
            raise DecodeError(f"frame range [{start}, {end}) is outside [0, {self.frame_count})")
        return start, end

    def _still(self, L: LayerInfo) -> tuple[np.ndarray, np.ndarray]:
        if L.index not in self._stills:
            res = self.meta["layers"][L.index]["resource"]
            data = self._r.resource(res["offset"], res["length"])
            ctx = av.CodecContext.create("png", "r")
            frames = ctx.decode(av.Packet(data))
            rgba = frames[0].to_ndarray(format="rgba")
            self._stills[L.index] = (rgba[:, :, :3], rgba[:, :, 3])
        return self._stills[L.index]

    # -- decoding ----------------------------------------------------------------------------------
    def _to_arrays(self, L: LayerInfo, color: VideoFrame, alpha: VideoFrame | None):
        cw, ch = L.content_size
        rgb = frame_to_rgb(color)[:ch, :cw]
        a = None
        if alpha is not None:
            p = alpha.planes[0]
            y = np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:ch, :cw]
            if L.alpha_range == "full":
                a = y.copy()
            else:
                a = np.clip((y.astype(np.int32) - 16) * 255 * 2 // 219 + 1 >> 1, 0, 255).astype(np.uint8)
        return rgb, a

    def decode(self, start: int = 0, end: int | None = None, layers: Iterable | None = None
               ) -> Iterator[tuple[int, dict[int, tuple[np.ndarray, np.ndarray | None]]]]:
        """Yield (frame, {layer_index: (rgb, alpha)}) for the chosen video layers active in each frame
        (content size, before scaling to the rect). Other layers are never decoded."""
        start, end = self._range(start, end)
        chosen = [L for L in (self.select(layers) if layers is not None else self.layers) if L.kind == "video"]
        want = {L.index: L for L in chosen}
        decoders: dict[tuple[int, str], av.CodecContext] = {}

        def dec(key, data: bytes) -> VideoFrame:
            ctx = decoders.get(key)
            if ctx is None:
                ctx = decoders[key] = av.CodecContext.create("vp9", "r")
                ctx.thread_count = 1  # one frame out per packet in
            out = ctx.decode(av.Packet(data))
            if len(out) != 1:
                raise DecodeError(f"layer {key[0]} {key[1]}: decoder returned {len(out)} frames for one packet")
            return out[0]

        rap = self.rap_at_or_before(start)
        pos = self._offsets[rap]
        stop = self._offsets[end] if end < self.frame_count else self._r.header.index_offset
        for _off, cau, _size in self._r.iter_caus(pos, stop):
            f = cau.frame_index
            jobs = [(e.layer_index, e) for e in cau.entries if e.layer_index in want and e.type == ENTRY_FRAME]
            if not jobs:
                if f >= start:
                    yield f, {}
                continue

            def work(job):
                li, e = job
                c = dec((li, "color"), e.color)
                a = dec((li, "alpha"), e.alpha) if e.alpha else None
                return li, (self._to_arrays(want[li], c, a) if f >= start else None)

            results = dict(self._pool.map(work, jobs))
            if f >= start:
                yield f, results

    def frames(self, start: int = 0, end: int | None = None, *, layers: Iterable | None = None,
               hide: Iterable | None = None, transparent: bool = False) -> Iterator[tuple[int, np.ndarray]]:
        """Composite frames (uint8 RGB, or RGBA if `transparent`) of the chosen layers."""
        shown = sorted(self.select(layers, hide), key=lambda L: (L.z, L.index))
        video_keys = [L.index for L in shown if L.kind == "video"]
        for f, planes in self.decode(start, end, video_keys):
            cv = Canvas(*self.size, None if transparent else self.background)
            for L in shown:
                if not L.active(f):
                    continue
                if L.kind == "video":
                    if L.index not in planes:
                        continue
                    rgb, a = planes[L.index]
                else:
                    rgb, a = self._still(L)
                cv.draw(rgb, a, L.rect, L.blend, L.opacity)
            yield f, cv.image(transparent)

    def frame(self, index: int, layers: Iterable | None = None, *, hide: Iterable | None = None,
              transparent: bool = False) -> np.ndarray:
        return next(self.frames(index, index + 1, layers=layers, hide=hide, transparent=transparent))[1]

    def layer_frames(self, key, start: int | None = None, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """A layer's own pixels as RGBA uint8 (content size), for the frames where it is active."""
        L = self.layer(key)
        s = max(L.start, start if start is not None else L.start)
        e = min(L.end, end if end is not None else L.end)
        if s >= e:
            return
        if L.kind == "still":
            rgb, a = self._still(L)
            img = np.dstack([rgb, a])
            for f in range(s, e):
                yield f, img
            return
        for f, planes in self.decode(s, e, [L.index]):
            rgb, a = planes[L.index]
            yield f, np.dstack([rgb, a if a is not None else np.full(rgb.shape[:2], 255, np.uint8)])

    def check(self):
        return validate(self.path)


def open(path: str | os.PathLike, **kw) -> Reader:  # noqa: A001 (mirrors built-in open, like av.open)
    return Reader(path, **kw)
