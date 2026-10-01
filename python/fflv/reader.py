"""Decoding .lvd files: selected layers only, composited like the player.

    with fflv.open("debug.lvd") as f:
        img = f.frame(120)                                  # composite RGB, file's default layers
        img = f.frame(120, layers=["rgb", "mask"])          # any subset
        for i, rgba in f.layer_frames("mask", 100, 200):    # a layer's own pixels (RGBA)
            ...

Only the layers you ask for are decoded — the packets of all other layers are skipped, so
switching between layer subsets costs nothing extra. Decoding starts at the nearest random-access
point at or before the first requested frame; the planes of a frame are decoded in parallel,
without the GIL.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator

import numpy as np

from . import _fflv, _util
from ._fflv import MetaError
from .report import Report


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


class Reader:
    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._r = _fflv._Reader(self.path)
        self.meta: dict = json.loads(self._r.meta_json())
        m = self.meta
        self.fps = Fraction(m["fps"]["num"], m["fps"]["den"])
        self.frame_count: int = m["frame_count"]
        self.size = (m["canvas"]["width"], m["canvas"]["height"])
        self.background: str = m["canvas"]["background"]
        self.layers = [_layer_info(i, L) for i, L in enumerate(m["layers"])]
        # frame(): the decoding of the last call, (selection, next frame, iterator), so that reading
        # consecutive frames continues it instead of starting over from a random-access point
        self._next_frames = None
        self._next_frames_lock = threading.Lock()

    def close(self) -> None:
        """Close the file. Iterators already started keep their own handle until they are done."""
        with self._next_frames_lock:
            self._next_frames = None
        self._r.close()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<fflv.Reader {os.path.basename(self.path)} {self.size[0]}x{self.size[1]} @ {self.fps} fps, "
                f"{self.frame_count} frames, layers {[L.id for L in self.layers]}>")

    # -- helpers -----------------------------------------------------------------------------------
    def layer(self, key) -> LayerInfo:
        """A layer by id or index (int or numeric string)."""
        k = _util.key(key)
        if k.isdigit():
            i = int(k)
            if not 0 <= i < len(self.layers):
                raise MetaError(f"layer index {i} out of range (file has {len(self.layers)} layers)")
            return self.layers[i]
        for L in self.layers:
            if L.id == k:
                return L
        raise MetaError(f"no layer {k!r}; layers: {', '.join(L.id for L in self.layers)}")

    def select(self, layers: Iterable | None = None, hide: Iterable | None = None) -> list[LayerInfo]:
        """Layers to show: `layers` (ids/indices) if given, else the file's visible layers; minus `hide`."""
        chosen = [self.layer(k) for k in _util.keys(layers)] if layers is not None else \
            [L for L in self.layers if L.visible]
        hidden = {self.layer(k).index for k in (_util.keys(hide, "hide") if hide is not None else [])}
        return [L for L in chosen if L.index not in hidden]

    @property
    def raps(self) -> list[int]:
        return self._r.raps()

    def rap_at_or_before(self, frame: int) -> int:
        return self._r.rap_at_or_before(_util.uint(frame, "frame"))

    def pts_us(self, frame: int) -> int:
        frame = _util.uint(frame, "frame")
        num, den = self.fps.numerator, self.fps.denominator
        return (2 * frame * 1_000_000 * den + num) // (2 * num)

    def check(self) -> Report:
        return Report.from_json(self._r.check())

    # -- decoding ----------------------------------------------------------------------------------
    def decode(self, start: int = 0, end: int | None = None, layers: Iterable | None = None
               ) -> Iterator[tuple[int, dict[int, tuple[np.ndarray, np.ndarray | None]]]]:
        """Yield (frame, {layer_index: (rgb, alpha)}) for the chosen video layers active in each
        frame (content size, before scaling to the rect). Other layers are never decoded."""
        chosen = self.select(layers) if layers is not None else self.layers
        idx = [L.index for L in chosen if L.kind == "video"]
        return iter(self._r.decode(_util.uint(start, "start"), _util.frame_arg(end, "end"), idx))

    def frames(self, start: int = 0, end: int | None = None, *, layers: Iterable | None = None,
               hide: Iterable | None = None, transparent: bool = False) -> Iterator[tuple[int, np.ndarray]]:
        """Composite frames (uint8 RGB, or RGBA if `transparent`) of the chosen layers."""
        keys = None if layers is None else _util.keys(layers)
        hidden = [] if hide is None else _util.keys(hide, "hide")
        return iter(self._r.frames(_util.uint(start, "start"), _util.frame_arg(end, "end"), keys, hidden,
                                   bool(transparent)))

    def frame(self, index: int, layers: Iterable | None = None, *, hide: Iterable | None = None,
              transparent: bool = False) -> np.ndarray:
        """Composite frame `index` (see `frames`). Reading frames one after another with the same
        layers continues one decoding instead of starting over from a random-access point."""
        index = _util.uint(index, "index")
        keys = None if layers is None else tuple(_util.keys(layers))
        hidden = () if hide is None else tuple(_util.keys(hide, "hide"))
        selection = (keys, hidden, bool(transparent))
        with self._next_frames_lock:
            cached, self._next_frames = self._next_frames, None
        it = None
        if cached is not None and cached[0] == selection and cached[1] == index:
            it = cached[2]
            item = next(it, None)
            if item is None or item[0] != index:
                it = None
        if it is None:
            it = iter(self._r.frames(index, None, None if keys is None else list(keys), list(hidden),
                                     selection[2]))
            item = next(it)
        with self._next_frames_lock:
            self._next_frames = (selection, index + 1, it)
        return item[1]

    def layer_frames(self, key, start: int | None = None, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """A layer's own pixels as RGBA uint8 (content size), for the frames where it is active."""
        return iter(self._r.layer_frames(_util.key(key), _util.frame_arg(start, "start"),
                                         _util.frame_arg(end, "end")))

    def still(self, key) -> np.ndarray:
        """A still layer's image (RGBA)."""
        L = self.layer(key)
        if L.kind != "still":
            raise MetaError(f"layer {L.id!r} is not a still layer")
        return self._r.still(L.index)


def open(path: str | os.PathLike) -> Reader:  # noqa: A001 (mirrors built-in open, like av.open)
    return Reader(path)
