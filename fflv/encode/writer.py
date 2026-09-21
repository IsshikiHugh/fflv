"""Streaming writer: numpy images in, .lvd out.

    with fflv.Writer("debug.lvd", size=(1280, 720), fps=30) as w:
        w.add_layer("rgb")                                  # opaque, full canvas
        w.add_layer("mask", alpha=True, lossless=True)      # exact pixel values
        w.add_still("legend", "legend.png", rect=(1100, 20, 160, 80))
        for img, mask in data:
            w.write(rgb=img, mask=mask)

Semantics
  * Every `write()` appends one composite frame; all layers are encoded (in parallel) and the frame
    is written immediately, so memory use does not grow with the length of the video.
  * A layer starts at the first frame it is given an image. If it is omitted later it keeps showing
    its last image ("sticky"), until `end_layer()` or the end of the file.
  * Key frames sit on the global grid (every `gop` frames) plus each layer's first frame, so every
    multiple of `gop` is a random-access point.
  * The file is written to "<path>.part" and renamed on `close()`; then it is validated.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .. import meta as M
from ..format import AudioPacket, Cau, LVFWriter, VideoEntry, encode_meta, meta_capacity_for, parse_fps, pts_us, validate
from ..format.constants import CAU_FLAG_RAP, ENTRY_EMPTY, ENTRY_FRAME, FRAME_FLAG_KEY
from .audio import AudioTrack, encode_audio
from .media import png_size, still_png
from .vp9enc import EncodeOptions, LayerEncoder


class WriterError(RuntimeError):
    pass


@dataclass
class _Video:
    id: str
    name: str
    z: float
    rect: dict
    alpha: bool
    lossless: bool
    blend: str
    opacity: float
    visible: bool
    encoder: LayerEncoder
    start: int | None = None
    end: int | None = None
    last: tuple | None = None


@dataclass
class _Still:
    id: str
    name: str
    z: float
    rect: dict
    png: bytes
    start: int
    end: int | None
    blend: str
    opacity: float
    visible: bool
    offset: int = 0


class Writer:
    def __init__(self, path: str | os.PathLike, size: tuple[int, int], fps=30, *, gop: int | None = None,
                 background: str = "#000000", crf: int = 32, speed: str = "balanced", check: bool = True,
                 threads: int | None = None):
        self.path = Path(path)
        self.width, self.height = int(size[0]), int(size[1])
        if self.width <= 0 or self.height <= 0:
            raise WriterError(f"size must be positive, got {size!r}")
        self.fps: Fraction = parse_fps(fps)
        self.gop = int(gop) if gop else max(1, round(2 * self.fps))
        self.background = M.check_background(background)
        self.options = EncodeOptions(crf=crf, speed=speed)
        self.check = check
        self._layers: list[_Video | _Still] = []
        self._audio: AudioTrack | None = None
        self._audio_pos = 0
        self._out: LVFWriter | None = None
        self._part = self.path.with_name(self.path.name + ".part")
        self._frames = 0
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=threads or min(16, (os.cpu_count() or 4)))
        self.report = None

    # ------------------------------------------------------------------ declaration (before writing)
    def _declare(self, what: str) -> None:
        if self._out is not None:
            raise WriterError(f"{what} must be called before the first write(): every composite frame "
                              f"has to list every layer")
        if self._closed:
            raise WriterError("writer is closed")

    def _rect(self, rect) -> dict:
        return M.check_rect(rect if rect is not None else (0, 0, self.width, self.height))

    def add_layer(self, id: str, *, alpha: bool = False, lossless: bool = False, rect=None, z: float | None = None,
                  name: str | None = None, blend: str = "normal", opacity: float = 1.0, visible: bool = True,
                  crf: int | None = None, speed: str | None = None) -> None:
        """Declare a video layer. `rect` = (x, y, w, h) on the canvas (default: the whole canvas);
        images written to it must be w×h. `lossless` keeps pixel values exactly (bigger files)."""
        self._declare("add_layer()")
        M.check_id(id, {L.id for L in self._layers})
        r = self._rect(rect)
        opts = EncodeOptions(crf=self.options.crf if crf is None else crf, speed=speed or self.options.speed)
        enc = LayerEncoder(r["w"], r["h"], self.fps, alpha=alpha, lossless=lossless, options=opts, name=id)
        self._layers.append(_Video(id, name or id, len(self._layers) if z is None else M.check_z(z), r, alpha, lossless,
                                   M.check_blend(blend), M.check_opacity(opacity), bool(visible), enc))

    def add_still(self, id: str, image, *, rect=None, start: int = 0, end: int | None = None,
                  z: float | None = None, name: str | None = None, blend: str = "normal", opacity: float = 1.0,
                  visible: bool = True) -> None:
        """Declare a still layer shown in frames [start, end) (end None = to the end of the file).
        `image`: a PNG/other image path, PNG bytes, or a uint8 array (H×W, H×W×3, H×W×4).
        Default rect: the image's own size at the top-left corner."""
        self._declare("add_still()")
        M.check_id(id, {L.id for L in self._layers})
        png = still_png(image)
        if rect is None:
            w, h = png_size(png)
            rect = (0, 0, w, h)
        if start < 0 or (end is not None and end <= start):
            raise WriterError(f"still {id!r}: bad frame range [{start}, {end})")
        self._layers.append(_Still(id, name or id, len(self._layers) if z is None else M.check_z(z), M.check_rect(rect), png,
                                   int(start), end, M.check_blend(blend), M.check_opacity(opacity), bool(visible)))

    def set_audio(self, src: str | os.PathLike, *, bitrate: str = "128k", channels: int = 2) -> None:
        """Audio track from any file FFmpeg can read (cut to the video length on close)."""
        self._declare("set_audio()")
        self._audio = encode_audio(src, bitrate=bitrate, channels=channels)

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def layer_ids(self) -> list[str]:
        return [L.id for L in self._layers]

    # ------------------------------------------------------------------ writing
    def _meta(self, frame_count: int, final: bool) -> dict:
        big = 10 ** 9  # placeholder numbers while drafting, so the reserved space is large enough
        layers = []
        for L in self._layers:
            if isinstance(L, _Video):
                start = L.start if final else big
                end = (L.end if L.end is not None else frame_count) if final else big
                layers.append(M.video_layer(id=L.id, name=L.name, z=L.z, rect=L.rect, start=start, end=end,
                                            fps=self.fps, alpha=L.alpha, lossless=L.lossless, blend=L.blend,
                                            opacity=L.opacity, visible=L.visible))
            else:
                end = (L.end if L.end is not None else frame_count) if final else big
                layers.append(M.still_layer(id=L.id, name=L.name, z=L.z, rect=L.rect, start=L.start if final else big,
                                            end=end, offset=L.offset, length=len(L.png), blend=L.blend,
                                            opacity=L.opacity, visible=L.visible))
        return M.file_meta(width=self.width, height=self.height, background=self.background, fps=self.fps,
                           frame_count=frame_count if final else big, gop=self.gop, layers=layers,
                           audio=self._audio.meta() if self._audio else None)

    def _begin(self) -> None:
        if not any(isinstance(L, _Video) for L in self._layers) and not self._layers:
            raise WriterError("declare at least one layer before writing")
        offset = 0
        for L in self._layers:
            if isinstance(L, _Still):
                L.offset = offset
                offset += len(L.png)
        resources = b"".join(L.png for L in self._layers if isinstance(L, _Still))
        draft = encode_meta(self._meta(0, final=False))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._out = LVFWriter(self._part)
        self._out.begin(draft, resources, meta_capacity=meta_capacity_for(draft))

    def write(self, images: dict | None = None, /, **kw) -> int:
        """Append one composite frame. Pass images by layer id (keyword or dict). Returns the frame index."""
        if self._closed:
            raise WriterError("writer is closed")
        imgs = dict(images or {})
        imgs.update(kw)
        videos = [L for L in self._layers if isinstance(L, _Video)]
        known = {L.id for L in videos}
        unknown = set(imgs) - known
        if unknown:
            stills = {L.id for L in self._layers if isinstance(L, _Still)}
            hint = " (still layers take no per-frame images)" if unknown & stills else ""
            raise WriterError(f"unknown video layer(s) {sorted(unknown)}{hint}; declared: {sorted(known)}")
        if self._out is None:
            self._begin()
        f = self._frames
        jobs = []
        for L in videos:
            if L.id in imgs:
                if L.end is not None:
                    raise WriterError(f"layer {L.id!r} was ended at frame {L.end}; a layer is one contiguous range")
                L.last = L.encoder.prepare(imgs[L.id])
                if L.start is None:
                    L.start = f
            if L.start is not None and L.end is None:
                key = f == L.start or f % self.gop == 0
                jobs.append((L, key))
        results = dict(zip((L.id for L, _ in jobs),
                           self._pool.map(lambda j: j[0].encoder.encode_prepared(*j[0].last, j[1]), jobs)))
        keys = {L.id: key for L, key in jobs}
        entries = []
        for i, L in enumerate(self._layers):
            if not isinstance(L, _Video):
                continue
            if L.id in results:
                color, alpha = results[L.id]
                entries.append(VideoEntry(i, ENTRY_FRAME, FRAME_FLAG_KEY if keys[L.id] else 0, color, alpha))
            else:
                entries.append(VideoEntry(i, ENTRY_EMPTY))
        rap = all(keys.values())
        audio: list[AudioPacket] = []
        if self._audio:
            hi = pts_us(f + 1, self.fps.numerator, self.fps.denominator)
            pk = self._audio.packets
            while self._audio_pos < len(pk) and pk[self._audio_pos].pts_us < hi:
                audio.append(pk[self._audio_pos])
                self._audio_pos += 1
        self._out.write_cau(Cau(f, CAU_FLAG_RAP if rap else 0, entries, audio))
        self._frames += 1
        return f

    def end_layer(self, id: str) -> None:
        """The layer's last frame was the previous write(); it is empty from now on."""
        for L in self._layers:
            if L.id == id and isinstance(L, _Video):
                if L.start is None:
                    raise WriterError(f"layer {id!r} has not started yet")
                if L.end is None:
                    L.end = self._frames
                return
        raise WriterError(f"no video layer {id!r}")

    # ------------------------------------------------------------------ finishing
    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._out is None:
                raise WriterError("no frames were written")
            n = self._frames
            for L in self._layers:
                if isinstance(L, _Video):
                    if L.start is None:
                        raise WriterError(f"layer {L.id!r} never received an image")
                    L.encoder.close()
                elif L.start >= n or (L.end is not None and L.end > n):
                    raise WriterError(f"still {L.id!r} range [{L.start}, {L.end}) is outside the {n} frames written")
            self._out.finish(meta=self._meta(n, final=True))
            self._out = None
            os.replace(self._part, self.path)
        except BaseException:
            self.abort()
            raise
        finally:
            self._closed = True
            self._pool.shutdown(wait=True)
        if self.check:
            self.report = validate(str(self.path))
            if not self.report.ok:
                raise WriterError(f"{self.path} failed validation: " + "; ".join(str(i) for i in self.report.errors[:5]))

    def abort(self) -> None:
        """Discard everything written so far."""
        self._closed = True
        if self._out is not None:
            self._out.abort()
            self._out = None
        try:
            self._part.unlink()
        except FileNotFoundError:
            pass
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()
