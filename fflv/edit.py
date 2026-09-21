"""Editing .lvd files without re-encoding what is already there.

  add_layer    new video layer from a media file, a sequence/iterator of numpy images or a callable
  add_still    new still layer (PNG / image / array)
  remove_layers, set_audio (replace or remove), set_layer (id, name, z, rect, blend, opacity, visible)

Existing layers are never re-encoded: their packets are copied bit for bit into a rewritten file
(written beside the output, validated, then atomically renamed over it). Only a newly added video
layer is encoded, with key frames exactly on the file's existing random-access points, so random
access stays the same. `set_layer` rewrites the metadata in place — instant, whatever the file size —
when it fits the space reserved after the metadata (it always does for ordinary edits).
"""

from __future__ import annotations

import copy
import itertools
import os
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from . import meta as M
from .encode.audio import AudioTrack, encode_audio
from .encode.media import open_video_source, png_size, raw_frames, still_png
from .encode.vp9enc import EncodeOptions, LayerEncoder
from .format import Cau, LVFReader, LVFWriter, Report, VideoEntry, encode_meta, pts_us, rewrite_meta_in_place, validate
from .format.constants import CAU_FLAG_RAP, ENTRY_EMPTY, ENTRY_FRAME, FRAME_FLAG_KEY


class EditError(RuntimeError):
    pass


@dataclass
class _NewVideo:
    meta: dict
    entries: Iterator[tuple[bytes, bytes, bool]]  # one per active frame, in order


@dataclass
class _NewStill:
    meta: dict
    png: bytes


KEEP = "keep"


def _tmp_path(dst: Path) -> Path:
    return dst.with_name(f".{dst.name}.fflv-tmp")


def remux(src: str | os.PathLike, dst: str | os.PathLike | None = None, *, drop: Iterable[int] = (),
          add_video: Iterable[_NewVideo] = (), add_still: Iterable[_NewStill] = (),
          audio: str | AudioTrack | None = KEEP, meta_patch: Callable[[dict], None] | None = None,
          check: bool = True) -> Report | None:
    """Rewrite `src` into `dst` (default: in place): drop layers, append layers, replace audio."""
    src, dst = Path(src), Path(dst) if dst is not None else Path(src)
    drop = set(drop)
    add_video, add_still = list(add_video), list(add_still)
    tmp = _tmp_path(dst)
    with LVFReader(src) as r:
        meta = copy.deepcopy(r.meta)
        num, den = meta["fps"]["num"], meta["fps"]["den"]
        old = meta["layers"]
        keep = [i for i in range(len(old)) if i not in drop]
        remap = {o: n for n, o in enumerate(keep)}
        layers = [old[i] for i in keep] + [v.meta for v in add_video] + [s.meta for s in add_still]
        ids = [L["id"] for L in layers]
        if len(set(ids)) != len(ids):
            raise EditError(f"duplicate layer ids after the edit: {ids}")

        # resources: kept stills first, then new ones
        blobs = []
        for L in layers:
            if L["kind"] != "still":
                continue
            s = next((x for x in add_still if x.meta is L), None)
            data = s.png if s else r.resource(L["resource"]["offset"], L["resource"]["length"])
            L["resource"] = {"offset": sum(len(b) for b in blobs), "length": len(data), "mime": "image/png"}
            blobs.append(data)

        new_track = None
        if audio is None:
            meta["audio"] = None
        elif isinstance(audio, AudioTrack):
            meta["audio"] = audio.meta()
            new_track = audio.packets_before(pts_us(meta["frame_count"], num, den))
        meta["layers"] = layers
        meta["generator"] = M.GENERATOR
        if meta_patch:
            meta_patch(meta)

        first_new = len(keep)
        new_streams = [(first_new + k, v.meta, v.entries) for k, v in enumerate(add_video)]
        tmp.parent.mkdir(parents=True, exist_ok=True)
        w = LVFWriter(tmp)
        try:
            w.begin(encode_meta(meta), b"".join(blobs))
            ai = 0
            for _off, cau, _size in r.iter_caus():
                f = cau.frame_index
                entries = []
                for e in cau.entries:
                    if e.layer_index in remap:
                        e.layer_index = remap[e.layer_index]
                        entries.append(e)
                for idx, L, stream in new_streams:
                    if L["start_frame"] <= f < L["end_frame"]:
                        try:
                            color, alpha, key = next(stream)
                        except StopIteration:
                            raise EditError(f"layer {L['id']!r}: source ran out of images at frame {f}") from None
                        entries.append(VideoEntry(idx, ENTRY_FRAME, FRAME_FLAG_KEY if key else 0, color, alpha))
                    else:
                        entries.append(VideoEntry(idx, ENTRY_EMPTY))
                rap = all(e.is_key for e in entries if e.type == ENTRY_FRAME)
                if audio == KEEP:
                    pk = cau.audio
                elif new_track is not None:
                    hi = pts_us(f + 1, num, den)
                    pk = []
                    while ai < len(new_track) and new_track[ai].pts_us < hi:
                        pk.append(new_track[ai])
                        ai += 1
                else:
                    pk = []
                w.write_cau(Cau(f, CAU_FLAG_RAP if rap else 0, entries, pk))
            w.finish()
        except BaseException:
            w.abort()
            raise
    rep = None
    if check:
        rep = validate(str(tmp))
        if not rep.ok:
            tmp.unlink(missing_ok=True)
            raise EditError("edited file failed validation: " + "; ".join(str(i) for i in rep.errors[:5]))
    os.replace(tmp, dst)
    return rep


# --------------------------------------------------------------------------------------------------
# Adding layers
# --------------------------------------------------------------------------------------------------
def _file_info(path) -> tuple[dict, list[int]]:
    with LVFReader(path) as r:
        _, _, entries = r.index()
        return copy.deepcopy(r.meta), [i for i, e in enumerate(entries) if e.is_rap]


def _range(meta: dict, start: int, end: int | None, n_images: int | None) -> tuple[int, int]:
    fc = meta["frame_count"]
    if end is None:
        end = start + n_images if n_images is not None else fc
    if not 0 <= start < end <= fc:
        raise EditError(f"frame range [{start}, {end}) is outside the file's [0, {fc})")
    return start, end


def _encode_stream(images: Iterator, enc: LayerEncoder, start: int, end: int, raps: list[int]
                   ) -> Iterator[tuple[bytes, bytes, bool]]:
    """Encode images for frames [start, end): key frames at `start` and at every existing RAP."""
    keys = {start} | {r for r in raps if start < r < end}
    try:
        for f in range(start, end):
            img = next(images, None)
            if img is None:
                raise EditError(f"source ran out of images at frame {f} (needed [{start}, {end}))")
            color, alpha = enc.encode(img, f in keys)
            yield color, alpha, f in keys
        enc.close()
    finally:
        close = getattr(images, "close", None)
        if close:
            close()


def add_layer(path, id: str, source, *, output=None, start: int = 0, end: int | None = None,
              alpha: bool | None = None, lossless: bool = False, rect=None, z: float | None = None,
              name: str | None = None, blend: str = "normal", opacity: float = 1.0, visible: bool = True,
              crf: int = 32, speed: str = "balanced", check: bool = True) -> Report | None:
    """Append a video layer.

    `source`: a media file (scaled to `rect`, default the whole canvas), a sequence or iterator of
    images, or a callable `frame -> image`. Images are H×W / H×W×3 / H×W×4 arrays (see
    `fflv.Writer.write`); with arrays, `rect` defaults to the first image's size at (0, 0).
    `alpha=None` picks alpha if the source has an alpha channel. Frames: [start, end), `end`
    defaults to the source length (sequences) or the end of the file.
    """
    meta, raps = _file_info(path)
    M.check_id(id, {L["id"] for L in meta["layers"]})
    fps = Fraction(meta["fps"]["num"], meta["fps"]["den"])
    W, H = meta["canvas"]["width"], meta["canvas"]["height"]
    opts = EncodeOptions(crf=crf, speed=speed)

    if isinstance(source, (str, os.PathLike)):
        src = open_video_source(source)
        if alpha is None:
            alpha = src.has_alpha
        elif alpha and not src.has_alpha:
            raise EditError(f"{src.path.name} ({src.codec}, {src.pix_fmt}) has no alpha channel")
        r = M.check_rect(rect if rect is not None else (0, 0, W, H))
        start, end = _range(meta, start, end, None)
        images = raw_frames(src, fps, r["w"], r["h"], end - start, alpha)
    else:
        if callable(source):
            first_f = start
            images = (source(f) for f in itertools.count(first_f))
            n = None
        else:
            n = len(source) if hasattr(source, "__len__") else None
            images = iter(source)
        first = next(images, None)
        if first is None:
            raise EditError("the image source is empty")
        first = np.asarray(first)
        images = itertools.chain([first], images)
        if alpha is None:
            alpha = first.ndim == 3 and first.shape[2] == 4
        r = M.check_rect(rect if rect is not None else (0, 0, first.shape[1], first.shape[0]))
        start, end = _range(meta, start, end, n)

    enc = LayerEncoder(r["w"], r["h"], fps, alpha=alpha, lossless=lossless, options=opts, name=id)
    top = max((L["z"] for L in meta["layers"]), default=-1)
    L = M.video_layer(id=id, name=name or id, z=top + 1 if z is None else z, rect=r, start=start, end=end, fps=fps,
                      alpha=alpha, lossless=lossless, blend=M.check_blend(blend),
                      opacity=M.check_opacity(opacity), visible=bool(visible))
    stream = _encode_stream(images, enc, start, end, raps)
    return remux(path, output, add_video=[_NewVideo(L, stream)], check=check)


def add_still(path, id: str, image, *, output=None, rect=None, start: int = 0, end: int | None = None,
              z: float | None = None, name: str | None = None, blend: str = "normal", opacity: float = 1.0,
              visible: bool = True, check: bool = True) -> Report | None:
    meta, _ = _file_info(path)
    M.check_id(id, {L["id"] for L in meta["layers"]})
    png = still_png(image)
    if rect is None:
        w, h = png_size(png)
        rect = (0, 0, w, h)
    start, end = _range(meta, start, end, None)
    top = max((L["z"] for L in meta["layers"]), default=-1)
    L = M.still_layer(id=id, name=name or id, z=top + 1 if z is None else z, rect=M.check_rect(rect), start=start,
                      end=end, offset=0, length=len(png), blend=M.check_blend(blend),
                      opacity=M.check_opacity(opacity), visible=bool(visible))
    return remux(path, output, add_still=[_NewStill(L, png)], check=check)


def remove_layers(path, keys: Iterable, *, output=None, check: bool = True) -> Report | None:
    meta, _ = _file_info(path)
    drop = {M.resolve_layer(meta, k) for k in keys}
    if not drop:
        raise EditError("no layers given")
    return remux(path, output, drop=drop, check=check)


def set_audio(path, source, *, output=None, bitrate: str = "128k", channels: int = 2,
              check: bool = True) -> Report | None:
    """Replace the audio track with `source` (any file FFmpeg reads), or remove it (None)."""
    if source is None:
        return remux(path, output, audio=None, check=check)
    meta, _ = _file_info(path)
    seconds = meta["frame_count"] * meta["fps"]["den"] / meta["fps"]["num"]
    return remux(path, output, audio=encode_audio(source, max_seconds=seconds, bitrate=bitrate, channels=channels),
                 check=check)


def set_layer(path, key, *, output=None, **fields) -> bool:
    """Change id / name / z / rect / blend / opacity / visible of one layer.
    Returns True when done in place (metadata only), False when the file had to be rewritten."""
    if output is not None and Path(output) != Path(path):
        shutil.copyfile(path, output)
        path = output
    meta, _ = _file_info(path)
    M.apply_edits(meta, key, fields)
    meta["generator"] = M.GENERATOR
    if rewrite_meta_in_place(path, meta):
        return True

    def patch(m: dict) -> None:
        m.clear()
        m.update(meta)

    remux(path, None, meta_patch=patch)
    return False
