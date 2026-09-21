"""LVF v1 validator: structure, metadata and the format invariants I1–I10.

Every problem is reported as an Issue tagged with a code. Invariant violations
use the invariant's own name ("I1" … "I10"); other codes:

    HDR   file header            META  metadata JSON
    RES   resource region        CAU   composite-frame structure
    VP9   bitstream vs. metadata AUD   audio packets / audio metadata
"""

from __future__ import annotations

import base64
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .binary import FormatError
from .constants import (
    ALPHA_RANGES,
    BLEND_MODES,
    CAU_FLAG_RAP,
    ENTRY_EMPTY,
    ENTRY_FRAME,
    FRAME_FLAG_KEY,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    INDEX_FLAG_RAP,
    INDEX_HEADER_SIZE,
    LAYER_KINDS,
    MAGIC_FILE,
    MAGIC_INDEX,
    PNG_SIGNATURE,
    VERSION,
)
from .container import LVFReader
from .timing import pts_us
from .vp9 import CS_RGB, Vp9Error, codec_profile, inspect_packet

INVARIANTS = [f"I{i}" for i in range(1, 11)]
MAX_ISSUES_PER_CODE = 25


@dataclass
class Issue:
    code: str
    message: str
    frame: int | None = None
    severity: str = "error"

    def __str__(self) -> str:
        where = f" frame {self.frame}:" if self.frame is not None else ""
        return f"[{self.code}]{where} {self.message}"


@dataclass
class LayerStats:
    frames: int = 0
    keyframes: int = 0
    color_bytes: int = 0
    alpha_bytes: int = 0


@dataclass
class Report:
    path: str
    issues: list[Issue] = field(default_factory=list)
    suppressed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fatal: bool = False
    meta: dict | None = None
    cau_count: int = 0
    rap_frames: list[int] = field(default_factory=list)
    layer_stats: dict[int, LayerStats] = field(default_factory=dict)
    audio_packets: int = 0
    audio_bytes: int = 0
    file_size: int = 0

    def add(self, code: str, message: str, frame: int | None = None, severity: str = "error") -> None:
        same = sum(1 for i in self.issues if i.code == code and i.severity == severity)
        if same >= MAX_ISSUES_PER_CODE:
            self.suppressed[f"{code}/{severity}"] += 1
            return
        self.issues.append(Issue(code, message, frame, severity))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def error_codes(self) -> set[str]:
        codes = {i.code for i in self.errors}
        codes |= {k.split("/")[0] for k in self.suppressed if k.endswith("/error")}
        return codes

    @property
    def ok(self) -> bool:
        return not self.fatal and not self.error_codes()


_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_meta(meta: dict, rep: Report, resources_size: int, reader: LVFReader) -> bool:
    """Returns False when the metadata is too broken to walk the CAUs."""
    ok = True

    def bad(msg: str, fatal: bool = False) -> None:
        nonlocal ok
        rep.add("META", msg)
        if fatal:
            ok = False

    if meta.get("format") != "LVF":
        bad(f"format must be \"LVF\", got {meta.get('format')!r}")
    if meta.get("version") != VERSION:
        bad(f"version must be {VERSION}, got {meta.get('version')!r}")
    canvas = meta.get("canvas")
    if not isinstance(canvas, dict) or not _is_int(canvas.get("width")) or not _is_int(canvas.get("height")) \
            or canvas["width"] <= 0 or canvas["height"] <= 0:
        bad(f"canvas must have positive integer width/height, got {canvas!r}")
    elif not isinstance(canvas.get("background"), str) or not _HEX_COLOR.match(canvas["background"]):
        bad(f"canvas.background must be #RRGGBB, got {canvas.get('background')!r}")
    fps = meta.get("fps")
    if not isinstance(fps, dict) or not _is_int(fps.get("num")) or not _is_int(fps.get("den")) \
            or fps["num"] <= 0 or fps["den"] <= 0:
        bad(f"fps must be {{num, den}} positive integers, got {fps!r}", fatal=True)
    fc = meta.get("frame_count")
    if not _is_int(fc) or fc <= 0:
        bad(f"frame_count must be a positive integer, got {fc!r}", fatal=True)
        fc = 0
    mri = meta.get("max_rap_interval")
    if not _is_int(mri) or mri <= 0:
        bad(f"max_rap_interval must be a positive integer, got {mri!r}", fatal=True)

    layers = meta.get("layers")
    if not isinstance(layers, list):
        bad("layers must be a list", fatal=True)
        return ok
    ids = set()
    for li, L in enumerate(layers):
        tag = f"layer {li}"
        if not isinstance(L, dict):
            bad(f"{tag} is not an object", fatal=True)
            continue
        tag = f"layer {li} ({L.get('id')!r})"
        if not isinstance(L.get("id"), str) or not L["id"]:
            bad(f"{tag}: id must be a non-empty string")
        elif L["id"] in ids:
            bad(f"{tag}: duplicate id")
        ids.add(L.get("id"))
        if not isinstance(L.get("name", ""), str):
            bad(f"{tag}: name must be a string")
        kind = L.get("kind")
        if kind not in LAYER_KINDS:
            bad(f"{tag}: kind must be one of {LAYER_KINDS}, got {kind!r}", fatal=True)
        if not isinstance(L.get("z"), (int, float)) or isinstance(L.get("z"), bool):
            bad(f"{tag}: z must be a number")
        r = L.get("rect")
        if not isinstance(r, dict) or not all(_is_int(r.get(k)) for k in "xywh") or r["w"] <= 0 or r["h"] <= 0:
            bad(f"{tag}: rect must be integer {{x,y,w,h}} with w,h > 0, got {r!r}")
        s, e = L.get("start_frame"), L.get("end_frame")
        if not _is_int(s) or not _is_int(e) or not (0 <= s < e <= fc):
            bad(f"{tag}: need integers 0 <= start_frame < end_frame <= frame_count ({fc}), got [{s!r}, {e!r})",
                fatal=True)
        if L.get("blend") not in BLEND_MODES:
            bad(f"{tag}: blend must be one of {BLEND_MODES}, got {L.get('blend')!r}")
        op = L.get("opacity")
        if not isinstance(op, (int, float)) or isinstance(op, bool) or not 0.0 <= op <= 1.0:
            bad(f"{tag}: opacity must be a number in [0, 1], got {op!r}")
        if not isinstance(L.get("visible"), bool):
            bad(f"{tag}: visible must be a boolean, got {L.get('visible')!r}")
        if kind == "video":
            if not isinstance(L.get("codec"), str) or codec_profile(L["codec"]) not in (0, 1):
                bad(f"{tag}: codec must be a vp09.00.* or vp09.01.* string, got {L.get('codec')!r}")
            if not isinstance(L.get("lossless", False), bool):
                bad(f"{tag}: lossless must be a boolean")
            cs = L.get("content_size")
            if cs is not None and not (isinstance(cs, list) and len(cs) == 2 and all(_is_int(v) and v > 0 for v in cs)
                                       and cs[0] <= (L.get("coded_width") or 0) and cs[1] <= (L.get("coded_height") or 0)):
                bad(f"{tag}: content_size must be [w, h] within the coded size, got {cs!r}")
            if L.get("alpha_range", "limited") not in ALPHA_RANGES:
                bad(f"{tag}: alpha_range must be one of {ALPHA_RANGES}, got {L.get('alpha_range')!r}")
            cw, ch = L.get("coded_width"), L.get("coded_height")
            if not _is_int(cw) or not _is_int(ch) or cw <= 0 or ch <= 0:
                bad(f"{tag}: coded_width/coded_height must be positive integers")
            ha = L.get("has_alpha")
            if not isinstance(ha, bool):
                bad(f"{tag}: has_alpha must be a boolean", fatal=True)
            elif ha and not (isinstance(L.get("alpha_codec"), str) and L["alpha_codec"].startswith("vp09.")):
                bad(f"{tag}: has_alpha is true but alpha_codec is {L.get('alpha_codec')!r}")
            elif not ha and L.get("alpha_codec") is not None:
                bad(f"{tag}: has_alpha is false but alpha_codec is {L.get('alpha_codec')!r}")
        elif kind == "still":
            res = L.get("resource")
            if not isinstance(res, dict) or not _is_int(res.get("offset")) or not _is_int(res.get("length")):
                rep.add("RES", f"{tag}: resource must be {{offset, length, mime}} integers")
                continue
            if res["offset"] < 0 or res["length"] <= 0 or res["offset"] + res["length"] > resources_size:
                rep.add("RES", f"{tag}: resource [{res['offset']}, +{res['length']}) is outside the "
                               f"{resources_size}-byte resource region")
                continue
            if res.get("mime") != "image/png":
                rep.add("RES", f"{tag}: mime must be image/png, got {res.get('mime')!r}")
            elif reader.resource(res["offset"], 8) != PNG_SIGNATURE:
                rep.add("RES", f"{tag}: resource does not start with the PNG signature")

    audio = meta.get("audio", "missing")
    if audio == "missing":
        bad("audio key is missing (use null for no audio)")
    elif audio is not None:
        if not isinstance(audio, dict):
            rep.add("AUD", "audio must be an object or null")
        else:
            if audio.get("codec") != "opus":
                rep.add("AUD", f"audio.codec must be \"opus\", got {audio.get('codec')!r}")
            if audio.get("sample_rate") != 48000:
                rep.add("AUD", f"audio.sample_rate must be 48000, got {audio.get('sample_rate')!r}")
            if not _is_int(audio.get("channels")) or not 1 <= audio["channels"] <= 2:
                rep.add("AUD", f"audio.channels must be 1 or 2, got {audio.get('channels')!r}")
            d = audio.get("description_b64")
            if d is not None:
                try:
                    head = base64.b64decode(d, validate=True)
                    if head[:8] != b"OpusHead":
                        rep.add("AUD", "audio.description_b64 is not an OpusHead")
                except (ValueError, TypeError):
                    rep.add("AUD", "audio.description_b64 is not valid base64")
    return ok


def validate(path: str, check_bitstream: bool = True) -> Report:
    rep = Report(path)
    try:
        reader = LVFReader(path)
    except (OSError, FormatError) as exc:
        rep.add("HDR", f"cannot read file header: {exc}")
        rep.fatal = True
        return rep
    with reader:
        return _validate(reader, rep, check_bitstream)


def _validate(reader: LVFReader, rep: Report, check_bitstream: bool) -> Report:
    h = reader.header
    rep.file_size = reader.file_size

    # ---- header -------------------------------------------------------------------
    if h.magic != MAGIC_FILE:
        rep.add("HDR", f"magic is {h.magic!r}, expected {MAGIC_FILE!r}")
        rep.fatal = True
        return rep
    if h.version != VERSION:
        rep.add("HDR", f"version is {h.version}, expected {VERSION}")
    if h.flags != 0:
        rep.add("HDR", f"flags is {h.flags:#x}, must be 0 in v1")
    if h.reserved_20 != 0 or any(h.reserved_48):
        rep.add("HDR", "reserved header bytes are not zero", severity="warning")
    order = [("meta_offset", h.meta_offset), ("resources_offset", h.resources_offset),
             ("cau_offset", h.cau_offset), ("index_offset", h.index_offset)]
    if h.meta_offset < HEADER_SIZE:
        rep.add("HDR", f"meta_offset {h.meta_offset} overlaps the header")
        rep.fatal = True
    if h.meta_offset + h.meta_length > h.resources_offset:
        rep.add("HDR", f"metadata [{h.meta_offset}, +{h.meta_length}) runs into resources_offset "
                       f"{h.resources_offset}")
        rep.fatal = True
    for (na, a), (nb, b) in zip(order, order[1:]):
        if a > b:
            rep.add("HDR", f"{na} ({a}) > {nb} ({b})")
            rep.fatal = True
    if h.index_offset > reader.file_size:
        rep.add("HDR", f"index_offset {h.index_offset} is past the end of the file ({reader.file_size})")
        rep.fatal = True
    if rep.fatal:
        return rep

    # ---- metadata -----------------------------------------------------------------
    try:
        meta = reader.meta
    except (ValueError, UnicodeDecodeError) as exc:
        rep.add("META", f"metadata is not valid UTF-8 JSON: {exc}")
        rep.fatal = True
        return rep
    rep.meta = meta
    if not isinstance(meta, dict):
        rep.add("META", "metadata is not a JSON object")
        rep.fatal = True
        return rep
    if not _check_meta(meta, rep, h.cau_offset - h.resources_offset, reader):
        rep.fatal = True
        return rep

    num, den = meta["fps"]["num"], meta["fps"]["den"]
    frame_count = meta["frame_count"]
    max_rap = meta["max_rap_interval"]
    layers = meta["layers"]
    video_layers = [i for i, L in enumerate(layers) if L["kind"] == "video"]
    has_audio_meta = meta.get("audio") is not None
    rep.layer_stats = {i: LayerStats() for i in video_layers}

    # ---- composite frames (sequential walk, independent of the index) ---------------
    actual: list[tuple[int, int, int]] = []  # (offset, frame_index, flags) in file order
    last_audio_pts = None
    expect_next = 0
    try:
        for k, (offset, cau, _size) in enumerate(reader.iter_caus()):
            actual.append((offset, cau.frame_index, cau.flags))
            f = cau.frame_index
            if f != expect_next:
                # Report each discontinuity once, then follow the file's own numbering.
                if f == expect_next + 1:
                    what = f"frame {expect_next} is missing"
                elif f > expect_next:
                    what = f"frames {expect_next}..{f - 1} are missing"
                else:
                    what = "frame number goes backwards or repeats"
                rep.add("I1", f"CAU #{k} (offset {offset}) has frame_index {f}, expected {expect_next} "
                              f"({what})", k)
            expect_next = f + 1
            if cau.flags & ~CAU_FLAG_RAP:
                rep.add("CAU", f"undefined flag bits set: {cau.flags:#04x}", f)
            if cau.reserved_13 or cau.reserved_18:
                rep.add("CAU", "reserved CAU header fields are not zero", f, "warning")
            if not 0 <= f < frame_count:
                rep.add("I1", f"frame_index {f} is outside [0, {frame_count})", k)
                continue

            # I2: exactly one entry per video layer, ascending layer_index
            got = [e.layer_index for e in cau.entries]
            if cau.video_entry_count != len(video_layers) or got != video_layers:
                missing = sorted(set(video_layers) - set(got))
                extra = sorted(set(got) - set(video_layers))
                detail = []
                if missing:
                    detail.append(f"missing layers {missing}")
                if extra:
                    detail.append(f"entries for non-video/unknown layers {extra}")
                if not missing and not extra:
                    detail.append(f"layer order {got} (expected {video_layers})")
                rep.add("I2", f"video_entry_count={cau.video_entry_count}, expected {len(video_layers)}; "
                              + "; ".join(detail), f)

            rap_should = True
            seen = set()
            for e in cau.entries:
                li = e.layer_index
                if li in seen or li not in rep.layer_stats:
                    continue  # already reported under I2
                seen.add(li)
                L = layers[li]
                active = L["start_frame"] <= f < L["end_frame"]
                if e.type not in (ENTRY_EMPTY, ENTRY_FRAME):
                    rep.add("CAU", f"layer {li}: entry type {e.type} is not valid in v1", f)
                    rap_should = False
                    continue
                if (e.type == ENTRY_FRAME) != active:
                    rep.add("I3", f"layer {li} ({L['id']}) is {'active' if active else 'inactive'} in "
                                  f"[{L['start_frame']}, {L['end_frame']}) but its entry is "
                                  f"{'FRAME' if e.type == ENTRY_FRAME else 'EMPTY'}", f)
                if e.type == ENTRY_EMPTY:
                    if e.color or e.alpha or e.frame_flags:
                        rep.add("CAU", f"layer {li}: EMPTY entry carries data or flags", f)
                    continue

                st = rep.layer_stats[li]
                st.frames += 1
                st.color_bytes += len(e.color)
                st.alpha_bytes += len(e.alpha)
                if e.frame_flags & ~FRAME_FLAG_KEY:
                    rep.add("CAU", f"layer {li}: undefined frame_flags bits {e.frame_flags:#04x}", f)
                if not e.color:
                    rep.add("CAU", f"layer {li}: FRAME entry without color data", f)
                    rap_should = False
                    continue
                if L["has_alpha"] and not e.alpha:
                    rep.add("CAU", f"layer {li}: has_alpha layer without alpha data", f)
                if not L["has_alpha"] and e.alpha:
                    rep.add("CAU", f"layer {li}: alpha data on a layer without has_alpha", f)

                flag_key = e.is_key
                color_key = alpha_key = flag_key
                if check_bitstream:
                    color_key = _check_packet(rep, f, li, "color", e.color, flag_key, L)
                    if e.alpha:
                        alpha_key = _check_packet(rep, f, li, "alpha", e.alpha, flag_key, L)
                planes_key = color_key and (alpha_key if e.alpha else True)
                if planes_key:
                    st.keyframes += 1
                if e.alpha and color_key != alpha_key:
                    rep.add("I5", f"layer {li} ({L['id']}): color is {'key' if color_key else 'delta'} "
                                  f"but alpha is {'key' if alpha_key else 'delta'}", f)
                if f == L["start_frame"] and not (flag_key and planes_key):
                    rep.add("I4", f"layer {li} ({L['id']}) starts here but its "
                                  f"{'color' if not color_key else 'alpha'} frame is not a key frame", f)
                rap_should = rap_should and flag_key and planes_key

            is_rap = bool(cau.flags & CAU_FLAG_RAP)
            if is_rap != rap_should:
                rep.add("I6", f"RAP flag is {int(is_rap)} but the frame's entries say it should be "
                              f"{int(rap_should)}", f)
            if is_rap:
                rep.rap_frames.append(f)
            if f == 0 and not is_rap:
                rep.add("I7", "frame 0 is not a RAP", f)

            # I9 + audio sanity
            lo, hi = pts_us(f, num, den), pts_us(f + 1, num, den)
            if cau.audio and not has_audio_meta:
                rep.add("AUD", f"{len(cau.audio)} audio packets but metadata audio is null", f)
            for a in cau.audio:
                rep.audio_packets += 1
                rep.audio_bytes += len(a.data)
                if not lo <= a.pts_us < hi:
                    rep.add("I9", f"audio packet pts {a.pts_us} us is outside this frame's window "
                                  f"[{lo}, {hi})", f)
                if not a.data:
                    rep.add("AUD", "empty audio packet", f)
                if a.duration_us <= 0:
                    rep.add("AUD", f"audio packet with duration {a.duration_us} us", f, "warning")
                if last_audio_pts is not None and a.pts_us <= last_audio_pts:
                    rep.add("AUD", f"audio pts {a.pts_us} does not increase (previous {last_audio_pts})", f)
                last_audio_pts = a.pts_us
    except FormatError as exc:
        rep.add("CAU", f"composite-frame region is structurally broken after {len(actual)} CAUs: {exc}")

    rep.cau_count = len(actual)
    if rep.cau_count != frame_count:
        rep.add("I1", f"file holds {rep.cau_count} composite frames, frame_count is {frame_count}")

    # I8: distance between adjacent RAPs
    raps = rep.rap_frames
    for a, b in zip(raps, raps[1:]):
        if b - a > max_rap:
            rep.add("I8", f"RAPs at {a} and {b} are {b - a} frames apart (max_rap_interval {max_rap})", b)
    if raps and frame_count - raps[-1] > max_rap:
        rep.add("I8", f"last RAP at {raps[-1]} leaves {frame_count - raps[-1]} frames to the end "
                      f"(max_rap_interval {max_rap})", raps[-1], "warning")

    # ---- index (I10) ------------------------------------------------------------------
    _check_index(reader, rep, actual, frame_count)
    return rep


def _check_packet(rep: Report, f: int, li: int, plane: str, data: bytes, flag_key: bool, L: dict) -> bool:
    """Checks one VP9 packet; returns whether the bitstream says it is a key frame.

    Color planes are profile 0 (8-bit 4:2:0) or, for lossless layers, profile 1 (8-bit 4:4:4),
    as their codec string says. Alpha planes are profile 0 luma whose signalled range must match
    the layer's alpha_range.
    """
    try:
        pk = inspect_packet(data)
    except Vp9Error as exc:
        rep.add("VP9", f"layer {li} {plane}: cannot parse VP9 header: {exc}", f)
        return flag_key
    if pk.shown_count != 1:
        rep.add("VP9", f"layer {li} {plane}: packet shows {pk.shown_count} frames (must be exactly 1)", f)
    if pk.key_frame != flag_key:
        rep.add("VP9", f"layer {li} {plane}: frame_flags says {'key' if flag_key else 'delta'} but the "
                       f"bitstream is a {'key' if pk.key_frame else 'delta'} frame", f)
    ki = pk.key_info
    if ki is not None:
        if (ki.width, ki.height) != (L["coded_width"], L["coded_height"]):
            rep.add("VP9", f"layer {li} {plane}: key frame is {ki.width}x{ki.height}, metadata says "
                           f"{L['coded_width']}x{L['coded_height']}", f)
        codec = L.get("codec") if plane == "color" else L.get("alpha_codec")
        profile = codec_profile(codec or "") or 0
        want_sub = (1, 1) if profile == 0 else (0, 0)
        if ki.profile != profile or ki.bit_depth != 8 or ki.subsampling != want_sub:
            rep.add("VP9", f"layer {li} {plane}: bitstream is profile {ki.profile}, {ki.bit_depth}-bit, "
                           f"subsampling {ki.subsampling}; codec {codec!r} requires profile {profile}, "
                           f"8-bit, subsampling {want_sub}", f)
        if plane == "alpha":
            want_range = 1 if L.get("alpha_range", "limited") == "full" else 0
            if ki.color_range != want_range:
                rep.add("VP9", f"layer {li} alpha: bitstream signals "
                               f"{'full' if ki.color_range else 'limited'} range, metadata alpha_range is "
                               f"{L.get('alpha_range', 'limited')!r}", f)
        elif profile == 1 and ki.color_space != CS_RGB:
            rep.add("VP9", f"layer {li} color: profile-1 planes must be RGB (color_space 7), got "
                           f"{ki.color_space}", f)
    return pk.key_frame


def _check_index(reader: LVFReader, rep: Report, actual: list[tuple[int, int, int]], frame_count: int) -> None:
    h = reader.header
    region = reader.file_size - h.index_offset
    try:
        magic, count, entries = reader.index()
    except FormatError as exc:
        rep.add("I10", f"index table unreadable: {exc}")
        return
    if magic != MAGIC_INDEX:
        rep.add("I10", f"index magic is {magic!r}, expected {MAGIC_INDEX!r}")
    if count != frame_count:
        rep.add("I10", f"index count is {count}, frame_count is {frame_count}")
    expect_size = INDEX_HEADER_SIZE + count * INDEX_ENTRY_SIZE
    if region != expect_size:
        rep.add("I10", f"{region - expect_size} trailing bytes after the index table")
    if len(entries) != len(actual):
        rep.add("I10", f"index has {len(entries)} entries but the file holds {len(actual)} composite frames")
    prev = -1
    for i, e in enumerate(entries):
        if e.frame_index != i and e.frame_index != prev + 1:  # report each discontinuity once
            rep.add("I10", f"index entry {i} has frame_index {e.frame_index}", i)
        prev = e.frame_index
        if e.flags & ~INDEX_FLAG_RAP:
            rep.add("I10", f"index entry {i} has undefined flag bits {e.flags:#04x}", i)
        if i < len(actual):
            off, fi, fl = actual[i]
            if e.cau_offset != off:
                rep.add("I10", f"index entry {i} points to offset {e.cau_offset}, CAU #{i} is at {off}", i)
            if bool(e.flags & INDEX_FLAG_RAP) != bool(fl & CAU_FLAG_RAP):
                rep.add("I10", f"index entry {i} RAP flag {e.flags & 1} differs from CAU header {fl & 1}", i)
