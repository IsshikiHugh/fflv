"""`fflv info`: human-readable dump of an .lvd file (header, metadata, one composite frame)."""

from __future__ import annotations

import copy
import json

from .format import FormatError, LVFReader, pts_us
from .format.constants import ENTRY_EMPTY, ENTRY_FRAME, ENTRY_HOLD
from .format.vp9 import Vp9Error, inspect_packet

ENTRY_NAMES = {ENTRY_EMPTY: "EMPTY", ENTRY_FRAME: "FRAME", ENTRY_HOLD: "HOLD"}


def print_header(r: LVFReader) -> None:
    h = r.header
    print(f"magic {h.magic!r}  version {h.version}  flags {h.flags:#06x}")
    print(f"meta       offset {h.meta_offset:>12,}  length {h.meta_length:,}")
    print(f"resources  offset {h.resources_offset:>12,}  length {h.cau_offset - h.resources_offset:,}")
    print(f"CAUs       offset {h.cau_offset:>12,}  length {h.index_offset - h.cau_offset:,}")
    print(f"index      offset {h.index_offset:>12,}  length {r.file_size - h.index_offset:,}")


def print_meta(meta: dict) -> None:
    m = copy.deepcopy(meta)
    a = m.get("audio")
    if isinstance(a, dict) and isinstance(a.get("description_b64"), str) and len(a["description_b64"]) > 40:
        a["description_b64"] = a["description_b64"][:40] + "…"
    print(json.dumps(m, ensure_ascii=False, indent=2))


def locate_cau(r: LVFReader, n: int) -> int:
    try:
        _, count, entries = r.index()
        if n < len(entries):
            return entries[n].cau_offset
    except FormatError:
        pass
    for k, (off, _cau, _size) in enumerate(r.iter_caus()):
        if k == n:
            return off
    raise FormatError(f"file has no composite frame #{n}")


def describe_vp9(data: bytes) -> str:
    try:
        pk = inspect_packet(data)
    except Vp9Error as exc:
        return f"unparseable ({exc})"
    fr = pk.frames[0]
    s = "key" if pk.key_frame else "inter"
    if pk.superframe:
        s += f", superframe of {len(pk.frames)}"
    if fr.width:
        s += f", {fr.width}x{fr.height}, cs={fr.color_space} range={'full' if fr.color_range else 'limited'}"
    return s


def print_frame(r: LVFReader, n: int) -> None:
    meta = r.meta
    layers = meta["layers"]
    num, den = meta["fps"]["num"], meta["fps"]["den"]
    off = locate_cau(r, n)
    cau, size = r.cau_at(off)
    f = cau.frame_index
    print(f"composite frame #{n} at offset {off:,}, {size:,} bytes")
    print(f"  frame_index {f}  flags {cau.flags:#04x}{' (RAP)' if cau.is_rap else ''}  "
          f"pts {pts_us(f, num, den)} us  window [{pts_us(f, num, den)}, {pts_us(f + 1, num, den)})")
    print(f"  video entries {cau.video_entry_count}, audio packets {len(cau.audio)}")
    for e in cau.entries:
        L = layers[e.layer_index] if 0 <= e.layer_index < len(layers) else {}
        line = f"    layer {e.layer_index:<2} {L.get('id', '?'):<12} {ENTRY_NAMES.get(e.type, e.type)!s:<5}"
        if e.type == ENTRY_FRAME:
            line += f" {'KEY' if e.is_key else '   '} color {len(e.color):>7,} B [{describe_vp9(e.color)}]"
            if e.alpha:
                line += f"\n{'':30}alpha {len(e.alpha):>7,} B [{describe_vp9(e.alpha)}]"
        print(line)
    for a in cau.audio:
        print(f"    audio pts {a.pts_us:>12} us  duration {a.duration_us:>6} us  {len(a.data):>5} B")
    stills = [f"{L['id']}" for L in layers if L.get("kind") == "still" and L["start_frame"] <= f < L["end_frame"]]
    if stills:
        print(f"  still layers shown: {', '.join(stills)}")
