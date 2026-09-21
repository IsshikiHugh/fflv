"""Derive deliberately broken .lvd files from a valid one (acceptance test 11.2-2).

    fflv corrupt test_assets/test.lvd [--out test_assets/bad] [--check]

Each variant breaks exactly one thing (missing entry, misaligned key frames,
non-contiguous frame numbers, ...). `--check` runs the validator on every
variant and verifies that it reports the expected invariant.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..format import Cau, IndexEntry, LVFReader, LVFWriter, VideoEntry, validate
from ..format.binary import pack_index
from ..format.constants import CAU_FLAG_RAP, ENTRY_EMPTY


@dataclass
class Source:
    meta: dict
    meta_bytes: bytes
    resources: bytes
    caus: list[Cau]

    @property
    def video_layers(self) -> list[int]:
        return [i for i, L in enumerate(self.meta["layers"]) if L["kind"] == "video"]

    def gop(self) -> int:
        return self.meta["max_rap_interval"]

    def late_layer(self) -> int:
        """A video layer that does not start at frame 0 (the test material has one)."""
        for i in self.video_layers:
            if self.meta["layers"][i]["start_frame"] > 0:
                return i
        raise SystemExit("source file has no video layer starting after frame 0")

    def alpha_layer(self) -> int:
        for i in self.video_layers:
            L = self.meta["layers"][i]
            if L["has_alpha"] and L["start_frame"] == 0:
                return i
        raise SystemExit("source file has no alpha layer starting at frame 0")

    def entry(self, frame: int, layer: int) -> VideoEntry:
        return next(e for e in self.caus[frame].entries if e.layer_index == layer)


IndexFix = Callable[[list[IndexEntry]], None]


@dataclass
class Mutation:
    name: str
    description: str
    expect: set[str]          # invariant codes the validator must report
    frame: int | None         # frame the primary error must point at (None: don't care)
    apply: Callable[[Source], IndexFix | None]


def _missing_entry(s: Source):
    f, li = 30, s.video_layers[-1]
    s.caus[f].entries = [e for e in s.caus[f].entries if e.layer_index != li]
    s.caus[f].video_entry_count = None


def _entry_order(s: Source):
    f = 31
    e = s.caus[f].entries
    e[0], e[1] = e[1], e[0]


def _active_entry_empty(s: Source):
    f, li = 32, s.video_layers[0]
    e = s.entry(f, li)
    e.type, e.frame_flags, e.color, e.alpha = ENTRY_EMPTY, 0, b"", b""


def _alpha_key_mismatch(s: Source):
    # At a RAP, swap the alpha plane of one layer for the next frame's (inter-coded) alpha.
    f, li = s.gop(), s.alpha_layer()
    s.entry(f, li).alpha = s.entry(f + 1, li).alpha


def _layer_start_not_key(s: Source):
    li = s.late_layer()
    f = s.meta["layers"][li]["start_frame"]
    e, nxt = s.entry(f, li), s.entry(f + 1, li)
    e.color, e.alpha, e.frame_flags = nxt.color, nxt.alpha, 0


def _rap_flag_cleared(s: Source):
    s.caus[s.gop()].flags &= ~CAU_FLAG_RAP


def _frame0_not_rap(s: Source):
    s.caus[0].flags &= ~CAU_FLAG_RAP


def _frame_index_gap(s: Source):
    s.caus[40].frame_index = 41


def _dropped_cau(s: Source):
    del s.caus[40]


def _audio_wrong_frame(s: Source):
    f = 10
    if not s.caus[f].audio:
        raise SystemExit("source file has no audio in frame 10")
    pk = s.caus[f].audio.pop()
    s.caus[f + 1].audio.insert(0, pk)


def _index_wrong_offset(s: Source) -> IndexFix:
    def fix(entries: list[IndexEntry]) -> None:
        entries[50].cau_offset += 4
    return fix


def _index_wrong_rap(s: Source) -> IndexFix:
    def fix(entries: list[IndexEntry]) -> None:
        entries[s.gop()].flags = 0
    return fix


def _hold_entry(s: Source):
    s.entry(33, s.video_layers[0]).type = 2


MUTATIONS = [
    Mutation("missing_entry", "frame 30 lacks the entry of the last video layer", {"I2"}, 30, _missing_entry),
    Mutation("entry_order", "frame 31 lists its first two video entries in the wrong order", {"I2"}, 31,
             _entry_order),
    Mutation("active_entry_empty", "frame 32: an active layer's entry is EMPTY", {"I3"}, 32, _active_entry_empty),
    Mutation("alpha_key_mismatch", "at the second RAP one layer's alpha plane is an inter frame", {"I5", "I6"},
             None, _alpha_key_mismatch),
    Mutation("layer_start_not_key", "the late-starting layer begins with an inter frame", {"I4"}, None,
             _layer_start_not_key),
    Mutation("rap_flag_cleared", "the RAP flag of the second RAP is cleared", {"I6", "I8"}, None,
             _rap_flag_cleared),
    Mutation("frame0_not_rap", "frame 0 is not marked RAP", {"I6", "I7"}, 0, _frame0_not_rap),
    Mutation("frame_index_gap", "composite frame 40 claims to be frame 41", {"I1"}, 40, _frame_index_gap),
    Mutation("dropped_cau", "composite frame 40 is missing", {"I1"}, 40, _dropped_cau),
    Mutation("audio_wrong_frame", "an audio packet of frame 10 is stored in frame 11", {"I9"}, 11,
             _audio_wrong_frame),
    Mutation("index_wrong_offset", "index entry 50 points 4 bytes too far", {"I10"}, 50, _index_wrong_offset),
    Mutation("index_wrong_rap", "index entry of the second RAP lost its RAP flag", {"I10"}, None,
             _index_wrong_rap),
    Mutation("hold_entry", "frame 33 uses the reserved HOLD entry type", {"CAU"}, 33, _hold_entry),
]


def load(path: str) -> Source:
    with LVFReader(path) as r:
        caus = [cau for _off, cau, _size in r.iter_caus()]
        h = r.header
        return Source(r.meta, r.meta_bytes(), r.read(h.resources_offset, h.cau_offset - h.resources_offset), caus)


def write_variant(src_path: str, m: Mutation, out: Path) -> Path:
    s = load(src_path)
    fix = m.apply(s)
    dst = out / f"{m.name}.lvd"
    w = LVFWriter(dst)
    w.begin(s.meta_bytes, s.resources)
    for cau in s.caus:
        w.write_cau(cau)
    index = list(w.index)
    if fix:
        fix(index)
    w.finish(pack_index(index))
    return dst


def check(path: Path, m: Mutation) -> tuple[bool, str]:
    rep = validate(str(path))
    codes = rep.error_codes()
    missing = m.expect - codes
    if missing:
        return False, f"expected {sorted(m.expect)}, validator reported {sorted(codes)}"
    if m.frame is not None:
        frames = {i.frame for i in rep.errors if i.code in m.expect}
        if m.frame not in frames:
            return False, f"expected an error at frame {m.frame}, got frames {sorted(f for f in frames if f is not None)}"
    first = next(i for i in rep.errors if i.code in m.expect)
    return True, str(first)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fflv corrupt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a valid .lvd (e.g. test_assets/test.lvd)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <source dir>/bad)")
    ap.add_argument("--check", action="store_true", help="validate each variant against its expectation")
    args = ap.parse_args(argv)
    out = args.out or Path(args.source).resolve().parent / "bad"
    out.mkdir(parents=True, exist_ok=True)
    if not validate(args.source).ok:
        print(f"error: {args.source} is not valid to begin with", file=sys.stderr)
        return 2
    failures = 0
    for m in MUTATIONS:
        path = write_variant(args.source, m, out)
        line = f"{m.name:<22} {m.description}"
        if args.check:
            ok, msg = check(path, m)
            failures += not ok
            line += f"\n{'':22} {'OK  ' if ok else 'FAIL'} {msg}"
        print(line)
    if args.check:
        print(f"\n{len(MUTATIONS) - failures}/{len(MUTATIONS)} broken files reported as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
