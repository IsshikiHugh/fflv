"""fflv — command line for LVF layered video files (.lvd).

  fflv info    FILE [--frame N] [--json]         show header, layers, statistics; check invariants
  fflv check   FILE...                           validate (exit 1 if invalid)
  fflv pack    PROJECT.json [-o OUT]             build a file from a project (spec 8.1)
  fflv add     FILE --src MEDIA | --still IMG | --audio MEDIA  [layer options] [-o OUT]
  fflv rm      FILE LAYER... | --audio [-o OUT]  remove layers / the audio track
  fflv set     FILE LAYER key=value... [-o OUT]  id, name, z, rect, blend, opacity, visible (in place)
  fflv render  FILE -o OUT [-l LAYERS] [--hide LAYERS] [-f RANGE]   composite to images / video / .npy
  fflv extract FILE LAYER -o OUT [-f RANGE]      one layer's own pixels (RGBA)
  fflv view    FILE                              open the interactive player in Chrome / Edge
  fflv testsrc / fflv corrupt                    generate test material / broken files (development)

LAYER is a layer id or index. RANGE is `N`, `A:B` (B excluded), `A:` or `:B`, in frames or,
with an `s` suffix, in seconds (`1.5s:3s`). Edits rewrite the file in place (atomically) unless
-o is given; existing layers are never re-encoded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction

from . import __version__
from . import meta as M
from .format import FormatError, LVFReader, seconds_to_frame, validate
from .format.report import print_report


class CliError(Exception):
    pass


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _point(v: str, fps: Fraction) -> int:
    v = v.strip()
    try:
        return seconds_to_frame(v[:-1], fps) if v.endswith("s") else int(v)
    except (ValueError, ZeroDivisionError):
        raise CliError(f"bad frame/time {v!r} (use a frame number or seconds like 1.5s)") from None


def parse_range(spec: str | None, fps: Fraction, frame_count: int) -> tuple[int, int]:
    if not spec:
        return 0, frame_count
    if ":" not in spec:
        n = _point(spec, fps)
        return n, n + 1
    a, b = spec.split(":", 1)
    start = _point(a, fps) if a.strip() else 0
    end = _point(b, fps) if b.strip() else frame_count
    if not 0 <= start < end <= frame_count:
        raise CliError(f"range {spec!r} = frames [{start}, {end}) is outside [0, {frame_count})")
    return start, end


def _file_timing(path: str) -> tuple[Fraction, int]:
    with LVFReader(path) as r:
        m = r.meta
        return Fraction(m["fps"]["num"], m["fps"]["den"]), m["frame_count"]


def _layers_arg(v: str | None) -> list[str] | None:
    return [s for s in v.split(",") if s] if v else None


class _Progress:
    def __init__(self, what: str):
        self.what, self.t0, self.last = what, time.monotonic(), 0.0
        self.tty = sys.stderr.isatty()

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if self.tty and (now - self.last > 0.2 or done == total):
            self.last = now
            fps = done / max(1e-6, now - self.t0)
            sys.stderr.write(f"\r{self.what}: {done}/{total} frames ({fps:.0f} fps)  ")
            if done == total:
                sys.stderr.write("\n")
            sys.stderr.flush()


def _report_edit(path: str, rep) -> None:
    fps, n = _file_timing(path)
    with LVFReader(path) as r:
        ids = [f"{L['id']}({L['kind'][0]})" for L in r.meta["layers"]]
        audio = "audio" if r.meta.get("audio") else "no audio"
    print(f"{path}: {n} frames, layers {', '.join(ids)}; {audio}" + ("" if rep is None or rep.ok else " — INVALID"))


# --------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------
def cmd_info(a) -> int:
    from .inspect import print_frame, print_header, print_meta

    if a.json:
        rep = validate(a.file)
        out = {"file": a.file, "valid": rep.ok, "meta": rep.meta, "rap_frames": rep.rap_frames,
               "layer_stats": {str(k): vars(v) for k, v in rep.layer_stats.items()},
               "issues": [dict(code=i.code, frame=i.frame, severity=i.severity, message=i.message) for i in rep.issues]}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if rep.ok else 1
    with LVFReader(a.file) as r:
        print(f"== {a.file}")
        print_header(r)
        if a.meta:
            print("\n== metadata")
            print_meta(r.meta)
        for n in a.frame or []:
            print()
            print_frame(r, n)
    print("\n== statistics & invariants")
    rep = validate(a.file)
    print_report(rep, verbose=True)
    return 0 if rep.ok else 1


def cmd_check(a) -> int:
    bad = 0
    for path in a.files:
        rep = validate(path)
        if a.quiet:
            print(f"{'ok     ' if rep.ok else 'INVALID'} {path}")
        else:
            print(f"== {path}")
            print_report(rep, verbose=False)
        bad += not rep.ok
    return 1 if bad else 0


def cmd_pack(a) -> int:
    from .project import pack

    rep = pack(a.project, a.output, jobs=a.jobs, keep_temp=a.keep_temp)
    print_report(rep, verbose=False)
    return 0 if rep.ok else 1


def cmd_add(a) -> int:
    from . import edit

    sources = [x for x in (a.src, a.still, a.audio) if x]
    if len(sources) != 1:
        raise CliError("give exactly one of --src (video layer), --still (image layer) or --audio")
    if a.audio:
        rep = edit.set_audio(a.file, a.audio, output=a.output, bitrate=a.bitrate)
        _report_edit(a.output or a.file, rep)
        return 0
    if not a.id:
        raise CliError("--id is required for a new layer")
    fps, n = _file_timing(a.file)
    start = _point(a.start, fps) if a.start else 0
    end = _point(a.end, fps) if a.end else None
    common = dict(output=a.output, rect=a.rect, start=start, end=end, z=a.z, name=a.name, blend=a.blend,
                  opacity=a.opacity, visible=not a.hidden)
    t0 = time.monotonic()
    if a.src:
        rep = edit.add_layer(a.file, a.id, a.src, alpha=a.alpha, lossless=a.lossless, crf=a.crf, speed=a.speed,
                             **common)
    else:
        rep = edit.add_still(a.file, a.id, a.still, **common)
    print(f"added {a.id!r} in {time.monotonic() - t0:.1f} s")
    _report_edit(a.output or a.file, rep)
    return 0


def cmd_rm(a) -> int:
    from . import edit

    if not a.layers and not a.audio:
        raise CliError("nothing to remove: give layer ids/indices and/or --audio")
    out = a.output
    rep = None
    if a.layers:
        rep = edit.remove_layers(a.file, a.layers, output=out)
    if a.audio:
        rep = edit.set_audio(out or a.file, None)
    _report_edit(out or a.file, rep)
    return 0


def cmd_set(a) -> int:
    from . import edit

    fields = {}
    for kv in a.fields:
        if "=" not in kv:
            raise CliError(f"expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        fields[k.strip()] = v.strip()
    if not fields:
        raise CliError(f"nothing to set; editable fields: {', '.join(M.EDITABLE_FIELDS)}")
    t0 = time.monotonic()
    in_place = edit.set_layer(a.file, a.layer, output=a.output, **fields)
    how = "metadata rewritten in place" if in_place else "file rewritten (metadata outgrew its reserved space)"
    print(f"{a.output or a.file}: {how} in {(time.monotonic() - t0) * 1000:.0f} ms")
    return 0


def cmd_render(a) -> int:
    from .render import render

    fps, n = _file_timing(a.file)
    start, end = parse_range(a.frames, fps, n)
    t0 = time.monotonic()
    count = render(a.file, a.output, layers=_layers_arg(a.layers), hide=_layers_arg(a.hide), start=start, end=end,
                   transparent=a.transparent, crf=a.crf, progress=_Progress("render"))
    print(f"wrote {count} frame(s) to {a.output} in {time.monotonic() - t0:.1f} s")
    return 0


def cmd_extract(a) -> int:
    from .render import extract

    fps, n = _file_timing(a.file)
    start, end = parse_range(a.frames, fps, n) if a.frames else (None, None)
    t0 = time.monotonic()
    count = extract(a.file, a.layer, a.output, start=start, end=end, crf=a.crf, progress=_Progress("extract"))
    print(f"wrote {count} frame(s) of {a.layer!r} to {a.output} in {time.monotonic() - t0:.1f} s")
    return 0


def cmd_view(a) -> int:
    from .view import serve

    with LVFReader(a.file):
        pass  # fail early on something that is not an .lvd
    serve(a.file, host=a.host, port=a.port, browser=a.browser, open_page=not a.no_open, quiet=not a.verbose)
    return 0


def cmd_testsrc(a, rest: list[str]) -> int:
    from .devtools import testsrc

    rc = testsrc.main(rest)
    if rc or a.no_pack:
        return rc
    from .project import pack

    out = testsrc.parse_out(rest)
    rep = pack(out / "test_project.json")
    print_report(rep, verbose=False)
    return 0 if rep.ok else 1


def cmd_corrupt(_a, rest: list[str]) -> int:
    from .devtools import corrupt

    return corrupt.main(rest)


# --------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fflv", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"fflv {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    s = sub.add_parser("info", help="show a file and check its invariants")
    s.add_argument("file")
    s.add_argument("--frame", type=int, action="append", help="dump composite frame N (repeatable)")
    s.add_argument("--no-meta", dest="meta", action="store_false", help="do not print the metadata JSON")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("check", help="validate files (exit 1 if any is invalid)")
    s.add_argument("files", nargs="+")
    s.add_argument("-q", "--quiet", action="store_true", help="one line per file")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("pack", help="build a file from a project JSON")
    s.add_argument("project")
    s.add_argument("-o", "--output")
    s.add_argument("-j", "--jobs", type=int, default=4, help="parallel FFmpeg encodes (default 4)")
    s.add_argument("--keep-temp", action="store_true")
    s.set_defaults(fn=cmd_pack)

    s = sub.add_parser("add", help="add a video / still layer or the audio track")
    s.add_argument("file")
    g = s.add_argument_group("what to add (one of)")
    g.add_argument("--src", metavar="MEDIA", help="video layer from any file FFmpeg reads")
    g.add_argument("--still", metavar="IMAGE", help="still layer from an image")
    g.add_argument("--audio", metavar="MEDIA", help="replace the audio track")
    g = s.add_argument_group("layer")
    g.add_argument("--id", help="new layer id")
    g.add_argument("--name")
    g.add_argument("--rect", help="x,y,w,h on the canvas (default: whole canvas / image size)")
    g.add_argument("--start", help="first frame (or seconds with s suffix)")
    g.add_argument("--end", help="end frame, excluded (default: end of file)")
    g.add_argument("--z", type=float, help="draw order (default: on top)")
    g.add_argument("--blend", default="normal", choices=["normal", "add", "multiply", "screen"])
    g.add_argument("--opacity", type=float, default=1.0)
    g.add_argument("--hidden", action="store_true", help="hidden by default in the player")
    g = s.add_argument_group("encoding")
    g.add_argument("--alpha", dest="alpha", action="store_true", default=None, help="keep the alpha channel")
    g.add_argument("--no-alpha", dest="alpha", action="store_false", help="drop it (default: auto)")
    g.add_argument("--lossless", action="store_true", help="bit-exact RGB and alpha (bigger)")
    g.add_argument("--crf", type=int, default=32, help="VP9 quality, lower is better (default 32)")
    g.add_argument("--speed", default="balanced", choices=["fast", "balanced", "best"])
    g.add_argument("--bitrate", default="128k", help="Opus bitrate for --audio")
    s.add_argument("-o", "--output", help="write here instead of editing FILE in place")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("rm", help="remove layers and/or the audio track")
    s.add_argument("file")
    s.add_argument("layers", nargs="*", help="layer ids or indices")
    s.add_argument("--audio", action="store_true", help="remove the audio track")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_rm)

    s = sub.add_parser("set", help="change layer properties (in place, instant)")
    s.add_argument("file")
    s.add_argument("layer")
    s.add_argument("fields", nargs="*", metavar="key=value",
                   help=f"one or more of: {', '.join(M.EDITABLE_FIELDS)} (rect=x,y,w,h, visible=true/false)")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_set)

    s = sub.add_parser("render", help="composite layers to images, a video or .npy")
    s.add_argument("file")
    s.add_argument("-o", "--output", required=True, help="out.png, frames/%%05d.png, dir/, out.mp4/.mov/.webm/.mkv, out.npy")
    s.add_argument("-l", "--layers", help="comma-separated layers to show (default: the file's visible layers)")
    s.add_argument("--hide", help="comma-separated layers to hide")
    s.add_argument("-f", "--frames", help="frame or range, e.g. 120, 100:200, 2s:5s (default: all)")
    s.add_argument("--transparent", action="store_true", help="no background: RGBA output")
    s.add_argument("--crf", type=int, default=18, help="quality of lossy video outputs (default 18)")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("extract", help="one layer's own pixels (RGBA)")
    s.add_argument("file")
    s.add_argument("layer")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("-f", "--frames", help="frame or range (default: where the layer is active)")
    s.add_argument("--crf", type=int, default=18)
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("view", help="open the interactive player (Chrome / Edge)")
    s.add_argument("file")
    s.add_argument("--port", type=int, default=0, help="default: first free port from 8765")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--browser", choices=["chrome", "edge", "chromium", "default"], help="default: Chrome, else Edge")
    s.add_argument("--no-open", action="store_true", help="only print the URL")
    s.add_argument("-v", "--verbose", action="store_true", help="log requests")
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("testsrc", help="(dev) generate the test material and pack test.lvd", add_help=False)
    s.add_argument("--no-pack", action="store_true")
    s.set_defaults(fn=cmd_testsrc, passthrough=True)

    s = sub.add_parser("corrupt", help="(dev) derive broken files from a valid one", add_help=False)
    s.set_defaults(fn=cmd_corrupt, passthrough=True)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    if rest and not getattr(args, "passthrough", False):
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    try:
        if getattr(args, "passthrough", False):
            return args.fn(args, rest)
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except (CliError, M.MetaError, FormatError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"fflv {args.cmd}: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # PackError, EditError, … all carry a readable message
        print(f"fflv {args.cmd}: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
