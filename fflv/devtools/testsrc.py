"""Generate the LVF test material (spec 11.1) and a matching project file.

    fflv testsrc [--out test_assets] [--duration 20]

Sources are rendered with FFmpeg (lavfi `testsrc2` / `aevalsrc`) plus numpy
overlays, and stored losslessly (FFV1 in Matroska; BGRA where alpha matters).
This FFmpeg build has no `drawtext`, so frame numbers are drawn with a small
built-in bitmap font instead.

Every video layer also carries a *barcode* of its global frame number
(1 start cell + 16 data bits + 1 parity bit, 8x8-px cells, white = 1). The
player's end-to-end test reads the barcodes back from the rendered canvas to
prove that all layers on screen come from the same composite frame; their
canvas positions are written to `barcodes.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..format.timing import parse_fps, seconds_to_frame

# 5x7 bitmap font -------------------------------------------------------------
_GLYPHS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    " ": ["00000"] * 7,
}
_MASKS = {c: np.array([[ch == "1" for ch in row] for row in g], dtype=bool) for c, g in _GLYPHS.items()}


def text_size(text: str, scale: int) -> tuple[int, int]:
    return (len(text) * 6 - 1) * scale, 7 * scale


def draw_text(img: np.ndarray, x: int, y: int, text: str, scale: int, color, box=None, pad: int = 0) -> None:
    """Draw `text` with its top-left at (x, y). `color`/`box` match img's channel count."""
    w, h = text_size(text, scale)
    if box is not None:
        img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = box
    for i, ch in enumerate(text):
        m = np.kron(_MASKS[ch], np.ones((scale, scale), dtype=bool))
        gx = x + i * 6 * scale
        region = img[y:y + h, gx:gx + 5 * scale]
        region[m[:region.shape[0], :region.shape[1]]] = color


# Barcode ---------------------------------------------------------------------
BAR_CELL = 8
BAR_BITS = 16
BAR_CELLS = BAR_BITS + 2  # start marker + data + parity


def barcode_cells(n: int) -> list[int]:
    bits = [(n >> (BAR_BITS - 1 - i)) & 1 for i in range(BAR_BITS)]
    return [1] + bits + [sum(bits) & 1]


def draw_barcode(img: np.ndarray, x: int, y: int, n: int, one, zero) -> None:
    for i, bit in enumerate(barcode_cells(n)):
        img[y:y + BAR_CELL, x + i * BAR_CELL:x + (i + 1) * BAR_CELL] = one if bit else zero


# FFmpeg plumbing -------------------------------------------------------------
def ffv1_writer(path: Path, w: int, h: int, fps: Fraction, rgba: bool) -> subprocess.Popen:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgba" if rgba else "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgra" if rgba else "bgr0", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def finish(proc: subprocess.Popen, what: str) -> None:
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed while writing {what}")


def testsrc2_frames(w: int, h: int, fps: Fraction, n: int):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps.numerator}/{fps.denominator}",
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    for _ in range(n):
        buf = proc.stdout.read(size)
        if len(buf) != size:
            raise SystemExit("testsrc2 ended early")
        yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
    proc.stdout.close()
    proc.wait()


# Layer renderers ---------------------------------------------------------------
WHITE4, BLACK4 = (255, 255, 255, 255), (0, 0, 0, 255)


def make_background(path: Path, W: int, H: int, fps: Fraction, N: int, fps_int: int, flash_frames: int) -> dict:
    proc = ffv1_writer(path, W, H, fps, rgba=False)
    digits = (24, 24)  # clear of the 14-px flash border
    bar = (24, 24 + 7 * 6 + 14)
    for n, img in enumerate(testsrc2_frames(W, H, fps, N)):
        draw_text(img, digits[0], digits[1], f"F{n:05d}", 6, (255, 255, 255), box=(0, 0, 0), pad=6)
        draw_barcode(img, bar[0], bar[1], n, (255, 255, 255), (0, 0, 0))
        if n % fps_int < flash_frames:  # white frame around the picture, together with the beep
            t = 14
            img[:t] = 255
            img[-t:] = 255
            img[:, :t] = 255
            img[:, -t:] = 255
        proc.stdin.write(img.tobytes())
    finish(proc, path.name)
    return {"x": bar[0], "y": bar[1]}


def make_sync_layer(path: Path, W: int, H: int, fps: Fraction, N: int, which: str) -> dict:
    """Layer A: wide white line, B: narrow red line at the same x = (n*16) mod W."""
    proc = ffv1_writer(path, W, H, fps, rgba=True)
    line_h = H - 60
    lw, color = (12, (255, 255, 255, 255)) if which == "A" else (4, (255, 0, 0, 255))
    tw, _ = text_size(f"{which} 00000", 4)
    digits_x = 16 if which == "A" else W - 16 - tw
    bar_x = 16 if which == "A" else W - 16 - BAR_CELLS * BAR_CELL
    for n in range(N):
        img = np.zeros((H, W, 4), dtype=np.uint8)
        cx = (n * 16) % W
        img[0:line_h, max(0, cx - lw // 2):min(W, cx + lw // 2)] = color
        draw_text(img, digits_x, line_h + 6, f"{which} {n:05d}", 4, WHITE4, box=BLACK4, pad=3)
        draw_barcode(img, bar_x, H - BAR_CELL - 4, n, WHITE4, BLACK4)
        proc.stdin.write(img.tobytes())
    finish(proc, path.name)
    return {"x": bar_x, "y": H - BAR_CELL - 4, "line_rows": [0, line_h], "line_width": lw}


def make_square_layer(path: Path, W: int, H: int, fps: Fraction, start: int, end: int) -> dict:
    """A moving square that exists only in [start, end); its numbers are global frame numbers."""
    proc = ffv1_writer(path, W, H, fps, rgba=True)
    side = 100
    bar = (16, H - BAR_CELL - 4)
    for n in range(start, end):
        img = np.zeros((H, W, 4), dtype=np.uint8)
        x = ((n - start) * 8) % (W - side)
        img[0:side, x:x + side] = (40, 170, 255, 230)
        draw_text(img, x + 5, 36, f"{n:05d}", 3, WHITE4)
        draw_barcode(img, bar[0], bar[1], n, WHITE4, BLACK4)
        proc.stdin.write(img.tobytes())
    finish(proc, path.name)
    return {"x": bar[0], "y": bar[1]}


def make_calibration_layer(path: Path, W: int, H: int, fps: Fraction, N: int) -> dict:
    """White with alpha ramping 0 → 255 left to right (top 40 rows), plus a barcode strip."""
    proc = ffv1_writer(path, W, H, fps, rgba=True)
    ramp_rows = 40
    base = np.zeros((H, W, 4), dtype=np.uint8)
    base[:ramp_rows, :, :3] = 255
    base[:ramp_rows, :, 3] = np.round(np.arange(W) * 255.0 / (W - 1)).astype(np.uint8)[None, :]
    bar = (W // 2 - BAR_CELLS * BAR_CELL // 2, ramp_rows + 8)
    for n in range(N):
        img = base.copy()
        draw_barcode(img, bar[0], bar[1], n, WHITE4, BLACK4)
        proc.stdin.write(img.tobytes())
    finish(proc, path.name)
    return {"x": bar[0], "y": bar[1], "ramp_rows": [0, ramp_rows]}


def exact_pattern(f: int, w: int = 256, h: int = 64) -> np.ndarray:
    """Per-pixel RGB of the lossless test layer; player/e2e computes the same values."""
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x * 7 + f) & 255, (y * 13 + 3 * f) & 255, (x ^ y ^ f) & 255], -1).astype(np.uint8)


def make_exact_layer(path: Path, fps: Fraction, N: int) -> dict:
    """Lossless layer (256×96): rows 0..63 an RGB pattern that changes every frame (alpha 255),
    rows 64..79 white with alpha = x (0..255), then a barcode; everything else transparent."""
    W, H = 256, 96
    proc = ffv1_writer(path, W, H, fps, rgba=True)
    bar = (0, 84)
    for n in range(N):
        img = np.zeros((H, W, 4), dtype=np.uint8)
        img[0:64, :, :3] = exact_pattern(n)
        img[0:64, :, 3] = 255
        img[64:80, :, :3] = 255
        img[64:80, :, 3] = np.arange(W, dtype=np.uint8)[None, :]
        draw_barcode(img, bar[0], bar[1], n, WHITE4, BLACK4)
        proc.stdin.write(img.tobytes())
    finish(proc, path.name)
    return {"x": bar[0], "y": bar[1], "pattern_rows": [0, 64], "ramp_rows": [64, 80]}


def make_logo(path: Path, w: int, h: int) -> None:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = (230, 60, 120, 200)
    b = 5
    img[:b], img[-b:], img[:, :b], img[:, -b:] = WHITE4, WHITE4, WHITE4, WHITE4
    tw, th = text_size("LVF", 8)
    draw_text(img, (w - tw) // 2, (h - th) // 2, "LVF", 8, WHITE4)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-i", "-",
           "-frames:v", "1", "-c:v", "png", "-pix_fmt", "rgba", str(path)]
    subprocess.run(cmd, input=img.tobytes(), check=True)


def make_beeps(path: Path, duration: float, beep_s: float) -> None:
    # 1 kHz tone during the first `beep_s` of every second (commas escaped for the lavfi parser).
    e = f"if(lt(mod(t\\,1)\\,{beep_s:.6f})\\,0.6*sin(2*PI*1000*t)\\,0)"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"aevalsrc={e}|{e}:s=48000:d={duration}",
           "-c:a", "pcm_s16le", str(path)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fflv testsrc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("test_assets"), help="output directory (default: ./test_assets)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", default="30/1")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds")
    ap.add_argument("--gop", type=int, default=60)
    ap.add_argument("--crf", type=int, default=32)
    args = ap.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    W, H = args.width, args.height
    fps = parse_fps(args.fps)
    if fps.denominator != 1:
        raise SystemExit("the beep/flash pattern needs an integer frame rate")
    fps_int = fps.numerator
    N = seconds_to_frame(args.duration, fps)
    flash = 2  # frames of white border per second; the beep lasts exactly as long
    beep_s = flash / fps_int

    # canvas layout
    band = (0, 250, W, 160)
    square = (0, 430, W, 150)
    calib = (0, H - 104, W, 64)
    logo = (W - 240, 20, 220, 100)
    exact = (420, 110, 256, 96)
    sq_start, sq_end = 45, min(N, N - 45) if N > 90 else N  # frame 45: not on the GOP grid
    logo_start, logo_end = min(N - 1, 3 * fps_int), min(N, 12 * fps_int)

    print(f"rendering {N} frames at {W}x{H} into {out} ...")
    bars = {}
    print("  bg.mkv")
    b = make_background(out / "bg.mkv", W, H, fps, N, fps_int, flash)
    bars["bg"] = {"x": b["x"], "y": b["y"], "start_frame": 0, "end_frame": N}
    for which in ("A", "B"):
        print(f"  sync_{which.lower()}.mkv")
        b = make_sync_layer(out / f"sync_{which.lower()}.mkv", band[2], band[3], fps, N, which)
        bars[f"sync_{which.lower()}"] = {"x": band[0] + b["x"], "y": band[1] + b["y"], "start_frame": 0,
                                         "end_frame": N}
    print("  square.mkv")
    b = make_square_layer(out / "square.mkv", square[2], square[3], fps, sq_start, sq_end)
    bars["square"] = {"x": square[0] + b["x"], "y": square[1] + b["y"], "start_frame": sq_start,
                      "end_frame": sq_end}
    print("  calib.mkv")
    b = make_calibration_layer(out / "calib.mkv", calib[2], calib[3], fps, N)
    bars["calib"] = {"x": calib[0] + b["x"], "y": calib[1] + b["y"], "start_frame": 0, "end_frame": N}
    print("  exact.mkv (lossless)")
    ex = make_exact_layer(out / "exact.mkv", fps, N)
    bars["exact"] = {"x": exact[0] + ex["x"], "y": exact[1] + ex["y"], "start_frame": 0, "end_frame": N}
    print("  logo.png, beeps.wav")
    make_logo(out / "logo.png", logo[2], logo[3])
    make_beeps(out / "beeps.wav", N / fps_int, beep_s)

    def secs(frame: int) -> float:
        return frame / fps_int

    project = {
        "output": "test.lvd",
        "canvas": {"width": W, "height": H, "background": "#000000"},
        "fps": f"{fps.numerator}/{fps.denominator}",
        "duration": N / fps_int,
        "gop": args.gop,
        "quality": {"crf": args.crf},
        "layers": [
            {"id": "bg", "name": "background (testsrc2)", "kind": "video", "src": "bg.mkv", "z": 0,
             "rect": [0, 0, W, H], "alpha": False},
            {"id": "sync_a", "name": "sync A (white line)", "kind": "video", "src": "sync_a.mkv", "z": 1,
             "rect": list(band), "alpha": True},
            {"id": "sync_b", "name": "sync B (red line)", "kind": "video", "src": "sync_b.mkv", "z": 2,
             "rect": list(band), "alpha": True},
            {"id": "square", "name": "square (from frame 45)", "kind": "video", "src": "square.mkv", "z": 3,
             "rect": list(square), "start": secs(sq_start), "end": secs(sq_end), "alpha": True},
            {"id": "calib", "name": "alpha calibration", "kind": "video", "src": "calib.mkv", "z": 4,
             "rect": list(calib), "alpha": True},
            {"id": "logo", "name": "logo (still)", "kind": "still", "src": "logo.png", "z": 5,
             "rect": list(logo), "start": secs(logo_start), "end": secs(logo_end)},
            {"id": "exact", "name": "lossless pattern", "kind": "video", "src": "exact.mkv", "z": 6,
             "rect": list(exact), "alpha": True, "lossless": True},
        ],
        "audio": {"src": "beeps.wav"},
    }
    (out / "test_project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")

    probes = {
        "canvas": [W, H], "fps": fps_int, "frame_count": N, "gop": args.gop,
        "barcode": {"cell": BAR_CELL, "bits": BAR_BITS, "cells": BAR_CELLS, "layers": bars},
        "sync_band": {"rect": list(band), "line_rows": [0, band[3] - 60]},
        "calib": {"rect": list(calib), "ramp_rows": [0, 40]},
        "logo": {"rect": list(logo), "start_frame": logo_start, "end_frame": logo_end},
        "exact": {"rect": list(exact), "pattern_rows": ex["pattern_rows"], "ramp_rows": ex["ramp_rows"]},
        "flash": {"frames_per_second": flash, "border": 14},
        "beep": {"hz": 1000, "seconds": beep_s},
    }
    (out / "barcodes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    print(f"wrote {out / 'test_project.json'} and {out / 'barcodes.json'}")
    return 0





def parse_out(argv: list[str] | None) -> Path:
    """The --out directory main() would use for these arguments."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", type=Path, default=Path("test_assets"))
    return ap.parse_known_args(argv)[0].out
