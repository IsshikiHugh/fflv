"""Human-readable output for validation reports (`fflv info`, `fflv check`, `fflv pack`)."""

from __future__ import annotations

from .validate import INVARIANTS, Report


def compress_positions(values: list[int], limit: int = 12) -> str:
    """"0, 60, 120, … 540 (every 60)" for regular series, else the first `limit` values."""
    if not values:
        return "none"
    if len(values) > 2:
        steps = {b - a for a, b in zip(values, values[1:])}
        if len(steps) == 1:
            return f"{values[0]}, {values[1]}, … {values[-1]} (every {steps.pop()})"
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f", … (+{len(values) - limit} more)" if len(values) > limit else "")


def print_stats(rep: Report) -> None:
    meta = rep.meta or {}
    fps = meta.get("fps") or {}
    num, den = fps.get("num", 1), fps.get("den", 1)
    fc = meta.get("frame_count", 0)
    dur = fc * den / num if num else 0
    print(f"file size       {rep.file_size:,} bytes ({rep.file_size / 1e6:.2f} MB)")
    print(f"composite frames {rep.cau_count} (frame_count {fc}, {dur:.3f} s)")
    print(f"RAPs            {len(rep.rap_frames)}: {compress_positions(rep.rap_frames)}")
    layers = meta.get("layers") or []
    for li, st in sorted(rep.layer_stats.items()):
        L = layers[li]
        secs = st.frames * den / num if num else 0
        kbps = lambda b: b * 8 / secs / 1000 if secs else 0.0  # noqa: E731
        alpha = f", alpha {kbps(st.alpha_bytes):8.1f} kbit/s" if L.get("has_alpha") else ""
        print(f"layer {li:<2} {L.get('id', '?'):<12} {st.frames:>6} frames, {st.keyframes:>4} key, "
              f"color {kbps(st.color_bytes):8.1f} kbit/s{alpha}")
    for li, L in enumerate(layers):
        if L.get("kind") == "still":
            r = L.get("resource") or {}
            print(f"layer {li:<2} {L.get('id', '?'):<12} still, {r.get('length', 0):,} byte PNG, "
                  f"frames [{L.get('start_frame')}, {L.get('end_frame')})")
    if meta.get("audio"):
        kbps = rep.audio_bytes * 8 / dur / 1000 if dur else 0
        print(f"audio           {rep.audio_packets} Opus packets, {kbps:.1f} kbit/s")


def print_report(rep: Report, verbose: bool = True) -> None:
    if verbose and rep.meta is not None:
        print_stats(rep)
        print()
    codes = rep.error_codes()
    if rep.fatal:
        print("FATAL: file could not be validated completely")
    for inv in INVARIANTS:
        state = "FAIL" if inv in codes else ("n/a " if rep.fatal else "ok  ")
        print(f"  {inv:<4} {state}")
    other = sorted(c for c in codes if c not in INVARIANTS)
    if other:
        print(f"  other failures: {', '.join(other)}")
    for issue in rep.issues:
        print(f"  {issue.severity.upper():7} {issue}")
    for key, n in rep.suppressed.items():
        print(f"  ... {n} more {key} issues not shown")
    print("VALID" if rep.ok else f"INVALID ({len(rep.errors) + sum(n for k, n in rep.suppressed.items() if k.endswith('/error'))} errors)")
