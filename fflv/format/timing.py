"""Frame/time arithmetic. All rounding is exact integer math, round-half-up.

The player (player/src/format/timing.ts) implements the same formulas; the two
must agree bit-for-bit because pts_us(f) is what ties decoder output back to
its composite frame.
"""

from __future__ import annotations

from fractions import Fraction


def round_half_up(x: Fraction) -> int:
    return (x.numerator * 2 + x.denominator) // (x.denominator * 2)


def pts_us(frame: int, fps_num: int, fps_den: int) -> int:
    """pts_us(f) = round(f * 1_000_000 * den / num)."""
    return (2 * frame * 1_000_000 * fps_den + fps_num) // (2 * fps_num)


def seconds_to_frame(seconds: float | int | str | Fraction, fps: Fraction) -> int:
    """frame = round(seconds * fps). Seconds are taken at their decimal value
    (0.1 means exactly 1/10), so `3.0` at 30 fps is exactly frame 90."""
    if not isinstance(seconds, Fraction):
        seconds = Fraction(str(seconds))
    return round_half_up(seconds * fps)


def parse_fps(value) -> Fraction:
    """Accepts "30/1", "30000/1001", "29.97", 30, 30.0 or {"num":..,"den":..}."""
    if isinstance(value, dict):
        fps = Fraction(int(value["num"]), int(value["den"]))
    elif isinstance(value, (int, float)):
        fps = Fraction(str(value))
    elif isinstance(value, str):
        fps = Fraction(value.strip())
    else:
        raise ValueError(f"cannot parse fps {value!r}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {value!r}")
    return fps
