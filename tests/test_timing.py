from fractions import Fraction

import pytest

from fflv.format.timing import parse_fps, pts_us, round_half_up, seconds_to_frame


def test_pts_us_integer_fps():
    assert [pts_us(f, 30, 1) for f in range(4)] == [0, 33333, 66667, 100000]
    assert pts_us(3600, 30, 1) == 120_000_000


def test_pts_us_ntsc():
    # 1001/30000 s per frame = 33366.666… us
    assert pts_us(1, 30000, 1001) == 33367
    assert pts_us(2, 30000, 1001) == 66733
    assert pts_us(30000, 30000, 1001) == 1_001_000_000


def test_pts_us_rounds_half_up():
    # 128 fps: odd frames land exactly on .5 us; must round up like JS Math.round.
    assert pts_us(1, 128, 1) == 7813
    assert pts_us(3, 128, 1) == 23438


def test_round_half_up():
    assert round_half_up(Fraction(5, 2)) == 3
    assert round_half_up(Fraction(7, 2)) == 4
    assert round_half_up(Fraction(-1, 2)) == 0


@pytest.mark.parametrize("value,expected", [
    ("30/1", Fraction(30)), ("30000/1001", Fraction(30000, 1001)), (25, Fraction(25)),
    ("29.97", Fraction(2997, 100)), ({"num": 24, "den": 1}, Fraction(24)),
])
def test_parse_fps(value, expected):
    assert parse_fps(value) == expected


def test_parse_fps_rejects_zero():
    with pytest.raises(ValueError):
        parse_fps("0/1")


def test_seconds_to_frame_is_exact_for_decimals():
    assert seconds_to_frame(3.0, Fraction(30)) == 90
    assert seconds_to_frame(1.5, Fraction(30)) == 45
    assert seconds_to_frame(0.1, Fraction(30)) == 3
    assert seconds_to_frame(120.0, Fraction(30000, 1001)) == 3596
