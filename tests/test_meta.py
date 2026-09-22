"""Arguments are checked before anything is written; metadata stays standard JSON (RFC 8259)."""

from fractions import Fraction

import numpy as np
import pytest

import fflv
from fflv import _util


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", float("nan"), float("inf"), "abc", True])
def test_z_must_be_a_finite_number(tmp_path, bad):
    with pytest.raises(fflv.MetaError, match="z must be"):
        fflv.Writer(tmp_path / "a.lvd", (16, 16)).add_layer("a", z=bad)


def test_numpy_scalars_become_plain_json_numbers(tmp_path):
    path = tmp_path / "b.lvd"
    with fflv.Writer(path, (16, 16), gop=np.int64(4)) as w:
        w.add_layer("a", z=np.int64(3), rect=tuple(np.int64(v) for v in (0, 0, 16, 16)), opacity=np.float32(0.5))
        w.add_still("s", np.zeros((2, 2, 3), np.uint8), z=np.float32(1.5), rect=np.array([1, 2, 2, 2]))
        w.write(a=np.zeros((16, 16, 3), np.uint8))
    with fflv.open(path) as r:
        L, S = r.meta["layers"]
    assert (L["z"], L["rect"], L["opacity"]) == (3, {"x": 0, "y": 0, "w": 16, "h": 16}, 0.5)
    assert type(L["z"]) is int and (S["z"], S["rect"]["x"]) == (1.5, 1)
    assert r.meta["max_rap_interval"] == 4


def test_rects():
    assert _util.rect("1,2,30,40") == (1, 2, 30, 40)
    assert _util.rect({"x": -1, "y": 2, "w": 3, "h": 4}) == (-1, 2, 3, 4)
    assert _util.rect(np.array([1, 2, 30, 40])) == (1, 2, 30, 40)
    for bad in ([1, 2, 0, 4], [1, 2, 3.5, 4], "1,2,3", [True, 1, 2, 3], 5):
        with pytest.raises(fflv.MetaError, match="rect"):
            _util.rect(bad)


def test_frame_rates():
    assert _util.fps(30) == 30 and _util.fps("30000/1001").denominator == 1001
    assert _util.fps(29.97) == _util.fps("29.97") == Fraction(2997, 100)
    for bad in (0, -1, "abc", True, None):
        with pytest.raises(fflv.MetaError):
            _util.fps(bad)


def test_layer_options_are_checked(tmp_path):
    w = fflv.Writer(tmp_path / "c.lvd", (16, 16))
    for kw, msg in [({"blend": "overlay"}, "blend"), ({"opacity": 1.5}, "opacity"), ({"speed": "warp"}, "speed"),
                    ({"crf": 64}, "crf"), ({"rect": (0, 0, 0, 4)}, "rect")]:
        with pytest.raises(fflv.MetaError, match=msg):
            w.add_layer("a", **kw)
    for bad_id in ("12", "-x", "", "a b"):
        with pytest.raises(fflv.MetaError, match="layer id"):
            w.add_layer(bad_id)
    with pytest.raises(fflv.MetaError, match="background"):
        fflv.Writer(tmp_path / "d.lvd", (16, 16), background="red")
    with pytest.raises(fflv.MediaError, match="not a PNG"):
        w.add_still("s", b"GIF89a")
