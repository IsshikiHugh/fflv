"""Metadata must be standard JSON (RFC 8259) that every reader — including browsers — can parse."""

import json
import math
import shutil

import numpy as np
import pytest

import fflv
from fflv.format import LVFReader, rewrite_meta_in_place, validate
from fflv.meta import MetaError, check_rect, check_z


@pytest.fixture
def small(packed, tmp_path):
    path = tmp_path / "s.lvd"
    shutil.copy(packed["path"], path)
    return path


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", float("nan"), float("inf"), "abc", True, None])
def test_z_must_be_a_finite_number(bad):
    with pytest.raises(MetaError, match="z must be"):
        check_z(bad)


def test_numbers_are_normalised_to_plain_json_types():
    assert check_z(np.int64(3)) == 3 and type(check_z(np.int64(3))) is int
    assert check_z(np.float32(1.5)) == 1.5 and type(check_z(np.float32(1.5))) is float
    assert check_z("-2") == -2 and check_z(" 2.5 ") == 2.5
    assert check_rect(np.array([1, 2, 30, 40])) == {"x": 1, "y": 2, "w": 30, "h": 40}
    assert all(type(v) is int for v in check_rect(tuple(np.int32(v) for v in (1, 2, 3, 4))).values())


def test_set_rejects_non_finite_z(small):
    for bad in ("nan", "inf", "1e999"):
        with pytest.raises(MetaError, match="finite"):
            fflv.set_layer(small, "bg", z=bad)
    assert validate(str(small)).ok


def test_writer_rejects_nan_and_accepts_numpy_scalars(tmp_path):
    with pytest.raises(MetaError):
        fflv.Writer(tmp_path / "a.lvd", (16, 16)).add_layer("a", z=float("nan"))
    path = tmp_path / "b.lvd"
    with fflv.Writer(path, (16, 16), gop=np.int64(4)) as w:
        w.add_layer("a", z=np.int64(3), rect=tuple(np.int64(v) for v in (0, 0, 16, 16)), opacity=np.float32(0.5))
        w.write(a=np.zeros((16, 16, 3), np.uint8))
    with LVFReader(path) as r:
        L = r.meta["layers"][0]
    assert (L["z"], L["rect"], L["opacity"]) == (3, {"x": 0, "y": 0, "w": 16, "h": 16}, 0.5)


@pytest.mark.parametrize("raw, why", [
    (b'{"format": "LVF", "z": NaN}', "NaN"),
    (b'{"format": "LVF", "z": Infinity}', "Infinity"),
    (b'{"format": "LVF", "z": 1e999}', "not finite"),
])
def test_validator_rejects_non_standard_json(small, raw, why):
    meta = json.loads(LVFReader(small).meta_bytes())
    meta["layers"][0]["z"] = 12345.0
    text = json.dumps(meta).encode().replace(b"12345.0", raw.split(b'"z": ')[1][:-1])
    assert rewrite_meta_in_place(small, text)
    rep = validate(str(small))
    assert not rep.ok and "META" in rep.error_codes()
    assert why in str(rep.errors[0])


def test_validator_rejects_a_byte_order_mark(small):
    raw = LVFReader(small).meta_bytes()
    assert rewrite_meta_in_place(small, b"\xef\xbb\xbf" + raw)
    rep = validate(str(small))
    assert not rep.ok and "byte-order mark" in str(rep.errors[0])


def test_encode_meta_refuses_nan():
    from fflv.format import encode_meta

    with pytest.raises(ValueError):
        encode_meta({"z": math.nan})
    assert b"3" in encode_meta({"z": np.int64(3)})
