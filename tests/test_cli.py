"""The `fflv` command line, end to end."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from fflv.cli import CliError, parse_range


def fflv(*args, ok=True):
    res = subprocess.run([sys.executable, "-m", "fflv", *map(str, args)], capture_output=True, text=True)
    if ok:
        assert res.returncode == 0, res.stderr + res.stdout
    return res


def test_parse_range():
    from fractions import Fraction

    fps = Fraction(30)
    assert parse_range(None, fps, 100) == (0, 100)
    assert parse_range("7", fps, 100) == (7, 8)
    assert parse_range("10:20", fps, 100) == (10, 20)
    assert parse_range("10:", fps, 100) == (10, 100)
    assert parse_range(":5", fps, 100) == (0, 5)
    assert parse_range("1s:2.5s", fps, 100) == (30, 75)
    with pytest.raises(CliError):
        parse_range("90:120", fps, 100)
    with pytest.raises(CliError):
        parse_range("abc", fps, 100)


def test_cli_workflow(packed, tmp_path):
    f = tmp_path / "t.lvd"
    shutil.copy(packed["path"], f)

    info = json.loads(fflv("info", f, "--json").stdout)
    assert info["valid"] and info["rap_frames"] == [0, 30, 60, 90]
    assert "I1   ok" in fflv("info", f, "--frame", "45", "--no-meta").stdout
    assert "ok" in fflv("check", "-q", f).stdout

    img = tmp_path / "note.png"
    fflv("render", packed["path"], "-f", "0", "-l", "logo", "-o", img)  # any PNG will do
    fflv("add", f, "--still", img, "--id", "note", "--rect", "0,0,64,36", "--start", "1s", "--end", "2s")
    fflv("add", f, "--src", packed["dir"] / "sync_a.mkv", "--id", "copy", "--start", "15", "--speed", "fast")
    out = fflv("set", f, "copy", "opacity=0.5", "blend=screen", "visible=false").stdout
    assert "in place" in out
    fflv("rm", f, "calib", "--audio")
    meta = json.loads(fflv("info", f, "--json").stdout)["meta"]
    layers = {L["id"]: L for L in meta["layers"]}
    assert "calib" not in layers and meta["audio"] is None
    assert (layers["note"]["start_frame"], layers["note"]["end_frame"]) == (30, 60)
    assert layers["copy"]["has_alpha"] and layers["copy"]["start_frame"] == 15
    assert (layers["copy"]["opacity"], layers["copy"]["blend"], layers["copy"]["visible"]) == (0.5, "screen", False)

    fflv("render", f, "-f", "40:44", "-l", "bg,copy", "-o", tmp_path / "r.npy")
    assert np.load(tmp_path / "r.npy").shape == (4, 360, 640, 3)
    fflv("extract", f, "exact", "-f", "3", "-o", tmp_path / "e.npy")
    assert np.load(tmp_path / "e.npy").shape == (1, 96, 256, 4)

    bad = fflv("set", f, "copy", "start_frame=3", ok=False)
    assert bad.returncode == 2 and "cannot be edited" in bad.stderr
    bad = fflv("rm", f, "nope", ok=False)
    assert bad.returncode == 2 and "no layer" in bad.stderr
    bad = fflv("render", f, "-f", "999", "-o", tmp_path / "x.png", ok=False)
    assert bad.returncode == 2


def test_corrupt_devtool(packed, tmp_path):
    res = fflv("corrupt", packed["path"], "--out", tmp_path, "--check")
    assert "13/13 broken files reported as expected" in res.stdout
