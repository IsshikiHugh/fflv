"""The `fflv` command line, end to end."""

import json
import shutil

import numpy as np

import fflv
from conftest import run_fflv as fflv_cmd


def test_frame_ranges(packed, tmp_path):
    out = tmp_path / "r.npy"
    for spec, n in [("7", 1), ("10:20", 10), ("110:", 10), (":5", 5), ("1s:2.5s", 45)]:
        fflv_cmd("render", packed["path"], "-f", spec, "-l", "logo", "-o", out)
        assert np.load(out).shape[0] == n, spec
    for bad in ("90:130", "abc", "5:5", "-1"):
        res = fflv_cmd("render", packed["path"], "-f", bad, "-o", out, ok=False)
        assert res.returncode == 2 and "error" in res.stderr, bad


def test_cli_workflow(packed, tmp_path):
    f = tmp_path / "t.lvd"
    shutil.copy(packed["path"], f)

    info = json.loads(fflv_cmd("info", f, "--json").stdout)
    assert info["valid"] and info["rap_frames"] == [0, 30, 60, 90]
    text = fflv_cmd("info", f, "--frame", "45", "--no-meta").stdout
    assert "I1   ok" in text and "composite frame #45" in text and "KEY" in text
    assert "ok" in fflv_cmd("check", "-q", f).stdout

    img = tmp_path / "note.png"
    fflv_cmd("render", packed["path"], "-f", "100", "-l", "logo", "-o", img)  # any PNG will do
    fflv_cmd("add", f, "--still", img, "--id", "note", "--rect", "0,0,64,36", "--start", "1s", "--end", "2s")
    fflv_cmd("add", f, "--src", packed["dir"] / "sync_a.mkv", "--id", "copy", "--start", "15", "--speed", "fast")
    out = fflv_cmd("set", f, "copy", "opacity=0.5", "blend=screen", "visible=false").stdout
    assert "in place" in out
    fflv_cmd("rm", f, "calib", "--audio")
    meta = json.loads(fflv_cmd("info", f, "--json").stdout)["meta"]
    layers = {L["id"]: L for L in meta["layers"]}
    assert "calib" not in layers and meta["audio"] is None
    assert (layers["note"]["start_frame"], layers["note"]["end_frame"]) == (30, 60)
    assert layers["copy"]["has_alpha"] and layers["copy"]["start_frame"] == 15
    assert (layers["copy"]["opacity"], layers["copy"]["blend"], layers["copy"]["visible"]) == (0.5, "screen", False)

    fflv_cmd("render", f, "-f", "40:44", "-l", "bg,copy", "-o", tmp_path / "r.npy")
    assert np.load(tmp_path / "r.npy").shape == (4, 360, 640, 3)
    fflv_cmd("extract", f, "exact", "-f", "3", "-o", tmp_path / "e.npy")
    assert np.load(tmp_path / "e.npy").shape == (1, 96, 256, 4)

    bad = fflv_cmd("set", f, "copy", "start_frame=3", ok=False)
    assert bad.returncode == 2 and "cannot be edited" in bad.stderr
    bad = fflv_cmd("rm", f, "nope", ok=False)
    assert bad.returncode == 2 and "no layer" in bad.stderr
    bad = fflv_cmd("render", f, "-f", "999", "-o", tmp_path / "x.png", ok=False)
    assert bad.returncode == 2
    assert fflv.cli.main(["check", "-q", str(f)]) == 0  # the same command line, in process


def test_rm_audio_to_output_reads_the_source(packed, tmp_path):
    """`rm FILE --audio -o OUT` writes FILE minus its audio to OUT (with or without layers), leaving
    FILE as it was."""
    out = tmp_path / "out.lvd"
    out.write_bytes(b"an unrelated file")
    fflv_cmd("rm", packed["path"], "--audio", "-o", out)
    meta = json.loads(fflv_cmd("info", out, "--json").stdout)["meta"]
    src = json.loads(fflv_cmd("info", packed["path"], "--json").stdout)["meta"]
    assert meta["audio"] is None and src["audio"] is not None
    assert [L["id"] for L in meta["layers"]] == [L["id"] for L in src["layers"]]
    fflv_cmd("rm", packed["path"], "calib", "--audio", "-o", out)
    meta = json.loads(fflv_cmd("info", out, "--json").stdout)["meta"]
    assert meta["audio"] is None and "calib" not in [L["id"] for L in meta["layers"]]
    assert fflv_cmd("rm", packed["path"], ok=False).returncode == 2


def test_broken_files_are_reported_precisely(packed, tmp_path):
    """Acceptance 11.2-2: every deliberately broken variant is reported with the right invariant."""
    res = fflv_cmd("corrupt", packed["path"], "--out", tmp_path, "--check")
    assert "13/13 broken files reported as expected" in res.stdout
    res = fflv_cmd("check", "-q", *sorted(tmp_path.glob("*.lvd")), ok=False)
    assert res.returncode == 1 and res.stdout.count("INVALID") == 13


def test_truncated_and_garbage_files(packed, tmp_path):
    data = packed["path"].read_bytes()
    bad = tmp_path / "truncated.lvd"
    bad.write_bytes(data[: len(data) // 2])
    rep = fflv.validate(bad)
    assert not rep.ok and rep.error_codes() & {"HDR", "CAU", "I1", "I10"}
    bad = tmp_path / "garbage.lvd"
    bad.write_bytes(b"NOPE" + bytes(100))
    rep = fflv.validate(bad)
    assert rep.fatal and "HDR" in rep.error_codes()
    res = fflv_cmd("info", bad, ok=False)
    assert res.returncode == 2 and "error" in res.stderr
