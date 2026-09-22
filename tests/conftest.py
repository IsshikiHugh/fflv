import json
import subprocess
import sys

import numpy as np
import pytest

import fflv
from fflv import _fflv


def run_fflv(*args, ok=True):
    """The `fflv` command line (as installed with the package)."""
    res = subprocess.run([sys.executable, "-m", "fflv", *map(str, args)], capture_output=True, text=True)
    if ok:
        assert res.returncode == 0, res.stderr + res.stdout
    return res


def read_png(path) -> np.ndarray:
    return _fflv.decode_png(open(path, "rb").read())


def read_video(path) -> list[np.ndarray]:
    """Every frame of a video file as RGB, decoded by FFmpeg."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "json", str(path)], capture_output=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    w, h = st["width"], st["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3))


@pytest.fixture(scope="session")
def packed(tmp_path_factory):
    """A small but complete test file: 640x360, 4 s @ 30 fps, gop 30, all layer kinds, audio."""
    out = tmp_path_factory.mktemp("assets")
    run_fflv("testsrc", "--out", out, "--width", 640, "--height", 360, "--duration", 4, "--gop", 30, "--no-pack")
    rep = fflv.pack(out / "test_project.json", out / "small.lvd", log=None)
    assert rep.ok, [str(i) for i in rep.issues]
    return {
        "dir": out,
        "path": out / "small.lvd",
        "probes": json.loads((out / "barcodes.json").read_text()),
        "project": json.loads((out / "test_project.json").read_text()),
    }
