"""Check an installed fflv (a wheel on a clean machine): write, read back, validate, and run the
command line. No FFmpeg, no build tools, nothing from the source tree."""

import os
import shutil
import subprocess
import tempfile

import numpy as np

import fflv

W, H, N = 64, 48, 12
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "smoke.lvd")
    with fflv.Writer(path, size=(W, H), fps=10) as w:
        w.add_layer("frame")
        w.add_layer("mask", alpha=True, lossless=True)
        for i in range(N):
            mask = np.zeros((H, W, 4), np.uint8)
            mask[:, : W // 2] = (255, 0, 0, 255)
            w.write(frame=np.full((H, W, 3), i * 20, np.uint8), mask=mask)
    with fflv.open(path) as f:
        rgb = f.frame(5, layers=["frame", "mask"])
    assert rgb.shape[:2] == (H, W), rgb.shape
    assert tuple(rgb[0, 0, :3]) == (255, 0, 0), rgb[0, 0]  # the lossless mask covers the left half
    assert fflv.validate(path).ok
    exe = shutil.which("fflv")
    assert exe, "the fflv command is not on PATH"
    subprocess.run([exe, "check", path], check=True)
    subprocess.run([exe, "info", path], check=True, stdout=subprocess.DEVNULL)
print(f"fflv {fflv.__version__}: write, read, validate and the command line work")
