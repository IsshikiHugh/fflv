"""render / extract outputs."""

import json
import subprocess

import numpy as np
import pytest

import fflv
from conftest import read_png, read_video

W, H, N = 96, 64, 12


@pytest.fixture
def src(tmp_path):
    path = tmp_path / "r.lvd"
    with fflv.Writer(path, (W, H), gop=4, background="#102030") as w:
        w.add_layer("bg", lossless=True)
        w.add_layer("dot", alpha=True, lossless=True, rect=(10, 10, 16, 16))
        w.add_layer("hidden", visible=False, rect=(0, 0, 8, 8))
        for f in range(N):
            bg = np.zeros((H, W, 3), np.uint8)
            bg[:, :, 0] = f * 10
            dot = np.zeros((16, 16, 4), np.uint8)
            dot[4:12, 4:12] = (255, 255, 255, 255)
            w.write(bg=bg, dot=dot, hidden=np.full((8, 8, 3), 255, np.uint8))
    return path


def test_render_single_frame_png(src, tmp_path):
    out = tmp_path / "f.png"
    assert fflv.render(src, out, start=5, end=6) == 1
    img = read_png(out)
    assert img.shape == (H, W, 4)
    assert tuple(img[0, 0]) == (50, 0, 0, 255)          # bg of frame 5, exact (lossless)
    assert tuple(img[18, 18]) == (255, 255, 255, 255)   # the dot
    assert tuple(img[2, 2]) == (50, 0, 0, 255)          # "hidden" layer is not shown by default


def test_render_layer_selection(src, tmp_path):
    fflv.render(src, tmp_path / "a.png", start=3, end=4, layers=["hidden"])
    img = read_png(tmp_path / "a.png")
    assert tuple(img[2, 2, :3]) == (255, 255, 255)
    assert tuple(img[30, 30, :3]) == (0x10, 0x20, 0x30)  # canvas background only
    with pytest.raises(fflv.OutputError, match="one frame"):
        fflv.render(src, tmp_path / "b.png", start=0, end=2)


def test_render_sequences_and_videos(src, tmp_path):
    assert fflv.render(src, tmp_path / "seq" / "%03d.png", start=2, end=6) == 4
    assert sorted(p.name for p in (tmp_path / "seq").iterdir()) == ["002.png", "003.png", "004.png", "005.png"]
    assert fflv.render(src, str(tmp_path / "dir") + "/", start=0, end=2) == 2
    assert len(list((tmp_path / "dir").iterdir())) == 2
    fflv.render(src, tmp_path / "v.mkv")  # FFV1: lossless
    frames = read_video(tmp_path / "v.mkv")
    assert len(frames) == N and all(tuple(fr[0, 0]) == (f * 10, 0, 0) for f, fr in enumerate(frames))
    fflv.render(src, tmp_path / "v.mp4")
    assert len(read_video(tmp_path / "v.mp4")) == N
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(tmp_path / "v.mp4")],
                           capture_output=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    assert (st["color_space"], st["color_range"], st["pix_fmt"]) == ("bt709", "tv", "yuv420p")
    fflv.render(src, tmp_path / "v.npy", start=1, end=4)
    arr = np.load(tmp_path / "v.npy")
    assert arr.shape == (3, H, W, 3) and arr[0, 0, 0, 0] == 10
    fflv.render(src, tmp_path / "t.webm", transparent=True, layers=["dot"], end=3)
    with pytest.raises(fflv.OutputError, match="alpha"):
        fflv.render(src, tmp_path / "t.mp4", transparent=True)
    with pytest.raises(fflv.OutputError, match="how to write"):
        fflv.render(src, tmp_path / "t.xyz")


def test_render_progress_and_jpeg(src, tmp_path):
    calls = []
    assert fflv.render(src, tmp_path / "j" / "%02d.jpg", start=0, end=3, progress=lambda d, t: calls.append((d, t))) == 3
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert sorted(p.name for p in (tmp_path / "j").iterdir()) == ["00.jpg", "01.jpg", "02.jpg"]


def test_extract_layer_exact(src, tmp_path):
    assert fflv.extract(src, "dot", tmp_path / "dot.npy") == N
    arr = np.load(tmp_path / "dot.npy")
    assert arr.shape == (N, 16, 16, 4)
    assert (arr[:, 4:12, 4:12] == 255).all() and (arr[:, :4] == 0).all()
    assert fflv.extract(src, "dot", tmp_path / "dot" / "%02d.png", start=10) == 2
    assert np.array_equal(read_png(tmp_path / "dot" / "11.png"), arr[11])


def test_a_raising_progress_callback_stops_the_render(src, tmp_path):
    seen = []

    def progress(done, total):
        seen.append(done)
        if done == 2:
            raise KeyError("enough")

    with pytest.raises(KeyError, match="enough"):
        fflv.render(src, tmp_path / "p" / "%02d.png", progress=progress)
    assert seen == [1, 2]
    assert len(list((tmp_path / "p").iterdir())) <= 2
    with pytest.raises(fflv.MetaError, match="non-negative"):
        fflv.render(src, tmp_path / "x.png", start=-1)
    with fflv.open(src) as r:
        with pytest.raises(fflv.MetaError):
            r.frame(-1)


def test_frame_by_frame_matches_frames(src):
    """frame(i) reuses the decoding of frame(i - 1) (same selection); any order gives the same pixels."""
    with fflv.open(src) as r:
        want = dict(r.frames())
        for order in (range(N), [5, 6, 7, 2, 3, 11, 0, 1]):
            for i in order:
                assert np.array_equal(r.frame(i), want[i]), i
        assert r.frame(3, transparent=True).shape == (H, W, 4) and r.frame(4).shape == (H, W, 3)
        assert r.frame(5, layers=["hidden"])[2, 2, 0] == 255 and r.frame(6)[2, 2, 0] == 60
        with pytest.raises(fflv.DecodeError, match="outside"):
            r.frame(N)
