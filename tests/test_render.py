"""render / extract outputs, and decoding only the selected layers."""

import av
import numpy as np
import pytest

import fflv
from fflv.render import OutputError

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


def read_video(path):
    with av.open(str(path)) as c:
        return [fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)]


def test_render_single_frame_png(src, tmp_path):
    out = tmp_path / "f.png"
    assert fflv.render(src, out, start=5, end=6) == 1
    img = read_video(out)[0]
    assert img.shape == (H, W, 3)
    assert tuple(img[0, 0]) == (50, 0, 0)          # bg of frame 5, exact (lossless)
    assert tuple(img[18, 18]) == (255, 255, 255)   # the dot
    assert tuple(img[2, 2]) == (50, 0, 0)          # "hidden" layer is not shown by default


def test_render_layer_selection(src, tmp_path):
    fflv.render(src, tmp_path / "a.png", start=3, end=4, layers=["hidden"])
    img = read_video(tmp_path / "a.png")[0]
    assert tuple(img[2, 2]) == (255, 255, 255)
    assert tuple(img[30, 30]) == (0x10, 0x20, 0x30)  # canvas background only
    with pytest.raises(OutputError, match="one frame"):
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
    fflv.render(src, tmp_path / "v.npy", start=1, end=4)
    arr = np.load(tmp_path / "v.npy")
    assert arr.shape == (3, H, W, 3) and arr[0, 0, 0, 0] == 10
    fflv.render(src, tmp_path / "t.webm", transparent=True, layers=["dot"], end=3)
    with pytest.raises(OutputError, match="alpha"):
        fflv.render(src, tmp_path / "t.mp4", transparent=True)


def test_extract_layer_exact(src, tmp_path):
    assert fflv.extract(src, "dot", tmp_path / "dot.npy") == N
    arr = np.load(tmp_path / "dot.npy")
    assert arr.shape == (N, 16, 16, 4)
    assert (arr[:, 4:12, 4:12] == 255).all() and (arr[:, :4] == 0).all()
    assert fflv.extract(src, "dot", tmp_path / "dot" / "%02d.png", start=10) == 2


def test_only_selected_layers_are_decoded(src, monkeypatch):
    created = []
    real = av.CodecContext.create

    def spy(name, mode="r"):
        if mode == "r" and name == "vp9":
            created.append(name)
        return real(name, mode)

    monkeypatch.setattr(av.CodecContext, "create", staticmethod(spy))
    with fflv.open(src) as r:
        r.frame(7, layers=["dot"])
    assert len(created) == 2  # dot color + dot alpha; bg and hidden were never decoded
    created.clear()
    with fflv.open(src) as r:
        list(r.frames(0, N))  # default: visible layers only
    assert len(created) == 3  # bg color, dot color + alpha
