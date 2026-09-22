"""fflv.Writer (numpy → .lvd) and fflv.open (decode) round trips."""

import numpy as np
import pytest

import fflv

W, H = 320, 180


def gradient(f: int) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    return np.stack([(x + 2 * f) % 256, (y * 2) % 256, np.full_like(x, 128)], -1).astype(np.uint8)


def noise(f: int, w: int, h: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed * 1000 + f).integers(0, 256, (h, w, 4), dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


@pytest.fixture
def written(tmp_path):
    """40 frames, gop 10: a lossy full-canvas layer, a lossless odd-sized alpha layer updated every
    third frame (sticky), a lossy alpha layer living in [7, 30), and a still in [5, 15)."""
    path = tmp_path / "w.lvd"
    masks, lates = {}, {}
    with fflv.Writer(path, (W, H), fps=30, gop=10) as w:
        w.add_layer("bg")
        w.add_layer("mask", alpha=True, lossless=True, rect=(10, 20, 101, 51))
        w.add_layer("late", alpha=True, rect=(200, 100, 64, 64), blend="screen")
        w.add_still("legend", np.full((10, 20, 4), 200, np.uint8), rect=(300, 0, 20, 10), start=5, end=15)
        for f in range(40):
            imgs = {"bg": gradient(f)}
            if f % 3 == 0:
                masks[f] = noise(f, 101, 51, 1)
                imgs["mask"] = masks[f]
            if f == 30:
                w.end_layer("late")
            if 7 <= f < 30:
                lates[f] = np.dstack([gradient(f)[:64, :64], np.full((64, 64), 180, np.uint8)])
                imgs["late"] = lates[f]
            assert w.write(imgs) == f
    assert w.report.ok and w.report.path == str(path)
    return path, masks, lates


def test_written_file_is_valid_and_on_the_grid(written):
    path, _, _ = written
    rep = fflv.validate(path)
    assert rep.ok, [str(i) for i in rep.issues]
    assert rep.rap_frames == [0, 10, 20, 30]
    layers = {L["id"]: L for L in rep.meta["layers"]}
    assert (layers["late"]["start_frame"], layers["late"]["end_frame"]) == (7, 30)
    assert (layers["legend"]["start_frame"], layers["legend"]["end_frame"]) == (5, 15)
    m = layers["mask"]
    assert m["lossless"] and m["alpha_range"] == "full" and m["codec"].startswith("vp09.01.")
    assert (m["coded_width"], m["coded_height"], m["content_size"]) == (102, 52, [101, 51])
    assert rep.meta["generator"] == "fflv"


def test_lossless_layer_is_bit_exact_and_sticky(written):
    path, masks, _ = written
    with fflv.open(path) as r:
        got = dict(r.layer_frames("mask"))
    assert sorted(got) == list(range(40))
    for f, rgba in got.items():
        src = masks[f - f % 3]  # the last image given at or before f
        assert np.array_equal(rgba, src), f"frame {f}"


def test_lossy_layers_are_close(written):
    path, _, lates = written
    with fflv.open(path) as r:
        for f, rgba in r.layer_frames("late"):
            assert psnr(rgba[:, :, :3], lates[f][:, :, :3]) > 30
            assert np.abs(rgba[:, :, 3].astype(int) - 180).max() <= 2
        for f, rgba in r.layer_frames("bg", 0, 12):
            assert psnr(rgba[:, :, :3], gradient(f)) > 30
            assert (rgba[:, :, 3] == 255).all()


def test_composite_matches_the_formulas(written):
    path, masks, _ = written
    with fflv.open(path) as r:
        only_mask = r.frame(13, layers=["mask"])
        bg = np.zeros((H, W, 3), np.float64)
        m = masks[12].astype(np.float64) / 255
        region = m[:, :, :3] * m[:, :, 3:] + bg[20:71, 10:111] * (1 - m[:, :, 3:])
        want = np.clip(np.floor(region * 255 + 0.5), 0, 255)
        assert np.abs(only_mask[20:71, 10:111].astype(int) - want).max() <= 1
        assert (only_mask[:20] == 0).all()
        # the still is drawn only inside its range; hiding works
        assert (r.frame(6, layers=["legend"])[0:10, 300:320] > 150).all()
        assert (r.frame(20, layers=["legend"])[0:10, 300:320] == 0).all()
        full = r.frame(13)
        assert full.shape == (H, W, 3)
        assert np.array_equal(r.frame(13, hide=["late", "legend", "bg"]), only_mask)
        assert np.array_equal(r.still("legend"), np.full((10, 20, 4), 200, np.uint8))


def test_transparent_composite_keeps_alpha(written):
    path, masks, _ = written
    with fflv.open(path) as r:
        rgba = r.frame(3, layers=["mask"], transparent=True)
    assert rgba.shape == (H, W, 4)
    assert (rgba[:20, :, 3] == 0).all()
    assert np.array_equal(rgba[20:71, 10:111, 3], masks[3][:, :, 3])


def test_decode_gives_each_layers_own_planes(written):
    path, masks, _ = written
    with fflv.open(path) as r:
        mask = r.layer("mask")
        out = list(r.decode(11, 14, layers=["mask", "legend"]))
    assert [f for f, _ in out] == [11, 12, 13]
    for f, planes in out:
        assert list(planes) == [mask.index]
        rgb, alpha = planes[mask.index]
        src = masks[f - f % 3]
        assert np.array_equal(rgb, src[:, :, :3]) and np.array_equal(alpha, src[:, :, 3])


def test_reader_api(written):
    path, _, _ = written
    with fflv.open(path) as r:
        assert r.frame_count == 40 and r.size == (W, H) and r.fps == 30
        assert [L.id for L in r.select()] == ["bg", "mask", "late", "legend"]
        assert [L.id for L in r.select(["late", 0], hide=[0])] == ["late"]
        late = r.layer("late")
        assert (late.start, late.end, late.blend, late.active(29), late.active(30)) == (7, 30, "screen", True, False)
        assert r.layer(1).id == r.layer("1").id == "mask"
        assert r.raps == [0, 10, 20, 30] and r.rap_at_or_before(19) == 10
        assert r.pts_us(1) == 33333
        assert r.check().ok
        with pytest.raises(fflv.MetaError, match="no layer"):
            r.layer("nope")
        with pytest.raises(fflv.DecodeError, match="outside"):
            r.frame(40)
        assert "w.lvd" in repr(r)


def test_writer_errors(tmp_path):
    with pytest.raises(fflv.WriterError, match="unknown video layer"):
        with fflv.Writer(tmp_path / "a.lvd", (64, 64)) as w:
            w.add_layer("a")
            w.write(b=np.zeros((64, 64, 3), np.uint8))
    assert not (tmp_path / "a.lvd").exists() and not list(tmp_path.glob(".*fflv-tmp"))

    w = fflv.Writer(tmp_path / "b.lvd", (64, 64))
    w.add_layer("a")
    w.write(a=np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(fflv.WriterError, match="before the first write"):
        w.add_layer("late")
    with pytest.raises(ValueError, match="64x64"):
        w.write(a=np.zeros((32, 32, 3), np.uint8))
    w.abort()
    assert not (tmp_path / "b.lvd").exists()

    with pytest.raises(fflv.WriterError, match="never received an image"):
        with fflv.Writer(tmp_path / "c.lvd", (64, 64)) as w:
            w.add_layer("a")
            w.add_layer("b")
            w.write(a=np.zeros((64, 64, 3), np.uint8))
    assert not (tmp_path / "c.lvd").exists() and not list(tmp_path.glob(".*fflv-tmp"))

    with pytest.raises(fflv.WriterError, match="contiguous"):
        with fflv.Writer(tmp_path / "d.lvd", (64, 64)) as w:
            w.add_layer("a")
            w.write(a=np.zeros((64, 64, 3), np.uint8))
            w.end_layer("a")
            w.write(a=np.zeros((64, 64, 3), np.uint8))

    with pytest.raises(fflv.MetaError, match="dtype"):
        with fflv.Writer(tmp_path / "e.lvd", (4, 4)) as w:
            w.add_layer("a")
            w.write(a=np.zeros((4, 4, 3), np.int32))
    w = fflv.Writer(tmp_path / "f.lvd", (4, 4))
    w.add_layer("x")
    with pytest.raises(fflv.MetaError, match="already used"):
        w.add_layer("x")


def test_image_types_are_normalised(tmp_path):
    path = tmp_path / "t.lvd"
    with fflv.Writer(path, (32, 16), gop=4) as w:
        w.add_layer("g", lossless=True)
        w.add_layer("m", alpha=True, lossless=True)
        w.write(g=np.full((16, 32), 0.5, np.float32), m=np.ones((16, 32), bool))
        w.write(g=np.full((16, 32, 1), 7, np.uint8), m=np.zeros((16, 32, 4), np.uint8))
        w.write(g=np.full((16, 64, 3), 9, np.uint8)[:, ::2], m=np.zeros((16, 32, 4), np.uint8))  # a strided view
    with fflv.open(path) as r:
        g = [rgba for _, rgba in r.layer_frames("g")]
        m = [rgba for _, rgba in r.layer_frames("m")]
    assert (g[0][:, :, :3] == 128).all() and (g[1][:, :, :3] == 7).all() and (g[2][:, :, :3] == 9).all()
    assert (m[0] == 255).all() and (m[1] == 0).all()


def test_a_rejected_image_changes_nothing(tmp_path):
    """A bad image for one layer must not update another layer's sticky image or start frame."""
    path = tmp_path / "keep.lvd"
    with fflv.Writer(path, (16, 16)) as w:
        w.add_layer("a", lossless=True)
        w.add_layer("b", lossless=True)
        w.write(a=np.full((16, 16, 3), 10, np.uint8), b=np.zeros((16, 16, 3), np.uint8))
        with pytest.raises(ValueError):
            w.write(a=np.full((16, 16, 3), 99, np.uint8), b=np.zeros((8, 8, 3), np.uint8))
        w.write(b=np.zeros((16, 16, 3), np.uint8))  # "a" omitted: repeats its last *written* image
    with fflv.open(path) as r:
        a = dict(r.layer_frames("a"))
    assert sorted(a) == [0, 1] and (a[1][:, :, :3] == 10).all()


def test_audio_track(tmp_path):
    import subprocess

    wav = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=3", str(wav)], check=True)
    path = tmp_path / "a.lvd"
    with fflv.Writer(path, (16, 16), fps=25) as w:
        w.add_layer("v")
        w.set_audio(wav, bitrate="64k", channels=1)
        for _ in range(25):
            w.write(v=np.zeros((16, 16, 3), np.uint8))
    rep = w.report
    assert rep.ok and rep.meta["audio"]["channels"] == 1 and rep.meta["audio"]["pre_skip"] > 0
    assert 48 <= rep.audio_packets <= 51  # 1 s of 20-ms packets: cut to the video
