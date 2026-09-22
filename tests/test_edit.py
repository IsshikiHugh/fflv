"""Editing: add / remove layers and audio, set properties — without re-encoding existing layers.

(That existing packets stay bit-identical is checked at the packet level by the Rust tests,
crates/fflv/tests/workflow.rs; here decoded pixels must not change.)
"""

import shutil
import subprocess

import numpy as np
import pytest

import fflv

W, H, N, GOP = 160, 96, 30, 10


def pixels_of(path, layer):
    with fflv.open(path) as r:
        return dict(r.layer_frames(layer))


def same_pixels(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


@pytest.fixture
def base(tmp_path):
    path = tmp_path / "base.lvd"
    with fflv.Writer(path, (W, H), fps=30, gop=GOP) as w:
        w.add_layer("a")
        w.add_layer("b", alpha=True, rect=(10, 10, 50, 40))
        w.add_still("s", np.full((8, 8, 3), 99, np.uint8), rect=(0, 0, 8, 8))
        for f in range(N):
            w.write(a=np.full((H, W, 3), f * 5, np.uint8), b=np.full((40, 50, 4), 200, np.uint8))
    return path


def test_add_numpy_layer_keeps_existing_layers(base, tmp_path):
    before = {k: pixels_of(base, k) for k in ("a", "b")}
    imgs = [np.random.default_rng(f).integers(0, 256, (20, 30, 4), dtype=np.uint8) for f in range(13, 27)]
    out = tmp_path / "added.lvd"
    rep = fflv.add_layer(base, "new", imgs, start=13, lossless=True, rect=(100, 50, 30, 20), output=out)
    assert rep.ok and rep.rap_frames == fflv.validate(base).rap_frames == [0, 10, 20]
    for k in ("a", "b"):
        assert same_pixels(pixels_of(out, k), before[k]), f"layer {k} changed"
    new_index = 3
    assert rep.layer_stats[new_index].keyframes == 2  # the layer start (13) and the existing RAP 20
    with fflv.open(out) as r:
        assert r.layer("new").z > r.layer("b").z  # on top by default
        got = dict(r.layer_frames("new"))
    assert sorted(got) == list(range(13, 27))
    for f, rgba in got.items():
        assert np.array_equal(rgba, imgs[f - 13])


def test_add_from_media_file_and_callable(base, tmp_path):
    media = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=80x60:rate=25:duration=0.5", "-c:v", "ffv1", str(media)], check=True)
    fflv.add_layer(base, "clip", str(media), rect=(0, 0, 80, 60), start=5)  # 12 source frames, cloned to the end
    fflv.add_layer(base, "fn", lambda f: np.full((10, 10), f, np.uint8), start=2, end=9, lossless=True)
    rep = fflv.validate(base)
    assert rep.ok, [str(i) for i in rep.issues]
    with fflv.open(base) as r:
        clip = r.layer("clip")
        assert (clip.start, clip.end, clip.has_alpha) == (5, N, False)  # alpha auto-detected: none
        fn = dict(r.layer_frames("fn"))
    assert sorted(fn) == list(range(2, 9))
    assert all((img[:, :, :3] == f).all() for f, img in fn.items())


def test_remove_layer_remaps_indices(base):
    before = pixels_of(base, "b")
    fflv.remove_layers(base, ["a"])
    assert same_pixels(pixels_of(base, "b"), before)
    with fflv.open(base) as r:
        assert [L.id for L in r.layers] == ["b", "s"]
        assert r.layer(0).id == "b"
    with pytest.raises(fflv.MetaError, match="no layer"):
        fflv.remove_layers(base, ["nope"])


def test_set_is_in_place_and_instant(base):
    size = base.stat().st_size
    assert fflv.set_layer(base, "b", opacity=0.25, visible="false", z=-1, name="máscara ü", rect="5,6,100,80") is True
    assert base.stat().st_size == size
    with fflv.open(base) as r:
        b = r.layer("b")
        assert (b.opacity, b.visible, b.z, b.name, b.rect) == (0.25, False, -1, "máscara ü", (5, 6, 100, 80))
    assert fflv.validate(base).ok
    fflv.set_layer(base, 1, id="mask", rect=(1, 2, 3, 4))
    with fflv.open(base) as r:
        assert r.layer("mask").index == 1 and r.layer("mask").rect == (1, 2, 3, 4)
    with pytest.raises(fflv.MetaError, match="cannot be edited"):
        fflv.set_layer(base, "mask", start_frame=3)
    with pytest.raises(fflv.MetaError, match="opacity"):
        fflv.set_layer(base, "mask", opacity=2)
    for bad in ("nan", "inf", "1e999"):
        with pytest.raises(fflv.MetaError, match="finite"):
            fflv.set_layer(base, "mask", z=bad)
    assert fflv.validate(base).ok


def test_set_falls_back_to_rewrite_when_metadata_outgrows_its_space(base):
    before = pixels_of(base, "b")
    assert fflv.set_layer(base, "a", name="x" * 20000) is False
    assert fflv.validate(base).ok
    assert same_pixels(pixels_of(base, "b"), before)
    with fflv.open(base) as r:
        assert r.layer("a").name == "x" * 20000
        assert (r.still("s")[:, :, :3] == 99).all()


def test_audio_add_replace_remove(base, tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=3", str(wav)], check=True)
    before = pixels_of(base, "a")
    rep = fflv.set_audio(base, str(wav))
    assert same_pixels(pixels_of(base, "a"), before)
    assert 45 <= rep.audio_packets <= 52  # 1 s of 20-ms packets (the file is 1 s long)
    assert rep.meta["audio"]["codec"] == "opus"
    rep = fflv.set_audio(base, None)
    assert rep.audio_packets == 0 and rep.meta["audio"] is None


def test_edit_errors(base):
    with pytest.raises(fflv.MetaError, match="already used"):
        fflv.add_still(base, "s", np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(fflv.EditError, match="outside"):
        fflv.add_layer(base, "x", [np.zeros((4, 4, 3), np.uint8)] * 5, start=28)
    with pytest.raises(fflv.EditError, match="ran out"):
        fflv.add_layer(base, "y", iter([np.zeros((4, 4, 3), np.uint8)] * 3), start=0, end=10)
    with pytest.raises(KeyError):  # an exception from the image source comes through as it is
        fflv.add_layer(base, "z", lambda f: {}[f], start=0)
    assert fflv.validate(base).ok  # a failed edit leaves the file untouched
    assert not list(base.parent.glob(".*fflv-tmp"))


def test_packed_file_can_be_edited(packed, tmp_path):
    path = tmp_path / "copy.lvd"
    shutil.copy(packed["path"], path)
    fflv.add_still(path, "note", np.full((20, 40, 4), 255, np.uint8), rect=(0, 0, 40, 20), start=10, end=20)
    fflv.remove_layers(path, ["calib"])
    rep = fflv.validate(path)
    assert rep.ok and rep.rap_frames == [0, 30, 60, 90]
    with fflv.open(path) as r:
        assert "calib" not in [L.id for L in r.layers] and r.layer("note").kind == "still"
        assert r.meta["audio"] is not None


def test_in_place_edits_alternate_and_never_grow_the_file(base):
    """Copy-on-write metadata: edits keep fitting beside the current copy (spec B.5)."""
    size = base.stat().st_size
    for n in (200, 3, 200, 10, 100, 200):
        assert fflv.set_layer(base, "b", name="y" * n) is True
        assert fflv.validate(base).ok
        with fflv.open(base) as r:
            assert r.layer("b").name == "y" * n
    assert base.stat().st_size == size
