"""Editing: add / remove layers and audio, set properties — without re-encoding existing layers."""

import shutil
import subprocess

import numpy as np
import pytest

import fflv
from fflv.edit import EditError
from fflv.format import LVFReader, validate
from fflv.meta import MetaError

W, H, N, GOP = 160, 96, 30, 10


def frames_of(path):
    """{frame: {layer_id: (type, flags, color, alpha)}} plus audio packets, for byte comparisons."""
    out, audio = {}, []
    with LVFReader(path) as r:
        ids = [L["id"] for L in r.meta["layers"]]
        for _o, cau, _s in r.iter_caus():
            out[cau.frame_index] = {ids[e.layer_index]: (e.type, e.frame_flags, e.color, e.alpha) for e in cau.entries}
            audio += [(a.pts_us, a.data) for a in cau.audio]
    return out, audio


def raps_of(path):
    return validate(str(path)).rap_frames


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


def test_add_numpy_layer_keeps_existing_packets(base, tmp_path):
    before, _ = frames_of(base)
    imgs = [np.random.default_rng(f).integers(0, 256, (20, 30, 4), dtype=np.uint8) for f in range(13, 27)]
    out = tmp_path / "added.lvd"
    fflv.add_layer(base, "new", imgs, start=13, lossless=True, rect=(100, 50, 30, 20), output=out)
    after, _ = frames_of(out)
    for f in range(N):
        for lid in ("a", "b"):
            assert after[f][lid] == before[f][lid], f"layer {lid} frame {f} was re-encoded"
        typ, flags, _c, _a = after[f]["new"]
        assert typ == (1 if 13 <= f < 27 else 0)
        if typ:
            assert bool(flags & 1) == (f in (13, 20)), f"key frames must be the layer start + existing RAPs ({f})"
    assert raps_of(out) == raps_of(base) == [0, 10, 20]
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
    rep = validate(str(base))
    assert rep.ok, [str(i) for i in rep.issues]
    with fflv.open(base) as r:
        clip = r.layer("clip")
        assert (clip.start, clip.end, clip.has_alpha) == (5, N, False)  # alpha auto-detected: none
        fn = dict(r.layer_frames("fn"))
    assert sorted(fn) == list(range(2, 9))
    assert all((img[:, :, :3] == f).all() for f, img in fn.items())


def test_remove_layer_remaps_indices(base, tmp_path):
    before, _ = frames_of(base)
    fflv.remove_layers(base, ["a"])
    after, _ = frames_of(base)
    for f in range(N):
        assert after[f]["b"] == before[f]["b"]
        assert "a" not in after[f]
    with fflv.open(base) as r:
        assert [L.id for L in r.layers] == ["b", "s"]
        assert r.layer(0).id == "b"
    with pytest.raises(MetaError, match="no layer"):
        fflv.remove_layers(base, ["nope"])


def test_set_is_in_place_and_instant(base):
    size = base.stat().st_size
    assert fflv.set_layer(base, "b", opacity=0.25, visible="false", z=-1, name="máscara ü", rect="5,6,100,80") is True
    assert base.stat().st_size == size
    with fflv.open(base) as r:
        b = r.layer("b")
        assert (b.opacity, b.visible, b.z, b.name, b.rect) == (0.25, False, -1, "máscara ü", (5, 6, 100, 80))
    assert validate(str(base)).ok
    fflv.set_layer(base, 1, id="mask")
    with fflv.open(base) as r:
        assert r.layer("mask").index == 1
    with pytest.raises(MetaError, match="cannot be edited"):
        fflv.set_layer(base, "mask", start_frame=3)
    with pytest.raises(MetaError, match="opacity"):
        fflv.set_layer(base, "mask", opacity=2)


def test_set_falls_back_to_rewrite_when_metadata_outgrows_its_space(base):
    before, _ = frames_of(base)
    assert fflv.set_layer(base, "a", name="x" * 20000) is False
    assert validate(str(base)).ok
    after, _ = frames_of(base)
    assert after == before
    with fflv.open(base) as r:
        assert r.layer("a").name == "x" * 20000


def test_audio_add_replace_remove(base, tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=3", str(wav)], check=True)
    video_before, _ = frames_of(base)
    fflv.set_audio(base, str(wav))
    video_after, audio = frames_of(base)
    assert video_after == video_before
    assert 45 <= len(audio) <= 52  # 1 s of 20-ms packets (the file is 1 s long)
    assert validate(str(base)).meta["audio"]["codec"] == "opus"
    fflv.set_audio(base, None)
    _, audio = frames_of(base)
    assert audio == [] and validate(str(base)).meta["audio"] is None


def test_edit_errors(base):
    with pytest.raises(MetaError, match="already used"):
        fflv.add_still(base, "s", np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(EditError, match="outside"):
        fflv.add_layer(base, "x", [np.zeros((4, 4, 3), np.uint8)] * 5, start=28)
    with pytest.raises(EditError, match="ran out"):
        fflv.add_layer(base, "y", iter([np.zeros((4, 4, 3), np.uint8)] * 3), start=0, end=10)
    assert validate(str(base)).ok  # a failed edit leaves the file untouched
    assert not list(base.parent.glob(".*fflv-tmp"))


def test_packed_file_can_be_edited(packed, tmp_path):
    path = tmp_path / "copy.lvd"
    shutil.copy(packed["path"], path)
    fflv.add_still(path, "note", np.full((20, 40, 4), 255, np.uint8), rect=(0, 0, 40, 20), start=10, end=20)
    fflv.remove_layers(path, ["calib"])
    rep = validate(str(path))
    assert rep.ok and rep.rap_frames == [0, 30, 60, 90]
    with fflv.open(path) as r:
        assert "calib" not in [L.id for L in r.layers] and r.layer("note").kind == "still"
        assert r.meta["audio"] is not None


def test_in_place_edits_are_crash_safe_and_alternate(base, monkeypatch):
    """Copy-on-write metadata: a crash before or during the switch leaves the old metadata; edits
    keep fitting beside the current copy, so the file never has to be rewritten (spec B.5)."""
    import fflv.format.container as C

    def name_of(path):
        with fflv.open(path) as r:
            return r.layer("b").name

    with LVFReader(base) as r:
        capacity = r.header.resources_offset - 64
        grow = capacity // 2 - r.header.meta_length - 64

    def crash(*_a):
        raise RuntimeError("power cut")

    def half_written(f, offset, data):
        f.seek(offset)
        f.write(data[: len(data) // 2])
        raise RuntimeError("power cut")

    for step in ("_switch_header", "_write_meta_copy"):
        with monkeypatch.context() as m:
            m.setattr(C, step, crash if step == "_switch_header" else half_written)
            with pytest.raises(RuntimeError, match="power cut"):
                fflv.set_layer(base, "b", name="x" * grow)
        assert validate(str(base)).ok and name_of(base) == "b", f"crash in {step} broke the file"

    size = base.stat().st_size
    for n in (grow, 3, grow, 10, grow // 2, grow):
        assert fflv.set_layer(base, "b", name="y" * n) is True
        assert validate(str(base)).ok and name_of(base) == "y" * n
    assert base.stat().st_size == size


def test_set_fallback_keeps_stills_stored_out_of_layer_order(tmp_path):
    """A (third-party) file whose resource region stores stills in another order than the layers:
    the rewrite fallback of set_layer must keep every still with its own image."""
    import json

    from fflv.format import LVFWriter

    src = tmp_path / "canon.lvd"
    # two colors whose PNGs have the same length (a swap would then go unnoticed by validation)
    red, blue = np.full((8, 8, 3), (30, 20, 10), np.uint8), np.full((8, 8, 3), (10, 20, 30), np.uint8)
    with fflv.Writer(src, (16, 8)) as w:
        w.add_layer("v", rect=(0, 0, 2, 2))
        w.add_still("red", red, rect=(0, 0, 8, 8))
        w.add_still("blue", blue, rect=(8, 0, 8, 8))
        w.write(v=np.zeros((2, 2, 3), np.uint8))
    with LVFReader(src) as r:
        meta = r.meta
        res = {L["id"]: r.resource(L["resource"]["offset"], L["resource"]["length"])
               for L in meta["layers"] if L["kind"] == "still"}
        caus = [c for _o, c, _s in r.iter_caus()]
    assert len(res["red"]) == len(res["blue"])
    for L in meta["layers"]:  # store blue first, red second
        if L["kind"] == "still":
            L["resource"]["offset"] = 0 if L["id"] == "blue" else len(res["blue"])
    odd = tmp_path / "odd.lvd"
    w = LVFWriter(odd)
    w.begin(json.dumps(meta).encode(), res["blue"] + res["red"], meta_capacity=0)  # no room: forces a rewrite
    for c in caus:
        w.write_cau(c)
    w.finish()
    assert validate(str(odd)).ok
    assert fflv.set_layer(odd, "v", name="renamed") is False
    with fflv.open(odd) as r:
        img = r.frame(0, layers=["red", "blue"])
    assert tuple(img[4, 4]) == (30, 20, 10) and tuple(img[4, 12]) == (10, 20, 30)
