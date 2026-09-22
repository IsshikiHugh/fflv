"""End-to-end checks of `fflv pack` output on the generated test material."""

import json
import subprocess

import numpy as np
import pytest

import fflv


def test_packed_file_is_valid(packed):
    rep = fflv.validate(packed["path"])
    assert rep.ok, [str(i) for i in rep.issues]
    assert not rep.warnings
    meta = rep.meta
    assert meta["frame_count"] == 120
    assert meta["max_rap_interval"] == 30
    # all layers are on the 30-frame grid, including the one that starts at frame 45
    assert rep.rap_frames == [0, 30, 60, 90]
    square = next(L for L in meta["layers"] if L["id"] == "square")
    assert (square["start_frame"], square["end_frame"]) == (45, 75)
    assert rep.layer_stats[meta["layers"].index(square)].keyframes == 2  # its start (45) and the grid point 60
    assert meta["audio"]["pre_skip"] > 0
    assert rep.audio_packets >= 4 * 50 - 2  # ~20 ms packets over 4 s


def _read_barcode(rgb: np.ndarray, x: int, y: int, cell: int, cells: int):
    bits = [int(rgb[y + cell // 2, x + i * cell + cell // 2, :3].mean() > 128) for i in range(cells)]
    data = bits[1:-1]
    if bits[0] != 1 or sum(data) % 2 != bits[-1]:
        return None
    return int("".join(map(str, data)), 2)


def test_every_layer_frame_carries_its_own_frame_number(packed):
    """Decode every video layer of every composite frame and read the barcode drawn into it."""
    probes = packed["probes"]["barcode"]
    seen = 0
    with fflv.open(packed["path"]) as r:
        for L in r.layers:
            if L.kind != "video":
                continue
            p = probes["layers"][L.id]
            for f, rgba in r.layer_frames(L.id):
                n = _read_barcode(rgba, p["x"] - L.rect[0], p["y"] - L.rect[1], probes["cell"], probes["cells"])
                assert n == f, f"layer {L.id} at frame {f} shows {n}"
                seen += 1
        assert seen == sum(L.end - L.start for L in r.layers if L.kind == "video")


def test_alpha_ramp_spans_0_to_255(packed):
    """The calibration ramp must decode to alpha ≈ 0 at the transparent end and ≈ 255 at the opaque
    end: exactly the range the VP9 header signals (color_range = limited)."""
    with fflv.open(packed["path"]) as r:
        _, calib = next(r.layer_frames("calib"))
    row = calib[10, :, 3].astype(int)
    assert row[0] <= 3 and row[-1] >= 252, (row[0], row[-1])
    assert np.all(np.diff(row[::16]) >= -3)  # monotonic ramp (allowing coding noise)


def test_lossless_layer_is_exact(packed):
    with fflv.open(packed["path"]) as r:
        for f, rgba in r.layer_frames("exact", 28, 33):
            y, x = np.mgrid[0:64, 0:256]
            want = np.stack([(x * 7 + f) & 255, (y * 13 + 3 * f) & 255, (x ^ y ^ f) & 255], -1).astype(np.uint8)
            assert np.array_equal(rgba[:64, :, :3], want) and (rgba[:64, :, 3] == 255).all()
            assert np.array_equal(rgba[64:80, :, 3], np.tile(np.arange(256, dtype=np.uint8), (16, 1)))


def test_pack_rejects_alpha_request_on_opaque_source(packed, tmp_path):
    proj = dict(packed["project"])
    proj["layers"] = [dict(proj["layers"][0], alpha=True, src=str(packed["dir"] / "bg.mkv"))]
    proj.pop("audio")
    p = tmp_path / "p.json"
    p.write_text(json.dumps(proj))
    out = tmp_path / "x.lvd"
    out.write_bytes(b"previous version")
    with pytest.raises(fflv.PackError, match="no alpha channel"):
        fflv.pack(p, out, log=None)
    assert out.read_bytes() == b"previous version"
    assert not list(tmp_path.glob(".*fflv-tmp"))


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.mark.parametrize("lossless", [False, True])
def test_pack_pads_odd_layers_by_repeating_the_edge(tmp_path, lossless):
    """Odd rects from 4:2:0 sources: the last row/column must keep the real content (spec B.4),
    and a transparent source must stay transparent there."""
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x48:r=30:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            tmp_path / "red.mp4")
    _ffmpeg("-f", "lavfi", "-i", "color=c=red@0.0:s=64x48:r=30:d=1,format=yuva420p", "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", tmp_path / "clear.webm")
    proj = {"canvas": {"width": 80, "height": 60}, "fps": "30/1", "duration": 1, "gop": 30, "layers": [
        {"id": "red", "src": "red.mp4", "rect": [0, 0, 63, 47], "lossless": lossless},
        {"id": "clear", "src": "clear.webm", "rect": [0, 0, 63, 47], "alpha": True, "lossless": lossless}]}
    (tmp_path / "p.json").write_text(json.dumps(proj))
    out = tmp_path / "odd.lvd"
    lines = []
    assert fflv.pack(tmp_path / "p.json", out, log=lines.append).ok
    assert any("2 layers" in line for line in lines)
    with fflv.open(out) as r:
        assert r.layer("red").content_size == (63, 47)
        _, red = next(r.layer_frames("red"))
        _, clear = next(r.layer_frames("clear"))
    assert red.shape == (47, 63, 4)
    assert red[:, :, 0].min() > 200 and red[:, :, 1].max() < 40, "edge pixels lost their color"
    assert clear[:, :, 3].max() <= 3, "a transparent source became opaque at the padded edge"


def test_pack_reports_project_errors(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"canvas": {"width": 16, "height": 16}, "layers": []}))
    with pytest.raises(fflv.PackError, match="duration"):
        fflv.pack(tmp_path / "p.json", log=None)
