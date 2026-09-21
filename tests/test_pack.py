"""End-to-end checks of `fflv pack` output on the generated test material."""

import av
import numpy as np
import pytest

from fflv.format import LVFReader, pts_us, validate
from fflv.format.constants import ENTRY_FRAME
from fflv.project import PackError, pack


def test_packed_file_is_valid(packed):
    rep = validate(str(packed["path"]))
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


def _decoder():
    ctx = av.CodecContext.create("vp9", "r")
    ctx.thread_count = 1
    return ctx


def _read_barcode(rgb: np.ndarray, x: int, y: int, cell: int, cells: int):
    bits = [int(rgb[y + cell // 2, x + i * cell + cell // 2].mean() > 128) for i in range(cells)]
    data = bits[1:-1]
    if bits[0] != 1 or sum(data) % 2 != bits[-1]:
        return None
    return int("".join(map(str, data)), 2)


def test_every_layer_frame_carries_its_own_frame_number(packed):
    """Decode every plane of every composite frame and read the barcode drawn into it."""
    probes = packed["probes"]["barcode"]
    with LVFReader(packed["path"]) as r:
        layers = r.meta["layers"]
        decoders = {}
        seen = 0
        for _off, cau, _size in r.iter_caus():
            for e in cau.entries:
                if e.type != ENTRY_FRAME:
                    continue
                L = layers[e.layer_index]
                dec = decoders.setdefault(e.layer_index, _decoder())
                frames = dec.decode(av.Packet(e.color))
                assert len(frames) == 1, "one packet must yield exactly one frame"
                rgb = frames[0].to_ndarray(format="rgb24")
                p = probes["layers"][L["id"]]
                n = _read_barcode(rgb, p["x"] - L["rect"]["x"], p["y"] - L["rect"]["y"],
                                  probes["cell"], probes["cells"])
                assert n == cau.frame_index, f"layer {L['id']} at frame {cau.frame_index} shows {n}"
                seen += 1
        assert seen == sum(L["end_frame"] - L["start_frame"] for L in layers if L["kind"] == "video")


def test_alpha_plane_is_limited_range_and_spans_0_to_255(packed):
    """The calibration ramp must decode to Y≈16 at the transparent end and Y≈235 at the opaque end,
    i.e. exactly the range the VP9 header signals (color_range = limited)."""
    with LVFReader(packed["path"]) as r:
        li = next(i for i, L in enumerate(r.meta["layers"]) if L["id"] == "calib")
        _off, cau, _ = next(r.iter_caus())
        e = next(e for e in cau.entries if e.layer_index == li)
        y = _decoder().decode(av.Packet(e.alpha))[0].to_ndarray(format="yuv420p")
        row = y[10, :].astype(int)  # a ramp row; luma plane comes first in the yuv420p array
        assert abs(row[0] - 16) <= 2 and abs(row[-1] - 235) <= 2, (row[0], row[-1])
        assert np.all(np.diff(row[::16]) >= -2)  # monotonic ramp (allowing coding noise)


def test_audio_packets_sit_in_their_frame_window(packed):
    with LVFReader(packed["path"]) as r:
        num, den = r.meta["fps"]["num"], r.meta["fps"]["den"]
        total = 0
        for _off, cau, _ in r.iter_caus():
            lo, hi = pts_us(cau.frame_index, num, den), pts_us(cau.frame_index + 1, num, den)
            for a in cau.audio:
                assert lo <= a.pts_us < hi
                total += 1
        assert total >= 4 * 50 - 2  # ~20 ms packets over 4 s


def test_pack_rejects_alpha_request_on_opaque_source(packed, tmp_path):
    import json
    proj = dict(packed["project"])
    proj["layers"] = [dict(proj["layers"][0], alpha=True, src=str(packed["dir"] / "bg.mkv"))]
    proj.pop("audio")
    p = tmp_path / "p.json"
    p.write_text(json.dumps(proj))
    with pytest.raises(PackError, match="no alpha channel"):
        pack(p, str(tmp_path / "x.lvd"), jobs=1, log=lambda s: None)
