"""Minimal VP9 bitstream inspection: superframe split + uncompressed header.

Used to double-check what the encoder produced ("do not trust the encoder"):
key-frame-ness, that each packet shows exactly one frame, and the coded size
of key frames. Reference: VP9 Bitstream Specification v0.6, sections 6.2 and
Annex B (superframes).
"""

from __future__ import annotations

from dataclasses import dataclass

KEY_FRAME = 0
CS_RGB = 7
SYNC_CODE = (0x49, 0x83, 0x42)


class Vp9Error(ValueError):
    pass


class _Bits:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def f(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.pos >> 3
            if byte >= len(self.data):
                raise Vp9Error("uncompressed header truncated")
            bit = (self.data[byte] >> (7 - (self.pos & 7))) & 1
            v = (v << 1) | bit
            self.pos += 1
        return v


@dataclass
class Vp9Frame:
    profile: int
    show_existing_frame: bool
    key_frame: bool
    show_frame: bool
    # Only known for key frames:
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    color_space: int | None = None
    color_range: int | None = None  # 0 = studio (limited), 1 = full
    subsampling: tuple[int, int] | None = None

    @property
    def shown(self) -> bool:
        return self.show_existing_frame or self.show_frame


@dataclass
class Vp9Packet:
    frames: list[Vp9Frame]
    superframe: bool

    @property
    def key_frame(self) -> bool:
        """A packet is decodable on its own iff its first frame is a key frame."""
        return self.frames[0].key_frame

    @property
    def shown_count(self) -> int:
        return sum(1 for fr in self.frames if fr.shown)

    @property
    def key_info(self) -> Vp9Frame | None:
        return next((fr for fr in self.frames if fr.key_frame), None)


def split_superframe(data: bytes) -> tuple[list[bytes], bool]:
    """Annex B: returns (frames, is_superframe)."""
    if not data:
        raise Vp9Error("empty packet")
    marker = data[-1]
    if marker & 0xE0 == 0xC0:
        frames_in_sf = (marker & 0x7) + 1
        mag = ((marker >> 3) & 0x3) + 1
        index_sz = 2 + mag * frames_in_sf
        if len(data) >= index_sz and data[-index_sz] == marker:
            sizes = []
            p = len(data) - index_sz + 1
            for _ in range(frames_in_sf):
                sizes.append(int.from_bytes(data[p:p + mag], "little"))
                p += mag
            frames, off = [], 0
            for s in sizes:
                if off + s > len(data) - index_sz:
                    raise Vp9Error("superframe index points past the data")
                frames.append(data[off:off + s])
                off += s
            return [fr for fr in frames if fr], True
    return [data], False


def parse_frame(data: bytes) -> Vp9Frame:
    b = _Bits(data)
    if b.f(2) != 2:
        raise Vp9Error("frame_marker is not 2 (not a VP9 frame)")
    low = b.f(1)
    high = b.f(1)
    profile = (high << 1) | low
    if profile == 3:
        b.f(1)
    if b.f(1):  # show_existing_frame
        b.f(3)
        return Vp9Frame(profile, True, False, True)
    frame_type = b.f(1)
    show_frame = bool(b.f(1))
    b.f(1)  # error_resilient_mode
    fr = Vp9Frame(profile, False, frame_type == KEY_FRAME, show_frame)
    if not fr.key_frame:
        return fr
    if (b.f(8), b.f(8), b.f(8)) != SYNC_CODE:
        raise Vp9Error("key frame without the VP9 sync code")
    bit_depth = 8
    if profile >= 2:
        bit_depth = 12 if b.f(1) else 10
    color_space = b.f(3)
    if color_space != CS_RGB:
        color_range = b.f(1)
        if profile in (1, 3):
            sub = (b.f(1), b.f(1))
            b.f(1)
        else:
            sub = (1, 1)
    else:
        color_range = 1
        sub = (0, 0)
        if profile in (1, 3):
            b.f(1)
    fr.bit_depth = bit_depth
    fr.color_space = color_space
    fr.color_range = color_range
    fr.subsampling = sub
    fr.width = b.f(16) + 1
    fr.height = b.f(16) + 1
    return fr


def inspect_packet(data: bytes) -> Vp9Packet:
    frames, sf = split_superframe(data)
    if not frames:
        raise Vp9Error("superframe without frames")
    return Vp9Packet([parse_frame(fr) for fr in frames], sf)


# --------------------------------------------------------------------------- #
# WebCodecs codec string
# --------------------------------------------------------------------------- #
# (level, max luma picture size, max luma sample rate) — VP9 levels (webmproject.org/vp9/levels)
_LEVELS = [
    (10, 36864, 829440),
    (11, 73728, 2764800),
    (20, 122880, 4608000),
    (21, 245760, 9216000),
    (30, 552960, 20736000),
    (31, 983040, 36864000),
    (40, 2228224, 83558400),
    (41, 2228224, 160432128),
    (50, 8912896, 311951360),
    (51, 8912896, 588251136),
    (52, 8912896, 1176502272),
    (60, 35651584, 1176502272),
    (61, 35651584, 2353004544),
    (62, 35651584, 4706009088),
]


def vp9_level(width: int, height: int, fps: float) -> int:
    """The smallest VP9 level that fits picture size and luma sample rate."""
    size = width * height
    rate = size * fps
    for lv, max_size, max_rate in _LEVELS:
        if size <= max_size and rate <= max_rate:
            return lv
    return _LEVELS[-1][0]


def codec_string(width: int, height: int, fps: float, lossless: bool = False) -> str:
    """WebCodecs codec string.

    Regular planes: profile 0, 8-bit 4:2:0 ("vp09.00.LL.08").
    Lossless color planes: profile 1, 8-bit 4:4:4 RGB (identity matrix, BT.709 primaries, sRGB
    transfer, full range): "vp09.01.LL.08.03.01.13.00.01".
    """
    level = vp9_level(width, height, fps)
    if lossless:
        return f"vp09.01.{level:02d}.08.03.01.13.00.01"
    return f"vp09.00.{level:02d}.08"


def codec_profile(codec: str) -> int | None:
    """Profile number of a vp09.* codec string (None if malformed)."""
    parts = codec.split(".")
    if len(parts) < 4 or parts[0] != "vp09" or not parts[1].isdigit():
        return None
    return int(parts[1])
