import pytest

from fflv.format.binary import (AudioPacket, Cau, FileHeader, FormatError, IndexEntry, VideoEntry, pack_index,
                        peek_cau_size, unpack_index)
from fflv.format.constants import CAU_FLAG_RAP, ENTRY_EMPTY, ENTRY_FRAME, FRAME_FLAG_KEY, HEADER_SIZE


def test_header_roundtrip():
    h = FileHeader(meta_offset=64, meta_length=100, resources_offset=164, cau_offset=500, index_offset=9000)
    raw = h.pack()
    assert len(raw) == HEADER_SIZE and raw[:4] == b"LVF1"
    assert raw[4:6] == b"\x01\x00"  # little-endian version
    assert FileHeader.unpack(raw) == h


def test_cau_layout_matches_spec():
    cau = Cau(7, CAU_FLAG_RAP,
              [VideoEntry(0, ENTRY_FRAME, FRAME_FLAG_KEY, b"abc", b"de"), VideoEntry(2, ENTRY_EMPTY)],
              [AudioPacket(-5, 20000, b"xyz")])
    raw = cau.pack()
    # 20 header + (12 + 3 + 2) + 12 + (16 + 3)
    assert len(raw) == 20 + 17 + 12 + 19
    assert raw[:4] == b"CAUF"
    assert int.from_bytes(raw[4:8], "little") == len(raw) - 8
    assert int.from_bytes(raw[8:12], "little") == 7
    assert raw[12] == 1
    assert int.from_bytes(raw[14:16], "little") == 2
    assert int.from_bytes(raw[16:18], "little") == 1
    assert peek_cau_size(raw) == len(raw)
    back, size = Cau.unpack(raw)
    assert size == len(raw)
    assert back.frame_index == 7 and back.is_rap
    assert back.entries == cau.entries
    assert back.audio == cau.audio


def test_cau_unpack_rejects_overrun():
    raw = bytearray(Cau(0, 0, [VideoEntry(0, ENTRY_FRAME, 0, b"a" * 10)]).pack())
    raw[20 + 4:20 + 8] = (1000).to_bytes(4, "little")  # color_len too large
    with pytest.raises(FormatError):
        Cau.unpack(bytes(raw))


def test_cau_unpack_rejects_bad_magic():
    raw = bytearray(Cau(0).pack())
    raw[:4] = b"XXXX"
    with pytest.raises(FormatError):
        Cau.unpack(bytes(raw))


def test_cau_unpack_rejects_trailing_bytes():
    raw = bytearray(Cau(0).pack() + b"\0\0")
    raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    with pytest.raises(FormatError):
        Cau.unpack(bytes(raw))


def test_index_roundtrip():
    entries = [IndexEntry(0, 1, 100), IndexEntry(1, 0, 250)]
    raw = pack_index(entries)
    assert len(raw) == 8 + 2 * 16
    magic, count, back = unpack_index(raw)
    assert magic == b"IDX1" and count == 2 and back == entries
    with pytest.raises(FormatError):
        unpack_index(raw[:-1])
