"""Binary structures of LVF v1: file header, composite frames (CAU), index.

Everything is little-endian. The pack functions write exactly what they are
given (they do not enforce the format invariants), so that the test suite can
build deliberately broken files; `fflv.format.validate` is where invariants live.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .constants import (
    AUDIO_PACKET_HEADER_SIZE,
    CAU_HEADER_SIZE,
    CAU_PAYLOAD_BASE,
    ENTRY_FRAME,
    FRAME_FLAG_KEY,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    INDEX_HEADER_SIZE,
    MAGIC_CAU,
    MAGIC_FILE,
    MAGIC_INDEX,
    VERSION,
    VIDEO_ENTRY_HEADER_SIZE,
    CAU_FLAG_RAP,
)


class FormatError(ValueError):
    """Raised when bytes cannot be parsed as the structure they claim to be."""


_HEADER = struct.Struct("<4sHHQIIQQQ16s")
_CAU_HEADER = struct.Struct("<4sIIBBHHH")
_VIDEO_ENTRY = struct.Struct("<HBBII")
_AUDIO_PACKET = struct.Struct("<qII")
_INDEX_HEADER = struct.Struct("<4sI")
_INDEX_ENTRY = struct.Struct("<IB3xQ")

assert _HEADER.size == HEADER_SIZE
assert _CAU_HEADER.size == CAU_HEADER_SIZE
assert _VIDEO_ENTRY.size == VIDEO_ENTRY_HEADER_SIZE
assert _AUDIO_PACKET.size == AUDIO_PACKET_HEADER_SIZE
assert _INDEX_HEADER.size == INDEX_HEADER_SIZE
assert _INDEX_ENTRY.size == INDEX_ENTRY_SIZE


# --------------------------------------------------------------------------- #
# File header
# --------------------------------------------------------------------------- #
@dataclass
class FileHeader:
    meta_offset: int = HEADER_SIZE
    meta_length: int = 0
    resources_offset: int = 0
    cau_offset: int = 0
    index_offset: int = 0
    magic: bytes = MAGIC_FILE
    version: int = VERSION
    flags: int = 0
    reserved_20: int = 0
    reserved_48: bytes = b"\x00" * 16

    def pack(self) -> bytes:
        return _HEADER.pack(
            self.magic, self.version, self.flags,
            self.meta_offset, self.meta_length, self.reserved_20,
            self.resources_offset, self.cau_offset, self.index_offset,
            self.reserved_48,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "FileHeader":
        if len(buf) < HEADER_SIZE:
            raise FormatError(f"file header needs {HEADER_SIZE} bytes, got {len(buf)}")
        (magic, version, flags, meta_offset, meta_length, reserved_20,
         resources_offset, cau_offset, index_offset, reserved_48) = _HEADER.unpack_from(buf, 0)
        return cls(meta_offset, meta_length, resources_offset, cau_offset, index_offset,
                   magic, version, flags, reserved_20, reserved_48)


# --------------------------------------------------------------------------- #
# Composite frame (CAU)
# --------------------------------------------------------------------------- #
@dataclass
class VideoEntry:
    layer_index: int
    type: int
    frame_flags: int = 0
    color: bytes = b""
    alpha: bytes = b""

    @property
    def is_key(self) -> bool:
        return bool(self.frame_flags & FRAME_FLAG_KEY)

    @property
    def is_frame(self) -> bool:
        return self.type == ENTRY_FRAME


@dataclass
class AudioPacket:
    pts_us: int
    duration_us: int
    data: bytes


@dataclass
class Cau:
    frame_index: int
    flags: int = 0
    entries: list[VideoEntry] = field(default_factory=list)
    audio: list[AudioPacket] = field(default_factory=list)
    # Raw header fields, kept so that the validator can report them verbatim.
    magic: bytes = MAGIC_CAU
    reserved_13: int = 0
    reserved_18: int = 0
    video_entry_count: int | None = None  # None: derive from entries when packing

    @property
    def is_rap(self) -> bool:
        return bool(self.flags & CAU_FLAG_RAP)

    def pack(self) -> bytes:
        body = bytearray()
        for e in self.entries:
            body += _VIDEO_ENTRY.pack(e.layer_index, e.type, e.frame_flags, len(e.color), len(e.alpha))
            body += e.color
            body += e.alpha
        for a in self.audio:
            body += _AUDIO_PACKET.pack(a.pts_us, a.duration_us, len(a.data))
            body += a.data
        count = len(self.entries) if self.video_entry_count is None else self.video_entry_count
        payload_size = CAU_HEADER_SIZE - CAU_PAYLOAD_BASE + len(body)
        head = _CAU_HEADER.pack(self.magic, payload_size, self.frame_index, self.flags,
                                self.reserved_13, count, len(self.audio), self.reserved_18)
        return head + bytes(body)

    @classmethod
    def unpack(cls, buf: bytes | memoryview, offset: int = 0) -> tuple["Cau", int]:
        """Parse one CAU starting at `offset`. Returns (cau, total_size_in_bytes)."""
        mv = memoryview(buf)
        end_of_buf = len(mv)
        if offset + CAU_HEADER_SIZE > end_of_buf:
            raise FormatError(f"truncated CAU header at offset {offset}")
        (magic, payload_size, frame_index, flags, reserved_13,
         n_video, n_audio, reserved_18) = _CAU_HEADER.unpack_from(mv, offset)
        if magic != MAGIC_CAU:
            raise FormatError(f"bad CAU magic {bytes(magic)!r} at offset {offset}")
        total = CAU_PAYLOAD_BASE + payload_size
        end = offset + total
        if payload_size < CAU_HEADER_SIZE - CAU_PAYLOAD_BASE or end > end_of_buf:
            raise FormatError(f"CAU at offset {offset} has invalid payload_size {payload_size}")
        pos = offset + CAU_HEADER_SIZE
        entries: list[VideoEntry] = []
        for _ in range(n_video):
            if pos + VIDEO_ENTRY_HEADER_SIZE > end:
                raise FormatError(f"CAU {frame_index}: video entry header overruns the CAU")
            layer_index, etype, fflags, color_len, alpha_len = _VIDEO_ENTRY.unpack_from(mv, pos)
            pos += VIDEO_ENTRY_HEADER_SIZE
            if pos + color_len + alpha_len > end:
                raise FormatError(f"CAU {frame_index}: layer {layer_index} data overruns the CAU")
            color = bytes(mv[pos:pos + color_len]); pos += color_len
            alpha = bytes(mv[pos:pos + alpha_len]); pos += alpha_len
            entries.append(VideoEntry(layer_index, etype, fflags, color, alpha))
        audio: list[AudioPacket] = []
        for _ in range(n_audio):
            if pos + AUDIO_PACKET_HEADER_SIZE > end:
                raise FormatError(f"CAU {frame_index}: audio packet header overruns the CAU")
            pts_us, duration_us, length = _AUDIO_PACKET.unpack_from(mv, pos)
            pos += AUDIO_PACKET_HEADER_SIZE
            if pos + length > end:
                raise FormatError(f"CAU {frame_index}: audio packet data overruns the CAU")
            audio.append(AudioPacket(pts_us, duration_us, bytes(mv[pos:pos + length])))
            pos += length
        if pos != end:
            raise FormatError(
                f"CAU {frame_index}: {end - pos} trailing bytes after the declared entries")
        cau = cls(frame_index, flags, entries, audio, bytes(magic), reserved_13, reserved_18, n_video)
        return cau, total


def peek_cau_size(buf: bytes | memoryview, offset: int = 0) -> int:
    """Total byte size of the CAU starting at `offset` (needs its first 8 bytes)."""
    magic, payload_size = struct.unpack_from("<4sI", buf, offset)
    if magic != MAGIC_CAU:
        raise FormatError(f"bad CAU magic {bytes(magic)!r} at offset {offset}")
    return CAU_PAYLOAD_BASE + payload_size


# --------------------------------------------------------------------------- #
# Index table
# --------------------------------------------------------------------------- #
@dataclass
class IndexEntry:
    frame_index: int
    flags: int
    cau_offset: int

    @property
    def is_rap(self) -> bool:
        return bool(self.flags & CAU_FLAG_RAP)


def pack_index(entries: list[IndexEntry], magic: bytes = MAGIC_INDEX, count: int | None = None) -> bytes:
    out = bytearray(_INDEX_HEADER.pack(magic, len(entries) if count is None else count))
    for e in entries:
        out += _INDEX_ENTRY.pack(e.frame_index, e.flags, e.cau_offset)
    return bytes(out)


def unpack_index(buf: bytes) -> tuple[bytes, int, list[IndexEntry]]:
    """Returns (magic, declared_count, entries). Raises if the table is truncated."""
    if len(buf) < INDEX_HEADER_SIZE:
        raise FormatError("truncated index header")
    magic, count = _INDEX_HEADER.unpack_from(buf, 0)
    need = INDEX_HEADER_SIZE + count * INDEX_ENTRY_SIZE
    if len(buf) < need:
        raise FormatError(f"index declares {count} entries but only {len(buf)} bytes are present")
    entries = [IndexEntry(*_INDEX_ENTRY.unpack_from(buf, INDEX_HEADER_SIZE + i * INDEX_ENTRY_SIZE))
               for i in range(count)]
    return bytes(magic), count, entries
