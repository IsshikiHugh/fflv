"""Streaming LVF writer and random-access reader."""

from __future__ import annotations

import json
import os
from typing import BinaryIO, Iterator

from .binary import (
    Cau,
    FileHeader,
    FormatError,
    IndexEntry,
    pack_index,
    peek_cau_size,
    unpack_index,
)
from .constants import CAU_FLAG_RAP, HEADER_SIZE, INDEX_FLAG_RAP


def _json_default(o):
    item = getattr(o, "item", None)  # numpy scalars
    if callable(item):
        return item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def encode_meta(meta: dict) -> bytes:
    """Standard JSON (RFC 8259): NaN / Infinity are refused rather than written as non-JSON."""
    return json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default).encode("utf-8")


def meta_capacity_for(meta_bytes: bytes) -> int:
    """Room reserved for the metadata: twice its size (at least 4 KiB), in 4 KiB steps, so that an
    edited copy fits beside the current one (see rewrite_meta_in_place)."""
    need = max(4096, 2 * len(meta_bytes))
    return (need + 4095) // 4096 * 4096


def _free_slot(region: tuple[int, int], current: tuple[int, int], size: int) -> int | None:
    """Offset of `size` bytes inside `region` that do not overlap `current` (both [start, end))."""
    lo, hi = region
    cur_start, cur_end = current
    if cur_start - lo >= size:  # at the start, before the current copy
        return lo
    end = (hi - size) // 8 * 8  # right-aligned at the end, so copies alternate between the two ends
    if end >= max(cur_end, lo):
        return end
    return None


def _sync(f: BinaryIO) -> None:
    f.flush()
    os.fsync(f.fileno())


def _write_meta_copy(f: BinaryIO, offset: int, data: bytes) -> None:
    f.seek(offset)
    f.write(data)
    _sync(f)


def _switch_header(f: BinaryIO, header: FileHeader) -> None:
    f.seek(0)
    f.write(header.pack())
    _sync(f)


def rewrite_meta_in_place(path: str | os.PathLike, meta: dict | bytes) -> bool:
    """Replace the metadata of an existing file without touching anything else (spec B.5).

    Copy-on-write, so a crash at any point leaves a valid file: the new JSON is written to free
    space in the reserved region [end of header, resources_offset) — never over the current copy —
    and synced; only then is the 64-byte header switched to it (meta_offset, meta_length).
    Returns False (and changes nothing) when there is no room beside the current copy."""
    data = meta if isinstance(meta, bytes) else encode_meta(meta)
    with open(path, "r+b") as f:
        header = FileHeader.unpack(f.read(HEADER_SIZE))
        region = (HEADER_SIZE, header.resources_offset)
        current = (header.meta_offset, header.meta_offset + header.meta_length)
        offset = _free_slot(region, current, len(data))
        if offset is None:
            return False
        _write_meta_copy(f, offset, data)
        header.meta_offset, header.meta_length = offset, len(data)
        _switch_header(f, header)
    return True


class LVFWriter:
    """Writes header placeholder → meta → resources → CAUs → index, then back-patches the header.

    It writes exactly what it is given; invariants are the caller's job (and the validator's).
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self.f: BinaryIO = open(self.path, "wb")
        self.header = FileHeader()
        self.index: list[IndexEntry] = []
        self._begun = False

    def begin(self, meta: dict | bytes, resources: bytes = b"", meta_capacity: int | None = None) -> None:
        """`meta_capacity`: bytes reserved for the metadata (default: meta_capacity_for(meta));
        0 means exactly the metadata, no padding."""
        meta_bytes = meta if isinstance(meta, bytes) else encode_meta(meta)
        if meta_capacity is None:
            meta_capacity = meta_capacity_for(meta_bytes)
        meta_capacity = max(meta_capacity, len(meta_bytes))
        self.f.write(b"\x00" * HEADER_SIZE)
        self.header.meta_offset = self.f.tell()
        self.header.meta_length = len(meta_bytes)
        self.f.write(meta_bytes + b" " * (meta_capacity - len(meta_bytes)))
        self.header.resources_offset = self.f.tell()
        self.f.write(resources)
        self.header.cau_offset = self.f.tell()
        self._begun = True

    def write_cau(self, cau: Cau | bytes, frame_index: int | None = None, flags: int | None = None) -> int:
        """Append one composite frame. Raw bytes may be passed (tests), then frame_index/flags
        must be given for the index. Returns the CAU's absolute offset."""
        assert self._begun, "begin() first"
        offset = self.f.tell()
        if isinstance(cau, Cau):
            data = cau.pack()
            frame_index = cau.frame_index if frame_index is None else frame_index
            flags = cau.flags if flags is None else flags
        else:
            data = cau
        self.f.write(data)
        idx_flags = INDEX_FLAG_RAP if (flags or 0) & CAU_FLAG_RAP else 0
        self.index.append(IndexEntry(int(frame_index or 0), idx_flags, offset))
        return offset

    def finish(self, index_bytes: bytes | None = None, meta: dict | bytes | None = None) -> None:
        """Write the index and back-patch the header. `meta` replaces the metadata written by
        begin() (it must fit in the reserved region) — used by streaming writers that only know
        the final frame count at the end."""
        self.header.index_offset = self.f.tell()
        self.f.write(pack_index(self.index) if index_bytes is None else index_bytes)
        if meta is not None:
            data = meta if isinstance(meta, bytes) else encode_meta(meta)
            capacity = self.header.resources_offset - self.header.meta_offset
            if len(data) > capacity:
                raise ValueError(f"metadata ({len(data)} bytes) exceeds the reserved {capacity} bytes")
            self.f.seek(self.header.meta_offset)
            self.f.write(data + b" " * (capacity - len(data)))
            self.header.meta_length = len(data)
        self.f.seek(0)
        self.f.write(self.header.pack())
        self.f.close()

    def abort(self) -> None:
        self.f.close()
        try:
            os.remove(self.path)
        except OSError:
            pass


class LVFReader:
    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self.f: BinaryIO = open(self.path, "rb")
        self.file_size = os.fstat(self.f.fileno()).st_size
        self.header = FileHeader.unpack(self.read(0, HEADER_SIZE))
        self._meta: dict | None = None

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "LVFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self, offset: int, length: int) -> bytes:
        self.f.seek(offset)
        data = self.f.read(length)
        if len(data) != length:
            raise FormatError(f"short read: wanted {length} bytes at {offset}, file has {self.file_size}")
        return data

    # ---- metadata / resources -------------------------------------------------
    def meta_bytes(self) -> bytes:
        return self.read(self.header.meta_offset, self.header.meta_length)

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = json.loads(self.meta_bytes().decode("utf-8"))
        return self._meta

    def resource(self, offset: int, length: int) -> bytes:
        return self.read(self.header.resources_offset + offset, length)

    # ---- index -----------------------------------------------------------------
    def index(self) -> tuple[bytes, int, list[IndexEntry]]:
        return unpack_index(self.read(self.header.index_offset, self.file_size - self.header.index_offset))

    # ---- composite frames ----------------------------------------------------------
    def cau_at(self, offset: int) -> tuple[Cau, int]:
        size = peek_cau_size(self.read(offset, 8))
        if offset + size > self.file_size:
            raise FormatError(f"CAU at {offset} claims {size} bytes, past end of file")
        return Cau.unpack(self.read(offset, size))

    def iter_caus(self, start: int | None = None, end: int | None = None) -> Iterator[tuple[int, Cau, int]]:
        """Walk the CAU region sequentially (independent of the index).
        Yields (offset, cau, size). Raises FormatError on structural damage."""
        pos = self.header.cau_offset if start is None else start
        end = self.header.index_offset if end is None else end
        while pos < end:
            if pos + 8 > end:
                raise FormatError(f"{end - pos} stray bytes before the index at offset {pos}")
            size = peek_cau_size(self.read(pos, 8))
            if pos + size > end:
                raise FormatError(f"CAU at offset {pos} ({size} bytes) runs past index_offset {end}")
            cau, size = Cau.unpack(self.read(pos, size))
            yield pos, cau, size
            pos += size
