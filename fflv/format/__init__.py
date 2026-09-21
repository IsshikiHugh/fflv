"""The LVF v1 container (.lvd files): binary structures, reader/writer, validator."""

from .binary import AudioPacket, Cau, FileHeader, FormatError, IndexEntry, VideoEntry
from .container import LVFReader, LVFWriter, encode_meta, meta_capacity_for, rewrite_meta_in_place
from .timing import parse_fps, pts_us, seconds_to_frame
from .validate import INVARIANTS, Report, validate

__all__ = [
    "AudioPacket", "Cau", "FileHeader", "FormatError", "IndexEntry", "VideoEntry",
    "LVFReader", "LVFWriter", "encode_meta", "meta_capacity_for", "rewrite_meta_in_place",
    "parse_fps", "pts_us", "seconds_to_frame", "INVARIANTS", "Report", "validate",
]
