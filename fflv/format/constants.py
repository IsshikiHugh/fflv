"""Constants of the LVF v1 container (see LVF_SPEC.md). Files use the extension .lvd."""

MAGIC_FILE = b"LVF1"
MAGIC_CAU = b"CAUF"
MAGIC_INDEX = b"IDX1"

VERSION = 1

HEADER_SIZE = 64
CAU_HEADER_SIZE = 20
VIDEO_ENTRY_HEADER_SIZE = 12
AUDIO_PACKET_HEADER_SIZE = 16
INDEX_HEADER_SIZE = 8
INDEX_ENTRY_SIZE = 16

# payload_size counts bytes from CAU offset 8 to the end of the CAU.
CAU_PAYLOAD_BASE = 8

# Video entry types (5.2)
ENTRY_EMPTY = 0
ENTRY_FRAME = 1
ENTRY_HOLD = 2  # reserved for a future version; invalid in v1

# Flag bits
CAU_FLAG_RAP = 0x01
FRAME_FLAG_KEY = 0x01
INDEX_FLAG_RAP = 0x01

FILE_EXTENSION = ".lvd"

LAYER_KINDS = ("video", "still")
ALPHA_RANGES = ("limited", "full")
BLEND_MODES = ("normal", "add", "multiply", "screen")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

OPUS_SAMPLE_RATE = 48000
