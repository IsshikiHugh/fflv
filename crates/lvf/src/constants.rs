//! Constants of the LVF v1 container (LVF_SPEC.md sections 3–7). Files use the extension `.lvd`.

pub const MAGIC_FILE: &[u8; 4] = b"LVF1";
pub const MAGIC_CAU: &[u8; 4] = b"CAUF";
pub const MAGIC_INDEX: &[u8; 4] = b"IDX1";
pub const VERSION: u16 = 1;
pub const FILE_EXTENSION: &str = "lvd";

pub const HEADER_SIZE: usize = 64;
pub const CAU_HEADER_SIZE: usize = 20;
pub const VIDEO_ENTRY_HEADER_SIZE: usize = 12;
pub const AUDIO_PACKET_HEADER_SIZE: usize = 16;
pub const INDEX_HEADER_SIZE: usize = 8;
pub const INDEX_ENTRY_SIZE: usize = 16;
/// `payload_size` counts the bytes from offset 8 of a composite frame to its end.
pub const CAU_PAYLOAD_BASE: usize = 8;

pub const ENTRY_EMPTY: u8 = 0;
pub const ENTRY_FRAME: u8 = 1;
/// Reserved for a future version; invalid in v1.
pub const ENTRY_HOLD: u8 = 2;

pub const CAU_FLAG_RAP: u8 = 0x01;
pub const FRAME_FLAG_KEY: u8 = 0x01;
pub const INDEX_FLAG_RAP: u8 = 0x01;

pub const BLEND_MODES: [&str; 4] = ["normal", "add", "multiply", "screen"];
pub const ALPHA_RANGES: [&str; 2] = ["limited", "full"];
pub const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
pub const OPUS_SAMPLE_RATE: u32 = 48_000;
