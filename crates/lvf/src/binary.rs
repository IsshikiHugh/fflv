//! Binary structures of LVF v1: file header, composite frames (CAU), index. Little-endian.
//!
//! Packing writes exactly what it is given (it does not enforce the format invariants), so tests
//! can build deliberately broken files; the invariants live in [`crate::validate`].

use std::io::Write;

use crate::constants::*;
use crate::error::{format_err, Error, Result};

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

// ------------------------------------------------------------------------------------------------
// File header
// ------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub meta_offset: u64,
    pub meta_length: u32,
    pub reserved_20: u32,
    pub resources_offset: u64,
    pub cau_offset: u64,
    pub index_offset: u64,
    pub reserved_48: [u8; 16],
}

impl Default for FileHeader {
    fn default() -> Self {
        FileHeader {
            magic: *MAGIC_FILE,
            version: VERSION,
            flags: 0,
            meta_offset: HEADER_SIZE as u64,
            meta_length: 0,
            reserved_20: 0,
            resources_offset: 0,
            cau_offset: 0,
            index_offset: 0,
            reserved_48: [0; 16],
        }
    }
}

impl FileHeader {
    pub fn pack(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..4].copy_from_slice(&self.magic);
        b[4..6].copy_from_slice(&self.version.to_le_bytes());
        b[6..8].copy_from_slice(&self.flags.to_le_bytes());
        b[8..16].copy_from_slice(&self.meta_offset.to_le_bytes());
        b[16..20].copy_from_slice(&self.meta_length.to_le_bytes());
        b[20..24].copy_from_slice(&self.reserved_20.to_le_bytes());
        b[24..32].copy_from_slice(&self.resources_offset.to_le_bytes());
        b[32..40].copy_from_slice(&self.cau_offset.to_le_bytes());
        b[40..48].copy_from_slice(&self.index_offset.to_le_bytes());
        b[48..64].copy_from_slice(&self.reserved_48);
        b
    }

    pub fn unpack(b: &[u8]) -> Result<FileHeader> {
        if b.len() < HEADER_SIZE {
            return format_err(format!("file header needs {HEADER_SIZE} bytes, got {}", b.len()));
        }
        Ok(FileHeader {
            magic: b[0..4].try_into().unwrap(),
            version: u16_at(b, 4),
            flags: u16_at(b, 6),
            meta_offset: u64_at(b, 8),
            meta_length: u32_at(b, 16),
            reserved_20: u32_at(b, 20),
            resources_offset: u64_at(b, 24),
            cau_offset: u64_at(b, 32),
            index_offset: u64_at(b, 40),
            reserved_48: b[48..64].try_into().unwrap(),
        })
    }
}

// ------------------------------------------------------------------------------------------------
// Composite frame
// ------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VideoEntry {
    pub layer_index: u16,
    pub kind: u8,
    pub frame_flags: u8,
    pub color: Vec<u8>,
    pub alpha: Vec<u8>,
}

impl VideoEntry {
    pub fn empty(layer_index: u16) -> Self {
        VideoEntry { layer_index, kind: ENTRY_EMPTY, ..Default::default() }
    }
    pub fn frame(layer_index: u16, key: bool, color: Vec<u8>, alpha: Vec<u8>) -> Self {
        VideoEntry { layer_index, kind: ENTRY_FRAME, frame_flags: if key { FRAME_FLAG_KEY } else { 0 }, color, alpha }
    }
    pub fn is_key(&self) -> bool {
        self.frame_flags & FRAME_FLAG_KEY != 0
    }
    pub fn is_frame(&self) -> bool {
        self.kind == ENTRY_FRAME
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPacket {
    pub pts_us: i64,
    pub duration_us: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cau {
    pub frame_index: u32,
    pub flags: u8,
    pub entries: Vec<VideoEntry>,
    pub audio: Vec<AudioPacket>,
    /// Raw header fields, kept so the validator can report them verbatim.
    pub magic: [u8; 4],
    pub reserved_13: u8,
    pub reserved_18: u16,
    /// `None`: derive from `entries` when packing.
    pub video_entry_count: Option<u16>,
}

impl Cau {
    pub fn new(frame_index: u32, rap: bool, entries: Vec<VideoEntry>, audio: Vec<AudioPacket>) -> Cau {
        Cau {
            frame_index,
            flags: if rap { CAU_FLAG_RAP } else { 0 },
            entries,
            audio,
            magic: *MAGIC_CAU,
            reserved_13: 0,
            reserved_18: 0,
            video_entry_count: None,
        }
    }

    pub fn is_rap(&self) -> bool {
        self.flags & CAU_FLAG_RAP != 0
    }

    pub fn packed_len(&self) -> usize {
        CAU_HEADER_SIZE
            + self.entries.iter().map(|e| VIDEO_ENTRY_HEADER_SIZE + e.color.len() + e.alpha.len()).sum::<usize>()
            + self.audio.iter().map(|a| AUDIO_PACKET_HEADER_SIZE + a.data.len()).sum::<usize>()
    }

    /// The packed bytes. Errors (writing nothing) when a count or size does not fit its field.
    pub fn pack(&self) -> Result<Vec<u8>> {
        let mut b = Vec::with_capacity(self.packed_len());
        self.write_to(&mut b)?;
        Ok(b)
    }

    /// Write the packed bytes straight to `w` (no intermediate copy of the payloads); returns the
    /// number of bytes written. The field ranges are checked before anything is written.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<usize> {
        let total = self.packed_len();
        let too_big =
            |what: &str, n: usize| Error::Value(format!("CAU {}: {what} ({n}) does not fit", self.frame_index));
        let payload = u32::try_from(total - CAU_PAYLOAD_BASE).map_err(|_| too_big("payload size", total))?;
        let count = match self.video_entry_count {
            Some(n) => n,
            None => u16::try_from(self.entries.len()).map_err(|_| too_big("video entry count", self.entries.len()))?,
        };
        let n_audio = u16::try_from(self.audio.len()).map_err(|_| too_big("audio packet count", self.audio.len()))?;
        // every data length is at most the payload size, which fits in a u32
        let mut h = [0u8; CAU_HEADER_SIZE];
        h[0..4].copy_from_slice(&self.magic);
        h[4..8].copy_from_slice(&payload.to_le_bytes());
        h[8..12].copy_from_slice(&self.frame_index.to_le_bytes());
        h[12] = self.flags;
        h[13] = self.reserved_13;
        h[14..16].copy_from_slice(&count.to_le_bytes());
        h[16..18].copy_from_slice(&n_audio.to_le_bytes());
        h[18..20].copy_from_slice(&self.reserved_18.to_le_bytes());
        w.write_all(&h)?;
        for e in &self.entries {
            let mut h = [0u8; VIDEO_ENTRY_HEADER_SIZE];
            h[0..2].copy_from_slice(&e.layer_index.to_le_bytes());
            h[2] = e.kind;
            h[3] = e.frame_flags;
            h[4..8].copy_from_slice(&(e.color.len() as u32).to_le_bytes());
            h[8..12].copy_from_slice(&(e.alpha.len() as u32).to_le_bytes());
            w.write_all(&h)?;
            w.write_all(&e.color)?;
            w.write_all(&e.alpha)?;
        }
        for a in &self.audio {
            let mut h = [0u8; AUDIO_PACKET_HEADER_SIZE];
            h[0..8].copy_from_slice(&a.pts_us.to_le_bytes());
            h[8..12].copy_from_slice(&a.duration_us.to_le_bytes());
            h[12..16].copy_from_slice(&(a.data.len() as u32).to_le_bytes());
            w.write_all(&h)?;
            w.write_all(&a.data)?;
        }
        Ok(total)
    }

    /// Parse one composite frame starting at `offset`. Returns (cau, total size in bytes).
    pub fn unpack(buf: &[u8], offset: usize) -> Result<(Cau, usize)> {
        let (cau, size) = Cau::unpack_ref(buf, offset)?;
        Ok((cau.to_cau(), size))
    }

    /// Like [`Cau::unpack`], borrowing the payloads from `buf` instead of copying them.
    pub fn unpack_ref(buf: &[u8], offset: usize) -> Result<(CauRef<'_>, usize)> {
        let end_of_buf = buf.len();
        if offset > end_of_buf || end_of_buf - offset < CAU_HEADER_SIZE {
            return format_err(format!("truncated CAU header at offset {offset}"));
        }
        let h = &buf[offset..];
        let magic: [u8; 4] = h[0..4].try_into().unwrap();
        if &magic != MAGIC_CAU {
            return format_err(format!("bad CAU magic {magic:?} at offset {offset}"));
        }
        let payload_size = u32_at(h, 4) as usize;
        let frame_index = u32_at(h, 8);
        let flags = h[12];
        let reserved_13 = h[13];
        let n_video = u16_at(h, 14);
        let n_audio = u16_at(h, 16);
        let reserved_18 = u16_at(h, 18);
        let total = CAU_PAYLOAD_BASE + payload_size;
        if payload_size < CAU_HEADER_SIZE - CAU_PAYLOAD_BASE || total > end_of_buf - offset {
            return format_err(format!("CAU at offset {offset} has invalid payload_size {payload_size}"));
        }
        let end = offset + total;
        let mut pos = offset + CAU_HEADER_SIZE;
        // The counts are untrusted: reserve no more entries than the remaining bytes can hold.
        let mut entries = Vec::with_capacity((n_video as usize).min((end - pos) / VIDEO_ENTRY_HEADER_SIZE));
        for _ in 0..n_video {
            if end - pos < VIDEO_ENTRY_HEADER_SIZE {
                return format_err(format!("CAU {frame_index}: video entry header overruns the CAU"));
            }
            let layer_index = u16_at(buf, pos);
            let kind = buf[pos + 2];
            let frame_flags = buf[pos + 3];
            let color_len = u32_at(buf, pos + 4) as usize;
            let alpha_len = u32_at(buf, pos + 8) as usize;
            pos += VIDEO_ENTRY_HEADER_SIZE;
            if color_len > end - pos || alpha_len > end - pos - color_len {
                return format_err(format!("CAU {frame_index}: layer {layer_index} data overruns the CAU"));
            }
            let color = &buf[pos..pos + color_len];
            pos += color_len;
            let alpha = &buf[pos..pos + alpha_len];
            pos += alpha_len;
            entries.push(VideoEntryRef { layer_index, kind, frame_flags, color, alpha });
        }
        let mut audio = Vec::with_capacity((n_audio as usize).min((end - pos) / AUDIO_PACKET_HEADER_SIZE));
        for _ in 0..n_audio {
            if end - pos < AUDIO_PACKET_HEADER_SIZE {
                return format_err(format!("CAU {frame_index}: audio packet header overruns the CAU"));
            }
            let pts_us = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            let duration_us = u32_at(buf, pos + 8);
            let len = u32_at(buf, pos + 12) as usize;
            pos += AUDIO_PACKET_HEADER_SIZE;
            if len > end - pos {
                return format_err(format!("CAU {frame_index}: audio packet data overruns the CAU"));
            }
            audio.push(AudioPacketRef { pts_us, duration_us, data: &buf[pos..pos + len] });
            pos += len;
        }
        if pos != end {
            return format_err(format!("CAU {frame_index}: {} trailing bytes after the declared entries", end - pos));
        }
        let cau = CauRef {
            frame_index,
            flags,
            entries,
            audio,
            magic,
            reserved_13,
            reserved_18,
            video_entry_count: n_video,
            raw: &buf[offset..end],
        };
        Ok((cau, total))
    }
}

/// A video entry whose data is borrowed from the buffer it was parsed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoEntryRef<'a> {
    pub layer_index: u16,
    pub kind: u8,
    pub frame_flags: u8,
    pub color: &'a [u8],
    pub alpha: &'a [u8],
}

impl VideoEntryRef<'_> {
    pub fn is_key(&self) -> bool {
        self.frame_flags & FRAME_FLAG_KEY != 0
    }
    pub fn is_frame(&self) -> bool {
        self.kind == ENTRY_FRAME
    }
    pub fn to_entry(&self) -> VideoEntry {
        let (layer_index, kind, frame_flags) = (self.layer_index, self.kind, self.frame_flags);
        VideoEntry { layer_index, kind, frame_flags, color: self.color.to_vec(), alpha: self.alpha.to_vec() }
    }
}

/// An audio packet whose data is borrowed from the buffer it was parsed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioPacketRef<'a> {
    pub pts_us: i64,
    pub duration_us: u32,
    pub data: &'a [u8],
}

impl AudioPacketRef<'_> {
    pub fn to_packet(&self) -> AudioPacket {
        AudioPacket { pts_us: self.pts_us, duration_us: self.duration_us, data: self.data.to_vec() }
    }
}

/// A parsed composite frame that borrows its payloads ([`Cau::unpack_ref`]): only the entry
/// headers are allocated, so a reader can copy just the layers it needs (or none).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CauRef<'a> {
    pub frame_index: u32,
    pub flags: u8,
    pub entries: Vec<VideoEntryRef<'a>>,
    pub audio: Vec<AudioPacketRef<'a>>,
    pub magic: [u8; 4],
    pub reserved_13: u8,
    pub reserved_18: u16,
    /// As stored (it equals `entries.len()` once parsed).
    pub video_entry_count: u16,
    /// The whole composite frame as stored (header included), e.g. to copy it unchanged.
    pub raw: &'a [u8],
}

impl CauRef<'_> {
    pub fn is_rap(&self) -> bool {
        self.flags & CAU_FLAG_RAP != 0
    }

    /// An owned copy.
    pub fn to_cau(&self) -> Cau {
        Cau {
            frame_index: self.frame_index,
            flags: self.flags,
            entries: self.entries.iter().map(VideoEntryRef::to_entry).collect(),
            audio: self.audio.iter().map(AudioPacketRef::to_packet).collect(),
            magic: self.magic,
            reserved_13: self.reserved_13,
            reserved_18: self.reserved_18,
            video_entry_count: Some(self.video_entry_count),
        }
    }
}

/// Total size of the composite frame starting at `offset` (needs its first 8 bytes).
pub fn peek_cau_size(buf: &[u8], offset: usize) -> Result<usize> {
    if offset + 8 > buf.len() {
        return format_err(format!("truncated CAU header at offset {offset}"));
    }
    if &buf[offset..offset + 4] != MAGIC_CAU {
        return format_err(format!("bad CAU magic {:?} at offset {offset}", &buf[offset..offset + 4]));
    }
    Ok(CAU_PAYLOAD_BASE + u32_at(buf, offset + 4) as usize)
}

// ------------------------------------------------------------------------------------------------
// Index
// ------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    pub frame_index: u32,
    pub flags: u8,
    pub cau_offset: u64,
}

impl IndexEntry {
    pub fn is_rap(&self) -> bool {
        self.flags & INDEX_FLAG_RAP != 0
    }
}

pub fn pack_index(entries: &[IndexEntry], magic: &[u8; 4], count: Option<u32>) -> Vec<u8> {
    let mut b = Vec::with_capacity(INDEX_HEADER_SIZE + entries.len() * INDEX_ENTRY_SIZE);
    b.extend_from_slice(magic);
    b.extend_from_slice(&count.unwrap_or(entries.len() as u32).to_le_bytes());
    for e in entries {
        b.extend_from_slice(&e.frame_index.to_le_bytes());
        b.push(e.flags);
        b.extend_from_slice(&[0, 0, 0]);
        b.extend_from_slice(&e.cau_offset.to_le_bytes());
    }
    b
}

/// (magic, declared count, entries). Errors if the table is truncated.
pub fn unpack_index(b: &[u8]) -> Result<([u8; 4], u32, Vec<IndexEntry>)> {
    if b.len() < INDEX_HEADER_SIZE {
        return format_err("truncated index header");
    }
    let magic: [u8; 4] = b[0..4].try_into().unwrap();
    let count = u32_at(b, 4);
    let need = INDEX_HEADER_SIZE + count as usize * INDEX_ENTRY_SIZE;
    if b.len() < need {
        return format_err(format!("index declares {count} entries but only {} bytes are present", b.len()));
    }
    let entries = (0..count as usize)
        .map(|i| {
            let p = INDEX_HEADER_SIZE + i * INDEX_ENTRY_SIZE;
            IndexEntry { frame_index: u32_at(b, p), flags: b[p + 4], cau_offset: u64_at(b, p + 8) }
        })
        .collect();
    Ok((magic, count, entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = FileHeader {
            meta_length: 100,
            resources_offset: 164,
            cau_offset: 500,
            index_offset: 9000,
            ..Default::default()
        };
        let raw = h.pack();
        assert_eq!(&raw[0..4], b"LVF1");
        assert_eq!(&raw[4..6], &[1, 0]);
        assert_eq!(FileHeader::unpack(&raw).unwrap(), h);
    }

    #[test]
    fn cau_layout_matches_the_spec() {
        let cau = Cau::new(
            7,
            true,
            vec![VideoEntry::frame(0, true, b"abc".to_vec(), b"de".to_vec()), VideoEntry::empty(2)],
            vec![AudioPacket { pts_us: -5, duration_us: 20000, data: b"xyz".to_vec() }],
        );
        let raw = cau.pack().unwrap();
        assert_eq!(raw.len(), 20 + 17 + 12 + 19);
        assert_eq!(&raw[0..4], b"CAUF");
        assert_eq!(u32_at(&raw, 4) as usize, raw.len() - 8);
        assert_eq!(u32_at(&raw, 8), 7);
        assert_eq!(raw[12], 1);
        assert_eq!(u16_at(&raw, 14), 2);
        assert_eq!(u16_at(&raw, 16), 1);
        assert_eq!(peek_cau_size(&raw, 0).unwrap(), raw.len());
        let (back, size) = Cau::unpack(&raw, 0).unwrap();
        assert_eq!(size, raw.len());
        assert!(back.is_rap() && back.frame_index == 7);
        assert_eq!(back.entries, cau.entries);
        assert_eq!(back.audio, cau.audio);
        let mut streamed = Vec::new();
        assert_eq!(cau.write_to(&mut streamed).unwrap(), raw.len());
        assert_eq!(streamed, raw);
        let (r, size) = Cau::unpack_ref(&raw, 0).unwrap();
        assert_eq!((size, r.raw, r.video_entry_count), (raw.len(), &raw[..], 2));
        assert_eq!((r.entries[0].color, r.entries[0].alpha, r.audio[0].data), (&b"abc"[..], &b"de"[..], &b"xyz"[..]));
        assert_eq!(r.to_cau(), back);
    }

    #[test]
    fn cau_pack_rejects_counts_that_do_not_fit() {
        let cau = Cau::new(0, false, vec![VideoEntry::empty(0); 1 << 16], vec![]);
        assert!(cau.pack().is_err());
        let mut out = Vec::new();
        assert!(cau.write_to(&mut out).is_err() && out.is_empty(), "nothing is written");
        let audio = AudioPacket { pts_us: 0, duration_us: 0, data: vec![] };
        assert!(Cau::new(0, false, vec![], vec![audio; 1 << 16]).pack().is_err());
    }

    #[test]
    fn cau_unpack_rejects_damage() {
        let mut raw =
            Cau::new(0, false, vec![VideoEntry::frame(0, false, vec![b'a'; 10], vec![])], vec![]).pack().unwrap();
        raw[24..28].copy_from_slice(&1000u32.to_le_bytes()); // color_len too large
        assert!(Cau::unpack(&raw, 0).is_err());
        let mut raw = Cau::new(0, false, vec![], vec![]).pack().unwrap();
        raw[0..4].copy_from_slice(b"XXXX");
        assert!(Cau::unpack(&raw, 0).is_err());
        let mut raw = Cau::new(0, false, vec![], vec![]).pack().unwrap();
        raw.extend_from_slice(&[0, 0]);
        let n = (raw.len() - 8) as u32;
        raw[4..8].copy_from_slice(&n.to_le_bytes());
        assert!(Cau::unpack(&raw, 0).is_err(), "trailing bytes");
        let mut raw = Cau::new(0, false, vec![], vec![]).pack().unwrap();
        raw[14..16].copy_from_slice(&u16::MAX.to_le_bytes()); // a count with no bytes behind it
        assert!(Cau::unpack(&raw, 0).is_err());
        assert!(Cau::unpack(&raw, raw.len() + 1).is_err());
    }

    #[test]
    fn index_roundtrip() {
        let entries = vec![
            IndexEntry { frame_index: 0, flags: 1, cau_offset: 100 },
            IndexEntry { frame_index: 1, flags: 0, cau_offset: 250 },
        ];
        let raw = pack_index(&entries, MAGIC_INDEX, None);
        assert_eq!(raw.len(), 8 + 2 * 16);
        let (magic, count, back) = unpack_index(&raw).unwrap();
        assert_eq!((&magic, count, back), (MAGIC_INDEX, 2, entries));
        assert!(unpack_index(&raw[..raw.len() - 1]).is_err());
    }
}
