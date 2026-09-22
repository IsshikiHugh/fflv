//! Minimal VP9 bitstream inspection: superframe split + uncompressed header (VP9 bitstream spec
//! v0.6, section 6.2 and Annex B). Used to double-check encoder output ("do not trust the encoder"):
//! key-frame-ness, one shown frame per packet, coded size and format of key frames.

pub const KEY_FRAME: u32 = 0;
pub const CS_RGB: u32 = 7;
const SYNC_CODE: [u32; 3] = [0x49, 0x83, 0x42];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp9Frame {
    pub profile: u32,
    pub show_existing_frame: bool,
    pub key_frame: bool,
    pub show_frame: bool,
    /// Only known for key frames:
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bit_depth: Option<u32>,
    pub color_space: Option<u32>,
    /// 0 = studio (limited), 1 = full.
    pub color_range: Option<u32>,
    pub subsampling: Option<(u32, u32)>,
}

impl Vp9Frame {
    pub fn shown(&self) -> bool {
        self.show_existing_frame || self.show_frame
    }
}

#[derive(Debug, Clone)]
pub struct Vp9Packet {
    pub frames: Vec<Vp9Frame>,
    pub superframe: bool,
}

impl Vp9Packet {
    /// A packet is decodable on its own iff its first frame is a key frame.
    pub fn key_frame(&self) -> bool {
        self.frames[0].key_frame
    }
    pub fn shown_count(&self) -> usize {
        self.frames.iter().filter(|f| f.shown()).count()
    }
    pub fn key_info(&self) -> Option<&Vp9Frame> {
        self.frames.iter().find(|f| f.key_frame)
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn f(&mut self, n: u32) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.pos >> 3;
            if byte >= self.data.len() {
                return Err("uncompressed header truncated".into());
            }
            v = (v << 1) | ((self.data[byte] >> (7 - (self.pos & 7))) & 1) as u32;
            self.pos += 1;
        }
        Ok(v)
    }
}

/// Annex B: (frames, is_superframe).
pub fn split_superframe(data: &[u8]) -> Result<(Vec<&[u8]>, bool), String> {
    let Some(&marker) = data.last() else { return Err("empty packet".into()) };
    if marker & 0xE0 == 0xC0 {
        let frames = (marker & 0x7) as usize + 1;
        let mag = ((marker >> 3) & 0x3) as usize + 1;
        let index_sz = 2 + mag * frames;
        if data.len() >= index_sz && data[data.len() - index_sz] == marker {
            let mut p = data.len() - index_sz + 1;
            let mut sizes = Vec::with_capacity(frames);
            for _ in 0..frames {
                let mut s = 0usize;
                for k in 0..mag {
                    s |= (data[p + k] as usize) << (8 * k);
                }
                sizes.push(s);
                p += mag;
            }
            let mut out = Vec::with_capacity(frames);
            let mut off = 0;
            for s in sizes {
                if off + s > data.len() - index_sz {
                    return Err("superframe index points past the data".into());
                }
                if s > 0 {
                    out.push(&data[off..off + s]);
                }
                off += s;
            }
            return Ok((out, true));
        }
    }
    Ok((vec![data], false))
}

pub fn parse_frame(data: &[u8]) -> Result<Vp9Frame, String> {
    let mut b = Bits { data, pos: 0 };
    if b.f(2)? != 2 {
        return Err("frame_marker is not 2 (not a VP9 frame)".into());
    }
    let low = b.f(1)?;
    let high = b.f(1)?;
    let profile = (high << 1) | low;
    if profile == 3 {
        b.f(1)?;
    }
    let mut fr = Vp9Frame {
        profile,
        show_existing_frame: false,
        key_frame: false,
        show_frame: true,
        width: None,
        height: None,
        bit_depth: None,
        color_space: None,
        color_range: None,
        subsampling: None,
    };
    if b.f(1)? == 1 {
        b.f(3)?;
        fr.show_existing_frame = true;
        return Ok(fr);
    }
    let frame_type = b.f(1)?;
    fr.show_frame = b.f(1)? == 1;
    b.f(1)?; // error_resilient_mode
    fr.key_frame = frame_type == KEY_FRAME;
    if !fr.key_frame {
        return Ok(fr);
    }
    if [b.f(8)?, b.f(8)?, b.f(8)?] != SYNC_CODE {
        return Err("key frame without the VP9 sync code".into());
    }
    let mut bit_depth = 8;
    if profile >= 2 {
        bit_depth = if b.f(1)? == 1 { 12 } else { 10 };
    }
    let color_space = b.f(3)?;
    let (color_range, sub);
    if color_space != CS_RGB {
        color_range = b.f(1)?;
        if profile == 1 || profile == 3 {
            sub = (b.f(1)?, b.f(1)?);
            b.f(1)?;
        } else {
            sub = (1, 1);
        }
    } else {
        color_range = 1;
        sub = (0, 0);
        if profile == 1 || profile == 3 {
            b.f(1)?;
        }
    }
    fr.bit_depth = Some(bit_depth);
    fr.color_space = Some(color_space);
    fr.color_range = Some(color_range);
    fr.subsampling = Some(sub);
    fr.width = Some(b.f(16)? + 1);
    fr.height = Some(b.f(16)? + 1);
    Ok(fr)
}

pub fn inspect_packet(data: &[u8]) -> Result<Vp9Packet, String> {
    let (frames, superframe) = split_superframe(data)?;
    if frames.is_empty() {
        return Err("superframe without frames".into());
    }
    let frames = frames.into_iter().map(parse_frame).collect::<Result<Vec<_>, _>>()?;
    Ok(Vp9Packet { frames, superframe })
}

// ------------------------------------------------------------------------------------------------
// WebCodecs codec strings
// ------------------------------------------------------------------------------------------------
/// (level, max luma picture size, max luma sample rate) — webmproject.org/vp9/levels
const LEVELS: [(u32, u64, u64); 14] = [
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
];

/// The smallest VP9 level that fits the picture size and luma sample rate.
pub fn vp9_level(width: u32, height: u32, fps: f64) -> u32 {
    let size = width as u64 * height as u64;
    let rate = size as f64 * fps;
    LEVELS.iter().find(|(_, s, r)| size <= *s && rate <= *r as f64).map(|l| l.0).unwrap_or(62)
}

/// Regular planes: profile 0, 8-bit 4:2:0 ("vp09.00.LL.08"). Lossless color planes: profile 1,
/// 8-bit 4:4:4 RGB (identity matrix, BT.709 primaries, sRGB transfer, full range).
pub fn codec_string(width: u32, height: u32, fps: f64, lossless: bool) -> String {
    let level = vp9_level(width, height, fps);
    if lossless {
        format!("vp09.01.{level:02}.08.03.01.13.00.01")
    } else {
        format!("vp09.00.{level:02}.08")
    }
}

/// Profile number of a vp09.* codec string.
pub fn codec_profile(codec: &str) -> Option<u32> {
    let parts: Vec<&str> = codec.split('.').collect();
    if parts.len() < 4 || parts[0] != "vp09" {
        return None;
    }
    parts[1].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_string_levels() {
        assert_eq!(codec_string(1920, 1080, 30.0, false), "vp09.00.40.08");
        assert_eq!(codec_string(1920, 1080, 60.0, false), "vp09.00.41.08");
        assert_eq!(codec_string(1280, 720, 30.0, false), "vp09.00.31.08");
        assert_eq!(codec_string(640, 360, 30.0, false), "vp09.00.21.08");
        assert_eq!(codec_string(3840, 2160, 30.0, false), "vp09.00.50.08");
        assert_eq!(codec_string(64, 64, 30.0, false), "vp09.00.10.08");
        assert_eq!(codec_string(1280, 720, 30.0, true), "vp09.01.31.08.03.01.13.00.01");
        assert_eq!(codec_profile("vp09.01.31.08.03.01.13.00.01"), Some(1));
        assert_eq!(codec_profile("vp09.00.40.08"), Some(0));
        assert_eq!(codec_profile("avc1.42E01E"), None);
    }

    #[test]
    fn superframe_split() {
        let (a, b) = (&b"\x82\x01\x02"[..], &b"\x86\x05"[..]);
        let marker = 0xC1u8; // 2 frames, 1-byte sizes
        let mut data = [a, b].concat();
        data.extend_from_slice(&[marker, a.len() as u8, b.len() as u8, marker]);
        let (frames, sf) = split_superframe(&data).unwrap();
        assert!(sf && frames == vec![a, b]);
        let (frames, sf) = split_superframe(a).unwrap();
        assert!(!sf && frames == vec![a]);
    }
}
