//! The Opus audio track (spec 8.2 step 4): FFmpeg transcodes any source to 48 kHz Opus in Ogg; the
//! Ogg pages are parsed here. Packets are stored on the decoder timeline (first packet at 0), the
//! Opus pre-skip goes into the metadata (spec A.4).

use std::path::Path;

use lvf::constants::OPUS_SAMPLE_RATE;
use lvf::meta::AudioMeta;
use lvf::validate::{base64_decode, base64_encode};
use lvf::AudioPacket;

use crate::error::{Error, Result};
use crate::media::{ffmpeg, run};

#[derive(Clone, Debug)]
pub struct AudioTrack {
    pub packets: Vec<AudioPacket>,
    /// The OpusHead packet (WebCodecs `description`).
    pub extradata: Vec<u8>,
    pub channels: u32,
    pub pre_skip: u32,
}

impl AudioTrack {
    pub fn meta(&self) -> AudioMeta {
        AudioMeta {
            codec: "opus".into(),
            sample_rate: OPUS_SAMPLE_RATE,
            channels: self.channels,
            description_b64: Some(base64_encode(&self.extradata)),
            pre_skip: Some(self.pre_skip),
            extra: Default::default(),
        }
    }

    /// Only the packets that start before `end_us` (the end of the video).
    pub fn truncate_to(&mut self, end_us: i64) {
        self.packets.retain(|p| p.pts_us < end_us);
    }

    pub fn from_meta(meta: &AudioMeta, packets: Vec<AudioPacket>) -> AudioTrack {
        let extradata = meta.description_b64.as_deref().and_then(base64_decode).unwrap_or_default();
        AudioTrack { packets, extradata, channels: meta.channels, pre_skip: meta.pre_skip.unwrap_or(0) }
    }
}

/// 48 kHz samples → microseconds, round half up.
pub fn samples_to_us(samples: u64) -> i64 {
    ((samples * 2_000_000 + OPUS_SAMPLE_RATE as u64) / (2 * OPUS_SAMPLE_RATE as u64)) as i64
}

/// Duration of an Opus packet in 48 kHz samples, from its TOC byte (RFC 6716 section 3.1).
pub fn opus_packet_samples(p: &[u8]) -> Result<u32> {
    let toc = *p.first().ok_or_else(|| Error::Media("empty Opus packet".into()))?;
    let config = toc >> 3;
    let frame = match config {
        0..=11 => [480, 960, 1920, 2880][(config % 4) as usize],
        12..=15 => [480, 960][(config % 2) as usize],
        _ => [120, 240, 480, 960][(config % 4) as usize],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => (*p.get(1).ok_or_else(|| Error::Media("truncated Opus packet".into()))? & 0x3f) as u32,
    };
    Ok(frame * frames)
}

/// The packets of the first logical stream of an Ogg file, and the granule position of its
/// end-of-stream page (if it has one).
pub fn ogg_packets(data: &[u8]) -> Result<(Vec<Vec<u8>>, Option<i64>)> {
    let bad = |m: &str| Error::Media(format!("Ogg stream: {m}"));
    let mut packets = Vec::new();
    let mut partial: Vec<u8> = Vec::new();
    let mut serial = None;
    let mut eos_granule = None;
    let mut pos = 0;
    while pos < data.len() {
        let page = &data[pos..];
        if page.len() < 27 || &page[..4] != b"OggS" {
            return Err(bad(&format!("no page at offset {pos}")));
        }
        let n_segments = page[26] as usize;
        if page.len() < 27 + n_segments {
            return Err(bad("truncated page header"));
        }
        let lacing = &page[27..27 + n_segments];
        let body_len: usize = lacing.iter().map(|&l| l as usize).sum();
        let body_start = 27 + n_segments;
        if page.len() < body_start + body_len {
            return Err(bad("truncated page"));
        }
        let page_serial = u32::from_le_bytes(page[14..18].try_into().unwrap());
        if *serial.get_or_insert(page_serial) == page_serial {
            if page[5] & 4 != 0 {
                eos_granule = Some(i64::from_le_bytes(page[6..14].try_into().unwrap()));
            }
            let mut off = body_start;
            for &l in lacing {
                partial.extend_from_slice(&page[off..off + l as usize]);
                off += l as usize;
                if l < 255 {
                    packets.push(std::mem::take(&mut partial));
                }
            }
        }
        pos += body_start + body_len;
    }
    Ok((packets, eos_granule))
}

/// Opus track from Ogg bytes: OpusHead, OpusTags, then audio packets.
pub fn parse_ogg_opus(data: &[u8]) -> Result<AudioTrack> {
    let (packets, eos_granule) = ogg_packets(data)?;
    let mut packets = packets.into_iter();
    let head = packets.next().ok_or_else(|| Error::Media("Opus stream is empty".into()))?;
    if head.len() < 19 || &head[..8] != b"OpusHead" {
        return Err(Error::Media("Opus stream has no OpusHead".into()));
    }
    let channels = head[9] as u32;
    let pre_skip = u16::from_le_bytes([head[10], head[11]]) as u32;
    if !packets.next().is_some_and(|t| t.starts_with(b"OpusTags")) {
        return Err(Error::Media("Opus stream has no OpusTags".into()));
    }
    let mut out = Vec::new();
    let mut t = 0u64; // decoder timeline, in samples
    for p in packets.filter(|p| !p.is_empty()) {
        let n = opus_packet_samples(&p)? as u64;
        out.push(AudioPacket { pts_us: samples_to_us(t), duration_us: samples_to_us(n) as u32, data: p });
        t += n;
    }
    // The end-of-stream granule position trims the last packet (like FFmpeg's Ogg demuxer).
    if let (Some(g), Some(last)) = (eos_granule, out.last_mut()) {
        let start = t - opus_packet_samples(&last.data)? as u64;
        if g > start as i64 && (g as u64) < t {
            last.duration_us = samples_to_us(g as u64 - start) as u32;
        }
    }
    Ok(AudioTrack { packets: out, extradata: head, channels, pre_skip })
}

/// Transcode any audio source to 48 kHz Opus.
pub fn encode_audio(src: &Path, max_seconds: Option<f64>, bitrate: &str, channels: u32) -> Result<AudioTrack> {
    if !src.exists() {
        return Err(Error::Media(format!("audio source not found: {}", src.display())));
    }
    let name = src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut cmd = ffmpeg();
    cmd.arg("-i").arg(src);
    cmd.args(["-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "libopus", "-b:a", bitrate]);
    cmd.args(["-ar", &OPUS_SAMPLE_RATE.to_string(), "-ac", &channels.to_string()]);
    if let Some(s) = max_seconds {
        cmd.args(["-t", &format!("{s:.6}")]);
    }
    cmd.args(["-f", "ogg", "-"]);
    let ogg = run(&mut cmd, &format!("audio transcode of {name}"))?;
    parse_ogg_opus(&ogg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toc_durations() {
        assert_eq!(opus_packet_samples(&[0xfc]).unwrap(), 960); // CELT 20 ms, one frame
        assert_eq!(opus_packet_samples(&[0xf8 | 1]).unwrap(), 1920); // two frames
        assert_eq!(opus_packet_samples(&[0x03, 0x03]).unwrap(), 480 * 3); // SILK 10 ms, code 3, 3 frames
        assert_eq!(opus_packet_samples(&[0x78]).unwrap(), 960); // hybrid 20 ms
    }

    #[test]
    fn microseconds_round_half_up() {
        assert_eq!(samples_to_us(960), 20_000);
        assert_eq!(samples_to_us(1), 21); // 20.83
        assert_eq!(samples_to_us(3), 63); // 62.5 → 63
    }
}
