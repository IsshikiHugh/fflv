//! Validator: structure, metadata and the format invariants I1–I10 (spec 5.4), plus Appendix B.
//!
//! Every problem is an [`Issue`] tagged with a code. Invariant violations use the invariant's own
//! name ("I1" … "I10"); other codes: HDR (file header), META (metadata JSON), RES (resources),
//! CAU (composite-frame structure), VP9 (bitstream vs. metadata), AUD (audio).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::constants::*;
use crate::container::LvfReader;
use crate::error::Error;
use crate::meta::alpha_range_ok;
use crate::timing::{pts_us, Fps};
use crate::vp9::{codec_profile, inspect_packet, CS_RGB};

pub const INVARIANTS: [&str; 10] = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"];
const MAX_ISSUES_PER_CODE: usize = 25;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Issue {
    pub code: String,
    pub message: String,
    pub frame: Option<u32>,
    pub severity: String,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.frame {
            Some(fr) => write!(f, "[{}] frame {fr}: {}", self.code, self.message),
            None => write!(f, "[{}] {}", self.code, self.message),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct LayerStats {
    pub frames: u64,
    pub keyframes: u64,
    pub color_bytes: u64,
    pub alpha_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub path: String,
    pub issues: Vec<Issue>,
    /// "CODE/severity" → number of issues not recorded
    pub suppressed: BTreeMap<String, u64>,
    pub fatal: bool,
    pub meta: Option<Value>,
    pub cau_count: u64,
    pub rap_frames: Vec<u32>,
    pub layer_stats: BTreeMap<usize, LayerStats>,
    pub audio_packets: u64,
    pub audio_bytes: u64,
    pub file_size: u64,
}

impl Report {
    fn add(&mut self, code: &str, message: impl Into<String>, frame: Option<u32>, severity: &str) {
        let same = self.issues.iter().filter(|i| i.code == code && i.severity == severity).count();
        if same >= MAX_ISSUES_PER_CODE {
            *self.suppressed.entry(format!("{code}/{severity}")).or_default() += 1;
            return;
        }
        self.issues.push(Issue { code: code.into(), message: message.into(), frame, severity: severity.into() });
    }
    fn error(&mut self, code: &str, message: impl Into<String>, frame: Option<u32>) {
        self.add(code, message, frame, "error");
    }
    fn warn(&mut self, code: &str, message: impl Into<String>, frame: Option<u32>) {
        self.add(code, message, frame, "warning");
    }

    pub fn errors(&self) -> Vec<&Issue> {
        self.issues.iter().filter(|i| i.severity == "error").collect()
    }
    pub fn warnings(&self) -> Vec<&Issue> {
        self.issues.iter().filter(|i| i.severity == "warning").collect()
    }
    pub fn error_codes(&self) -> BTreeSet<String> {
        let mut codes: BTreeSet<String> = self.errors().iter().map(|i| i.code.clone()).collect();
        for k in self.suppressed.keys() {
            if let Some(code) = k.strip_suffix("/error") {
                codes.insert(code.to_string());
            }
        }
        codes
    }
    pub fn error_count(&self) -> u64 {
        self.errors().len() as u64
            + self.suppressed.iter().filter(|(k, _)| k.ends_with("/error")).map(|(_, n)| n).sum::<u64>()
    }
    pub fn ok(&self) -> bool {
        !self.fatal && self.error_codes().is_empty()
    }
}

fn is_int(v: &Value) -> bool {
    v.is_i64() || v.is_u64()
}
fn int(v: &Value) -> Option<i64> {
    if is_int(v) {
        v.as_i64()
    } else {
        None
    }
}
fn is_number(v: &Value) -> bool {
    v.is_number()
}

fn check_meta(meta: &Value, rep: &mut Report, resources_size: u64, reader: &LvfReader) -> bool {
    let mut ok = true;
    let mut bad = |rep: &mut Report, msg: String, fatal: bool| {
        rep.error("META", msg, None);
        if fatal {
            ok = false;
        }
    };
    if meta.get("format") != Some(&Value::from("LVF")) {
        bad(rep, format!("format must be \"LVF\", got {}", meta.get("format").unwrap_or(&Value::Null)), false);
    }
    if meta.get("version").and_then(int) != Some(VERSION as i64) {
        bad(rep, format!("version must be {VERSION}, got {}", meta.get("version").unwrap_or(&Value::Null)), false);
    }
    let canvas = meta.get("canvas");
    let cw = canvas.and_then(|c| c.get("width")).and_then(int);
    let ch = canvas.and_then(|c| c.get("height")).and_then(int);
    if cw.is_none_or(|w| w <= 0) || ch.is_none_or(|h| h <= 0) {
        bad(
            rep,
            format!("canvas must have positive integer width/height, got {}", canvas.unwrap_or(&Value::Null)),
            false,
        );
    } else {
        let bg = canvas.and_then(|c| c.get("background")).and_then(Value::as_str).unwrap_or("");
        if !(bg.len() == 7 && bg.starts_with('#') && bg[1..].chars().all(|c| c.is_ascii_hexdigit())) {
            bad(
                rep,
                format!(
                    "canvas.background must be #RRGGBB, got {}",
                    canvas.and_then(|c| c.get("background")).unwrap_or(&Value::Null)
                ),
                false,
            );
        }
    }
    let fps = meta.get("fps");
    let (fnum, fden) = (fps.and_then(|f| f.get("num")).and_then(int), fps.and_then(|f| f.get("den")).and_then(int));
    if fnum.is_none_or(|n| n <= 0) || fden.is_none_or(|d| d <= 0) {
        bad(rep, format!("fps must be {{num, den}} positive integers, got {}", fps.unwrap_or(&Value::Null)), true);
    }
    let fc = match meta.get("frame_count").and_then(int) {
        Some(n) if n > 0 => n,
        _ => {
            bad(
                rep,
                format!(
                    "frame_count must be a positive integer, got {}",
                    meta.get("frame_count").unwrap_or(&Value::Null)
                ),
                true,
            );
            0
        }
    };
    if meta.get("max_rap_interval").and_then(int).is_none_or(|n| n <= 0) {
        bad(
            rep,
            format!(
                "max_rap_interval must be a positive integer, got {}",
                meta.get("max_rap_interval").unwrap_or(&Value::Null)
            ),
            true,
        );
    }

    let Some(layers) = meta.get("layers").and_then(Value::as_array) else {
        bad(rep, "layers must be a list".into(), true);
        return ok;
    };
    let mut ids = BTreeSet::new();
    for (li, l) in layers.iter().enumerate() {
        let Some(lo) = l.as_object() else {
            bad(rep, format!("layer {li} is not an object"), true);
            continue;
        };
        let get = |k: &str| lo.get(k).unwrap_or(&Value::Null);
        let tag = format!("layer {li} ({})", get("id"));
        match get("id").as_str() {
            Some(id) if !id.is_empty() => {
                if !ids.insert(id.to_string()) {
                    bad(rep, format!("{tag}: duplicate id"), false);
                }
            }
            _ => bad(rep, format!("{tag}: id must be a non-empty string"), false),
        }
        if lo.contains_key("name") && !get("name").is_string() {
            bad(rep, format!("{tag}: name must be a string"), false);
        }
        let kind = get("kind").as_str().unwrap_or("");
        if kind != "video" && kind != "still" {
            bad(rep, format!("{tag}: kind must be one of (\"video\", \"still\"), got {}", get("kind")), true);
        }
        if !is_number(get("z")) {
            bad(rep, format!("{tag}: z must be a number"), false);
        }
        let r = get("rect");
        let rect_ok = ["x", "y", "w", "h"].iter().all(|k| r.get(k).is_some_and(is_int))
            && r.get("w").and_then(int).unwrap_or(0) > 0
            && r.get("h").and_then(int).unwrap_or(0) > 0;
        if !rect_ok {
            bad(rep, format!("{tag}: rect must be integer {{x,y,w,h}} with w,h > 0, got {r}"), false);
        }
        let (s, e) = (get("start_frame"), get("end_frame"));
        let range_ok = match (int(s), int(e)) {
            (Some(s), Some(e)) => 0 <= s && s < e && e <= fc,
            _ => false,
        };
        if !range_ok {
            bad(
                rep,
                format!("{tag}: need integers 0 <= start_frame < end_frame <= frame_count ({fc}), got [{s}, {e})"),
                true,
            );
        }
        if !get("blend").as_str().is_some_and(|b| BLEND_MODES.contains(&b)) {
            bad(rep, format!("{tag}: blend must be one of {BLEND_MODES:?}, got {}", get("blend")), false);
        }
        if !get("opacity").as_f64().is_some_and(|o| (0.0..=1.0).contains(&o)) {
            bad(rep, format!("{tag}: opacity must be a number in [0, 1], got {}", get("opacity")), false);
        }
        if !get("visible").is_boolean() {
            bad(rep, format!("{tag}: visible must be a boolean, got {}", get("visible")), false);
        }
        if kind == "video" {
            if !get("codec").as_str().and_then(codec_profile).is_some_and(|p| p <= 1) {
                bad(rep, format!("{tag}: codec must be a vp09.00.* or vp09.01.* string, got {}", get("codec")), false);
            }
            if lo.contains_key("lossless") && !get("lossless").is_boolean() {
                bad(rep, format!("{tag}: lossless must be a boolean"), false);
            }
            let (cw, ch) = (int(get("coded_width")), int(get("coded_height")));
            if cw.is_none_or(|w| w <= 0) || ch.is_none_or(|h| h <= 0) {
                bad(rep, format!("{tag}: coded_width/coded_height must be positive integers"), false);
            }
            if let Some(cs) = lo.get("content_size") {
                let ok_cs = cs.as_array().is_some_and(|a| {
                    a.len() == 2
                        && a.iter().all(|v| int(v).is_some_and(|n| n > 0))
                        && int(&a[0]).unwrap_or(i64::MAX) <= cw.unwrap_or(0)
                        && int(&a[1]).unwrap_or(i64::MAX) <= ch.unwrap_or(0)
                });
                if !ok_cs {
                    bad(rep, format!("{tag}: content_size must be [w, h] within the coded size, got {cs}"), false);
                }
            }
            if lo.contains_key("alpha_range") && !get("alpha_range").as_str().is_some_and(alpha_range_ok) {
                bad(
                    rep,
                    format!("{tag}: alpha_range must be one of {ALPHA_RANGES:?}, got {}", get("alpha_range")),
                    false,
                );
            }
            match get("has_alpha").as_bool() {
                None => bad(rep, format!("{tag}: has_alpha must be a boolean"), true),
                Some(true) if !get("alpha_codec").as_str().is_some_and(|c| c.starts_with("vp09.")) => {
                    bad(rep, format!("{tag}: has_alpha is true but alpha_codec is {}", get("alpha_codec")), false)
                }
                Some(false) if !get("alpha_codec").is_null() => {
                    bad(rep, format!("{tag}: has_alpha is false but alpha_codec is {}", get("alpha_codec")), false)
                }
                _ => {}
            }
        } else if kind == "still" {
            let res = get("resource");
            let (off, len) = (res.get("offset").and_then(int), res.get("length").and_then(int));
            let (Some(off), Some(len)) = (off, len) else {
                rep.error("RES", format!("{tag}: resource must be {{offset, length, mime}} integers"), None);
                continue;
            };
            if off < 0 || len <= 0 || off.checked_add(len).is_none_or(|end| end as u64 > resources_size) {
                rep.error(
                    "RES",
                    format!("{tag}: resource [{off}, +{len}) is outside the {resources_size}-byte resource region"),
                    None,
                );
                continue;
            }
            if res.get("mime").and_then(Value::as_str) != Some("image/png") {
                rep.error(
                    "RES",
                    format!("{tag}: mime must be image/png, got {}", res.get("mime").unwrap_or(&Value::Null)),
                    None,
                );
            } else if reader.resource(off as u64, 8).ok().as_deref() != Some(&PNG_SIGNATURE[..]) {
                rep.error("RES", format!("{tag}: resource does not start with the PNG signature"), None);
            }
        }
    }

    match meta.get("audio") {
        None => bad(rep, "audio key is missing (use null for no audio)".into(), false),
        Some(Value::Null) => {}
        Some(Value::Object(a)) => {
            if a.get("codec").and_then(Value::as_str) != Some("opus") {
                rep.error(
                    "AUD",
                    format!("audio.codec must be \"opus\", got {}", a.get("codec").unwrap_or(&Value::Null)),
                    None,
                );
            }
            if a.get("sample_rate").and_then(int) != Some(48000) {
                rep.error(
                    "AUD",
                    format!("audio.sample_rate must be 48000, got {}", a.get("sample_rate").unwrap_or(&Value::Null)),
                    None,
                );
            }
            if !a.get("channels").and_then(int).is_some_and(|c| (1..=2).contains(&c)) {
                rep.error(
                    "AUD",
                    format!("audio.channels must be 1 or 2, got {}", a.get("channels").unwrap_or(&Value::Null)),
                    None,
                );
            }
            match a.get("description_b64") {
                None | Some(Value::Null) => {}
                Some(Value::String(s)) => match base64_decode(s) {
                    Some(head) if head.starts_with(b"OpusHead") => {}
                    Some(_) => rep.error("AUD", "audio.description_b64 is not an OpusHead", None),
                    None => rep.error("AUD", "audio.description_b64 is not valid base64", None),
                },
                Some(_) => rep.error("AUD", "audio.description_b64 is not valid base64", None),
            }
        }
        Some(_) => rep.error("AUD", "audio must be an object or null", None),
    }
    ok
}

/// Standard base64 (RFC 4648) with padding.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let b = s.as_bytes();
    if b.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for chunk in b.chunks(4) {
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 {
            return None;
        }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n <<= 6;
            if i < 4 - pad {
                n |= val(c)?;
            }
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    Some(out)
}

pub fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

pub fn validate(path: impl AsRef<Path>) -> Report {
    validate_opts(path, true)
}

pub fn validate_opts(path: impl AsRef<Path>, check_bitstream: bool) -> Report {
    let path = path.as_ref();
    let mut rep = Report { path: path.display().to_string(), ..Default::default() };
    let reader = match LvfReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            rep.error("HDR", format!("cannot read file header: {e}"), None);
            rep.fatal = true;
            return rep;
        }
    };
    run(&reader, &mut rep, check_bitstream);
    rep
}

struct LayerInfo {
    start: u32,
    end: u32,
    has_alpha: bool,
    id: String,
    codec: String,
    alpha_codec: String,
    coded: (u32, u32),
    alpha_full: bool,
}

fn run(reader: &LvfReader, rep: &mut Report, check_bitstream: bool) {
    let h = reader.header.clone();
    rep.file_size = reader.file_size;

    // ---- header -------------------------------------------------------------------------------
    if &h.magic != MAGIC_FILE {
        rep.error("HDR", format!("magic is {:?}, expected {:?}", String::from_utf8_lossy(&h.magic), "LVF1"), None);
        rep.fatal = true;
        return;
    }
    if h.version != VERSION {
        rep.error("HDR", format!("version is {}, expected {VERSION}", h.version), None);
    }
    if h.flags != 0 {
        rep.error("HDR", format!("flags is {:#x}, must be 0 in v1", h.flags), None);
    }
    if h.reserved_20 != 0 || h.reserved_48.iter().any(|&b| b != 0) {
        rep.warn("HDR", "reserved header bytes are not zero", None);
    }
    if h.meta_offset < HEADER_SIZE as u64 {
        rep.error("HDR", format!("meta_offset {} overlaps the header", h.meta_offset), None);
        rep.fatal = true;
    }
    if h.meta_offset.saturating_add(h.meta_length as u64) > h.resources_offset {
        rep.error(
            "HDR",
            format!(
                "metadata [{}, +{}) runs into resources_offset {}",
                h.meta_offset, h.meta_length, h.resources_offset
            ),
            None,
        );
        rep.fatal = true;
    }
    let order = [
        ("meta_offset", h.meta_offset),
        ("resources_offset", h.resources_offset),
        ("cau_offset", h.cau_offset),
        ("index_offset", h.index_offset),
    ];
    for w in order.windows(2) {
        if w[0].1 > w[1].1 {
            rep.error("HDR", format!("{} ({}) > {} ({})", w[0].0, w[0].1, w[1].0, w[1].1), None);
            rep.fatal = true;
        }
    }
    if h.index_offset > reader.file_size {
        rep.error(
            "HDR",
            format!("index_offset {} is past the end of the file ({})", h.index_offset, reader.file_size),
            None,
        );
        rep.fatal = true;
    }
    if rep.fatal {
        return;
    }

    // ---- metadata (standard JSON, spec B.12) ----------------------------------------------------
    let raw = match reader.meta_bytes() {
        Ok(b) => b,
        Err(e) => {
            rep.error("META", format!("cannot read the metadata: {e}"), None);
            rep.fatal = true;
            return;
        }
    };
    if raw.starts_with(b"\xef\xbb\xbf") {
        rep.error("META", "metadata starts with a UTF-8 byte-order mark", None);
        rep.fatal = true;
        return;
    }
    let meta: Value = match std::str::from_utf8(&raw)
        .map_err(|e| e.to_string())
        .and_then(|s| serde_json::from_str(s).map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(e) => {
            rep.error("META", format!("metadata is not valid UTF-8 JSON (RFC 8259): {e}"), None);
            rep.fatal = true;
            return;
        }
    };
    if !meta.is_object() {
        rep.error("META", "metadata is not a JSON object", None);
        rep.fatal = true;
        return;
    }
    rep.meta = Some(meta.clone());
    if !check_meta(&meta, rep, h.cau_offset.saturating_sub(h.resources_offset), reader) {
        rep.fatal = true;
        return;
    }
    // The readers' own schema (types and ranges): what it rejects cannot be read, so it is fatal.
    if let Err(e) = serde_json::from_value::<crate::meta::Meta>(meta.clone()) {
        rep.error("META", format!("metadata does not match the LVF schema: {e}"), None);
        rep.fatal = true;
        return;
    }
    let fps = match Fps::new(meta["fps"]["num"].as_u64().unwrap(), meta["fps"]["den"].as_u64().unwrap()) {
        Ok(f) => f,
        Err(e) => {
            rep.error("META", e, None);
            rep.fatal = true;
            return;
        }
    };
    let frame_count = meta["frame_count"].as_u64().unwrap() as u32;
    let max_rap = meta["max_rap_interval"].as_u64().unwrap() as u32;
    let layers: Vec<&Value> = meta["layers"].as_array().unwrap().iter().collect();
    let mut infos = BTreeMap::new();
    for (i, l) in layers.iter().enumerate() {
        if l["kind"] == "video" {
            infos.insert(
                i,
                LayerInfo {
                    start: l["start_frame"].as_u64().unwrap() as u32,
                    end: l["end_frame"].as_u64().unwrap() as u32,
                    has_alpha: l["has_alpha"].as_bool().unwrap_or(false),
                    id: l["id"].as_str().unwrap_or("?").to_string(),
                    codec: l["codec"].as_str().unwrap_or("").to_string(),
                    alpha_codec: l["alpha_codec"].as_str().unwrap_or("").to_string(),
                    coded: (
                        l["coded_width"].as_u64().unwrap_or(0) as u32,
                        l["coded_height"].as_u64().unwrap_or(0) as u32,
                    ),
                    alpha_full: l.get("alpha_range").and_then(Value::as_str) == Some("full"),
                },
            );
        }
    }
    let video_layers: Vec<usize> = infos.keys().copied().collect();
    let has_audio_meta = !meta["audio"].is_null();
    for &i in &video_layers {
        rep.layer_stats.insert(i, LayerStats::default());
    }

    // ---- composite frames (sequential walk, independent of the index) -----------------------------
    let mut actual: Vec<(u64, u32, u8)> = Vec::new();
    let mut last_audio_pts: Option<i64> = None;
    let mut expect_next: u32 = 0;
    // Frames are borrowed from the iterator's buffer: validation copies no payloads.
    let mut caus = reader.caus(None, None);
    for k in 0usize.. {
        let Some(item) = caus.next_ref() else { break };
        let (offset, cau, _size) = match item {
            Ok(x) => x,
            Err(e) => {
                rep.error(
                    "CAU",
                    format!("composite-frame region is structurally broken after {} CAUs: {e}", actual.len()),
                    None,
                );
                break;
            }
        };
        let k = k as u32;
        actual.push((offset, cau.frame_index, cau.flags));
        let f = cau.frame_index;
        if f != expect_next {
            let what = if Some(f) == expect_next.checked_add(1) {
                format!("frame {expect_next} is missing")
            } else if f > expect_next {
                format!("frames {expect_next}..{} are missing", f - 1)
            } else {
                "frame number goes backwards or repeats".to_string()
            };
            rep.error(
                "I1",
                format!("CAU #{k} (offset {offset}) has frame_index {f}, expected {expect_next} ({what})"),
                Some(k),
            );
        }
        expect_next = f.wrapping_add(1);
        if cau.flags & !CAU_FLAG_RAP != 0 {
            rep.error("CAU", format!("undefined flag bits set: {:#04x}", cau.flags), Some(f));
        }
        if cau.reserved_13 != 0 || cau.reserved_18 != 0 {
            rep.warn("CAU", "reserved CAU header fields are not zero", Some(f));
        }
        if f >= frame_count {
            rep.error("I1", format!("frame_index {f} is outside [0, {frame_count})"), Some(k));
            continue;
        }

        // I2
        let got: Vec<usize> = cau.entries.iter().map(|e| e.layer_index as usize).collect();
        let count = cau.video_entry_count as usize;
        if count != video_layers.len() || got != video_layers {
            let want: BTreeSet<usize> = video_layers.iter().copied().collect();
            let have: BTreeSet<usize> = got.iter().copied().collect();
            let missing: Vec<_> = want.difference(&have).collect();
            let extra: Vec<_> = have.difference(&want).collect();
            let mut detail = Vec::new();
            if !missing.is_empty() {
                detail.push(format!("missing layers {missing:?}"));
            }
            if !extra.is_empty() {
                detail.push(format!("entries for non-video/unknown layers {extra:?}"));
            }
            if missing.is_empty() && extra.is_empty() {
                detail.push(format!("layer order {got:?} (expected {video_layers:?})"));
            }
            rep.error(
                "I2",
                format!("video_entry_count={count}, expected {}; {}", video_layers.len(), detail.join("; ")),
                Some(f),
            );
        }

        let mut rap_should = true;
        let mut seen = BTreeSet::new();
        for e in &cau.entries {
            let li = e.layer_index as usize;
            let Some(l) = infos.get(&li) else { continue };
            if !seen.insert(li) {
                continue;
            }
            let active = l.start <= f && f < l.end;
            if e.kind != ENTRY_EMPTY && e.kind != ENTRY_FRAME {
                rep.error("CAU", format!("layer {li}: entry type {} is not valid in v1", e.kind), Some(f));
                rap_should = false;
                continue;
            }
            if (e.kind == ENTRY_FRAME) != active {
                rep.error(
                    "I3",
                    format!(
                        "layer {li} ({}) is {} in [{}, {}) but its entry is {}",
                        l.id,
                        if active { "active" } else { "inactive" },
                        l.start,
                        l.end,
                        if e.kind == ENTRY_FRAME { "FRAME" } else { "EMPTY" }
                    ),
                    Some(f),
                );
            }
            if e.kind == ENTRY_EMPTY {
                if !e.color.is_empty() || !e.alpha.is_empty() || e.frame_flags != 0 {
                    rep.error("CAU", format!("layer {li}: EMPTY entry carries data or flags"), Some(f));
                }
                continue;
            }
            let st = rep.layer_stats.get_mut(&li).unwrap();
            st.frames += 1;
            st.color_bytes += e.color.len() as u64;
            st.alpha_bytes += e.alpha.len() as u64;
            if e.frame_flags & !FRAME_FLAG_KEY != 0 {
                rep.error("CAU", format!("layer {li}: undefined frame_flags bits {:#04x}", e.frame_flags), Some(f));
            }
            if e.color.is_empty() {
                rep.error("CAU", format!("layer {li}: FRAME entry without color data"), Some(f));
                rap_should = false;
                continue;
            }
            if l.has_alpha && e.alpha.is_empty() {
                rep.error("CAU", format!("layer {li}: has_alpha layer without alpha data"), Some(f));
            }
            if !l.has_alpha && !e.alpha.is_empty() {
                rep.error("CAU", format!("layer {li}: alpha data on a layer without has_alpha"), Some(f));
            }
            let flag_key = e.is_key();
            let (mut color_key, mut alpha_key) = (flag_key, flag_key);
            if check_bitstream {
                color_key = check_packet(rep, f, li, "color", e.color, flag_key, l);
                if !e.alpha.is_empty() {
                    alpha_key = check_packet(rep, f, li, "alpha", e.alpha, flag_key, l);
                }
            }
            let planes_key = color_key && (e.alpha.is_empty() || alpha_key);
            if planes_key {
                rep.layer_stats.get_mut(&li).unwrap().keyframes += 1;
            }
            if !e.alpha.is_empty() && color_key != alpha_key {
                let kd = |k: bool| if k { "key" } else { "delta" };
                rep.error(
                    "I5",
                    format!("layer {li} ({}): color is {} but alpha is {}", l.id, kd(color_key), kd(alpha_key)),
                    Some(f),
                );
            }
            if f == l.start && !(flag_key && planes_key) {
                let which = if !color_key { "color" } else { "alpha" };
                rep.error(
                    "I4",
                    format!("layer {li} ({}) starts here but its {which} frame is not a key frame", l.id),
                    Some(f),
                );
            }
            rap_should = rap_should && flag_key && planes_key;
        }

        let is_rap = cau.is_rap();
        if is_rap != rap_should {
            rep.error(
                "I6",
                format!("RAP flag is {} but the frame's entries say it should be {}", is_rap as u8, rap_should as u8),
                Some(f),
            );
        }
        if is_rap {
            rep.rap_frames.push(f);
        }
        if f == 0 && !is_rap {
            rep.error("I7", "frame 0 is not a RAP", Some(f));
        }

        // I9 + audio sanity
        let (lo, hi) = (pts_us(f as u64, fps), pts_us(f as u64 + 1, fps));
        if !cau.audio.is_empty() && !has_audio_meta {
            rep.error("AUD", format!("{} audio packets but metadata audio is null", cau.audio.len()), Some(f));
        }
        for a in &cau.audio {
            rep.audio_packets += 1;
            rep.audio_bytes += a.data.len() as u64;
            if !(lo <= a.pts_us && a.pts_us < hi) {
                rep.error(
                    "I9",
                    format!("audio packet pts {} us is outside this frame's window [{lo}, {hi})", a.pts_us),
                    Some(f),
                );
            }
            if a.data.is_empty() {
                rep.error("AUD", "empty audio packet", Some(f));
            }
            if a.duration_us == 0 {
                rep.warn("AUD", format!("audio packet with duration {} us", a.duration_us), Some(f));
            }
            if let Some(prev) = last_audio_pts {
                if a.pts_us <= prev {
                    rep.error("AUD", format!("audio pts {} does not increase (previous {prev})", a.pts_us), Some(f));
                }
            }
            last_audio_pts = Some(a.pts_us);
        }
    }

    rep.cau_count = actual.len() as u64;
    if rep.cau_count != frame_count as u64 {
        rep.error("I1", format!("file holds {} composite frames, frame_count is {frame_count}", rep.cau_count), None);
    }

    // I8
    let raps = rep.rap_frames.clone();
    for w in raps.windows(2) {
        if w[1].saturating_sub(w[0]) > max_rap {
            rep.error(
                "I8",
                format!("RAPs at {} and {} are {} frames apart (max_rap_interval {max_rap})", w[0], w[1], w[1] - w[0]),
                Some(w[1]),
            );
        }
    }
    if let Some(&last) = raps.last() {
        if frame_count.saturating_sub(last) > max_rap {
            rep.warn(
                "I8",
                format!(
                    "last RAP at {last} leaves {} frames to the end (max_rap_interval {max_rap})",
                    frame_count - last
                ),
                Some(last),
            );
        }
    }

    check_index(reader, rep, &actual, frame_count);
}

fn check_packet(rep: &mut Report, f: u32, li: usize, plane: &str, data: &[u8], flag_key: bool, l: &LayerInfo) -> bool {
    let pk = match inspect_packet(data) {
        Ok(p) => p,
        Err(e) => {
            rep.error("VP9", format!("layer {li} {plane}: cannot parse VP9 header: {e}"), Some(f));
            return flag_key;
        }
    };
    if pk.shown_count() != 1 {
        rep.error(
            "VP9",
            format!("layer {li} {plane}: packet shows {} frames (must be exactly 1)", pk.shown_count()),
            Some(f),
        );
    }
    if pk.key_frame() != flag_key {
        let kd = |k: bool| if k { "key" } else { "delta" };
        rep.error(
            "VP9",
            format!(
                "layer {li} {plane}: frame_flags says {} but the bitstream is a {} frame",
                kd(flag_key),
                kd(pk.key_frame())
            ),
            Some(f),
        );
    }
    if let Some(ki) = pk.key_info() {
        let (w, h) = (ki.width.unwrap_or(0), ki.height.unwrap_or(0));
        if (w, h) != l.coded {
            rep.error(
                "VP9",
                format!("layer {li} {plane}: key frame is {w}x{h}, metadata says {}x{}", l.coded.0, l.coded.1),
                Some(f),
            );
        }
        let codec = if plane == "color" { &l.codec } else { &l.alpha_codec };
        let profile = codec_profile(codec).unwrap_or(0);
        let want_sub = if profile == 0 { (1, 1) } else { (0, 0) };
        if ki.profile != profile || ki.bit_depth != Some(8) || ki.subsampling != Some(want_sub) {
            rep.error(
                "VP9",
                format!(
                    "layer {li} {plane}: bitstream is profile {}, {}-bit, subsampling {:?}; codec {codec:?} requires profile {profile}, 8-bit, subsampling {want_sub:?}",
                    ki.profile,
                    ki.bit_depth.unwrap_or(0),
                    ki.subsampling.unwrap_or((9, 9))
                ),
                Some(f),
            );
        }
        if plane == "alpha" {
            let want_range = l.alpha_full as u32;
            if ki.color_range != Some(want_range) {
                let r = |x: u32| if x == 1 { "full" } else { "limited" };
                rep.error(
                    "VP9",
                    format!(
                        "layer {li} alpha: bitstream signals {} range, metadata alpha_range is {:?}",
                        r(ki.color_range.unwrap_or(0)),
                        if l.alpha_full { "full" } else { "limited" }
                    ),
                    Some(f),
                );
            }
        } else if profile == 1 && ki.color_space != Some(CS_RGB) {
            rep.error(
                "VP9",
                format!(
                    "layer {li} color: profile-1 planes must be RGB (color_space 7), got {}",
                    ki.color_space.unwrap_or(0)
                ),
                Some(f),
            );
        }
    }
    pk.key_frame()
}

fn check_index(reader: &LvfReader, rep: &mut Report, actual: &[(u64, u32, u8)], frame_count: u32) {
    let region = reader.file_size.saturating_sub(reader.header.index_offset);
    let (magic, count, entries) = match reader.index() {
        Ok(x) => x,
        Err(Error::Format(e)) | Err(Error::Value(e)) => {
            rep.error("I10", format!("index table unreadable: {e}"), None);
            return;
        }
        Err(e) => {
            rep.error("I10", format!("index table unreadable: {e}"), None);
            return;
        }
    };
    if &magic != MAGIC_INDEX {
        rep.error("I10", format!("index magic is {:?}, expected \"IDX1\"", String::from_utf8_lossy(&magic)), None);
    }
    if count != frame_count {
        rep.error("I10", format!("index count is {count}, frame_count is {frame_count}"), None);
    }
    let expect_size = (INDEX_HEADER_SIZE + count as usize * INDEX_ENTRY_SIZE) as u64;
    if region != expect_size {
        rep.error("I10", format!("{} trailing bytes after the index table", region as i64 - expect_size as i64), None);
    }
    if entries.len() != actual.len() {
        rep.error(
            "I10",
            format!("index has {} entries but the file holds {} composite frames", entries.len(), actual.len()),
            None,
        );
    }
    let mut prev: i64 = -1;
    for (i, e) in entries.iter().enumerate() {
        let iu = i as u32;
        if e.frame_index != iu && e.frame_index as i64 != prev + 1 {
            rep.error("I10", format!("index entry {i} has frame_index {}", e.frame_index), Some(iu));
        }
        prev = e.frame_index as i64;
        if e.flags & !INDEX_FLAG_RAP != 0 {
            rep.error("I10", format!("index entry {i} has undefined flag bits {:#04x}", e.flags), Some(iu));
        }
        if let Some(&(off, _fi, fl)) = actual.get(i) {
            if e.cau_offset != off {
                rep.error(
                    "I10",
                    format!("index entry {i} points to offset {}, CAU #{i} is at {off}", e.cau_offset),
                    Some(iu),
                );
            }
            if e.is_rap() != (fl & CAU_FLAG_RAP != 0) {
                rep.error(
                    "I10",
                    format!("index entry {i} RAP flag {} differs from CAU header {}", e.flags & 1, fl & 1),
                    Some(iu),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        for data in [&b""[..], b"f", b"fo", b"foo", b"foob", b"OpusHead\x01\x02\x38\x01"] {
            let s = base64_encode(data);
            assert_eq!(base64_decode(&s).unwrap(), data, "{s}");
        }
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert!(base64_decode("abc").is_none());
    }
}
