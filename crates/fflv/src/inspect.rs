//! Human-readable output: validation reports (`fflv info`, `fflv check`, `fflv pack`) and dumps of
//! the header, the metadata and single composite frames (`fflv info`).

use std::fmt::Write as _;

use lvf::constants::{ENTRY_EMPTY, ENTRY_FRAME, ENTRY_HOLD};
use lvf::validate::INVARIANTS;
use lvf::vp9::inspect_packet;
use lvf::{pts_us, LvfReader, Report};
use serde_json::Value;

use crate::error::{Error, Result};

/// "0, 60, 120, … 540 (every 60)" for regular series, else the first `limit` values.
pub fn compress_positions(values: &[u32], limit: usize) -> String {
    if values.is_empty() {
        return "none".into();
    }
    if values.len() > 2 {
        let step = values[1].wrapping_sub(values[0]);
        if step > 0 && step < u32::MAX / 2 && values.windows(2).all(|w| w[1].wrapping_sub(w[0]) == step) {
            return format!("{}, {}, … {} (every {step})", values[0], values[1], values[values.len() - 1]);
        }
    }
    let shown: Vec<String> = values.iter().take(limit).map(|v| v.to_string()).collect();
    let more = if values.len() > limit { format!(", … (+{} more)", values.len() - limit) } else { String::new() };
    shown.join(", ") + &more
}

pub fn stats(rep: &Report) -> String {
    let mut s = String::new();
    let null = Value::Null;
    let meta = rep.meta.as_ref().unwrap_or(&null);
    let num = meta["fps"]["num"].as_f64().unwrap_or(1.0);
    let den = meta["fps"]["den"].as_f64().unwrap_or(1.0);
    let fc = meta["frame_count"].as_u64().unwrap_or(0);
    let dur = if num > 0.0 { fc as f64 * den / num } else { 0.0 };
    let _ = writeln!(s, "file size       {} bytes ({:.2} MB)", thousands(rep.file_size), rep.file_size as f64 / 1e6);
    let _ = writeln!(s, "composite frames {} (frame_count {fc}, {dur:.3} s)", rep.cau_count);
    let _ = writeln!(s, "RAPs            {}: {}", rep.rap_frames.len(), compress_positions(&rep.rap_frames, 12));
    let empty = Vec::new();
    let layers = meta["layers"].as_array().unwrap_or(&empty);
    for (&li, st) in &rep.layer_stats {
        let l = layers.get(li).unwrap_or(&null);
        let secs = if num > 0.0 { st.frames as f64 * den / num } else { 0.0 };
        let kbps = |b: u64| if secs > 0.0 { b as f64 * 8.0 / secs / 1000.0 } else { 0.0 };
        let alpha = if l["has_alpha"].as_bool() == Some(true) {
            format!(", alpha {:8.1} kbit/s", kbps(st.alpha_bytes))
        } else {
            String::new()
        };
        let _ = writeln!(
            s,
            "layer {li:<2} {:<12} {:>6} frames, {:>4} key, color {:8.1} kbit/s{alpha}",
            l["id"].as_str().unwrap_or("?"),
            st.frames,
            st.keyframes,
            kbps(st.color_bytes)
        );
    }
    for (li, l) in layers.iter().enumerate() {
        if l["kind"].as_str() == Some("still") {
            let _ = writeln!(
                s,
                "layer {li:<2} {:<12} still, {} byte PNG, frames [{}, {})",
                l["id"].as_str().unwrap_or("?"),
                thousands(l["resource"]["length"].as_u64().unwrap_or(0)),
                l["start_frame"],
                l["end_frame"]
            );
        }
    }
    if !meta["audio"].is_null() {
        let kbps = if dur > 0.0 { rep.audio_bytes as f64 * 8.0 / dur / 1000.0 } else { 0.0 };
        let _ = writeln!(s, "audio           {} Opus packets, {kbps:.1} kbit/s", rep.audio_packets);
    }
    s
}

pub fn report(rep: &Report, verbose: bool) -> String {
    let mut s = String::new();
    if verbose && rep.meta.is_some() {
        s += &stats(rep);
        s.push('\n');
    }
    let codes = rep.error_codes();
    if rep.fatal {
        s += "FATAL: file could not be validated completely\n";
    }
    for inv in INVARIANTS {
        let state = if codes.contains(inv) {
            "FAIL"
        } else if rep.fatal {
            "n/a "
        } else {
            "ok  "
        };
        let _ = writeln!(s, "  {inv:<4} {state}");
    }
    let other: Vec<&str> = codes.iter().map(String::as_str).filter(|c| !INVARIANTS.contains(c)).collect();
    if !other.is_empty() {
        let _ = writeln!(s, "  other failures: {}", other.join(", "));
    }
    for i in &rep.issues {
        let _ = writeln!(s, "  {:7} {i}", i.severity.to_uppercase());
    }
    for (key, n) in &rep.suppressed {
        let _ = writeln!(s, "  ... {n} more {key} issues not shown");
    }
    if rep.ok() {
        s += "VALID\n";
    } else {
        let _ = writeln!(s, "INVALID ({} errors)", rep.error_count());
    }
    s
}

pub fn thousands(n: u64) -> String {
    let d = n.to_string();
    let mut out = String::new();
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn header(r: &LvfReader) -> String {
    let h = &r.header;
    let mut s = String::new();
    let magic = String::from_utf8_lossy(&h.magic);
    let _ = writeln!(s, "magic {magic:?}  version {}  flags {:#06x}", h.version, h.flags);
    let _ =
        writeln!(s, "meta       offset {:>12}  length {}", thousands(h.meta_offset), thousands(h.meta_length as u64));
    let _ = writeln!(
        s,
        "resources  offset {:>12}  length {}",
        thousands(h.resources_offset),
        thousands(h.cau_offset.saturating_sub(h.resources_offset))
    );
    let _ = writeln!(
        s,
        "CAUs       offset {:>12}  length {}",
        thousands(h.cau_offset),
        thousands(h.index_offset.saturating_sub(h.cau_offset))
    );
    let _ = writeln!(
        s,
        "index      offset {:>12}  length {}",
        thousands(h.index_offset),
        thousands(r.file_size.saturating_sub(h.index_offset))
    );
    s
}

/// The metadata JSON, with the (long) Opus description shortened.
pub fn meta_dump(meta: &Value) -> String {
    let mut m = meta.clone();
    if let Some(d) = m["audio"]["description_b64"].as_str() {
        if d.chars().count() > 40 {
            let short: String = d.chars().take(40).collect();
            m["audio"]["description_b64"] = Value::String(short + "…");
        }
    }
    serde_json::to_string_pretty(&m).unwrap_or_default()
}

fn locate_cau(r: &LvfReader, n: u32) -> Result<u64> {
    if let Ok((_, _, entries)) = r.index() {
        if let Some(e) = entries.get(n as usize) {
            return Ok(e.cau_offset);
        }
    }
    for (k, item) in r.caus(None, None).enumerate() {
        let (off, _, _) = item?;
        if k == n as usize {
            return Ok(off);
        }
    }
    Err(Error::Format(format!("file has no composite frame #{n}")))
}

pub fn describe_vp9(data: &[u8]) -> String {
    let pk = match inspect_packet(data) {
        Ok(p) => p,
        Err(e) => return format!("unparseable ({e})"),
    };
    let fr = &pk.frames[0];
    let mut s = if pk.key_frame() { "key".to_string() } else { "inter".to_string() };
    if pk.superframe {
        let _ = write!(s, ", superframe of {}", pk.frames.len());
    }
    if let (Some(w), Some(h)) = (fr.width, fr.height) {
        let range = if fr.color_range == Some(1) { "full" } else { "limited" };
        let _ = write!(s, ", {w}x{h}, cs={} range={range}", fr.color_space.map_or("?".into(), |c| c.to_string()));
    }
    s
}

pub fn frame_dump(r: &LvfReader, n: u32) -> Result<String> {
    let meta = r.meta_json()?;
    let empty = Vec::new();
    let layers = meta["layers"].as_array().unwrap_or(&empty);
    let fps = lvf::Fps::new(meta["fps"]["num"].as_u64().unwrap_or(1), meta["fps"]["den"].as_u64().unwrap_or(1))
        .map_err(Error::Format)?;
    let off = locate_cau(r, n)?;
    let (cau, size) = r.cau_at(off)?;
    let f = cau.frame_index;
    let mut s = String::new();
    let _ = writeln!(s, "composite frame #{n} at offset {}, {} bytes", thousands(off), thousands(size as u64));
    let (a, b) = (pts_us(f as u64, fps), pts_us(f as u64 + 1, fps));
    let rap = if cau.is_rap() { " (RAP)" } else { "" };
    let _ = writeln!(s, "  frame_index {f}  flags {:#04x}{rap}  pts {a} us  window [{a}, {b})", cau.flags);
    let count = cau.video_entry_count.map_or(cau.entries.len(), |c| c as usize);
    let _ = writeln!(s, "  video entries {count}, audio packets {}", cau.audio.len());
    for e in &cau.entries {
        let id = layers.get(e.layer_index as usize).and_then(|l| l["id"].as_str()).unwrap_or("?");
        let kind = match e.kind {
            ENTRY_EMPTY => "EMPTY".to_string(),
            ENTRY_FRAME => "FRAME".to_string(),
            ENTRY_HOLD => "HOLD".to_string(),
            k => k.to_string(),
        };
        let mut line = format!("    layer {:<2} {id:<12} {kind:<5}", e.layer_index);
        if e.kind == ENTRY_FRAME {
            let key = if e.is_key() { "KEY" } else { "   " };
            let _ = write!(line, " {key} color {:>7} B [{}]", thousands(e.color.len() as u64), describe_vp9(&e.color));
            if !e.alpha.is_empty() {
                let _ = write!(
                    line,
                    "\n{:30}alpha {:>7} B [{}]",
                    "",
                    thousands(e.alpha.len() as u64),
                    describe_vp9(&e.alpha)
                );
            }
        }
        let _ = writeln!(s, "{line}");
    }
    for p in &cau.audio {
        let _ =
            writeln!(s, "    audio pts {:>12} us  duration {:>6} us  {:>5} B", p.pts_us, p.duration_us, p.data.len());
    }
    let stills: Vec<&str> = layers
        .iter()
        .filter(|l| {
            l["kind"].as_str() == Some("still")
                && l["start_frame"].as_u64().is_some_and(|a| a <= f as u64)
                && l["end_frame"].as_u64().is_some_and(|b| (f as u64) < b)
        })
        .filter_map(|l| l["id"].as_str())
        .collect();
    if !stills.is_empty() {
        let _ = writeln!(s, "  still layers shown: {}", stills.join(", "));
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions() {
        assert_eq!(compress_positions(&[0, 60, 120, 180], 12), "0, 60, … 180 (every 60)");
        assert_eq!(compress_positions(&[0, 5, 7], 2), "0, 5, … (+1 more)");
        assert_eq!(compress_positions(&[], 12), "none");
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(12), "12");
    }
}
