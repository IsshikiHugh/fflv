//! `fflv pack`: build an .lvd from a project JSON (spec section 8.1).
//!
//! Every video layer's source is decoded by its own FFmpeg process (scaled to the layer's rect,
//! at the file's frame rate) and encoded in-process by the [`Writer`], all layers in parallel;
//! audio is transcoded to Opus at the same time. The Writer places key frames exactly (layer
//! starts + the GOP grid), verifies every packet, validates the file and publishes it atomically.
//!
//! Project additions over spec 8.1: per layer `lossless` (bit-exact RGB / alpha), `start_frame` /
//! `end_frame` instead of seconds; `quality.alpha_crf`, `quality.cpu_used`, `quality.speed`;
//! `audio.bitrate`, `audio.channels`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lvf::constants::FILE_EXTENSION;
use lvf::meta::{self, Rect};
use lvf::timing::seconds_to_frame;
use lvf::{Fps, Report};
use serde_json::Value;

use crate::audio::{encode_audio, AudioTrack};
use crate::codec::Speed;
use crate::error::{Error, Result};
use crate::image::Image;
use crate::media::{open_video_source, raw_frames, still_png_from_file, RawFrames};
use crate::render::Progress;
use crate::writer::{LayerOptions, StillOptions, Writer, WriterOptions};

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Pack(msg.into()))
}

#[derive(Clone, Debug)]
pub struct ProjectLayer {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub still: bool,
    pub src: PathBuf,
    pub z: f64,
    pub rect: Rect,
    pub start_frame: u32,
    pub end_frame: u32,
    pub alpha: bool,
    pub lossless: bool,
    pub blend: String,
    pub opacity: f64,
    pub visible: bool,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub path: PathBuf,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub background: String,
    pub fps: Fps,
    pub frame_count: u32,
    pub gop: u32,
    pub crf: u32,
    pub alpha_crf: u32,
    pub speed: Speed,
    pub cpu_used: Option<i32>,
    pub layers: Vec<ProjectLayer>,
    pub audio_src: Option<PathBuf>,
    pub audio_bitrate: String,
    pub audio_channels: u32,
}

fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn uint(v: &Value, what: &str) -> Result<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::Pack(format!("{what} must be a non-negative integer, got {v}")))
}

/// `start` / `end` of a layer: `<key>_frame` (frames) or `<key>` (seconds), else `default`.
fn frame_of(l: &Value, key: &str, fps: Fps, default: u32) -> Result<u32> {
    if let Some(v) = l.get(format!("{key}_frame")) {
        return uint(v, &format!("{key}_frame"));
    }
    match l.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let f = seconds_to_frame(&text(v), fps).map_err(Error::Pack)?;
            u32::try_from(f).map_err(|_| Error::Pack(format!("{key} {v} is before the start")))
        }
    }
}

pub fn load_project(path: &Path, output: Option<&Path>) -> Result<Project> {
    let data = std::fs::read(path).map_err(|e| Error::Pack(format!("cannot read project {}: {e}", path.display())))?;
    let p: Value = serde_json::from_slice(&data)
        .map_err(|e| Error::Pack(format!("cannot read project {}: {e}", path.display())))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let src_path = |v: Option<&Value>| -> Result<PathBuf> {
        let s = v.and_then(Value::as_str).filter(|s| !s.is_empty());
        let s =
            s.ok_or_else(|| Error::Pack(format!("src must be a path string, got {}", v.unwrap_or(&Value::Null))))?;
        let p = Path::new(s);
        let p = if p.is_absolute() { p.to_path_buf() } else { base.join(p) };
        if !p.exists() {
            return err(format!("source file not found: {}", p.display()));
        }
        Ok(p)
    };

    let canvas = &p["canvas"];
    let (width, height) = match (canvas["width"].as_u64(), canvas["height"].as_u64()) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w as u32, h as u32),
        _ => return err(format!("canvas needs positive integer width/height, got {canvas}")),
    };
    let fps = Fps::parse(&p.get("fps").map_or("30/1".into(), text)).map_err(Error::Pack)?;
    let duration = p.get("duration").ok_or_else(|| Error::Pack("project needs a duration (seconds)".into()))?;
    let frame_count = seconds_to_frame(&text(duration), fps).map_err(Error::Pack)?;
    if frame_count <= 0 {
        return err(format!("duration {duration} gives {frame_count} frames"));
    }
    let frame_count = frame_count as u32;
    let gop = match p.get("gop") {
        Some(v) => uint(v, "gop")?,
        None => ((2.0 * fps.as_f64()).round() as u32).max(1),
    };
    if gop == 0 {
        return err("gop must be positive");
    }
    let q = &p["quality"];
    let crf = q.get("crf").map_or(Ok(32), |v| uint(v, "quality.crf"))?;
    let alpha_crf = q.get("alpha_crf").map_or(Ok(crf), |v| uint(v, "quality.alpha_crf"))?;
    let speed = q.get("speed").and_then(Value::as_str).map_or(Ok(Speed::Balanced), Speed::parse)?;
    let cpu_used = match q.get("cpu_used") {
        Some(v) => Some(
            v.as_i64()
                .filter(|c| (-8..=8).contains(c))
                .ok_or_else(|| Error::Pack(format!("quality.cpu_used must be an integer in -8..8, got {v}")))?
                as i32,
        ),
        None => None,
    };

    let mut layers = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    for (i, l) in p["layers"].as_array().map(Vec::as_slice).unwrap_or(&[]).iter().enumerate() {
        let tag = format!("layer {i} ({})", l.get("id").unwrap_or(&Value::Null));
        let taken: Vec<&str> = ids.iter().map(String::as_str).collect();
        let id = meta::check_id(l["id"].as_str().unwrap_or(""), &taken)?;
        ids.push(id.clone());
        let still = match l.get("kind").and_then(Value::as_str).unwrap_or("video") {
            "video" => false,
            "still" => true,
            _ => return err(format!("{tag}: kind must be video or still")),
        };
        let start = frame_of(l, "start", fps, 0)?;
        let end = frame_of(l, "end", fps, frame_count)?.min(frame_count);
        if start >= end {
            return err(format!("{tag}: empty or negative interval [{start}, {end}) (frame_count {frame_count})"));
        }
        let flag = |k: &str, default: bool| l.get(k).and_then(Value::as_bool).unwrap_or(default);
        layers.push(ProjectLayer {
            index: i,
            name: l.get("name").map_or(id.clone(), text),
            id,
            still,
            src: src_path(l.get("src"))?,
            z: meta::check_z(l.get("z").unwrap_or(&Value::from(i)))?.0,
            rect: meta::rect_from_json(l.get("rect").unwrap_or(&Value::Null))?,
            start_frame: start,
            end_frame: end,
            alpha: flag("alpha", false),
            lossless: flag("lossless", false),
            blend: meta::check_blend(l.get("blend").and_then(Value::as_str).unwrap_or("normal"))?,
            opacity: meta::check_opacity(l.get("opacity").and_then(Value::as_f64).unwrap_or(1.0))?,
            visible: flag("visible", true),
        });
    }
    if layers.is_empty() {
        return err("project has no layers");
    }
    let background = meta::check_background(canvas.get("background").and_then(Value::as_str).unwrap_or("#000000"))?;
    let audio = p.get("audio").filter(|a| !a.is_null());
    let out = match (output, p.get("output").and_then(Value::as_str)) {
        (Some(o), _) => std::env::current_dir()?.join(o),
        (None, Some(o)) => base.join(o),
        (None, None) => {
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into());
            base.join(format!("{stem}.{FILE_EXTENSION}"))
        }
    };
    Ok(Project {
        path: path.to_path_buf(),
        output: out,
        width,
        height,
        background,
        fps,
        frame_count,
        gop,
        crf,
        alpha_crf,
        speed,
        cpu_used,
        layers,
        audio_src: audio.map(|a| src_path(a.get("src"))).transpose()?,
        audio_bitrate: audio.and_then(|a| a.get("bitrate")).map_or("128k".into(), text),
        audio_channels: match audio.and_then(|a| a.get("channels")) {
            Some(v) => uint(v, "audio.channels")?,
            None => 2,
        },
    })
}

pub type Log<'a> = &'a mut dyn FnMut(&str);

pub struct PackOptions {
    pub output: Option<PathBuf>,
    /// Encoding threads (default: all cores).
    pub threads: Option<usize>,
}

/// Build the project's file. Returns the validation report of the published file.
pub fn pack(project: &Path, o: &PackOptions, log: Log) -> Result<Report> {
    pack_with_progress(project, o, log, None)
}

/// [`pack`], reporting (frames encoded, frames in total) after each frame; a progress error stops
/// the pack (the previous output stays).
pub fn pack_with_progress(project: &Path, o: &PackOptions, log: Log, mut progress: Option<Progress>) -> Result<Report> {
    let t0 = Instant::now();
    let project = std::path::absolute(project)?;
    let proj = load_project(&project, o.output.as_deref())?;
    let name = project.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    log(&format!(
        "project {name}: {}x{} @ {} fps, {} frames, gop {}, {} layers",
        proj.width,
        proj.height,
        proj.fps,
        proj.frame_count,
        proj.gop,
        proj.layers.len()
    ));
    let (bitrate, channels) = (proj.audio_bitrate.clone(), proj.audio_channels);
    let seconds = proj.frame_count as f64 * proj.fps.den as f64 / proj.fps.num as f64;
    let audio_job = proj
        .audio_src
        .clone()
        .map(|src| std::thread::spawn(move || encode_audio(&src, Some(seconds), &bitrate, channels)));

    let wopts = WriterOptions {
        gop: Some(proj.gop),
        background: proj.background.clone(),
        crf: proj.crf,
        speed: proj.speed,
        check: true,
        threads: o.threads,
    };
    let mut w = Writer::create(&proj.output, proj.width, proj.height, proj.fps, wopts)?;
    let mut sources: Vec<Option<RawFrames>> = Vec::new();
    for l in &proj.layers {
        log(&format!(
            "  layer {} {:<14} {:<5} frames [{}, {}){}{}",
            l.index,
            format!("{:?}", l.id),
            if l.still { "still" } else { "video" },
            l.start_frame,
            l.end_frame,
            if l.alpha { "  +alpha" } else { "" },
            if l.lossless { "  lossless" } else { "" }
        ));
        if l.still {
            let png = still_png_from_file(&l.src).map_err(|e| e.into_stage(Error::Pack))?;
            let so = StillOptions {
                rect: Some(l.rect),
                start: l.start_frame,
                end: Some(l.end_frame),
                z: Some(l.z),
                name: Some(l.name.clone()),
                blend: l.blend.clone(),
                opacity: l.opacity,
                visible: l.visible,
            };
            w.add_still(&l.id, png, so)?;
            sources.push(None);
            continue;
        }
        let src = open_video_source(&l.src).map_err(|e| e.into_stage(Error::Pack))?;
        if l.alpha && !src.has_alpha {
            let file = l.src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            return err(format!(
                "layer {:?}: alpha=true but {file} ({}, {}) has no alpha channel",
                l.id, src.codec, src.pix_fmt
            ));
        }
        let lo = LayerOptions {
            alpha: l.alpha,
            lossless: l.lossless,
            rect: Some(l.rect),
            z: Some(l.z),
            name: Some(l.name.clone()),
            blend: l.blend.clone(),
            opacity: l.opacity,
            visible: l.visible,
            crf: Some(proj.crf),
            alpha_crf: Some(proj.alpha_crf),
            speed: Some(proj.speed),
            cpu_used: proj.cpu_used,
        };
        w.add_layer(&l.id, lo)?;
        let n = (l.end_frame - l.start_frame) as u64;
        let frames =
            raw_frames(&src, proj.fps, l.rect.w, l.rect.h, n, l.alpha).map_err(|e| e.into_stage(Error::Pack))?;
        sources.push(Some(frames));
    }
    if let Some(job) = audio_job {
        let track: AudioTrack = job
            .join()
            .map_err(|_| Error::Pack("audio thread panicked".into()))?
            .map_err(|e| e.into_stage(Error::Pack))?;
        log(&format!(
            "  audio: {} Opus packets, {} ch, pre-skip {}",
            track.packets.len(),
            track.channels,
            track.pre_skip
        ));
        w.set_audio_track(track)?;
    }

    log(&format!("encoding {} frames ...", proj.frame_count));
    for f in 0..proj.frame_count {
        let mut images: Vec<(usize, Image)> = Vec::new();
        for (l, src) in proj.layers.iter().zip(sources.iter_mut()) {
            let Some(src) = src else { continue };
            if f == l.end_frame {
                w.end_layer(&l.id)?;
            }
            if l.start_frame <= f && f < l.end_frame {
                let img = src.next().unwrap_or_else(|| err("source ended early"));
                images.push((l.index, img.map_err(|e| e.into_stage(Error::Pack))?));
            }
        }
        let refs: Vec<(&str, crate::image::ImageRef)> =
            images.iter().map(|(i, img)| (proj.layers[*i].id.as_str(), img.view())).collect();
        w.write(&refs)?;
        if let Some(p) = progress.as_mut() {
            p(f + 1, proj.frame_count)?;
        }
    }
    log(&format!("  done in {:.1} s; writing {} ...", t0.elapsed().as_secs_f64(), proj.output.display()));
    let rep = match w.close() {
        Ok(r) => r.cloned(),
        Err(e) => return Err(e.into_stage(Error::Pack)),
    };
    log(&format!("total {:.1} s", t0.elapsed().as_secs_f64()));
    rep.ok_or_else(|| Error::Pack("the file was not validated".into()))
}
