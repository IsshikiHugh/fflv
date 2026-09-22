//! Streaming writer: images in, .lvd out.
//!
//! Semantics
//!  * Every [`Writer::write`] appends one composite frame; all layers are encoded (in parallel) and
//!    the frame is written immediately, so memory use does not grow with the length of the video.
//!  * A layer starts at the first frame it is given an image. If it is omitted later it keeps
//!    showing its last image ("sticky"), until [`Writer::end_layer`] or the end of the file.
//!  * Key frames sit on the global grid (every `gop` frames) plus each layer's first frame, so
//!    every multiple of `gop` is a random-access point.
//!  * The file is written to a hidden temporary file beside the destination; [`Writer::close`]
//!    validates it and then atomically renames it over the destination (spec B.11).
//!  * If encoding fails inside `write`, the layers' encoders may be out of step with the file, so
//!    the writer refuses to continue: further `write` / `close` calls fail; use [`Writer::abort`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lvf::meta::{self, AudioMeta, Canvas, Rect, VideoLayerSpec, Z};
use lvf::{encode_meta, meta_capacity_for, pts_us, publish, temp_path_for, Cau, Fps, LvfWriter, Report, VideoEntry};
use rayon::prelude::*;

use crate::audio::{encode_audio, AudioTrack};
use crate::codec::{EncodeOptions, Speed};
use crate::encode::{LayerEncoder, Prepared};
use crate::error::{Error, Result};
use crate::image::{png_size, ImageRef};

#[derive(Clone, Debug)]
pub struct WriterOptions {
    /// Random-access interval in frames (default: 2 seconds).
    pub gop: Option<u32>,
    pub background: String,
    pub crf: u32,
    pub speed: Speed,
    /// Validate the finished file before publishing it.
    pub check: bool,
    /// Encoding threads (default: all cores, at most 16).
    pub threads: Option<usize>,
}

impl Default for WriterOptions {
    fn default() -> Self {
        WriterOptions {
            gop: None,
            background: "#000000".into(),
            crf: 32,
            speed: Speed::Balanced,
            check: true,
            threads: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerOptions {
    pub alpha: bool,
    /// Keep pixel values exactly (bigger files).
    pub lossless: bool,
    /// Where the layer sits on the canvas (default: the whole canvas); images must be w×h.
    pub rect: Option<Rect>,
    pub z: Option<f64>,
    pub name: Option<String>,
    pub blend: String,
    pub opacity: f64,
    pub visible: bool,
    pub crf: Option<u32>,
    /// Quality of the alpha plane (default: `crf`).
    pub alpha_crf: Option<u32>,
    pub speed: Option<Speed>,
    /// Overrides the cpu-used of the speed preset.
    pub cpu_used: Option<i32>,
}

impl Default for LayerOptions {
    fn default() -> Self {
        LayerOptions {
            alpha: false,
            lossless: false,
            rect: None,
            z: None,
            name: None,
            blend: "normal".into(),
            opacity: 1.0,
            visible: true,
            crf: None,
            alpha_crf: None,
            speed: None,
            cpu_used: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StillOptions {
    /// Default: the image's own size at the top-left corner.
    pub rect: Option<Rect>,
    pub start: u32,
    /// Frames [start, end); None: to the end of the file.
    pub end: Option<u32>,
    pub z: Option<f64>,
    pub name: Option<String>,
    pub blend: String,
    pub opacity: f64,
    pub visible: bool,
}

impl Default for StillOptions {
    fn default() -> Self {
        StillOptions {
            rect: None,
            start: 0,
            end: None,
            z: None,
            name: None,
            blend: "normal".into(),
            opacity: 1.0,
            visible: true,
        }
    }
}

struct VideoLayer {
    id: String,
    name: String,
    z: Z,
    rect: Rect,
    alpha: bool,
    lossless: bool,
    blend: String,
    opacity: f64,
    visible: bool,
    encoder: LayerEncoder,
    start: Option<u32>,
    end: Option<u32>,
    last: Option<Prepared>,
}

struct StillLayer {
    id: String,
    name: String,
    z: Z,
    rect: Rect,
    png: Vec<u8>,
    start: u32,
    end: Option<u32>,
    blend: String,
    opacity: f64,
    visible: bool,
    offset: u64,
}

enum Layer {
    Video(Box<VideoLayer>),
    Still(StillLayer),
}

impl Layer {
    fn id(&self) -> &str {
        match self {
            Layer::Video(v) => &v.id,
            Layer::Still(s) => &s.id,
        }
    }
}

pub struct Writer {
    path: PathBuf,
    part: PathBuf,
    width: u32,
    height: u32,
    fps: Fps,
    gop: u32,
    background: String,
    options: EncodeOptions,
    check: bool,
    layers: Vec<Layer>,
    audio: Option<AudioTrack>,
    audio_pos: usize,
    out: Option<LvfWriter>,
    frames: u32,
    closed: bool,
    broken: Option<String>,
    pool: rayon::ThreadPool,
    pub report: Option<Report>,
    /// Test hook: the encoder of this layer fails.
    #[cfg(test)]
    fail_encode: Option<String>,
}

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Writer(msg.into()))
}

impl Writer {
    pub fn create(path: impl AsRef<Path>, width: u32, height: u32, fps: Fps, opts: WriterOptions) -> Result<Writer> {
        if width == 0 || height == 0 {
            return err(format!("size must be positive, got {width}x{height}"));
        }
        let gop = match opts.gop {
            Some(0) => return err("gop must be positive"),
            Some(g) => g,
            None => ((2.0 * fps.as_f64()).round() as u32).max(1),
        };
        let background = meta::check_background(&opts.background)?;
        let options = EncodeOptions::new(opts.crf, opts.speed)?;
        let threads =
            opts.threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()).min(16));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .map_err(|e| Error::Writer(format!("cannot start encoding threads: {e}")))?;
        let path = path.as_ref().to_path_buf();
        Ok(Writer {
            part: temp_path_for(&path),
            path,
            width,
            height,
            fps,
            gop,
            background,
            options,
            check: opts.check,
            layers: Vec::new(),
            audio: None,
            audio_pos: 0,
            out: None,
            frames: 0,
            closed: false,
            broken: None,
            pool,
            report: None,
            #[cfg(test)]
            fail_encode: None,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn fps(&self) -> Fps {
        self.fps
    }

    pub fn gop(&self) -> u32 {
        self.gop
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn frame_count(&self) -> u32 {
        self.frames
    }

    pub fn layer_ids(&self) -> Vec<String> {
        self.layers.iter().map(|l| l.id().to_string()).collect()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    // -------------------------------------------------------------- declaration (before writing)
    fn declare(&self, what: &str, id: &str) -> Result<String> {
        if self.closed {
            return err("writer is closed");
        }
        if self.out.is_some() {
            return err(format!(
                "{what} must be called before the first write(): every composite frame has to list every layer"
            ));
        }
        let taken: Vec<&str> = self.layers.iter().map(|l| l.id()).collect();
        Ok(meta::check_id(id, &taken)?)
    }

    fn z_or_next(&self, z: Option<f64>) -> Result<Z> {
        match z {
            Some(z) => Ok(meta::check_z_f64(z)?),
            None => Ok(Z(self.layers.len() as f64)),
        }
    }

    /// Declare a video layer.
    pub fn add_layer(&mut self, id: &str, o: LayerOptions) -> Result<()> {
        let id = self.declare("add_layer()", id)?;
        let rect = o.rect.unwrap_or(Rect { x: 0, y: 0, w: self.width, h: self.height });
        let rect = meta::make_rect(rect.x, rect.y, rect.w as i64, rect.h as i64)?;
        let crf = o.crf.unwrap_or(self.options.crf);
        let speed = o.speed.unwrap_or(self.options.speed);
        let opts = EncodeOptions { cpu_used: o.cpu_used, ..EncodeOptions::new(crf, speed)? };
        let alpha_opts =
            EncodeOptions { cpu_used: o.cpu_used, ..EncodeOptions::new(o.alpha_crf.unwrap_or(crf), speed)? };
        let blend = meta::check_blend(&o.blend)?;
        let opacity = meta::check_opacity(o.opacity)?;
        let z = self.z_or_next(o.z)?;
        let (w, h, fps) = (rect.w, rect.h, self.fps);
        let encoder = LayerEncoder::with_alpha_options(w, h, fps, o.alpha, o.lossless, &opts, &alpha_opts, &id)?;
        self.layers.push(Layer::Video(Box::new(VideoLayer {
            name: o.name.unwrap_or_else(|| id.clone()),
            id,
            z,
            rect,
            alpha: o.alpha,
            lossless: o.lossless,
            blend,
            opacity,
            visible: o.visible,
            encoder,
            start: None,
            end: None,
            last: None,
        })));
        Ok(())
    }

    /// Declare a still layer (PNG bytes) shown in frames [start, end).
    pub fn add_still(&mut self, id: &str, png: Vec<u8>, o: StillOptions) -> Result<()> {
        let id = self.declare("add_still()", id)?;
        let rect = match o.rect {
            Some(r) => meta::make_rect(r.x, r.y, r.w as i64, r.h as i64)?,
            None => {
                let (w, h) = png_size(&png)?;
                meta::make_rect(0, 0, w as i64, h as i64)?
            }
        };
        if o.end.is_some_and(|e| e <= o.start) {
            return err(format!("still {id:?}: bad frame range [{}, {})", o.start, o.end.unwrap()));
        }
        let blend = meta::check_blend(&o.blend)?;
        let opacity = meta::check_opacity(o.opacity)?;
        let z = self.z_or_next(o.z)?;
        self.layers.push(Layer::Still(StillLayer {
            name: o.name.unwrap_or_else(|| id.clone()),
            id,
            z,
            rect,
            png,
            start: o.start,
            end: o.end,
            blend,
            opacity,
            visible: o.visible,
            offset: 0,
        }));
        Ok(())
    }

    /// Audio track from any file FFmpeg can read (cut to the video length).
    pub fn set_audio(&mut self, src: &Path, bitrate: &str, channels: u32) -> Result<()> {
        self.declare("set_audio()", "audio")?;
        self.audio = Some(encode_audio(src, None, bitrate, channels)?);
        Ok(())
    }

    pub fn set_audio_track(&mut self, track: AudioTrack) -> Result<()> {
        self.declare("set_audio()", "audio")?;
        self.audio = Some(track);
        Ok(())
    }

    // -------------------------------------------------------------- writing
    fn meta(&self, frame_count: u32, last: bool) -> lvf::Meta {
        // Placeholder numbers while drafting, so the space reserved for the metadata is large enough.
        let big = 1_000_000_000;
        let layers = self
            .layers
            .iter()
            .map(|l| match l {
                Layer::Video(v) => meta::video_layer(VideoLayerSpec {
                    id: &v.id,
                    name: &v.name,
                    z: v.z,
                    rect: v.rect,
                    start: if last { v.start.unwrap_or(0) } else { big },
                    end: if last { v.end.unwrap_or(frame_count) } else { big },
                    fps: self.fps,
                    alpha: v.alpha,
                    lossless: v.lossless,
                    blend: &v.blend,
                    opacity: v.opacity,
                    visible: v.visible,
                }),
                Layer::Still(s) => meta::still_layer(
                    &s.id,
                    &s.name,
                    s.z,
                    s.rect,
                    if last { s.start } else { big },
                    if last { s.end.unwrap_or(frame_count) } else { big },
                    s.offset,
                    s.png.len() as u64,
                    &s.blend,
                    s.opacity,
                    s.visible,
                ),
            })
            .collect();
        let canvas = Canvas { width: self.width, height: self.height, background: self.background.clone() };
        let audio: Option<AudioMeta> = self.audio.as_ref().map(|a| a.meta());
        meta::file_meta(canvas, self.fps, if last { frame_count } else { big }, self.gop, layers, audio)
    }

    fn begin(&mut self) -> Result<()> {
        if self.layers.is_empty() {
            return err("declare at least one layer before writing");
        }
        let mut offset = 0;
        let mut resources = Vec::new();
        for l in &mut self.layers {
            if let Layer::Still(s) = l {
                s.offset = offset;
                offset += s.png.len() as u64;
                resources.extend_from_slice(&s.png);
            }
        }
        let draft = encode_meta(&self.meta(0, false))?;
        let mut out = LvfWriter::create(&self.part)?;
        out.begin(&draft, &resources, Some(meta_capacity_for(draft.len())))?;
        self.out = Some(out);
        Ok(())
    }

    fn check_usable(&self) -> Result<()> {
        if self.closed {
            return err("writer is closed");
        }
        if let Some(e) = &self.broken {
            return err(format!(
                "a previous write() failed while encoding ({e}); the encoders are out of step with the file, so it cannot be completed — call abort()"
            ));
        }
        Ok(())
    }

    /// Append one composite frame; images by layer id. Returns the frame index.
    pub fn write(&mut self, images: &[(&str, ImageRef)]) -> Result<u32> {
        self.check_usable()?;
        let mut given: HashMap<&str, ImageRef> = HashMap::new();
        for (id, img) in images {
            given.insert(id, *img);
        }
        let unknown: Vec<&str> = {
            let mut u: Vec<&str> = given
                .keys()
                .copied()
                .filter(|id| !self.layers.iter().any(|l| matches!(l, Layer::Video(v) if v.id == *id)))
                .collect();
            u.sort();
            u
        };
        if !unknown.is_empty() {
            let stills =
                unknown.iter().any(|id| self.layers.iter().any(|l| matches!(l, Layer::Still(s) if s.id == *id)));
            let hint = if stills { " (still layers take no per-frame images)" } else { "" };
            let mut known: Vec<&str> = self
                .layers
                .iter()
                .filter_map(|l| if let Layer::Video(v) = l { Some(v.id.as_str()) } else { None })
                .collect();
            known.sort();
            return err(format!("unknown video layer(s) {unknown:?}{hint}; declared: {known:?}"));
        }
        if self.layers.is_empty() {
            return err("declare at least one layer before writing");
        }
        let f = self.frames;
        // Check and convert every image before changing any state: a bad image leaves the writer as it was.
        let mut prepared: HashMap<usize, Prepared> = HashMap::new();
        {
            let layers = &self.layers;
            let jobs: Vec<(usize, &VideoLayer, ImageRef)> = layers
                .iter()
                .enumerate()
                .filter_map(|(i, l)| match l {
                    Layer::Video(v) => given.get(v.id.as_str()).map(|img| (i, v.as_ref(), *img)),
                    _ => None,
                })
                .collect();
            for (_, v, _) in &jobs {
                if let Some(end) = v.end {
                    return err(format!("layer {:?} was ended at frame {end}; a layer is one contiguous range", v.id));
                }
            }
            let results: Vec<Result<(usize, Prepared)>> =
                self.pool.install(|| jobs.par_iter().map(|(i, v, img)| Ok((*i, v.encoder.prepare(*img)?))).collect());
            for r in results {
                let (i, p) = r?;
                prepared.insert(i, p);
            }
        }
        if self.out.is_none() {
            self.begin()?;
        }
        let gop = self.gop;
        for (i, l) in self.layers.iter_mut().enumerate() {
            if let Layer::Video(v) = l {
                if let Some(p) = prepared.remove(&i) {
                    v.last = Some(p);
                    v.start.get_or_insert(f);
                }
            }
        }
        match self.encode_and_write(f, gop) {
            Ok(f) => Ok(f),
            Err(e) => {
                self.broken = Some(e.to_string());
                Err(e)
            }
        }
    }

    fn encode_and_write(&mut self, f: u32, gop: u32) -> Result<u32> {
        #[cfg(test)]
        let fail = self.fail_encode.clone();
        let mut jobs: Vec<(usize, &mut VideoLayer, bool)> = self
            .layers
            .iter_mut()
            .enumerate()
            .filter_map(|(i, l)| match l {
                Layer::Video(v) if v.start.is_some() && v.end.is_none() => {
                    let key = Some(f) == v.start || f % gop == 0;
                    Some((i, v.as_mut(), key))
                }
                _ => None,
            })
            .collect();
        let encoded: Vec<Result<(usize, bool, Vec<u8>, Vec<u8>)>> = self.pool.install(|| {
            jobs.par_iter_mut()
                .map(|(i, v, key)| {
                    #[cfg(test)]
                    if fail.as_deref() == Some(v.id.as_str()) {
                        return Err(Error::Encode("encoder crashed".into()));
                    }
                    let p = v.last.as_ref().expect("an active layer has an image");
                    let (c, a) = v.encoder.encode_prepared(p, *key)?;
                    Ok((*i, *key, c, a))
                })
                .collect()
        });
        let mut results: HashMap<usize, (bool, Vec<u8>, Vec<u8>)> = HashMap::new();
        for r in encoded {
            let (i, key, c, a) = r?;
            results.insert(i, (key, c, a));
        }
        let rap = results.values().all(|(key, _, _)| *key);
        let mut entries = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            if let Layer::Video(_) = l {
                entries.push(match results.remove(&i) {
                    Some((key, c, a)) => VideoEntry::frame(i as u16, key, c, a),
                    None => VideoEntry::empty(i as u16),
                });
            }
        }
        let mut audio = Vec::new();
        if let Some(track) = &self.audio {
            let hi = pts_us(f as u64 + 1, self.fps);
            while self.audio_pos < track.packets.len() && track.packets[self.audio_pos].pts_us < hi {
                audio.push(track.packets[self.audio_pos].clone());
                self.audio_pos += 1;
            }
        }
        self.out.as_mut().unwrap().write_cau(&Cau::new(f, rap, entries, audio))?;
        self.frames += 1;
        Ok(f)
    }

    /// The layer's last frame was the previous write(); it is empty from now on.
    pub fn end_layer(&mut self, id: &str) -> Result<()> {
        let frames = self.frames;
        for l in &mut self.layers {
            if let Layer::Video(v) = l {
                if v.id == id {
                    if v.start.is_none() {
                        return err(format!("layer {id:?} has not started yet"));
                    }
                    v.end.get_or_insert(frames);
                    return Ok(());
                }
            }
        }
        err(format!("no video layer {id:?}"))
    }

    // -------------------------------------------------------------- finishing
    /// Finish, validate and publish the file. Returns the validation report (None with check off).
    pub fn close(&mut self) -> Result<Option<&Report>> {
        if self.closed {
            return Ok(self.report.as_ref());
        }
        if let Some(e) = self.broken.clone() {
            self.abort();
            return err(format!("not written: a write() failed while encoding ({e})"));
        }
        let result = self.finish();
        self.closed = true;
        match result {
            Ok(rep) => {
                self.report = rep;
                Ok(self.report.as_ref())
            }
            Err(e) => {
                self.abort();
                Err(match e {
                    Error::Invalid { .. } => Error::Writer(e.to_string()),
                    e => e,
                })
            }
        }
    }

    fn finish(&mut self) -> Result<Option<Report>> {
        if self.out.is_none() {
            return err("no frames were written");
        }
        let n = self.frames;
        for l in &mut self.layers {
            match l {
                Layer::Video(v) => {
                    if v.start.is_none() {
                        return err(format!("layer {:?} never received an image", v.id));
                    }
                    v.encoder.finish()?;
                }
                Layer::Still(s) => {
                    if s.start >= n || s.end.is_some_and(|e| e > n) {
                        let end = s.end.map_or("None".to_string(), |e| e.to_string());
                        return err(format!(
                            "still {:?} range [{}, {end}) is outside the {n} frames written",
                            s.id, s.start
                        ));
                    }
                }
            }
        }
        let data = encode_meta(&self.meta(n, true))?;
        self.out.take().unwrap().finish(None, Some(&data))?;
        Ok(publish(&self.part, &self.path, self.check)?)
    }

    /// Discard everything written so far.
    pub fn abort(&mut self) {
        self.closed = true;
        if let Some(out) = self.out.take() {
            out.abort();
        }
        let _ = std::fs::remove_file(&self.part);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.closed {
            self.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    /// After an encoder error mid-frame the other layers' encoders are a frame ahead of the file;
    /// continuing would produce a file that validates but decodes wrongly.
    #[test]
    fn a_failed_encode_makes_the_writer_refuse_to_continue() {
        let dir = std::env::temp_dir().join(format!("fflv-writer-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("broken.lvd");
        let opts = WriterOptions { gop: Some(4), speed: Speed::Fast, ..Default::default() };
        let mut w = Writer::create(&path, 32, 32, Fps::new(30, 1).unwrap(), opts).unwrap();
        let lossless = LayerOptions { lossless: true, ..Default::default() };
        w.add_layer("a", lossless.clone()).unwrap();
        w.add_layer("b", lossless).unwrap();
        let img = |v: u8| Image::filled(32, 32, &[v, v, v]);
        for f in 0..3 {
            let (a, b) = (img(f), img(f));
            w.write(&[("a", a.view()), ("b", b.view())]).unwrap();
        }
        w.fail_encode = Some("b".into());
        let (a, b) = (img(3), img(3));
        assert!(w.write(&[("a", a.view()), ("b", b.view())]).unwrap_err().to_string().contains("encoder crashed"));
        w.fail_encode = None;
        let e = w.write(&[("a", a.view()), ("b", b.view())]).unwrap_err();
        assert!(e.to_string().contains("failed while encoding"), "{e}");
        let e = w.close().unwrap_err();
        assert!(e.to_string().contains("not written"), "{e}");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }
}
