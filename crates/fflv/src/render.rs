//! `fflv render` / `fflv extract`: decode selected layers to images, videos or numpy arrays.
//!
//! The output type follows the path:
//!   out.png / out.jpg          a single frame
//!   frames/%05d.png, dir/      one image per frame (named by frame index)
//!   out.mp4 / .m4v / .mov      H.264 (4:2:0, BT.709); .mov with alpha → PNG-in-MOV (RGBA)
//!   out.webm                   VP9 (with alpha: yuva420p)
//!   out.mkv                    FFV1, lossless (RGB or RGBA)
//!   out.npy                    all frames stacked (N×H×W×C uint8), written as they come

use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use lvf::Fps;

use crate::decode::Reader;
use crate::error::{Error, Result};
use crate::image::{encode_png, Image};
use crate::media::ffmpeg;

/// Called with (frames done, frames in total) after each frame; an error stops the work.
pub type Progress<'a> = &'a mut dyn FnMut(u32, u32) -> Result<()>;

fn out_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Output(msg.into()))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

/// printf-style frame pattern: `%d`, `%5d`, `%05d` and `%%`.
pub fn format_pattern(pattern: &str, index: u32) -> Result<String> {
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            out.push('%');
            continue;
        }
        let mut spec = String::new();
        while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
            spec.push(d);
            chars.next();
        }
        if chars.next() != Some('d') {
            return out_err(format!("unsupported pattern {pattern:?}: use %d or %05d"));
        }
        let zero = spec.starts_with('0');
        let width: usize = spec.parse().unwrap_or(0);
        out.push_str(&if zero { format!("{index:0width$}") } else { format!("{index:width$}") });
    }
    Ok(out)
}

pub trait Sink {
    fn write(&mut self, index: u32, img: &Image) -> Result<()>;
    fn close(self: Box<Self>) -> Result<()>;
}

// ------------------------------------------------------------------------------------------------
// PNG files (encoded on worker threads, overlapping with decoding)
// ------------------------------------------------------------------------------------------------
struct PngSink {
    pattern: Option<String>,
    single: Option<PathBuf>,
    count: u32,
    tx: Option<SyncSender<(PathBuf, Image)>>,
    workers: Vec<JoinHandle<Result<()>>>,
}

impl PngSink {
    fn new(pattern: Option<String>, single: Option<PathBuf>) -> PngSink {
        let (tx, rx): (SyncSender<(PathBuf, Image)>, Receiver<(PathBuf, Image)>) = sync_channel(8);
        let rx = Arc::new(Mutex::new(rx));
        let n = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 6);
        let workers = (0..n)
            .map(|_| {
                let rx = rx.clone();
                std::thread::spawn(move || -> Result<()> {
                    loop {
                        let job = rx.lock().unwrap().recv();
                        let Ok((path, img)) = job else { return Ok(()) };
                        ensure_parent(&path)?;
                        fs::write(&path, encode_png(img.view(), true)?)?;
                    }
                })
            })
            .collect();
        PngSink { pattern, single, count: 0, tx: Some(tx), workers }
    }
}

impl Sink for PngSink {
    fn write(&mut self, index: u32, img: &Image) -> Result<()> {
        let path = match (&self.single, &self.pattern) {
            (Some(p), _) => {
                if self.count > 0 {
                    return out_err(format!(
                        "{} holds one frame; use a pattern like frames/%05d.png or a directory",
                        p.display()
                    ));
                }
                p.clone()
            }
            (None, Some(pat)) => PathBuf::from(format_pattern(pat, index)?),
            (None, None) => unreachable!(),
        };
        self.count += 1;
        if self.tx.as_ref().unwrap().send((path, img.clone())).is_err() {
            // a worker failed; its error is reported by close()
            return Ok(());
        }
        Ok(())
    }

    fn close(mut self: Box<Self>) -> Result<()> {
        drop(self.tx.take());
        let mut first = Ok(());
        for w in self.workers.drain(..) {
            let r = w.join().unwrap_or_else(|_| out_err("PNG writer thread panicked"));
            if first.is_ok() {
                first = r;
            }
        }
        first
    }
}

// ------------------------------------------------------------------------------------------------
// FFmpeg pipes (JPEG, video)
// ------------------------------------------------------------------------------------------------
struct FfmpegSink {
    output: PathBuf,
    kind: FfKind,
    fps: Fps,
    crf: u32,
    alpha: bool,
    child: Option<(Child, ChildStdin, JoinHandle<Vec<u8>>)>,
    size: (u32, u32),
    count: u32,
    single: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum FfKind {
    Jpeg,
    H264,
    PngMov,
    Vp9,
    Ffv1,
}

impl FfKind {
    /// 4:2:0 outputs need even sizes.
    fn even(self) -> bool {
        matches!(self, FfKind::H264 | FfKind::Vp9)
    }
}

const BT709: [&str; 8] =
    ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"];

impl FfmpegSink {
    fn start(&mut self, index: u32, w: u32, h: u32, channels: u8) -> Result<()> {
        let in_fmt = if channels == 4 { "rgba" } else { "rgb24" };
        let mut cmd = ffmpeg();
        cmd.arg("-y").args(["-f", "rawvideo", "-pix_fmt", in_fmt, "-s", &format!("{w}x{h}")]);
        cmd.args(["-r", &format!("{}/{}", self.fps.num, self.fps.den), "-i", "-"]);
        let yuv = "scale=out_color_matrix=bt709:out_range=tv";
        match self.kind {
            FfKind::Jpeg => {
                cmd.args(["-q:v", "2"]);
                if self.single {
                    cmd.args(["-frames:v", "1", "-update", "1"]);
                } else {
                    cmd.args(["-start_number", &index.to_string()]);
                }
            }
            FfKind::H264 => {
                cmd.args(["-vf", yuv, "-c:v", "libx264", "-preset", "medium", "-crf", &self.crf.to_string()]);
                cmd.args(["-pix_fmt", "yuv420p"]).args(BT709);
            }
            FfKind::Vp9 => {
                let pix = if self.alpha { "yuva420p" } else { "yuv420p" };
                cmd.args(["-vf", yuv, "-c:v", "libvpx-vp9", "-crf", &self.crf.to_string(), "-b:v", "0"]);
                cmd.args(["-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-pix_fmt", pix]).args(BT709);
            }
            FfKind::Ffv1 => {
                cmd.args(["-c:v", "ffv1", "-pix_fmt", if self.alpha { "bgra" } else { "bgr0" }]);
            }
            FfKind::PngMov => {
                cmd.args(["-c:v", "png", "-pix_fmt", "rgba"]);
            }
        }
        cmd.arg(&self.output);
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::Media(format!("cannot start ffmpeg for {}: {e}", self.output.display())))?;
        let stdin = child.stdin.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let log = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });
        self.child = Some((child, stdin, log));
        self.size = (w, h);
        Ok(())
    }

    fn failure(&mut self) -> Error {
        let Some((mut child, stdin, log)) = self.child.take() else {
            return Error::Media("ffmpeg is not running".into());
        };
        drop(stdin);
        let status = child.wait();
        let err = log.join().unwrap_or_default();
        Error::Media(format!(
            "ffmpeg failed writing {} ({}): {}",
            self.output.display(),
            status.map(|s| s.to_string()).unwrap_or_else(|e| e.to_string()),
            String::from_utf8_lossy(&err).trim()
        ))
    }
}

/// Pad to even width/height by repeating the last column / row.
fn pad_even(img: &Image) -> Option<Image> {
    let (w, h, c) = (img.width as usize, img.height as usize, img.channels as usize);
    if w % 2 == 0 && h % 2 == 0 {
        return None;
    }
    let (pw, ph) = (w + w % 2, h + h % 2);
    let mut data = Vec::with_capacity(pw * ph * c);
    for y in 0..ph {
        let row = &img.data[y.min(h - 1) * w * c..][..w * c];
        data.extend_from_slice(row);
        if pw > w {
            data.extend_from_slice(&row[(w - 1) * c..]);
        }
    }
    Some(Image { width: pw as u32, height: ph as u32, channels: img.channels, data })
}

impl Sink for FfmpegSink {
    fn write(&mut self, index: u32, img: &Image) -> Result<()> {
        if self.single && self.count > 0 {
            return out_err(format!(
                "{} holds one frame; use a pattern like frames/%05d.png or a directory",
                self.output.display()
            ));
        }
        let padded = if self.kind.even() { pad_even(img) } else { None };
        let img = padded.as_ref().unwrap_or(img);
        if self.child.is_none() {
            self.start(index, img.width, img.height, img.channels)?;
        } else if self.size != (img.width, img.height) {
            return out_err(format!("frame size changed to {}x{} within one video", img.width, img.height));
        }
        let (_, stdin, _) = self.child.as_mut().unwrap();
        if stdin.write_all(&img.data).is_err() {
            return Err(self.failure());
        }
        self.count += 1;
        Ok(())
    }

    fn close(mut self: Box<Self>) -> Result<()> {
        let Some((mut child, stdin, log)) = self.child.take() else { return Ok(()) };
        drop(stdin);
        let status = child.wait()?;
        let err = log.join().unwrap_or_default();
        if !status.success() {
            return Err(Error::Media(format!(
                "ffmpeg failed writing {} ({status}): {}",
                self.output.display(),
                String::from_utf8_lossy(&err).trim()
            )));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------
// .npy
// ------------------------------------------------------------------------------------------------
struct NpySink {
    path: PathBuf,
    out: Option<BufWriter<File>>,
    expected: u32,
    count: u32,
    shape: (u32, u32, u8),
}

/// A NumPy v1.0 header for uint8 data of `shape`, padded to `len` bytes (a multiple of 64).
fn npy_header(shape: &str, len: usize) -> Vec<u8> {
    let dict = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': {shape}, }}");
    let mut h = b"\x93NUMPY\x01\x00".to_vec();
    h.extend_from_slice(&((len - 10) as u16).to_le_bytes());
    h.extend_from_slice(dict.as_bytes());
    h.resize(len - 1, b' ');
    h.push(b'\n');
    h
}

const NPY_HEADER_LEN: usize = 128;

impl Sink for NpySink {
    fn write(&mut self, _index: u32, img: &Image) -> Result<()> {
        if self.out.is_none() {
            ensure_parent(&self.path)?;
            self.shape = (img.width, img.height, img.channels);
            let mut f = BufWriter::new(File::create(&self.path)?);
            f.write_all(&npy_header(&self.shape_str(self.expected), NPY_HEADER_LEN))?;
            self.out = Some(f);
        } else if (img.width, img.height, img.channels) != self.shape {
            return out_err("frame shape changed within one .npy");
        }
        self.out.as_mut().unwrap().write_all(&img.data)?;
        self.count += 1;
        Ok(())
    }

    fn close(mut self: Box<Self>) -> Result<()> {
        match self.out.take() {
            None => {
                ensure_parent(&self.path)?;
                fs::write(&self.path, npy_header("(0,)", 64))?;
            }
            Some(f) => {
                let mut f = f.into_inner().map_err(|e| Error::Io(e.into_error()))?;
                if self.count != self.expected {
                    f.seek(SeekFrom::Start(0))?;
                    f.write_all(&npy_header(&self.shape_str(self.count), NPY_HEADER_LEN))?;
                }
                f.flush()?;
            }
        }
        Ok(())
    }
}

impl NpySink {
    fn shape_str(&self, n: u32) -> String {
        let (w, h, c) = self.shape;
        format!("({n}, {h}, {w}, {c})")
    }
}

// ------------------------------------------------------------------------------------------------
/// A sink for `output`; `frames` is how many frames will be written (for .npy).
pub fn open_sink(output: &str, fps: Fps, alpha: bool, crf: u32, frames: u32) -> Result<Box<dyn Sink>> {
    let path = Path::new(output);
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if output.ends_with('/') || output.ends_with(std::path::MAIN_SEPARATOR) || path.is_dir() {
        let pattern = path.join("%06d.png").to_string_lossy().into_owned();
        return Ok(Box::new(PngSink::new(Some(pattern), None)));
    }
    let ff = |kind, single| -> Box<dyn Sink> {
        Box::new(FfmpegSink {
            output: path.to_path_buf(),
            kind,
            fps,
            crf,
            alpha,
            child: None,
            size: (0, 0),
            count: 0,
            single,
        })
    };
    match ext.as_str() {
        "png" if output.contains('%') => Ok(Box::new(PngSink::new(Some(output.into()), None))),
        "png" => Ok(Box::new(PngSink::new(None, Some(path.to_path_buf())))),
        "jpg" | "jpeg" => {
            if alpha {
                return out_err("JPEG has no alpha channel; use .png");
            }
            let single = !output.contains('%');
            if !single {
                format_pattern(output, 0)?;
            }
            ensure_parent(path)?;
            Ok(ff(FfKind::Jpeg, single))
        }
        "npy" => Ok(Box::new(NpySink { path: path.into(), out: None, expected: frames, count: 0, shape: (0, 0, 0) })),
        "mp4" | "m4v" | "mov" | "webm" | "mkv" => {
            let kind = match ext.as_str() {
                "webm" => FfKind::Vp9,
                "mkv" => FfKind::Ffv1,
                "mov" if alpha => FfKind::PngMov,
                _ if alpha => {
                    return out_err("H.264 has no alpha channel: use .mov, .webm, .mkv or PNG frames for --transparent")
                }
                _ => FfKind::H264,
            };
            ensure_parent(path)?;
            Ok(ff(kind, false))
        }
        _ => out_err(format!(
            "don't know how to write {output:?}: use .png/.jpg (optionally with %d), a directory, .mp4/.mov/.webm/.mkv or .npy"
        )),
    }
}

#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// Layers to show (default: the file's visible layers).
    pub layers: Option<Vec<String>>,
    pub hide: Vec<String>,
    /// Layer opacities to use instead of the file's (layer id or index, 0–1).
    pub opacity: Vec<(String, f32)>,
    /// Draw order (layer ids or indices, bottom first) instead of z; layers not listed go on top.
    pub order: Option<Vec<String>>,
    pub start: u32,
    pub end: Option<u32>,
    /// No background: RGBA output.
    pub transparent: bool,
    /// Quality of lossy video outputs.
    pub crf: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            layers: None,
            hide: Vec::new(),
            opacity: Vec::new(),
            order: None,
            start: 0,
            end: None,
            transparent: false,
            crf: 18,
        }
    }
}

fn drain(
    frames: impl Iterator<Item = Result<(u32, Image)>>,
    mut sink: Box<dyn Sink>,
    total: u32,
    mut progress: Option<Progress>,
) -> Result<u32> {
    let mut n = 0;
    let mut result = Ok(());
    for item in frames {
        match item.and_then(|(f, img)| sink.write(f, &img)) {
            Ok(()) => {
                n += 1;
                if let Some(Err(e)) = progress.as_mut().map(|p| p(n, total)) {
                    result = Err(e);
                    break;
                }
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }
    let closed = sink.close();
    result?;
    closed?;
    Ok(n)
}

/// Composite the chosen layers over [start, end) into `output`. Returns the frames written.
pub fn render(path: &Path, output: &str, o: &RenderOptions, progress: Option<Progress>) -> Result<u32> {
    let r = Reader::open(path)?;
    let end = o.end.unwrap_or(r.frame_count());
    let mut frames = r.frames(o.start, Some(end), o.layers.as_deref(), &o.hide, o.transparent)?;
    for (key, opacity) in &o.opacity {
        frames.set_opacity(r.layer(key)?, *opacity);
    }
    if let Some(order) = &o.order {
        frames.set_order(&order.iter().map(|k| r.layer(k)).collect::<Result<Vec<_>>>()?);
    }
    let total = end.saturating_sub(o.start);
    let sink = open_sink(output, r.fps(), o.transparent, o.crf, total)?;
    drain(frames, sink, total, progress)
}

/// One layer's own pixels (RGBA, content size) for the frames where it is active.
pub fn extract(
    path: &Path,
    layer: &str,
    output: &str,
    start: Option<u32>,
    end: Option<u32>,
    crf: u32,
    progress: Option<Progress>,
) -> Result<u32> {
    let r = Reader::open(path)?;
    let l = &r.layers()[r.layer(layer)?];
    let s = start.map_or(l.start_frame, |s| s.max(l.start_frame));
    let e = end.map_or(l.end_frame, |e| e.min(l.end_frame));
    let total = e.saturating_sub(s);
    let frames = r.layer_frames(layer, start, end)?;
    let sink = open_sink(output, r.fps(), true, crf, total)?;
    drain(frames, sink, total.max(1), progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns() {
        assert_eq!(format_pattern("f/%05d.png", 42).unwrap(), "f/00042.png");
        assert_eq!(format_pattern("%d_%%.png", 7).unwrap(), "7_%.png");
        assert_eq!(format_pattern("%3d.png", 7).unwrap(), "  7.png");
        assert!(format_pattern("%s.png", 7).is_err());
    }

    #[test]
    fn npy_header_is_aligned() {
        let h = npy_header("(3, 2, 4, 3)", NPY_HEADER_LEN);
        assert_eq!(h.len(), 128);
        assert_eq!(h[127], b'\n');
        assert_eq!(npy_header("(0,)", 64).len(), 64);
    }
}
