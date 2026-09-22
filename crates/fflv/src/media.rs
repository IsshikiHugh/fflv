//! FFmpeg and ffprobe as subprocesses: probing source media, decoding video to raw frames at a
//! layer's size, converting still images to PNG.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

use lvf::Fps;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::image::{Image, PNG_SIGNATURE};

pub fn ffmpeg() -> Command {
    let mut c = Command::new("ffmpeg");
    c.args(["-hide_banner", "-nostdin", "-loglevel", "error"]);
    c
}

fn describe(cmd: &Command) -> String {
    let mut parts = vec![cmd.get_program().to_string_lossy().into_owned()];
    parts.extend(cmd.get_args().map(|a| a.to_string_lossy().into_owned()));
    parts.join(" ")
}

fn spawn_error(cmd: &Command, what: &str, e: std::io::Error) -> Error {
    let prog = cmd.get_program().to_string_lossy().into_owned();
    if e.kind() == std::io::ErrorKind::NotFound {
        Error::Media(format!("{what}: {prog} not found on PATH (install FFmpeg)"))
    } else {
        Error::Media(format!("{what}: cannot start {prog}: {e}"))
    }
}

fn stderr_tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.trim().lines().collect();
    lines[lines.len().saturating_sub(15)..].join("\n  ")
}

/// Run to completion; returns stdout. Fails with the tail of stderr and the command line.
pub fn run(cmd: &mut Command, what: &str) -> Result<Vec<u8>> {
    let out = cmd.stdin(Stdio::null()).output().map_err(|e| spawn_error(cmd, what, e))?;
    if !out.status.success() {
        return Err(Error::Media(format!(
            "{what} failed ({} {}):\n  {}\n  command: {}",
            cmd.get_program().to_string_lossy(),
            out.status,
            stderr_tail(&out.stderr),
            describe(cmd)
        )));
    }
    Ok(out.stdout)
}

/// Pixel formats FFmpeg knows to carry alpha.
fn alpha_formats() -> &'static HashSet<String> {
    static FORMATS: OnceLock<HashSet<String>> = OnceLock::new();
    FORMATS.get_or_init(|| {
        let mut cmd = Command::new("ffprobe");
        cmd.args(["-v", "error", "-show_pixel_formats", "-of", "json"]);
        let Ok(out) = run(&mut cmd, "ffprobe -show_pixel_formats") else { return HashSet::new() };
        let v: Value = serde_json::from_slice(&out).unwrap_or(Value::Null);
        v["pixel_formats"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|f| f["flags"]["alpha"].as_i64() == Some(1))
                    .filter_map(|f| f["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    })
}

pub fn pix_fmt_has_alpha(name: &str) -> bool {
    alpha_formats().contains(name)
}

#[derive(Clone, Debug)]
pub struct VideoSource {
    pub path: PathBuf,
    pub codec: String,
    pub pix_fmt: String,
    /// Decoder to force (libvpx for WebM with alpha: FFmpeg's own VP8/VP9 decoders drop it).
    pub decoder: Option<String>,
    pub has_alpha: bool,
}

pub fn open_video_source(path: &Path) -> Result<VideoSource> {
    if !path.exists() {
        return Err(Error::Media(format!("source file not found: {}", path.display())));
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut cmd = Command::new("ffprobe");
    cmd.args(["-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json"]).arg(path);
    let out = run(&mut cmd, &format!("ffprobe {name}"))?;
    let v: Value = serde_json::from_slice(&out).map_err(|e| Error::Media(format!("ffprobe {name}: {e}")))?;
    let info = v["streams"].get(0).ok_or_else(|| Error::Media(format!("{} has no video stream", path.display())))?;
    let codec = info["codec_name"].as_str().unwrap_or("?").to_string();
    let pix_fmt = info["pix_fmt"].as_str().unwrap_or("").to_string();
    let webm_alpha = info["tags"]
        .as_object()
        .and_then(|t| t.iter().find(|(k, _)| k.eq_ignore_ascii_case("alpha_mode")))
        .is_some_and(|(_, v)| v.as_str() == Some("1") || v.as_i64() == Some(1));
    let decoder = if webm_alpha {
        match codec.as_str() {
            "vp9" => Some("libvpx-vp9".to_string()),
            "vp8" => Some("libvpx".to_string()),
            _ => None,
        }
    } else {
        None
    };
    let has_alpha = webm_alpha || pix_fmt_has_alpha(&pix_fmt);
    Ok(VideoSource { path: path.to_path_buf(), codec, pix_fmt, decoder, has_alpha })
}

/// Decoded frames of a source, scaled to a layer's size, from an ffmpeg pipe.
pub struct RawFrames {
    child: Child,
    stdout: ChildStdout,
    /// ffmpeg's stderr, drained on a thread (a full pipe would stall ffmpeg and with it the read)
    stderr: Option<std::thread::JoinHandle<Vec<u8>>>,
    name: String,
    width: u32,
    height: u32,
    channels: u8,
    count: u64,
    done: u64,
}

impl RawFrames {
    fn fail(&mut self) -> Error {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let err = self.stderr.take().and_then(|t| t.join().ok()).unwrap_or_default();
        Error::Media(format!(
            "decoding {} stopped after {} of {} frames: {}",
            self.name,
            self.done,
            self.count,
            stderr_tail(&err)
        ))
    }
}

impl Iterator for RawFrames {
    type Item = Result<Image>;
    fn next(&mut self) -> Option<Result<Image>> {
        if self.done >= self.count {
            return None;
        }
        let mut buf = vec![0u8; self.width as usize * self.height as usize * self.channels as usize];
        if self.stdout.read_exact(&mut buf).is_err() {
            self.done = self.count.max(self.done);
            return Some(Err(self.fail()));
        }
        self.done += 1;
        Some(Ok(Image { width: self.width, height: self.height, channels: self.channels, data: buf }))
    }
}

impl Drop for RawFrames {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Exactly `count` frames at `fps`, scaled to w×h, as RGBA (`alpha`) or RGB. Short sources are
/// extended by repeating their last frame.
pub fn raw_frames(src: &VideoSource, fps: Fps, w: u32, h: u32, count: u64, alpha: bool) -> Result<RawFrames> {
    let (fmt, channels) = if alpha { ("rgba", 4) } else { ("rgb24", 3) };
    let vf = format!("fps={fps},scale={w}:{h}:flags=bicubic,setsar=1,format={fmt},tpad=stop_mode=clone:stop=-1");
    let mut cmd = ffmpeg();
    if let Some(d) = &src.decoder {
        cmd.args(["-c:v", d]);
    }
    cmd.arg("-i").arg(&src.path);
    cmd.args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", &vf, "-frames:v", &count.to_string()]);
    cmd.args(["-f", "rawvideo", "-pix_fmt", fmt, "-"]);
    let name = src.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| spawn_error(&cmd, &format!("decoding {name}"), e))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().map(|e| std::thread::spawn(move || tail_of(e, 64 << 10)));
    Ok(RawFrames { child, stdout, stderr, name, width: w, height: h, channels, count, done: 0 })
}

/// Read a stream to its end, keeping the last `keep` bytes.
fn tail_of(mut r: impl Read, keep: usize) -> Vec<u8> {
    let mut tail = Vec::new();
    let mut buf = [0u8; 8192];
    while let Ok(n) = r.read(&mut buf) {
        if n == 0 {
            break;
        }
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > 2 * keep {
            tail.drain(..tail.len() - keep);
        }
    }
    if tail.len() > keep {
        tail.drain(..tail.len() - keep);
    }
    tail
}

/// PNG bytes for a still layer: a PNG file as is, any other image converted by FFmpeg.
pub fn still_png_from_file(path: &Path) -> Result<Vec<u8>> {
    let data = std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => Error::Media(format!("still image not found: {}", path.display())),
        _ => Error::Io(e),
    })?;
    if data.starts_with(PNG_SIGNATURE) {
        return Ok(data);
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut cmd = ffmpeg();
    cmd.arg("-i").arg(path).args(["-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"]);
    let png = run(&mut cmd, &format!("convert {name} to PNG"))?;
    if !png.starts_with(PNG_SIGNATURE) {
        return Err(Error::Media(format!("convert {name} to PNG: FFmpeg produced no image")));
    }
    Ok(png)
}

/// PNG bytes given directly (must be a PNG).
pub fn still_png_from_bytes(data: &[u8]) -> Result<Vec<u8>> {
    if !data.starts_with(PNG_SIGNATURE) {
        return Err(Error::Media("still image bytes are not a PNG".into()));
    }
    Ok(data.to_vec())
}
