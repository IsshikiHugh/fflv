//! Generate the LVF test material (spec 11.1) and a matching project file.
//!
//! ```text
//! fflv testsrc [--out test_assets] [--duration 20]
//! ```
//!
//! Sources are rendered with FFmpeg (lavfi `testsrc2` / `aevalsrc`) plus drawn overlays, and
//! stored losslessly (FFV1 in Matroska; BGRA where alpha matters). Frame numbers are drawn with a
//! small built-in bitmap font.
//!
//! Every video layer also carries a *barcode* of its global frame number (1 start cell + 16 data
//! bits + 1 parity bit, 8x8-px cells, white = 1). The player's end-to-end test reads the barcodes
//! back from the rendered canvas to prove that all layers on screen come from the same composite
//! frame; their canvas positions are written to `barcodes.json`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};

use lvf::timing::seconds_to_frame;
use lvf::Fps;
use serde_json::json;

use crate::error::{Error, Result};
use crate::image::{encode_png, Image};
use crate::media::{ffmpeg, run};

// 5x7 bitmap font --------------------------------------------------------------------------------
fn glyph(c: char) -> [&'static str; 7] {
    match c {
        '0' => ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        '1' => ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        '2' => ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        '3' => ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
        '4' => ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        '5' => ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        '6' => ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
        '7' => ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        '8' => ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        '9' => ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
        'A' => ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        'B' => ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        'F' => ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        'L' => ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        'S' => ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        'V' => ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        _ => ["00000"; 7],
    }
}

pub fn text_size(text: &str, scale: u32) -> (u32, u32) {
    ((text.chars().count() as u32 * 6 - 1) * scale, 7 * scale)
}

/// Fill [x0, x1) × [y0, y1) (clipped to the image) with `color`.
fn fill(img: &mut Image, x0: i64, y0: i64, x1: i64, y1: i64, color: &[u8]) {
    let c = img.channels as usize;
    let (w, h) = (img.width as i64, img.height as i64);
    for y in y0.max(0)..y1.min(h) {
        for x in x0.max(0)..x1.min(w) {
            let i = (y * w + x) as usize * c;
            img.data[i..i + c].copy_from_slice(&color[..c]);
        }
    }
}

/// Draw `text` with its top-left at (x, y); `bg`: box behind it, `pad` pixels larger.
fn draw_text(img: &mut Image, x: i64, y: i64, text: &str, scale: u32, color: &[u8], bg: Option<&[u8]>, pad: i64) {
    let (w, h) = text_size(text, scale);
    if let Some(bg) = bg {
        fill(img, x - pad, y - pad, x + w as i64 + pad, y + h as i64 + pad, bg);
    }
    let s = scale as i64;
    for (i, ch) in text.chars().enumerate() {
        let gx = x + i as i64 * 6 * s;
        for (r, row) in glyph(ch).iter().enumerate() {
            for (k, bit) in row.bytes().enumerate() {
                if bit == b'1' {
                    let (px, py) = (gx + k as i64 * s, y + r as i64 * s);
                    fill(img, px, py, px + s, py + s, color);
                }
            }
        }
    }
}

// Barcode ----------------------------------------------------------------------------------------
pub const BAR_CELL: u32 = 8;
pub const BAR_BITS: u32 = 16;
pub const BAR_CELLS: u32 = BAR_BITS + 2; // start marker + data + parity

pub fn barcode_cells(n: u32) -> Vec<u8> {
    let bits: Vec<u8> = (0..BAR_BITS).map(|i| ((n >> (BAR_BITS - 1 - i)) & 1) as u8).collect();
    let parity = bits.iter().sum::<u8>() & 1;
    std::iter::once(1).chain(bits).chain(std::iter::once(parity)).collect()
}

fn draw_barcode(img: &mut Image, x: i64, y: i64, n: u32, one: &[u8], zero: &[u8]) {
    let c = BAR_CELL as i64;
    for (i, bit) in barcode_cells(n).into_iter().enumerate() {
        let x0 = x + i as i64 * c;
        fill(img, x0, y, x0 + c, y + c, if bit == 1 { one } else { zero });
    }
}

// FFmpeg plumbing --------------------------------------------------------------------------------
struct Ffv1 {
    child: Child,
    what: String,
}

impl Ffv1 {
    fn new(path: &Path, w: u32, h: u32, fps: Fps, rgba: bool) -> Result<Ffv1> {
        let mut cmd = ffmpeg();
        cmd.arg("-y").args(["-f", "rawvideo", "-pix_fmt", if rgba { "rgba" } else { "rgb24" }]);
        cmd.args(["-s", &format!("{w}x{h}"), "-r", &format!("{}/{}", fps.num, fps.den), "-i", "-"]);
        cmd.args(["-c:v", "ffv1", "-level", "3", "-pix_fmt", if rgba { "bgra" } else { "bgr0" }]).arg(path);
        let child = cmd.stdin(Stdio::piped()).spawn().map_err(|e| Error::Media(format!("cannot start ffmpeg: {e}")))?;
        Ok(Ffv1 { child, what: path.display().to_string() })
    }

    fn write(&mut self, img: &Image) -> Result<()> {
        self.child.stdin.as_mut().unwrap().write_all(&img.data).map_err(|e| Error::Media(format!("{}: {e}", self.what)))
    }

    fn finish(mut self) -> Result<()> {
        drop(self.child.stdin.take());
        if !self.child.wait()?.success() {
            return Err(Error::Media(format!("ffmpeg failed while writing {}", self.what)));
        }
        Ok(())
    }
}

fn testsrc2_frames(w: u32, h: u32, fps: Fps, n: u32) -> Result<impl Iterator<Item = Result<Image>>> {
    let mut cmd = ffmpeg();
    cmd.args(["-f", "lavfi", "-i", &format!("testsrc2=size={w}x{h}:rate={}/{}", fps.num, fps.den)]);
    cmd.args(["-frames:v", &n.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
    let mut child =
        cmd.stdout(Stdio::piped()).spawn().map_err(|e| Error::Media(format!("cannot start ffmpeg: {e}")))?;
    let mut out = child.stdout.take().unwrap();
    let mut left = n;
    Ok(std::iter::from_fn(move || {
        if left == 0 {
            let _ = child.wait();
            return None;
        }
        left -= 1;
        let mut buf = vec![0u8; (w * h * 3) as usize];
        Some(match out.read_exact(&mut buf) {
            Ok(()) => Ok(Image { width: w, height: h, channels: 3, data: buf }),
            Err(_) => Err(Error::Media("testsrc2 ended early".into())),
        })
    }))
}

// Layer renderers --------------------------------------------------------------------------------
const WHITE4: [u8; 4] = [255, 255, 255, 255];
const BLACK4: [u8; 4] = [0, 0, 0, 255];

fn make_background(path: &Path, w: u32, h: u32, fps: Fps, n: u32, fps_int: u32, flash: u32) -> Result<(i64, i64)> {
    let mut out = Ffv1::new(path, w, h, fps, false)?;
    let digits = (24, 24); // clear of the 14-px flash border
    let bar = (24, 24 + 7 * 6 + 14);
    for (f, img) in testsrc2_frames(w, h, fps, n)?.enumerate() {
        let mut img = img?;
        let f = f as u32;
        draw_text(&mut img, digits.0, digits.1, &format!("F{f:05}"), 6, &[255; 3], Some(&[0; 3]), 6);
        draw_barcode(&mut img, bar.0, bar.1, f, &[255; 3], &[0; 3]);
        if f % fps_int < flash {
            // white frame around the picture, together with the beep
            let (t, wi, hi) = (14, w as i64, h as i64);
            fill(&mut img, 0, 0, wi, t, &[255; 3]);
            fill(&mut img, 0, hi - t, wi, hi, &[255; 3]);
            fill(&mut img, 0, 0, t, hi, &[255; 3]);
            fill(&mut img, wi - t, 0, wi, hi, &[255; 3]);
        }
        out.write(&img)?;
    }
    out.finish()?;
    Ok(bar)
}

/// Layer A: wide white line, B: narrow red line at the same x = (n*16) mod W.
fn make_sync_layer(path: &Path, w: u32, h: u32, fps: Fps, n: u32, which: char) -> Result<(i64, i64)> {
    let mut out = Ffv1::new(path, w, h, fps, true)?;
    let line_h = h as i64 - 60;
    let (lw, color): (i64, [u8; 4]) = if which == 'A' { (12, [255, 255, 255, 255]) } else { (4, [255, 0, 0, 255]) };
    let (tw, _) = text_size(&format!("{which} 00000"), 4);
    let digits_x = if which == 'A' { 16 } else { w as i64 - 16 - tw as i64 };
    let bar_x = if which == 'A' { 16 } else { w as i64 - 16 - (BAR_CELLS * BAR_CELL) as i64 };
    let bar_y = h as i64 - BAR_CELL as i64 - 4;
    for f in 0..n {
        let mut img = Image::filled(w, h, &[0, 0, 0, 0]);
        let cx = (f as i64 * 16) % w as i64;
        fill(&mut img, (cx - lw / 2).max(0), 0, (cx + lw / 2).min(w as i64), line_h, &color);
        draw_text(&mut img, digits_x, line_h + 6, &format!("{which} {f:05}"), 4, &WHITE4, Some(&BLACK4), 3);
        draw_barcode(&mut img, bar_x, bar_y, f, &WHITE4, &BLACK4);
        out.write(&img)?;
    }
    out.finish()?;
    Ok((bar_x, bar_y))
}

/// A moving square that exists only in [start, end); its numbers are global frame numbers.
fn make_square_layer(path: &Path, w: u32, h: u32, fps: Fps, start: u32, end: u32) -> Result<(i64, i64)> {
    let mut out = Ffv1::new(path, w, h, fps, true)?;
    let side = 100i64;
    let bar = (16, h as i64 - BAR_CELL as i64 - 4);
    for f in start..end {
        let mut img = Image::filled(w, h, &[0, 0, 0, 0]);
        let x = ((f - start) as i64 * 8) % (w as i64 - side);
        fill(&mut img, x, 0, x + side, side, &[40, 170, 255, 230]);
        draw_text(&mut img, x + 5, 36, &format!("{f:05}"), 3, &WHITE4, None, 0);
        draw_barcode(&mut img, bar.0, bar.1, f, &WHITE4, &BLACK4);
        out.write(&img)?;
    }
    out.finish()?;
    Ok(bar)
}

/// White with alpha ramping 0 → 255 left to right (top 40 rows), plus a barcode strip.
fn make_calibration_layer(path: &Path, w: u32, h: u32, fps: Fps, n: u32) -> Result<(i64, i64)> {
    let mut out = Ffv1::new(path, w, h, fps, true)?;
    let ramp_rows = 40;
    let mut base = Image::filled(w, h, &[0, 0, 0, 0]);
    for y in 0..ramp_rows {
        for x in 0..w {
            let a = (x as f64 * 255.0 / (w - 1) as f64).round() as u8;
            let i = ((y * w + x) * 4) as usize;
            base.data[i..i + 4].copy_from_slice(&[255, 255, 255, a]);
        }
    }
    let bar = ((w / 2 - BAR_CELLS * BAR_CELL / 2) as i64, ramp_rows as i64 + 8);
    for f in 0..n {
        let mut img = base.clone();
        draw_barcode(&mut img, bar.0, bar.1, f, &WHITE4, &BLACK4);
        out.write(&img)?;
    }
    out.finish()?;
    Ok(bar)
}

/// Per-pixel RGB of the lossless test layer; player/e2e computes the same values.
pub fn exact_pattern(f: u32, x: u32, y: u32) -> [u8; 3] {
    [((x * 7 + f) & 255) as u8, ((y * 13 + 3 * f) & 255) as u8, ((x ^ y ^ f) & 255) as u8]
}

/// Lossless layer (256×96): rows 0..63 an RGB pattern that changes every frame (alpha 255), rows
/// 64..79 white with alpha = x (0..255), then a barcode; everything else transparent.
fn make_exact_layer(path: &Path, fps: Fps, n: u32) -> Result<(i64, i64)> {
    let (w, h) = (256u32, 96u32);
    let mut out = Ffv1::new(path, w, h, fps, true)?;
    let bar = (0, 84);
    for f in 0..n {
        let mut img = Image::filled(w, h, &[0, 0, 0, 0]);
        for y in 0..80 {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let px = if y < 64 {
                    let [r, g, b] = exact_pattern(f, x, y);
                    [r, g, b, 255]
                } else {
                    [255, 255, 255, x as u8]
                };
                img.data[i..i + 4].copy_from_slice(&px);
            }
        }
        draw_barcode(&mut img, bar.0, bar.1, f, &WHITE4, &BLACK4);
        out.write(&img)?;
    }
    out.finish()?;
    Ok(bar)
}

fn make_logo(path: &Path, w: u32, h: u32) -> Result<()> {
    let mut img = Image::filled(w, h, &[230, 60, 120, 200]);
    let (b, wi, hi) = (5, w as i64, h as i64);
    fill(&mut img, 0, 0, wi, b, &WHITE4);
    fill(&mut img, 0, hi - b, wi, hi, &WHITE4);
    fill(&mut img, 0, 0, b, hi, &WHITE4);
    fill(&mut img, wi - b, 0, wi, hi, &WHITE4);
    let (tw, th) = text_size("LVF", 8);
    draw_text(&mut img, (wi - tw as i64) / 2, (hi - th as i64) / 2, "LVF", 8, &WHITE4, None, 0);
    std::fs::write(path, encode_png(img.view(), false)?)?;
    Ok(())
}

fn make_beeps(path: &Path, duration: f64, beep_s: f64) -> Result<()> {
    // 1 kHz tone during the first `beep_s` of every second (commas escaped for the lavfi parser).
    let e = format!("if(lt(mod(t\\,1)\\,{beep_s:.6})\\,0.6*sin(2*PI*1000*t)\\,0)");
    let mut cmd = ffmpeg();
    cmd.arg("-y").args(["-f", "lavfi", "-i", &format!("aevalsrc={e}|{e}:s=48000:d={duration}")]);
    cmd.args(["-c:a", "pcm_s16le"]).arg(path);
    run(&mut cmd, "beeps.wav")?;
    Ok(())
}

pub struct TestsrcOptions {
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: String,
    pub duration: String,
    pub gop: u32,
    pub crf: u32,
}

impl Default for TestsrcOptions {
    fn default() -> Self {
        TestsrcOptions {
            out: PathBuf::from("test_assets"),
            width: 1280,
            height: 720,
            fps: "30/1".into(),
            duration: "20".into(),
            gop: 60,
            crf: 32,
        }
    }
}

/// Render the material into `o.out`; returns the path of the project file.
pub fn generate(o: &TestsrcOptions) -> Result<PathBuf> {
    let out = &o.out;
    std::fs::create_dir_all(out)?;
    let (w, h) = (o.width, o.height);
    let fps = Fps::parse(&o.fps).map_err(Error::Meta)?;
    if fps.den != 1 {
        return Err(Error::Meta("the beep/flash pattern needs an integer frame rate".into()));
    }
    let fps_int = fps.num;
    let n = seconds_to_frame(&o.duration, fps).map_err(Error::Meta)? as u32;
    let flash = 2; // frames of white border per second; the beep lasts exactly as long
    let beep_s = flash as f64 / fps_int as f64;

    // canvas layout
    let band = [0, 250, w, 160];
    let square = [0, 430, w, 150];
    let calib = [0, h - 104, w, 64];
    let logo = [w - 240, 20, 220, 100];
    let exact = [420, 110, 256, 96];
    let (sq_start, sq_end) = (45, if n > 90 { n - 45 } else { n }); // frame 45: not on the GOP grid
    let (logo_start, logo_end) = ((n - 1).min(3 * fps_int), n.min(12 * fps_int));

    println!("rendering {n} frames at {w}x{h} into {} ...", out.display());
    let p = |name: &str| out.join(name);
    let (bg, sync_a, sync_b, sq, cal, ex) = std::thread::scope(|s| {
        let bg = s.spawn(|| make_background(&p("bg.mkv"), w, h, fps, n, fps_int, flash));
        let a = s.spawn(|| make_sync_layer(&p("sync_a.mkv"), band[2], band[3], fps, n, 'A'));
        let b = s.spawn(|| make_sync_layer(&p("sync_b.mkv"), band[2], band[3], fps, n, 'B'));
        let sq = s.spawn(|| make_square_layer(&p("square.mkv"), square[2], square[3], fps, sq_start, sq_end));
        let cal = s.spawn(|| make_calibration_layer(&p("calib.mkv"), calib[2], calib[3], fps, n));
        let ex = s.spawn(|| make_exact_layer(&p("exact.mkv"), fps, n));
        let j = |t: std::thread::ScopedJoinHandle<'_, Result<_>>| {
            t.join().unwrap_or_else(|_| Err(Error::Media("renderer panicked".into())))
        };
        (j(bg), j(a), j(b), j(sq), j(cal), j(ex))
    });
    let (bg, sync_a, sync_b, sq, cal, ex) = (bg?, sync_a?, sync_b?, sq?, cal?, ex?);
    println!("  bg.mkv, sync_a.mkv, sync_b.mkv, square.mkv, calib.mkv, exact.mkv (lossless)");
    make_logo(&p("logo.png"), logo[2], logo[3])?;
    make_beeps(&p("beeps.wav"), n as f64 / fps_int as f64, beep_s)?;
    println!("  logo.png, beeps.wav");

    let secs = |f: u32| f as f64 / fps_int as f64;
    let bars = json!({
        "bg": {"x": bg.0, "y": bg.1, "start_frame": 0, "end_frame": n},
        "sync_a": {"x": band[0] as i64 + sync_a.0, "y": band[1] as i64 + sync_a.1, "start_frame": 0, "end_frame": n},
        "sync_b": {"x": band[0] as i64 + sync_b.0, "y": band[1] as i64 + sync_b.1, "start_frame": 0, "end_frame": n},
        "square": {"x": square[0] as i64 + sq.0, "y": square[1] as i64 + sq.1, "start_frame": sq_start, "end_frame": sq_end},
        "calib": {"x": calib[0] as i64 + cal.0, "y": calib[1] as i64 + cal.1, "start_frame": 0, "end_frame": n},
        "exact": {"x": exact[0] as i64 + ex.0, "y": exact[1] as i64 + ex.1, "start_frame": 0, "end_frame": n},
    });
    let project = json!({
        "output": "test.lvd",
        "canvas": {"width": w, "height": h, "background": "#000000"},
        "fps": format!("{}/{}", fps.num, fps.den),
        "duration": n as f64 / fps_int as f64,
        "gop": o.gop,
        "quality": {"crf": o.crf},
        "layers": [
            {"id": "bg", "name": "background (testsrc2)", "kind": "video", "src": "bg.mkv", "z": 0,
             "rect": [0, 0, w, h], "alpha": false},
            {"id": "sync_a", "name": "sync A (white line)", "kind": "video", "src": "sync_a.mkv", "z": 1,
             "rect": band, "alpha": true},
            {"id": "sync_b", "name": "sync B (red line)", "kind": "video", "src": "sync_b.mkv", "z": 2,
             "rect": band, "alpha": true},
            {"id": "square", "name": "square (from frame 45)", "kind": "video", "src": "square.mkv", "z": 3,
             "rect": square, "start": secs(sq_start), "end": secs(sq_end), "alpha": true},
            {"id": "calib", "name": "alpha calibration", "kind": "video", "src": "calib.mkv", "z": 4,
             "rect": calib, "alpha": true},
            {"id": "logo", "name": "logo (still)", "kind": "still", "src": "logo.png", "z": 5,
             "rect": logo, "start": secs(logo_start), "end": secs(logo_end)},
            {"id": "exact", "name": "lossless pattern", "kind": "video", "src": "exact.mkv", "z": 6,
             "rect": exact, "alpha": true, "lossless": true},
        ],
        "audio": {"src": "beeps.wav"},
    });
    let project_path = p("test_project.json");
    std::fs::write(&project_path, serde_json::to_string_pretty(&project).unwrap() + "\n")?;
    let probes = json!({
        "canvas": [w, h], "fps": fps_int, "frame_count": n, "gop": o.gop,
        "barcode": {"cell": BAR_CELL, "bits": BAR_BITS, "cells": BAR_CELLS, "layers": bars},
        "sync_band": {"rect": band, "line_rows": [0, band[3] - 60]},
        "calib": {"rect": calib, "ramp_rows": [0, 40]},
        "logo": {"rect": logo, "start_frame": logo_start, "end_frame": logo_end},
        "exact": {"rect": exact, "pattern_rows": [0, 64], "ramp_rows": [64, 80]},
        "flash": {"frames_per_second": flash, "border": 14},
        "beep": {"hz": 1000, "seconds": beep_s},
    });
    std::fs::write(p("barcodes.json"), serde_json::to_string_pretty(&probes).unwrap() + "\n")?;
    println!("wrote {} and {}", project_path.display(), p("barcodes.json").display());
    Ok(project_path)
}
