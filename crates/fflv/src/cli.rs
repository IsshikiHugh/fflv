//! The `fflv` command line.
//!
//!   fflv info    FILE [--frame N] [--json]         show header, layers, statistics; check invariants
//!   fflv check   FILE...                           validate (exit 1 if invalid)
//!   fflv pack    PROJECT.json [-o OUT]             build a file from a project (spec 8.1)
//!   fflv add     FILE --src MEDIA | --still IMG | --audio MEDIA  [layer options] [-o OUT]
//!   fflv rm      FILE LAYER... | --audio [-o OUT]  remove layers / the audio track
//!   fflv set     FILE LAYER key=value... [-o OUT]  id, name, z, rect, blend, opacity, visible (in place)
//!   fflv render  FILE -o OUT [-l LAYERS] [--hide LAYERS] [-f RANGE]   composite to images / video / .npy
//!   fflv extract FILE LAYER -o OUT [-f RANGE]      one layer's own pixels (RGBA)
//!   fflv view    FILE [--open]                     serve the interactive player (Chrome / Edge); print its URL
//!   fflv testsrc / fflv corrupt                    generate test material / broken files (development)
//!
//! LAYER is a layer id or index. RANGE is `N`, `A:B` (B excluded), `A:` or `:B`, in frames or,
//! with an `s` suffix, in seconds (`1.5s:3s`). Edits rewrite the file in place (atomically) unless
//! -o is given; existing layers are never re-encoded.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{ArgAction, Parser, Subcommand};
use lvf::meta::{self, EDITABLE_FIELDS};
use lvf::timing::seconds_to_frame;
use lvf::{validate, Fps, LvfReader};
use serde_json::{json, Value};

use crate::codec::Speed;
use crate::edit::{self, AddOptions, Source};
use crate::error::{Error, Result};
use crate::inspect;
use crate::media::still_png_from_file;
use crate::project::{pack, PackOptions};
use crate::render::{extract, render, RenderOptions};

#[derive(Parser)]
#[command(name = "fflv", version, about = "Pack, edit, decode, render and view LVF layered video files (.lvd)")]
#[command(after_help = "LAYER is a layer id or index. RANGE is N, A:B (B excluded), A: or :B, in frames or, with an \
    s suffix, in seconds (1.5s:3s). Edits rewrite the file in place (atomically) unless -o is given; existing layers \
    are never re-encoded.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show a file and check its invariants
    Info {
        file: PathBuf,
        /// Dump composite frame N (repeatable)
        #[arg(long = "frame")]
        frames: Vec<u32>,
        /// Do not print the metadata JSON
        #[arg(long = "no-meta")]
        no_meta: bool,
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },
    /// Validate files (exit 1 if any is invalid)
    Check {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// One line per file
        #[arg(short, long)]
        quiet: bool,
    },
    /// Build a file from a project JSON
    Pack {
        project: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Encoding threads (default: all cores)
        #[arg(short = 'j', long = "jobs")]
        jobs: Option<usize>,
    },
    /// Add a video / still layer or the audio track
    Add(AddArgs),
    /// Remove layers and/or the audio track
    Rm {
        file: PathBuf,
        /// Layer ids or indices
        layers: Vec<String>,
        /// Remove the audio track
        #[arg(long)]
        audio: bool,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Change layer properties (in place, instant)
    Set {
        file: PathBuf,
        layer: String,
        /// One or more of: id, name, z, rect, blend, opacity, visible (rect=x,y,w,h, visible=true/false)
        #[arg(value_name = "key=value")]
        fields: Vec<String>,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Composite layers to images, a video or .npy
    Render {
        file: PathBuf,
        /// out.png, frames/%05d.png, dir/, out.mp4/.mov/.webm/.mkv, out.npy
        #[arg(short, long)]
        output: String,
        /// Comma-separated layers to show (default: the file's visible layers)
        #[arg(short, long)]
        layers: Option<String>,
        /// Comma-separated layers to hide
        #[arg(long)]
        hide: Option<String>,
        /// Frame or range, e.g. 120, 100:200, 2s:5s (default: all)
        #[arg(short, long)]
        frames: Option<String>,
        /// No background: RGBA output
        #[arg(long)]
        transparent: bool,
        /// Quality of lossy video outputs
        #[arg(long, default_value_t = 18)]
        crf: u32,
    },
    /// One layer's own pixels (RGBA)
    Extract {
        file: PathBuf,
        layer: String,
        #[arg(short, long)]
        output: String,
        /// Frame or range (default: where the layer is active)
        #[arg(short, long)]
        frames: Option<String>,
        #[arg(long, default_value_t = 18)]
        crf: u32,
    },
    /// Serve the interactive player (Chrome / Edge) and print its URL
    View {
        file: PathBuf,
        /// Default: the first free port from 8765
        #[arg(long, default_value_t = 0)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Also open the page in a browser (Chrome, else Edge)
        #[arg(long)]
        open: bool,
        /// Open the page in this browser (implies --open)
        #[arg(long, value_parser = ["chrome", "edge", "chromium", "default"])]
        browser: Option<String>,
        /// Only print the URL (the default; kept for older scripts)
        #[arg(long = "no-open", hide = true, conflicts_with_all = ["open", "browser"])]
        no_open: bool,
        /// Log requests
        #[arg(short, long)]
        verbose: bool,
    },
    /// (dev) Generate the test material and pack test.lvd
    Testsrc {
        /// Output directory
        #[arg(long, default_value = "test_assets")]
        out: PathBuf,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        #[arg(long, default_value = "30/1")]
        fps: String,
        /// Seconds
        #[arg(long, default_value = "20")]
        duration: String,
        #[arg(long, default_value_t = 60)]
        gop: u32,
        #[arg(long, default_value_t = 32)]
        crf: u32,
        /// Only render the sources and the project file
        #[arg(long = "no-pack")]
        no_pack: bool,
    },
    /// (dev) Derive broken files from a valid one
    Corrupt {
        /// A valid .lvd (e.g. test_assets/test.lvd)
        source: PathBuf,
        /// Output directory (default: <source dir>/bad)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Validate each variant against its expectation
        #[arg(long)]
        check: bool,
    },
}

#[derive(clap::Args)]
struct AddArgs {
    file: PathBuf,
    /// Video layer from any file FFmpeg reads
    #[arg(long, value_name = "MEDIA", help_heading = "What to add (one of)")]
    src: Option<PathBuf>,
    /// Still layer from an image
    #[arg(long, value_name = "IMAGE", help_heading = "What to add (one of)")]
    still: Option<PathBuf>,
    /// Replace the audio track
    #[arg(long, value_name = "MEDIA", help_heading = "What to add (one of)")]
    audio: Option<PathBuf>,
    /// New layer id
    #[arg(long, help_heading = "Layer")]
    id: Option<String>,
    #[arg(long, help_heading = "Layer")]
    name: Option<String>,
    /// x,y,w,h on the canvas (default: whole canvas / image size)
    #[arg(long, help_heading = "Layer")]
    rect: Option<String>,
    /// First frame (or seconds with s suffix)
    #[arg(long, help_heading = "Layer")]
    start: Option<String>,
    /// End frame, excluded (default: end of file)
    #[arg(long, help_heading = "Layer")]
    end: Option<String>,
    /// Draw order (default: on top)
    #[arg(long, allow_negative_numbers = true, help_heading = "Layer")]
    z: Option<f64>,
    #[arg(long, default_value = "normal", value_parser = ["normal", "add", "multiply", "screen"], help_heading = "Layer")]
    blend: String,
    #[arg(long, default_value_t = 1.0, help_heading = "Layer")]
    opacity: f64,
    /// Hidden by default in the player
    #[arg(long, help_heading = "Layer")]
    hidden: bool,
    /// Keep the alpha channel
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_alpha", help_heading = "Encoding")]
    alpha: bool,
    /// Drop it (default: auto)
    #[arg(long = "no-alpha", action = ArgAction::SetTrue, help_heading = "Encoding")]
    no_alpha: bool,
    /// Bit-exact RGB and alpha (bigger)
    #[arg(long, help_heading = "Encoding")]
    lossless: bool,
    /// VP9 quality, lower is better
    #[arg(long, default_value_t = 32, help_heading = "Encoding")]
    crf: u32,
    #[arg(long, default_value = "balanced", value_parser = ["fast", "balanced", "best"], help_heading = "Encoding")]
    speed: String,
    /// Opus bitrate for --audio
    #[arg(long, default_value = "128k", help_heading = "Encoding")]
    bitrate: String,
    /// Write here instead of editing FILE in place
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn cli_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Meta(msg.into()))
}

fn point(v: &str, fps: Fps) -> Result<u32> {
    let v = v.trim();
    let n = match v.strip_suffix('s') {
        Some(secs) => seconds_to_frame(secs, fps).ok(),
        None => v.parse::<i64>().ok(),
    };
    match n.and_then(|n| u32::try_from(n).ok()) {
        Some(n) => Ok(n),
        None => cli_err(format!("bad frame/time {v:?} (use a frame number or seconds like 1.5s)")),
    }
}

/// `N`, `A:B`, `A:`, `:B` (frames, or seconds with an `s` suffix) → [start, end).
pub fn parse_range(spec: Option<&str>, fps: Fps, frame_count: u32) -> Result<(u32, u32)> {
    let Some(spec) = spec.filter(|s| !s.is_empty()) else { return Ok((0, frame_count)) };
    let (start, end) = match spec.split_once(':') {
        None => {
            let n = point(spec, fps)?;
            (n, n.saturating_add(1))
        }
        Some((a, b)) => {
            let start = if a.trim().is_empty() { 0 } else { point(a, fps)? };
            let end = if b.trim().is_empty() { frame_count } else { point(b, fps)? };
            (start, end)
        }
    };
    if start >= end || end > frame_count {
        return cli_err(format!("range {spec:?} = frames [{start}, {end}) is outside [0, {frame_count})"));
    }
    Ok((start, end))
}

fn file_timing(path: &Path) -> Result<(Fps, u32)> {
    let m = LvfReader::open(path)?.meta()?;
    Ok((m.fps(), m.frame_count))
}

fn layers_arg(v: &Option<String>) -> Option<Vec<String>> {
    v.as_ref().map(|s| s.split(',').filter(|p| !p.is_empty()).map(String::from).collect())
}

struct Progress {
    what: &'static str,
    t0: Instant,
    last: f64,
    tty: bool,
}

impl Progress {
    fn new(what: &'static str) -> Progress {
        Progress { what, t0: Instant::now(), last: -1.0, tty: std::io::stderr().is_terminal() }
    }

    fn update(&mut self, done: u32, total: u32) -> Result<()> {
        let now = self.t0.elapsed().as_secs_f64();
        if self.tty && (now - self.last > 0.2 || done == total) {
            self.last = now;
            let fps = done as f64 / now.max(1e-6);
            eprint!("\r{}: {done}/{total} frames ({fps:.0} fps)  ", self.what);
            if done == total {
                eprintln!();
            }
        }
        Ok(())
    }
}

fn report_edit(path: &Path, rep: Option<&lvf::Report>) -> Result<()> {
    let m = LvfReader::open(path)?.meta()?;
    let ids: Vec<String> =
        m.layers.iter().map(|l| format!("{}({})", l.id, if l.is_video() { "v" } else { "s" })).collect();
    let audio = if m.audio.is_some() { "audio" } else { "no audio" };
    let invalid = if rep.is_some_and(|r| !r.ok()) { " — INVALID" } else { "" };
    println!("{}: {} frames, layers {}; {audio}{invalid}", path.display(), m.frame_count, ids.join(", "));
    Ok(())
}

fn cmd_info(file: &Path, frames: &[u32], no_meta: bool, as_json: bool) -> Result<i32> {
    if as_json {
        let rep = validate(file);
        let stats: serde_json::Map<String, Value> =
            rep.layer_stats.iter().map(|(k, v)| (k.to_string(), serde_json::to_value(v).unwrap())).collect();
        let issues: Vec<Value> = rep
            .issues
            .iter()
            .map(|i| json!({"code": i.code, "frame": i.frame, "severity": i.severity, "message": i.message}))
            .collect();
        let out = json!({"file": file, "valid": rep.ok(), "meta": rep.meta, "rap_frames": rep.rap_frames,
                         "layer_stats": stats, "issues": issues});
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(if rep.ok() { 0 } else { 1 });
    }
    let r = LvfReader::open(file)?;
    println!("== {}", file.display());
    print!("{}", inspect::header(&r));
    if !no_meta {
        println!("\n== metadata");
        println!("{}", inspect::meta_dump(&r.meta_json()?));
    }
    for &n in frames {
        println!();
        print!("{}", inspect::frame_dump(&r, n)?);
    }
    println!("\n== statistics & invariants");
    let rep = validate(file);
    print!("{}", inspect::report(&rep, true));
    Ok(if rep.ok() { 0 } else { 1 })
}

fn cmd_add(a: &AddArgs) -> Result<i32> {
    let given = [a.src.is_some(), a.still.is_some(), a.audio.is_some()].iter().filter(|&&b| b).count();
    if given != 1 {
        return cli_err("give exactly one of --src (video layer), --still (image layer) or --audio");
    }
    let out = a.output.as_deref().unwrap_or(&a.file);
    if let Some(audio) = &a.audio {
        let rep = edit::set_audio(&a.file, Some(audio), a.output.as_deref(), &a.bitrate, 2, true)?;
        report_edit(out, rep.as_ref())?;
        return Ok(0);
    }
    let Some(id) = &a.id else { return cli_err("--id is required for a new layer") };
    let (fps, _) = file_timing(&a.file)?;
    let opts = AddOptions {
        output: a.output.clone(),
        start: a.start.as_deref().map(|s| point(s, fps)).transpose()?.unwrap_or(0),
        end: a.end.as_deref().map(|s| point(s, fps)).transpose()?,
        alpha: if a.no_alpha {
            Some(false)
        } else if a.alpha {
            Some(true)
        } else {
            None
        },
        lossless: a.lossless,
        rect: a.rect.as_deref().map(meta::parse_rect).transpose()?,
        z: a.z,
        name: a.name.clone(),
        blend: a.blend.clone(),
        opacity: a.opacity,
        visible: !a.hidden,
        crf: a.crf,
        speed: Speed::parse(&a.speed)?,
        check: true,
    };
    let t0 = Instant::now();
    let rep = if let Some(src) = &a.src {
        edit::add_layer(&a.file, id, Source::Media(src), &opts)?
    } else {
        let png = still_png_from_file(a.still.as_ref().unwrap())?;
        edit::add_still(&a.file, id, png, &opts)?
    };
    println!("added {id:?} in {:.1} s", t0.elapsed().as_secs_f64());
    report_edit(out, rep.as_ref())?;
    Ok(0)
}

fn run(cli: Cli) -> Result<i32> {
    match cli.cmd {
        Cmd::Info { file, frames, no_meta, json } => cmd_info(&file, &frames, no_meta, json),
        Cmd::Check { files, quiet } => {
            let mut bad = 0;
            for path in &files {
                let rep = validate(path);
                if quiet {
                    println!("{} {}", if rep.ok() { "ok     " } else { "INVALID" }, path.display());
                } else {
                    println!("== {}", path.display());
                    print!("{}", inspect::report(&rep, false));
                }
                bad += usize::from(!rep.ok());
            }
            Ok(if bad > 0 { 1 } else { 0 })
        }
        Cmd::Pack { project, output, jobs } => {
            let rep = pack(&project, &PackOptions { output, threads: jobs }, &mut |line| println!("{line}"))?;
            print!("{}", inspect::report(&rep, false));
            Ok(if rep.ok() { 0 } else { 1 })
        }
        Cmd::Add(a) => cmd_add(&a),
        Cmd::Rm { file, layers, audio, output } => {
            if layers.is_empty() && !audio {
                return cli_err("nothing to remove: give layer ids/indices and/or --audio");
            }
            let mut rep = None;
            if !layers.is_empty() {
                rep = edit::remove_layers(&file, &layers, output.as_deref(), true)?;
            }
            let target = output.as_deref().unwrap_or(&file);
            if audio {
                rep = edit::set_audio(target, None, None, "128k", 2, true)?;
            }
            report_edit(target, rep.as_ref())?;
            Ok(0)
        }
        Cmd::Set { file, layer, fields, output } => {
            let mut kv = Vec::new();
            for f in &fields {
                let Some((k, v)) = f.split_once('=') else { return cli_err(format!("expected key=value, got {f:?}")) };
                kv.push((k.trim().to_string(), Value::String(v.trim().to_string())));
            }
            if kv.is_empty() {
                return cli_err(format!("nothing to set; editable fields: {}", EDITABLE_FIELDS.join(", ")));
            }
            let t0 = Instant::now();
            let in_place = edit::set_layer(&file, &layer, &kv, output.as_deref())?;
            let how = if in_place {
                "metadata rewritten in place"
            } else if output.is_some() {
                "written"
            } else {
                "file rewritten (metadata outgrew its reserved space)"
            };
            let target = output.as_deref().unwrap_or(&file);
            println!("{}: {how} in {:.0} ms", target.display(), t0.elapsed().as_secs_f64() * 1000.0);
            Ok(0)
        }
        Cmd::Render { file, output, layers, hide, frames, transparent, crf } => {
            let (fps, n) = file_timing(&file)?;
            let (start, end) = parse_range(frames.as_deref(), fps, n)?;
            let t0 = Instant::now();
            let o = RenderOptions {
                layers: layers_arg(&layers),
                hide: layers_arg(&hide).unwrap_or_default(),
                start,
                end: Some(end),
                transparent,
                crf,
                ..Default::default()
            };
            let mut p = Progress::new("render");
            let count = render(&file, &output, &o, Some(&mut |d, t| p.update(d, t)))?;
            println!("wrote {count} frame(s) to {output} in {:.1} s", t0.elapsed().as_secs_f64());
            Ok(0)
        }
        Cmd::Extract { file, layer, output, frames, crf } => {
            let (fps, n) = file_timing(&file)?;
            let (start, end) = match frames {
                Some(f) => {
                    let (a, b) = parse_range(Some(&f), fps, n)?;
                    (Some(a), Some(b))
                }
                None => (None, None),
            };
            let t0 = Instant::now();
            let mut p = Progress::new("extract");
            let count = extract(&file, &layer, &output, start, end, crf, Some(&mut |d, t| p.update(d, t)))?;
            println!("wrote {count} frame(s) of {layer:?} to {output} in {:.1} s", t0.elapsed().as_secs_f64());
            Ok(0)
        }
        Cmd::View { file, port, host, open, browser, no_open: _, verbose } => {
            LvfReader::open(&file)?; // fail early on something that is not an .lvd
            crate::view::serve(&file, &host, port, browser.as_deref(), open || browser.is_some(), !verbose)?;
            Ok(0)
        }
        Cmd::Testsrc { out, width, height, fps, duration, gop, crf, no_pack } => {
            let o = crate::devtools::testsrc::TestsrcOptions { out, width, height, fps, duration, gop, crf };
            let project = crate::devtools::testsrc::generate(&o)?;
            if no_pack {
                return Ok(0);
            }
            let rep = pack(&project, &PackOptions { output: None, threads: None }, &mut |line| println!("{line}"))?;
            print!("{}", inspect::report(&rep, false));
            Ok(if rep.ok() { 0 } else { 1 })
        }
        Cmd::Corrupt { source, out, check } => {
            let failures = crate::devtools::corrupt::run(&source, out.as_deref(), check)?;
            Ok(if failures > 0 { 1 } else { 0 })
        }
    }
}

fn command_name(cmd: &Cmd) -> &'static str {
    match cmd {
        Cmd::Info { .. } => "info",
        Cmd::Check { .. } => "check",
        Cmd::Pack { .. } => "pack",
        Cmd::Add(_) => "add",
        Cmd::Rm { .. } => "rm",
        Cmd::Set { .. } => "set",
        Cmd::Render { .. } => "render",
        Cmd::Extract { .. } => "extract",
        Cmd::View { .. } => "view",
        Cmd::Testsrc { .. } => "testsrc",
        Cmd::Corrupt { .. } => "corrupt",
    }
}

/// Run the command line (`args[0]` is the program name); returns the exit code: 0 ok, 1 invalid
/// file(s), 2 error.
pub fn main_with_args(args: Vec<String>) -> i32 {
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { 2 } else { 0 };
        }
    };
    let name = command_name(&cli.cmd);
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("fflv {name}: error: {e}");
            2
        }
    }
}
