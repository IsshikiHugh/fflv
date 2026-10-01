//! Editing .lvd files without re-encoding what is already there.
//!
//!   add_layer      new video layer from a media file or a stream of images
//!   add_still      new still layer (PNG)
//!   remove_layers, set_audio (replace or remove), set_layer (id, name, z, rect, blend, opacity,
//!   visible)
//!
//! Existing layers are never re-encoded: their packets are copied bit for bit into a rewritten
//! file (written beside the output, validated, then atomically renamed over it). Only a newly
//! added video layer is encoded, with key frames exactly on the file's existing random-access
//! points, so random access stays the same. `set_layer` rewrites the metadata in place — instant,
//! whatever the file size — when it fits the space reserved after the metadata (it always does
//! for ordinary edits).
//!
//! Edits of one file are serialized: a rewrite holds an exclusive lock on the source file from
//! the moment it opens it until the result is published, and lvf's in-place metadata rewrite
//! takes the same lock. A rewritten file keeps the source's permissions; editing in place through
//! a symbolic link rewrites the file it points to and keeps the link.

use std::cell::RefCell;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use lvf::meta::{self, Layer, Rect, VideoLayerSpec, Z};
use lvf::{encode_meta, pts_us, publish, rewrite_meta_with, temp_path_for, Cau, LvfReader, Meta, Report, VideoEntry};
use serde_json::Value;

use crate::audio::{encode_audio, AudioTrack};
use crate::codec::{EncodeOptions, Speed};
use crate::encode::LayerEncoder;
use crate::error::{Error, Result};
use crate::image::{png_size, Image};
use crate::media::{open_video_source, raw_frames};

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Edit(msg.into()))
}

/// Encoded frames of a new video layer, one call per active frame, in order:
/// (color packet, alpha packet, key).
pub type Stream<'a> = Box<dyn FnMut(u32) -> Result<(Vec<u8>, Vec<u8>, bool)> + 'a>;

pub struct NewVideo<'a> {
    pub meta: Layer,
    pub stream: Stream<'a>,
}

pub struct NewStill {
    pub meta: Layer,
    pub png: Vec<u8>,
}

pub enum AudioEdit {
    Keep,
    Remove,
    Replace(AudioTrack),
}

thread_local! {
    /// See [`with_interrupt_check`].
    static INTERRUPT_CHECK: RefCell<Option<Box<dyn FnMut() -> Result<()>>>> = const { RefCell::new(None) };
}

/// Run `f` with `check` called every ~100 ms by the rewrites it makes on this thread (between
/// frames); an error from `check` stops the rewrite (nothing is published) and is returned. For
/// callers that must stay interruptible, like the Python bindings (Ctrl+C).
pub fn with_interrupt_check<R>(check: impl FnMut() -> Result<()> + 'static, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Box<dyn FnMut() -> Result<()>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            INTERRUPT_CHECK.with(|c| *c.borrow_mut() = previous);
        }
    }
    let previous = INTERRUPT_CHECK.with(|c| c.borrow_mut().replace(Box::new(check)));
    let _restore = Restore(previous);
    f()
}

fn interrupt_check() -> Result<()> {
    INTERRUPT_CHECK.with(|c| match c.borrow_mut().as_mut() {
        Some(check) => check(),
        None => Ok(()),
    })
}

/// Open `path` and take an exclusive lock on it (waiting for other edits of it to finish). When
/// the file was replaced (renamed over) while waiting, the new one is locked instead. Unix only:
/// Windows locks are mandatory, so there the lock would also stop the edit's own reader (and the
/// viewer) from reading the file; elsewhere edits run unlocked.
fn lock_file(path: &Path) -> Result<File> {
    // (read-only, unlike lvf::container::open_locked: the source of an edit written elsewhere
    // need not be writable)
    for _ in 0..100 {
        let f = File::open(path)?;
        if cfg!(not(unix)) {
            return Ok(f);
        }
        match f.lock() {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Unsupported => return Ok(f),
            Err(e) => return Err(e.into()),
        }
        if is_file_at(&f, path) {
            return Ok(f);
        }
    }
    err(format!("{} keeps being replaced", path.display()))
}

/// Whether the open file `f` is still the one `path` names.
#[cfg(unix)]
fn is_file_at(f: &File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (f.metadata(), fs::metadata(path)) {
        (Ok(a), Ok(b)) => (a.dev(), a.ino()) == (b.dev(), b.ino()),
        // gone: the next open reports it
        _ => false,
    }
}

#[cfg(not(unix))]
fn is_file_at(_f: &File, _path: &Path) -> bool {
    true
}

/// Where a rewrite of `src` into `dst` is published: `dst`, or, when `dst` is a symbolic link to
/// `src` itself (an in-place edit), the file it points to — the link stays a link.
fn publish_target(src: &Path, dst: &Path) -> PathBuf {
    let is_link = fs::symlink_metadata(dst).is_ok_and(|m| m.file_type().is_symlink());
    if is_link && same_file(src, dst) {
        if let Ok(real) = fs::canonicalize(dst) {
            return real;
        }
    }
    dst.to_path_buf()
}

/// Rewrite `src` into `dst` (default: in place): drop layers, append layers, replace audio,
/// patch the metadata. Packets of kept layers are copied as they are.
///
/// `lock` is `src` locked by [`lock_file`], taken by the caller before it read anything it plans
/// the edit from (layer indices, ids, random-access points), so concurrent edits never plan
/// against a file that is replaced underneath them. It is held until the result is published.
/// When the result replaces `src`, the lock is released on the replaced file: an edit that was
/// waiting for it notices the replacement and locks the new file.
pub fn remux(
    lock: File,
    src: &Path,
    dst: Option<&Path>,
    remove: &[usize],
    add_video: Vec<NewVideo>,
    add_still: Vec<NewStill>,
    audio: AudioEdit,
    meta_patch: Option<&dyn Fn(&mut Meta) -> Result<()>>,
    check: bool,
) -> Result<Option<Report>> {
    let dst = publish_target(src, dst.unwrap_or(src));
    let tmp = temp_path_for(&dst);
    let r = LvfReader::open(src)?;
    let mut meta = r.meta()?;
    let fps = meta.fps();
    let old = std::mem::take(&mut meta.layers);
    let keep: Vec<usize> = (0..old.len()).filter(|i| !remove.contains(i)).collect();
    let remap: Vec<Option<u16>> = (0..old.len()).map(|i| keep.iter().position(|&k| k == i).map(|n| n as u16)).collect();
    let n_kept = keep.len();
    let mut layers: Vec<Layer> = keep.iter().map(|&i| old[i].clone()).collect();
    let mut streams = Vec::new();
    for v in add_video {
        layers.push(v.meta);
        streams.push(v.stream);
    }
    let n_new_video = streams.len();
    let mut new_pngs: Vec<Option<Vec<u8>>> = vec![None; layers.len()];
    for s in add_still {
        layers.push(s.meta);
        new_pngs.push(Some(s.png));
    }
    let ids: Vec<&str> = layers.iter().map(|l| l.id.as_str()).collect();
    if (1..ids.len()).any(|i| ids[..i].contains(&ids[i])) {
        return err(format!("duplicate layer ids after the edit: {ids:?}"));
    }

    // resources: stills in their new order
    let mut resources = Vec::new();
    for (l, png) in layers.iter_mut().zip(&new_pngs) {
        if l.kind != lvf::Kind::Still {
            continue;
        }
        let offset = resources.len() as u64;
        match png {
            Some(p) => resources.extend_from_slice(p),
            None => {
                let res =
                    l.resource.as_ref().ok_or_else(|| Error::Format(format!("still {:?} has no resource", l.id)))?;
                resources.extend_from_slice(&r.resource(res.offset, res.length)?);
            }
        }
        let res =
            l.resource.get_or_insert_with(|| lvf::meta::Resource { offset: 0, length: 0, mime: "image/png".into() });
        res.offset = offset;
        res.length = resources.len() as u64 - offset;
        res.mime = "image/png".into();
    }

    let keep_audio = matches!(audio, AudioEdit::Keep);
    let mut new_track = None;
    match audio {
        AudioEdit::Keep => {}
        AudioEdit::Remove => meta.audio = None,
        AudioEdit::Replace(mut t) => {
            meta.audio = Some(t.meta());
            t.truncate_to(pts_us(meta.frame_count as u64, fps));
            new_track = Some(t);
        }
    }
    meta.layers = layers;
    meta.generator = Some(meta::GENERATOR.into());
    if let Some(patch) = meta_patch {
        patch(&mut meta)?;
    }

    let new_ranges: Vec<(u16, u32, u32)> = (0..n_new_video)
        .map(|k| {
            let l = &meta.layers[n_kept + k];
            ((n_kept + k) as u16, l.start_frame, l.end_frame)
        })
        .collect();
    let mut w = lvf::LvfWriter::create(&tmp)?;
    let result = (|| -> Result<()> {
        w.begin(&encode_meta(&meta)?, &resources, None)?;
        let mut ai = 0;
        let mut last_check = Instant::now();
        for item in r.caus(None, None) {
            if last_check.elapsed() >= Duration::from_millis(100) {
                interrupt_check()?;
                last_check = Instant::now();
            }
            let (_, cau, _) = item?;
            let f = cau.frame_index;
            let mut entries: Vec<VideoEntry> = cau
                .entries
                .into_iter()
                .filter_map(|mut e| {
                    remap.get(e.layer_index as usize).copied().flatten().map(|n| {
                        e.layer_index = n;
                        e
                    })
                })
                .collect();
            for (k, &(idx, start, end)) in new_ranges.iter().enumerate() {
                if start <= f && f < end {
                    let (color, alpha, key) = (streams[k])(f)?;
                    entries.push(VideoEntry::frame(idx, key, color, alpha));
                } else {
                    entries.push(VideoEntry::empty(idx));
                }
            }
            let rap = entries.iter().filter(|e| e.is_frame()).all(|e| e.is_key());
            let packets = if keep_audio {
                cau.audio
            } else if let Some(t) = &new_track {
                let hi = pts_us(f as u64 + 1, fps);
                let mut pk = Vec::new();
                while ai < t.packets.len() && t.packets[ai].pts_us < hi {
                    pk.push(t.packets[ai].clone());
                    ai += 1;
                }
                pk
            } else {
                Vec::new()
            };
            w.write_cau(&Cau::new(f, rap, entries, packets))?;
        }
        Ok(())
    })();
    if let Err(e) = result {
        w.abort();
        return Err(e);
    }
    if let Err(e) = w.finish(None, None) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    drop(r);
    // like the file it replaces (best effort: not every file system has permissions)
    if let Ok(m) = fs::metadata(src) {
        let _ = fs::set_permissions(&tmp, m.permissions());
    }
    let published = publish(&tmp, &dst, check).map_err(|e| match Error::from(e) {
        e @ Error::Invalid { .. } => Error::Edit(e.to_string()),
        e => e,
    });
    drop(lock);
    published
}

// ------------------------------------------------------------------------------------------------
// Adding layers
// ------------------------------------------------------------------------------------------------
fn file_info(path: &Path) -> Result<(Meta, Vec<u32>)> {
    let r = LvfReader::open(path)?;
    let meta = r.meta()?;
    let (_, _, entries) = r.index()?;
    let raps = entries.iter().enumerate().filter(|(_, e)| e.is_rap()).map(|(i, _)| i as u32).collect();
    Ok((meta, raps))
}

fn frame_range(meta: &Meta, start: u32, end: Option<u32>, n_images: Option<u32>) -> Result<(u32, u32)> {
    let fc = meta.frame_count;
    let end = end.unwrap_or_else(|| n_images.map_or(fc, |n| start.saturating_add(n)));
    if start >= end || end > fc {
        return err(format!("frame range [{start}, {end}) is outside the file's [0, {fc})"));
    }
    Ok((start, end))
}

/// Where the images of a new video layer come from.
pub enum Source<'a> {
    /// Any file FFmpeg reads (scaled to the layer's rect).
    Media(&'a Path),
    /// Images for consecutive frames from `start`; `len` if known (sets the default end).
    Images { images: Box<dyn Iterator<Item = Result<Image>> + 'a>, len: Option<u32> },
}

#[derive(Clone, Debug)]
pub struct AddOptions {
    /// Write here instead of editing the file in place.
    pub output: Option<std::path::PathBuf>,
    pub start: u32,
    /// Default: the source length (images) or the end of the file.
    pub end: Option<u32>,
    /// None: alpha if the source has an alpha channel.
    pub alpha: Option<bool>,
    pub lossless: bool,
    /// Default: the whole canvas (media) or the first image's size at (0, 0).
    pub rect: Option<Rect>,
    /// Default: on top.
    pub z: Option<f64>,
    pub name: Option<String>,
    pub blend: String,
    pub opacity: f64,
    pub visible: bool,
    pub crf: u32,
    pub speed: Speed,
    pub check: bool,
}

impl Default for AddOptions {
    fn default() -> Self {
        AddOptions {
            output: None,
            start: 0,
            end: None,
            alpha: None,
            lossless: false,
            rect: None,
            z: None,
            name: None,
            blend: "normal".into(),
            opacity: 1.0,
            visible: true,
            crf: 32,
            speed: Speed::Balanced,
            check: true,
        }
    }
}

/// Append a video layer; key frames go exactly on the file's random-access points.
pub fn add_layer(path: &Path, id: &str, source: Source, o: &AddOptions) -> Result<Option<Report>> {
    let lock = lock_file(path)?;
    let (meta, raps) = file_info(path)?;
    let taken: Vec<&str> = meta.layers.iter().map(|l| l.id.as_str()).collect();
    let id = meta::check_id(id, &taken)?;
    let fps = meta.fps();
    let (cw, ch) = (meta.canvas.width, meta.canvas.height);
    let opts = EncodeOptions::new(o.crf, o.speed)?;
    let blend = meta::check_blend(&o.blend)?;
    let opacity = meta::check_opacity(o.opacity)?;
    let check_rect = |r: Rect| meta::make_rect(r.x, r.y, r.w as i64, r.h as i64);

    let (alpha, rect, start, end, mut images): (bool, Rect, u32, u32, Box<dyn Iterator<Item = Result<Image>>>) =
        match source {
            Source::Media(p) => {
                let src = open_video_source(p)?;
                let alpha = match o.alpha {
                    None => src.has_alpha,
                    Some(true) if !src.has_alpha => {
                        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        return err(format!("{name} ({}, {}) has no alpha channel", src.codec, src.pix_fmt));
                    }
                    Some(a) => a,
                };
                let rect = check_rect(o.rect.unwrap_or(Rect { x: 0, y: 0, w: cw, h: ch }))?;
                let (start, end) = frame_range(&meta, o.start, o.end, None)?;
                let frames = raw_frames(&src, fps, rect.w, rect.h, (end - start) as u64, alpha)?;
                (alpha, rect, start, end, Box::new(frames))
            }
            Source::Images { mut images, len } => {
                let first = images.next().ok_or_else(|| Error::Edit("the image source is empty".into()))??;
                let alpha = o.alpha.unwrap_or(first.channels == 4);
                let rect = check_rect(o.rect.unwrap_or(Rect { x: 0, y: 0, w: first.width, h: first.height }))?;
                let (start, end) = frame_range(&meta, o.start, o.end, len)?;
                (alpha, rect, start, end, Box::new(std::iter::once(Ok(first)).chain(images)))
            }
        };
    let mut enc = LayerEncoder::new(rect.w, rect.h, fps, alpha, o.lossless, &opts, &id)?;
    let z = match o.z {
        Some(z) => meta::check_z_f64(z)?,
        None => Z(meta.top_z() + 1.0),
    };
    let layer = meta::video_layer(VideoLayerSpec {
        id: &id,
        name: o.name.as_deref().unwrap_or(&id),
        z,
        rect,
        start,
        end,
        fps,
        alpha,
        lossless: o.lossless,
        blend: &blend,
        opacity,
        visible: o.visible,
    });
    let keys: Vec<u32> = raps.into_iter().filter(|&r| start < r && r < end).collect();
    let stream: Stream = Box::new(move |f: u32| {
        let img = match images.next() {
            Some(img) => img?,
            None => return err(format!("source ran out of images at frame {f} (needed [{start}, {end}))")),
        };
        let key = f == start || keys.binary_search(&f).is_ok();
        let (c, a) = enc.encode(img.view(), key)?;
        if f + 1 == end {
            enc.finish()?;
        }
        Ok((c, a, key))
    });
    remux(
        lock,
        path,
        o.output.as_deref(),
        &[],
        vec![NewVideo { meta: layer, stream }],
        vec![],
        AudioEdit::Keep,
        None,
        o.check,
    )
}

/// Append a still layer (PNG bytes) shown in frames [start, end).
pub fn add_still(path: &Path, id: &str, png: Vec<u8>, o: &AddOptions) -> Result<Option<Report>> {
    let lock = lock_file(path)?;
    let (meta, _) = file_info(path)?;
    let taken: Vec<&str> = meta.layers.iter().map(|l| l.id.as_str()).collect();
    let id = meta::check_id(id, &taken)?;
    let rect = match o.rect {
        Some(r) => meta::make_rect(r.x, r.y, r.w as i64, r.h as i64)?,
        None => {
            let (w, h) = png_size(&png)?;
            meta::make_rect(0, 0, w as i64, h as i64)?
        }
    };
    let (start, end) = frame_range(&meta, o.start, o.end, None)?;
    let z = match o.z {
        Some(z) => meta::check_z_f64(z)?,
        None => Z(meta.top_z() + 1.0),
    };
    let layer = meta::still_layer(
        &id,
        o.name.as_deref().unwrap_or(&id),
        z,
        rect,
        start,
        end,
        0,
        png.len() as u64,
        &meta::check_blend(&o.blend)?,
        meta::check_opacity(o.opacity)?,
        o.visible,
    );
    remux(
        lock,
        path,
        o.output.as_deref(),
        &[],
        vec![],
        vec![NewStill { meta: layer, png }],
        AudioEdit::Keep,
        None,
        o.check,
    )
}

pub fn remove_layers(path: &Path, keys: &[String], output: Option<&Path>, check: bool) -> Result<Option<Report>> {
    if keys.is_empty() {
        return err("no layers given");
    }
    remove(path, keys, false, output, check)
}

/// Remove layers and/or (`audio`) the audio track, in one rewrite.
pub fn remove(path: &Path, keys: &[String], audio: bool, output: Option<&Path>, check: bool) -> Result<Option<Report>> {
    if keys.is_empty() && !audio {
        return err("nothing to remove");
    }
    let lock = lock_file(path)?;
    let (meta, _) = file_info(path)?;
    let remove: Vec<usize> = keys.iter().map(|k| meta.resolve_layer(k)).collect::<std::result::Result<_, _>>()?;
    let audio = if audio { AudioEdit::Remove } else { AudioEdit::Keep };
    remux(lock, path, output, &remove, vec![], vec![], audio, None, check)
}

/// Replace the audio track with `source` (any file FFmpeg reads), or remove it (None).
pub fn set_audio(
    path: &Path,
    source: Option<&Path>,
    output: Option<&Path>,
    bitrate: &str,
    channels: u32,
    check: bool,
) -> Result<Option<Report>> {
    let lock = lock_file(path)?;
    let Some(src) = source else {
        return remux(lock, path, output, &[], vec![], vec![], AudioEdit::Remove, None, check);
    };
    let (meta, _) = file_info(path)?;
    let seconds = meta.frame_count as f64 * meta.fps.den as f64 / meta.fps.num as f64;
    let track = encode_audio(src, Some(seconds), bitrate, channels)?;
    remux(lock, path, output, &[], vec![], vec![], AudioEdit::Replace(track), None, check)
}

/// Validate and apply field changes (id, name, z, rect, blend, opacity, visible) to one layer.
pub fn apply_edits(meta: &mut Meta, key: &str, fields: &[(String, Value)]) -> Result<()> {
    let i = meta.resolve_layer(key)?.to_string();
    for (field, value) in fields {
        meta::apply_edit(meta, &i, field, value)?;
    }
    Ok(())
}

/// Change layer properties. Returns true when done in place (metadata only), false when the file
/// had to be rewritten.
pub fn set_layer(path: &Path, key: &str, fields: &[(String, Value)], output: Option<&Path>) -> Result<bool> {
    let patch = |m: &mut Meta| -> Result<()> {
        apply_edits(m, key, fields)?;
        m.generator = Some(meta::GENERATOR.into());
        Ok(())
    };
    if let Some(out) = output.filter(|out| !same_file(path, out)) {
        // A new file: written like every other edit (temporary file, validated, renamed).
        remux(lock_file(path)?, path, Some(out), &[], vec![], vec![], AudioEdit::Keep, Some(&patch), true)?;
        return Ok(false);
    }
    // In place: the metadata is read, edited and rewritten through one open file.
    let mut failure = None;
    let in_place = rewrite_meta_with(path, |current| {
        let edited = serde_json::from_slice::<Meta>(current)
            .map_err(|e| Error::Format(format!("metadata does not match the LVF schema: {e}")))
            .and_then(|mut m| patch(&mut m).map(|()| m))
            .and_then(|m| Ok(encode_meta(&m)?));
        edited.map_err(|e| {
            let msg = e.to_string();
            failure = Some(e);
            lvf::Error::Value(msg)
        })
    });
    match (in_place, failure) {
        (_, Some(e)) => return Err(e),
        (Err(e), None) => return Err(e.into()),
        (Ok(true), None) => return Ok(true),
        (Ok(false), None) => {}
    }
    // No room beside the current metadata: rewrite the file, applying the edits to the metadata
    // remux() builds (its still-resource offsets are recomputed for the rewritten resources).
    remux(lock_file(path)?, path, None, &[], vec![], vec![], AudioEdit::Keep, Some(&patch), true)?;
    Ok(false)
}

/// Whether two paths name the same existing file (however they are spelled).
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
