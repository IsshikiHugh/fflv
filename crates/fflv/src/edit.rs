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

use std::path::Path;

use lvf::meta::{self, Layer, Rect, VideoLayerSpec, Z};
use lvf::{
    encode_meta, pts_us, publish, rewrite_meta_in_place, temp_path_for, Cau, LvfReader, Meta, Report, VideoEntry,
};
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

/// Rewrite `src` into `dst` (default: in place): drop layers, append layers, replace audio,
/// patch the metadata. Packets of kept layers are copied as they are.
pub fn remux(
    src: &Path,
    dst: Option<&Path>,
    remove: &[usize],
    add_video: Vec<NewVideo>,
    add_still: Vec<NewStill>,
    audio: AudioEdit,
    meta_patch: Option<&dyn Fn(&mut Meta) -> Result<()>>,
    check: bool,
) -> Result<Option<Report>> {
    let dst = dst.unwrap_or(src);
    let tmp = temp_path_for(dst);
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
        let data = match png {
            Some(p) => p.clone(),
            None => {
                let res =
                    l.resource.as_ref().ok_or_else(|| Error::Format(format!("still {:?} has no resource", l.id)))?;
                r.resource(res.offset, res.length)?
            }
        };
        let res =
            l.resource.get_or_insert_with(|| lvf::meta::Resource { offset: 0, length: 0, mime: "image/png".into() });
        res.offset = resources.len() as u64;
        res.length = data.len() as u64;
        res.mime = "image/png".into();
        resources.extend_from_slice(&data);
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
        for item in r.caus(None, None) {
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
    publish(&tmp, dst, check).map_err(|e| match Error::from(e) {
        e @ Error::Invalid { .. } => Error::Edit(e.to_string()),
        e => e,
    })
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
    remux(path, o.output.as_deref(), &[], vec![], vec![NewStill { meta: layer, png }], AudioEdit::Keep, None, o.check)
}

pub fn remove_layers(path: &Path, keys: &[String], output: Option<&Path>, check: bool) -> Result<Option<Report>> {
    let (meta, _) = file_info(path)?;
    let remove: Vec<usize> = keys.iter().map(|k| meta.resolve_layer(k)).collect::<std::result::Result<_, _>>()?;
    if remove.is_empty() {
        return err("no layers given");
    }
    remux(path, output, &remove, vec![], vec![], AudioEdit::Keep, None, check)
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
    let Some(src) = source else {
        return remux(path, output, &[], vec![], vec![], AudioEdit::Remove, None, check);
    };
    let (meta, _) = file_info(path)?;
    let seconds = meta.frame_count as f64 * meta.fps.den as f64 / meta.fps.num as f64;
    let track = encode_audio(src, Some(seconds), bitrate, channels)?;
    remux(path, output, &[], vec![], vec![], AudioEdit::Replace(track), None, check)
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
    let path = match output {
        Some(out) if out != path => {
            std::fs::copy(path, out)?;
            out
        }
        _ => path,
    };
    let (mut meta, _) = file_info(path)?;
    apply_edits(&mut meta, key, fields)?;
    meta.generator = Some(meta::GENERATOR.into());
    if rewrite_meta_in_place(path, &encode_meta(&meta)?)? {
        return Ok(true);
    }
    // No room beside the current metadata: rewrite the file, applying the edits to the metadata
    // remux() builds (its still-resource offsets are recomputed for the rewritten resources).
    let patch = |m: &mut Meta| apply_edits(m, key, fields);
    remux(path, None, &[], vec![], vec![], AudioEdit::Keep, Some(&patch), true)?;
    Ok(false)
}
