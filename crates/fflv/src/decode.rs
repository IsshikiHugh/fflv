//! Decoding .lvd files: selected layers only, composited like the player.
//!
//! Only the layers asked for are decoded — the packets of all other layers are skipped, so
//! switching between layer subsets costs nothing extra. Decoding starts at the nearest
//! random-access point at or before the first requested frame; the planes of a frame are decoded
//! and converted in parallel.
//!
//! A [`Reader`] is cheap to share (`&self` everywhere); every [`Reader::decode`] /
//! [`Reader::frames`] call is an independent session with its own decoders.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lvf::constants::ENTRY_FRAME;
use lvf::container::CauIter;
use lvf::{pts_us, Fps, Kind, Layer, LvfReader, Meta, Report};
use rayon::prelude::*;

use crate::codec::Decoder;
use crate::composite::{parse_color, Blend, Canvas, Pixels};
use crate::error::{Error, Result};
use crate::image::{decode_png, Image};
use crate::pixel::{frame_to_alpha, frame_to_rgb};

/// A video layer's own pixels in one frame (content size, before scaling to its rect).
#[derive(Clone, Debug)]
pub struct LayerFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub alpha: Option<Vec<u8>>,
}

impl LayerFrame {
    pub fn pixels(&self) -> Pixels<'_> {
        Pixels { width: self.width, height: self.height, color: &self.rgb, channels: 3, alpha: self.alpha.as_deref() }
    }

    /// RGBA (opaque when the layer has no alpha).
    pub fn to_rgba(&self) -> Image {
        let n = (self.width * self.height) as usize;
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            data.extend_from_slice(&self.rgb[i * 3..i * 3 + 3]);
            data.push(self.alpha.as_ref().map_or(255, |a| a[i]));
        }
        Image { width: self.width, height: self.height, channels: 4, data }
    }
}

pub struct Reader {
    path: PathBuf,
    file: Arc<LvfReader>,
    meta: Arc<Meta>,
    offsets: Arc<Vec<u64>>,
    raps: Vec<u32>,
    stills: Mutex<HashMap<usize, Arc<Image>>>,
}

fn range_error<T>(start: u32, end: u32, n: u32) -> Result<T> {
    Err(Error::Decode(format!("frame range [{start}, {end}) is outside [0, {n})")))
}

impl Reader {
    pub fn open(path: impl AsRef<Path>) -> Result<Reader> {
        let path = path.as_ref().to_path_buf();
        let file = LvfReader::open(&path)?;
        let meta = file.meta()?;
        let (_, _, entries) = file.index()?;
        if entries.len() != meta.frame_count as usize {
            return Err(Error::Decode("index does not match frame_count (run `fflv check`)".into()));
        }
        check_layers(&meta)?;
        let offsets = entries.iter().map(|e| e.cau_offset).collect();
        let raps = entries.iter().enumerate().filter(|(_, e)| e.is_rap()).map(|(i, _)| i as u32).collect();
        Ok(Reader {
            path,
            file: Arc::new(file),
            meta: Arc::new(meta),
            offsets: Arc::new(offsets),
            raps,
            stills: Mutex::new(HashMap::new()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    pub fn fps(&self) -> Fps {
        self.meta.fps()
    }

    pub fn frame_count(&self) -> u32 {
        self.meta.frame_count
    }

    pub fn size(&self) -> (u32, u32) {
        (self.meta.canvas.width, self.meta.canvas.height)
    }

    pub fn layers(&self) -> &[Layer] {
        &self.meta.layers
    }

    pub fn raps(&self) -> &[u32] {
        &self.raps
    }

    pub fn pts_us(&self, frame: u32) -> i64 {
        pts_us(frame as u64, self.fps())
    }

    pub fn check(&self) -> Report {
        lvf::validate(&self.path)
    }

    /// Layer index from an id or an index ("3").
    pub fn layer(&self, key: &str) -> Result<usize> {
        Ok(self.meta.resolve_layer(key)?)
    }

    /// Layers to show: `layers` (ids / indices) if given, else the file's visible layers; minus
    /// `hide`. In the order given (or file order).
    pub fn select(&self, layers: Option<&[String]>, hide: &[String]) -> Result<Vec<usize>> {
        let chosen: Vec<usize> = match layers {
            Some(keys) => keys.iter().map(|k| self.layer(k)).collect::<Result<_>>()?,
            None => (0..self.meta.layers.len()).filter(|&i| self.meta.layers[i].visible).collect(),
        };
        let hidden: Vec<usize> = hide.iter().map(|k| self.layer(k)).collect::<Result<_>>()?;
        Ok(chosen.into_iter().filter(|i| !hidden.contains(i)).collect())
    }

    pub fn rap_at_or_before(&self, frame: u32) -> u32 {
        match self.raps.binary_search(&frame) {
            Ok(i) => self.raps[i],
            Err(0) => self.raps.first().copied().unwrap_or(0),
            Err(i) => self.raps[i - 1],
        }
    }

    /// A still layer's image (RGBA), decoded once.
    pub fn still(&self, index: usize) -> Result<Arc<Image>> {
        if let Some(img) = self.stills.lock().unwrap().get(&index) {
            return Ok(img.clone());
        }
        let l = &self.meta.layers[index];
        let res = l.resource.as_ref().ok_or_else(|| Error::Decode(format!("still {:?} has no resource", l.id)))?;
        let img = Arc::new(decode_png(&self.file.resource(res.offset, res.length)?)?);
        self.stills.lock().unwrap().insert(index, img.clone());
        Ok(img)
    }

    fn check_range(&self, start: u32, end: Option<u32>) -> Result<(u32, u32)> {
        let n = self.frame_count();
        let end = end.unwrap_or(n);
        if start >= end || end > n {
            return range_error(start, end, n);
        }
        Ok((start, end))
    }

    /// The chosen video layers' own pixels, frame by frame, over [start, end). Layers that are
    /// not active in a frame are absent from its map. Other layers are never decoded.
    pub fn decode(&self, start: u32, end: Option<u32>, layers: &[usize]) -> Result<Decoding> {
        let (start, end) = self.check_range(start, end)?;
        let mut want: Vec<usize> = layers.iter().copied().filter(|&i| self.meta.layers[i].is_video()).collect();
        want.sort_unstable();
        want.dedup();
        let rap = self.rap_at_or_before(start);
        let from = self.offsets[rap as usize];
        let to = if end < self.frame_count() { self.offsets[end as usize] } else { self.file.header.index_offset };
        let planes = want
            .iter()
            .map(|&i| {
                let l = &self.meta.layers[i];
                let color = Decoder::new(1, &format!("layer {:?} color", l.id))?;
                let alpha =
                    if l.has_alpha() { Some(Decoder::new(1, &format!("layer {:?} alpha", l.id))?) } else { None };
                let (w, h) = l.content_size();
                Ok(LayerDecoder { index: i, width: w, height: h, full: l.alpha_full_range(), color, alpha })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Decoding { caus: CauIter::new(self.file.clone(), Some(from), Some(to)), start, end, planes, done: false })
    }

    /// Composite frames (RGB, or RGBA when `transparent`) of the chosen layers over [start, end).
    pub fn frames(
        &self,
        start: u32,
        end: Option<u32>,
        layers: Option<&[String]>,
        hide: &[String],
        transparent: bool,
    ) -> Result<Frames> {
        let mut shown = self.select(layers, hide)?;
        shown.sort_by(|&a, &b| {
            let (la, lb) = (&self.meta.layers[a], &self.meta.layers[b]);
            la.z.0.partial_cmp(&lb.z.0).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        let mut stills = HashMap::new();
        let mut draw = Vec::new();
        for &i in &shown {
            let l = &self.meta.layers[i];
            if l.kind == Kind::Still {
                stills.insert(i, self.still(i)?);
            }
            draw.push(Draw {
                index: i,
                video: l.is_video(),
                start: l.start_frame,
                end: l.end_frame,
                rect: l.rect,
                blend: Blend::parse(&l.blend)?,
                opacity: l.opacity as f32,
            });
        }
        let videos: Vec<usize> = shown.iter().copied().filter(|&i| self.meta.layers[i].is_video()).collect();
        let background = if transparent { None } else { Some(parse_color(&self.meta.canvas.background)?) };
        let (w, h) = self.size();
        Ok(Frames {
            decoding: self.decode(start, end, &videos)?,
            draw,
            stills,
            width: w,
            height: h,
            background,
            transparent,
        })
    }

    pub fn frame(&self, index: u32, layers: Option<&[String]>, hide: &[String], transparent: bool) -> Result<Image> {
        let end = index.checked_add(1).ok_or_else(|| Error::Decode(format!("frame {index} is out of range")))?;
        let mut it = self.frames(index, Some(end), layers, hide, transparent)?;
        it.next().ok_or_else(|| Error::Decode(format!("frame {index} was not decoded")))?.map(|(_, img)| img)
    }

    /// A layer's own pixels as RGBA (content size) for the frames in [start, end) where it is
    /// active.
    pub fn layer_frames(&self, key: &str, start: Option<u32>, end: Option<u32>) -> Result<LayerFrames> {
        let i = self.layer(key)?;
        let l = &self.meta.layers[i];
        let s = start.map_or(l.start_frame, |s| s.max(l.start_frame));
        let e = end.map_or(l.end_frame, |e| e.min(l.end_frame));
        if s >= e {
            return Ok(LayerFrames::Empty);
        }
        if l.kind == Kind::Still {
            return Ok(LayerFrames::Still { image: self.still(i)?, next: s, end: e });
        }
        Ok(LayerFrames::Video { decoding: self.decode(s, Some(e), &[i])?, index: i })
    }
}

/// What decoding relies on in the metadata (the validator checks much more).
fn check_layers(meta: &Meta) -> Result<()> {
    for l in meta.layers.iter().filter(|l| l.is_video()) {
        let ((cw, ch), (w, h)) = (l.coded_size(), l.content_size());
        if cw == 0 || ch == 0 || cw > 16384 || ch > 16384 || w == 0 || h == 0 || w > cw || h > ch {
            return Err(Error::Decode(format!(
                "layer {:?}: bad coded size {cw}x{ch} / content size {w}x{h} (run `fflv check`)",
                l.id
            )));
        }
    }
    Ok(())
}

struct LayerDecoder {
    index: usize,
    width: u32,
    height: u32,
    full: bool,
    color: Decoder,
    alpha: Option<Decoder>,
}

impl LayerDecoder {
    fn run(&mut self, color: &[u8], alpha: &[u8], convert: bool) -> Result<Option<LayerFrame>> {
        let (w, h, full) = (self.width, self.height, self.full);
        let n = w as usize * h as usize;
        // the metadata's content size must lie within what the stream decodes to
        let fits = |f: &crate::codec::Frame, plane: &str| -> Result<()> {
            if f.width < w || f.height < h {
                return Err(Error::Decode(format!(
                    "layer {} {plane}: decoded {}x{}, smaller than its content size {w}x{h} (run `fflv check`)",
                    self.index, f.width, f.height
                )));
            }
            Ok(())
        };
        let (cdec, adec) = (&mut self.color, &mut self.alpha);
        let (rgb, a) = rayon::join(
            || -> Result<Option<Vec<u8>>> {
                let f = cdec.decode(color)?;
                fits(&f, "color")?;
                Ok(convert.then(|| {
                    let mut out = vec![0; n * 3];
                    frame_to_rgb(&f, w, h, &mut out);
                    out
                }))
            },
            || -> Result<Option<Vec<u8>>> {
                match adec {
                    Some(d) if !alpha.is_empty() => {
                        let f = d.decode(alpha)?;
                        fits(&f, "alpha")?;
                        Ok(convert.then(|| {
                            let mut out = vec![0; n];
                            frame_to_alpha(&f, w, h, full, &mut out);
                            out
                        }))
                    }
                    _ => Ok(None),
                }
            },
        );
        let (rgb, a) = (rgb?, a?);
        Ok(rgb.map(|rgb| LayerFrame { width: w, height: h, rgb, alpha: a }))
    }
}

/// Decoding session (see [`Reader::decode`]).
pub struct Decoding {
    caus: CauIter<Arc<LvfReader>>,
    start: u32,
    end: u32,
    planes: Vec<LayerDecoder>,
    done: bool,
}

impl Decoding {
    fn step(&mut self) -> Option<Result<(u32, BTreeMap<usize, LayerFrame>)>> {
        loop {
            let (_, cau, _) = match self.caus.next()? {
                Ok(c) => c,
                Err(e) => return Some(Err(e.into())),
            };
            let f = cau.frame_index;
            if f >= self.end {
                return None;
            }
            let convert = f >= self.start;
            let mut jobs: Vec<(&mut LayerDecoder, &lvf::VideoEntry)> = Vec::new();
            let mut entries = cau.entries.iter().filter(|e| e.kind == ENTRY_FRAME).peekable();
            for d in self.planes.iter_mut() {
                while entries.peek().is_some_and(|e| (e.layer_index as usize) < d.index) {
                    entries.next();
                }
                if let Some(e) = entries.peek().filter(|e| e.layer_index as usize == d.index) {
                    jobs.push((d, e));
                }
            }
            let results: Vec<Result<(usize, Option<LayerFrame>)>> =
                jobs.into_par_iter().map(|(d, e)| Ok((d.index, d.run(&e.color, &e.alpha, convert)?))).collect();
            if !convert {
                if let Some(Err(e)) = results.into_iter().find(|r| r.is_err()) {
                    return Some(Err(e));
                }
                continue;
            }
            let mut out = BTreeMap::new();
            for r in results {
                match r {
                    Ok((i, Some(frame))) => {
                        out.insert(i, frame);
                    }
                    Ok((_, None)) => {}
                    Err(e) => return Some(Err(e)),
                }
            }
            return Some(Ok((f, out)));
        }
    }
}

impl Iterator for Decoding {
    type Item = Result<(u32, BTreeMap<usize, LayerFrame>)>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let r = self.step();
        if !matches!(r, Some(Ok(_))) {
            self.done = true;
        }
        r
    }
}

struct Draw {
    index: usize,
    video: bool,
    start: u32,
    end: u32,
    rect: lvf::Rect,
    blend: Blend,
    opacity: f32,
}

/// Composite frames (see [`Reader::frames`]).
pub struct Frames {
    decoding: Decoding,
    draw: Vec<Draw>,
    stills: HashMap<usize, Arc<Image>>,
    width: u32,
    height: u32,
    background: Option<[u8; 3]>,
    transparent: bool,
}

impl Frames {
    /// Draw the layers in this order (layer indices, bottom first) instead of by z; layers shown
    /// but not listed are drawn above them, by z.
    pub fn set_order(&mut self, order: &[usize]) {
        self.draw.sort_by_key(|d| order.iter().position(|&i| i == d.index).unwrap_or(usize::MAX));
    }

    /// Draw layer `index` (if it is shown) with this opacity instead of its own.
    pub fn set_opacity(&mut self, index: usize, opacity: f32) {
        for d in self.draw.iter_mut().filter(|d| d.index == index) {
            d.opacity = opacity.clamp(0.0, 1.0);
        }
    }
}

impl Iterator for Frames {
    type Item = Result<(u32, Image)>;
    fn next(&mut self) -> Option<Self::Item> {
        let (f, planes) = match self.decoding.next()? {
            Ok(x) => x,
            Err(e) => return Some(Err(e)),
        };
        let mut cv = Canvas::new(self.width, self.height, self.background);
        for d in &self.draw {
            if !(d.start <= f && f < d.end) {
                continue;
            }
            let px = if d.video {
                match planes.get(&d.index) {
                    Some(frame) => frame.pixels(),
                    None => continue,
                }
            } else {
                Pixels::rgba(&self.stills[&d.index])
            };
            cv.draw(&px, d.rect, d.blend, d.opacity);
        }
        Some(Ok((f, cv.image(self.transparent))))
    }
}

/// A layer's own frames (see [`Reader::layer_frames`]).
pub enum LayerFrames {
    Empty,
    Still { image: Arc<Image>, next: u32, end: u32 },
    Video { decoding: Decoding, index: usize },
}

impl Iterator for LayerFrames {
    type Item = Result<(u32, Image)>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LayerFrames::Empty => None,
            LayerFrames::Still { image, next, end } => {
                if *next >= *end {
                    return None;
                }
                *next += 1;
                Some(Ok((*next - 1, (**image).clone())))
            }
            LayerFrames::Video { decoding, index } => loop {
                match decoding.next()? {
                    Ok((f, mut planes)) => {
                        if let Some(frame) = planes.remove(index) {
                            return Some(Ok((f, frame.to_rgba())));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            },
        }
    }
}
