//! Layer encoding: the color (+ alpha) VP9 streams of one video layer, from 8-bit images.

use lvf::Fps;

use crate::codec::{EncodeOptions, Encoder, Planar, PlaneFormat, Signal};
use crate::error::{Error, Result};
use crate::image::ImageRef;
use crate::pixel;

pub use lvf::meta::even;

/// A frame converted to the layer's coded pictures, ready to encode.
#[derive(Clone)]
pub struct Prepared {
    pub color: Planar,
    pub alpha: Option<Planar>,
}

/// Color (+ alpha) encoder pair for one video layer of size width×height (any parity; odd sizes
/// are padded to the even coded size).
pub struct LayerEncoder {
    pub width: u32,
    pub height: u32,
    pub lossless: bool,
    name: String,
    color: Encoder,
    alpha: Option<Encoder>,
}

impl LayerEncoder {
    pub fn new(
        width: u32,
        height: u32,
        fps: Fps,
        alpha: bool,
        lossless: bool,
        opts: &EncodeOptions,
        name: &str,
    ) -> Result<LayerEncoder> {
        LayerEncoder::with_alpha_options(width, height, fps, alpha, lossless, opts, opts, name)
    }

    /// Like [`LayerEncoder::new`] with separate encoding options for the alpha plane.
    #[allow(clippy::too_many_arguments)]
    pub fn with_alpha_options(
        width: u32,
        height: u32,
        fps: Fps,
        alpha: bool,
        lossless: bool,
        opts: &EncodeOptions,
        alpha_opts: &EncodeOptions,
        name: &str,
    ) -> Result<LayerEncoder> {
        let (cw, ch) = (even(width), even(height));
        let color = if lossless {
            Encoder::new(cw, ch, fps, PlaneFormat::I444, Signal::Rgb, true, opts, &format!("layer {name:?} color"))?
        } else {
            Encoder::new(
                cw,
                ch,
                fps,
                PlaneFormat::I420,
                Signal::Bt709Limited,
                false,
                opts,
                &format!("layer {name:?} color"),
            )?
        };
        let alpha = if alpha {
            let signal = if lossless { Signal::Bt709Full } else { Signal::Bt709Limited };
            let what = format!("layer {name:?} alpha");
            Some(Encoder::new(cw, ch, fps, PlaneFormat::I420, signal, lossless, alpha_opts, &what)?)
        } else {
            None
        };
        Ok(LayerEncoder { width, height, lossless, name: name.into(), color, alpha })
    }

    pub fn has_alpha(&self) -> bool {
        self.alpha.is_some()
    }

    /// Number of plane streams (and encoders): 1, or 2 with alpha.
    pub fn streams(&self) -> usize {
        1 + self.alpha.is_some() as usize
    }

    /// libvpx threads of each plane encoder; before the first frame only.
    pub fn set_threads(&mut self, threads: u32) -> Result<()> {
        self.color.set_threads(threads)?;
        if let Some(a) = &mut self.alpha {
            a.set_threads(threads)?;
        }
        Ok(())
    }

    /// Check and convert an image (gray, RGB or RGBA; without alpha it is opaque).
    pub fn prepare(&self, img: ImageRef) -> Result<Prepared> {
        if (img.width, img.height) != (self.width, self.height) {
            return Err(Error::Meta(format!(
                "{}: image is {}x{}, the layer is {}x{}",
                self.name, img.width, img.height, self.width, self.height
            )));
        }
        let (cw, ch) = (even(self.width), even(self.height));
        let (lossless, alpha) = (self.lossless, self.alpha.is_some());
        let (color, alpha) = rayon::join(
            || if lossless { pixel::color_gbr(img, cw, ch) } else { pixel::color_i420(img, cw, ch, alpha) },
            || alpha.then(|| pixel::alpha_i420(img, cw, ch, lossless)),
        );
        Ok(Prepared { color, alpha })
    }

    /// (color packet, alpha packet — empty without alpha); both planes encode in parallel.
    pub fn encode_prepared(&mut self, p: &Prepared, key: bool) -> Result<(Vec<u8>, Vec<u8>)> {
        let (color_enc, alpha_enc, name) = (&mut self.color, &mut self.alpha, &self.name);
        let (color, alpha) = rayon::join(
            || color_enc.encode(&p.color, key),
            || match (alpha_enc, &p.alpha) {
                (Some(enc), Some(pic)) => enc.encode(pic, key),
                (None, _) => Ok(Vec::new()),
                (Some(_), None) => Err(Error::Encode(format!("{name}: prepared frame has no alpha plane"))),
            },
        );
        Ok((color?, alpha?))
    }

    pub fn encode(&mut self, img: ImageRef, key: bool) -> Result<(Vec<u8>, Vec<u8>)> {
        let p = self.prepare(img)?;
        self.encode_prepared(&p, key)
    }

    pub fn finish(&mut self) -> Result<()> {
        self.color.finish()?;
        if let Some(a) = &mut self.alpha {
            a.finish()?;
        }
        Ok(())
    }
}
