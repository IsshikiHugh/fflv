//! Compositing layers onto a canvas with the same formulas as the WebGL player
//! (player/src/render): straight-alpha layers, bilinear sampling when a layer is scaled to its
//! rect, blend modes normal / add / multiply / screen, layer opacity.

use lvf::Rect;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::image::Image;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    Normal,
    Add,
    Multiply,
    Screen,
}

impl Blend {
    pub fn parse(s: &str) -> Result<Blend> {
        match s {
            "normal" => Ok(Blend::Normal),
            "add" => Ok(Blend::Add),
            "multiply" => Ok(Blend::Multiply),
            "screen" => Ok(Blend::Screen),
            _ => Err(Error::Meta(format!("blend must be one of normal, add, multiply, screen, got {s:?}"))),
        }
    }
}

/// `#RRGGBB` → [r, g, b].
pub fn parse_color(s: &str) -> Result<[u8; 3]> {
    let hex = s.strip_prefix('#').filter(|h| h.len() == 6 && h.bytes().all(|b| b.is_ascii_hexdigit()));
    let hex = hex.ok_or_else(|| Error::Meta(format!("background must be #RRGGBB, got {s:?}")))?;
    let c = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    Ok([c(0), c(2), c(4)])
}

/// A layer's own pixels (straight alpha): `color` has 3 (RGB) or 4 (RGBA) channels; a separate
/// `alpha` plane, if given, overrides the fourth channel. No alpha at all means opaque.
#[derive(Clone, Copy)]
pub struct Pixels<'a> {
    pub width: u32,
    pub height: u32,
    pub color: &'a [u8],
    pub channels: usize,
    pub alpha: Option<&'a [u8]>,
}

impl<'a> Pixels<'a> {
    pub fn rgba(img: &'a Image) -> Pixels<'a> {
        Pixels { width: img.width, height: img.height, color: &img.data, channels: img.channels as usize, alpha: None }
    }

    #[inline]
    fn texel(&self, x: usize, y: usize) -> [f32; 4] {
        let i = y * self.width as usize + x;
        let c = &self.color[i * self.channels..];
        let a = match (self.alpha, self.channels) {
            (Some(a), _) => a[i],
            (None, 4) => c[3],
            _ => 255,
        };
        [c[0] as f32, c[1] as f32, c[2] as f32, a as f32]
    }
}

/// Sampling positions along one axis: (i0, i1, weight of i1) per destination pixel, like GL
/// LINEAR filtering with clamp-to-edge.
fn taps(dst: u32, src: u32, first: i64, origin: i64, count: usize) -> Vec<(usize, usize, f32)> {
    (0..count)
        .map(|k| {
            let local = (first as i128 + k as i128 - origin as i128) as f64;
            if dst == src {
                let i = local as usize;
                return (i, i, 0.0);
            }
            let s = ((local + 0.5) * src as f64 / dst as f64 - 0.5).clamp(0.0, (src - 1) as f64);
            let i0 = s.floor() as usize;
            (i0, (i0 + 1).min(src as usize - 1), (s - i0 as f64) as f32)
        })
        .collect()
}

/// Premultiplied float canvas.
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    color: Vec<f32>,
    alpha: Vec<f32>,
}

impl Canvas {
    /// Opaque `background`, or transparent (None).
    pub fn new(width: u32, height: u32, background: Option<[u8; 3]>) -> Canvas {
        let n = width as usize * height as usize;
        let (color, alpha) = match background {
            Some(bg) => {
                let px = [bg[0] as f32 / 255.0, bg[1] as f32 / 255.0, bg[2] as f32 / 255.0];
                (px.repeat(n), vec![1.0; n])
            }
            None => (vec![0.0; n * 3], vec![0.0; n]),
        };
        Canvas { width, height, color, alpha }
    }

    pub fn draw(&mut self, px: &Pixels, rect: Rect, blend: Blend, opacity: f32) {
        let (cw, ch) = (self.width as i64, self.height as i64);
        let (x0, y0) = (rect.x.max(0), rect.y.max(0));
        let (x1, y1) = (rect.x.saturating_add(rect.w as i64).min(cw), rect.y.saturating_add(rect.h as i64).min(ch));
        if x0 >= x1 || y0 >= y1 || px.width == 0 || px.height == 0 {
            return;
        }
        let cols = taps(rect.w, px.width, x0, rect.x, (x1 - x0) as usize);
        let rows = taps(rect.h, px.height, y0, rect.y, (y1 - y0) as usize);
        let w = self.width as usize;
        let (ya, yb) = (y0 as usize, y1 as usize);
        let color = &mut self.color[ya * w * 3..yb * w * 3];
        let alpha = &mut self.alpha[ya * w..yb * w];
        color.par_chunks_mut(w * 3).zip(alpha.par_chunks_mut(w)).zip(rows.par_iter()).for_each(
            |((crow, arow), &(r0, r1, fy))| {
                for (k, &(c0, c1, fx)) in cols.iter().enumerate() {
                    let s = if fx == 0.0 && fy == 0.0 {
                        px.texel(c0, r0)
                    } else {
                        let (a, b, c, d) = (px.texel(c0, r0), px.texel(c1, r0), px.texel(c0, r1), px.texel(c1, r1));
                        let mut s = [0.0; 4];
                        for i in 0..4 {
                            let top = a[i] + (b[i] - a[i]) * fx;
                            let bot = c[i] + (d[i] - c[i]) * fx;
                            s[i] = top + (bot - top) * fy;
                        }
                        s
                    };
                    let x = x0 as usize + k;
                    let src = [s[0] / 255.0, s[1] / 255.0, s[2] / 255.0];
                    let a = s[3] / 255.0 * opacity;
                    let p = &mut crow[x * 3..x * 3 + 3];
                    let big_a = arow[x];
                    match blend {
                        Blend::Add => {
                            for i in 0..3 {
                                p[i] = (p[i] + a * src[i]).min(1.0);
                            }
                        }
                        Blend::Normal => {
                            for i in 0..3 {
                                p[i] = a * src[i] + p[i] * (1.0 - a);
                            }
                        }
                        Blend::Multiply | Blend::Screen => {
                            let inv = 1.0 / big_a.max(1e-6);
                            for i in 0..3 {
                                let cb = p[i] * inv;
                                let bf = if blend == Blend::Multiply { cb * src[i] } else { cb + src[i] - cb * src[i] };
                                let mixed = (1.0 - big_a) * src[i] + big_a * bf;
                                p[i] = a * mixed + p[i] * (1.0 - a);
                            }
                        }
                    }
                    arow[x] = a + big_a * (1.0 - a);
                }
            },
        );
    }

    /// RGB, or straight-alpha RGBA when `transparent`.
    pub fn image(&self, transparent: bool) -> Image {
        let n = self.width as usize * self.height as usize;
        let q = |v: f32| (v * 255.0 + 0.5) as u8;
        let data = if !transparent {
            self.color.par_iter().map(|&v| q(v)).collect()
        } else {
            let mut out = vec![0u8; n * 4];
            out.par_chunks_mut(4).zip(self.color.par_chunks(3)).zip(self.alpha.par_iter()).for_each(|((o, c), &a)| {
                let inv = 1.0 / a.max(1e-6);
                o[0] = q(c[0] * inv);
                o[1] = q(c[1] * inv);
                o[2] = q(c[2] * inv);
                o[3] = q(a);
            });
            out
        };
        Image { width: self.width, height: self.height, channels: if transparent { 4 } else { 3 }, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i64, y: i64, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn normal_blend_and_clipping() {
        let mut cv = Canvas::new(4, 2, Some([0, 0, 255]));
        let layer = Image::filled(2, 2, &[255, 0, 0, 128]);
        cv.draw(&Pixels::rgba(&layer), rect(3, 0, 2, 2), Blend::Normal, 1.0);
        let img = cv.image(false);
        assert_eq!(&img.data[..3], &[0, 0, 255]);
        assert_eq!(&img.data[9..12], &[128, 0, 127]);
    }

    #[test]
    fn transparent_canvas_keeps_straight_colors() {
        let mut cv = Canvas::new(1, 1, None);
        let layer = Image::filled(1, 1, &[200, 100, 50, 64]);
        cv.draw(&Pixels::rgba(&layer), rect(0, 0, 1, 1), Blend::Normal, 1.0);
        assert_eq!(cv.image(true).data, vec![200, 100, 50, 64]);
    }

    #[test]
    fn blend_modes_on_opaque_canvas() {
        let run = |blend| {
            let mut cv = Canvas::new(1, 1, Some([100, 100, 100]));
            let layer = Image::filled(1, 1, &[200, 200, 200, 255]);
            cv.draw(&Pixels::rgba(&layer), rect(0, 0, 1, 1), blend, 1.0);
            cv.image(false).data[0]
        };
        assert_eq!(run(Blend::Add), 255);
        assert_eq!(run(Blend::Multiply), (100.0f32 * 200.0 / 255.0 + 0.5) as u8);
        assert_eq!(run(Blend::Screen), (255.0 - 155.0f32 * 55.0 / 255.0 + 0.5) as u8);
    }

    #[test]
    fn scaling_is_bilinear_with_clamped_edges() {
        let layer = Image::new(2, 1, 3, vec![0, 0, 0, 255, 255, 255]).unwrap();
        let mut cv = Canvas::new(4, 1, Some([0, 0, 0]));
        cv.draw(&Pixels::rgba(&layer), rect(0, 0, 4, 1), Blend::Normal, 1.0);
        let r: Vec<u8> = cv.image(false).data.chunks(3).map(|p| p[0]).collect();
        assert_eq!(r, vec![0, 64, 191, 255]);
    }
}
