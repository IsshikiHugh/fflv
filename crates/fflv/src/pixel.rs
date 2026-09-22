//! Conversions between interleaved 8-bit images and the planar pictures of the VP9 streams.
//!
//! Encoding side: RGB → BT.709 limited-range 4:2:0 (chroma = mean of each 2×2 block), RGB → G/B/R
//! planes (lossless), alpha → luma. Odd images are padded to the even coded size by repeating the
//! last column / row (cropped away again on playback, spec B.4).
//!
//! Decoding side: the exact matrix formula (BT.601 / 709 / 2020, limited or full range, as the
//! stream says) with nearest-neighbour chroma, and luma → alpha. Done here rather than by a
//! library so that the result does not depend on the frame width or the CPU.

use rayon::prelude::*;

use crate::codec::{Frame, Planar, PlaneFormat};
use crate::image::ImageRef;

const KR: f64 = 0.2126;
const KB: f64 = 0.0722;
const KG: f64 = 1.0 - KR - KB;
const Y_SCALE: f64 = 219.0 / 255.0;
const C_SCALE: f64 = 224.0 / 255.0;

/// 16-bit fixed point, rounded to nearest.
const fn fx(v: f64) -> i32 {
    (v * 65536.0 + if v < 0.0 { -0.5 } else { 0.5 }) as i32
}

const Y_R: i32 = fx(KR * Y_SCALE);
const Y_G: i32 = fx(KG * Y_SCALE);
const Y_B: i32 = fx(KB * Y_SCALE);
const CB_R: i32 = fx(-KR * C_SCALE / (2.0 * (1.0 - KB)));
const CB_G: i32 = fx(-KG * C_SCALE / (2.0 * (1.0 - KB)));
const CB_B: i32 = fx(C_SCALE / 2.0);
const CR_R: i32 = fx(C_SCALE / 2.0);
const CR_G: i32 = fx(-KG * C_SCALE / (2.0 * (1.0 - KR)));
const CR_B: i32 = fx(-KB * C_SCALE / (2.0 * (1.0 - KR)));

#[inline]
fn luma(r: i32, g: i32, b: i32) -> u8 {
    (((16 << 16) + Y_R * r + Y_G * g + Y_B * b + (1 << 15)) >> 16) as u8
}

/// Chroma from the sums of 4 samples.
#[inline]
fn chroma4(cr: i32, cg: i32, cb: i32, r: i32, g: i32, b: i32) -> u8 {
    (((128 << 18) + cr * r + cg * g + cb * b + (1 << 17)) >> 18).clamp(0, 255) as u8
}

/// Limited-range luma for an alpha value: 0..255 → 16..235.
#[inline]
pub fn alpha_to_limited(a: u8) -> u8 {
    ((a as u32 * 219 + 127) / 255 + 16) as u8
}

/// BT.709 limited-range 4:2:0 picture of size `cw`×`ch` (even) from `src` (padded by repetition).
pub fn color_i420(src: ImageRef, cw: u32, ch: u32) -> Planar {
    debug_assert!(cw % 2 == 0 && ch % 2 == 0 && cw >= src.width && ch >= src.height);
    let mut out = Planar::new(PlaneFormat::I420, cw, ch);
    let (w, h) = (src.width, src.height);
    let (cwu, half) = (cw as usize, (cw / 2) as usize);
    let (yp, up, vp) = out.planes_mut();
    yp.par_chunks_mut(2 * cwu).zip(up.par_chunks_mut(half)).zip(vp.par_chunks_mut(half)).enumerate().for_each(
        |(j, ((yrows, urow), vrow))| {
            let rows = [(2 * j as u32).min(h - 1), (2 * j as u32 + 1).min(h - 1)];
            for i in 0..half {
                let (mut sr, mut sg, mut sb) = (0, 0, 0);
                for (dy, &sy) in rows.iter().enumerate() {
                    for dx in 0..2 {
                        let x = 2 * i + dx;
                        let p = src.rgba_at((x as u32).min(w - 1), sy);
                        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
                        yrows[dy * cwu + x] = luma(r, g, b);
                        sr += r;
                        sg += g;
                        sb += b;
                    }
                }
                urow[i] = chroma4(CB_R, CB_G, CB_B, sr, sg, sb);
                vrow[i] = chroma4(CR_R, CR_G, CR_B, sr, sg, sb);
            }
        },
    );
    out
}

/// Lossless RGB picture (4:4:4, planes G, B, R as VP9 stores RGB) of size `cw`×`ch`.
pub fn color_gbr(src: ImageRef, cw: u32, ch: u32) -> Planar {
    let mut out = Planar::new(PlaneFormat::I444, cw, ch);
    let (w, h) = (src.width, src.height);
    let cwu = cw as usize;
    let (gp, bp, rp) = out.planes_mut();
    gp.par_chunks_mut(cwu).zip(bp.par_chunks_mut(cwu)).zip(rp.par_chunks_mut(cwu)).enumerate().for_each(
        |(y, ((g, b), r))| {
            let sy = (y as u32).min(h - 1);
            for x in 0..cwu {
                let p = src.rgba_at((x as u32).min(w - 1), sy);
                r[x] = p[0];
                g[x] = p[1];
                b[x] = p[2];
            }
        },
    );
    out
}

/// Alpha as the luma of a 4:2:0 picture (neutral chroma): full range (Y = alpha, lossless
/// layers) or limited range (Y = 16..235). Images without an alpha channel are opaque.
pub fn alpha_i420(src: ImageRef, cw: u32, ch: u32, full: bool) -> Planar {
    let mut out = Planar::new(PlaneFormat::I420, cw, ch);
    let (w, h) = (src.width, src.height);
    let cwu = cw as usize;
    let (yp, up, vp) = out.planes_mut();
    up.fill(128);
    vp.fill(128);
    yp.par_chunks_mut(cwu).enumerate().for_each(|(y, row)| {
        let sy = (y as u32).min(h - 1);
        for (x, v) in row.iter_mut().enumerate() {
            let a = src.rgba_at((x as u32).min(w - 1), sy)[3];
            *v = if full { a } else { alpha_to_limited(a) };
        }
    });
    out
}

// ------------------------------------------------------------------------------------------------
// Decoding
// ------------------------------------------------------------------------------------------------
/// (Kr, Kb) for a libvpx color space.
fn matrix(color_space: u32) -> (f64, f64) {
    match color_space {
        1 | 3 => (0.299, 0.114), // BT.601, SMPTE 170M
        4 => (0.212, 0.087),     // SMPTE 240M
        5 => (0.2627, 0.0593),   // BT.2020
        _ => (0.2126, 0.0722),   // BT.709, unknown
    }
}

/// Per-sample-value contributions of the matrix formula, so each pixel is a few additions.
struct YuvTables {
    y: [f32; 256],
    rv: [f32; 256],
    gu: [f32; 256],
    gv: [f32; 256],
    bu: [f32; 256],
}

impl YuvTables {
    fn new(color_space: u32, full: bool) -> YuvTables {
        let (kr, kb) = matrix(color_space);
        let kg = 1.0 - kr - kb;
        let (ys, cs) = if full { (1.0f32, 1.0f64) } else { ((255.0 / 219.0) as f32, 255.0 / 224.0) };
        let c_rv = (2.0 * (1.0 - kr) * cs) as f32;
        let c_gu = (-2.0 * kb * (1.0 - kb) / kg * cs) as f32;
        let c_gv = (-2.0 * kr * (1.0 - kr) / kg * cs) as f32;
        let c_bu = (2.0 * (1.0 - kb) * cs) as f32;
        let mut t = YuvTables { y: [0.0; 256], rv: [0.0; 256], gu: [0.0; 256], gv: [0.0; 256], bu: [0.0; 256] };
        for i in 0..256 {
            let v = i as f32;
            t.y[i] = if full { v } else { (v - 16.0) * ys };
            let c = v - 128.0;
            t.rv[i] = c * c_rv;
            t.gu[i] = c * c_gu;
            t.gv[i] = c * c_gv;
            t.bu[i] = c * c_bu;
        }
        t
    }
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v + 0.5) as u8 // saturating: clamps to 0..255, NaN → 0
}

/// RGB (w×h×3, the top-left w×h of the picture) of a decoded color frame.
pub fn frame_to_rgb(f: &Frame, w: u32, h: u32, out: &mut [u8]) {
    let (w, h) = (w.min(f.width) as usize, h.min(f.height) as usize);
    let out = &mut out[..w * h * 3];
    let [yp, up, vp] = f.planes;
    let [ys, us, vs] = f.strides;
    let (xs, ysh) = f.chroma_shift();
    if f.is_rgb() {
        out.par_chunks_mut(w * 3).enumerate().for_each(|(r, row)| {
            let (g, b, rr) = (&yp[r * ys..], &up[r * us..], &vp[r * vs..]);
            for (x, px) in row.chunks_exact_mut(3).enumerate() {
                px[0] = rr[x];
                px[1] = g[x];
                px[2] = b[x];
            }
        });
        return;
    }
    let t = YuvTables::new(f.color_space, f.full_range);
    out.par_chunks_mut(w * 3).enumerate().for_each(|(r, row)| {
        let yrow = &yp[r * ys..];
        let cr = r >> ysh;
        let (urow, vrow) = (&up[cr * us..], &vp[cr * vs..]);
        for (x, px) in row.chunks_exact_mut(3).enumerate() {
            let yv = t.y[yrow[x] as usize];
            let (u, v) = (urow[x >> xs] as usize, vrow[x >> xs] as usize);
            px[0] = to_u8(yv + t.rv[v]);
            px[1] = to_u8(yv + (t.gu[u] + t.gv[v]));
            px[2] = to_u8(yv + t.bu[u]);
        }
    });
}

/// Alpha (w×h) from the luma of a decoded alpha frame.
pub fn frame_to_alpha(f: &Frame, w: u32, h: u32, full: bool, out: &mut [u8]) {
    let (w, h) = (w.min(f.width) as usize, h.min(f.height) as usize);
    let mut lut = [0u8; 256];
    for (y, v) in lut.iter_mut().enumerate() {
        *v = if full { y as u8 } else { ((((y as i32 - 16) * 510).div_euclid(219) + 1) >> 1).clamp(0, 255) as u8 };
    }
    let (yp, ys) = (f.planes[0], f.strides[0]);
    out[..w * h].par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = lut[yp[r * ys + x] as usize];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_range_extremes() {
        assert_eq!(luma(0, 0, 0), 16);
        assert_eq!(luma(255, 255, 255), 235);
        assert_eq!(chroma4(CB_R, CB_G, CB_B, 1020, 1020, 1020), 128);
        assert_eq!(chroma4(CR_R, CR_G, CR_B, 0, 0, 0), 128);
        assert_eq!(chroma4(CB_R, CB_G, CB_B, 0, 0, 1020), 240);
        assert_eq!(chroma4(CR_R, CR_G, CR_B, 1020, 0, 0), 240);
        assert_eq!(alpha_to_limited(0), 16);
        assert_eq!(alpha_to_limited(255), 235);
    }

    #[test]
    fn limited_alpha_roundtrip_is_exact() {
        // 220 luma levels carry 256 alpha levels: the round trip must be within one step and
        // exact at both ends.
        let t: Vec<u8> = (0..=255u8).map(alpha_to_limited).collect();
        for (a, &y) in t.iter().enumerate() {
            let back = ((((y as i32 - 16) * 510).div_euclid(219) + 1) >> 1).clamp(0, 255);
            assert!((back - a as i32).abs() <= 1, "{a} -> {y} -> {back}");
        }
        assert_eq!(t[0], 16);
        assert_eq!(t[255], 235);
    }

    #[test]
    fn padding_repeats_the_last_column_and_row() {
        let data: Vec<u8> = (0..3 * 3).flat_map(|i| [i as u8 * 20, 0, 0, 255]).collect();
        let img = ImageRef::new(3, 3, 4, &data).unwrap();
        let p = color_gbr(img, 4, 4);
        let [_, _, r] = p.planes();
        assert_eq!(&r[..4], &[0, 20, 40, 40]);
        assert_eq!(&r[12..16], &[120, 140, 160, 160]);
    }
}
