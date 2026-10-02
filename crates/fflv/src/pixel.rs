//! Conversions between interleaved 8-bit images and the planar pictures of the VP9 streams.
//!
//! Encoding side: RGB → BT.709 limited-range 4:2:0 (chroma = mean of each 2×2 block, weighted by
//! alpha for layers with alpha), RGB → G/B/R planes (lossless), alpha → luma. Odd images are padded
//! to the even coded size by repeating the last column / row (cropped away again on playback,
//! spec B.4).
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

/// One pixel of a `C`-channel image as RGBA (gray → R = G = B, missing alpha → 255).
#[inline(always)]
fn rgba<const C: usize>(p: &[u8]) -> [u8; 4] {
    match C {
        1 => [p[0], p[0], p[0], 255],
        3 => [p[0], p[1], p[2], 255],
        _ => [p[0], p[1], p[2], p[3]],
    }
}

/// Luma of a 2×2 block `p` (top-left, top-right, bottom-left, bottom-right) into `ya` / `yb`,
/// and its chroma. `weighted`: the chroma is the alpha-weighted mean of the four colors, so fully
/// transparent pixels (whose color is arbitrary) do not bleed into the visible edge of a layer;
/// when the four alphas are equal (opaque images) this is exactly the plain mean.
#[inline(always)]
fn block(p: &[[u8; 4]; 4], ya: &mut [u8], yb: &mut [u8], u: &mut u8, v: &mut u8, weighted: bool) {
    let (mut sr, mut sg, mut sb) = (0, 0, 0);
    let (mut wr, mut wg, mut wb, mut sa) = (0, 0, 0, 0);
    for (k, q) in p.iter().enumerate() {
        let (r, g, b) = (q[0] as i32, q[1] as i32, q[2] as i32);
        let y = luma(r, g, b);
        if k < 2 {
            ya[k] = y;
        } else {
            yb[k - 2] = y;
        }
        sr += r;
        sg += g;
        sb += b;
        if weighted {
            let a = q[3] as i32;
            wr += r * a;
            wg += g * a;
            wb += b * a;
            sa += a;
        }
    }
    if weighted && sa > 0 {
        // the weighted mean, scaled like a sum of 4 samples and rounded
        (sr, sg, sb) = ((4 * wr + sa / 2) / sa, (4 * wg + sa / 2) / sa, (4 * wb + sa / 2) / sa);
    }
    *u = chroma4(CB_R, CB_G, CB_B, sr, sg, sb);
    *v = chroma4(CR_R, CR_G, CR_B, sr, sg, sb);
}

/// One row pair of [`color_i420`]: `r0` / `r1` hold `w` pixels of `C` channels each; `y0` / `y1`
/// / `u` / `v` are the output rows (2·`u.len()` luma samples each).
#[allow(clippy::too_many_arguments)]
fn i420_rows<const C: usize>(
    r0: &[u8],
    r1: &[u8],
    w: usize,
    y0: &mut [u8],
    y1: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    weighted: bool,
) {
    let weighted = weighted && C == 4;
    let half = u.len();
    let pairs = (w / 2).min(half);
    let src = r0.chunks_exact(2 * C).zip(r1.chunks_exact(2 * C));
    let dst = y0.chunks_exact_mut(2).zip(y1.chunks_exact_mut(2)).zip(u.iter_mut().zip(v.iter_mut()));
    for ((a, b), ((ya, yb), (cu, cv))) in src.zip(dst).take(pairs) {
        let p = [rgba::<C>(&a[..C]), rgba::<C>(&a[C..]), rgba::<C>(&b[..C]), rgba::<C>(&b[C..])];
        block(&p, ya, yb, cu, cv, weighted);
    }
    // an odd last column and the padding repeat the last pixel
    for i in pairs..half {
        let (x0, x1) = ((2 * i).min(w - 1) * C, (2 * i + 1).min(w - 1) * C);
        let p = [rgba::<C>(&r0[x0..]), rgba::<C>(&r0[x1..]), rgba::<C>(&r1[x0..]), rgba::<C>(&r1[x1..])];
        block(&p, &mut y0[2 * i..2 * i + 2], &mut y1[2 * i..2 * i + 2], &mut u[i], &mut v[i], weighted);
    }
}

/// BT.709 limited-range 4:2:0 picture of size `cw`×`ch` (even) from `src` (padded by repetition).
/// `weighted` (a layer with alpha, from an RGBA image): chroma is alpha-weighted (see [`block`]).
/// Without it the image's alpha is ignored, as the layer is drawn opaque.
pub fn color_i420(src: ImageRef, cw: u32, ch: u32, weighted: bool) -> Planar {
    debug_assert!(cw.is_multiple_of(2) && ch.is_multiple_of(2) && cw >= src.width && ch >= src.height);
    let mut out = Planar::new(PlaneFormat::I420, cw, ch);
    let w = src.width as usize;
    let (cwu, half) = (cw as usize, (cw / 2) as usize);
    let (yp, up, vp) = out.planes_mut();
    yp.par_chunks_mut(2 * cwu).zip(up.par_chunks_mut(half)).zip(vp.par_chunks_mut(half)).enumerate().for_each(
        |(j, ((yrows, urow), vrow))| {
            let (r0, r1) = (src.row(2 * j as u32), src.row(2 * j as u32 + 1));
            let (y0, y1) = yrows.split_at_mut(cwu);
            match src.channels {
                1 => i420_rows::<1>(r0, r1, w, y0, y1, urow, vrow, weighted),
                3 => i420_rows::<3>(r0, r1, w, y0, y1, urow, vrow, weighted),
                _ => i420_rows::<4>(r0, r1, w, y0, y1, urow, vrow, weighted),
            }
        },
    );
    out
}

/// One row of [`color_gbr`]: `src` holds the pixels (`C` channels), the planes are padded by
/// repeating the last one.
fn gbr_row<const C: usize>(src: &[u8], g: &mut [u8], b: &mut [u8], r: &mut [u8]) {
    let w = src.len() / C;
    for (((p, g), b), r) in src.chunks_exact(C).zip(g.iter_mut()).zip(b.iter_mut()).zip(r.iter_mut()) {
        let p = rgba::<C>(p);
        (*r, *g, *b) = (p[0], p[1], p[2]);
    }
    if g.len() > w {
        let p = rgba::<C>(&src[(w - 1) * C..]);
        r[w..].fill(p[0]);
        g[w..].fill(p[1]);
        b[w..].fill(p[2]);
    }
}

/// Lossless RGB picture (4:4:4, planes G, B, R as VP9 stores RGB) of size `cw`×`ch`.
pub fn color_gbr(src: ImageRef, cw: u32, ch: u32) -> Planar {
    let mut out = Planar::new(PlaneFormat::I444, cw, ch);
    let cwu = cw as usize;
    let (gp, bp, rp) = out.planes_mut();
    gp.par_chunks_mut(cwu).zip(bp.par_chunks_mut(cwu)).zip(rp.par_chunks_mut(cwu)).enumerate().for_each(
        |(y, ((g, b), r))| {
            let row = src.row(y as u32);
            match src.channels {
                1 => gbr_row::<1>(row, g, b, r),
                3 => gbr_row::<3>(row, g, b, r),
                _ => gbr_row::<4>(row, g, b, r),
            }
        },
    );
    out
}

/// Alpha as the luma of a 4:2:0 picture (neutral chroma): full range (Y = alpha, lossless
/// layers) or limited range (Y = 16..235). Images without an alpha channel are opaque.
pub fn alpha_i420(src: ImageRef, cw: u32, ch: u32, full: bool) -> Planar {
    let mut out = Planar::new(PlaneFormat::I420, cw, ch);
    let w = src.width as usize;
    let cwu = cw as usize;
    let lut: [u8; 256] = std::array::from_fn(|a| if full { a as u8 } else { alpha_to_limited(a as u8) });
    let (yp, up, vp) = out.planes_mut();
    up.fill(128);
    vp.fill(128);
    if src.channels != 4 {
        yp.fill(lut[255]);
        return out;
    }
    yp.par_chunks_mut(cwu).enumerate().for_each(|(y, row)| {
        for (v, p) in row.iter_mut().zip(src.row(y as u32).chunks_exact(4)) {
            *v = lut[p[3] as usize];
        }
        if row.len() > w {
            let last = row[w - 1];
            row[w..].fill(last);
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
    use crate::image::Image;

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
    fn chroma_is_alpha_weighted() {
        // one opaque red pixel among transparent black ones keeps the chroma of red
        let mut data = vec![0u8; 16];
        data[..4].copy_from_slice(&[255, 0, 0, 255]);
        let edge = color_i420(ImageRef::new(2, 2, 4, &data).unwrap(), 2, 2, true);
        let red = Image::filled(2, 2, &[255, 0, 0, 255]);
        let solid = color_i420(red.view(), 2, 2, true);
        assert_eq!(&edge.planes()[1..], &solid.planes()[1..]);
        // a layer without alpha shows the transparent pixels too: the plain mean of all four
        let opaque: Vec<u8> = data.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        let plain = color_i420(ImageRef::new(2, 2, 3, &opaque).unwrap(), 2, 2, false);
        assert_eq!(color_i420(ImageRef::new(2, 2, 4, &data).unwrap(), 2, 2, false), plain);
        assert_ne!(&plain.planes()[1..], &solid.planes()[1..]);
        // equal alphas: the plain mean, as without alpha
        let rgba: Vec<u8> =
            [10u8, 200, 30, 90, 250, 0, 60, 120].iter().flat_map(|&v| [v, 255 - v, v / 2, 77]).collect();
        let rgb: Vec<u8> = rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        let a = color_i420(ImageRef::new(4, 2, 4, &rgba).unwrap(), 4, 2, true);
        let b = color_i420(ImageRef::new(4, 2, 3, &rgb).unwrap(), 4, 2, false);
        assert_eq!(a, b);
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
