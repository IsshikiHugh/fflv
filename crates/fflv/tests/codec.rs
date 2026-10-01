//! VP9 streams through libvpx: exact key-frame placement, lossless round trips, value ranges.

use fflv::codec::{Decoder, EncodeOptions, Speed};
use fflv::encode::LayerEncoder;
use fflv::image::Image;
use fflv::pixel::{frame_to_alpha, frame_to_rgb};
use lvf::vp9::inspect_packet;
use lvf::Fps;

fn fps() -> Fps {
    Fps::new(30, 1).unwrap()
}

/// Deterministic noisy RGBA test image (hard for a lossy codec, trivial to compare).
fn pattern(w: u32, h: u32, f: u32) -> Image {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 7 + y * 13 + f * 31) ^ (x * y + f);
            data.extend_from_slice(&[
                (v & 255) as u8,
                ((v >> 3) & 255) as u8,
                ((x ^ y ^ f) & 255) as u8,
                ((x * 9 + f) & 255) as u8,
            ]);
        }
    }
    Image::new(w, h, 4, data).unwrap()
}

fn decode_rgba(color: &mut Decoder, alpha: &mut Decoder, c: &[u8], a: &[u8], w: u32, h: u32, full: bool) -> Vec<u8> {
    let mut rgb = vec![0; (w * h * 3) as usize];
    frame_to_rgb(&color.decode(c).unwrap(), w, h, &mut rgb);
    let mut al = vec![0; (w * h) as usize];
    frame_to_alpha(&alpha.decode(a).unwrap(), w, h, full, &mut al);
    rgb.chunks(3).zip(al).flat_map(|(p, a)| [p[0], p[1], p[2], a]).collect()
}

#[test]
fn lossless_layers_round_trip_exactly_even_at_odd_sizes() {
    for (w, h) in [(34, 18), (33, 17)] {
        let opts = EncodeOptions { speed: Speed::Fast, ..Default::default() };
        let mut enc = LayerEncoder::new(w, h, fps(), true, true, &opts, "exact").unwrap();
        let (mut dc, mut da) = (Decoder::new(1, "c").unwrap(), Decoder::new(1, "a").unwrap());
        for f in 0..6 {
            let img = pattern(w, h, f);
            let (c, a) = enc.encode(img.view(), f.is_multiple_of(4)).unwrap();
            let info = inspect_packet(&c).unwrap();
            assert_eq!(info.frames[0].profile, 1);
            let got = decode_rgba(&mut dc, &mut da, &c, &a, w, h, true);
            assert_eq!(got, img.data, "{w}x{h} frame {f}");
        }
        enc.finish().unwrap();
    }
}

#[test]
fn key_frames_exactly_where_asked() {
    let keys = [true, false, false, true, true, false, false, false, false, false, true, false];
    let opts = EncodeOptions { speed: Speed::Fast, ..Default::default() };
    let mut enc = LayerEncoder::new(64, 32, fps(), true, false, &opts, "keys").unwrap();
    for (f, &key) in keys.iter().enumerate() {
        let (c, a) = enc.encode(pattern(64, 32, f as u32).view(), key).unwrap();
        for p in [&c, &a] {
            let info = inspect_packet(p).unwrap();
            assert_eq!(info.key_frame(), key, "frame {f}");
            assert_eq!(info.shown_count(), 1);
        }
    }
    enc.finish().unwrap();
}

#[test]
fn lossy_color_uses_limited_range_and_keeps_extremes() {
    // Flat black / white / gray areas must come back exactly: limited range maps 0 → 16 and
    // 255 → 235, and decoding maps them back.
    let (w, h) = (64, 32);
    let mut data = Vec::new();
    for _y in 0..h {
        for x in 0..w {
            let v = match x / 16 {
                0 => 0,
                1 => 255,
                2 => 128,
                _ => 64,
            };
            data.extend_from_slice(&[v, v, v]);
        }
    }
    let img = Image::new(w, h, 3, data).unwrap();
    let opts = EncodeOptions { crf: 10, speed: Speed::Fast, ..Default::default() };
    let mut enc = LayerEncoder::new(w, h, fps(), false, false, &opts, "gray").unwrap();
    let (c, a) = enc.encode(img.view(), true).unwrap();
    assert!(a.is_empty());
    let info = inspect_packet(&c).unwrap();
    let k = info.key_info().unwrap();
    assert_eq!((k.profile, k.color_space, k.color_range), (0, Some(2), Some(0)));
    let mut dec = Decoder::new(1, "c").unwrap();
    let mut rgb = vec![0; (w * h * 3) as usize];
    frame_to_rgb(&dec.decode(&c).unwrap(), w, h, &mut rgb);
    for (x, want) in [(4, 0u8), (20, 255), (36, 128), (52, 64)] {
        let i = ((10 * w + x) * 3) as usize;
        for ch in 0..3 {
            assert!((rgb[i + ch] as i32 - want as i32).abs() <= 1, "x={x}: {:?} vs {want}", &rgb[i..i + 3]);
        }
    }
}

#[test]
fn lossy_alpha_is_limited_and_lossless_alpha_is_full_range() {
    let img = Image::filled(32, 16, &[255, 255, 255, 255]);
    for (lossless, range) in [(false, 0), (true, 1)] {
        let opts = EncodeOptions { speed: Speed::Fast, ..Default::default() };
        let mut enc = LayerEncoder::new(32, 16, fps(), true, lossless, &opts, "a").unwrap();
        let (_, a) = enc.encode(img.view(), true).unwrap();
        let info = inspect_packet(&a).unwrap();
        assert_eq!(info.key_info().unwrap().color_range, Some(range));
        let mut dec = Decoder::new(1, "a").unwrap();
        let fr = dec.decode(&a).unwrap();
        let fr_y = fr.planes[0][..8].to_vec();
        let mut al = vec![0; 32 * 16];
        frame_to_alpha(&fr, 32, 16, lossless, &mut al);
        let lo = al.iter().min().unwrap();
        assert!(*lo >= 254, "lossless={lossless}: min alpha {lo}, luma {fr_y:?}");
    }
}

#[test]
fn wrong_image_size_is_rejected_before_encoding() {
    let opts = EncodeOptions::default();
    let mut enc = LayerEncoder::new(32, 16, fps(), false, false, &opts, "sz").unwrap();
    let err = enc.encode(Image::filled(16, 16, &[0, 0, 0]).view(), true).unwrap_err();
    assert!(err.to_string().contains("image is 16x16, the layer is 32x16"), "{err}");
    // the encoder is still usable
    enc.encode(Image::filled(32, 16, &[0, 0, 0]).view(), true).unwrap();
    enc.finish().unwrap();
}
