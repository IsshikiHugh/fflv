//! fflv::Reader: selective decoding and compositing of files made by fflv::Writer.

use fflv::codec::Speed;
use fflv::image::{encode_png, Image};
use fflv::{LayerOptions, Reader, StillOptions, Writer, WriterOptions};
use lvf::{Fps, Rect};

const W: u32 = 48;
const H: u32 = 32;
const N: u32 = 14;

fn exact(f: u32, w: u32, h: u32) -> Image {
    let mut d = Vec::new();
    for y in 0..h {
        for x in 0..w {
            d.extend_from_slice(&[
                ((x * 7 + f) & 255) as u8,
                ((y * 13 + 3 * f) & 255) as u8,
                ((x ^ y ^ f) & 255) as u8,
                (x * 5 % 256) as u8,
            ]);
        }
    }
    Image::new(w, h, 4, d).unwrap()
}

/// bg (lossy, opaque, full canvas), exact (lossless + alpha, odd size, starts at frame 3),
/// hidden (not visible by default), logo (still, frames 2..9)
fn sample(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fflv-decode-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s.lvd");
    let opts = WriterOptions { gop: Some(5), speed: Speed::Fast, background: "#102030".into(), ..Default::default() };
    let mut w = Writer::create(&path, W, H, Fps::new(30, 1).unwrap(), opts).unwrap();
    w.add_layer("bg", LayerOptions::default()).unwrap();
    let rect = Some(Rect { x: 5, y: 3, w: 21, h: 13 });
    w.add_layer("exact", LayerOptions { alpha: true, lossless: true, rect, ..Default::default() }).unwrap();
    w.add_layer("hidden", LayerOptions { visible: false, ..Default::default() }).unwrap();
    let logo = encode_png(Image::filled(4, 4, &[255, 0, 0, 255]).view(), false).unwrap();
    let rect = Some(Rect { x: 40, y: 0, w: 4, h: 4 });
    w.add_still("logo", logo, StillOptions { rect, start: 2, end: Some(9), ..Default::default() }).unwrap();
    for f in 0..N {
        let bg = Image::filled(W, H, &[(f * 10) as u8, 100, 50]);
        let hidden = Image::filled(W, H, &[255, 255, 255]);
        let ex = exact(f, 21, 13);
        let mut images = vec![("bg", bg.view()), ("hidden", hidden.view())];
        if f >= 3 {
            images.push(("exact", ex.view()));
        }
        w.write(&images).unwrap();
    }
    assert!(w.close().unwrap().unwrap().ok());
    path
}

#[test]
fn lossless_layer_pixels_are_exact_from_any_start() {
    let path = sample("exact");
    let r = Reader::open(&path).unwrap();
    let all: Vec<(u32, Image)> = r.layer_frames("exact", None, None).unwrap().map(|x| x.unwrap()).collect();
    assert_eq!(all.iter().map(|(f, _)| *f).collect::<Vec<_>>(), (3..N).collect::<Vec<_>>());
    for (f, img) in &all {
        assert_eq!(img, &exact(*f, 21, 13), "frame {f}");
    }
    // starting between random-access points decodes from the RAP before
    let (f, img) = r.layer_frames("exact", Some(8), Some(9)).unwrap().next().unwrap().unwrap();
    assert_eq!((f, img), (8, exact(8, 21, 13)));
}

#[test]
fn composite_uses_visibility_z_and_stills() {
    let path = sample("composite");
    let r = Reader::open(&path).unwrap();
    let px = |img: &Image, x: u32, y: u32| img.data[((y * W + x) * 3) as usize..][..3].to_vec();
    let img = r.frame(4, None, &[], false).unwrap();
    assert_eq!((img.width, img.height, img.channels), (W, H, 3));
    assert_eq!(px(&img, 41, 1), vec![255, 0, 0]); // logo (still) on top
    let bg = px(&img, 1, 30);
    assert!((bg[0] as i32 - 40).abs() <= 2 && (bg[1] as i32 - 100).abs() <= 2, "{bg:?}");
    // the lossless layer blended over bg: alpha at local x=0 is 0 → bg shows through
    let bg_only = r.frame(4, Some(&["bg".to_string()]), &[], false).unwrap();
    assert_eq!(px(&img, 5, 3), px(&bg_only, 5, 3));
    assert_ne!(px(&img, 6, 3), px(&bg_only, 6, 3));
    // an explicit layer list overrides visibility; hide removes layers
    let img = r.frame(4, Some(&["hidden".to_string()]), &[], false).unwrap();
    assert!(px(&img, 1, 30).iter().all(|&v| v >= 253));
    let img = r.frame(4, None, &["logo".into(), "bg".into()], false).unwrap();
    // the background color shows where the hidden layers were
    assert_eq!(px(&img, 41, 1), vec![0x10, 0x20, 0x30]);
    // transparent output: nothing drawn → alpha 0
    let img = r.frame(4, Some(&["logo".to_string()]), &[], true).unwrap();
    assert_eq!(img.channels, 4);
    assert_eq!(img.data[3], 0);
    assert_eq!(&img.data[((W + 41) * 4) as usize..][..4], &[255, 0, 0, 255]);
}

#[test]
fn only_requested_layers_are_decoded_and_ranges_are_checked() {
    let path = sample("select");
    let r = Reader::open(&path).unwrap();
    let frames: Vec<_> = r.decode(0, Some(5), &[1]).unwrap().map(|x| x.unwrap()).collect();
    assert_eq!(frames.len(), 5);
    for (f, planes) in &frames {
        assert_eq!(planes.keys().copied().collect::<Vec<_>>(), if *f >= 3 { vec![1] } else { vec![] });
    }
    assert!(r.decode(3, Some(3), &[0]).is_err());
    assert!(r.decode(0, Some(N + 1), &[0]).is_err());
    assert!(r.layer("nope").unwrap_err().to_string().contains("no layer"));
    assert_eq!(r.rap_at_or_before(7), 5);
}
