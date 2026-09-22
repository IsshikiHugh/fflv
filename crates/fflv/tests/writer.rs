//! fflv::Writer: images in, valid .lvd out.

use fflv::codec::Speed;
use fflv::image::{encode_png, Image};
use fflv::{LayerOptions, StillOptions, Writer, WriterOptions};
use lvf::{validate, Fps, LvfReader, Rect};

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fflv-writer-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("out.lvd")
}

fn opts(gop: u32) -> WriterOptions {
    WriterOptions { gop: Some(gop), speed: Speed::Fast, ..Default::default() }
}

fn gray(w: u32, h: u32, v: u8) -> Image {
    Image::filled(w, h, &[v, v, v])
}

#[test]
fn layers_start_late_end_early_and_stay_sticky() {
    let path = tmp("sticky");
    let fps = Fps::new(30, 1).unwrap();
    let mut w = Writer::create(&path, 64, 48, fps, opts(4)).unwrap();
    w.add_layer("bg", LayerOptions::default()).unwrap();
    let rect = Some(Rect { x: 10, y: 5, w: 21, h: 13 });
    w.add_layer("mask", LayerOptions { alpha: true, lossless: true, rect, ..Default::default() }).unwrap();
    let logo = encode_png(Image::filled(8, 8, &[255, 0, 0, 128]).view(), false).unwrap();
    w.add_still("logo", logo, StillOptions { start: 2, end: Some(9), ..Default::default() }).unwrap();
    let mask = Image::filled(21, 13, &[1, 2, 3, 200]);
    for f in 0..12u8 {
        let bg = gray(64, 48, f * 10);
        let mut images = vec![("bg", bg.view())];
        if f == 5 {
            images.push(("mask", mask.view()));
        }
        if f == 10 {
            w.end_layer("mask").unwrap();
        }
        assert_eq!(w.write(&images).unwrap(), f as u32);
    }
    let rep = w.close().unwrap().unwrap().clone();
    assert!(rep.ok(), "{:?}", rep.errors());
    assert_eq!(rep.rap_frames, vec![0, 4, 8]);

    let r = LvfReader::open(&path).unwrap();
    let m = r.meta().unwrap();
    let mask_meta = &m.layers[1];
    assert_eq!((mask_meta.start_frame, mask_meta.end_frame), (5, 10));
    assert_eq!(mask_meta.content_size, Some([21, 13]));
    assert_eq!((m.layers[2].start_frame, m.layers[2].end_frame), (2, 9));
    assert_eq!(m.frame_count, 12);
    // frame 5 (layer start) is a key frame although it is not on the grid
    let (_, _, idx) = r.index().unwrap();
    let (cau, _) = r.cau_at(idx[5].cau_offset).unwrap();
    assert!(cau.entries[1].is_key() && !cau.is_rap());
    assert!(!path.with_file_name(".out.lvd.fflv-tmp").exists());
}

#[test]
fn declaration_and_image_errors() {
    let path = tmp("errors");
    let fps = Fps::new(25, 1).unwrap();
    let mut w = Writer::create(&path, 16, 16, fps, opts(5)).unwrap();
    w.add_layer("a", LayerOptions::default()).unwrap();
    assert!(w.add_layer("a", LayerOptions::default()).unwrap_err().to_string().contains("already used"));
    let e = w.write(&[("nope", gray(16, 16, 0).view())]).unwrap_err();
    assert!(e.to_string().contains("unknown video layer"), "{e}");
    // a wrong size is rejected before anything changes
    let e = w.write(&[("a", gray(8, 8, 0).view())]).unwrap_err();
    assert!(e.to_string().contains("image is 8x8"), "{e}");
    w.write(&[("a", gray(16, 16, 0).view())]).unwrap();
    let e = w.add_layer("b", LayerOptions::default()).unwrap_err();
    assert!(e.to_string().contains("before the first write"), "{e}");
    w.end_layer("a").unwrap();
    let e = w.write(&[("a", gray(16, 16, 0).view())]).unwrap_err();
    assert!(e.to_string().contains("contiguous"), "{e}");
    w.write(&[]).unwrap();
    w.close().unwrap();
    assert!(validate(&path).ok());
}

#[test]
fn a_layer_without_images_fails_the_close_and_writes_nothing() {
    let path = tmp("never");
    let mut w = Writer::create(&path, 16, 16, Fps::new(30, 1).unwrap(), opts(4)).unwrap();
    w.add_layer("a", LayerOptions::default()).unwrap();
    w.add_layer("b", LayerOptions::default()).unwrap();
    w.write(&[("a", gray(16, 16, 9).view())]).unwrap();
    let e = w.close().unwrap_err();
    assert!(e.to_string().contains("never received an image"), "{e}");
    assert!(!path.exists());
    assert!(std::fs::read_dir(path.parent().unwrap()).unwrap().next().is_none());
}

#[test]
fn dropping_an_unclosed_writer_discards_it() {
    let path = tmp("drop");
    {
        let mut w = Writer::create(&path, 16, 16, Fps::new(30, 1).unwrap(), opts(4)).unwrap();
        w.add_layer("a", LayerOptions::default()).unwrap();
        w.write(&[("a", gray(16, 16, 9).view())]).unwrap();
    }
    assert!(std::fs::read_dir(path.parent().unwrap()).unwrap().next().is_none());
}
