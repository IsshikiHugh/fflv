//! End to end on generated test material: pack, decode, edit, render, and the command line.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use fflv::codec::Speed;
use fflv::devtools::testsrc::{self, exact_pattern, TestsrcOptions};
use fflv::edit::{self, AddOptions, Source};
use fflv::image::Image;
use fflv::project::{pack, PackOptions};
use fflv::render::{render, RenderOptions};
use fflv::Reader;
use lvf::{pts_us, validate, LvfReader};
use serde_json::Value;

struct Packed {
    dir: PathBuf,
    path: PathBuf,
    probes: Value,
}

/// A small but complete test file: 640x360, 4 s @ 30 fps, gop 30, all layer kinds, audio.
fn packed() -> &'static Packed {
    static P: OnceLock<Packed> = OnceLock::new();
    P.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("fflv-workflow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let o = TestsrcOptions {
            out: dir.clone(),
            width: 640,
            height: 360,
            duration: "4".into(),
            gop: 30,
            ..Default::default()
        };
        let project = testsrc::generate(&o).unwrap();
        let path = dir.join("small.lvd");
        let rep = pack(&project, &PackOptions { output: Some(path.clone()), threads: None }, &mut |_| {}).unwrap();
        assert!(rep.ok(), "{:?}", rep.errors());
        let probes = serde_json::from_slice(&std::fs::read(dir.join("barcodes.json")).unwrap()).unwrap();
        Packed { dir, path, probes }
    })
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fflv-workflow-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn copy_of(name: &str) -> PathBuf {
    let p = scratch(name).join("t.lvd");
    std::fs::copy(&packed().path, &p).unwrap();
    p
}

fn read_barcode(img: &Image, x: u32, y: u32, cell: u32, cells: u32) -> Option<u32> {
    let c = img.channels as usize;
    let bits: Vec<u32> = (0..cells)
        .map(|i| {
            let (px, py) = (x + i * cell + cell / 2, y + cell / 2);
            let p = &img.data[((py * img.width + px) as usize) * c..][..3];
            u32::from((p[0] as u32 + p[1] as u32 + p[2] as u32) / 3 > 128)
        })
        .collect();
    let data = &bits[1..bits.len() - 1];
    if bits[0] != 1 || data.iter().sum::<u32>() % 2 != bits[bits.len() - 1] {
        return None;
    }
    Some(data.iter().fold(0, |n, b| n * 2 + b))
}

#[test]
fn packed_file_is_valid_with_the_expected_structure() {
    let rep = validate(&packed().path);
    assert!(rep.ok(), "{:?}", rep.errors());
    assert!(rep.warnings().is_empty(), "{:?}", rep.warnings());
    let meta = rep.meta.as_ref().unwrap();
    assert_eq!(meta["frame_count"], 120);
    assert_eq!(rep.rap_frames, vec![0, 30, 60, 90]);
    let layers = meta["layers"].as_array().unwrap();
    let sq = layers.iter().position(|l| l["id"] == "square").unwrap();
    assert_eq!((layers[sq]["start_frame"].as_u64(), layers[sq]["end_frame"].as_u64()), (Some(45), Some(75)));
    assert_eq!(rep.layer_stats[&sq].keyframes, 2); // its start (45) and the grid point 60
    assert!(meta["audio"]["pre_skip"].as_u64().unwrap() > 0);
}

#[test]
fn every_layer_frame_carries_its_own_frame_number() {
    let p = packed();
    let r = Reader::open(&p.path).unwrap();
    let bc = &p.probes["barcode"];
    let (cell, cells) = (bc["cell"].as_u64().unwrap() as u32, bc["cells"].as_u64().unwrap() as u32);
    let mut seen = 0;
    for l in r.layers().iter().filter(|l| l.is_video()) {
        let probe = &bc["layers"][&l.id];
        let x = (probe["x"].as_i64().unwrap() - l.rect.x) as u32;
        let y = (probe["y"].as_i64().unwrap() - l.rect.y) as u32;
        for item in r.layer_frames(&l.id, None, None).unwrap() {
            let (f, img) = item.unwrap();
            assert_eq!(read_barcode(&img, x, y, cell, cells), Some(f), "layer {} frame {f}", l.id);
            seen += 1;
        }
    }
    assert_eq!(seen, r.layers().iter().filter(|l| l.is_video()).map(|l| l.end_frame - l.start_frame).sum::<u32>());
}

#[test]
fn lossless_layer_is_exact_and_alpha_ramp_spans_the_range() {
    let r = Reader::open(&packed().path).unwrap();
    for item in r.layer_frames("exact", Some(55), Some(65)).unwrap() {
        let (f, img) = item.unwrap();
        for y in 0..80 {
            for x in 0..256 {
                let px = &img.data[((y * 256 + x) * 4) as usize..][..4];
                let want = if y < 64 {
                    let [r, g, b] = exact_pattern(f, x, y);
                    [r, g, b, 255]
                } else {
                    [255, 255, 255, x as u8]
                };
                assert_eq!(px, want, "frame {f} at {x},{y}");
            }
        }
    }
    let (_, calib) = r.layer_frames("calib", Some(0), Some(1)).unwrap().next().unwrap().unwrap();
    let alpha: Vec<u8> = (0..calib.width).map(|x| calib.data[((10 * calib.width + x) * 4 + 3) as usize]).collect();
    assert!(alpha[0] <= 3 && alpha[alpha.len() - 1] >= 252, "{} .. {}", alpha[0], alpha[alpha.len() - 1]);
    assert!(alpha.windows(16).step_by(16).all(|w| w[15] as i32 - w[0] as i32 >= -3));
}

#[test]
fn audio_packets_sit_in_their_frame_window() {
    let r = LvfReader::open(&packed().path).unwrap();
    let fps = r.meta().unwrap().fps();
    let mut total = 0;
    for item in r.caus(None, None) {
        let (_, cau, _) = item.unwrap();
        let (lo, hi) = (pts_us(cau.frame_index as u64, fps), pts_us(cau.frame_index as u64 + 1, fps));
        for a in &cau.audio {
            assert!(lo <= a.pts_us && a.pts_us < hi);
            total += 1;
        }
    }
    assert!(total >= 4 * 50 - 2);
}

#[test]
fn pack_rejects_alpha_on_an_opaque_source_and_keeps_the_old_output() {
    let p = packed();
    let d = scratch("alpha");
    let mut proj: Value = serde_json::from_slice(&std::fs::read(p.dir.join("test_project.json")).unwrap()).unwrap();
    let mut layer = proj["layers"][0].clone();
    layer["alpha"] = Value::Bool(true);
    layer["src"] = Value::String(p.dir.join("bg.mkv").display().to_string());
    proj["layers"] = Value::Array(vec![layer]);
    proj.as_object_mut().unwrap().remove("audio");
    std::fs::write(d.join("p.json"), proj.to_string()).unwrap();
    let out = d.join("x.lvd");
    std::fs::write(&out, b"previous").unwrap();
    let e =
        pack(&d.join("p.json"), &PackOptions { output: Some(out.clone()), threads: None }, &mut |_| {}).unwrap_err();
    assert!(e.to_string().contains("no alpha channel"), "{e}");
    assert_eq!(std::fs::read(&out).unwrap(), b"previous");
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 2);
}

fn ffmpeg(args: &[&str]) {
    let st = Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-y"]).args(args).status().unwrap();
    assert!(st.success());
}

#[test]
fn pack_pads_odd_layers_by_repeating_the_edge() {
    let d = scratch("odd");
    let red = d.join("red.mp4");
    let clear = d.join("clear.webm");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=64x48:r=30:d=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        red.to_str().unwrap(),
    ]);
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "color=c=red@0.0:s=64x48:r=30:d=1,format=yuva420p",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        clear.to_str().unwrap(),
    ]);
    for lossless in [false, true] {
        let proj = serde_json::json!({"canvas": {"width": 80, "height": 60}, "fps": "30/1", "duration": 1, "gop": 30,
            "layers": [{"id": "red", "src": "red.mp4", "rect": [0, 0, 63, 47], "lossless": lossless},
                       {"id": "clear", "src": "clear.webm", "rect": [0, 0, 63, 47], "alpha": true, "lossless": lossless}]});
        std::fs::write(d.join("p.json"), proj.to_string()).unwrap();
        let out = d.join("odd.lvd");
        assert!(pack(&d.join("p.json"), &PackOptions { output: Some(out.clone()), threads: None }, &mut |_| {})
            .unwrap()
            .ok());
        let r = Reader::open(&out).unwrap();
        assert_eq!(r.layers()[0].content_size(), (63, 47));
        let (_, red) = r.layer_frames("red", None, None).unwrap().next().unwrap().unwrap();
        let (_, clear) = r.layer_frames("clear", None, None).unwrap().next().unwrap().unwrap();
        assert_eq!((red.width, red.height), (63, 47));
        let px = |img: &Image, i: usize| img.data[i * 4..i * 4 + 4].to_vec();
        assert!((0..63 * 47).all(|i| px(&red, i)[0] > 200 && px(&red, i)[1] < 40), "edge pixels lost their color");
        assert!((0..63 * 47).all(|i| px(&clear, i)[3] <= 3), "a transparent source became opaque");
    }
}

/// {frame: [(layer id, kind, flags, color, alpha)]} for byte comparisons.
fn entries_of(path: &Path) -> Vec<Vec<(String, u8, u8, Vec<u8>, Vec<u8>)>> {
    let r = LvfReader::open(path).unwrap();
    let ids: Vec<String> = r.meta().unwrap().layers.iter().map(|l| l.id.clone()).collect();
    r.caus(None, None)
        .map(|c| {
            let (_, cau, _) = c.unwrap();
            cau.entries
                .into_iter()
                .map(|e| (ids[e.layer_index as usize].clone(), e.kind, e.frame_flags, e.color, e.alpha))
                .collect()
        })
        .collect()
}

#[test]
fn edits_never_touch_existing_packets() {
    let p = packed();
    let path = copy_of("edits");
    let before = entries_of(&path);
    let opts = AddOptions { start: 15, speed: Speed::Fast, ..Default::default() };
    edit::add_layer(&path, "copy", Source::Media(&p.dir.join("sync_a.mkv")), &opts).unwrap();
    let images = (0..12).map(|f| Ok(Image::filled(10, 10, &[f as u8, 0, 0])));
    let opts = AddOptions { start: 2, lossless: true, ..Default::default() };
    edit::add_layer(&path, "fn", Source::Images { images: Box::new(images), len: Some(12) }, &opts).unwrap();
    assert_eq!(validate(&path).rap_frames, vec![0, 30, 60, 90]);
    let after = entries_of(&path);
    for (f, (a, b)) in before.iter().zip(&after).enumerate() {
        assert_eq!(&b[..a.len()], &a[..], "frame {f}: existing packets changed");
        let copy = &b[a.len()];
        assert_eq!(copy.1, u8::from(f >= 15));
        if f >= 15 {
            assert_eq!(copy.2 & 1 == 1, [15, 30, 60, 90].contains(&f), "frame {f}");
        }
    }
    let r = Reader::open(&path).unwrap();
    let copy = &r.layers()[r.layer("copy").unwrap()];
    assert!(copy.has_alpha() && copy.start_frame == 15 && copy.z.0 > 6.0);
    let fn_frames: Vec<(u32, Image)> = r.layer_frames("fn", None, None).unwrap().map(|x| x.unwrap()).collect();
    assert_eq!(fn_frames.iter().map(|x| x.0).collect::<Vec<_>>(), (2..14).collect::<Vec<_>>());
    assert!(fn_frames.iter().all(|(f, img)| img.data[0] as u32 == f - 2));
    drop(r);

    edit::remove_layers(&path, &["calib".into(), "fn".into()], None, true).unwrap();
    let removed = entries_of(&path);
    for (a, b) in after.iter().zip(&removed) {
        let kept: Vec<_> = a.iter().filter(|e| e.0 != "calib" && e.0 != "fn").cloned().collect();
        assert_eq!(b, &kept);
    }
    // metadata edits are in place (copy-on-write): same size, same packets
    let size = std::fs::metadata(&path).unwrap().len();
    let fields = vec![
        ("opacity".to_string(), Value::from("0.5")),
        ("name".into(), Value::from("máscara ü")),
        ("rect".into(), Value::from("5,6,100,80")),
        ("id".into(), Value::from("mask")),
        ("visible".into(), Value::from("false")),
    ];
    assert!(edit::set_layer(&path, "copy", &fields, None).unwrap());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), size);
    assert!(validate(&path).ok());
    let m = LvfReader::open(&path).unwrap().meta().unwrap();
    let l = &m.layers[m.resolve_layer("mask").unwrap()];
    assert_eq!((l.opacity, l.visible, l.name.as_str(), l.rect.w), (0.5, false, "máscara ü", 100));
    let e = edit::set_layer(&path, "mask", &[("start_frame".into(), Value::from("3"))], None).unwrap_err();
    assert!(e.to_string().contains("cannot be edited"), "{e}");
    let renamed = entries_of(&path);
    assert_eq!(
        renamed.iter().map(|f| f.len()).collect::<Vec<_>>(),
        removed.iter().map(|f| f.len()).collect::<Vec<_>>()
    );
    // metadata too large for its reserved space: the file is rewritten, packets unchanged
    let big = vec![("name".to_string(), Value::from("x".repeat(20000)))];
    assert!(!edit::set_layer(&path, "bg", &big, None).unwrap());
    assert_eq!(entries_of(&path), renamed);
    edit::set_audio(&path, None, None, "128k", 2, true).unwrap();
    assert!(LvfReader::open(&path).unwrap().meta().unwrap().audio.is_none());
}

#[test]
fn render_outputs() {
    let p = packed();
    let d = scratch("render");
    let o = |start, end| RenderOptions { start, end: Some(end), ..Default::default() };
    let s = |p: &Path| p.to_str().unwrap().to_string();
    assert_eq!(render(&p.path, &s(&d.join("one.png")), &o(5, 6), None).unwrap(), 1);
    let e = render(&p.path, &s(&d.join("two.png")), &o(0, 2), None).unwrap_err();
    assert!(e.to_string().contains("one frame"), "{e}");
    assert_eq!(render(&p.path, &s(&d.join("seq/%03d.png")), &o(2, 6), None).unwrap(), 4);
    assert!(d.join("seq/005.png").exists() && !d.join("seq/006.png").exists());
    assert_eq!(render(&p.path, &format!("{}/", s(&d.join("dir"))), &o(0, 2), None).unwrap(), 2);
    assert!(d.join("dir/000001.png").exists());
    assert_eq!(render(&p.path, &s(&d.join("v.npy")), &o(1, 4), None).unwrap(), 3);
    let npy = std::fs::read(d.join("v.npy")).unwrap();
    assert!(String::from_utf8_lossy(&npy[..128]).contains("'shape': (3, 360, 640, 3)"));
    assert_eq!(npy.len(), 128 + 3 * 360 * 640 * 3);
    render(&p.path, &s(&d.join("v.mp4")), &o(0, 10), None).unwrap();
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(d.join("v.mp4"))
        .output()
        .unwrap();
    let v: Value = serde_json::from_slice(&probe.stdout).unwrap();
    let st = &v["streams"][0];
    assert_eq!(
        (st["nb_frames"].as_str(), st["color_space"].as_str(), st["color_range"].as_str()),
        (Some("10"), Some("bt709"), Some("tv"))
    );
    let t = RenderOptions { transparent: true, layers: Some(vec!["logo".into()]), ..o(100, 102) };
    let e = render(&p.path, &s(&d.join("t.mp4")), &t, None).unwrap_err();
    assert!(e.to_string().contains("alpha"), "{e}");
    render(&p.path, &s(&d.join("t.webm")), &t, None).unwrap();
    render(&p.path, &s(&d.join("t.mkv")), &t, None).unwrap();
}

fn fflv(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_fflv")).args(args).output().unwrap()
}

fn ok(args: &[&str]) -> String {
    let out = fflv(args);
    assert!(
        out.status.success(),
        "{args:?}: {}{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn command_line_workflow() {
    let p = packed();
    let f = copy_of("cli");
    let fs = f.to_str().unwrap();
    let d = f.parent().unwrap();
    let info: Value = serde_json::from_str(&ok(&["info", fs, "--json"])).unwrap();
    assert_eq!(info["valid"], true);
    assert_eq!(info["rap_frames"], serde_json::json!([0, 30, 60, 90]));
    assert!(ok(&["info", fs, "--frame", "45", "--no-meta"]).contains("I1   ok"));
    assert!(ok(&["check", "-q", fs]).starts_with("ok"));

    let img = d.join("note.png");
    ok(&["render", p.path.to_str().unwrap(), "-f", "100", "-l", "logo", "-o", img.to_str().unwrap()]);
    ok(&[
        "add",
        fs,
        "--still",
        img.to_str().unwrap(),
        "--id",
        "note",
        "--rect",
        "0,0,64,36",
        "--start",
        "1s",
        "--end",
        "2s",
    ]);
    let src = p.dir.join("sync_a.mkv");
    ok(&["add", fs, "--src", src.to_str().unwrap(), "--id", "copy", "--start", "15", "--speed", "fast"]);
    assert!(ok(&["set", fs, "copy", "opacity=0.5", "blend=screen", "visible=false"]).contains("in place"));
    ok(&["rm", fs, "calib", "--audio"]);
    let meta = &serde_json::from_str::<Value>(&ok(&["info", fs, "--json"])).unwrap()["meta"];
    let layer = |id: &str| meta["layers"].as_array().unwrap().iter().find(|l| l["id"] == id).cloned();
    assert!(layer("calib").is_none() && meta["audio"].is_null());
    let note = layer("note").unwrap();
    assert_eq!((note["start_frame"].as_u64(), note["end_frame"].as_u64()), (Some(30), Some(60)));
    let copy = layer("copy").unwrap();
    assert_eq!((copy["has_alpha"].as_bool(), copy["start_frame"].as_u64()), (Some(true), Some(15)));
    assert_eq!(
        (copy["opacity"].as_f64(), copy["blend"].as_str(), copy["visible"].as_bool()),
        (Some(0.5), Some("screen"), Some(false))
    );

    let npy = d.join("r.npy");
    ok(&["render", fs, "-f", "40:44", "-l", "bg,copy", "-o", npy.to_str().unwrap()]);
    assert!(String::from_utf8_lossy(&std::fs::read(&npy).unwrap()[..128]).contains("(4, 360, 640, 3)"));
    let e = d.join("e.npy");
    ok(&["extract", fs, "exact", "-f", "3", "-o", e.to_str().unwrap()]);
    assert!(String::from_utf8_lossy(&std::fs::read(&e).unwrap()[..128]).contains("(1, 96, 256, 4)"));

    let bad = fflv(&["set", fs, "copy", "start_frame=3"]);
    assert_eq!(bad.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("cannot be edited"));
    let bad = fflv(&["rm", fs, "nope"]);
    assert_eq!(bad.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("no layer"));
    assert_eq!(fflv(&["render", fs, "-f", "999", "-o", d.join("x.png").to_str().unwrap()]).status.code(), Some(2));
}

/// A (third-party) file whose resource region stores stills in another order than the layers:
/// the rewrite fallback of set_layer must keep every still with its own image.
#[test]
fn rewrites_keep_stills_stored_out_of_layer_order() {
    use fflv::image::encode_png;
    use fflv::{LayerOptions, StillOptions, Writer, WriterOptions};
    use lvf::{Fps, LvfWriter, Rect};

    let d = scratch("stills");
    let src = d.join("canon.lvd");
    // two colors whose PNGs have the same length (a swap would then go unnoticed by validation)
    let red = encode_png(Image::filled(8, 8, &[30, 20, 10]).view(), false).unwrap();
    let blue = encode_png(Image::filled(8, 8, &[10, 20, 30]).view(), false).unwrap();
    assert_eq!(red.len(), blue.len());
    let mut w = Writer::create(&src, 16, 8, Fps::new(30, 1).unwrap(), WriterOptions::default()).unwrap();
    w.add_layer("v", LayerOptions { rect: Some(Rect { x: 0, y: 0, w: 2, h: 2 }), ..Default::default() }).unwrap();
    let at = |x| Some(Rect { x, y: 0, w: 8, h: 8 });
    w.add_still("red", red.clone(), StillOptions { rect: at(0), ..Default::default() }).unwrap();
    w.add_still("blue", blue.clone(), StillOptions { rect: at(8), ..Default::default() }).unwrap();
    w.write(&[("v", Image::filled(2, 2, &[0, 0, 0]).view())]).unwrap();
    w.close().unwrap();

    let r = LvfReader::open(&src).unwrap();
    let mut meta = r.meta().unwrap();
    for l in meta.layers.iter_mut().filter(|l| l.resource.is_some()) {
        l.resource.as_mut().unwrap().offset = if l.id == "blue" { 0 } else { blue.len() as u64 };
    }
    let odd = d.join("odd.lvd");
    let mut w = LvfWriter::create(&odd).unwrap();
    // store blue first, red second, with no room beside the metadata: set_layer must rewrite
    w.begin(&lvf::encode_meta(&meta).unwrap(), &[blue.as_slice(), red.as_slice()].concat(), Some(0)).unwrap();
    for c in r.caus(None, None) {
        w.write_cau(&c.unwrap().1).unwrap();
    }
    w.finish(None, None).unwrap();
    assert!(validate(&odd).ok());
    assert!(!edit::set_layer(&odd, "v", &[("name".into(), Value::from("renamed"))], None).unwrap());
    let img = Reader::open(&odd).unwrap().frame(0, Some(&["red".into(), "blue".into()]), &[], false).unwrap();
    let px = |x: u32| img.data[((4 * 16 + x) * 3) as usize..][..3].to_vec();
    assert_eq!((px(4), px(12)), (vec![30, 20, 10], vec![10, 20, 30]));
}

#[test]
fn corrupt_devtool_reports_every_variant() {
    let d = scratch("corrupt");
    let out = ok(&["corrupt", packed().path.to_str().unwrap(), "--out", d.to_str().unwrap(), "--check"]);
    assert!(out.contains("13/13 broken files reported as expected"), "{out}");
}
