//! Validator tests on synthetic files: a valid file passes; each deliberate breakage (spec 11.2-2)
//! is reported under the right invariant and frame; metadata must be standard JSON; copy-on-write
//! metadata edits survive a crash.
//!
//! The VP9 packets are minimal hand-made uncompressed headers — the validator only parses those.

use std::path::{Path, PathBuf};

use lvf::binary::pack_index;
use lvf::constants::*;
use lvf::container::rewrite_meta_in_place_hooked;
use lvf::meta::{self, VideoLayerSpec};
use lvf::*;

// ------------------------------------------------------------------------------------------------
// synthetic VP9 headers
// ------------------------------------------------------------------------------------------------
struct Bits(Vec<u8>, usize);
impl Bits {
    fn put(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            if self.1 % 8 == 0 {
                self.0.push(0);
            }
            let bit = ((v >> i) & 1) as u8;
            *self.0.last_mut().unwrap() |= bit << (7 - self.1 % 8);
            self.1 += 1;
        }
    }
}

fn vp9(key: bool, w: u32, h: u32, lossless_rgb: bool, full_range: bool) -> Vec<u8> {
    let mut b = Bits(Vec::new(), 0);
    b.put(2, 2); // frame_marker
    b.put(lossless_rgb as u32, 1); // profile_low_bit
    b.put(0, 1); // profile_high_bit
    b.put(0, 1); // show_existing_frame
    b.put(if key { 0 } else { 1 }, 1); // frame_type
    b.put(1, 1); // show_frame
    b.put(0, 1); // error_resilient_mode
    if key {
        b.put(0x49, 8);
        b.put(0x83, 8);
        b.put(0x42, 8);
        if lossless_rgb {
            b.put(7, 3); // CS_RGB
            b.put(0, 1); // reserved (profile 1)
        } else {
            b.put(2, 3); // BT.709
            b.put(full_range as u32, 1);
        }
        b.put(w - 1, 16);
        b.put(h - 1, 16);
    }
    b.0.extend_from_slice(&[0; 4]);
    b.0
}

// ------------------------------------------------------------------------------------------------
// a small valid file
// ------------------------------------------------------------------------------------------------
const N: u32 = 40;
const GOP: u32 = 10;

struct Layout {
    meta: Meta,
    caus: Vec<Cau>,
}

fn fps() -> Fps {
    Fps::new(30, 1).unwrap()
}

fn layout() -> Layout {
    let spec = |id: &'static str, rect: Rect, start: u32, end: u32, alpha: bool, lossless: bool, z: f64| {
        meta::video_layer(VideoLayerSpec {
            id,
            name: id,
            z: Z(z),
            rect,
            start,
            end,
            fps: fps(),
            alpha,
            lossless,
            blend: "normal",
            opacity: 1.0,
            visible: true,
        })
    };
    let layers = vec![
        spec("bg", Rect { x: 0, y: 0, w: 64, h: 32 }, 0, N, false, false, 0.0),
        spec("ov", Rect { x: 4, y: 4, w: 16, h: 16 }, 0, N, true, false, 1.0),
        spec("late", Rect { x: 30, y: 2, w: 21, h: 13 }, 15, 35, true, true, 2.0),
        meta::still_layer("s", "s", Z(3.0), Rect { x: 0, y: 0, w: 8, h: 8 }, 5, 25, 0, 16, "normal", 1.0, true),
    ];
    let audio = meta::AudioMeta {
        codec: "opus".into(),
        sample_rate: 48000,
        channels: 2,
        description_b64: Some(validate::base64_encode(b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00")),
        pre_skip: Some(312),
        extra: Default::default(),
    };
    let meta = meta::file_meta(
        meta::Canvas { width: 64, height: 32, background: "#000000".into() },
        fps(),
        N,
        GOP,
        layers,
        Some(audio),
    );
    let mut caus = Vec::new();
    let mut apts = 0i64;
    for f in 0..N {
        let mut entries = Vec::new();
        let mut rap = true;
        for (i, l) in meta.layers.iter().enumerate().filter(|(_, l)| l.is_video()) {
            if !l.active(f) {
                entries.push(VideoEntry::empty(i as u16));
                continue;
            }
            let key = f == l.start_frame || f % GOP == 0;
            rap &= key;
            let (w, h) = l.coded_size();
            let color = vp9(key, w, h, l.lossless, false);
            let alpha = if l.has_alpha() { vp9(key, w, h, false, l.alpha_full_range()) } else { vec![] };
            entries.push(VideoEntry::frame(i as u16, key, color, alpha));
        }
        let hi = pts_us(f as u64 + 1, fps());
        let mut audio = Vec::new();
        while apts < hi {
            audio.push(AudioPacket { pts_us: apts, duration_us: 20000, data: vec![0xfc, 1, 2] });
            apts += 20000;
        }
        caus.push(Cau::new(f, rap, entries, audio));
    }
    Layout { meta, caus }
}

fn png16() -> Vec<u8> {
    let mut p = PNG_SIGNATURE.to_vec();
    p.extend_from_slice(&[0; 8]);
    p
}

fn write(path: &Path, meta_bytes: &[u8], caus: &[Cau], fix_index: impl FnOnce(&mut Vec<IndexEntry>)) {
    let mut w = LvfWriter::create(path).unwrap();
    w.begin(meta_bytes, &png16(), None).unwrap();
    for c in caus {
        w.write_cau(c).unwrap();
    }
    let mut idx = w.index.clone();
    fix_index(&mut idx);
    w.finish(Some(pack_index(&idx, MAGIC_INDEX, None)), None).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lvf-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn build(name: &str, mutate: impl FnOnce(&mut Layout), fix_index: impl FnOnce(&mut Vec<IndexEntry>)) -> PathBuf {
    let mut l = layout();
    mutate(&mut l);
    let path = tmp(name);
    write(&path, &encode_meta(&l.meta).unwrap(), &l.caus, fix_index);
    path
}

fn codes(rep: &Report) -> Vec<String> {
    rep.error_codes().into_iter().collect()
}

fn frames_of(rep: &Report, code: &str) -> Vec<u32> {
    rep.errors().iter().filter(|i| i.code == code).filter_map(|i| i.frame).collect()
}

// ------------------------------------------------------------------------------------------------
#[test]
fn the_synthetic_file_is_valid() {
    let path = build("valid.lvd", |_| {}, |_| {});
    let rep = validate(&path);
    assert!(rep.ok(), "{:?}", rep.issues);
    assert!(rep.warnings().is_empty(), "{:?}", rep.warnings());
    assert_eq!(rep.rap_frames, vec![0, 10, 20, 30]);
    assert_eq!(rep.layer_stats[&2].frames, 20);
    assert_eq!(rep.layer_stats[&2].keyframes, 3); // 15, 20, 30
    let r = LvfReader::open(&path).unwrap();
    let m = r.meta().unwrap();
    assert_eq!(m.layers[2].content_size, Some([21, 13]));
    assert_eq!(r.caus(None, None).count(), N as usize);
}

fn entry(l: &mut Layout, f: usize, layer: u16) -> &mut VideoEntry {
    l.caus[f].entries.iter_mut().find(|e| e.layer_index == layer).unwrap()
}

#[test]
fn broken_files_are_reported_precisely() {
    type Mutate = Box<dyn Fn(&mut Layout)>;
    let cases: Vec<(&str, Mutate, &[&str], Option<u32>)> = vec![
        (
            "missing_entry",
            Box::new(|l| {
                l.caus[30].entries.retain(|e| e.layer_index != 2);
                l.caus[30].video_entry_count = None;
            }),
            &["I2"],
            Some(30),
        ),
        ("entry_order", Box::new(|l| l.caus[31].entries.swap(0, 1)), &["I2"], Some(31)),
        ("active_entry_empty", Box::new(|l| *entry(l, 32, 0) = VideoEntry::empty(0)), &["I3"], Some(32)),
        (
            "alpha_key_mismatch",
            Box::new(|l| {
                let a = entry(l, 11, 1).alpha.clone();
                entry(l, 10, 1).alpha = a;
            }),
            &["I5", "I6"],
            Some(10),
        ),
        (
            "layer_start_not_key",
            Box::new(|l| {
                let next = entry(l, 16, 2).clone();
                let e = entry(l, 15, 2);
                e.color = next.color;
                e.alpha = next.alpha;
                e.frame_flags = 0;
            }),
            &["I4"],
            Some(15),
        ),
        ("rap_flag_cleared", Box::new(|l| l.caus[10].flags = 0), &["I6", "I8"], Some(10)),
        ("frame0_not_rap", Box::new(|l| l.caus[0].flags = 0), &["I6", "I7"], Some(0)),
        ("frame_index_gap", Box::new(|l| l.caus[20].frame_index = 21), &["I1"], Some(20)),
        (
            "dropped_cau",
            Box::new(|l| {
                l.caus.remove(20);
            }),
            &["I1"],
            Some(20),
        ),
        (
            "audio_wrong_frame",
            Box::new(|l| {
                let pk = l.caus[10].audio.pop().unwrap();
                l.caus[11].audio.insert(0, pk);
            }),
            &["I9"],
            Some(11),
        ),
        ("hold_entry", Box::new(|l| entry(l, 33, 0).kind = ENTRY_HOLD), &["CAU"], Some(33)),
    ];
    for (name, mutate, expect, frame) in cases {
        let path = build(&format!("{name}.lvd"), |l| mutate(l), |_| {});
        let rep = validate(&path);
        let got = codes(&rep);
        for code in expect {
            assert!(got.contains(&code.to_string()), "{name}: expected {code}, got {got:?}: {:?}", rep.issues);
        }
        if let Some(f) = frame {
            assert!(
                frames_of(&rep, expect[0]).contains(&f),
                "{name}: expected an error at frame {f}: {:?}",
                rep.issues
            );
        }
    }
    let path = build("index_wrong_offset.lvd", |_| {}, |idx| idx[25].cau_offset += 4);
    assert_eq!(frames_of(&validate(&path), "I10"), vec![25]);
    let path = build("index_wrong_rap.lvd", |_| {}, |idx| idx[10].flags = 0);
    assert_eq!(frames_of(&validate(&path), "I10"), vec![10]);
}

#[test]
fn a_dropped_frame_is_reported_once_not_as_a_cascade() {
    let path = build(
        "dropped_once.lvd",
        |l| {
            l.caus.remove(20);
        },
        |_| {},
    );
    let rep = validate(&path);
    let i1: Vec<_> = rep.errors().into_iter().filter(|i| i.code == "I1").collect();
    assert_eq!(i1.len(), 2, "{i1:?}"); // the gap, and the count
    assert!(i1[0].message.contains("frame 20 is missing"));
}

#[test]
fn bitstream_must_match_the_metadata() {
    // alpha plane signals full range, metadata says limited
    let path = build(
        "alpha_range.lvd",
        |l| {
            let a = vp9(true, 16, 16, false, true);
            entry(l, 0, 1).alpha = a;
        },
        |_| {},
    );
    let rep = validate(&path);
    assert!(rep.errors().iter().any(|i| i.code == "VP9" && i.message.contains("alpha_range")), "{:?}", rep.issues);
    // wrong coded size
    let path = build("size.lvd", |l| entry(l, 0, 0).color = vp9(true, 32, 32, false, false), |_| {});
    assert!(validate(&path).errors().iter().any(|i| i.code == "VP9" && i.message.contains("32x32")));
}

#[test]
fn metadata_must_be_standard_json() {
    let l = layout();
    let good = String::from_utf8(encode_meta(&l.meta).unwrap()).unwrap();
    for (bad, why) in [
        (good.replacen("\"z\": 1", "\"z\": NaN", 1), "RFC 8259"),
        (good.replacen("\"z\": 1", "\"z\": 1e999", 1), "RFC 8259"),
    ] {
        let path = tmp("json.lvd");
        write(&path, bad.as_bytes(), &l.caus, |_| {});
        let rep = validate(&path);
        assert!(!rep.ok() && rep.errors()[0].message.contains(why), "{:?}", rep.issues);
    }
    let path = tmp("bom.lvd");
    let mut bom = b"\xef\xbb\xbf".to_vec();
    bom.extend_from_slice(good.as_bytes());
    write(&path, &bom, &l.caus, |_| {});
    assert!(validate(&path).errors()[0].message.contains("byte-order mark"));
}

#[test]
fn copy_on_write_metadata_edits_survive_a_crash() {
    let path = build("cow.lvd", |_| {}, |_| {});
    let r = LvfReader::open(&path).unwrap();
    let mut m = r.meta().unwrap();
    let (cap, len) = (r.header.resources_offset - HEADER_SIZE as u64, r.header.meta_length as u64);
    drop(r);
    let size = std::fs::metadata(&path).unwrap().len();
    // crash after the new copy is written, before the header switches: old metadata intact
    m.layers[0].name = "x".repeat((cap / 2 - len - 64) as usize);
    let data = encode_meta(&m).unwrap();
    let res = rewrite_meta_in_place_hooked(&path, &data, &mut |_| Err(Error::Value("power cut".into())));
    assert!(res.is_err());
    assert!(validate(&path).ok());
    assert_eq!(LvfReader::open(&path).unwrap().meta().unwrap().layers[0].name, "bg");
    // successive edits alternate between the two ends of the region and never grow the file
    for n in [(cap / 2 - len - 64) as usize, 3, 400, 10, (cap / 2 - len - 64) as usize] {
        m.layers[0].name = "y".repeat(n);
        assert!(rewrite_meta_in_place(&path, &encode_meta(&m).unwrap()).unwrap());
        let rep = validate(&path);
        assert!(rep.ok(), "{:?}", rep.issues);
        assert_eq!(LvfReader::open(&path).unwrap().meta().unwrap().layers[0].name.len(), n);
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), size);
    // no room beside the current copy: refused, nothing changed
    m.layers[0].name = "z".repeat(cap as usize);
    assert!(!rewrite_meta_in_place(&path, &encode_meta(&m).unwrap()).unwrap());
    assert!(validate(&path).ok());
}

#[test]
fn garbage_and_truncation_are_reported() {
    let path = tmp("garbage.lvd");
    std::fs::write(&path, [b"NOPE".as_slice(), &[0u8; 100]].concat()).unwrap();
    let rep = validate(&path);
    assert!(rep.fatal && codes(&rep).contains(&"HDR".to_string()));
    let good = build("trunc_src.lvd", |_| {}, |_| {});
    let data = std::fs::read(&good).unwrap();
    let path = tmp("truncated.lvd");
    std::fs::write(&path, &data[..data.len() / 2]).unwrap();
    let rep = validate(&path);
    assert!(!rep.ok());
}

#[test]
fn unknown_metadata_fields_survive_a_roundtrip() {
    let l = layout();
    let mut v = serde_json::to_value(&l.meta).unwrap();
    v["x_custom"] = serde_json::json!({"k": [1, 2]});
    v["layers"][0]["x_note"] = serde_json::json!("keep me");
    let m: Meta = serde_json::from_value(v.clone()).unwrap();
    assert_eq!(serde_json::to_value(&m).unwrap(), v);
}

#[test]
fn an_invalid_result_never_replaces_the_destination() {
    let dst = tmp("published.lvd");
    std::fs::write(&dst, b"previous version").unwrap();
    let part = temp_path_for(&dst);
    std::fs::write(&part, [b"LVF1".as_slice(), &[0u8; 100]].concat()).unwrap();
    match publish(&part, &dst, true) {
        Err(PublishError::Invalid { report, .. }) => assert!(!report.ok()),
        other => panic!("expected an invalid result, got {other:?}"),
    }
    assert_eq!(std::fs::read(&dst).unwrap(), b"previous version");
    assert!(!part.exists());
    // a valid file is published, and its report names the destination
    let good = build("publish_src.lvd", |_| {}, |_| {});
    std::fs::copy(&good, &part).unwrap();
    let rep = publish(&part, &dst, true).unwrap().unwrap();
    assert!(rep.ok() && rep.path == dst.display().to_string());
    assert_eq!(std::fs::read(&dst).unwrap(), std::fs::read(&good).unwrap());
}

#[test]
fn temporary_files_are_unique_per_writer() {
    let a = temp_path_for("x/out.lvd");
    let b = temp_path_for("x/out.lvd");
    assert_ne!(a, b);
    let name = a.file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.starts_with(".out.lvd.") && name.ends_with(".fflv-tmp"), "{name}");
}

/// Offsets and numbers from a hostile file must give errors, never panics or false VALIDs.
#[test]
fn out_of_range_offsets_and_numbers() {
    let good = build("range_src.lvd", |_| {}, |_| {});
    let data = std::fs::read(&good).unwrap();

    let mut h = FileHeader::unpack(&data[..HEADER_SIZE]).unwrap();
    h.index_offset = 1 << 52;
    let path = tmp("far_index.lvd");
    std::fs::write(&path, [h.pack().as_slice(), &data[HEADER_SIZE..]].concat()).unwrap();
    assert!(LvfReader::open(&path).unwrap().index().is_err());
    assert!(LvfReader::open(&path).unwrap().read(u64::MAX - 3, 8).is_err());
    let rep = validate(&path);
    assert!(!rep.ok() && codes(&rep).contains(&"HDR".to_string()));

    let edit = |name: &str, f: &dyn Fn(&mut serde_json::Value)| {
        let path = tmp(name);
        std::fs::copy(&good, &path).unwrap();
        let mut m: serde_json::Value =
            serde_json::from_slice(&LvfReader::open(&path).unwrap().meta_bytes().unwrap()).unwrap();
        f(&mut m);
        assert!(rewrite_meta_in_place(&path, &serde_json::to_vec(&m).unwrap()).unwrap());
        validate(&path)
    };
    let rep = edit("big_frame_count.lvd", &|m| m["frame_count"] = serde_json::json!(4294967416u64));
    assert!(!rep.ok() && rep.fatal, "a frame count past u32 must not validate");
    let rep = edit("big_fps.lvd", &|m| m["fps"]["num"] = serde_json::json!(5000000000u64));
    assert!(!rep.ok() && rep.fatal);
    let rep = edit("big_end.lvd", &|m| m["layers"][0]["end_frame"] = serde_json::json!(1u64 << 40));
    assert!(!rep.ok());
    // whatever the readers' schema rejects is invalid, so `check` and `render` agree
    let cases: [(&str, &dyn Fn(&mut serde_json::Value)); 4] = [
        ("generator.lvd", &|m| m["generator"] = serde_json::json!(5)),
        ("pre_skip.lvd", &|m| m["audio"]["pre_skip"] = serde_json::json!(-1)),
        ("canvas.lvd", &|m| m["canvas"]["width"] = serde_json::json!(4294967616u64)),
        ("rect.lvd", &|m| m["layers"][0]["rect"]["x"] = serde_json::json!(1u64 << 63)),
    ];
    for (name, f) in cases {
        let rep = edit(name, f);
        assert!(!rep.ok() && rep.fatal, "{name}");
        assert!(LvfReader::open(tmp(name)).unwrap().meta().is_err(), "{name}");
    }
}

#[test]
fn metadata_edits_read_and_write_through_one_handle() {
    let path = build("edit_with.lvd", |_| {}, |_| {});
    let edited = lvf::rewrite_meta_with(&path, |current| {
        let mut m: serde_json::Value = serde_json::from_slice(current).unwrap();
        m["layers"][0]["name"] = serde_json::json!("renamed");
        Ok(serde_json::to_vec(&m).unwrap())
    })
    .unwrap();
    assert!(edited && validate(&path).ok());
    assert_eq!(LvfReader::open(&path).unwrap().meta().unwrap().layers[0].name, "renamed");
    // an error from the edit leaves the file as it was
    let before = std::fs::read(&path).unwrap();
    assert!(lvf::rewrite_meta_with(&path, |_| Err(Error::Value("no".into()))).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
}
