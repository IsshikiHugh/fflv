//! Derive deliberately broken .lvd files from a valid one (acceptance test 11.2-2).
//!
//! ```text
//! fflv corrupt test_assets/test.lvd [--out test_assets/bad] [--check]
//! ```
//!
//! Each variant breaks exactly one thing (missing entry, misaligned key frames, non-contiguous
//! frame numbers, ...). `--check` runs the validator on every variant and verifies that it reports
//! the expected invariant.

use std::path::{Path, PathBuf};

use lvf::binary::pack_index;
use lvf::constants::{CAU_FLAG_RAP, ENTRY_EMPTY, MAGIC_INDEX};
use lvf::{validate, Cau, IndexEntry, LvfReader, LvfWriter, Meta, VideoEntry};

use crate::error::{Error, Result};

#[derive(Clone)]
pub struct Source {
    pub meta: Meta,
    pub meta_bytes: Vec<u8>,
    pub resources: Vec<u8>,
    pub caus: Vec<Cau>,
}

impl Source {
    pub fn load(path: &Path) -> Result<Source> {
        let r = LvfReader::open(path)?;
        let caus = r.caus(None, None).map(|c| c.map(|(_, cau, _)| cau)).collect::<lvf::Result<Vec<_>>>()?;
        Ok(Source { meta: r.meta()?, meta_bytes: r.meta_bytes()?, resources: r.resources()?, caus })
    }

    fn video_layers(&self) -> Vec<usize> {
        self.meta.video_layers()
    }

    fn gop(&self) -> usize {
        self.meta.max_rap_interval as usize
    }

    /// A video layer that does not start at frame 0 (the test material has one).
    fn late_layer(&self) -> Result<usize> {
        self.video_layers()
            .into_iter()
            .find(|&i| self.meta.layers[i].start_frame > 0)
            .ok_or_else(|| Error::Edit("source file has no video layer starting after frame 0".into()))
    }

    fn alpha_layer(&self) -> Result<usize> {
        self.video_layers()
            .into_iter()
            .find(|&i| self.meta.layers[i].has_alpha() && self.meta.layers[i].start_frame == 0)
            .ok_or_else(|| Error::Edit("source file has no alpha layer starting at frame 0".into()))
    }

    fn entry(&mut self, frame: usize, layer: usize) -> &mut VideoEntry {
        self.caus[frame].entries.iter_mut().find(|e| e.layer_index as usize == layer).expect("layer entry")
    }
}

type IndexFix = Box<dyn Fn(&mut Vec<IndexEntry>)>;

pub struct Mutation {
    pub name: &'static str,
    pub description: &'static str,
    /// Invariant codes the validator must report.
    pub expect: &'static [&'static str],
    /// Frame the primary error must point at (None: don't care).
    pub frame: Option<u32>,
    apply: fn(&mut Source) -> Result<Option<IndexFix>>,
}

fn missing_entry(s: &mut Source) -> Result<Option<IndexFix>> {
    let li = *s.video_layers().last().unwrap() as u16;
    s.caus[30].entries.retain(|e| e.layer_index != li);
    s.caus[30].video_entry_count = None;
    Ok(None)
}

fn entry_order(s: &mut Source) -> Result<Option<IndexFix>> {
    s.caus[31].entries.swap(0, 1);
    Ok(None)
}

fn active_entry_empty(s: &mut Source) -> Result<Option<IndexFix>> {
    let li = s.video_layers()[0];
    let e = s.entry(32, li);
    e.kind = ENTRY_EMPTY;
    e.frame_flags = 0;
    e.color.clear();
    e.alpha.clear();
    Ok(None)
}

fn alpha_key_mismatch(s: &mut Source) -> Result<Option<IndexFix>> {
    // At a RAP, swap the alpha plane of one layer for the next frame's (inter-coded) alpha.
    let (f, li) = (s.gop(), s.alpha_layer()?);
    let next = s.entry(f + 1, li).alpha.clone();
    s.entry(f, li).alpha = next;
    Ok(None)
}

fn layer_start_not_key(s: &mut Source) -> Result<Option<IndexFix>> {
    let li = s.late_layer()?;
    let f = s.meta.layers[li].start_frame as usize;
    let next = s.entry(f + 1, li).clone();
    let e = s.entry(f, li);
    e.color = next.color;
    e.alpha = next.alpha;
    e.frame_flags = 0;
    Ok(None)
}

fn rap_flag_cleared(s: &mut Source) -> Result<Option<IndexFix>> {
    let g = s.gop();
    s.caus[g].flags &= !CAU_FLAG_RAP;
    Ok(None)
}

fn frame0_not_rap(s: &mut Source) -> Result<Option<IndexFix>> {
    s.caus[0].flags &= !CAU_FLAG_RAP;
    Ok(None)
}

fn frame_index_gap(s: &mut Source) -> Result<Option<IndexFix>> {
    s.caus[40].frame_index = 41;
    Ok(None)
}

fn dropped_cau(s: &mut Source) -> Result<Option<IndexFix>> {
    s.caus.remove(40);
    Ok(None)
}

fn audio_wrong_frame(s: &mut Source) -> Result<Option<IndexFix>> {
    let pk = s.caus[10].audio.pop().ok_or_else(|| Error::Edit("source file has no audio in frame 10".into()))?;
    s.caus[11].audio.insert(0, pk);
    Ok(None)
}

fn index_wrong_offset(_s: &mut Source) -> Result<Option<IndexFix>> {
    Ok(Some(Box::new(|entries: &mut Vec<IndexEntry>| entries[50].cau_offset += 4)))
}

fn index_wrong_rap(s: &mut Source) -> Result<Option<IndexFix>> {
    let g = s.gop();
    Ok(Some(Box::new(move |entries: &mut Vec<IndexEntry>| entries[g].flags = 0)))
}

fn hold_entry(s: &mut Source) -> Result<Option<IndexFix>> {
    let li = s.video_layers()[0];
    s.entry(33, li).kind = 2;
    Ok(None)
}

pub const MUTATIONS: [Mutation; 13] = [
    Mutation {
        name: "missing_entry",
        description: "frame 30 lacks the entry of the last video layer",
        expect: &["I2"],
        frame: Some(30),
        apply: missing_entry,
    },
    Mutation {
        name: "entry_order",
        description: "frame 31 lists its first two video entries in the wrong order",
        expect: &["I2"],
        frame: Some(31),
        apply: entry_order,
    },
    Mutation {
        name: "active_entry_empty",
        description: "frame 32: an active layer's entry is EMPTY",
        expect: &["I3"],
        frame: Some(32),
        apply: active_entry_empty,
    },
    Mutation {
        name: "alpha_key_mismatch",
        description: "at the second RAP one layer's alpha plane is an inter frame",
        expect: &["I5", "I6"],
        frame: None,
        apply: alpha_key_mismatch,
    },
    Mutation {
        name: "layer_start_not_key",
        description: "the late-starting layer begins with an inter frame",
        expect: &["I4"],
        frame: None,
        apply: layer_start_not_key,
    },
    Mutation {
        name: "rap_flag_cleared",
        description: "the RAP flag of the second RAP is cleared",
        expect: &["I6", "I8"],
        frame: None,
        apply: rap_flag_cleared,
    },
    Mutation {
        name: "frame0_not_rap",
        description: "frame 0 is not marked RAP",
        expect: &["I6", "I7"],
        frame: Some(0),
        apply: frame0_not_rap,
    },
    Mutation {
        name: "frame_index_gap",
        description: "composite frame 40 claims to be frame 41",
        expect: &["I1"],
        frame: Some(40),
        apply: frame_index_gap,
    },
    Mutation {
        name: "dropped_cau",
        description: "composite frame 40 is missing",
        expect: &["I1"],
        frame: Some(40),
        apply: dropped_cau,
    },
    Mutation {
        name: "audio_wrong_frame",
        description: "an audio packet of frame 10 is stored in frame 11",
        expect: &["I9"],
        frame: Some(11),
        apply: audio_wrong_frame,
    },
    Mutation {
        name: "index_wrong_offset",
        description: "index entry 50 points 4 bytes too far",
        expect: &["I10"],
        frame: Some(50),
        apply: index_wrong_offset,
    },
    Mutation {
        name: "index_wrong_rap",
        description: "index entry of the second RAP lost its RAP flag",
        expect: &["I10"],
        frame: None,
        apply: index_wrong_rap,
    },
    Mutation {
        name: "hold_entry",
        description: "frame 33 uses the reserved HOLD entry type",
        expect: &["CAU"],
        frame: Some(33),
        apply: hold_entry,
    },
];

/// Write `src` with `m` applied (to a copy: `src` is loaded once and shared by every variant).
pub fn write_variant(src: &Source, m: &Mutation, out: &Path) -> Result<PathBuf> {
    let mut s = src.clone();
    let fix = (m.apply)(&mut s)?;
    let dst = out.join(format!("{}.lvd", m.name));
    let mut w = LvfWriter::create(&dst)?;
    w.begin(&s.meta_bytes, &s.resources, None)?;
    for cau in &s.caus {
        w.write_cau(cau)?;
    }
    let mut index = w.index.clone();
    if let Some(fix) = fix {
        fix(&mut index);
    }
    w.finish(Some(pack_index(&index, MAGIC_INDEX, None)), None)?;
    Ok(dst)
}

/// (as expected, the first matching issue or what went wrong)
pub fn check(path: &Path, m: &Mutation) -> (bool, String) {
    let rep = validate(path);
    let codes = rep.error_codes();
    let missing: Vec<&str> = m.expect.iter().copied().filter(|c| !codes.contains(*c)).collect();
    if !missing.is_empty() {
        return (false, format!("expected {:?}, validator reported {:?}", m.expect, codes));
    }
    let errors = rep.errors();
    if let Some(frame) = m.frame {
        let frames: Vec<u32> =
            errors.iter().filter(|i| m.expect.contains(&i.code.as_str())).filter_map(|i| i.frame).collect();
        if !frames.contains(&frame) {
            return (false, format!("expected an error at frame {frame}, got frames {frames:?}"));
        }
    }
    let first = errors.iter().find(|i| m.expect.contains(&i.code.as_str())).unwrap();
    (true, first.to_string())
}

/// Returns the number of variants that were not reported as expected (with `check`).
pub fn run(source: &Path, out: Option<&Path>, check_variants: bool) -> Result<usize> {
    let out = match out {
        Some(o) => o.to_path_buf(),
        None => std::path::absolute(source)?.parent().unwrap_or(Path::new(".")).join("bad"),
    };
    std::fs::create_dir_all(&out)?;
    if !validate(source).ok() {
        return Err(Error::Edit(format!("{} is not valid to begin with", source.display())));
    }
    let src = Source::load(source)?;
    let mut failures = 0;
    for m in &MUTATIONS {
        let path = write_variant(&src, m, &out)?;
        let mut line = format!("{:<22} {}", m.name, m.description);
        if check_variants {
            let (ok, msg) = check(&path, m);
            failures += usize::from(!ok);
            line += &format!("\n{:22} {} {msg}", "", if ok { "OK  " } else { "FAIL" });
        }
        println!("{line}");
    }
    if check_variants {
        println!("\n{}/{} broken files reported as expected", MUTATIONS.len() - failures, MUTATIONS.len());
    }
    Ok(failures)
}
