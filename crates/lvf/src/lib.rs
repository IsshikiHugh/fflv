//! The LVF v1 layered video container (`.lvd` files, LVF_SPEC.md): binary structures, streaming
//! writer, random-access reader, copy-on-write metadata edits, validator.
//!
//! Synchronization is a structural property of the format: the unit of storage is the composite
//! frame (one entry per video layer for one frame instant, plus that frame's audio).

pub mod binary;
pub mod constants;
pub mod container;
pub mod error;
pub mod meta;
pub mod output;
pub mod timing;
pub mod validate;
pub mod vp9;

pub use binary::{AudioPacket, Cau, FileHeader, IndexEntry, VideoEntry};
pub use container::{
    encode_meta, meta_capacity_for, rewrite_meta_in_place, rewrite_meta_with, temp_path_for, LvfReader, LvfWriter,
};
pub use error::{Error, Result};
pub use meta::{Kind, Layer, Meta, MetaError, Rect, Z};
pub use output::{publish, PublishError};
pub use timing::{pts_us, seconds_to_frame, Fps};
pub use validate::{validate, Issue, Report};
