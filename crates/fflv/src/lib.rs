//! fflv: pack, edit, decode, render and view LVF layered video files (.lvd).
//!
//! The container itself (structures, reader/writer, validator) lives in the `lvf` crate; this
//! crate adds the codecs (libvpx VP9, FFmpeg for foreign media), the streaming [`Writer`] and
//! selective decoding and compositing ([`Reader`]).

pub mod audio;
pub mod codec;
pub mod composite;
pub mod decode;
pub mod encode;
pub mod error;
pub mod image;
pub mod media;
pub mod pixel;
pub mod writer;

pub use decode::Reader;
pub use error::{Error, Result};
pub use image::{Image, ImageRef};
pub use writer::{LayerOptions, StillOptions, Writer, WriterOptions};
