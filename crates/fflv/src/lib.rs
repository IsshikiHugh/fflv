//! fflv: pack, edit, decode, render and view LVF layered video files (.lvd).
//!
//! The container itself (structures, reader/writer, validator) lives in the `lvf` crate; this
//! crate adds the codecs (libvpx VP9, FFmpeg for foreign media), the streaming [`Writer`],
//! selective decoding and compositing ([`Reader`]), the edit operations and the command line.

pub mod audio;
pub mod cli;
pub mod codec;
pub mod composite;
pub mod decode;
pub mod devtools;
pub mod edit;
pub mod encode;
pub mod error;
pub mod image;
pub mod inspect;
pub mod media;
pub mod pixel;
pub mod project;
pub mod render;
pub mod view;
pub mod writer;

pub use decode::Reader;
pub use error::{Error, Result};
pub use image::{Image, ImageRef};
pub use writer::{LayerOptions, StillOptions, Writer, WriterOptions};
