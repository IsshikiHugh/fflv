//! 8-bit images (rows top to bottom, channels interleaved) and PNG encoding/decoding.

use crate::error::{Error, Result};

/// An owned 8-bit image with 1 (gray), 3 (RGB) or 4 (RGBA) channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub data: Vec<u8>,
}

/// A borrowed 8-bit image with 1 (gray), 3 (RGB) or 4 (RGBA) channels, tightly packed.
#[derive(Clone, Copy, Debug)]
pub struct ImageRef<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub data: &'a [u8],
}

fn check(width: u32, height: u32, channels: u8, len: usize) -> Result<()> {
    if !matches!(channels, 1 | 3 | 4) {
        return Err(Error::Meta(format!("images must have 1, 3 or 4 channels, got {channels}")));
    }
    if width == 0 || height == 0 {
        return Err(Error::Meta(format!("image size {width}x{height} is empty")));
    }
    let want = width as usize * height as usize * channels as usize;
    if len != want {
        return Err(Error::Meta(format!("{width}x{height}x{channels} image needs {want} bytes, got {len}")));
    }
    Ok(())
}

impl Image {
    pub fn new(width: u32, height: u32, channels: u8, data: Vec<u8>) -> Result<Image> {
        check(width, height, channels, data.len())?;
        Ok(Image { width, height, channels, data })
    }

    pub fn filled(width: u32, height: u32, pixel: &[u8]) -> Image {
        let data = pixel.repeat(width as usize * height as usize);
        Image { width, height, channels: pixel.len() as u8, data }
    }

    pub fn view(&self) -> ImageRef<'_> {
        ImageRef { width: self.width, height: self.height, channels: self.channels, data: &self.data }
    }

    /// The same pixels as RGBA (gray → R = G = B, missing alpha → 255).
    pub fn to_rgba(&self) -> Image {
        self.view().to_rgba()
    }

    /// The same pixels as RGB (alpha dropped).
    pub fn to_rgb(&self) -> Image {
        self.view().to_rgb()
    }
}

impl<'a> ImageRef<'a> {
    pub fn new(width: u32, height: u32, channels: u8, data: &'a [u8]) -> Result<ImageRef<'a>> {
        check(width, height, channels, data.len())?;
        Ok(ImageRef { width, height, channels, data })
    }

    #[inline]
    pub fn rgba_at(&self, x: u32, y: u32) -> [u8; 4] {
        let c = self.channels as usize;
        let i = (y as usize * self.width as usize + x as usize) * c;
        let p = &self.data[i..i + c];
        match c {
            1 => [p[0], p[0], p[0], 255],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        }
    }

    /// Row `y` (clamped to the last row): `width * channels` bytes.
    #[inline]
    pub fn row(&self, y: u32) -> &'a [u8] {
        let data: &'a [u8] = self.data;
        let len = self.width as usize * self.channels as usize;
        &data[y.min(self.height - 1) as usize * len..][..len]
    }

    pub fn has_alpha(&self) -> bool {
        self.channels == 4
    }

    pub fn to_rgba(&self) -> Image {
        if self.channels == 4 {
            return Image { width: self.width, height: self.height, channels: 4, data: self.data.to_vec() };
        }
        let mut data = vec![255u8; self.width as usize * self.height as usize * 4];
        let out = data.chunks_exact_mut(4);
        if self.channels == 1 {
            for (o, &v) in out.zip(self.data) {
                o[..3].fill(v);
            }
        } else {
            for (o, p) in out.zip(self.data.chunks_exact(3)) {
                o[..3].copy_from_slice(p);
            }
        }
        Image { width: self.width, height: self.height, channels: 4, data }
    }

    pub fn to_rgb(&self) -> Image {
        if self.channels == 3 {
            return Image { width: self.width, height: self.height, channels: 3, data: self.data.to_vec() };
        }
        let mut data = vec![0u8; self.width as usize * self.height as usize * 3];
        let out = data.chunks_exact_mut(3);
        if self.channels == 1 {
            for (o, &v) in out.zip(self.data) {
                o.fill(v);
            }
        } else {
            for (o, p) in out.zip(self.data.chunks_exact(4)) {
                o.copy_from_slice(&p[..3]);
            }
        }
        Image { width: self.width, height: self.height, channels: 3, data }
    }
}

// ------------------------------------------------------------------------------------------------
// PNG
// ------------------------------------------------------------------------------------------------
pub const PNG_SIGNATURE: &[u8; 8] = lvf::constants::PNG_SIGNATURE;

/// (width, height) from a PNG's IHDR chunk.
pub fn png_size(png: &[u8]) -> Result<(u32, u32)> {
    if png.len() < 24 || &png[..8] != PNG_SIGNATURE || &png[12..16] != b"IHDR" {
        return Err(Error::Media("not a PNG image".into()));
    }
    let be = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    Ok((be(&png[16..20]), be(&png[20..24])))
}

/// Encode an image as PNG (`fast`: quick compression for bulk output, else default).
pub fn encode_png(img: ImageRef, fast: bool) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, img.width, img.height);
        enc.set_color(match img.channels {
            1 => png::ColorType::Grayscale,
            3 => png::ColorType::Rgb,
            _ => png::ColorType::Rgba,
        });
        enc.set_depth(png::BitDepth::Eight);
        if fast {
            enc.set_compression(png::Compression::Fast);
        }
        let mut w = enc.write_header().map_err(|e| Error::Output(format!("PNG encoding failed: {e}")))?;
        w.write_image_data(img.data).map_err(|e| Error::Output(format!("PNG encoding failed: {e}")))?;
    }
    Ok(out)
}

/// Decode a PNG of any color type / bit depth into 8-bit RGBA.
pub fn decode_png(data: &[u8]) -> Result<Image> {
    let fail = |e: png::DecodingError| Error::Decode(format!("PNG decoding failed: {e}"));
    let mut dec = png::Decoder::new(std::io::Cursor::new(data));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(fail)?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(fail)?;
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width, info.height);
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => Image { width: w, height: h, channels: 3, data: buf }.to_rgba().data,
        png::ColorType::Grayscale => Image { width: w, height: h, channels: 1, data: buf }.to_rgba().data,
        png::ColorType::GrayscaleAlpha => buf.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Indexed => return Err(Error::Decode("PNG palette was not expanded".into())),
    };
    Image::new(w, h, 4, rgba)
}
