//! libvpx VP9: one encoder or decoder per plane stream (the color or the alpha plane of a layer).
//!
//! Encoding guarantees what the format needs (spec 5.4, 8.2): every [`Encoder::encode`] call yields
//! exactly one packet that shows exactly one frame, and that frame is a key frame exactly when
//! asked. libvpx's own key-frame placement is pushed out of reach and alt-ref / look-ahead are
//! off; both properties are also verified from the VP9 frame header of every packet.
//!
//! Stream formats:
//!   color, lossy      profile 0, 8-bit 4:2:0, BT.709 limited range
//!   color, lossless   profile 1, 8-bit 4:4:4 RGB (planes G, B, R), lossless: exact pixel values
//!   alpha             profile 0 luma; limited range (Y = 16..235) when lossy, full range and
//!                     lossless (Y = alpha) when the layer is lossless

use std::ffi::CStr;
use std::os::raw::{c_int, c_uint, c_ulong};
use std::ptr;

use lvf::vp9::inspect_packet;
use lvf::Fps;
use vpx_sys as vpx;

use crate::error::{Error, Result};

/// libvpx key-frame interval that is never reached on its own.
const NEVER: c_uint = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speed {
    Fast,
    Balanced,
    Best,
}

impl Speed {
    pub const NAMES: [&'static str; 3] = ["fast", "balanced", "best"];

    pub fn parse(s: &str) -> Result<Speed> {
        match s {
            "fast" => Ok(Speed::Fast),
            "balanced" => Ok(Speed::Balanced),
            "best" => Ok(Speed::Best),
            _ => Err(Error::Meta(format!("speed must be one of fast, balanced, best, got {s:?}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Speed::Fast => "fast",
            Speed::Balanced => "balanced",
            Speed::Best => "best",
        }
    }

    /// (libvpx deadline, cpu-used)
    fn preset(self) -> (c_ulong, c_int) {
        match self {
            Speed::Fast => (vpx::VPX_DL_REALTIME as c_ulong, 8),
            Speed::Balanced => (vpx::VPX_DL_GOOD_QUALITY as c_ulong, 4),
            Speed::Best => (vpx::VPX_DL_GOOD_QUALITY as c_ulong, 1),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// Constant quality, 0 (best) ..= 63. Ignored by lossless streams.
    pub crf: u32,
    pub speed: Speed,
    /// libvpx threads per stream; 0 picks automatically.
    pub threads: u32,
    /// Overrides the cpu-used of the speed preset (libvpx: -8..=8, higher is faster).
    pub cpu_used: Option<i32>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions { crf: 32, speed: Speed::Balanced, threads: 0, cpu_used: None }
    }
}

impl EncodeOptions {
    pub fn new(crf: u32, speed: Speed) -> Result<EncodeOptions> {
        if crf > 63 {
            return Err(Error::Meta(format!("crf must be in 0..63, got {crf}")));
        }
        Ok(EncodeOptions { crf, speed, ..Default::default() })
    }
}

// ------------------------------------------------------------------------------------------------
// Planar pictures
// ------------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneFormat {
    I420,
    I444,
}

/// An 8-bit planar picture: the Y (or G), U (or B) and V (or R) planes stored back to back
/// without row padding. Width and height are even for I420.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planar {
    pub format: PlaneFormat,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Planar {
    pub fn new(format: PlaneFormat, width: u32, height: u32) -> Planar {
        Planar { format, width, height, data: vec![0; Planar::byte_len(format, width, height)] }
    }

    /// Bytes of a `width`×`height` picture in `format`.
    pub fn byte_len(format: PlaneFormat, width: u32, height: u32) -> usize {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = match format {
            PlaneFormat::I420 => (w.div_ceil(2), h.div_ceil(2)),
            PlaneFormat::I444 => (w, h),
        };
        w * h + 2 * cw * ch
    }

    pub fn chroma_size(&self) -> (u32, u32) {
        match self.format {
            PlaneFormat::I420 => (self.width.div_ceil(2), self.height.div_ceil(2)),
            PlaneFormat::I444 => (self.width, self.height),
        }
    }

    fn split(&self) -> (usize, usize) {
        let (cw, ch) = self.chroma_size();
        let y = self.width as usize * self.height as usize;
        (y, y + cw as usize * ch as usize)
    }

    pub fn planes(&self) -> [&[u8]; 3] {
        let (a, b) = self.split();
        let (y, rest) = self.data.split_at(a);
        let (u, v) = rest.split_at(b - a);
        [y, u, v]
    }

    pub fn planes_mut(&mut self) -> (&mut [u8], &mut [u8], &mut [u8]) {
        let (a, b) = self.split();
        let (y, rest) = self.data.split_at_mut(a);
        let (u, v) = rest.split_at_mut(b - a);
        (y, u, v)
    }
}

/// What a stream's headers say about its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// BT.709 matrix, limited range (lossy color, lossy alpha)
    Bt709Limited,
    /// BT.709 matrix, full range (lossless alpha)
    Bt709Full,
    /// RGB (lossless color, 4:4:4 only)
    Rgb,
}

impl Signal {
    /// (VPX_CS_*, VPX_CR_*)
    fn vpx(self) -> (vpx::vpx_color_space_t, vpx::vpx_color_range_t) {
        match self {
            Signal::Bt709Limited => (vpx::VPX_CS_BT_709, vpx::VPX_CR_STUDIO_RANGE),
            Signal::Bt709Full => (vpx::VPX_CS_BT_709, vpx::VPX_CR_FULL_RANGE),
            Signal::Rgb => (vpx::VPX_CS_SRGB, vpx::VPX_CR_FULL_RANGE),
        }
    }
}

fn codec_message(ctx: *const vpx::vpx_codec_ctx_t, code: vpx::vpx_codec_err_t) -> String {
    // SAFETY: libvpx returns NUL-terminated static or context-owned strings (or NULL).
    unsafe {
        let text =
            |p: *const std::os::raw::c_char| (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().into_owned());
        let mut msg = if ctx.is_null() { None } else { text(vpx::vpx_codec_error(ctx)) }
            .unwrap_or_else(|| text(vpx::vpx_codec_err_to_string(code)).unwrap_or_default());
        if !ctx.is_null() {
            if let Some(d) = text(vpx::vpx_codec_error_detail(ctx)) {
                msg = format!("{msg} ({d})");
            }
        }
        msg
    }
}

fn default_threads() -> u32 {
    std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(2).min(8)
}

/// Largest time base term libvpx accepts.
const TIMEBASE_MAX: u64 = 1_000_000_000;

/// libvpx time base (seconds per tick) for `fps`: 1/fps, with both terms at most 10^9. A rate
/// whose terms are larger (e.g. 23.976023976 = 2997002997/125000000) is approximated by halving
/// both terms: the pts advance by one tick per frame, so the time base only informs rate control.
/// None when that is off by more than 1e-6 (absurd rates such as 4294967295/1).
fn timebase(fps: Fps) -> Option<vpx::vpx_rational> {
    let (mut num, mut den) = (fps.den as u64, fps.num as u64);
    while num > TIMEBASE_MAX || den > TIMEBASE_MAX {
        num = num.div_ceil(2);
        den = den.div_ceil(2);
    }
    let error = (num as f64 / den as f64) / (fps.den as f64 / fps.num as f64) - 1.0;
    (error.abs() <= 1e-6).then_some(vpx::vpx_rational { num: num as c_int, den: den as c_int })
}

// ------------------------------------------------------------------------------------------------
// Encoder
// ------------------------------------------------------------------------------------------------
pub struct Encoder {
    ctx: Box<vpx::vpx_codec_ctx_t>,
    /// the configuration the context runs with (for [`Encoder::set_threads`])
    cfg: Box<vpx::vpx_codec_enc_cfg_t>,
    width: u32,
    height: u32,
    format: PlaneFormat,
    signal: Signal,
    deadline: c_ulong,
    pts: i64,
    what: String,
}

// SAFETY: a libvpx context has no thread affinity, and every method that touches it takes
// &mut self (the &self methods only read plain fields).
unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

impl Encoder {
    /// `what` names the stream in error messages (e.g. "layer 'mask' alpha").
    pub fn new(
        width: u32,
        height: u32,
        fps: Fps,
        format: PlaneFormat,
        signal: Signal,
        lossless: bool,
        opts: &EncodeOptions,
        what: &str,
    ) -> Result<Encoder> {
        let fail = |m: String| Error::Encode(format!("{what}: {m}"));
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(fail(format!("unsupported size {width}x{height}")));
        }
        if format == PlaneFormat::I420 && (width % 2 != 0 || height % 2 != 0) {
            return Err(fail(format!("4:2:0 needs an even size, got {width}x{height}")));
        }
        if signal == Signal::Rgb && format != PlaneFormat::I444 {
            return Err(fail("RGB streams must be 4:4:4".into()));
        }
        if opts.crf > 63 {
            return Err(fail(format!("crf must be in 0..63, got {}", opts.crf)));
        }
        let (deadline, preset_cpu) = opts.speed.preset();
        let cpu_used = opts.cpu_used.unwrap_or(preset_cpu);
        // SAFETY: plain C calls on a zero-initialised config; the context is boxed so its address
        // stays fixed for libvpx.
        unsafe {
            let iface = vpx::vpx_codec_vp9_cx();
            let mut cfg: vpx::vpx_codec_enc_cfg_t = std::mem::zeroed();
            let rc = vpx::vpx_codec_enc_config_default(iface, &mut cfg, 0);
            if rc != vpx::VPX_CODEC_OK {
                return Err(fail(format!("no default config: {}", codec_message(ptr::null(), rc))));
            }
            cfg.g_w = width;
            cfg.g_h = height;
            cfg.g_profile = if format == PlaneFormat::I444 { 1 } else { 0 };
            cfg.g_timebase = timebase(fps)
                .ok_or_else(|| fail(format!("frame rate {}/{} is out of libvpx's range", fps.num, fps.den)))?;
            cfg.g_threads = if opts.threads > 0 { opts.threads } else { default_threads() };
            cfg.g_pass = vpx::VPX_RC_ONE_PASS;
            cfg.g_lag_in_frames = 0;
            cfg.g_error_resilient = 0;
            cfg.rc_end_usage = vpx::VPX_Q;
            cfg.kf_mode = vpx::VPX_KF_AUTO;
            cfg.kf_min_dist = NEVER;
            cfg.kf_max_dist = NEVER;
            let cfg = Box::new(cfg);
            let mut ctx: Box<vpx::vpx_codec_ctx_t> = Box::new(std::mem::zeroed());
            let rc = vpx::fflv_vp9_enc_init(&mut *ctx, &*cfg, 0);
            if rc != vpx::VPX_CODEC_OK {
                return Err(fail(format!("cannot start libvpx: {}", codec_message(&*ctx, rc))));
            }
            let mut enc = Encoder { ctx, cfg, width, height, format, signal, deadline, pts: 0, what: what.into() };
            let (cs, range) = signal.vpx();
            let mut controls = vec![
                (vpx::VP8E_SET_CPUUSED, cpu_used),
                (vpx::VP8E_SET_ENABLEAUTOALTREF, 0),
                (vpx::VP9E_SET_ROW_MT, 1),
                (vpx::VP9E_SET_COLOR_SPACE, cs as c_int),
                (vpx::VP9E_SET_COLOR_RANGE, range as c_int),
            ];
            if lossless {
                controls.push((vpx::VP9E_SET_LOSSLESS, 1));
            } else {
                controls.push((vpx::VP8E_SET_CQ_LEVEL, opts.crf as c_int));
            }
            for (id, value) in controls {
                let rc = vpx::fflv_vpx_control_int(&mut *enc.ctx, id as c_int, value);
                if rc != vpx::VPX_CODEC_OK {
                    return Err(fail(format!("libvpx control {id} = {value}: {}", codec_message(&*enc.ctx, rc))));
                }
            }
            Ok(enc)
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Change the number of libvpx threads before the first frame (e.g. once the number of
    /// streams encoded in parallel is known).
    pub fn set_threads(&mut self, threads: u32) -> Result<()> {
        let threads = threads.max(1);
        if self.cfg.g_threads == threads {
            return Ok(());
        }
        if self.pts != 0 {
            return Err(Error::Encode(format!("{}: threads can only change before the first frame", self.what)));
        }
        let old = self.cfg.g_threads;
        self.cfg.g_threads = threads;
        // SAFETY: the context was initialised in new(); libvpx copies the configuration.
        let rc = unsafe { vpx::vpx_codec_enc_config_set(&mut *self.ctx, &*self.cfg) };
        if rc != vpx::VPX_CODEC_OK {
            self.cfg.g_threads = old;
            return Err(Error::Encode(format!(
                "{}: cannot use {threads} threads: {}",
                self.what,
                codec_message(&*self.ctx, rc)
            )));
        }
        Ok(())
    }

    /// Encode one picture; returns its packet.
    pub fn encode(&mut self, pic: &Planar, key: bool) -> Result<Vec<u8>> {
        if (pic.width, pic.height, pic.format) != (self.width, self.height, self.format) {
            return Err(Error::Encode(format!(
                "{}: picture is {}x{} {:?}, the stream is {}x{} {:?}",
                self.what, pic.width, pic.height, pic.format, self.width, self.height, self.format
            )));
        }
        // libvpx reads the whole picture from `data`
        let expected = Planar::byte_len(pic.format, pic.width, pic.height);
        if pic.data.len() != expected {
            return Err(Error::Encode(format!(
                "{}: picture holds {} bytes, a {}x{} {:?} picture needs {expected}",
                self.what,
                pic.data.len(),
                pic.width,
                pic.height,
                pic.format
            )));
        }
        let fmt = match self.format {
            PlaneFormat::I420 => vpx::VPX_IMG_FMT_I420,
            PlaneFormat::I444 => vpx::VPX_IMG_FMT_I444,
        };
        let packets = unsafe {
            // SAFETY: the image only borrows `pic` for the duration of vpx_codec_encode, which
            // copies the samples into its own buffers (lag 0); libvpx does not write to it.
            let mut img: vpx::vpx_image_t = std::mem::zeroed();
            let data = pic.data.as_ptr() as *mut u8;
            if vpx::vpx_img_wrap(&mut img, fmt, self.width, self.height, 1, data).is_null() {
                return Err(Error::Encode(format!("{}: vpx_img_wrap failed", self.what)));
            }
            let (cs, range) = self.signal.vpx();
            img.cs = cs;
            img.range = range;
            let flags = if key { vpx::VPX_EFLAG_FORCE_KF as vpx::vpx_enc_frame_flags_t } else { 0 };
            let rc = vpx::vpx_codec_encode(&mut *self.ctx, &img, self.pts, 1, flags, self.deadline);
            if rc != vpx::VPX_CODEC_OK {
                return Err(Error::Encode(format!(
                    "{}: encoding failed: {}",
                    self.what,
                    codec_message(&*self.ctx, rc)
                )));
            }
            self.pts += 1;
            self.drain()
        };
        if packets.len() != 1 {
            return Err(Error::Encode(format!(
                "{} encoder returned {} packets for one frame",
                self.what,
                packets.len()
            )));
        }
        let data = packets.into_iter().next().unwrap();
        let info = inspect_packet(&data)
            .map_err(|e| Error::Encode(format!("{} encoder produced an unparseable packet: {e}", self.what)))?;
        if info.shown_count() != 1 {
            return Err(Error::Encode(format!("{} packet shows {} frames", self.what, info.shown_count())));
        }
        if info.key_frame() != key {
            return Err(Error::Encode(format!(
                "{} packet is {} frame, expected {}",
                self.what,
                if info.key_frame() { "a key" } else { "an inter" },
                if key { "key" } else { "inter" }
            )));
        }
        Ok(data)
    }

    /// Collect the compressed frames waiting in the encoder.
    unsafe fn drain(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut iter: vpx::vpx_codec_iter_t = ptr::null();
        loop {
            let pkt = vpx::vpx_codec_get_cx_data(&mut *self.ctx, &mut iter);
            if pkt.is_null() {
                break;
            }
            if (*pkt).kind == vpx::VPX_CODEC_CX_FRAME_PKT {
                let f = (*pkt).data.frame;
                out.push(std::slice::from_raw_parts(f.buf as *const u8, f.sz).to_vec());
            }
        }
        out
    }

    /// End of stream: fails if the encoder was holding frames back (it must not, with lag 0).
    pub fn finish(&mut self) -> Result<()> {
        let tail = unsafe {
            let rc = vpx::vpx_codec_encode(&mut *self.ctx, ptr::null(), self.pts, 1, 0, self.deadline);
            if rc != vpx::VPX_CODEC_OK {
                return Err(Error::Encode(format!("{}: flush failed: {}", self.what, codec_message(&*self.ctx, rc))));
            }
            self.drain()
        };
        if !tail.is_empty() {
            return Err(Error::Encode(format!("{} encoder held back {} packets", self.what, tail.len())));
        }
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: the context was initialised in new() and is destroyed exactly once.
        unsafe {
            vpx::vpx_codec_destroy(&mut *self.ctx);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Decoder
// ------------------------------------------------------------------------------------------------
/// A decoded picture, borrowed from the decoder until its next call.
pub struct Frame<'a> {
    pub format: PlaneFormat,
    pub width: u32,
    pub height: u32,
    /// libvpx color space (VPX_CS_*): 1 BT.601, 2 BT.709, 3 SMPTE 170, 4 SMPTE 240, 5 BT.2020, 7 RGB
    pub color_space: u32,
    pub full_range: bool,
    pub planes: [&'a [u8]; 3],
    pub strides: [usize; 3],
}

impl Frame<'_> {
    pub fn is_rgb(&self) -> bool {
        self.color_space == vpx::VPX_CS_SRGB
    }

    pub fn chroma_shift(&self) -> (u32, u32) {
        match self.format {
            PlaneFormat::I420 => (1, 1),
            PlaneFormat::I444 => (0, 0),
        }
    }

    /// Copy into a tightly packed [`Planar`] (tests, debugging).
    pub fn to_planar(&self) -> Planar {
        let mut p = Planar::new(self.format, self.width, self.height);
        let (cw, ch) = p.chroma_size();
        let dims = [(self.width, self.height), (cw, ch), (cw, ch)];
        let (y, u, v) = p.planes_mut();
        for (k, dst) in [y, u, v].into_iter().enumerate() {
            let (w, h) = (dims[k].0 as usize, dims[k].1 as usize);
            for r in 0..h {
                let s = r * self.strides[k];
                dst[r * w..(r + 1) * w].copy_from_slice(&self.planes[k][s..s + w]);
            }
        }
        p
    }
}

pub struct Decoder {
    ctx: Box<vpx::vpx_codec_ctx_t>,
    what: String,
}

// SAFETY: see Encoder.
unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

impl Decoder {
    /// `threads`: libvpx decoding threads (1 is best when many streams are decoded in parallel).
    pub fn new(threads: u32, what: &str) -> Result<Decoder> {
        unsafe {
            let cfg = vpx::vpx_codec_dec_cfg_t { threads: threads.max(1), w: 0, h: 0 };
            let mut ctx: Box<vpx::vpx_codec_ctx_t> = Box::new(std::mem::zeroed());
            let rc = vpx::fflv_vp9_dec_init(&mut *ctx, &cfg);
            if rc != vpx::VPX_CODEC_OK {
                return Err(Error::Decode(format!("{what}: cannot start libvpx: {}", codec_message(&*ctx, rc))));
            }
            Ok(Decoder { ctx, what: what.into() })
        }
    }

    /// Decode one packet; it must produce exactly one picture.
    pub fn decode(&mut self, data: &[u8]) -> Result<Frame<'_>> {
        let what = &self.what;
        unsafe {
            let rc = vpx::vpx_codec_decode(&mut *self.ctx, data.as_ptr(), data.len() as c_uint, ptr::null_mut(), 0);
            if rc != vpx::VPX_CODEC_OK {
                return Err(Error::Decode(format!("{what}: {}", codec_message(&*self.ctx, rc))));
            }
            let mut iter: vpx::vpx_codec_iter_t = ptr::null();
            let img = vpx::vpx_codec_get_frame(&mut *self.ctx, &mut iter);
            if img.is_null() {
                return Err(Error::Decode(format!("{what}: the decoder returned no picture for a packet")));
            }
            if !vpx::vpx_codec_get_frame(&mut *self.ctx, &mut iter).is_null() {
                return Err(Error::Decode(format!("{what}: the decoder returned several pictures for one packet")));
            }
            let img = &*img;
            let format = match img.fmt {
                vpx::VPX_IMG_FMT_I420 => PlaneFormat::I420,
                vpx::VPX_IMG_FMT_I444 => PlaneFormat::I444,
                other => return Err(Error::Decode(format!("{what}: unsupported picture format {other:#x}"))),
            };
            let (w, h) = (img.d_w, img.d_h);
            let (xs, ys) = if format == PlaneFormat::I420 { (1, 1) } else { (0, 0) };
            let dims = [(w, h), ((w + xs) >> xs, (h + ys) >> ys), ((w + xs) >> xs, (h + ys) >> ys)];
            let mut planes: [&[u8]; 3] = [&[]; 3];
            let mut strides = [0usize; 3];
            for k in 0..3 {
                let stride = img.stride[k];
                if stride <= 0 || img.planes[k].is_null() {
                    return Err(Error::Decode(format!("{what}: picture plane {k} is missing")));
                }
                let (pw, ph) = (dims[k].0 as usize, dims[k].1 as usize);
                strides[k] = stride as usize;
                planes[k] = std::slice::from_raw_parts(img.planes[k], strides[k] * (ph - 1) + pw);
            }
            Ok(Frame {
                format,
                width: w,
                height: h,
                color_space: img.cs,
                full_range: img.range == vpx::VPX_CR_FULL_RANGE,
                planes,
                strides,
            })
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: initialised in new(), destroyed once.
        unsafe {
            vpx::vpx_codec_destroy(&mut *self.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timebase_fits_libvpx_limits() {
        let tb = timebase(Fps::new(30000, 1001).unwrap()).unwrap();
        assert_eq!((tb.num, tb.den), (1001, 30000));
        let fps = Fps::new(23_976_023_976, 1_000_000_000).unwrap();
        assert!(fps.num as u64 > TIMEBASE_MAX);
        let tb = timebase(fps).unwrap();
        assert!(tb.num > 0 && tb.den > 0 && tb.den as u64 <= TIMEBASE_MAX);
        let exact = fps.den as f64 / fps.num as f64;
        assert!((tb.num as f64 / tb.den as f64 / exact - 1.0).abs() < 1e-8);
        let tb = timebase(Fps { num: u32::MAX, den: u32::MAX - 2 }).unwrap();
        assert!(tb.num as u64 <= TIMEBASE_MAX && tb.den as u64 <= TIMEBASE_MAX);
        assert!(timebase(Fps { num: u32::MAX, den: 1 }).is_none());
    }
}
