//! `fflv._fflv`: the Python extension module. Thin bindings of the `fflv` crate; the Pythonic API
//! (keyword arguments, dataclasses, numpy dtype handling) is in python/fflv/.
//!
//! The GIL is released while encoding, decoding and compositing; images cross the boundary as
//! uint8 numpy arrays, copied once on the way in (while the GIL is held) and moved on the way out.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fflv::codec::Speed;
use fflv::decode::{Decoding, Frames, LayerFrames};
use fflv::edit::{self, AddOptions, Source};
use fflv::image::{Image, ImageRef};
use fflv::project::{pack_with_progress, PackOptions};
use fflv::render::{self, RenderOptions};
use fflv::{Error, LayerOptions, StillOptions, WriterOptions};
use lvf::{Fps, Rect, Report};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

create_exception!(fflv, MetaError, PyValueError, "Bad layer id, rect, z, option ... (bad input, not a bad file).");
create_exception!(fflv, FormatError, PyValueError, "The file is not a well-formed .lvd.");
create_exception!(fflv, OutputError, PyValueError, "A render/extract output that cannot be written.");
create_exception!(fflv, EncodeError, PyRuntimeError, "The VP9 encoder failed or broke the format's rules.");
create_exception!(fflv, DecodeError, PyRuntimeError, "Decoding failed.");
create_exception!(fflv, MediaError, PyRuntimeError, "FFmpeg / ffprobe failed, or a source file is unusable.");
create_exception!(fflv, WriterError, PyRuntimeError, "fflv.Writer cannot continue.");
create_exception!(fflv, EditError, PyRuntimeError, "An edit could not be applied.");
create_exception!(fflv, PackError, PyRuntimeError, "fflv.pack failed.");
create_exception!(fflv, ViewError, PyRuntimeError, "The viewer could not start.");
create_exception!(fflv, InvalidOutput, PyRuntimeError, "The file just written failed validation (nothing replaced).");

thread_local! {
    /// A Python exception raised inside a callback (image source, progress, log), re-raised as
    /// it is once the Rust call returns.
    static CALLBACK_ERROR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

const CALLBACK_FAILED: &str = "a Python callback raised an exception";

fn callback_failed(e: PyErr) -> Error {
    CALLBACK_ERROR.with(|c| *c.borrow_mut() = Some(e));
    Error::Edit(CALLBACK_FAILED.into())
}

fn py_err(e: Error) -> PyErr {
    if matches!(&e, Error::Edit(m) if m == CALLBACK_FAILED) {
        if let Some(cb) = CALLBACK_ERROR.with(|c| c.borrow_mut().take()) {
            return cb;
        }
    }
    let msg = e.to_string();
    match e {
        Error::Io(io) => io.into(),
        Error::Format(_) => FormatError::new_err(msg),
        Error::Meta(_) => MetaError::new_err(msg),
        Error::Encode(_) => EncodeError::new_err(msg),
        Error::Decode(_) => DecodeError::new_err(msg),
        Error::Media(_) => MediaError::new_err(msg),
        Error::Writer(_) => WriterError::new_err(msg),
        Error::Edit(_) => EditError::new_err(msg),
        Error::Pack(_) => PackError::new_err(msg),
        Error::Output(_) => OutputError::new_err(msg),
        Error::View(_) => ViewError::new_err(msg),
        Error::Invalid { .. } => InvalidOutput::new_err(msg),
    }
}

fn clear_callback_error() {
    CALLBACK_ERROR.with(|c| c.borrow_mut().take());
}

fn report_json(r: Option<&Report>) -> Option<String> {
    r.map(|r| serde_json::to_string(r).expect("reports serialize"))
}

fn speed(s: &str) -> PyResult<Speed> {
    Speed::parse(s).map_err(py_err)
}

fn rect(r: Option<(i64, i64, i64, i64)>) -> PyResult<Option<Rect>> {
    r.map(|(x, y, w, h)| lvf::meta::make_rect(x, y, w, h).map_err(|e| MetaError::new_err(e.0))).transpose()
}

/// A borrowed view of a uint8 array shaped H×W, H×W×1, H×W×3 or H×W×4 (C-contiguous).
fn image_ref<'a>(a: &'a PyReadonlyArrayDyn<'_, u8>, what: &str) -> PyResult<ImageRef<'a>> {
    let shape = a.shape();
    let (h, w, c) = match *shape {
        [h, w] => (h, w, 1),
        [h, w, c] if matches!(c, 1 | 3 | 4) => (h, w, c),
        _ => return Err(PyValueError::new_err(format!("{what}: expected H×W, H×W×3 or H×W×4, got shape {shape:?}"))),
    };
    let data = a.as_slice().map_err(|_| PyValueError::new_err(format!("{what}: the array must be C-contiguous")))?;
    ImageRef::new(w as u32, h as u32, c as u8, data).map_err(py_err)
}

fn owned_image(a: &PyReadonlyArrayDyn<'_, u8>, what: &str) -> PyResult<Image> {
    let r = image_ref(a, what)?;
    Ok(Image { width: r.width, height: r.height, channels: r.channels, data: r.data.to_vec() })
}

fn to_array<'py>(py: Python<'py>, img: Image) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let (h, w, c) = (img.height as usize, img.width as usize, img.channels as usize);
    let arr = if c == 1 {
        Array2::from_shape_vec((h, w), img.data).map(|a| a.into_dyn())
    } else {
        Array3::from_shape_vec((h, w, c), img.data).map(|a| a.into_dyn())
    };
    Ok(arr.map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py))
}

fn fps(num: u64, den: u64) -> PyResult<Fps> {
    Fps::new(num, den).map_err(MetaError::new_err)
}

// ------------------------------------------------------------------------------------------------
// Writer
// ------------------------------------------------------------------------------------------------
#[pyclass(name = "_Writer", module = "fflv._fflv")]
struct Writer {
    inner: fflv::Writer,
}

#[pymethods]
impl Writer {
    #[new]
    #[pyo3(signature = (path, width, height, fps_num, fps_den, gop, background, crf, speed_name, check, threads))]
    fn new(
        path: PathBuf,
        width: u32,
        height: u32,
        fps_num: u64,
        fps_den: u64,
        gop: Option<u32>,
        background: String,
        crf: u32,
        speed_name: &str,
        check: bool,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let opts = WriterOptions { gop, background, crf, speed: speed(speed_name)?, check, threads };
        let inner = fflv::Writer::create(path, width, height, fps(fps_num, fps_den)?, opts).map_err(py_err)?;
        Ok(Writer { inner })
    }

    #[pyo3(signature = (id, alpha, lossless, rect_xywh, z, name, blend, opacity, visible, crf, speed_name))]
    fn add_layer(
        &mut self,
        id: &str,
        alpha: bool,
        lossless: bool,
        rect_xywh: Option<(i64, i64, i64, i64)>,
        z: Option<f64>,
        name: Option<String>,
        blend: String,
        opacity: f64,
        visible: bool,
        crf: Option<u32>,
        speed_name: Option<&str>,
    ) -> PyResult<()> {
        let o = LayerOptions {
            alpha,
            lossless,
            rect: rect(rect_xywh)?,
            z,
            name,
            blend,
            opacity,
            visible,
            crf,
            speed: speed_name.map(speed).transpose()?,
            ..Default::default()
        };
        self.inner.add_layer(id, o).map_err(py_err)
    }

    #[pyo3(signature = (id, png, rect_xywh, start, end, z, name, blend, opacity, visible))]
    fn add_still(
        &mut self,
        id: &str,
        png: &[u8],
        rect_xywh: Option<(i64, i64, i64, i64)>,
        start: u32,
        end: Option<u32>,
        z: Option<f64>,
        name: Option<String>,
        blend: String,
        opacity: f64,
        visible: bool,
    ) -> PyResult<()> {
        let o = StillOptions { rect: rect(rect_xywh)?, start, end, z, name, blend, opacity, visible };
        self.inner.add_still(id, png.to_vec(), o).map_err(py_err)
    }

    fn set_audio(&mut self, py: Python<'_>, src: PathBuf, bitrate: &str, channels: u32) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.set_audio(&src, bitrate, channels)).map_err(py_err)
    }

    /// `images`: [(layer id, uint8 array)]
    fn write(&mut self, py: Python<'_>, images: Vec<(String, PyReadonlyArrayDyn<'_, u8>)>) -> PyResult<u32> {
        // Copied while the GIL is held: other Python threads may change the arrays once it is
        // released (one memcpy, small next to encoding).
        let owned = images.iter().map(|(id, a)| owned_image(a, id)).collect::<PyResult<Vec<_>>>()?;
        let refs: Vec<(&str, ImageRef)> =
            images.iter().zip(&owned).map(|((id, _), img)| (id.as_str(), img.view())).collect();
        let inner = &mut self.inner;
        py.detach(|| inner.write(&refs)).map_err(py_err)
    }

    fn end_layer(&mut self, id: &str) -> PyResult<()> {
        self.inner.end_layer(id).map_err(py_err)
    }

    /// Validation report as JSON (None with check off).
    fn close(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        let inner = &mut self.inner;
        let rep = py.detach(|| inner.close().map(|r| r.cloned())).map_err(py_err)?;
        Ok(report_json(rep.as_ref()))
    }

    fn abort(&mut self) {
        self.inner.abort();
    }

    #[getter]
    fn frame_count(&self) -> u32 {
        self.inner.frame_count()
    }

    #[getter]
    fn layer_ids(&self) -> Vec<String> {
        self.inner.layer_ids()
    }

    #[getter]
    fn gop(&self) -> u32 {
        self.inner.gop()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }
}

// ------------------------------------------------------------------------------------------------
// Reader and its iterators
// ------------------------------------------------------------------------------------------------
#[pyclass(name = "_Reader", module = "fflv._fflv", frozen)]
struct Reader {
    /// None once closed; iterators keep their own handle to the file.
    inner: Mutex<Option<Arc<fflv::Reader>>>,
}

impl Reader {
    fn get(&self) -> PyResult<Arc<fflv::Reader>> {
        self.inner.lock().unwrap().clone().ok_or_else(|| PyValueError::new_err("I/O operation on a closed fflv.Reader"))
    }
}

#[pymethods]
impl Reader {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| fflv::Reader::open(path)).map_err(py_err)?;
        Ok(Reader { inner: Mutex::new(Some(Arc::new(inner))) })
    }

    fn close(&self) {
        self.inner.lock().unwrap().take();
    }

    fn meta_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self.get()?.meta()).expect("metadata serializes"))
    }

    fn raps(&self) -> PyResult<Vec<u32>> {
        Ok(self.get()?.raps().to_vec())
    }

    fn rap_at_or_before(&self, frame: u32) -> PyResult<u32> {
        Ok(self.get()?.rap_at_or_before(frame))
    }

    fn check(&self, py: Python<'_>) -> PyResult<String> {
        let r = self.get()?;
        let rep = py.detach(|| r.check());
        Ok(report_json(Some(&rep)).unwrap())
    }

    #[pyo3(signature = (start, end, layers, hide, transparent))]
    fn frames(
        &self,
        start: u32,
        end: Option<u32>,
        layers: Option<Vec<String>>,
        hide: Vec<String>,
        transparent: bool,
    ) -> PyResult<FrameIter> {
        let it = self.get()?.frames(start, end, layers.as_deref(), &hide, transparent).map_err(py_err)?;
        Ok(FrameIter { inner: it })
    }

    fn decode(&self, start: u32, end: Option<u32>, layers: Vec<usize>) -> PyResult<DecodeIter> {
        let r = self.get()?;
        if let Some(&bad) = layers.iter().find(|&&i| i >= r.layers().len()) {
            return Err(MetaError::new_err(format!("layer index {bad} out of range")));
        }
        Ok(DecodeIter { inner: r.decode(start, end, &layers).map_err(py_err)? })
    }

    fn layer_frames(&self, key: &str, start: Option<u32>, end: Option<u32>) -> PyResult<LayerFrameIter> {
        Ok(LayerFrameIter { inner: self.get()?.layer_frames(key, start, end).map_err(py_err)? })
    }

    /// A still layer's image, RGBA.
    fn still<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        let r = self.get()?;
        if index >= r.layers().len() {
            return Err(MetaError::new_err(format!("layer index {index} out of range")));
        }
        let img = py.detach(|| r.still(index).map(|img| (*img).clone())).map_err(py_err)?;
        to_array(py, img)
    }
}

#[pyclass(module = "fflv._fflv")]
struct FrameIter {
    inner: Frames,
}

#[pymethods]
impl FrameIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<(u32, Bound<'py, PyArrayDyn<u8>>)>> {
        let inner = &mut self.inner;
        match py.detach(|| inner.next()) {
            None => Ok(None),
            Some(Err(e)) => Err(py_err(e)),
            Some(Ok((f, img))) => Ok(Some((f, to_array(py, img)?))),
        }
    }
}

#[pyclass(module = "fflv._fflv")]
struct DecodeIter {
    inner: Decoding,
}

#[pymethods]
impl DecodeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// (frame, {layer index: (rgb H×W×3, alpha H×W or None)})
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<(u32, Bound<'py, PyDict>)>> {
        let inner = &mut self.inner;
        let (f, planes) = match py.detach(|| inner.next()) {
            None => return Ok(None),
            Some(Err(e)) => return Err(py_err(e)),
            Some(Ok(x)) => x,
        };
        let out = PyDict::new(py);
        for (i, lf) in planes {
            let (w, h) = (lf.width, lf.height);
            let rgb = to_array(py, Image { width: w, height: h, channels: 3, data: lf.rgb })?;
            let alpha = match lf.alpha {
                Some(a) => Some(to_array(py, Image { width: w, height: h, channels: 1, data: a })?),
                None => None,
            };
            out.set_item(i, (rgb, alpha))?;
        }
        Ok(Some((f, out)))
    }
}

#[pyclass(module = "fflv._fflv")]
struct LayerFrameIter {
    inner: LayerFrames,
}

#[pymethods]
impl LayerFrameIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<(u32, Bound<'py, PyArrayDyn<u8>>)>> {
        let inner = &mut self.inner;
        match py.detach(|| inner.next()) {
            None => Ok(None),
            Some(Err(e)) => Err(py_err(e)),
            Some(Ok((f, img))) => Ok(Some((f, to_array(py, img)?))),
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Edits, pack, render
// ------------------------------------------------------------------------------------------------
/// Run an edit (without the GIL) so that Ctrl+C stops it: the rewrite checks for signals every
/// ~100 ms (see `edit::with_interrupt_check`) and the KeyboardInterrupt comes through as it is.
fn interruptible<R>(f: impl FnOnce() -> R) -> R {
    clear_callback_error();
    edit::with_interrupt_check(|| Python::attach(|py| py.check_signals().map_err(callback_failed)), f)
}

#[allow(clippy::too_many_arguments)]
fn add_options(
    output: Option<PathBuf>,
    start: u32,
    end: Option<u32>,
    alpha: Option<bool>,
    lossless: bool,
    rect_xywh: Option<(i64, i64, i64, i64)>,
    z: Option<f64>,
    name: Option<String>,
    blend: String,
    opacity: f64,
    visible: bool,
    crf: u32,
    speed_name: &str,
    check: bool,
) -> PyResult<AddOptions> {
    Ok(AddOptions {
        output,
        start,
        end,
        alpha,
        lossless,
        rect: rect(rect_xywh)?,
        z,
        name,
        blend,
        opacity,
        visible,
        crf,
        speed: speed(speed_name)?,
        check,
    })
}

/// New video layer from a media file (`source` a path) or from a Python iterator of uint8 arrays.
#[pyfunction]
#[pyo3(signature = (path, id, source, images, length, output, start, end, alpha, lossless, rect_xywh, z, name, blend,
                    opacity, visible, crf, speed_name, check))]
fn add_layer(
    py: Python<'_>,
    path: PathBuf,
    id: &str,
    source: Option<PathBuf>,
    images: Option<Py<PyAny>>,
    length: Option<u32>,
    output: Option<PathBuf>,
    start: u32,
    end: Option<u32>,
    alpha: Option<bool>,
    lossless: bool,
    rect_xywh: Option<(i64, i64, i64, i64)>,
    z: Option<f64>,
    name: Option<String>,
    blend: String,
    opacity: f64,
    visible: bool,
    crf: u32,
    speed_name: &str,
    check: bool,
) -> PyResult<Option<String>> {
    let o = add_options(
        output, start, end, alpha, lossless, rect_xywh, z, name, blend, opacity, visible, crf, speed_name, check,
    )?;
    clear_callback_error();
    let rep = match (source, images) {
        (Some(src), None) => py.detach(|| interruptible(|| edit::add_layer(&path, id, Source::Media(&src), &o))),
        (None, Some(iter)) => {
            let next = move || -> Option<fflv::Result<Image>> {
                Python::attach(|py| {
                    if let Err(e) = py.check_signals() {
                        return Some(Err(callback_failed(e)));
                    }
                    let it = iter.bind(py);
                    let item = match it.call_method0("__next__") {
                        Ok(item) => item,
                        Err(e) if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) => return None,
                        Err(e) => return Some(Err(callback_failed(e))),
                    };
                    let arr = match item.extract::<PyReadonlyArrayDyn<'_, u8>>() {
                        Ok(a) => a,
                        Err(e) => return Some(Err(callback_failed(e.into()))),
                    };
                    Some(owned_image(&arr, "image").map_err(callback_failed))
                })
            };
            let images = Box::new(std::iter::from_fn(next));
            py.detach(|| interruptible(|| edit::add_layer(&path, id, Source::Images { images, len: length }, &o)))
        }
        _ => return Err(PyValueError::new_err("give a media file or an image iterator")),
    };
    Ok(report_json(rep.map_err(py_err)?.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (path, id, png, output, rect_xywh, start, end, z, name, blend, opacity, visible, check))]
fn add_still(
    py: Python<'_>,
    path: PathBuf,
    id: &str,
    png: &[u8],
    output: Option<PathBuf>,
    rect_xywh: Option<(i64, i64, i64, i64)>,
    start: u32,
    end: Option<u32>,
    z: Option<f64>,
    name: Option<String>,
    blend: String,
    opacity: f64,
    visible: bool,
    check: bool,
) -> PyResult<Option<String>> {
    let o = add_options(
        output, start, end, None, false, rect_xywh, z, name, blend, opacity, visible, 32, "balanced", check,
    )?;
    let rep = py.detach(|| interruptible(|| edit::add_still(&path, id, png.to_vec(), &o))).map_err(py_err)?;
    Ok(report_json(rep.as_ref()))
}

#[pyfunction]
fn remove_layers(
    py: Python<'_>,
    path: PathBuf,
    keys: Vec<String>,
    output: Option<PathBuf>,
    check: bool,
) -> PyResult<Option<String>> {
    let rep =
        py.detach(|| interruptible(|| edit::remove_layers(&path, &keys, output.as_deref(), check))).map_err(py_err)?;
    Ok(report_json(rep.as_ref()))
}

#[pyfunction]
fn set_audio(
    py: Python<'_>,
    path: PathBuf,
    source: Option<PathBuf>,
    output: Option<PathBuf>,
    bitrate: &str,
    channels: u32,
    check: bool,
) -> PyResult<Option<String>> {
    let rep = py
        .detach(|| {
            interruptible(|| edit::set_audio(&path, source.as_deref(), output.as_deref(), bitrate, channels, check))
        })
        .map_err(py_err)?;
    Ok(report_json(rep.as_ref()))
}

/// `fields_json`: a JSON array of [field, value] pairs.
#[pyfunction]
fn set_layer(py: Python<'_>, path: PathBuf, key: &str, fields_json: &str, output: Option<PathBuf>) -> PyResult<bool> {
    let pairs: Vec<(String, serde_json::Value)> =
        serde_json::from_str(fields_json).map_err(|e| MetaError::new_err(format!("bad field values: {e}")))?;
    py.detach(|| interruptible(|| edit::set_layer(&path, key, &pairs, output.as_deref()))).map_err(py_err)
}

#[pyfunction]
fn validate(py: Python<'_>, path: PathBuf) -> String {
    let rep = py.detach(|| lvf::validate(&path));
    report_json(Some(&rep)).unwrap()
}

/// With the GIL: raise a pending Ctrl+C, then call `f(*args)` (if given). Either exception stops
/// the Rust work (the sentinel error) and is re-raised as it is.
fn callback<A: for<'py> pyo3::call::PyCallArgs<'py>>(f: &Option<Py<PyAny>>, args: A) -> fflv::Result<()> {
    Python::attach(|py| {
        py.check_signals().map_err(callback_failed)?;
        if let Some(f) = f {
            f.bind(py).call1(args).map_err(callback_failed)?;
        }
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (project, output, threads, log, progress))]
fn pack(
    py: Python<'_>,
    project: PathBuf,
    output: Option<PathBuf>,
    threads: Option<usize>,
    log: Option<Py<PyAny>>,
    progress: Option<Py<PyAny>>,
) -> PyResult<String> {
    clear_callback_error();
    let o = PackOptions { output, threads };
    let mut p = progress_fn(progress);
    let rep = py.detach(|| {
        // a log line cannot stop the pack by itself; its exception stops it at the next frame
        let log_error: RefCell<Option<PyErr>> = RefCell::new(None);
        let mut log_fn = |line: &str| {
            if log_error.borrow().is_none() {
                if let Some(f) = &log {
                    *log_error.borrow_mut() = Python::attach(|py| f.bind(py).call1((line.to_string(),)).err());
                }
            }
        };
        let mut progress = |d, t| match log_error.borrow_mut().take() {
            Some(e) => Err(callback_failed(e)),
            None => p(d, t),
        };
        let rep = pack_with_progress(&project, &o, &mut log_fn, Some(&mut progress))?;
        let pending = log_error.borrow_mut().take();
        match pending {
            Some(e) => Err(callback_failed(e)),
            None => Ok(rep),
        }
    });
    Ok(report_json(Some(&rep.map_err(py_err)?)).unwrap())
}

fn progress_fn(progress: Option<Py<PyAny>>) -> impl FnMut(u32, u32) -> fflv::Result<()> {
    move |done, total| callback(&progress, (done, total))
}

#[pyfunction]
#[pyo3(signature = (path, output, layers, hide, start, end, transparent, crf, progress))]
fn render_file(
    py: Python<'_>,
    path: PathBuf,
    output: String,
    layers: Option<Vec<String>>,
    hide: Vec<String>,
    start: u32,
    end: Option<u32>,
    transparent: bool,
    crf: u32,
    progress: Option<Py<PyAny>>,
) -> PyResult<u32> {
    clear_callback_error();
    let o = RenderOptions { layers, hide, start, end, transparent, crf, ..Default::default() };
    let mut p = progress_fn(progress);
    py.detach(|| render::render(&path, &output, &o, Some(&mut p))).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (path, layer, output, start, end, crf, progress))]
fn extract_layer(
    py: Python<'_>,
    path: PathBuf,
    layer: &str,
    output: String,
    start: Option<u32>,
    end: Option<u32>,
    crf: u32,
    progress: Option<Py<PyAny>>,
) -> PyResult<u32> {
    clear_callback_error();
    let mut p = progress_fn(progress);
    py.detach(|| render::extract(&path, layer, &output, start, end, crf, Some(&mut p))).map_err(py_err)
}

// ------------------------------------------------------------------------------------------------
// Images, command line
// ------------------------------------------------------------------------------------------------
#[pyfunction]
fn encode_png<'py>(py: Python<'py>, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<Bound<'py, PyBytes>> {
    let img = owned_image(&image, "still image")?;
    let png = py.detach(|| fflv::image::encode_png(img.view(), false)).map_err(py_err)?;
    Ok(PyBytes::new(py, &png))
}

#[pyfunction]
fn decode_png<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let img = py.detach(|| fflv::image::decode_png(data)).map_err(py_err)?;
    to_array(py, img)
}

/// PNG bytes of an image file (PNGs as they are, other formats converted by FFmpeg).
#[pyfunction]
fn png_from_file<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyBytes>> {
    let png = py.detach(|| fflv::media::still_png_from_file(Path::new(&path))).map_err(py_err)?;
    Ok(PyBytes::new(py, &png))
}

#[pyfunction]
fn png_size(data: &[u8]) -> PyResult<(u32, u32)> {
    fflv::image::png_size(data).map_err(py_err)
}

/// Serve `path` with the player until Ctrl+C (KeyboardInterrupt) or another exception from a
/// signal handler; `ready(url)` is called once the server is listening.
#[pyfunction]
#[pyo3(signature = (path, host, port, browser, open_page, ready))]
fn view(
    py: Python<'_>,
    path: PathBuf,
    host: &str,
    port: u16,
    browser: Option<String>,
    open_page: bool,
    ready: Option<Py<PyAny>>,
) -> PyResult<()> {
    let server = py.detach(|| fflv::view::ViewServer::bind(&path, host, port, true)).map_err(py_err)?;
    let url = server.url();
    let running = server.start(8);
    let result = (|| -> PyResult<()> {
        if let Some(f) = &ready {
            f.bind(py).call1((url.clone(),))?;
        }
        if open_page {
            let u = url.clone();
            py.detach(|| fflv::view::open_browser(&u, browser.as_deref()));
        }
        loop {
            py.check_signals()?;
            py.detach(|| std::thread::sleep(Duration::from_millis(100)));
        }
    })();
    py.detach(|| running.stop());
    result
}

/// Run the `fflv` command line; returns the exit code.
#[pyfunction]
fn cli_main(py: Python<'_>, argv: Vec<String>) -> i32 {
    py.detach(|| fflv::cli::main_with_args(argv))
}

#[pymodule]
mod _fflv {
    #[pymodule_export]
    use super::{
        add_layer, add_still, cli_main, decode_png, encode_png, extract_layer, pack, png_from_file, png_size,
        remove_layers, render_file, set_audio, set_layer, validate, view, DecodeError, DecodeIter, EditError,
        EncodeError, FormatError, FrameIter, InvalidOutput, LayerFrameIter, MediaError, MetaError, OutputError,
        PackError, Reader, ViewError, Writer, WriterError,
    };

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        use pyo3::types::PyModuleMethods;
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
