//! The metadata JSON (spec section 4 + Appendix B) as typed structures, plus the rules for building
//! and editing it. Unknown fields of the file, its layers and its audio are kept (`extra`), so
//! readers and editors preserve them (unknown fields inside canvas, fps, rect and resource are not).
//!
//! The validator works on the raw JSON value (so it can report every problem); code that has a
//! validated file uses these types.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::constants::{ALPHA_RANGES, BLEND_MODES};
use crate::timing::Fps;
use crate::vp9::codec_string;

pub const GENERATOR: &str = "fflv";
pub const EDITABLE_FIELDS: [&str; 7] = ["id", "name", "z", "rect", "blend", "opacity", "visible"];

#[derive(Debug, Clone, PartialEq)]
pub struct MetaError(pub String);

impl std::fmt::Display for MetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for MetaError {}

fn err<T>(msg: impl Into<String>) -> Result<T, MetaError> {
    Err(MetaError(msg.into()))
}

// ------------------------------------------------------------------------------------------------
// Types
// ------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub background: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct FpsJson {
    pub num: u32,
    pub den: u32,
}

impl From<FpsJson> for Fps {
    fn from(f: FpsJson) -> Fps {
        Fps::new(f.num as u64, f.den as u64).unwrap_or(Fps { num: f.num.max(1), den: f.den.max(1) })
    }
}

impl From<Fps> for FpsJson {
    fn from(f: Fps) -> FpsJson {
        FpsJson { num: f.num, den: f.den }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: u32,
    pub h: u32,
}

/// Drawing order. Written as a JSON integer when integral ("z": 3, not 3.0), else a float.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Z(pub f64);

impl Serialize for Z {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.fract() == 0.0 && self.0.abs() < 9.0e15 {
            s.serialize_i64(self.0 as i64)
        } else {
            s.serialize_f64(self.0)
        }
    }
}

impl<'de> Deserialize<'de> for Z {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Z, D::Error> {
        f64::deserialize(d).map(Z)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Video,
    Still,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Resource {
    pub offset: u64,
    pub length: u64,
    pub mime: String,
}

fn double_option<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Option<String>>, D::Error> {
    Option::<String>::deserialize(d).map(Some)
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub kind: Kind,
    pub z: Z,
    pub rect: Rect,
    pub start_frame: u32,
    pub end_frame: u32,
    // video layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coded_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coded_height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_alpha: Option<bool>,
    /// Outer None: absent (still layers); Some(None): `null` (video layer without alpha).
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub alpha_codec: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_size: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub lossless: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_range: Option<String>,
    // still layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<Resource>,
    pub blend: String,
    pub opacity: f64,
    pub visible: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Layer {
    pub fn is_video(&self) -> bool {
        self.kind == Kind::Video
    }
    pub fn active(&self, frame: u32) -> bool {
        self.start_frame <= frame && frame < self.end_frame
    }
    pub fn has_alpha(&self) -> bool {
        self.has_alpha.unwrap_or(false)
    }
    pub fn coded_size(&self) -> (u32, u32) {
        (self.coded_width.unwrap_or(self.rect.w), self.coded_height.unwrap_or(self.rect.h))
    }
    /// Valid pixels of the coded frame (spec B.4).
    pub fn content_size(&self) -> (u32, u32) {
        match self.content_size {
            Some([w, h]) => (w, h),
            None => self.coded_size(),
        }
    }
    pub fn alpha_full_range(&self) -> bool {
        self.alpha_range.as_deref() == Some("full")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioMeta {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub description_b64: Option<String>,
    /// Opus pre-skip in 48 kHz samples (spec A.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_skip: Option<u32>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    pub format: String,
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    pub canvas: Canvas,
    pub fps: FpsJson,
    pub frame_count: u32,
    pub max_rap_interval: u32,
    pub layers: Vec<Layer>,
    pub audio: Option<AudioMeta>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Meta {
    pub fn fps(&self) -> Fps {
        self.fps.into()
    }
    pub fn video_layers(&self) -> Vec<usize> {
        self.layers.iter().enumerate().filter(|(_, l)| l.is_video()).map(|(i, _)| i).collect()
    }
    /// Layer index from an id or an index ("3").
    pub fn resolve_layer(&self, key: &str) -> Result<usize, MetaError> {
        if !key.is_empty() && key.bytes().all(|b| b.is_ascii_digit()) {
            let i: usize = key.parse().map_err(|_| MetaError(format!("bad layer index {key:?}")))?;
            if i >= self.layers.len() {
                return err(format!("layer index {i} out of range (file has {} layers)", self.layers.len()));
            }
            return Ok(i);
        }
        match self.layers.iter().position(|l| l.id == key) {
            Some(i) => Ok(i),
            None => err(format!(
                "no layer {key:?}; layers: {}",
                self.layers.iter().map(|l| l.id.as_str()).collect::<Vec<_>>().join(", ")
            )),
        }
    }
    /// Highest z in the file (for putting new layers on top).
    pub fn top_z(&self) -> f64 {
        self.layers.iter().map(|l| l.z.0).fold(-1.0, f64::max)
    }
}

// ------------------------------------------------------------------------------------------------
// Checks
// ------------------------------------------------------------------------------------------------
pub fn check_id(id: &str, taken: &[&str]) -> Result<String, MetaError> {
    let ok_first = id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    let ok_rest = id.chars().all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c));
    if !ok_first || !ok_rest {
        return err(format!("layer id {id:?} must be letters, digits, '_', '-', '.' (not starting with '-' or '.')"));
    }
    if id.bytes().all(|b| b.is_ascii_digit()) {
        return err(format!("layer id {id:?} must not be a plain number (numbers refer to layer indices)"));
    }
    if taken.contains(&id) {
        return err(format!("layer id {id:?} is already used"));
    }
    Ok(id.to_string())
}

pub fn make_rect(x: i64, y: i64, w: i64, h: i64) -> Result<Rect, MetaError> {
    if w <= 0 || h <= 0 || w > u32::MAX as i64 || h > u32::MAX as i64 {
        return err(format!("rect width/height must be positive, got [{x}, {y}, {w}, {h}]"));
    }
    Ok(Rect { x, y, w: w as u32, h: h as u32 })
}

/// "x,y,w,h" (also "x y w h", "x:y:w:h").
pub fn parse_rect(s: &str) -> Result<Rect, MetaError> {
    let parts: Vec<&str> =
        s.split(|c: char| c == ',' || c == ':' || c.is_whitespace()).filter(|p| !p.is_empty()).collect();
    let nums: Option<Vec<i64>> = parts.iter().map(|p| p.parse().ok()).collect();
    match nums {
        Some(n) if n.len() == 4 => make_rect(n[0], n[1], n[2], n[3]),
        _ => err(format!("rect must be four integers x,y,w,h, got {s:?}")),
    }
}

/// A rect from JSON: [x, y, w, h], {"x":..,"y":..,"w":..,"h":..} or "x,y,w,h".
pub fn rect_from_json(v: &Value) -> Result<Rect, MetaError> {
    let bad = || MetaError(format!("rect must be four integers [x, y, w, h], got {v}"));
    let int = |v: &Value| v.as_i64().filter(|_| !v.is_f64()).ok_or_else(bad);
    match v {
        Value::String(s) => parse_rect(s),
        Value::Array(a) if a.len() == 4 => make_rect(int(&a[0])?, int(&a[1])?, int(&a[2])?, int(&a[3])?),
        Value::Object(o) => {
            let g = |k: &str| o.get(k).ok_or_else(bad).and_then(int);
            make_rect(g("x")?, g("y")?, g("w")?, g("h")?)
        }
        _ => Err(bad()),
    }
}

/// A finite number (JSON number or numeric string).
fn finite_number(v: &Value, what: &str) -> Result<f64, MetaError> {
    let n = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    match n {
        Some(n) if n.is_finite() => Ok(n),
        _ => err(format!("{what} must be a finite number, got {v}")),
    }
}

pub fn check_z(v: &Value) -> Result<Z, MetaError> {
    finite_number(v, "z").map(Z)
}

pub fn check_z_f64(z: f64) -> Result<Z, MetaError> {
    if z.is_finite() {
        Ok(Z(z))
    } else {
        err(format!("z must be a finite number, got {z}"))
    }
}

pub fn check_opacity(v: f64) -> Result<f64, MetaError> {
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        err(format!("opacity must be in [0, 1], got {v}"))
    }
}

pub fn check_blend(b: &str) -> Result<String, MetaError> {
    if BLEND_MODES.contains(&b) {
        Ok(b.to_string())
    } else {
        err(format!("blend must be one of {BLEND_MODES:?}, got {b:?}"))
    }
}

pub fn check_background(c: &str) -> Result<String, MetaError> {
    let ok = c.len() == 7 && c.starts_with('#') && c[1..].chars().all(|ch| ch.is_ascii_hexdigit());
    if ok {
        Ok(c.to_string())
    } else {
        err(format!("background must be #RRGGBB, got {c:?}"))
    }
}

pub fn parse_bool(v: &Value) -> Result<bool, MetaError> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) if n.as_i64() == Some(0) || n.as_i64() == Some(1) => Ok(n.as_i64() == Some(1)),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => err(format!("expected true/false, got {s:?}")),
        },
        _ => err(format!("expected true/false, got {v}")),
    }
}

pub fn even(n: u32) -> u32 {
    n + (n & 1)
}

// ------------------------------------------------------------------------------------------------
// Builders
// ------------------------------------------------------------------------------------------------
pub struct VideoLayerSpec<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub z: Z,
    pub rect: Rect,
    pub start: u32,
    pub end: u32,
    pub fps: Fps,
    pub alpha: bool,
    pub lossless: bool,
    pub blend: &'a str,
    pub opacity: f64,
    pub visible: bool,
}

pub fn video_layer(s: VideoLayerSpec) -> Layer {
    let (cw, ch) = (even(s.rect.w), even(s.rect.h));
    let fps = s.fps.as_f64();
    Layer {
        id: s.id.into(),
        name: s.name.into(),
        kind: Kind::Video,
        z: s.z,
        rect: s.rect,
        start_frame: s.start,
        end_frame: s.end,
        codec: Some(codec_string(cw, ch, fps, s.lossless)),
        coded_width: Some(cw),
        coded_height: Some(ch),
        has_alpha: Some(s.alpha),
        alpha_codec: Some(s.alpha.then(|| codec_string(cw, ch, fps, false))),
        // odd sizes are padded to even for 4:2:0; players crop back to the content (spec B.4)
        content_size: ((cw, ch) != (s.rect.w, s.rect.h)).then_some([s.rect.w, s.rect.h]),
        lossless: s.lossless,
        alpha_range: s.alpha.then(|| if s.lossless { "full" } else { "limited" }.to_string()),
        resource: None,
        blend: s.blend.into(),
        opacity: s.opacity,
        visible: s.visible,
        extra: Map::new(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn still_layer(
    id: &str,
    name: &str,
    z: Z,
    rect: Rect,
    start: u32,
    end: u32,
    offset: u64,
    length: u64,
    blend: &str,
    opacity: f64,
    visible: bool,
) -> Layer {
    Layer {
        id: id.into(),
        name: name.into(),
        kind: Kind::Still,
        z,
        rect,
        start_frame: start,
        end_frame: end,
        codec: None,
        coded_width: None,
        coded_height: None,
        has_alpha: None,
        alpha_codec: None,
        content_size: None,
        lossless: false,
        alpha_range: None,
        resource: Some(Resource { offset, length, mime: "image/png".into() }),
        blend: blend.into(),
        opacity,
        visible,
        extra: Map::new(),
    }
}

pub fn file_meta(
    canvas: Canvas,
    fps: Fps,
    frame_count: u32,
    gop: u32,
    layers: Vec<Layer>,
    audio: Option<AudioMeta>,
) -> Meta {
    Meta {
        format: "LVF".into(),
        version: 1,
        generator: Some(GENERATOR.into()),
        canvas,
        fps: fps.into(),
        frame_count,
        max_rap_interval: gop,
        layers,
        audio,
        extra: Map::new(),
    }
}

/// Validate and apply one edit of an editable field (`fflv set`).
pub fn apply_edit(meta: &mut Meta, key: &str, field: &str, value: &Value) -> Result<(), MetaError> {
    let i = meta.resolve_layer(key)?;
    let as_str = |v: &Value| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match field {
        "id" => {
            let taken: Vec<&str> = meta.layers.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, l)| l.id.as_str()).collect();
            let id = check_id(&as_str(value), &taken)?;
            meta.layers[i].id = id;
        }
        "name" => meta.layers[i].name = as_str(value),
        "z" => meta.layers[i].z = check_z(value)?,
        "rect" => meta.layers[i].rect = rect_from_json(value)?,
        "blend" => meta.layers[i].blend = check_blend(&as_str(value))?,
        "opacity" => meta.layers[i].opacity = check_opacity(finite_number(value, "opacity")?)?,
        "visible" => meta.layers[i].visible = parse_bool(value)?,
        _ => {
            return err(format!(
                "field {field:?} cannot be edited in place (editable: {}); frame ranges and pixels need `fflv rm` + `fflv add`",
                EDITABLE_FIELDS.join(", ")
            ))
        }
    }
    Ok(())
}

pub fn alpha_range_ok(r: &str) -> bool {
    ALPHA_RANGES.contains(&r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn z_must_be_finite() {
        for bad in [json!("nan"), json!("inf"), json!("-inf"), json!("1e999"), json!("abc"), json!(true), Value::Null] {
            assert!(check_z(&bad).is_err(), "{bad}");
        }
        assert_eq!(check_z(&json!(3)).unwrap(), Z(3.0));
        assert_eq!(check_z(&json!(" 2.5 ")).unwrap(), Z(2.5));
    }

    #[test]
    fn rects() {
        assert_eq!(parse_rect("1,2,30,40").unwrap(), Rect { x: 1, y: 2, w: 30, h: 40 });
        assert_eq!(rect_from_json(&json!([1, 2, 30, 40])).unwrap().w, 30);
        assert_eq!(rect_from_json(&json!({"x": -1, "y": 2, "w": 3, "h": 4})).unwrap().x, -1);
        assert!(rect_from_json(&json!([1, 2, 0, 4])).is_err());
        assert!(rect_from_json(&json!([1, 2, 3.5, 4])).is_err());
    }

    #[test]
    fn ids() {
        assert!(check_id("mask", &[]).is_ok());
        assert!(check_id("12", &[]).is_err());
        assert!(check_id("-x", &[]).is_err());
        assert!(check_id("a", &["a"]).is_err());
    }

    #[test]
    fn layer_json_roundtrip_keeps_null_alpha_codec_and_unknown_fields() {
        let l = video_layer(VideoLayerSpec {
            id: "bg",
            name: "bg",
            z: Z(0.0),
            rect: Rect { x: 0, y: 0, w: 63, h: 47 },
            start: 0,
            end: 10,
            fps: Fps::new(30, 1).unwrap(),
            alpha: false,
            lossless: false,
            blend: "normal",
            opacity: 1.0,
            visible: true,
        });
        let mut v = serde_json::to_value(&l).unwrap();
        assert_eq!(v["alpha_codec"], Value::Null);
        assert_eq!(v["z"], json!(0));
        assert_eq!(v["content_size"], json!([63, 47]));
        v["future_field"] = json!({"a": 1});
        let back: Layer = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back.alpha_codec, Some(None));
        assert_eq!(serde_json::to_value(&back).unwrap(), v);
    }
}
