//! `fflv view FILE`: serve the web player and one .lvd on localhost; `--open` opens Chrome / Edge.
//!
//! The player reads the file through HTTP range requests, only the parts it needs, like it does
//! with a local File. Every response carries an ETag (inode + size + mtime); the player sends it
//! back with If-Match, and polls it, so when the file is rewritten (a new debug run, `fflv add`,
//! `fflv set`, ...) the page reloads the file and keeps the current frame and layer settings.
//!
//! Each request opens the file once and takes the ETag, the size and the bytes from that one open
//! file, so a file replaced while a response is sent (writers rename a finished file over it,
//! spec B.11) cannot mix into it: the response is all old file. An in-place metadata rewrite
//! (`fflv set`) changes the open file itself, so a range read while one happens can mix bytes from
//! before and after it; the rewrite changes the ETag (mtime), so the player's next request, sent
//! with the old ETag in If-Match, gets 412 and the page reloads the file.
//!
//! The player can also export what it shows (its visible layers, at their opacities) to a video:
//!   GET  /export              {"media", "formats"}: what this server can export
//!   POST /export              {"format", "layers": [id...], "opacity": {id: 0–1}, "order": [id...]}
//!                             → {"id"}; one at a time. "order" is the draw order, bottom first.
//!   GET  /export/ID           {"state": running | done | failed | cancelled, "done", "total", "error"}
//!   GET  /export/ID/file      the finished video, as a download named after the .lvd
//!   POST /export/ID/cancel
//! POST requests must be sent as application/json (a page on another site cannot send that
//! without a CORS preflight, which this server does not answer). The video is rendered by
//! [`render`] (like `fflv render`) into a private temporary directory, and kept until the next
//! export or until the server stops (a server killed by Ctrl+C leaves it behind).
//!
//! Listening on a loopback address (the default), only requests addressed to a loopback name are
//! served (a page elsewhere must not reach the file through DNS rebinding). On another address,
//! the Host must be an IP address, `localhost` or the name the server was bound to.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use include_dir::{include_dir, Dir};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::error::{Error, Result};
use crate::render::{render, RenderOptions};

/// The built player (player/, `npm run build`).
static VIEWER: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/viewer");

pub fn viewer_is_built() -> bool {
    VIEWER.get_file("index.html").is_some()
}

/// Changes whenever the file is replaced (inode) or rewritten in place (size, mtime).
pub fn etag_of(meta: &std::fs::Metadata) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let mtime_ns = meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128;
        format!("\"{:x}-{:x}-{:x}\"", meta.ino(), meta.size(), mtime_ns)
    }
    #[cfg(not(unix))]
    {
        let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
        format!("\"{:x}-{:x}\"", meta.len(), mtime.map_or(0, |d| d.as_nanos()))
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

/// A request header's value.
fn request_header(req: &Request, name: &'static str) -> Option<String> {
    req.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str().to_string())
}

/// The mutex's data, also after a panic while it was held (the state it guards stays usable).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

/// `%XX` escapes → bytes (paths only).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn percent_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

pub struct ViewServer {
    server: Arc<Server>,
    media: PathBuf,
    media_name: String,
    quiet: bool,
    /// Listening on a loopback address: only accept requests addressed to it (a page elsewhere
    /// must not reach the file through DNS rebinding).
    loopback_only: bool,
    /// The host the server was bound to (lowercase), also accepted as a Host.
    bound_host: String,
    /// The current (or last) export; shared with the thread that renders it.
    export: Arc<Mutex<Option<Export>>>,
    /// Private directory of the export files, created with the first export.
    export_dir: Mutex<Option<PathBuf>>,
}

/// Export ids, unique in the process (they also name the temporary files).
static NEXT_EXPORT: AtomicU64 = AtomicU64::new(1);

/// A started server (see [`ViewServer::start`]).
pub struct Running {
    server: Arc<Server>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Running {
    /// Block until the server stops (it only stops through [`Running::stop`]).
    pub fn wait(self) {
        for w in self.workers {
            let _ = w.join();
        }
    }

    /// Stop the worker threads; waits up to 2 s for requests in progress. A worker still busy
    /// then (sending a large file to a slow client, say) is left to stop by itself once done.
    pub fn stop(self) {
        for _ in &self.workers {
            self.server.unblock();
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        for w in self.workers {
            while !w.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if w.is_finished() {
                let _ = w.join();
            }
        }
    }
}

fn is_loopback_name(host: &str) -> bool {
    let ip = host.trim_start_matches('[').trim_end_matches(']');
    host == "localhost" || ip.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// An IP address (`[...]` for IPv6), not a DNS name.
fn is_ip_address(host: &str) -> bool {
    host.trim_start_matches('[').trim_end_matches(']').parse::<std::net::IpAddr>().is_ok()
}

/// The host name of a Host header value (without the port).
fn host_name(value: &str) -> &str {
    let v = value.trim();
    if v.starts_with('[') {
        return v.find(']').map_or(v, |i| &v[..=i]);
    }
    v.split(':').next().unwrap_or(v)
}

/// `bytes=a-b` / `bytes=a-` / `bytes=-n` → [start, end) within `size`; Err(()) when unsatisfiable,
/// None when malformed or not a single range (RFC 7233: then the whole file is sent).
fn parse_range(value: &str, size: u64) -> Option<std::result::Result<(u64, u64), ()>> {
    let spec = value.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let num = |s: &str| if s.is_empty() { Some(None) } else { s.parse::<u64>().ok().map(Some) };
    let (a, b) = (num(a)?, num(b)?);
    let (start, end) = match (a, b) {
        (None, None) => return None,
        (Some(a), Some(b)) if b < a => return None,
        (Some(a), b) => (a, b.map_or(size, |b| b.saturating_add(1).min(size))),
        (None, Some(n)) => (size.saturating_sub(n), size),
    };
    if start >= size || start >= end {
        return Some(Err(()));
    }
    Some(Ok((start, end)))
}

impl ViewServer {
    /// Create (not start) the server. Port 0: the first free port from 8765.
    pub fn bind(media: &Path, host: &str, port: u16, quiet: bool) -> Result<ViewServer> {
        let media = std::path::absolute(media)?;
        if !media.is_file() {
            return Err(Error::View(format!("no such file: {}", media.display())));
        }
        if !viewer_is_built() {
            return Err(Error::View("this fflv was built without the viewer (run `npm run build` in player/)".into()));
        }
        let last_error = std::cell::RefCell::new(String::new());
        let try_bind = |p: u16| Server::http((host, p)).map_err(|e| *last_error.borrow_mut() = e.to_string()).ok();
        let server = if port != 0 { try_bind(port) } else { (8765..8865).find_map(try_bind).or_else(|| try_bind(0)) };
        let server = server.ok_or_else(|| {
            let port = if port == 0 { "any port".to_string() } else { format!("port {port}") };
            Error::View(format!("cannot listen on {host}, {port}: {}", last_error.borrow()))
        })?;
        let media_name = media.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let bound_host = host.to_ascii_lowercase();
        let loopback_only = is_loopback_name(&bound_host);
        if !loopback_only {
            eprintln!(
                "warning: listening on {host}, not a loopback address: anyone who can reach this machine there can \
                 read {} and export from it (there is no authentication)",
                media.display()
            );
        }
        Ok(ViewServer {
            server: Arc::new(server),
            media,
            media_name,
            quiet,
            loopback_only,
            bound_host,
            export: Arc::new(Mutex::new(None)),
            export_dir: Mutex::new(None),
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.server.server_addr().to_ip().expect("an IP listener")
    }

    pub fn media(&self) -> &Path {
        &self.media
    }

    pub fn url(&self) -> String {
        let a = self.addr();
        format!("http://{a}/?src=/media/{}&watch=1", percent_encode(&self.media_name))
        // [::1]:port for IPv6
    }

    /// Serve on `threads` worker threads, in the background.
    pub fn start(self, threads: usize) -> Running {
        let server = self.server.clone();
        let this = Arc::new(self);
        let workers = (0..threads.max(1))
            .map(|_| {
                let s = this.clone();
                std::thread::spawn(move || {
                    while let Ok(req) = s.server.recv() {
                        s.handle(req);
                    }
                })
            })
            .collect();
        Running { server, workers }
    }

    /// Serve until the process ends.
    pub fn serve(self, threads: usize) {
        self.start(threads).wait();
    }

    fn handle(&self, req: Request) {
        let url = req.url().to_string();
        let path = url.split(['?', '#']).next().unwrap_or("/").to_string();
        let head = *req.method() == Method::Head;
        if !self.quiet {
            eprintln!("{} {} {}", req.remote_addr().map_or("-".into(), |a| a.to_string()), req.method(), url);
        }
        let host = request_header(&req, "Host").map(|h| host_name(&h).to_ascii_lowercase());
        if !host.is_some_and(|h| self.host_allowed(&h)) {
            let _ = req.respond(Response::empty(403));
            return;
        }
        // Request bodies are refused before anything is read: only POST /export has one, of a
        // stated length. (tiny_http has no socket timeouts, so a body being read ties up a
        // worker; these limits keep what one request can hold small.)
        let export_post = *req.method() == Method::Post && path == "/export";
        let max_body = if export_post { MAX_EXPORT_REQUEST } else { MAX_OTHER_BODY };
        if request_header(&req, "Transfer-Encoding").is_some() || (export_post && req.body_length().is_none()) {
            let _ = req.respond(json_error(411, "send the request with a Content-Length"));
            return;
        }
        if req.body_length().is_some_and(|n| n as u64 > max_body) {
            let _ = req.respond(json_error(413, "request body too large"));
            return;
        }
        if path == "/export" || path.starts_with("/export/") {
            let _ = self.handle_export(req, &path);
            return;
        }
        if !matches!(req.method(), Method::Get | Method::Head) {
            let _ = req.respond(Response::empty(405).with_header(header("Allow", "GET, HEAD")));
            return;
        }
        let result = if let Some(name) = path.strip_prefix("/media/") {
            if percent_decode(name) == self.media_name {
                self.serve_media(req, head)
            } else {
                req.respond(Self::not_found())
            }
        } else {
            let rel = percent_decode(path.trim_start_matches('/'));
            let rel = if rel.is_empty() { "index.html".to_string() } else { rel };
            match VIEWER.get_file(&rel).filter(|_| !rel.split('/').any(|c| c == "..")) {
                Some(f) => req.respond(
                    Response::from_data(f.contents())
                        .with_header(header("Content-Type", content_type(&rel)))
                        .with_header(header("Cache-Control", "no-store"))
                        .with_chunked_threshold(usize::MAX),
                ),
                None => req.respond(Self::not_found()),
            }
        };
        let _ = result;
    }

    /// Whether a request addressed to `host` (a Host header's name, lowercase) is served.
    fn host_allowed(&self, host: &str) -> bool {
        if self.loopback_only {
            return is_loopback_name(host);
        }
        // a DNS name other than the server's own could point anywhere (DNS rebinding)
        is_ip_address(host) || host == "localhost" || host == self.bound_host
    }

    fn not_found() -> Response<std::io::Empty> {
        Response::empty(404).with_header(header("Cache-Control", "no-store"))
    }

    fn serve_media(&self, req: Request, _head: bool) -> std::io::Result<()> {
        let Ok(mut f) = File::open(&self.media) else { return req.respond(Self::not_found()) };
        // ETag, size and bytes all come from this one open file.
        let meta = f.metadata()?;
        let (etag, size) = (etag_of(&meta), meta.len());
        let get = |name: &'static str| request_header(&req, name);
        if let Some(want) = get("If-Match") {
            if want.trim() != "*" && want.trim() != etag {
                return req.respond(media_headers(Response::empty(412), &etag));
            }
        }
        let (mut start, mut end, mut status) = (0, size, 200);
        if let Some(r) = get("Range") {
            match parse_range(&r, size) {
                None => {} // not a single valid range: the whole file
                Some(Err(())) => {
                    return req.respond(
                        media_headers(Response::empty(416), &etag)
                            .with_header(header("Content-Range", &format!("bytes */{size}"))),
                    )
                }
                Some(Ok((a, b))) => (start, end, status) = (a, b, 206),
            }
        }
        f.seek(SeekFrom::Start(start))?;
        let len = end - start;
        let mut resp = Response::new(StatusCode(status), vec![], f.take(len), Some(len as usize), None)
            .with_header(header("Content-Type", "application/octet-stream"));
        if status == 206 {
            resp = resp.with_header(header("Content-Range", &format!("bytes {start}-{}/{size}", end - 1)));
        }
        req.respond(media_headers(resp, &etag))
    }
}

impl Drop for ViewServer {
    fn drop(&mut self) {
        if let Some(e) = lock(&self.export).take() {
            let _ = fs::remove_file(&e.path);
        }
        if let Some(dir) = lock(&self.export_dir).take() {
            let _ = fs::remove_dir(&dir); // empty unless an export thread is still writing
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Export
// ------------------------------------------------------------------------------------------------

/// Video formats the player can export to (the extension picks the codec, see `render`).
const EXPORT_FORMATS: &[(&str, &str)] =
    &[("mp4", "video/mp4"), ("webm", "video/webm"), ("mov", "video/quicktime"), ("mkv", "video/x-matroska")];

/// Largest export request accepted (it lists layer ids and opacities).
const MAX_EXPORT_REQUEST: u64 = 1 << 20;

/// Largest body accepted (and ignored) with other requests.
const MAX_OTHER_BODY: u64 = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExportState {
    Running,
    Done,
    Failed,
    Cancelled,
}

struct Export {
    id: u64,
    state: ExportState,
    error: Option<String>,
    done: u32,
    total: u32,
    cancel: bool,
    path: PathBuf,
    format: &'static str,
}

impl Export {
    fn status(&self) -> Value {
        let state = match self.state {
            ExportState::Running => "running",
            ExportState::Done => "done",
            ExportState::Failed => "failed",
            ExportState::Cancelled => "cancelled",
        };
        json!({"id": self.id, "state": state, "done": self.done, "total": self.total, "error": self.error})
    }
}

/// `{"format", "layers", "opacity", "order"}` → the output format and what to render.
fn parse_export(body: &[u8]) -> std::result::Result<(&'static str, RenderOptions), String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("not JSON: {e}"))?;
    let format = v.get("format").and_then(Value::as_str).unwrap_or("mp4");
    let Some(&(format, _)) = EXPORT_FORMATS.iter().find(|(f, _)| *f == format) else {
        return Err(format!("unsupported format {format:?}"));
    };
    let mut opacity = Vec::new();
    if let Some(o) = v.get("opacity").filter(|o| !o.is_null()) {
        let o = o.as_object().ok_or("\"opacity\" must map layer ids to numbers")?;
        for (k, a) in o {
            let a = a.as_f64().filter(|a| (0.0..=1.0).contains(a)).ok_or("opacities must be numbers from 0 to 1")?;
            opacity.push((k.clone(), a as f32));
        }
    }
    let ids = |key: &str| -> std::result::Result<Option<Vec<String>>, String> {
        match v.get(key).filter(|o| !o.is_null()) {
            None => Ok(None),
            Some(o) => o
                .as_array()
                .and_then(|a| a.iter().map(|l| l.as_str().map(String::from)).collect())
                .map(Some)
                .ok_or_else(|| format!("\"{key}\" must be a list of layer ids")),
        }
    };
    let layers = ids("layers")?.ok_or("\"layers\" must be a list of layer ids")?;
    let order = ids("order")?;
    Ok((format, RenderOptions { layers: Some(layers), opacity, order, ..Default::default() }))
}

/// A file name for a header: the name itself (RFC 5987) and an ASCII fallback.
fn content_disposition(name: &str) -> String {
    let ascii: String =
        name.chars().map(|c| if c.is_ascii_graphic() && c != '"' && c != '\\' || c == ' ' { c } else { '_' }).collect();
    format!("attachment; filename=\"{ascii}\"; filename*=UTF-8''{}", percent_encode(name))
}

fn json_response(status: u16, v: &Value) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(v.to_string().into_bytes())
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json"))
        .with_header(header("Cache-Control", "no-store"))
}

fn json_error(status: u16, msg: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(status, &json!({ "error": msg }))
}

/// Sent as JSON: a page on another site cannot send that without a CORS preflight, which this
/// server does not answer (so it cannot start or cancel exports).
fn is_json(req: &Request) -> bool {
    request_header(req, "Content-Type").is_some_and(|t| t.trim_start().starts_with("application/json"))
}

/// A new private directory (owner only, on Unix) with an unguessable name, for export files.
fn private_temp_dir() -> std::io::Result<PathBuf> {
    use std::hash::{BuildHasher, Hasher};
    let mut last = None;
    for _ in 0..16 {
        // RandomState is seeded randomly (per process, then varied per instance)
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        h.write_u128(now);
        h.write_u32(std::process::id());
        let dir = std::env::temp_dir().join(format!("fflv-view-{:016x}", h.finish()));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.expect("tried"))
}

/// The message of a caught panic.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default()
}

impl ViewServer {
    fn handle_export(&self, mut req: Request, path: &str) -> std::io::Result<()> {
        let rest: Vec<&str> = path.trim_start_matches("/export").trim_start_matches('/').split('/').collect();
        let method = req.method().clone();
        let id = rest[0].parse::<u64>().ok();
        match (&method, rest.as_slice()) {
            (Method::Get, [""]) => {
                let formats: Vec<&str> = EXPORT_FORMATS.iter().map(|(f, _)| *f).collect();
                let media = format!("/media/{}", percent_encode(&self.media_name));
                req.respond(json_response(200, &json!({ "media": media, "formats": formats })))
            }
            (Method::Post, [""]) => {
                if !is_json(&req) {
                    return req.respond(json_error(415, "send the export request as application/json"));
                }
                // at most MAX_EXPORT_REQUEST bytes, the length stated (checked in handle())
                let mut body = Vec::new();
                req.as_reader().take(MAX_EXPORT_REQUEST + 1).read_to_end(&mut body)?;
                if body.len() as u64 > MAX_EXPORT_REQUEST {
                    return req.respond(json_error(413, "export request too large"));
                }
                match parse_export(&body) {
                    Ok((format, opts)) => match self.start_export(format, opts) {
                        Ok(id) => req.respond(json_response(202, &json!({ "id": id }))),
                        Err(msg) => req.respond(json_error(409, &msg)),
                    },
                    Err(msg) => req.respond(json_error(400, &msg)),
                }
            }
            (Method::Get, [_]) => match lock(&self.export).as_ref().filter(|e| Some(e.id) == id) {
                Some(e) => req.respond(json_response(200, &e.status())),
                None => req.respond(json_error(404, "no such export")),
            },
            (Method::Post, [_, "cancel"]) => {
                if !is_json(&req) {
                    return req.respond(json_error(415, "send the cancel request as application/json"));
                }
                let status = lock(&self.export).as_mut().filter(|e| Some(e.id) == id).map(|e| {
                    e.cancel = e.state == ExportState::Running;
                    e.status()
                });
                match status {
                    Some(s) => req.respond(json_response(200, &s)),
                    None => req.respond(json_error(404, "no such export")),
                }
            }
            (Method::Get | Method::Head, [_, "file"]) => {
                let done = lock(&self.export)
                    .as_ref()
                    .filter(|e| Some(e.id) == id && e.state == ExportState::Done)
                    .map(|e| (e.path.clone(), e.format));
                let Some((file, format)) = done else { return req.respond(json_error(404, "no such finished export")) };
                let Ok(f) = File::open(&file) else { return req.respond(json_error(404, "the exported file is gone")) };
                let mime =
                    EXPORT_FORMATS.iter().find(|(f, _)| *f == format).map_or("application/octet-stream", |(_, m)| m);
                let stem = Path::new(&self.media_name).file_stem().map_or("export".into(), |s| s.to_string_lossy());
                req.respond(
                    Response::from_file(f)
                        .with_header(header("Content-Type", mime))
                        .with_header(header("Content-Disposition", &content_disposition(&format!("{stem}.{format}"))))
                        .with_header(header("Cache-Control", "no-store"))
                        .with_chunked_threshold(usize::MAX),
                )
            }
            _ => req.respond(json_error(404, "unknown export request")),
        }
    }

    /// Start rendering in the background; Err when another export is still running.
    fn start_export(&self, format: &'static str, opts: RenderOptions) -> std::result::Result<u64, String> {
        let mut slot = lock(&self.export);
        if slot.as_ref().is_some_and(|e| e.state == ExportState::Running) {
            return Err("an export is already running".into());
        }
        if let Some(old) = slot.take() {
            let _ = fs::remove_file(&old.path);
        }
        let id = NEXT_EXPORT.fetch_add(1, Ordering::Relaxed);
        let out = {
            let mut dir = lock(&self.export_dir);
            if dir.is_none() {
                *dir = Some(private_temp_dir().map_err(|e| format!("cannot create a temporary directory: {e}"))?);
            }
            dir.as_ref().expect("just set").join(format!("export-{id}.{format}"))
        };
        // claimed here (the name must not exist), then overwritten by the render
        File::options()
            .write(true)
            .create_new(true)
            .open(&out)
            .map_err(|e| format!("cannot create {}: {e}", out.display()))?;
        *slot = Some(Export {
            id,
            state: ExportState::Running,
            error: None,
            done: 0,
            total: 0,
            cancel: false,
            path: out.clone(),
            format,
        });
        let (jobs, media) = (self.export.clone(), self.media.clone());
        std::thread::spawn(move || {
            let mut progress = |done: u32, total: u32| -> Result<()> {
                let mut slot = lock(&jobs);
                match slot.as_mut().filter(|e| e.id == id) {
                    Some(e) if !e.cancel => {
                        (e.done, e.total) = (done, total);
                        Ok(())
                    }
                    _ => Err(Error::Output("export cancelled".into())),
                }
            };
            // a panic must not leave the export "running" (refusing every later one) or its file
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                render(&media, &out.to_string_lossy(), &opts, Some(&mut progress))
            }))
            .unwrap_or_else(|p| Err(Error::Output(format!("the export failed: {}", panic_message(&*p)))));
            let mut slot = lock(&jobs);
            let Some(e) = slot.as_mut().filter(|e| e.id == id) else {
                let _ = fs::remove_file(&out);
                return;
            };
            e.state = match result {
                Ok(_) => ExportState::Done,
                Err(_) if e.cancel => ExportState::Cancelled,
                Err(err) => {
                    e.error = Some(err.to_string());
                    ExportState::Failed
                }
            };
            if e.state != ExportState::Done {
                let _ = fs::remove_file(&out);
            }
        });
        Ok(id)
    }
}

/// Media responses always carry a Content-Length (tiny_http would switch to chunked transfer for
/// large bodies, and HEAD responses would lose the file size the player reads from them).
fn media_headers<R: Read>(r: Response<R>, etag: &str) -> Response<R> {
    r.with_header(header("ETag", etag))
        .with_header(header("Accept-Ranges", "bytes"))
        .with_header(header("Cache-Control", "no-store"))
        .with_chunked_threshold(usize::MAX)
}

/// Open `url` in Chrome / Edge (WebCodecs); returns what was used.
pub fn open_browser(url: &str, browser: Option<&str>) -> String {
    use std::process::{Command, Stdio};
    let spawn = |cmd: &mut Command| cmd.stdout(Stdio::null()).stderr(Stdio::null()).spawn().is_ok();
    let default = |url: &str| -> bool {
        if cfg!(target_os = "macos") {
            spawn(Command::new("open").arg(url))
        } else if cfg!(windows) {
            spawn(Command::new("cmd").args(["/C", "start", "", url]))
        } else {
            spawn(Command::new("xdg-open").arg(url))
        }
    };
    if browser == Some("default") {
        default(url);
        return "default browser".into();
    }
    let names: Vec<&str> = match browser {
        Some(b) => vec![b],
        None => vec!["chrome", "edge", "chromium"],
    };
    if cfg!(target_os = "macos") {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        for n in &names {
            let app = match *n {
                "chrome" => "Google Chrome",
                "edge" => "Microsoft Edge",
                "chromium" => "Chromium",
                _ => continue,
            };
            let found = [PathBuf::from("/Applications"), home.join("Applications")]
                .iter()
                .any(|d| d.join(format!("{app}.app")).exists());
            if found && spawn(Command::new("open").args(["-a", app, url])) {
                return app.into();
            }
        }
    } else if cfg!(target_os = "linux") {
        for n in &names {
            let bins: &[&str] = match *n {
                "chrome" => &["google-chrome", "google-chrome-stable"],
                "edge" => &["microsoft-edge", "microsoft-edge-stable"],
                "chromium" => &["chromium", "chromium-browser"],
                _ => &[],
            };
            for b in bins {
                if spawn(Command::new(b).arg(url)) {
                    return (*b).into();
                }
            }
        }
    }
    default(url);
    "default browser (the player needs Chrome or Edge)".into()
}

/// Serve `media` and (optionally) open the page; runs until Ctrl+C.
pub fn serve(media: &Path, host: &str, port: u16, browser: Option<&str>, open_page: bool, quiet: bool) -> Result<()> {
    let server = ViewServer::bind(media, host, port, quiet)?;
    let url = server.url();
    println!(
        "serving {} at\n  {url}\n(the page follows changes to the file; Ctrl+C to stop)",
        server.media().display()
    );
    if open_page {
        let (u, b) = (url.clone(), browser.map(String::from));
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            println!("opened in {}", open_browser(&u, b.as_deref()));
        });
    }
    server.serve(8);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        assert_eq!(parse_range("bytes=0-9", 100), Some(Ok((0, 10))));
        assert_eq!(parse_range("bytes=90-", 100), Some(Ok((90, 100))));
        assert_eq!(parse_range("bytes=-10", 100), Some(Ok((90, 100))));
        assert_eq!(parse_range("bytes=95-200", 100), Some(Ok((95, 100))));
        assert_eq!(parse_range("bytes=100-", 100), Some(Err(())));
        assert_eq!(parse_range("bytes=-", 100), None);
        assert_eq!(parse_range("bytes=5-3", 100), None);
        assert_eq!(parse_range("bytes=0-1,5-6", 100), None);
        assert_eq!(parse_range("bytes=0-18446744073709551615", 100), Some(Ok((0, 100))));
        assert_eq!(parse_range("items=0-1", 100), None);
    }

    #[test]
    fn percent_coding() {
        assert_eq!(percent_decode("a%20b%2F.lvd"), "a b/.lvd");
        assert_eq!(percent_encode("a b.lvd"), "a%20b.lvd");
    }

    #[test]
    fn export_requests() {
        let (format, o) =
            parse_export(br#"{"format": "webm", "layers": ["bg", "1"], "opacity": {"bg": 0.5}}"#).unwrap();
        assert_eq!(format, "webm");
        assert_eq!(o.layers, Some(vec!["bg".to_string(), "1".to_string()]));
        assert_eq!(o.opacity, vec![("bg".to_string(), 0.5)]);
        assert!(!o.transparent && o.start == 0 && o.end.is_none() && o.order.is_none());
        let (_, o) = parse_export(br#"{"layers": ["a"], "order": ["b", "a"]}"#).unwrap();
        assert_eq!(o.order, Some(vec!["b".to_string(), "a".to_string()]));
        assert!(parse_export(br#"{"layers": [], "order": "a"}"#).is_err());
        assert_eq!(parse_export(br#"{"layers": []}"#).unwrap().0, "mp4");
        assert!(parse_export(br#"{"format": "gif", "layers": []}"#).is_err());
        assert!(parse_export(br#"{"format": "mp4"}"#).is_err());
        assert!(parse_export(br#"{"layers": [1]}"#).is_err());
        assert!(parse_export(br#"{"layers": [], "opacity": {"bg": 2}}"#).is_err());
        assert!(parse_export(b"layers").is_err());
        assert_eq!(
            content_disposition("débug \"a\".mp4"),
            "attachment; filename=\"d_bug _a_.mp4\"; filename*=UTF-8''d%C3%A9bug%20%22a%22.mp4"
        );
    }

    #[test]
    fn host_names() {
        assert_eq!(host_name("127.0.0.1:8765"), "127.0.0.1");
        assert_eq!(host_name("[::1]:80"), "[::1]");
        assert_eq!(host_name("localhost"), "localhost");
        assert!(!is_loopback_name(host_name("evil.example.com:8765")));
        assert!(is_loopback_name(host_name("127.0.0.2:80")) && is_loopback_name(host_name("[::1]:80")));
        assert!(is_ip_address(host_name("192.168.1.20:8765")) && is_ip_address(host_name("[fe80::1]:80")));
        assert!(!is_ip_address(host_name("evil.example.com:8765")) && !is_ip_address("localhost"));
    }

    #[test]
    fn private_dirs_and_panics() {
        let (a, b) = (private_temp_dir().unwrap(), private_temp_dir().unwrap());
        assert_ne!(a, b);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&a).unwrap().permissions().mode() & 0o077, 0);
        }
        fs::remove_dir(a).unwrap();
        fs::remove_dir(b).unwrap();
        let p = std::panic::catch_unwind(|| -> u8 { panic!("boom {}", 1) }).unwrap_err();
        assert_eq!(panic_message(&*p), "boom 1");
    }
}
