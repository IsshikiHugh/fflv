//! `fflv view FILE`: serve the web player and one .lvd on localhost and open Chrome / Edge.
//!
//! The player reads the file through HTTP range requests, only the parts it needs, like it does
//! with a local File. Every response carries an ETag (inode + size + mtime); the player sends it
//! back with If-Match, and polls it, so when the file is rewritten (a new debug run, `fflv add`,
//! `fflv set`, ...) the page reloads the file and keeps the current frame and layer settings.
//!
//! Each request opens the file once and takes the ETag, the size and the bytes from that one open
//! file, so a response can never mix the ETag of one version with the bytes of another — even
//! while the file is being replaced (writers rename a finished file over it, spec B.11).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use include_dir::{include_dir, Dir};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::error::{Error, Result};

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
}

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

    /// Stop the worker threads and wait for them.
    pub fn stop(self) {
        for _ in &self.workers {
            self.server.unblock();
        }
        self.wait();
    }
}

fn is_loopback_name(host: &str) -> bool {
    let ip = host.trim_start_matches('[').trim_end_matches(']');
    host == "localhost" || ip.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
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
        let loopback_only = is_loopback_name(&host.to_ascii_lowercase());
        Ok(ViewServer { server: Arc::new(server), media, media_name, quiet, loopback_only })
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
        if self.loopback_only {
            let host = req.headers().iter().find(|h| h.field.equiv("Host")).map(|h| h.value.as_str().to_string());
            if !host.as_deref().map(|h| host_name(h).to_ascii_lowercase()).is_some_and(|h| is_loopback_name(&h)) {
                let _ = req.respond(Response::empty(403));
                return;
            }
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

    fn not_found() -> Response<std::io::Empty> {
        Response::empty(404).with_header(header("Cache-Control", "no-store"))
    }

    fn serve_media(&self, req: Request, _head: bool) -> std::io::Result<()> {
        let Ok(mut f) = File::open(&self.media) else { return req.respond(Self::not_found()) };
        // ETag, size and bytes all come from this one open file.
        let meta = f.metadata()?;
        let (etag, size) = (etag_of(&meta), meta.len());
        let get = |name: &'static str| {
            req.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str().to_string())
        };
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
    fn host_names() {
        assert_eq!(host_name("127.0.0.1:8765"), "127.0.0.1");
        assert_eq!(host_name("[::1]:80"), "[::1]");
        assert_eq!(host_name("localhost"), "localhost");
        assert!(!is_loopback_name(host_name("evil.example.com:8765")));
        assert!(is_loopback_name(host_name("127.0.0.2:80")) && is_loopback_name(host_name("[::1]:80")));
    }
}
