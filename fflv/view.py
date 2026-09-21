"""`fflv view FILE`: serve the web player and one .lvd on localhost and open Chrome / Edge.

The player reads the file through HTTP range requests, only the parts it needs, like it does with
a local File. Every response carries an ETag (inode + size + mtime); the player sends it back with
If-Match, and polls it, so when the file is rewritten (a new debug run, `fflv add`, `fflv set`, …)
the page reloads the file and keeps the current frame and layer settings.

Each request opens the file once and takes the ETag, the size and the bytes from that one open
file, so a response can never mix the ETag of one version with the bytes of another — even while
the file is being replaced (writers rename a finished file over it, spec B.11).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 1 << 20


class ViewError(RuntimeError):
    pass


def etag_of(st: os.stat_result) -> str:
    """Changes whenever the file is replaced (inode) or rewritten in place (size, mtime)."""
    return f'"{st.st_ino:x}-{st.st_size:x}-{st.st_mtime_ns:x}"'


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, media: Path, quiet: bool):
        self.media = media
        self.media_name = media.name
        self.quiet = quiet
        super().__init__(addr, _Handler)


class _Handler(SimpleHTTPRequestHandler):
    server: _Server

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def log_message(self, fmt, *args):  # noqa: D401 - quiet by default
        if not self.server.quiet:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if urlsplit(self.path).path.startswith("/media/"):
            return self._media(head=False)
        return super().do_GET()

    def do_HEAD(self):
        if urlsplit(self.path).path.startswith("/media/"):
            return self._media(head=True)
        return super().do_HEAD()

    def _media(self, head: bool) -> None:
        name = unquote(urlsplit(self.path).path[len("/media/"):])
        if name != self.server.media_name:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            f = open(self.server.media, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with f:
            self._serve(f, head)

    def _serve(self, f, head: bool) -> None:
        st = os.fstat(f.fileno())  # ETag, size and bytes all come from this one open file
        etag, size = etag_of(st), st.st_size
        want = self.headers.get("If-Match")
        if want and want.strip() != etag:
            self.send_response(HTTPStatus.PRECONDITION_FAILED)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = 0, size  # [start, end)
        status = HTTPStatus.OK
        rng = self.headers.get("Range")
        if rng:
            m = _RANGE.match(rng.strip())
            if not m or (not m.group(1) and not m.group(2)):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if m.group(1):
                start = int(m.group(1))
                end = min(size, int(m.group(2)) + 1) if m.group(2) else size
            else:  # suffix range: the last N bytes
                start = max(0, size - int(m.group(2)))
            if start >= size or start >= end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(end - start))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end - 1}/{size}")
        self.end_headers()
        if head:
            return
        try:
            f.seek(start)
            left = end - start
            while left > 0:
                buf = f.read(min(_CHUNK, left))
                if not buf:
                    break
                self.wfile.write(buf)
                left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_server(media: str | os.PathLike, host: str = "127.0.0.1", port: int = 0, quiet: bool = True) -> _Server:
    """Create (not start) the server. port 0 = first free port from 8765."""
    media = Path(media).resolve()
    if not media.is_file():
        raise ViewError(f"no such file: {media}")
    if not (VIEWER_DIR / "index.html").exists():
        raise ViewError(f"viewer not built ({VIEWER_DIR} is missing); run `npm install && npm run build` in player/")
    if port:
        return _Server((host, port), media, quiet)
    for p in range(8765, 8865):
        try:
            return _Server((host, p), media, quiet)
        except OSError:
            continue
    return _Server((host, 0), media, quiet)


def viewer_url(server: _Server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/?src=/media/{quote(server.media_name)}&watch=1"


_MAC_APPS = {"chrome": "Google Chrome", "edge": "Microsoft Edge", "chromium": "Chromium"}
_LINUX_BINS = {"chrome": ["google-chrome", "google-chrome-stable"], "edge": ["microsoft-edge", "microsoft-edge-stable"],
               "chromium": ["chromium", "chromium-browser"]}


def open_browser(url: str, browser: str | None = None) -> str:
    """Open `url` in Chrome/Edge (WebCodecs); returns what was used."""
    names = [browser] if browser and browser != "default" else list(_MAC_APPS)
    if browser == "default":
        webbrowser.open(url)
        return "default browser"
    if sys.platform == "darwin":
        for n in names:
            app = _MAC_APPS.get(n)
            if app and any(Path(d, f"{app}.app").exists() for d in ("/Applications", Path.home() / "Applications")):
                subprocess.Popen(["open", "-a", app, url])
                return app
    elif sys.platform.startswith("linux"):
        for n in names:
            for b in _LINUX_BINS.get(n, []):
                if shutil.which(b):
                    subprocess.Popen([b, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return b
    webbrowser.open(url)
    return "default browser (the player needs Chrome or Edge)"


def serve(media: str | os.PathLike, *, host: str = "127.0.0.1", port: int = 0, browser: str | None = None,
          open_page: bool = True, quiet: bool = True) -> None:
    server = make_server(media, host, port, quiet)
    url = viewer_url(server)
    print(f"serving {server.media} at\n  {url}\n(the page follows changes to the file; Ctrl+C to stop)", flush=True)
    if open_page:
        threading.Timer(0.2, lambda: print(f"opened in {open_browser(url, browser)}", flush=True)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
