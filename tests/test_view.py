"""The `fflv view` server: HTTP range requests, ETags, If-Match."""

import threading
import urllib.error
import urllib.request

import pytest

from fflv import view


@pytest.fixture
def server(packed, monkeypatch, tmp_path):
    if not (view.VIEWER_DIR / "index.html").exists():  # tests of the media endpoint do not need the viewer
        (tmp_path / "viewer").mkdir()
        (tmp_path / "viewer" / "index.html").write_text("<html></html>")
        monkeypatch.setattr(view, "VIEWER_DIR", tmp_path / "viewer")
    srv = view.make_server(packed["path"], port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}", packed["path"].read_bytes()
    srv.shutdown()
    srv.server_close()


def get(url, headers=None, method="GET"):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, dict(res.headers), res.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_ranges(server):
    srv, base, data = server
    url = f"{base}/media/{srv.media_name}"
    st, h, body = get(url, {"Range": "bytes=0-63"})
    assert st == 206 and body == data[:64] and h["Content-Range"] == f"bytes 0-63/{len(data)}"
    st, h, body = get(url, {"Range": "bytes=100-"})
    assert st == 206 and body == data[100:]
    st, h, body = get(url, {"Range": "bytes=-16"})
    assert st == 206 and body == data[-16:]
    st, h, body = get(url)
    assert st == 200 and body == data and h["Accept-Ranges"] == "bytes"
    st, _, _ = get(url, {"Range": f"bytes={len(data)}-"})
    assert st == 416


def test_etag_and_if_match(server):
    srv, base, data = server
    url = f"{base}/media/{srv.media_name}"
    st, h, body = get(url, method="HEAD")
    assert st == 200 and int(h["Content-Length"]) == len(data) and body == b""
    etag = h["ETag"]
    st, _, body = get(url, {"Range": "bytes=0-3", "If-Match": etag})
    assert st == 206 and body == b"LVF1"
    st, h2, _ = get(url, {"Range": "bytes=0-3", "If-Match": '"stale"'})
    assert st == 412 and h2["ETag"] == etag
    st, _, _ = get(f"{base}/media/other.lvd")
    assert st == 404


def test_viewer_url(server):
    srv, base, _ = server
    url = view.viewer_url(srv)
    assert url.startswith(base + "/?src=/media/") and url.endswith("&watch=1")
    st, _, body = get(base + "/")
    assert st == 200 and b"<html" in body.lower()
