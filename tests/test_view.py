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


def test_responses_stay_consistent_while_the_file_is_replaced(tmp_path):
    """Writers rename a new version over the file (spec B.11). Every response must carry the bytes
    of the same version as its ETag and Content-Range, or If-Match cannot detect the change."""
    import os

    versions = {0xAA: 300_000, 0xBB: 400_000}  # fill byte -> size
    media = tmp_path / "watched.lvd"
    media.write_bytes(bytes([0xAA]) * versions[0xAA])
    srv = view.make_server(media, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    stop = threading.Event()

    def replace_forever():
        n = 0
        while not stop.is_set():
            byte = (0xAA, 0xBB)[n % 2]
            tmp = tmp_path / f".v{n % 2}"
            tmp.write_bytes(bytes([byte]) * versions[byte])
            os.replace(tmp, media)
            n += 1

    t = threading.Thread(target=replace_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/media/watched.lvd"
    bad, checked, versions_of_etag = [], 0, {}
    try:
        for i in range(1500):
            st, h, body = get(url, {"Range": f"bytes={(i * 7919) % 250_000}-{(i * 7919) % 250_000 + 4095}"})
            if st != 206:
                continue
            byte = body[0]
            total = int(h["Content-Range"].split("/")[1])
            if body != bytes([byte]) * len(body) or total != versions[byte]:
                bad.append((i, hex(byte), total))  # bytes of one version, size of another
            versions_of_etag.setdefault(h["ETag"], set()).add(byte)
            checked += 1
    finally:
        stop.set()
        t.join()
        srv.shutdown()
        srv.server_close()
    assert checked > 1000
    assert bad == [], f"{len(bad)} of {checked} responses mixed two versions, e.g. {bad[:3]}"
    shared = {etag: sorted(map(hex, b)) for etag, b in versions_of_etag.items() if len(b) > 1}
    assert not shared, f"one ETag was served with the bytes of different versions: {list(shared.items())[:3]}"
