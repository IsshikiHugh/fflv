"""The `fflv view` server: HTTP range requests, ETags, If-Match, the bundled player, export."""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

import fflv
from conftest import read_video


class Server:
    def __init__(self, media, *args):
        self.proc = subprocess.Popen([sys.executable, "-m", "fflv", "view", str(media), *args],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in self.proc.stdout:
            m = re.search(r"(http://[\d.]+:\d+)(/\?src=\S+)", line)
            if m:
                self.base, self.page = m.group(1), m.group(1) + m.group(2)
                break
        else:
            raise RuntimeError("fflv view did not start: " + self.proc.stderr.read())
        self.media_url = f"{self.base}/media/{os.path.basename(media)}"

    def stop(self):
        self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()
        self.proc.stderr.close()


@pytest.fixture
def server(packed):
    srv = Server(packed["path"])
    yield srv, packed["path"].read_bytes()
    srv.stop()


def get(url, headers=None, method="GET", data=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, dict(res.headers), res.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def post(url, body, content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return get(url, {"Content-Type": content_type}, method="POST", data=data)


def test_ranges(server):
    srv, data = server
    url = srv.media_url
    st, h, body = get(url, {"Range": "bytes=0-63"})
    assert st == 206 and body == data[:64] and h["Content-Range"] == f"bytes 0-63/{len(data)}"
    st, h, body = get(url, {"Range": "bytes=100-"})
    assert st == 206 and body == data[100:]
    st, h, body = get(url, {"Range": "bytes=-16"})
    assert st == 206 and body == data[-16:]
    st, h, body = get(url)
    assert st == 200 and body == data and h["Accept-Ranges"] == "bytes" and h["Cache-Control"] == "no-store"
    st, h, _ = get(url, {"Range": f"bytes={len(data)}-"})
    assert st == 416 and h["Content-Range"] == f"bytes */{len(data)}"


def test_etag_and_if_match(server):
    srv, data = server
    url = srv.media_url
    st, h, body = get(url, method="HEAD")
    assert st == 200 and int(h["Content-Length"]) == len(data) and body == b""
    etag = h["ETag"]
    st, _, body = get(url, {"Range": "bytes=0-3", "If-Match": etag})
    assert st == 206 and body == b"LVF1"
    st, h2, _ = get(url, {"Range": "bytes=0-3", "If-Match": '"stale"'})
    assert st == 412 and h2["ETag"] == etag
    st, _, _ = get(f"{srv.base}/media/other.lvd")
    assert st == 404


def test_the_player_is_served(server):
    srv, _ = server
    assert srv.page.startswith(srv.base + "/?src=/media/") and srv.page.endswith("&watch=1")
    st, h, body = get(srv.page)
    assert st == 200 and b"<html" in body.lower() and h["Content-Type"].startswith("text/html")
    script = re.search(rb'src="\./(assets/[^"]+\.js)"', body).group(1).decode()
    st, h, js = get(f"{srv.base}/{script}")
    assert st == 200 and h["Content-Type"].startswith("text/javascript") and len(js) > 10_000
    assert get(f"{srv.base}/../../etc/passwd")[0] == 404


def test_responses_stay_consistent_while_the_file_is_replaced(tmp_path):
    """Writers rename a new version over the file (spec B.11). Every response must carry the bytes
    of the same version as its ETag and Content-Range, or If-Match cannot detect the change."""
    versions = {0xAA: 300_000, 0xBB: 400_000}  # fill byte -> size
    media = tmp_path / "watched.lvd"
    media.write_bytes(bytes([0xAA]) * versions[0xAA])
    srv = Server(media)
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
    bad, checked, versions_of_etag = [], 0, {}
    try:
        for i in range(1500):
            st, h, body = get(srv.media_url, {"Range": f"bytes={(i * 7919) % 250_000}-{(i * 7919) % 250_000 + 4095}"})
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
        srv.stop()
    assert checked > 1000
    assert bad == [], f"{len(bad)} of {checked} responses mixed two versions, e.g. {bad[:3]}"
    shared = {etag: sorted(map(hex, b)) for etag, b in versions_of_etag.items() if len(b) > 1}
    assert not shared, f"one ETag was served with the bytes of different versions: {list(shared.items())[:3]}"


def test_requests_for_other_hosts_are_refused(server):
    srv, _ = server
    st, _, _ = get(srv.media_url, {"Host": "evil.example.com:80", "Range": "bytes=0-3"})
    assert st == 403
    st, _, body = get(srv.media_url, {"Host": "localhost", "Range": "bytes=0-3"})
    assert st == 206 and body == b"LVF1"
    st, h, body = get(srv.media_url, {"Range": "bytes=5-3"})  # not a valid range: the whole file
    assert st == 200 and len(body) == int(h["Content-Length"])


@pytest.mark.parametrize("args", [(), ("--no-open",)])
def test_the_page_is_not_opened_by_default(packed, args):
    srv = Server(packed["path"], *args)  # --no-open is still accepted
    threading.Event().wait(0.5)  # the page would be opened 200 ms after the server starts
    srv.proc.kill()
    out = srv.proc.stdout.read()
    srv.stop()
    assert "opened in" not in out


def test_open_and_no_open_conflict(packed):
    res = subprocess.run([sys.executable, "-m", "fflv", "view", str(packed["path"]), "--open", "--no-open"],
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 2 and "cannot be used with" in res.stderr


def test_view_from_python(packed, tmp_path):
    import signal

    import fflv

    with pytest.raises(fflv.ViewError, match="no such file"):
        fflv.view(tmp_path / "missing.lvd", open_page=False)
    urls = []

    def ready(url):
        urls.append(url)
        assert get(url)[0] == 200  # serving
        threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT)).start()

    with pytest.raises(KeyboardInterrupt):
        fflv.view(packed["path"], ready=ready)  # opens no browser by default
    assert urls and "/?src=/media/small.lvd" in urls[0]
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(urls[0], timeout=2)  # stopped


# ---- export ------------------------------------------------------------------------------------
W, H, N = 96, 64, 12


@pytest.fixture
def layered(tmp_path):
    path = tmp_path / "layered.lvd"
    with fflv.Writer(path, (W, H), gop=4, background="#102030") as w:
        w.add_layer("bg", lossless=True)
        w.add_layer("dot", alpha=True, lossless=True, rect=(10, 10, 16, 16))
        w.add_layer("hidden", visible=False, lossless=True, rect=(0, 0, 8, 8))
        for f in range(N):
            bg = np.zeros((H, W, 3), np.uint8)
            bg[:, :, 0] = f * 10
            dot = np.zeros((16, 16, 4), np.uint8)
            dot[4:12, 4:12] = (255, 255, 255, 255)
            w.write(bg=bg, dot=dot, hidden=np.full((8, 8, 3), 255, np.uint8))
    srv = Server(path)
    yield srv, path
    srv.stop()


def export(srv, request, tmp_path):
    """Run one export to the end; returns (final status, the downloaded file or None, its headers)."""
    st, _, body = post(f"{srv.base}/export", request)
    assert st == 202, body
    eid = json.loads(body)["id"]
    deadline = time.monotonic() + 60
    while True:
        st, _, body = get(f"{srv.base}/export/{eid}")
        status = json.loads(body)
        assert st == 200 and status["id"] == eid
        if status["state"] != "running":
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    if status["state"] != "done":
        return status, None, None
    st, h, data = get(f"{srv.base}/export/{eid}/file")
    assert st == 200 and int(h["Content-Length"]) == len(data)
    out = tmp_path / f"export-{eid}.{request.get('format', 'mp4')}"
    out.write_bytes(data)
    return status, out, h


def test_export_renders_the_layers_shown_at_their_opacities(layered, tmp_path):
    srv, path = layered
    st, _, body = get(f"{srv.base}/export")
    assert st == 200 and json.loads(body) == {"media": "/media/layered.lvd", "formats": ["mp4", "webm", "mov", "mkv"]}

    # a layer hidden in the file, shown in the player
    status, out, h = export(srv, {"format": "mkv", "layers": ["bg", "hidden"]}, tmp_path)
    assert status["done"] == status["total"] == N
    assert h["Content-Type"] == "video/x-matroska" and 'filename="layered.mkv"' in h["Content-Disposition"]
    fflv.render(path, tmp_path / "want.mkv", layers=["bg", "hidden"])
    got, want = read_video(out), read_video(tmp_path / "want.mkv")
    assert len(got) == N and all(np.array_equal(a, b) for a, b in zip(got, want))
    assert tuple(got[3][2, 2]) == (255, 255, 255)

    # opacity 0 is the same as hidden; 0.5 mixes with what is below
    _, out, _ = export(srv, {"format": "mkv", "layers": ["bg", "dot"], "opacity": {"dot": 0}}, tmp_path)
    fflv.render(path, tmp_path / "bg.mkv", layers=["bg"])
    assert all(np.array_equal(a, b) for a, b in zip(read_video(out), read_video(tmp_path / "bg.mkv")))
    _, out, _ = export(srv, {"format": "mkv", "layers": ["bg", "dot"], "opacity": {"dot": 0.5, "bg": 1}}, tmp_path)
    px = read_video(out)[4][18, 18].astype(int)  # the dot over bg (40, 0, 0)
    assert np.abs(px - (148, 128, 128)).max() <= 2

    # the draw order: bg (opaque, full canvas) moved over the dot covers it
    status, out, _ = export(srv, {"format": "mkv", "layers": ["bg", "dot"], "order": ["dot", "bg", "hidden"]}, tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(read_video(out), read_video(tmp_path / "bg.mkv")))

    # H.264 by default; the previous export's file is gone once a new one starts
    previous = f"{srv.base}/export/{status['id']}/file"
    assert get(previous)[0] == 200
    status, out, h = export(srv, {"layers": ["bg", "dot"]}, tmp_path)
    assert h["Content-Type"] == "video/mp4" and len(read_video(out)) == N
    assert get(previous)[0] == 404
    status, _, _ = export(srv, {"format": "mkv", "layers": ["bg"], "order": ["nope"]}, tmp_path)
    assert status["state"] == "failed" and "nope" in status["error"]


def test_export_errors_and_cancel(layered, tmp_path):
    srv, _ = layered
    url = f"{srv.base}/export"
    assert post(url, {"layers": []}, content_type="text/plain")[0] == 415  # no simple cross-site POSTs
    assert post(url, {"format": "gif", "layers": []})[0] == 400
    assert post(url, {"layers": "bg"})[0] == 400
    assert post(url, b"{")[0] == 400
    assert post(url, {"layers": [], "opacity": {"bg": 1.5}})[0] == 400
    assert get(url, {"Host": "evil.example.com"})[0] == 403
    assert get(f"{url}/99")[0] == 404 and get(f"{url}/99/file")[0] == 404
    status, out, _ = export(srv, {"layers": ["nope"]}, tmp_path)
    assert status["state"] == "failed" and "nope" in status["error"] and out is None

    st, _, body = post(url, {"layers": ["bg", "dot"], "format": "mkv"})
    eid = json.loads(body)["id"]
    assert post(url, {"layers": ["bg"]})[0] == 409  # one at a time
    assert post(f"{url}/{eid}/cancel", {})[0] == 200
    deadline = time.monotonic() + 30
    while (state := json.loads(get(f"{url}/{eid}")[2])["state"]) == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert state == "cancelled" and get(f"{url}/{eid}/file")[0] == 404
    assert post(url, {"layers": ["bg"]})[0] == 202  # free again
