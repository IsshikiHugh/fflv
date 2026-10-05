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
from pathlib import Path

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


def free_port():
    """A port free on every address: listening on 0.0.0.0 at a port another process holds on
    127.0.0.1 (an ssh tunnel, say) succeeds, but requests to it then reach that process."""
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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


def test_request_bodies_are_limited(server):
    srv, _ = server
    assert get(srv.media_url, {"Range": "bytes=0-3"}, data=b"x" * 2000)[0] == 413  # unexpected body
    assert get(srv.media_url, {"Range": "bytes=0-3"}, data=b"x" * 100)[0] == 206  # small: ignored
    assert post(f"{srv.base}/export", b" " * ((1 << 20) + 1))[0] == 413


def test_non_loopback_host_warns_and_still_checks_the_host(packed):
    srv = Server(packed["path"], "--host", "0.0.0.0", "--port", str(free_port()))
    try:
        assert get(srv.media_url, {"Range": "bytes=0-3"})[0] == 206  # Host: an IP address
        assert get(srv.media_url, {"Host": "evil.example.com", "Range": "bytes=0-3"})[0] == 403
    finally:
        srv.proc.kill()
        err = srv.proc.stderr.read()
        srv.stop()
    assert "not a loopback address" in err


@pytest.mark.parametrize("args", [(), ("--no-open",)])
def test_the_page_is_not_opened_by_default(packed, args):
    srv = Server(packed["path"], *args)  # --no-open is still accepted
    threading.Event().wait(0.5)  # the page would be opened 200 ms after the server starts
    srv.proc.kill()
    out = srv.proc.stdout.read()
    srv.stop()
    assert "opened in" not in out


def test_the_default_port_moves_on_at_most_ten_ports(packed):
    """8765, else the next free port up to 8775; a port another process holds on 127.0.0.1 is
    taken even for a server on 0.0.0.0 (binding there would succeed, but requests would not
    reach it)."""
    import socket

    held = {}
    for port in range(8765, 8776):
        s = socket.socket()
        # like a server: a port left in TIME_WAIT by an earlier test can be listened on again
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            s.listen()
            held[port] = s
        except OSError:
            s.close()  # in use already
    try:
        res = subprocess.run([sys.executable, "-m", "fflv", "view", str(packed["path"]), "--host", "0.0.0.0"],
                             capture_output=True, text=True, timeout=30)
        assert res.returncode != 0 and "8765–8775 are all in use" in res.stderr
        if not held:
            pytest.skip("every port from 8765 to 8775 is used by other processes")
        port = max(held)
        held.pop(port).close()
        srv = Server(packed["path"], "--host", "0.0.0.0")
        try:
            assert srv.base.endswith(f":{port}")
        finally:
            srv.stop()
    finally:
        for s in held.values():
            s.close()


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


def wait_export(srv, eid, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        st, _, body = get(f"{srv.base}/export/{eid}")
        status = json.loads(body)
        assert st == 200 and status["id"] == eid
        if status["state"] != "running":
            return status
        assert time.monotonic() < deadline, status
        time.sleep(0.05)


def export(srv, request, tmp_path):
    """Run one export to the end (into tmp_path/out unless the request names a path); returns
    (final status, the file written or None)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    request = {"path": str(out_dir / f"e{len(list(out_dir.iterdir()))}.{request.get('format', 'mp4')}"), **request}
    st, _, body = post(f"{srv.base}/export", request)
    assert st == 202, body
    started = json.loads(body)
    status = wait_export(srv, started["id"])
    assert status["path"] == started["path"]
    dest = Path(started["path"])
    assert not [p.name for p in dest.parent.iterdir() if "fflv-tmp" in p.name]  # no partial file left
    if status["state"] != "done":
        assert not dest.exists() or request.get("overwrite")
        return status, None
    return status, dest


def test_export_renders_the_layers_shown_at_their_opacities(layered, tmp_path):
    srv, path = layered
    st, _, body = get(f"{srv.base}/export")
    assert st == 200 and json.loads(body) == {"media": "/media/layered.lvd", "formats": ["mp4", "webm", "mov", "mkv"],
                                             "dir": str(path.parent), "stem": "layered", "sep": os.sep}

    # a layer hidden in the file, shown in the player
    status, out = export(srv, {"format": "mkv", "layers": ["bg", "hidden"]}, tmp_path)
    assert status["done"] == status["total"] == N
    fflv.render(path, tmp_path / "want.mkv", layers=["bg", "hidden"])
    got, want = read_video(out), read_video(tmp_path / "want.mkv")
    assert len(got) == N and all(np.array_equal(a, b) for a, b in zip(got, want))
    assert tuple(got[3][2, 2]) == (255, 255, 255)
    # the file can also be downloaded through the browser
    st, h, data = get(f"{srv.base}/export/{status['id']}/file")
    assert st == 200 and data == out.read_bytes() and h["Content-Type"] == "video/x-matroska"
    assert f'filename="{out.name}"' in h["Content-Disposition"]

    # opacity 0 is the same as hidden; 0.5 mixes with what is below
    _, out = export(srv, {"format": "mkv", "layers": ["bg", "dot"], "opacity": {"dot": 0}}, tmp_path)
    fflv.render(path, tmp_path / "bg.mkv", layers=["bg"])
    assert all(np.array_equal(a, b) for a, b in zip(read_video(out), read_video(tmp_path / "bg.mkv")))
    _, out = export(srv, {"format": "mkv", "layers": ["bg", "dot"], "opacity": {"dot": 0.5, "bg": 1}}, tmp_path)
    px = read_video(out)[4][18, 18].astype(int)  # the dot over bg (40, 0, 0)
    assert np.abs(px - (148, 128, 128)).max() <= 2

    # the draw order: bg (opaque, full canvas) moved over the dot covers it
    _, out = export(srv, {"format": "mkv", "layers": ["bg", "dot"], "order": ["dot", "bg", "hidden"]}, tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(read_video(out), read_video(tmp_path / "bg.mkv")))

    # a range of frames, [start, end)
    status, out = export(srv, {"format": "mkv", "layers": ["bg"], "start": 3, "end": 7}, tmp_path)
    assert status["total"] == 4
    assert all(np.array_equal(a, b) for a, b in zip(read_video(out), read_video(tmp_path / "bg.mkv")[3:7], strict=True))
    status, _ = export(srv, {"layers": ["bg"], "start": 5, "end": N + 1}, tmp_path)
    assert status["state"] == "failed"

    # H.264 by default; the path's extension picks the format
    _, out = export(srv, {"layers": ["bg", "dot"]}, tmp_path)
    assert out.suffix == ".mp4" and len(read_video(out)) == N
    _, out = export(srv, {"format": "mp4", "layers": ["bg"], "path": str(tmp_path / "out" / "x.webm")}, tmp_path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "vp9"
    status, _ = export(srv, {"format": "mkv", "layers": ["bg"], "order": ["nope"]}, tmp_path)
    assert status["state"] == "failed" and "nope" in status["error"]


def test_export_paths(layered, tmp_path):
    srv, path = layered
    url = f"{srv.base}/export"
    # no path: <stem>.<format> next to the .lvd; an existing file is only replaced when asked
    st, _, body = post(url, {"layers": ["bg"], "format": "mkv"})
    started = json.loads(body)
    assert st == 202 and started["path"] == str(path.with_suffix(".mkv"))
    assert wait_export(srv, started["id"])["state"] == "done"
    first = path.with_suffix(".mkv").read_bytes()
    st, _, body = post(url, {"layers": ["bg", "dot"], "format": "mkv"})
    assert st == 409 and json.loads(body)["exists"] is True and json.loads(body)["path"] == str(path.with_suffix(".mkv"))
    assert path.with_suffix(".mkv").read_bytes() == first
    st, _, body = post(url, {"layers": ["bg", "dot"], "format": "mkv", "overwrite": True})
    assert st == 202 and wait_export(srv, json.loads(body)["id"])["state"] == "done"
    assert path.with_suffix(".mkv").read_bytes() != first

    # relative to the .lvd's folder, which must exist
    assert post(url, {"layers": ["bg"], "path": "clips/a.mp4"})[0] == 400
    (tmp_path / "clips").mkdir()
    st, _, body = post(url, {"layers": ["bg"], "path": "clips/a.mp4"})
    assert st == 202 and json.loads(body)["path"] == str(tmp_path / "clips" / "a.mp4")
    assert wait_export(srv, json.loads(body)["id"])["state"] == "done" and (tmp_path / "clips" / "a.mp4").stat().st_size

    st, _, body = post(url, {"layers": ["bg"], "path": "clips/a.gif"})
    assert st == 400 and ".mp4" in json.loads(body)["error"]
    assert post(url, {"layers": ["bg"], "path": "clips/../b.mp4"})[0] == 400
    assert post(url, {"layers": ["bg"], "path": "clips"})[0] == 400
    assert post(url, {"layers": ["bg"], "path": str(tmp_path / "clips") + "/"})[0] == 400


def test_export_cancel_leaves_nothing(layered, tmp_path):
    srv, _ = layered
    url = f"{srv.base}/export"
    dest = tmp_path / "cancelled.mkv"
    st, _, body = post(url, {"layers": ["bg", "dot"], "path": str(dest)})
    eid = json.loads(body)["id"]
    assert post(url, {"layers": ["bg"], "path": str(tmp_path / "other.mp4")})[0] == 409  # one at a time
    assert post(f"{url}/{eid}/cancel", {}, content_type="text/plain")[0] == 415  # no cross-site cancel
    assert post(f"{url}/{eid}/cancel", {})[0] == 200
    assert wait_export(srv, eid, 30)["state"] == "cancelled"
    assert get(f"{url}/{eid}/file")[0] == 404
    assert not dest.exists() and not [p for p in tmp_path.iterdir() if "fflv-tmp" in p.name]
    assert post(url, {"layers": ["bg"], "path": str(tmp_path / "other.mp4")})[0] == 202  # free again


def test_export_on_a_network_address_stays_in_the_folder(tmp_path, layered):
    _, path = layered
    srv = Server(path, "--host", "0.0.0.0", "--port", str(free_port()))
    try:
        base = srv.base.replace("0.0.0.0", "127.0.0.1")
        st, _, body = post(f"{base}/export", {"layers": ["bg"], "path": str(tmp_path.parent / "outside.mp4")})
        assert st == 403 and "network" in json.loads(body)["error"]
        st, _, body = post(f"{base}/export", {"layers": ["bg"], "path": "inside.mp4"})
        assert st == 202
    finally:
        srv.stop()


def test_export_errors(layered, tmp_path):
    srv, _ = layered
    url = f"{srv.base}/export"
    assert post(url, {"layers": []}, content_type="text/plain")[0] == 415  # no simple cross-site POSTs
    assert post(url, {"format": "gif", "layers": []})[0] == 400
    assert post(url, {"layers": "bg"})[0] == 400
    assert post(url, b"{")[0] == 400
    assert post(url, {"layers": [], "opacity": {"bg": 1.5}})[0] == 400
    assert post(url, {"layers": [], "start": 4, "end": 4})[0] == 400
    assert get(url, {"Host": "evil.example.com"})[0] == 403
    assert get(f"{url}/99")[0] == 404 and get(f"{url}/99/file")[0] == 404
    status, out = export(srv, {"layers": ["nope"]}, tmp_path)
    assert status["state"] == "failed" and "nope" in status["error"] and out is None
