"""The `fflv view` server: HTTP range requests, ETags, If-Match, the bundled player."""

import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest


class Server:
    def __init__(self, media):
        self.proc = subprocess.Popen([sys.executable, "-m", "fflv", "view", str(media), "--no-open"],
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


def get(url, headers=None, method="GET"):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, dict(res.headers), res.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


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
        fflv.view(packed["path"], open_page=False, ready=ready)
    assert urls and "/?src=/media/small.lvd" in urls[0]
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(urls[0], timeout=2)  # stopped
