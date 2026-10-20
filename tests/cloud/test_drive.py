"""The real Drive client, against a fake that speaks Drive's protocol.

What matters is not that bytes are sent but that the client believes only
what the SERVER acknowledges: a 308 whose Range says less arrived than was
sent, a 503 mid-upload, a download that stops short or a server that ignores
Range. And a transfer that cannot be verified is never treated as done.
"""

from __future__ import annotations

import hashlib
import http.server
import os
import threading

import pytest

from cloud import gcp
from cloud.sanitize import CloudError

from fakes import DriveHttp, Tokens


@pytest.fixture
def small_chunks(monkeypatch):
    """512 KiB chunks so a 1.3 MB file takes several round trips."""
    monkeypatch.setattr(gcp, "CHUNK", 2 * gcp.CHUNK_ALIGN)


@pytest.fixture
def h():
    return DriveHttp()


@pytest.fixture
def drive(h):
    return gcp.Drive(h, Tokens(), shared_drive=True, sleep=lambda s: None)


def _file(tmp_path, n=1_300_000, name="v.mp4"):
    p = tmp_path / name
    p.write_bytes(os.urandom(n))
    return p


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()                  # noqa: S324


# ------------------------------------------------------------------ upload

def test_an_upload_arrives_whole_and_verified(drive, h, tmp_path,
                                              small_chunks):
    p = _file(tmp_path)
    meta = drive.upload(p, "v.mp4", "video/mp4", "folder")
    stored = h.files[meta["id"]]["data"]
    assert stored == p.read_bytes()
    puts = [c for c in h.calls if c[0] == "PUT"]
    assert len(puts) == 3, "1.3 MB in 512 KiB chunks"


def test_every_non_final_chunk_is_256k_aligned(drive, h, tmp_path,
                                               small_chunks, monkeypatch):
    sizes = []
    real = drive.put_chunk

    def spy(uri, data, offset, total):
        sizes.append(len(data))
        return real(uri, data, offset, total)

    monkeypatch.setattr(drive, "put_chunk", spy)
    drive.upload(_file(tmp_path), "v.mp4", "video/mp4", "folder")
    assert all(s % gcp.CHUNK_ALIGN == 0 for s in sizes[:-1])


def test_a_short_acknowledgement_is_believed(drive, h, tmp_path,
                                             small_chunks):
    """The server said it holds less than was sent. The client resends
    from where the SERVER says, never from where it hoped."""
    h.ack_short = 2
    p = _file(tmp_path)
    meta = drive.upload(p, "v.mp4", "video/mp4", "folder")
    assert h.files[meta["id"]]["data"] == p.read_bytes()


def test_a_503_mid_upload_is_resumed_not_restarted(drive, h, tmp_path,
                                                   small_chunks):
    h.fail_puts = 1
    p = _file(tmp_path)
    meta = drive.upload(p, "v.mp4", "video/mp4", "folder")
    assert h.files[meta["id"]]["data"] == p.read_bytes()
    inits = [c for c in h.calls if c[0] == "POST"]
    assert len(inits) == 1, "the same session, resumed"


def test_an_upload_drive_cannot_vouch_for_is_trashed_and_refused(
        drive, h, tmp_path, small_chunks):
    h.corrupt_md5 = True
    with pytest.raises(CloudError) as e:
        drive.upload(_file(tmp_path), "v.mp4", "video/mp4", "folder")
    assert e.value.category == "verification"
    assert all(f["trashed"] for f in h.files.values())


def test_an_unaligned_middle_chunk_is_refused_before_it_is_sent(drive):
    with pytest.raises(CloudError):
        drive.put_chunk("https://up.fake/s/1", b"x" * 1000, 0, 10_000)


def test_an_empty_file_is_refused(drive, tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    with pytest.raises(CloudError):
        drive.upload(p, "empty.mp4", "video/mp4", "folder")


# ---------------------------------------------------------------- download

def _stored(h, data: bytes) -> str:
    h.files["fileid00000001"] = {"data": data, "name": "s.mp4",
                                 "trashed": False}
    return "fileid00000001"


def test_a_download_is_verified_then_moved_into_place(drive, h, tmp_path):
    data = os.urandom(3_000_000)
    fid = _stored(h, data)
    dest = tmp_path / "01_s.mp4"
    drive.download(fid, dest, len(data), _md5(data))
    assert dest.read_bytes() == data
    assert not (tmp_path / "01_s.mp4.part").exists()


def test_a_partial_download_resumes_with_range(drive, h, tmp_path):
    data = os.urandom(3_000_000)
    fid = _stored(h, data)
    dest = tmp_path / "01_s.mp4"
    (tmp_path / "01_s.mp4.part").write_bytes(data[:1_000_000])
    drive.download(fid, dest, len(data), _md5(data))
    assert dest.read_bytes() == data


def test_a_server_that_ignores_range_restarts_cleanly(drive, h, tmp_path):
    data = os.urandom(2_000_000)
    fid = _stored(h, data)
    h.ignore_range = True
    (tmp_path / "s.mp4.part").write_bytes(data[:700_000])
    drive.download(fid, tmp_path / "s.mp4", len(data), _md5(data))
    assert (tmp_path / "s.mp4").read_bytes() == data


def test_a_stream_that_stops_short_is_continued(drive, h, tmp_path):
    data = os.urandom(2_000_000)
    fid = _stored(h, data)
    h.cut_stream_at = 500_000
    drive.download(fid, tmp_path / "s.mp4", len(data), _md5(data))
    assert (tmp_path / "s.mp4").read_bytes() == data


def test_a_corrupt_download_never_reaches_its_destination(drive, h,
                                                          tmp_path):
    data = os.urandom(1_000_000)
    fid = _stored(h, data)
    with pytest.raises(CloudError) as e:
        drive.download(fid, tmp_path / "s.mp4", len(data), "0" * 32)
    assert e.value.category == "verification"
    assert not (tmp_path / "s.mp4").exists()
    assert not (tmp_path / "s.mp4.part").exists(), "partial removed"


def test_an_already_verified_file_is_not_fetched_again(drive, h, tmp_path):
    data = os.urandom(500_000)
    fid = _stored(h, data)
    (tmp_path / "s.mp4").write_bytes(data)
    n = len(h.calls)
    drive.download(fid, tmp_path / "s.mp4", len(data), _md5(data))
    assert len(h.calls) == n


def test_a_malformed_file_id_never_reaches_a_url(drive):
    with pytest.raises(CloudError):
        drive.get("../../etc/passwd")


# ------------------------------------------------- the http layer itself

class _Says308(http.server.BaseHTTPRequestHandler):
    def do_PUT(self):
        # read the body first: closing a socket with unread bytes makes
        # Windows send a reset, and the client's read of this reply then
        # fails with WinError 10053 - a flaky test, not a client fault
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(308)
        self.send_header("Range", "bytes=0-262143")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):                        # quiet
        pass


def test_a_308_resume_incomplete_is_returned_not_followed():
    """urllib follows 308 as a redirect since 3.11. Drive's 308 is not a
    redirect, and following it would lose the Range that says how much
    arrived. Proven against a real socket, not a mock."""
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Says308)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        r = gcp.Http(timeout=5).request(
            "PUT", f"http://127.0.0.1:{srv.server_port}/s", {}, b"x")
        assert r.status == 308
        assert r.headers["range"] == "bytes=0-262143"
    finally:
        srv.shutdown()


# ----------------------------------------------------------------- compute

def test_compute_refuses_anything_but_a_plain_resource_name():
    for bad in ("../x", "inst; rm -rf /", "Inst", "a" * 70, ""):
        with pytest.raises(CloudError):
            gcp.Compute(DriveHttp(), Tokens(), "test-project-123",
                        "us-central1-a", bad)


def test_compute_only_ever_calls_its_three_urls():
    class Rec:
        def __init__(self):
            self.urls = []

        def request(self, method, url, headers=None, body=None,
                    timeout=None):
            self.urls.append((method, url))
            return gcp.Response(200, {}, b'{"status": "RUNNING"}')

    rec = Rec()
    c = gcp.Compute(rec, Tokens(), "test-project-123", "us-central1-a",
                    "at-worker")
    c.status()
    c.start()
    c.stop()
    base = ("https://compute.googleapis.com/compute/v1/projects/"
            "test-project-123/zones/us-central1-a/instances/at-worker")
    assert rec.urls == [("GET", base), ("POST", base + "/start"),
                        ("POST", base + "/stop")]
