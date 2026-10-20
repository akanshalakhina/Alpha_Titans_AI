"""Cloud Storage as the file store: the same protocol as Drive, different
addresses, md5 in base64, folders that are prefixes - and the control-plane
app that serves only the cloud routes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.parse

import pytest

from cloud import gcp
from cloud.config import CloudConfig
from cloud.gcs import GcsStore, check_object, normalise
from cloud.sanitize import CloudError

from fakes import API_TOKEN, FakeDrive, Tokens, cloud_env

BUCKET = "at-test-bucket"


def _b64md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()  # noqa: S324


class GcsHttp:
    """Speaks the GCS JSON API: resumable sessions, object metadata,
    ranged media."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.sessions: dict[str, dict] = {}
        self.ack_short = 0
        self.corrupt = False
        self.calls = []

    def _obj(self, name):
        data = self.objects[name]
        return {"name": name, "bucket": BUCKET, "size": str(len(data)),
                "md5Hash": (_b64md5(b"x") if self.corrupt
                            else _b64md5(data)),
                "contentType": "video/mp4"}

    def request(self, method, url, headers=None, body=None, timeout=None):
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.calls.append((method, url))
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if method == "POST" and "/upload/storage/v1/b/" in url:
            uri = f"https://storage.fake/s/{len(self.sessions)}?upload_id=X"
            self.sessions[uri] = {"name": q["name"], "buf": bytearray(),
                                  "size": int(headers[
                                      "x-upload-content-length"])}
            return gcp.Response(200, {"location": uri}, b"{}")
        if method == "PUT" and url in self.sessions:
            s = self.sessions[url]
            rng = headers["content-range"]
            if not rng.startswith("bytes */"):
                start = int(rng.split(" ")[1].split("-")[0])
                if start == len(s["buf"]):
                    data = body
                    if self.ack_short and len(data) > gcp.CHUNK_ALIGN:
                        self.ack_short -= 1
                        data = data[:gcp.CHUNK_ALIGN]
                    s["buf"] += data
            if len(s["buf"]) >= s["size"]:
                self.objects[s["name"]] = bytes(s["buf"])
                return gcp.Response(200, {}, json.dumps(
                    self._obj(s["name"])).encode())
            h = ({"range": f"bytes=0-{len(s['buf']) - 1}"} if s["buf"]
                 else {})
            return gcp.Response(308, h, b"")
        if method == "GET" and "/storage/v1/b/" in url:
            name = urllib.parse.unquote(u.path.split("/o/", 1)[1])
            if name not in self.objects:
                return gcp.Response(404, {}, b'{"error":"not found"}')
            return gcp.Response(200, {}, json.dumps(self._obj(name)).encode())
        return gcp.Response(400, {}, b"{}")

    def stream(self, method, url, headers=None, timeout=None):
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        name = urllib.parse.unquote(
            urllib.parse.urlparse(url).path.split("/o/", 1)[1])
        data = self.objects[name]
        start, status = 0, 200
        if "range" in headers:
            start = int(headers["range"].split("=")[1].rstrip("-"))
            status = 206
        return status, {"content-type": "video/mp4"}, io.BytesIO(data[start:])


@pytest.fixture
def h():
    return GcsHttp()


@pytest.fixture
def store(h):
    return GcsStore(h, Tokens(), BUCKET, sleep=lambda s: None)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(gcp, "CHUNK", 2 * gcp.CHUNK_ALIGN)


def test_upload_is_chunked_verified_and_named_by_prefix(store, h, tmp_path,
                                                        small):
    p = tmp_path / "v.mp4"
    p.write_bytes(os.urandom(1_300_000))
    meta = store.upload(p, "v.mp4", "video/mp4", "outputs/abcdef123456")
    assert meta["id"] == "outputs/abcdef123456/v.mp4"
    assert h.objects[meta["id"]] == p.read_bytes()
    assert meta["md5Checksum"] == hashlib.md5(p.read_bytes()).hexdigest()


def test_a_short_acknowledgement_is_believed(store, h, tmp_path, small):
    h.ack_short = 2
    p = tmp_path / "v.mp4"
    p.write_bytes(os.urandom(1_300_000))
    meta = store.upload(p, "v.mp4", "video/mp4", "outputs/x")
    assert h.objects[meta["id"]] == p.read_bytes()


def test_a_bad_md5_from_storage_fails_verification(store, h, tmp_path):
    h.corrupt = True
    p = tmp_path / "v.mp4"
    p.write_bytes(os.urandom(5000))
    with pytest.raises(CloudError) as e:
        store.upload(p, "v.mp4", "video/mp4", "outputs/x")
    assert e.value.category == "verification"


def test_download_resumes_and_verifies(store, h, tmp_path):
    data = os.urandom(2_000_000)
    h.objects["inputs/j_01_a.mp4"] = data
    (tmp_path / "a.mp4.part").write_bytes(data[:500_000])
    store.download("inputs/j_01_a.mp4", tmp_path / "a.mp4", len(data),
                   hashlib.md5(data).hexdigest())
    assert (tmp_path / "a.mp4").read_bytes() == data


def test_find_returns_nothing_for_a_missing_object(store):
    assert store.find("outputs/x", "nope.mp4") == []


def test_folders_are_prefixes(store, h):
    assert store.create_folder("abcdef123456", "outputs") == \
        "outputs/abcdef123456"
    assert h.calls == [], "no API call for a prefix"


@pytest.mark.parametrize("bad", ["../x", "a/../b", "/abs", "a//b", "a/",
                                 "", "x\x00y"])
def test_object_names_are_ours(bad):
    with pytest.raises(CloudError):
        check_object(bad)


def test_md5_is_turned_into_hex():
    m = normalise({"name": "p/q.mp4", "size": "3",
                   "md5Hash": _b64md5(b"abc")})
    assert m["md5Checksum"] == hashlib.md5(b"abc").hexdigest()
    assert m["parents"] == ["p"] and m["name"] == "q.mp4"


def test_a_private_bucket_offers_no_link(store):
    assert store.link("outputs/x/v.mp4") is None


# --------------------------------------------------------------- config

def test_gcs_needs_a_bucket_not_drive_folders(tmp_path):
    env = cloud_env(tmp_path, AT_CLOUD_STORAGE="gcs")
    for k in ("AT_DRIVE_INPUT_FOLDER_ID", "AT_DRIVE_OUTPUT_FOLDER_ID"):
        env.pop(k)
    problems = CloudConfig.from_env(env).problems("control")
    assert any("AT_GCS_BUCKET" in p for p in problems)
    assert not any("DRIVE" in p for p in problems)
    env["AT_GCS_BUCKET"] = BUCKET
    cfg = CloudConfig.from_env(env)
    assert cfg.problems("control") == []
    assert (cfg.drive_input_folder, cfg.drive_output_folder) == \
        ("inputs", "outputs")


def test_gcs_allows_relay_uploads_only(tmp_path):
    env = cloud_env(tmp_path, AT_CLOUD_STORAGE="gcs", AT_GCS_BUCKET=BUCKET,
                    AT_CLOUD_UPLOAD_MODE="direct")
    assert any("relay" in p for p in
               CloudConfig.from_env(env).problems("control"))


# --------------------------------------- outputs through the control plane

class PrivateStore(FakeDrive):
    """FakeDrive without public links, as a private bucket behaves."""

    def link(self, file_id):
        return None

    def open_media(self, file_id, rng=None):
        data = self.files[file_id]["data"]
        if rng:
            a, _, b = rng[len("bytes="):].partition("-")
            a = int(a or 0)
            b = int(b) if b else len(data) - 1
            part = data[a:b + 1]
            return (206, {"content-type": "video/mp4",
                          "content-length": str(len(part)),
                          "content-range": f"bytes {a}-{b}/{len(data)}"},
                    io.BytesIO(part))
        return 200, {"content-type": "video/mp4",
                     "content-length": str(len(data))}, io.BytesIO(data)


def test_private_outputs_are_streamed_to_an_authenticated_browser(tmp_path,
                                                                  clock,
                                                                  compute):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cloud.app import mount
    from test_queue import queued

    store = PrivateStore()
    app = FastAPI()
    ctx = mount(app, host="127.0.0.1", env=cloud_env(tmp_path), drive=store,
                compute=compute, clock=clock)
    jid = queued(ctx.queue)
    c = ctx.queue.claim("w")
    ctx.queue.heartbeat(jid, "w", c.claim_token, "processing")
    ctx.queue.heartbeat(jid, "w", c.claim_token, "uploading_results")
    meta = store.add(b"VIDEOBYTES" * 100, "v.mp4", "outputs")
    ctx.queue.complete(jid, "w", c.claim_token, [{
        "kind": "video", "name": "v.mp4", "size": 1000, "md5": "a" * 32,
        "drive_file_id": meta["id"], "verified": True}])
    auth = {"Authorization": f"Bearer {API_TOKEN}"}
    with TestClient(app) as client:
        view = client.get(f"/api/cloud/jobs/{jid}", headers=auth).json()
        link = view["outputs"][0]["link"]
        assert link == f"/api/cloud/jobs/{jid}/outputs/0"
        assert client.get(link).status_code == 401, "token required"
        r = client.get(link, headers=auth)
        assert r.status_code == 200 and r.content == b"VIDEOBYTES" * 100
        assert "attachment" in r.headers["content-disposition"]
        assert client.get(f"/api/cloud/jobs/{jid}/outputs/5",
                          headers=auth).status_code == 404


# ------------------------------------------------------ the control app

def test_the_control_app_serves_only_the_cloud(tmp_path, compute, drive,
                                               clock):
    from fastapi.testclient import TestClient

    from cloud.control_app import build

    app = build(env=cloud_env(tmp_path), drive=drive, compute=compute,
                clock=clock)
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"ok": True, "cloud": True}
        assert c.get("/cloud").status_code == 200
        assert c.get("/static/cloud.js").status_code == 200
        assert c.get("/api/cloud/config").json()["enabled"] is True
        for legacy in ("/api/jobs", "/api/v2/jobs", "/static/app.js",
                       "/static/studio.js", "/docs", "/openapi.json",
                       "/static/..%2F..%2F.env"):
            assert c.get(legacy).status_code == 404, legacy
        assert c.post("/api/jobs", json={}).status_code in (404, 405)


# ------------------------------------------------- download tickets

def _finished_private_job(tmp_path, clock, compute):
    from fastapi import FastAPI

    from cloud.app import mount
    from test_queue import queued

    store = PrivateStore()
    app = FastAPI()
    ctx = mount(app, host="127.0.0.1", env=cloud_env(tmp_path), drive=store,
                compute=compute, clock=clock)
    jid = queued(ctx.queue)
    c = ctx.queue.claim("w")
    ctx.queue.heartbeat(jid, "w", c.claim_token, "processing")
    ctx.queue.heartbeat(jid, "w", c.claim_token, "uploading_results")
    meta = store.add(b"V" * 4096, "v.mp4", "outputs")
    ctx.queue.complete(jid, "w", c.claim_token, [{
        "kind": "video", "name": "v.mp4", "size": 4096, "md5": "a" * 32,
        "drive_file_id": meta["id"], "verified": True}])
    return app, jid


def test_a_browser_download_ticket_works_once(tmp_path, clock, compute):
    """The browser's own download manager fetches the file, so no token
    can ride along; a one-time ticket stands in for it."""
    from fastapi.testclient import TestClient
    app, jid = _finished_private_job(tmp_path, clock, compute)
    auth = {"Authorization": f"Bearer {API_TOKEN}"}
    with TestClient(app) as c:
        assert c.post(f"/api/cloud/jobs/{jid}/outputs/0/ticket"
                      ).status_code == 401, "minting needs the token"
        t = c.post(f"/api/cloud/jobs/{jid}/outputs/0/ticket",
                   headers=auth).json()
        assert t["url"].startswith("/api/cloud/download/")
        assert API_TOKEN not in t["url"]
        r = c.get(t["url"])
        assert r.status_code == 200 and r.content == b"V" * 4096
        assert "attachment" in r.headers["content-disposition"]
        assert c.get(t["url"]).status_code == 404, "single use"


def test_a_ticket_expires(tmp_path, clock, compute):
    from fastapi.testclient import TestClient
    app, jid = _finished_private_job(tmp_path, clock, compute)
    auth = {"Authorization": f"Bearer {API_TOKEN}"}
    with TestClient(app) as c:
        t = c.post(f"/api/cloud/jobs/{jid}/outputs/0/ticket",
                   headers=auth).json()
        clock.advance(61)
        assert c.get(t["url"]).status_code == 404


def test_a_made_up_ticket_gets_nothing(tmp_path, clock, compute):
    from fastapi.testclient import TestClient
    app, _jid = _finished_private_job(tmp_path, clock, compute)
    with TestClient(app) as c:
        assert c.get("/api/cloud/download/not-a-real-ticket"
                     ).status_code == 404


def test_no_ticket_for_an_output_that_does_not_exist(tmp_path, clock,
                                                     compute):
    from fastapi.testclient import TestClient
    app, jid = _finished_private_job(tmp_path, clock, compute)
    with TestClient(app) as c:
        r = c.post(f"/api/cloud/jobs/{jid}/outputs/9/ticket",
                   headers={"Authorization": f"Bearer {API_TOKEN}"})
        assert r.status_code == 404


def test_a_view_link_plays_seeks_and_can_be_reused(tmp_path, clock, compute):
    """The page's own <video>: inline, reusable while it lives, and a
    Range request gets the slice it asked for, so the player can seek."""
    from fastapi.testclient import TestClient
    app, jid = _finished_private_job(tmp_path, clock, compute)
    auth = {"Authorization": f"Bearer {API_TOKEN}"}
    with TestClient(app) as c:
        t = c.post(f"/api/cloud/jobs/{jid}/outputs/0/ticket?mode=view",
                   headers=auth).json()
        assert t["expires_in"] == 30 * 60
        whole = c.get(t["url"])
        assert whole.status_code == 200 and len(whole.content) == 4096
        assert whole.headers["content-disposition"].startswith("inline")
        part = c.get(t["url"], headers={"Range": "bytes=100-199"})
        assert part.status_code == 206 and len(part.content) == 100
        assert part.headers["content-range"] == "bytes 100-199/4096"
        assert c.get(t["url"]).status_code == 200, "reusable while it lives"
        clock.advance(30 * 60 + 1)
        assert c.get(t["url"]).status_code == 404, "but not for ever"


def test_a_download_link_ignores_range_and_stays_single_use(tmp_path, clock,
                                                           compute):
    from fastapi.testclient import TestClient
    app, jid = _finished_private_job(tmp_path, clock, compute)
    auth = {"Authorization": f"Bearer {API_TOKEN}"}
    with TestClient(app) as c:
        t = c.post(f"/api/cloud/jobs/{jid}/outputs/0/ticket",
                   headers=auth).json()
        r = c.get(t["url"], headers={"Range": "bytes=0-9"})
        assert r.status_code == 200 and len(r.content) == 4096
        assert c.get(t["url"]).status_code == 404


def test_a_bad_mode_is_refused(tmp_path, clock, compute):
    from fastapi.testclient import TestClient
    app, jid = _finished_private_job(tmp_path, clock, compute)
    with TestClient(app) as c:
        r = c.post(f"/api/cloud/jobs/{jid}/outputs/0/ticket?mode=forever",
                   headers={"Authorization": f"Bearer {API_TOKEN}"})
        assert r.status_code == 400


def test_the_options_are_the_pipelines_own(tmp_path, clock, compute, drive):
    """The cloud Create form offers exactly what JobSettings accepts - the
    same derivation as the studio's /api/v2/produce/options."""
    import typing

    from fastapi.testclient import TestClient

    from agents.artifacts import JobSettings
    from cloud.control_app import build
    app = build(env=cloud_env(tmp_path), drive=drive, compute=compute,
                clock=clock)
    with TestClient(app) as c:
        o = c.get("/api/cloud/options").json()
    for k in ("style", "music", "orientation", "framing"):
        ann = JobSettings.model_fields[k].annotation
        assert o[k] == [a for a in typing.get_args(ann)
                        if isinstance(a, str)], k
