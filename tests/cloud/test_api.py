"""The /api/cloud routes, over the real router and the fakes.

The upload is driven exactly as the browser drives it: create, then chunks
with Content-Range, then complete. Progress is what the SERVER says it
holds.
"""

from __future__ import annotations

import os

import pytest

from cloud import queue as Q

from fakes import API_TOKEN, WORKER_TOKEN, cloud_env

CHUNK = 256 * 1024


def create(client, auth, files, key="intent-00000001", settings=None):
    r = client.post("/api/cloud/uploads", headers=auth, json={
        "idempotency_key": key, "files": files,
        "settings": settings or {"title": "Test", "target_seconds": 30}})
    assert r.status_code == 200, r.text
    return r.json()


def send(client, auth, url, data: bytes, chunk=CHUNK, start=0):
    """The browser's loop: PUT chunks, trust the acknowledged offset."""
    total, off, seen = len(data), start, []
    while True:
        end = min(off + chunk, total) - 1
        r = client.put(url, headers={**auth, "Content-Range":
                                     f"bytes {off}-{end}/{total}"},
                       content=data[off:end + 1])
        assert r.status_code in (200, 409), r.text
        body = r.json()
        seen.append(body["received"])
        if body["complete"]:
            return seen
        off = body["received"]


@pytest.fixture
def clip():
    return os.urandom(3 * CHUNK + 1234)


def spec(data, name="clip.mp4"):
    return {"name": name, "size": len(data), "mime": "video/mp4"}


# ----------------------------------------------------- the whole upload

def test_upload_to_queue_end_to_end(plane, auth, clip, drive, compute):
    client, ctx = plane
    made = create(client, auth, [spec(clip)])
    job_id = made["job"]["job_id"]
    assert made["job"]["state"] == Q.UPLOADING
    target = made["files"][0]
    assert target["mode"] == "relay" and "session_uri" not in target

    progress = send(client, auth, target["url"], clip)
    assert progress == sorted(progress), "progress only moves forward"
    assert progress[-1] == len(clip)

    r = client.post(f"/api/cloud/uploads/{job_id}/complete", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job"]["state"] == Q.STARTING_VM
    assert body["worker_start"] == "started" and compute.starts() == 1

    src = ctx.queue.get(job_id)["sources"][0]
    stored = drive.files[src["drive_file_id"]]
    assert stored["data"] == clip, "Drive holds exactly the bytes sent"
    assert src["md5"] == stored["md5Checksum"]


def test_progress_is_what_the_server_acknowledged(plane, auth, clip, drive):
    """Drive held back part of a chunk. The bar must not run ahead."""
    client, _ = plane
    drive.short_ack = 1
    made = create(client, auth, [spec(clip)])
    seen = send(client, auth, made["files"][0]["url"], clip,
                chunk=2 * CHUNK)
    assert seen[0] == CHUNK, "2 chunks sent, 1 acknowledged, 1 reported"


def test_a_resumed_upload_continues_from_the_server_offset(plane, auth,
                                                           clip):
    """The tab was closed half-way. The same intent finds the same job,
    and the server says where to carry on."""
    client, _ = plane
    made = create(client, auth, [spec(clip)])
    url = made["files"][0]["url"]
    client.put(url, headers={**auth, "Content-Range":
                             f"bytes 0-{CHUNK - 1}/{len(clip)}"},
               content=clip[:CHUNK])
    again = create(client, auth, [spec(clip)])
    assert again["job"]["job_id"] == made["job"]["job_id"]
    assert again["created"] is False
    assert again["files"][0]["received"] == CHUNK
    off = client.get(url, headers=auth).json()["received"]
    assert off == CHUNK
    send(client, auth, url, clip, start=off)


def test_a_chunk_at_the_wrong_offset_is_told_where_to_resume(plane, auth,
                                                             clip):
    client, _ = plane
    made = create(client, auth, [spec(clip)])
    url = made["files"][0]["url"]
    r = client.put(url, headers={**auth, "Content-Range":
                                 f"bytes {CHUNK}-{2 * CHUNK - 1}/"
                                 f"{len(clip)}"},
                   content=clip[CHUNK:2 * CHUNK])
    assert r.status_code == 409 and r.json() == {"received": 0,
                                                 "complete": False}


def test_complete_before_every_byte_is_refused(plane, auth, clip, compute):
    client, ctx = plane
    made = create(client, auth, [spec(clip)])
    client.put(made["files"][0]["url"], headers={
        **auth, "Content-Range": f"bytes 0-{CHUNK - 1}/{len(clip)}"},
        content=clip[:CHUNK])
    r = client.post(f"/api/cloud/uploads/{made['job']['job_id']}/complete",
                    headers=auth)
    assert r.status_code == 400
    assert ctx.queue.get(made["job"]["job_id"])["state"] == Q.UPLOADING
    assert compute.starts() == 0, "no worker for an incomplete upload"


def test_completing_twice_starts_one_worker(plane, auth, clip, compute):
    client, _ = plane
    made = create(client, auth, [spec(clip)])
    send(client, auth, made["files"][0]["url"], clip)
    for _ in range(3):
        client.post(f"/api/cloud/uploads/{made['job']['job_id']}/complete",
                    headers=auth)
    assert compute.starts() == 1


def test_two_files_both_have_to_arrive(plane, auth, compute):
    client, _ = plane
    a, b = os.urandom(CHUNK + 5), os.urandom(2 * CHUNK + 7)
    made = create(client, auth, [spec(a, "a.mp4"), spec(b, "b.mov")])
    send(client, auth, made["files"][0]["url"], a)
    jid = made["job"]["job_id"]
    assert client.post(f"/api/cloud/uploads/{jid}/complete",
                       headers=auth).status_code == 400
    send(client, auth, made["files"][1]["url"], b)
    assert client.post(f"/api/cloud/uploads/{jid}/complete",
                       headers=auth).status_code == 200


# ------------------------------------------------------------ validation

@pytest.mark.parametrize("files,why", [
    ([{"name": "notes.txt", "size": 10, "mime": "text/plain"}], "video"),
    ([{"name": "x.mp4", "size": 10, "mime": "application/zip"}], "video"),
    ([{"name": "x.mp4", "size": 10 ** 13, "mime": "video/mp4"}], "large"),
])
def test_unsupported_uploads_are_refused(plane, auth, files, why):
    client, _ = plane
    r = client.post("/api/cloud/uploads", headers=auth, json={
        "idempotency_key": "bad-upload-01", "files": files, "settings": {}})
    assert r.status_code == 400
    assert r.json()["category"] == "invalid"


def test_settings_are_validated_by_the_pipelines_own_schema(plane, auth):
    client, _ = plane
    r = client.post("/api/cloud/uploads", headers=auth, json={
        "idempotency_key": "bad-settings-1", "files": [
            {"name": "x.mp4", "size": 10, "mime": "video/mp4"}],
        "settings": {"resolution": "16k"}})
    assert r.status_code == 400


def test_an_oversized_chunk_is_refused(plane, auth, clip):
    client, _ = plane
    made = create(client, auth, [spec(clip)])
    big = b"x" * (17 * 1024 * 1024)
    r = client.put(made["files"][0]["url"], headers={
        **auth, "Content-Range": f"bytes 0-{len(big) - 1}/{len(clip)}"},
        content=big)
    assert r.status_code == 400


def test_a_malformed_job_id_never_reaches_the_queue(plane, auth):
    client, _ = plane
    for bad in ("..%2F..%2Fetc", "ABCDEF123456", "1234"):
        r = client.get(f"/api/cloud/jobs/{bad}", headers=auth)
        assert r.status_code in (400, 404)


# ------------------------------------------------------------------ auth

def test_browser_routes_need_the_operator_token(plane):
    client, _ = plane
    for method, path in (("get", "/api/cloud/jobs"),
                         ("post", "/api/cloud/reconcile"),
                         ("get", "/api/cloud/jobs/abcdef123456")):
        r = getattr(client, method)(path)
        assert r.status_code == 401
        r = getattr(client, method)(path, headers={
            "Authorization": "Bearer wrong-token"})
        assert r.status_code == 401


def test_the_token_is_never_accepted_from_a_url(plane):
    client, _ = plane
    r = client.get(f"/api/cloud/jobs?token={API_TOKEN}")
    assert r.status_code == 401


def test_worker_routes_refuse_the_operator_token(plane, auth):
    """Two audiences, two credentials. A browser token cannot claim work."""
    client, _ = plane
    r = client.post("/api/cloud/worker/claim", headers=auth,
                    json={"worker_id": "w1"})
    assert r.status_code == 401


def test_worker_routes_refuse_no_token(plane):
    client, _ = plane
    r = client.post("/api/cloud/worker/claim", json={"worker_id": "w1"})
    assert r.status_code == 401


def test_the_config_route_says_nothing_secret(plane):
    client, _ = plane
    body = client.get("/api/cloud/config").text
    assert API_TOKEN not in body and WORKER_TOKEN not in body
    assert '"auth_required":true' in body.replace(" ", "")


# ------------------------------------------------------- what leaks, nothing

def test_a_cloud_error_reaches_the_browser_as_a_sentence(plane, auth, clip,
                                                         drive):
    """Drive's raw 404 text - with its file ids and request ids - stays in
    the server log."""
    client, _ = plane
    made = create(client, auth, [spec(clip)])
    drive.sessions.clear()                       # the session expired
    r = client.put(made["files"][0]["url"], headers={
        **auth, "Content-Range": f"bytes 0-{CHUNK - 1}/{len(clip)}"},
        content=clip[:CHUNK])
    assert r.status_code == 409
    assert r.json()["restarted"] is True and r.json()["received"] == 0


def test_job_views_carry_no_session_and_no_raw_error(plane, auth, clip):
    client, ctx = plane
    made = create(client, auth, [spec(clip)])
    jid = made["job"]["job_id"]
    body = client.get(f"/api/cloud/jobs/{jid}", headers=auth).text
    assert "SECRETSESSION" not in body and "upload_id" not in body


def test_a_failed_job_shows_a_category_and_a_fixed_sentence(plane, auth,
                                                            clip):
    client, ctx = plane
    made = create(client, auth, [spec(clip)])
    send(client, auth, made["files"][0]["url"], clip)
    jid = made["job"]["job_id"]
    client.post(f"/api/cloud/uploads/{jid}/complete", headers=auth)
    c = ctx.queue.claim("w")
    ctx.queue.fail(jid, "w", c.claim_token, "pipeline",
                   "Traceback ... /home/alpha/server/agents/x.py line 9 "
                   "ya29.SECRET")
    view = client.get(f"/api/cloud/jobs/{jid}", headers=auth).json()
    assert view["error"]["category"] == "pipeline"
    assert view["error"]["message"].startswith("The video could not")
    assert "Traceback" not in str(view) and "ya29" not in str(view)


# ----------------------------------------------------------- cancel, list

def test_cancel_and_list(plane, auth, clip):
    client, _ = plane
    made = create(client, auth, [spec(clip)])
    jid = made["job"]["job_id"]
    r = client.post(f"/api/cloud/jobs/{jid}/cancel", headers=auth)
    assert r.json()["state"] == Q.CANCELLED
    jobs = client.get("/api/cloud/jobs", headers=auth).json()["jobs"]
    assert [j["job_id"] for j in jobs] == [jid]


# --------------------------------------------------- mounting, and not

def test_misconfiguration_mounts_only_a_report(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cloud.app import mount

    env = cloud_env(tmp_path)
    env.pop("AT_GCP_WORKER_INSTANCE")
    app = FastAPI()
    assert mount(app, host="127.0.0.1", env=env) is None
    with TestClient(app) as c:
        body = c.get("/api/cloud/config").json()
        assert body["enabled"] is False
        assert any("AT_GCP_WORKER_INSTANCE" in p for p in body["problems"])
        assert c.post("/api/cloud/uploads", json={}).status_code == 404


def test_a_public_host_without_a_token_does_not_mount(tmp_path):
    from fastapi import FastAPI

    from cloud.app import mount

    env = cloud_env(tmp_path)
    env.pop("AT_CLOUD_API_TOKEN")
    assert mount(FastAPI(), host="0.0.0.0", env=env) is None


def test_disabled_means_absent(tmp_path):
    from fastapi import FastAPI

    from cloud.app import mount

    app = FastAPI()
    before = len(app.routes)
    assert mount(app, host="127.0.0.1",
                 env={**cloud_env(tmp_path), "AT_CLOUD_ENABLED": "0"}) is None
    assert len(app.routes) == before
