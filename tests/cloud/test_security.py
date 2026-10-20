"""No credential reaches a log, an error response or a frontend asset.

Brief, section 13, test 14 - and test 15, that the local path is untouched.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import pytest

from cloud.config import CloudConfig
from cloud.sanitize import CloudError, classify, redact

from fakes import API_TOKEN, WORKER_TOKEN, cloud_env

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------- redaction

@pytest.mark.parametrize("secret,text", [
    ("ya29.a0AfB_byC-SECRET", "token ya29.a0AfB_byC-SECRET expired"),
    ("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",
     "id eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl bad"),
    ("abc123SESSION", "PUT https://x/upload?upload_id=abc123SESSION&x=1"),
    ("AIzaSyFAKEKEY", "GET https://x/?key=AIzaSyFAKEKEY"),
    ("sk-live-SECRET", "Authorization: Bearer sk-live-SECRET"),
    ("MIIEvQIBADANBg", "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n"
                       "-----END PRIVATE KEY-----"),
    ("MIIEsecret", '{"private_key": "MIIEsecret", "x": 1}'),
])
def test_redact_removes_every_kind_of_credential(secret, text):
    assert secret not in redact(text)


def test_a_classified_error_carries_no_credential():
    err = classify(CloudError("auth", "Bearer ya29.LEAK at "
                                      "https://u/?upload_id=LEAK2"))
    assert "LEAK" not in str(err) and "LEAK" not in repr(err.public())


def test_the_public_half_of_an_error_is_a_fixed_sentence():
    err = CloudError("permission", "Required 'compute.instances.start' on "
                                   "projects/secret-project-name")
    assert "secret-project-name" not in str(err.public())


# ------------------------------------------------------------ config

def test_config_never_shows_a_secret(tmp_path):
    cfg = CloudConfig.from_env(cloud_env(tmp_path))
    for view in (repr(cfg), str(cfg.summary())):
        assert API_TOKEN not in view and WORKER_TOKEN not in view
    assert cfg.summary()["api_token"] == "set"


def test_config_problems_never_quote_a_secret(tmp_path):
    env = cloud_env(tmp_path, AT_CLOUD_API_TOKEN="short-secret-x")
    problems = CloudConfig.from_env(env).problems("control")
    assert any("too short" in p for p in problems)
    assert all("short-secret-x" not in p for p in problems)


# ----------------------------------------------------------- frontend

_LEAKS = [re.compile(p) for p in (
    r"ya29\.", r"-----BEGIN", r"AIza[0-9A-Za-z_-]{20,}", r"private_key",
    r"[?&](token|access_token|key)=", r"client_secret",
    r"Bearer [A-Za-z0-9_\-.]{20,}")]


@pytest.mark.parametrize("name", ["cloud.js", "cloud.html"])
def test_the_cloud_page_carries_no_credential(name):
    src = (ROOT / "frontend" / name).read_text(encoding="utf-8")
    for rx in _LEAKS:
        assert not rx.search(src), f"{name}: {rx.pattern}"


def test_the_cloud_page_sends_its_token_in_a_header_only():
    src = (ROOT / "frontend" / "cloud.js").read_text(encoding="utf-8")
    assert "headers.Authorization = `Bearer ${S.token}`" in src
    assert "sessionStorage" in src, "per tab, not persisted"
    assert "localStorage" in src            # job ids and a salt only
    stored = re.findall(r"setItem\('local', (\w+)", src)
    assert set(stored) <= {"JOBS_KEY", "SALT_KEY"}, stored


def test_every_frontend_asset_is_free_of_credentials():
    for p in (ROOT / "frontend").rglob("*"):
        if p.suffix not in (".js", ".html", ".css"):
            continue
        src = p.read_text(encoding="utf-8", errors="ignore")
        for rx in (_LEAKS[0], _LEAKS[1], _LEAKS[3], _LEAKS[5]):
            assert not rx.search(src), f"{p.name}: {rx.pattern}"


# ---------------------------------------------------------------- logs

def test_a_whole_production_logs_no_credential(plane, worker_cp, drive,
                                               clock, tmp_path, caplog):
    """Upload, claim, render, upload, complete, idle, shut down - with
    every log line captured at DEBUG - and not one credential in them."""
    import os

    from cloud.worker import Worker

    from test_worker import AUTH, CHUNK, FakePipeline

    client, ctx = plane
    caplog.set_level(logging.DEBUG)
    data = os.urandom(CHUNK + 7)
    made = client.post("/api/cloud/uploads", headers=AUTH, json={
        "idempotency_key": "log-check-0001", "files": [
            {"name": "c.mp4", "size": len(data), "mime": "video/mp4"}],
        "settings": {}}).json()
    url = made["files"][0]["url"]
    jid = made["job"]["job_id"]
    client.put(url, headers={**AUTH, "Content-Range":
                             f"bytes 0-{CHUNK - 1}/{len(data)}"},
               content=data[:CHUNK])
    client.put(url, headers={**AUTH, "Content-Range":
                             f"bytes {CHUNK}-{len(data) - 1}/{len(data)}"},
               content=data[CHUNK:])
    client.post(f"/api/cloud/uploads/{jid}/complete", headers=AUTH)
    client.get("/api/cloud/jobs/abcdef123456", headers={
        "Authorization": "Bearer wrong-" + API_TOKEN[:10]})
    Worker(ctx.cfg, worker_cp, drive, FakePipeline(),
           uploads_dir=tmp_path / "u", outputs_dir=tmp_path / "o",
           worker_id="test-host:boot0001", clock=clock, sleep=clock.sleep,
           checks=lambda: ([], {})).run()
    assert ctx.queue.get(jid)["state"] == "completed"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (API_TOKEN, WORKER_TOKEN, "SECRETSESSION", "ya29."):
        assert secret not in logged, secret


# ------------------------------------------------- 15. the local path

def test_with_cloud_off_the_app_is_unchanged(monkeypatch):
    monkeypatch.delenv("AT_CLOUD_ENABLED", raising=False)
    import importlib
    import main
    importlib.reload(main)
    paths = set()

    def walk(routes):
        for r in routes:
            if getattr(r, "path", None):
                paths.add(r.path)
            sub = getattr(r, "routes", None)
            if sub:
                walk(sub)
    walk(main.app.routes)
    assert not any(p.startswith("/api/cloud") for p in paths)
    assert "/cloud" not in paths
    assert not [t for t in threading.enumerate()
                if t.name == "cloud-reconcile"]


def test_the_upload_formats_are_the_renderers_own():
    from agents.tools.pipeline_bridge import build_video
    from cloud.paths import VIDEO_EXT
    assert tuple(VIDEO_EXT) == tuple(build_video.VIDEO_EXT)
