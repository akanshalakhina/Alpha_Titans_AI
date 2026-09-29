"""Shared setup for the backend tests.

Every test session gets its own empty uploads/outputs/music folders, so the
tests never touch real builds. Config reads these at import time, which is
why they are set before the server modules are imported."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
_TMP = Path(tempfile.mkdtemp(prefix="at_tests_"))
os.environ["AT_UPLOADS_DIR"] = str(_TMP / "uploads")
os.environ["AT_OUTPUTS_DIR"] = str(_TMP / "outputs")
os.environ["AT_MUSIC_DIR"] = str(_TMP / "music")
os.environ.pop("AT_VISION_API_KEY", None)
sys.path.insert(0, str(ROOT / "server"))

import config  # noqa: E402

config.ensure_dirs()
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
import jobs  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def job(client):
    r = client.post("/api/jobs", json={})
    assert r.status_code == 200
    return r.json()["id"]


def upload(client, job_id, *names):
    files = [("files", (n, open(FIX / n, "rb"), "video/mp4")) for n in names]
    try:
        return client.post(f"/api/jobs/{job_id}/clips", files=files)
    finally:
        for _, (_, fh, _) in files:
            fh.close()


@pytest.fixture
def job_with_clips(client, job):
    r = upload(client, job, "Gear box build.mp4", "robot_arm.mp4", "vertical tower.mp4")
    assert r.status_code == 200, r.text
    assert len(r.json()["saved"]) == 3
    return job
