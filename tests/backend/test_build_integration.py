"""Integration: a real render through the real pipeline, driven by the API."""
import time

import pytest


def wait(client, job_id, timeout=600):
    seen, t0 = [], time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        seen.append(j["progress"])
        if j["status"] in ("done", "error", "cancelled"):
            return j, seen
        time.sleep(1)
    pytest.fail("build did not finish in time")


@pytest.mark.slow
def test_full_build_renders_verified_video(client, job_with_clips):
    client.post(f"/api/jobs/{job_with_clips}/script", json={"text": "Frame goes together\nMotor spins up"})
    r = client.post(f"/api/jobs/{job_with_clips}/start",
                    json={"resolution": "720p", "quality": "draft", "title": "TEST_BUILD"})
    assert r.status_code == 200 and r.json()["status"] == "running"
    j, seen = wait(client, job_with_clips)
    assert j["status"] == "done", j.get("error")
    assert seen == sorted(seen), "server progress went backwards"
    o = j["output"]
    assert o["frames_ok"] and not o["black_frames"]
    assert o["height"] == 720 and o["video_codec"] == "h264" and o["audio_codec"]
    assert j["started"] and j["finished"] and j["finished"] > j["started"]
    assert all(s["status"] in ("done", "skipped") for s in j["stages"])
    d = client.get(f"/api/jobs/{job_with_clips}/download")
    assert d.status_code == 200 and d.content[4:8] == b"ftyp"
    th = client.get(f"/api/jobs/{job_with_clips}/file/{o['thumbnail']}")
    assert th.status_code == 200
    # regenerate: new job, same clips, original kept
    rg = client.post(f"/api/jobs/{job_with_clips}/regenerate", json={"speed": "fast"})
    assert rg.status_code == 200
    new = rg.json()["id"]
    assert new != job_with_clips
    assert len(client.get(f"/api/jobs/{new}/clips").json()["clips"]) == 3
    j2, _ = wait(client, new)
    assert j2["status"] == "done", j2.get("error")
    assert j2["output"]["speed"] == 1.35
    assert client.get(f"/api/jobs/{job_with_clips}").json()["status"] == "done"


@pytest.mark.slow
def test_cancel_stops_a_running_build(client, job_with_clips):
    client.post(f"/api/jobs/{job_with_clips}/start", json={"resolution": "720p", "quality": "draft"})
    time.sleep(1.5)
    assert client.post(f"/api/jobs/{job_with_clips}/cancel").status_code == 200
    j, _ = wait(client, job_with_clips, 120)
    assert j["status"] in ("cancelled", "error")
