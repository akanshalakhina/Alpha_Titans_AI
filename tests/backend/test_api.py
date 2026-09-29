"""Unit / API tests for the Alpha Titans server (FastAPI TestClient)."""
import shutil
from pathlib import Path

import pytest

import adapter
import jobs
from conftest import FIX, upload


def names(client, job_id):
    return [c["name"] for c in client.get(f"/api/jobs/{job_id}/clips").json()["clips"]]


# ---------------------------------------------------------------- health / page

def test_health_reports_capabilities_and_options(client):
    h = client.get("/api/health").json()
    assert h["ok"] is True
    assert h["capabilities"]["ffmpeg"] and h["capabilities"]["pipeline"]
    for k in ("orientation", "resolution", "quality", "speed", "transition_style",
              "mood", "music_source", "keep_audio", "sequence"):
        assert h["options"][k], k
    assert h["options"]["quality"] == ["draft", "standard", "high", "master"]
    assert len(h["agents"]) == 5
    assert [s["id"] for s in h["stages"]][:2] == ["probe", "measure"]
    assert h["limits"]["clip_mb"] > 0


def test_index_is_never_cached(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-cache"
    assert 'id="loader"' in r.text and 'id="sessionNotice"' in r.text


def test_static_assets_served_with_version_query(client):
    for f in ("app.js", "styles.css"):
        assert client.get(f"/static/{f}?v=4").status_code == 200


# ---------------------------------------------------------------- jobs

def test_create_and_get_job(client, job):
    j = client.get(f"/api/jobs/{job}").json()
    assert j["id"] == job and j["status"] == "draft"
    assert len(j["stages"]) == 10


def test_unknown_job_returns_exact_no_job_message(client):
    # the frontend's session-recovery logic matches this exact text
    r = client.get("/api/jobs/deadbeef0000")
    assert r.status_code == 404
    assert r.json()["detail"] == "No job deadbeef0000"


def test_empty_draft_not_listed_but_draft_with_clips_is(client, job, job_with_clips):
    ids = [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert job not in ids or job == job_with_clips
    assert job_with_clips in ids


def test_job_survives_server_memory_loss(client, job_with_clips):
    jobs._JOBS.pop(job_with_clips, None)          # simulate a restart
    r = client.get(f"/api/jobs/{job_with_clips}")
    assert r.status_code == 200
    assert len(names(client, job_with_clips)) == 3


def test_lost_job_after_folders_cleared(client, job_with_clips):
    shutil.rmtree(adapter.build_dir(job_with_clips))
    shutil.rmtree(adapter.src_dir(job_with_clips))
    jobs._JOBS.pop(job_with_clips, None)
    for method, path in (("post", "/script"), ("post", "/start"), ("get", "/clips")):
        r = getattr(client, method)(f"/api/jobs/{job_with_clips}{path}", json={"text": ""}) \
            if method == "post" else client.get(f"/api/jobs/{job_with_clips}{path}")
        assert r.status_code == 404
        assert r.json()["detail"] == f"No job {job_with_clips}"


# ---------------------------------------------------------------- clips

def test_upload_numbers_clips_and_reads_metadata(client, job_with_clips):
    clips = client.get(f"/api/jobs/{job_with_clips}/clips").json()["clips"]
    assert [c["name"] for c in clips] == [
        "01_Gear box build.mp4", "02_robot_arm.mp4", "03_vertical tower.mp4"]
    by = {c["name"]: c for c in clips}
    assert by["03_vertical tower.mp4"]["orientation"] == "vertical"
    assert by["01_Gear box build.mp4"]["orientation"] == "horizontal"
    assert by["01_Gear box build.mp4"]["has_audio"] is True
    assert by["02_robot_arm.mp4"]["has_audio"] is False
    assert all(c["duration"] > 5 for c in clips)


def test_upload_rejects_non_video(client, job):
    r = client.post(f"/api/jobs/{job}/clips",
                    files=[("files", ("bad.txt", open(FIX / "bad.txt", "rb"), "text/plain"))])
    assert r.status_code == 200
    assert r.json()["saved"] == []
    assert r.json()["rejected"][0]["why"] == "not a video container"


def test_upload_rejects_unreadable_video(client, job, tmp_path):
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"\x00" * 5000)
    r = client.post(f"/api/jobs/{job}/clips",
                    files=[("files", ("broken.mp4", open(fake, "rb"), "video/mp4"))])
    assert r.json()["rejected"][0]["why"] == "ffprobe could not read it"
    assert names(client, job) == []


def test_second_upload_appends_numbering(client, job_with_clips):
    upload(client, job_with_clips, "robot_arm.mp4")
    assert names(client, job_with_clips)[-1] == "04_robot_arm.mp4"


def test_poster_is_a_jpeg(client, job_with_clips):
    r = client.get(f"/api/jobs/{job_with_clips}/poster/01_Gear box build.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"


def test_clip_stream_and_path_traversal_blocked(client, job_with_clips):
    assert client.get(f"/api/jobs/{job_with_clips}/clip/02_robot_arm.mp4").status_code == 200
    r = client.get(f"/api/jobs/{job_with_clips}/clip/..%2F..%2Fconfig.py")
    assert r.status_code in (400, 404)


def test_delete_clip(client, job_with_clips):
    r = client.delete(f"/api/jobs/{job_with_clips}/clips/02_robot_arm.mp4")
    assert r.status_code == 200
    assert "02_robot_arm.mp4" not in names(client, job_with_clips)
    assert client.delete(f"/api/jobs/{job_with_clips}/clips/nope.mp4").status_code == 404


# ---------------------------------------------------------------- reorder (bug 5)

def test_reorder_renumbers_in_new_order(client, job_with_clips):
    order = list(reversed(names(client, job_with_clips)))
    r = client.post(f"/api/jobs/{job_with_clips}/clips/order", json={"order": order})
    assert r.status_code == 200, r.text
    assert names(client, job_with_clips) == [
        "01_vertical tower.mp4", "02_robot_arm.mp4", "03_Gear box build.mp4"]


def test_reorder_rejects_wrong_set(client, job_with_clips):
    r = client.post(f"/api/jobs/{job_with_clips}/clips/order", json={"order": ["x.mp4"]})
    assert r.status_code == 400


def test_reorder_retries_when_file_briefly_locked(client, job_with_clips, monkeypatch):
    """Windows: a file still open by a preview raises PermissionError on rename."""
    real, fails = Path.rename, {"n": 0}

    def flaky(self, target):
        if fails["n"] < 3:
            fails["n"] += 1
            raise PermissionError("in use")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    order = list(reversed(names(client, job_with_clips)))
    r = client.post(f"/api/jobs/{job_with_clips}/clips/order", json={"order": order})
    assert r.status_code == 200, r.text
    assert fails["n"] == 3
    monkeypatch.setattr(Path, "rename", real)
    assert names(client, job_with_clips)[0] == "01_vertical tower.mp4"


def test_reorder_locked_file_gives_409_and_rolls_back(client, job_with_clips, monkeypatch):
    before = names(client, job_with_clips)
    real = Path.rename

    def locked(self, target):
        if "robot_arm" in self.name:                 # second clip never releases
            raise PermissionError("in use")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", locked)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = client.post(f"/api/jobs/{job_with_clips}/clips/order",
                    json={"order": list(reversed(before))})
    monkeypatch.setattr(Path, "rename", real)
    assert r.status_code == 409
    assert "still in use" in r.json()["detail"]
    assert names(client, job_with_clips) == before   # nothing left half-renamed
    assert not any(n.startswith("tmp_") for n in names(client, job_with_clips))


# ---------------------------------------------------------------- script / voice / music

def test_script_saved_and_persisted(client, job_with_clips):
    r = client.post(f"/api/jobs/{job_with_clips}/script", json={"text": "Frame\nMotor"})
    assert r.status_code == 200 and r.json()["script"] == "Frame\nMotor"
    jobs._JOBS.pop(job_with_clips, None)
    assert client.get(f"/api/jobs/{job_with_clips}").json()["script"] == "Frame\nMotor"


def test_script_file_upload(client, job, tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("Line one\nLine two")
    r = client.post(f"/api/jobs/{job}/script-file",
                    files={"file": ("s.txt", open(f, "rb"), "text/plain")})
    assert r.status_code == 200 and r.json()["script"].startswith("Line one")


def test_voice_note_upload_and_playback(client, job):
    r = client.post(f"/api/jobs/{job}/voice",
                    files={"file": ("voice_1.webm", open(FIX / "voice.webm", "rb"), "audio/webm")})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["bytes"] > 1000 and v["duration"] > 2
    a = client.get(f"/api/jobs/{job}/voice-file?v=x")
    assert a.status_code == 200 and len(a.content) == v["bytes"]


def test_second_voice_note_replaces_first(client, job, tmp_path):
    for i, freq in enumerate((1, 2)):
        client.post(f"/api/jobs/{job}/voice",
                    files={"file": (f"voice_{i}.webm", open(FIX / "voice.webm", "rb"), "audio/webm")})
    assert client.get(f"/api/jobs/{job}").json()["voice"]["filename"] == "voice_1.webm"


def test_music_library_roundtrip(client, tmp_path):
    import subprocess
    wav = tmp_path / "beat.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=220:duration=2", str(wav)], check=True)
    r = client.post("/api/music", files={"file": ("beat.wav", open(wav, "rb"), "audio/wav")},
                    data={"mood": "calm"})
    assert r.status_code == 200 and r.json()["duration"] >= 1.9
    assert "beat.wav" in [t["name"] for t in client.get("/api/music").json()["tracks"]]
    assert client.post("/api/music", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
    assert client.delete("/api/music/beat.wav").status_code == 200
