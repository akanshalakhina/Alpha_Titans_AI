"""The AI soundtrack: found silent in real builds because beatbed.py needs
`soundfile`, which was not in requirements.txt."""
import importlib.util
import json
import subprocess
import sys
import time

import pytest

import adapter
from conftest import ROOT

HAS_SF = importlib.util.find_spec("soundfile") is not None


def test_health_reports_beat_engine_only_when_it_can_run(client):
    assert client.get("/api/health").json()["capabilities"]["beat_engine"] is HAS_SF


def test_requirements_list_soundfile():
    assert "soundfile" in (ROOT / "requirements.txt").read_text()


@pytest.mark.skipif(not HAS_SF, reason="soundfile not installed")
def test_beatbed_writes_a_score(tmp_path):
    out = tmp_path / "score.wav"
    r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "beatbed.py"), str(out),
                        "12", "4.0", "8.0", "--hits", "4.0", "8.0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.stat().st_size > 100_000


def test_stdlib_engine_writes_a_score(tmp_path):
    out = tmp_path / "score.wav"
    r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "bgm_stdlib.py"), str(out),
                        "12", "4.0", "8.0", "--style", "bright"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.stat().st_size > 100_000


@pytest.mark.slow
@pytest.mark.parametrize("mood", ["beat", "calm"])
def test_ai_soundtrack_is_never_silent(client, job_with_clips, mood):
    client.post(f"/api/jobs/{job_with_clips}/start",
                json={"resolution": "720p", "quality": "draft", "mood": mood, "music_source": "generate"})
    t0 = time.time()
    while time.time() - t0 < 400:
        j = client.get(f"/api/jobs/{job_with_clips}").json()
        if j["status"] != "running":
            break
        time.sleep(1)
    assert j["status"] == "done", j.get("error")
    plan = json.loads((adapter.build_dir(job_with_clips) / "_audio.json").read_text())
    assert plan["track"], "AI soundtrack was selected but no score was produced"
    assert j["output"]["music"] != "silent"
    assert j["output"]["lufs"] > -40, f"score is silent: {j['output']['lufs']} LUFS"
