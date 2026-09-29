"""
Compatibility facade: an agent-built job seen through the EXISTING app.

The point of these tests is that nothing in server/jobs.py, server/main.py or
the frontend was changed - so they drive the real, unmodified endpoints with
FastAPI's TestClient and assert the agent's video comes back out of them.

The end-to-end test performs a real render.
"""

import json
import shutil
from pathlib import Path

import pytest

from agents import compat
from agents.artifacts import JobSettings, RenderResult
from agents.runner import make_job, run_job
from agents.store import ArtifactStore
from conftest import llm_returning_plan

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


# --------------------------------------------------------------- fixtures

# ------------------------------------ 1. projection into the job.json shape

def test_job_json_is_written_in_the_legacy_shape(published_job):
    job, payload = published_job
    p = Path(job.build_dir) / "job.json"
    assert p.exists(), "no job.json was written"

    on_disk = json.loads(p.read_text())
    # exactly the keys jobs._persist writes - no more, no fewer
    assert set(on_disk) == {"id", "settings", "created", "status", "error",
                            "started", "finished", "stages", "output",
                            "captions", "voice", "script"}
    assert on_disk == payload


def test_projected_status_and_timings_are_real(published_job):
    _, payload = published_job
    assert payload["status"] == "done"
    assert payload["error"] is None
    assert payload["started"] and payload["finished"]
    assert payload["finished"] > payload["started"]
    assert payload["created"] <= payload["started"] + 1


def test_stages_are_projected_honestly_not_all_marked_done(published_job):
    """Phase 1 has no Vision or Music agent and never builds contact sheets.
    Those three must read `skipped`, not `done`."""
    _, payload = published_job
    st = payload["stages"]
    assert len(st) == 10
    assert {k for k, v in st.items() if v["status"] == "skipped"} == {
        "vision", "sheets", "music"}
    assert {k for k, v in st.items() if v["status"] == "done"} == {
        "probe", "measure", "plan", "direct", "cut", "join", "master"}
    # the stages that ran carry real timings from the trace
    for sid in ("probe", "cut", "master"):
        assert st[sid]["started"] and st[sid]["finished"]


def test_output_is_master_stages_own_record_not_a_re_derivation(published_job):
    job, payload = published_job
    o = payload["output"]
    original = json.loads((Path(job.build_dir) / "output.json").read_text())
    for k, v in original.items():
        assert o[k] == v, f"output.{k} was altered"
    # provenance added, nothing else
    assert o["_produced_by"] == compat.MARKER
    assert o["attempt"] == 1 and o["attempts"] == 1


def test_output_carries_every_field_the_frontend_reads(published_job):
    _, payload = published_job
    o = payload["output"]
    for key in ("file", "path", "thumbnail", "duration", "frames",
                "frames_expected", "frames_ok", "width", "height", "fps",
                "video_codec", "audio_codec", "size", "lufs", "black_frames",
                "speed", "segments", "music", "orientation", "resolution",
                "keep_audio", "has_source_audio", "sample_rate"):
        assert key in o, f"frontend reads output.{key} and it is missing"


def test_settings_do_not_claim_capabilities_phase_1_lacks(published_job):
    _, payload = published_job
    s = payload["settings"]
    assert s["title"] == "COMPAT_TEST"
    assert s["_produced_by"] == compat.MARKER
    # there is no Music Agent, so the bed really is none
    assert s["music_source"] == "none" and s["keep_audio"] == "none"
    assert s["speed"] == "normal"


# ----------------------------------- 2/3. discovery and restore by the app

def test_existing_api_jobs_discovers_the_agent_build(client, published_job):
    job, _ = published_job
    r = client.get("/api/jobs")
    assert r.status_code == 200
    ids = {j["id"] for j in r.json()["jobs"]}
    assert job.job_id in ids, "the agent build is invisible to /api/jobs"


def test_the_drawer_card_has_everything_it_renders(client, published_job):
    job, _ = published_job
    card = next(j for j in client.get("/api/jobs").json()["jobs"]
                if j["id"] == job.job_id)
    assert card["title"] == "COMPAT_TEST"
    assert card["status"] == "done"
    assert card["clips"] == 2                    # read from manifest.json
    assert card["orientation"] == "horizontal"
    assert card["duration"] > 0
    assert card["thumbnail"] == "thumbnail.jpg"
    assert card["progress"] == 100.0, (
        "done + skipped must total 100%, or the drawer shows a stalled build")


def test_existing_job_endpoint_restores_the_full_snapshot(client,
                                                          published_job):
    job, _ = published_job
    r = client.get(f"/api/jobs/{job.job_id}")
    assert r.status_code == 200
    snap = r.json()
    assert snap["status"] == "done"
    assert snap["progress"] == 100.0
    assert len(snap["stages"]) == 10
    assert snap["output"]["frames_ok"] is True
    # analysis comes from manifest.json, which the agent path also writes
    assert len(snap["analysis"]["clips"]) == 2
    assert snap["analysis"]["orientation"] == "horizontal"


def test_restore_happens_from_disk_with_an_empty_registry(published_job):
    """Proves _restore did the work, not a leftover in-memory Job."""
    import jobs as legacy_jobs

    job, _ = published_job
    legacy_jobs._JOBS.clear()
    restored = legacy_jobs.get(job.job_id)
    assert restored is not None, "jobs._restore could not read the projection"
    assert restored.status == "done"
    assert restored.progress() == 100.0
    assert restored.output["frames_ok"] is True


# --------------------------------------------------- 4. the real download

def test_existing_download_returns_the_real_mp4(client, published_job):
    job, payload = published_job
    r = client.get(f"/api/jobs/{job.job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.content[4:8] == b"ftyp", "not an mp4"
    on_disk = Path(payload["output"]["path"]).read_bytes()
    assert r.content == on_disk, "download did not serve the produced file"


def test_existing_file_endpoint_serves_the_thumbnail(client, published_job):
    job, _ = published_job
    r = client.get(f"/api/jobs/{job.job_id}/file/thumbnail.jpg")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8", "not a jpeg"


def test_notes_endpoint_reads_the_agent_output_json(client, published_job):
    job, _ = published_job
    r = client.get(f"/api/jobs/{job.job_id}/notes")
    assert r.status_code == 200
    assert r.json()["frames_ok"] is True


# ------------------------------------------------------- honesty and safety

def test_an_escalated_build_is_not_presented_as_a_success(tmp_path,
                                                          published_job):
    """A video its own critic refused must not read as `done`."""
    job, _ = published_job
    graph = {"status": "escalated", "verdict": "ESCALATE",
             "error": "the critic re-issued a constraint that was already tried"}
    status, error = compat._project_status(graph, None, has_output=True)
    assert status == "error"
    assert "did not pass it" in error
    assert "already tried" in error


def test_a_build_with_no_output_is_an_error(published_job):
    status, error = compat._project_status({"status": "done"}, None,
                                           has_output=False)
    assert status == "error" and "no output file" in error


def test_a_legacy_job_json_is_never_overwritten(tmp_path):
    build = tmp_path / "outputs" / "legacyjob"
    build.mkdir(parents=True)
    (build / compat.STATE_DB).write_bytes(b"")          # looks like an agent job
    legacy = {"id": "legacyjob", "settings": {"title": "REAL BUILD"},
              "status": "done"}
    (build / "job.json").write_text(json.dumps(legacy))

    with pytest.raises(FileExistsError, match="legacy pipeline"):
        compat.project("legacyjob", tmp_path)

    # untouched
    assert json.loads((build / "job.json").read_text()) == legacy


def test_a_folder_with_no_agent_build_is_refused(tmp_path):
    (tmp_path / "outputs" / "nothing").mkdir(parents=True)
    with pytest.raises(compat.NotAnAgentJob):
        compat.project("nothing", tmp_path)


def test_dry_run_writes_nothing(tmp_path, published_job):
    job, _ = published_job
    target = Path(job.build_dir) / "job.json"
    before = target.read_bytes()
    compat.project(job.job_id, dry_run=True)
    assert target.read_bytes() == before


def test_republishing_is_idempotent(published_job):
    job, first = published_job
    second = compat.project(job.job_id)
    assert second == first


def test_agent_jobs_are_discoverable_by_marker(published_job):
    job, _ = published_job
    assert job.job_id in compat.agent_jobs()
    assert compat.is_agent_job(job.job_id)


def test_project_all_reports_per_job_outcomes(published_job):
    results = compat.project_all(dry_run=True)
    job, _ = published_job
    assert results[job.job_id].startswith("ok")
