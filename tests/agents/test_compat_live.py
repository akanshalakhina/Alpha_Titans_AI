"""
Phase 2 Step 4: EditPlan projection, auto-projection, live status,
agent cancellation, and the legacy /start safety guard.

The real-render pieces reuse the session's published job; the guard and
projection pieces run against isolated fixtures.
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agents import compat
from agents.artifacts import EditPlan, JobSettings
from agents.runner import make_job, run_job
from agents.store import ArtifactStore, resolve_job_db
from conftest import llm_returning_plan

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


# ------------------------------------------ 1. EditPlan projection (A)

def test_edit_plan_is_projected_into_the_legacy_shape(published_job):
    job, _ = published_job
    plan = compat.build_edit_plan(job.job_id)
    assert plan is not None
    # exactly the keys the legacy director writes, plus provenance
    for key in ("vision", "median_motion", "busy_threshold", "segments",
                "transitions", "decisions"):
        assert key in plan, f"legacy key {key} missing"
    assert plan["_produced_by"] == compat.MARKER


def test_every_decision_carries_the_fields_the_ui_renders(published_job):
    job, _ = published_job
    plan = compat.build_edit_plan(job.job_id)
    assert plan["decisions"], "no decisions projected"
    for d in plan["decisions"]:
        for key in ("agent", "what", "chose", "why", "decided_by"):
            assert key in d, f"UI reads d.{key}"
        assert d["what"], "a decision with no subject is useless"
        assert d["why"], "a decision with no reason is useless"
        # provenance preserved beyond what the legacy shape can express
        assert "decided_by_detail" in d
        assert "confidence" in d
        assert "produced_by" in d


def test_decisions_map_onto_legacy_agent_ids(published_job):
    """The UI looks the id up in /api/health's agent list."""
    job, _ = published_job
    plan = compat.build_edit_plan(job.job_id)
    legacy_ids = {"vision", "editor", "animation", "merge", "music"}
    seen = {d["agent"] for d in plan["decisions"]}
    assert seen <= legacy_ids, f"unknown agent ids: {seen - legacy_ids}"
    assert seen, "no agents attributed"


def test_decided_by_never_credits_the_operator_for_a_machine_choice(
        published_job):
    """`measurement` renders as "decided", anything unknown as "yours".
    A measurement-driven choice must never be badged as the operator's."""
    job, _ = published_job
    plan = compat.build_edit_plan(job.job_id)
    for d in plan["decisions"]:
        if d["decided_by_detail"] in ("measurement", "policy", "fallback"):
            assert d["decided_by"] == "measurement", (
                f"{d['what']} was machine-decided but is badged "
                f"{d['decided_by']!r}")
        if d["decided_by_detail"] == "human":
            assert d["decided_by"] == "you"


def test_a_source_the_legacy_badge_cannot_express_is_put_in_the_text():
    """`llm` and `fallback` have no honest legacy badge, so the real source
    goes into `why`, which the UI always shows."""
    for raw in ("llm", "fallback"):
        d = compat._project_decision(
            {"what": "W", "chose": "C", "why": "because", "decided_by": raw,
             "confidence": 0.5}, "editing_director")
        assert f"[decided by the {raw}]" in d["why"]


def test_decisions_are_in_production_order(published_job):
    job, _ = published_job
    plan = compat.build_edit_plan(job.job_id)
    producers = [d["produced_by"] for d in plan["decisions"]]
    order = ["clip_analysis", "editing_director", "rendering", "critic"]
    seen = [p for p in order if p in producers]
    positions = [producers.index(p) for p in seen]
    assert positions == sorted(positions), "decisions are out of build order"


def test_project_writes_edit_plan_json_alongside_job_json(published_job):
    job, _ = published_job
    p = Path(job.build_dir) / "edit_plan.json"
    assert p.exists(), "project() did not write edit_plan.json"
    assert json.loads(p.read_text())["_produced_by"] == compat.MARKER


def test_a_legacy_edit_plan_is_never_overwritten(tmp_path):
    build = tmp_path / "outputs" / "legacyplan"
    build.mkdir(parents=True)
    (build / compat.STATE_DB).write_bytes(b"")
    legacy = {"decisions": [{"what": "REAL"}], "median_motion": 1.0}
    (build / "edit_plan.json").write_text(json.dumps(legacy))
    with pytest.raises(FileExistsError, match="legacy director"):
        compat.project_edit_plan("legacyplan", tmp_path)
    assert json.loads((build / "edit_plan.json").read_text()) == legacy


# ---------------------------------------- 2. snapshot / API integration

def test_legacy_plan_endpoint_returns_200_for_an_agent_job(client,
                                                           published_job):
    job, _ = published_job
    r = client.get(f"/api/jobs/{job.job_id}/plan")
    assert r.status_code == 200, "the decisions panel would be blank"
    assert r.json()["decisions"]


def test_the_snapshot_carries_a_non_empty_edit_plan(client, published_job):
    """What the UI actually reads: job.edit_plan, not /plan."""
    job, _ = published_job
    snap = client.get(f"/api/jobs/{job.job_id}").json()
    plan = snap.get("edit_plan")
    assert plan and plan.get("decisions"), "renderPlan() would hide the card"


# ------------------------------------------- 3. auto-projection (B)

@pytest.mark.slow
def test_a_finished_run_projects_itself_with_no_manual_step(agent_root,
                                                            calm_clip,
                                                            lively_clip, pool):
    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lively_clip, src / LIVELY)
    shutil.copy2(calm_clip, src / CALM)
    job = make_job(src=src, job_id="autoproj", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=10.0, title="AUTO"))
    (Path(job.build_dir) / "agent_settings.json").write_text(
        job.settings.model_dump_json(indent=2))
    llm = llm_returning_plan(
        [{"clip_id": LIVELY, "target_seconds": 5.0, "reason": "open"},
         {"clip_id": CALM, "target_seconds": 5.0, "reason": "settle"}])

    final = run_job(job, llm=llm, pool=pool, root=agent_root)
    assert final["status"] == "done"

    # nobody ran `python -m agents.compat`
    jj = Path(job.build_dir) / "job.json"
    ep = Path(job.build_dir) / "edit_plan.json"
    assert jj.exists(), "job.json was not projected automatically"
    assert ep.exists(), "edit_plan.json was not projected automatically"
    assert json.loads(jj.read_text())["status"] == "done"


@pytest.mark.slow
def test_an_escalated_run_still_projects(agent_root, calm_clip, lively_clip,
                                         pool):
    """A build that failed its own review must still be visible."""
    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    job = make_job(src=src, job_id="escproj", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=10.0, title="ESC"))
    (Path(job.build_dir) / "agent_settings.json").write_text(
        job.settings.model_dump_json(indent=2))
    llm = llm_returning_plan(
        [{"clip_id": CALM, "target_seconds": 4.0, "reason": "slow"},
         {"clip_id": LIVELY, "target_seconds": 4.0, "reason": "payoff"}])

    final = run_job(job, llm=llm, pool=pool, pass_threshold=101.0,
                    root=agent_root)
    assert final["status"] == "escalated"

    projected = json.loads((Path(job.build_dir) / "job.json").read_text())
    assert projected["status"] == "error"
    assert "did not pass it" in projected["error"]
    assert projected["output"], "the video exists and must stay downloadable"


def test_a_projection_failure_cannot_fail_the_build(monkeypatch, tmp_path):
    """The rule that matters: job.json is a convenience, not the record."""
    from agents import runner
    from agents.artifacts import JobState

    def boom(*a, **kw):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(compat, "project", boom)
    job = JobState(job_id="x", src_dir=str(tmp_path),
                   build_dir=str(tmp_path / "outputs" / "x"))
    runner._auto_project(job, None)          # must not raise


# ------------------------------------------------ 4. live status (C)

def test_a_running_projection_reports_running_with_real_stages(tmp_path):
    from agents.trace import TraceRecorder

    build = tmp_path / "outputs" / "livejob"
    build.mkdir(parents=True)
    (build / "agent_settings.json").write_text(
        JobSettings(title="LIVE").model_dump_json())
    db = compat.shared_db_path(root=tmp_path)
    ArtifactStore(db)
    t = TraceRecorder(db)
    with t.node("livejob", "clip_analysis", "clip_analysis") as span:
        span.tool("pipeline.probe")

    payload = compat.project_running("livejob", root=tmp_path, node="render")
    assert payload is not None
    assert payload["status"] == "running"
    assert payload["output"] is None
    st = payload["stages"]
    assert st["probe"]["status"] == "done", "clip_analysis ran"
    assert st["cut"]["status"] == "pending", "render has not run yet"
    assert st["vision"]["status"] == "skipped"


def test_a_terminal_projection_is_never_dragged_back_to_running(published_job):
    """The ordering guarantee: a late heartbeat must not un-finish a build."""
    job, _ = published_job
    target = Path(job.build_dir) / "job.json"
    before = json.loads(target.read_text())
    assert before["status"] == "done"

    out = compat.project_running(job.job_id, node="render")
    assert out is None, "a heartbeat overwrote a finished job"
    assert json.loads(target.read_text())["status"] == "done"


def test_a_heartbeat_never_touches_a_legacy_job(tmp_path):
    build = tmp_path / "outputs" / "legacyjob"
    build.mkdir(parents=True)
    legacy = {"id": "legacyjob", "settings": {"title": "REAL"},
              "status": "running"}
    (build / "job.json").write_text(json.dumps(legacy))
    assert compat.project_running("legacyjob", root=tmp_path) is None
    assert json.loads((build / "job.json").read_text()) == legacy


@pytest.mark.slow
def test_events_reports_stage_transitions_for_an_agent_job(client,
                                                           published_job):
    """/events on a finished agent job: one snapshot, then final."""
    job, _ = published_job
    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as r:
        assert r.status_code == 200
        kinds, payloads = [], []
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            m = json.loads(line[6:])
            kinds.append(m.get("kind"))
            payloads.append(m)
            if m.get("kind") == "final":
                break
    assert kinds[0] == "snapshot"
    assert "final" in kinds
    final = payloads[-1]["job"]
    assert final["status"] == "done"
    assert final["progress"] == 100.0
    assert len(final["stages"]) == 10


# -------------------------------------------- 5. cancellation (D)

def test_cancel_marker_roundtrip(tmp_path):
    assert not compat.cancel_requested("j", root=tmp_path)
    assert compat.request_cancel("j", root=tmp_path)
    assert compat.cancel_requested("j", root=tmp_path)
    compat.clear_cancel_request("j", root=tmp_path)
    assert not compat.cancel_requested("j", root=tmp_path)


def test_v2_cancel_rejects_a_non_agent_job(client):
    r = client.post("/api/v2/jobs/definitelynotajob/cancel")
    assert r.status_code == 404


def test_v2_cancel_requests_and_reports_honestly(client, published_job):
    job, _ = published_job
    try:
        r = client.post(f"/api/v2/jobs/{job.job_id}/cancel")
        assert r.status_code == 200
        body = r.json()
        assert body["cancelled"] is True
        # the job is not running in this process, so it must say so
        assert body["mechanism"] in ("cross-process (marker)", "in-process")
        assert compat.cancel_requested(job.job_id)
    finally:
        compat.clear_cancel_request(job.job_id)
    assert client.delete(f"/api/v2/jobs/{job.job_id}/cancel").status_code == 200


@pytest.mark.slow
def test_a_cancel_request_kills_a_real_running_render(agent_root, calm_clip,
                                                      lively_clip):
    """End to end: a marker written from outside kills the real encoder."""
    from agents.render_pool import MediaPool

    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    job = make_job(src=src, job_id="killme", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="1080p", quality="master",
                                        target_seconds=10.0, title="KILL"))
    (Path(job.build_dir) / "agent_settings.json").write_text(
        job.settings.model_dump_json(indent=2))

    pool = MediaPool(max_workers=2, name="killtest")
    outcome: dict = {}

    def go():
        try:
            outcome["final"] = run_job(job, pool=pool, root=agent_root)
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}"

    th = threading.Thread(target=go, daemon=True)
    th.start()

    # wait until the pool genuinely owns a process, then cancel from outside
    deadline = time.time() + 180
    while time.time() < deadline and not pool.active_processes():
        time.sleep(0.5)
    assert pool.active_processes(), "no process ever started"

    compat.request_cancel(job.job_id, root=agent_root)
    th.join(timeout=180)

    assert not th.is_alive(), "the job did not stop"
    assert pool.active_processes() == [], "a process was orphaned"
    assert "error" in outcome or outcome.get("final", {}).get("status") != "done"
    pool.close()


# ------------------------------- 6. legacy /start protection (E2)

def test_legacy_start_is_refused_for_an_agent_job(client, published_job):
    """The guard that stops a button click destroying an agent build."""
    job, _ = published_job
    before = (Path(job.build_dir) / "manifest.json").read_bytes()

    r = client.post(f"/api/jobs/{job.job_id}/start", json={})
    assert r.status_code == 409
    assert "agent pipeline" in r.json()["detail"]

    # and it really did not run
    assert (Path(job.build_dir) / "manifest.json").read_bytes() == before
    assert client.get(f"/api/jobs/{job.job_id}").json()["status"] == "done"


def test_legacy_cancel_no_longer_lies_about_an_agent_job(client,
                                                          published_job):
    job, _ = published_job
    r = client.post(f"/api/jobs/{job.job_id}/cancel")
    assert r.status_code == 409, "a no-op cancel must not report success"
    assert "/api/v2/jobs" in r.json()["detail"]


def test_a_legacy_job_is_completely_unaffected_by_the_guard(client):
    """The guard must key on the marker, not on anything a legacy job has."""
    r = client.post("/api/jobs", json={})
    jid = r.json()["id"]
    # no clips, so start fails for the ORIGINAL reason - not the guard
    r = client.post(f"/api/jobs/{jid}/start", json={})
    assert r.status_code == 400
    assert "Add at least one clip" in r.json()["detail"]
    # and cancel still behaves exactly as before
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200


def test_the_guard_reads_only_the_marker(tmp_path, monkeypatch):
    """agent_owned() must not import the agent package - the legacy app has
    to start when the agent dependencies are absent."""
    import main

    src = Path(main.__file__).read_text()
    fn = src[src.index("def agent_owned("):src.index("def serve_within(")]
    assert "import agents" not in fn
    assert "_produced_by" in fn


# ------------------------------------------------- 7. v2 surface

def test_v2_health_reports_the_agent_layer(client):
    r = client.get("/api/v2/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert set(body["agents"]) == {"clip_analysis", "editing_director",
                                   "rendering", "critic"}


def test_v2_status_exposes_artifacts_and_verdict(client, published_job):
    job, _ = published_job
    body = client.get(f"/api/v2/jobs/{job.job_id}/status").json()
    assert body["job_id"] == job.job_id
    assert body["artifacts"]["EditPlan"]
    assert body["artifacts"]["RenderResult"]
    assert body["verdict"] in ("PASS", "REVISE", "ESCALATE")
    assert body["totals"]["nodes"] > 0


def test_v2_trace_returns_the_execution_trace(client, published_job):
    job, _ = published_job
    body = client.get(f"/api/v2/jobs/{job.job_id}/trace").json()
    assert body["trace"], "no trace rows"
    assert {"node", "agent", "status"} <= set(body["trace"][0])


def test_v2_project_is_idempotent(client, published_job):
    job, _ = published_job
    a = client.post(f"/api/v2/jobs/{job.job_id}/project")
    b = client.post(f"/api/v2/jobs/{job.job_id}/project")
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()


# ------------------------- 8. a job that stops without a render

def test_finalize_flips_a_stale_running_projection(tmp_path):
    """A job cancelled during analysis produced no video, so `project()`
    refuses it - without this the last heartbeat's "running" would stand
    forever and the UI would show a build that never ends."""
    build = tmp_path / "outputs" / "stuck"
    build.mkdir(parents=True)
    running = {
        "id": "stuck", "settings": {"_produced_by": compat.MARKER},
        "created": 1.0, "status": "running", "error": None,
        "started": 1.0, "finished": None,
        "stages": {"probe": {"id": "probe", "status": "done"},
                   "measure": {"id": "measure", "status": "running"}},
        "output": None, "captions": [], "voice": None, "script": "",
    }
    (build / "job.json").write_text(json.dumps(running))

    out = compat.finalize("stuck", tmp_path, status="cancelled",
                          error="cancelled before a video was produced")
    assert out["status"] == "cancelled"
    assert "cancelled" in out["error"]
    assert out["finished"] is not None
    assert out["stages"]["measure"]["status"] == "error"
    assert out["stages"]["probe"]["status"] == "done", "finished work kept"


def test_finalize_leaves_an_already_terminal_projection_alone(published_job):
    job, _ = published_job
    out = compat.finalize(job.job_id, status="error", error="nope")
    assert out["status"] == "done", "a finished build was overwritten"
    assert out["error"] is None


def test_finalize_never_touches_a_legacy_job(tmp_path):
    build = tmp_path / "outputs" / "legacyfin"
    build.mkdir(parents=True)
    legacy = {"id": "legacyfin", "settings": {"title": "REAL"},
              "status": "running"}
    (build / "job.json").write_text(json.dumps(legacy))
    assert compat.finalize("legacyfin", tmp_path) is None
    assert json.loads((build / "job.json").read_text()) == legacy


@pytest.mark.slow
def test_a_cancelled_run_is_not_left_reading_running(agent_root, calm_clip,
                                                     lively_clip):
    """End to end: cancel a real job during analysis, and the projection must
    end terminal rather than stuck."""
    from agents.render_pool import MediaPool

    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    job = make_job(src=src, job_id="stalecheck", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="1080p", quality="master",
                                        target_seconds=10.0, title="STALE"))
    (Path(job.build_dir) / "agent_settings.json").write_text(
        job.settings.model_dump_json(indent=2))

    pool = MediaPool(max_workers=2, name="stale")
    done = threading.Event()

    def go():
        try:
            run_job(job, pool=pool, root=agent_root)
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    deadline = time.time() + 120
    while time.time() < deadline and not pool.active_processes():
        time.sleep(0.25)
    compat.request_cancel(job.job_id, root=agent_root)
    assert done.wait(180), "the job never stopped"

    jj = Path(job.build_dir) / "job.json"
    if jj.exists():
        status = json.loads(jj.read_text())["status"]
        assert status != "running", (
            "the projection is stuck on running - the UI would show a build "
            "that never ends")
    pool.close()
