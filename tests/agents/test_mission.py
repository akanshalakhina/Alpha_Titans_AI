"""Mission Control: what a viewer is told must be what the trace supports.

The screen answers "which agent is working now and what is it doing". A wrong
answer is worse than none, so these tests pin the cases where it is easiest to
lie: a planned agent shown as working, a dead run shown as running, a fallback
shown as a success, a checker that only ever passes.

Trace rows are seeded directly, one per node attempt, exactly as the graph
writes them. The checkers read artifacts through `store.get_raw`, so a small
stand-in for the store is enough to exercise them without a render.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import mission, teams as T                  # noqa: E402
from agents.artifacts import new_id                     # noqa: E402
from agents.store import ArtifactStore                  # noqa: E402
from agents.trace import TraceRecorder, TraceSpan       # noqa: E402

JOB = "missionjob"


@pytest.fixture
def tracer(tmp_path):
    return TraceRecorder(tmp_path / "state.db")


def seed(tracer, node, *, status="ok", tools=(), notes=(), attempt=1,
         fallback=None, outputs=("art_1",), started=None, job=JOB):
    """One trace row, as the graph writes it."""
    w = T.worker_for_node(node)
    span = TraceSpan(recorder=tracer, trace_id=new_id("tr"), job_id=job,
                     node=node, agent=(w.agent_id if w else node),
                     attempt=attempt, input_artifacts=[],
                     seq=tracer._next_seq(job))
    if started is not None:
        span.started_at = started
    span._open()
    for n in notes:
        span.note(n)
    for name, detail in tools:
        span.tool(name, **detail)
    if fallback:
        span.fallback(fallback)
    for o in outputs:
        span.output(o)
    if status == "ok":
        span.ok()
    elif status == "error":
        span.fail("boom: the model reply was unusable")
    elif status == "skipped":
        span.skipped("no key")
    return span


class FakeStore:
    """Just enough of the store for the checkers: `get_raw`."""

    def __init__(self, **by_kind):
        self.by_kind = by_kind

    def get_raw(self, job_id, kind, revision=None):
        return self.by_kind.get(kind)


def snap(tracer, store=None, **kw):
    return mission.build_snapshot(JOB, tracer.for_job(JOB), store, **kw)


def team(s, tid):
    return next(t for t in s["teams"] if t["id"] == tid)


def worker(s, tid, key):
    return next(w for w in team(s, tid)["workers"] if w["key"] == key)


# ---------------------------------------------------------- an empty job

def test_a_job_with_no_trace_is_queued_and_nothing_is_working(tracer):
    s = snap(tracer)
    assert s["status"] == "queued"
    assert s["now"] is None and s["feed"] == []
    for t in s["teams"]:
        for w in t["workers"]:
            assert w["state"] == ("waiting" if w["built"] else "planned")


def test_a_team_of_only_planned_workers_is_planned_not_waiting(tracer):
    s = snap(tracer)
    assert team(s, "insight")["state"] == "planned"
    assert team(s, "insight")["checker"]["state"] == "planned"
    assert "none built yet" in team(s, "insight")["lead"]["summary"]


# ----------------------------------------------------------- working now

def test_a_running_row_is_working_and_says_what_it_is_doing(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision", status="running",
         tools=[("select.candidate_windows", {"clip": "01.MOV"})])
    s = snap(tracer, producing=True)

    w = worker(s, "footage", "vision")
    assert w["state"] == "working"
    assert w["doing"].startswith("tool: select.candidate_windows")
    assert "clip=01.MOV" in w["doing"]
    assert team(s, "footage")["state"] == "working"
    assert s["status"] == "running"
    assert s["now"]["worker"] == "Vision"
    assert s["now"]["team"] == "footage"


def test_a_planned_worker_is_never_shown_as_working(tracer):
    seed(tracer, "clip_analysis", status="running")
    s = snap(tracer, producing=True)
    planned = [w for t in s["teams"] for w in t["workers"] if not w["built"]]
    assert planned and all(w["state"] == "planned" for w in planned)
    assert all("doing" not in w for w in planned)


def test_two_agents_can_work_at_once(tracer):
    """Vision and Research run in parallel after clip_analysis."""
    seed(tracer, "clip_analysis")
    seed(tracer, "vision", status="running")
    seed(tracer, "research", status="running")
    s = snap(tracer, producing=True)
    assert worker(s, "footage", "vision")["state"] == "working"
    assert worker(s, "creative", "research")["state"] == "working"


def test_a_working_agent_with_only_a_note_shows_the_note(tracer):
    seed(tracer, "story", status="running", notes=["drafting six beats"])
    s = snap(tracer, producing=True)
    assert worker(s, "creative", "story")["doing"] == "drafting six beats"


def test_a_working_agent_with_nothing_yet_says_it_is_starting(tracer):
    seed(tracer, "story", status="running")
    assert worker(snap(tracer, producing=True), "creative",
                  "story")["doing"] == "starting"


# ----------------------------------------------------------- done, failed

def test_finished_workers_make_a_finished_team_with_a_lead_summary(tracer):
    seed(tracer, "animation")
    seed(tracer, "transition")
    seed(tracer, "caption")
    seed(tracer, "editing_director")
    s = snap(tracer, producing=True)
    t = team(s, "edit")
    assert t["state"] == "done"
    assert t["lead"]["summary"].startswith("4 of 4 workers done")


def test_an_essential_failure_fails_the_team_and_the_job(tracer):
    seed(tracer, "clip_analysis", status="error")
    s = snap(tracer)
    assert team(s, "footage")["state"] == "failed"
    assert s["status"] == "failed"
    assert worker(s, "footage", "clip_analysis")["error"].startswith("boom")


def test_an_optional_failure_is_not_a_failed_team(tracer):
    """Vision is never blocking: the graph carries on without it."""
    seed(tracer, "clip_analysis")
    seed(tracer, "vision", status="error")
    s = snap(tracer, producing=True)
    assert worker(s, "footage", "vision")["state"] == "failed"
    assert team(s, "footage")["state"] != "failed"


def test_a_retried_node_reports_its_retries(tracer):
    seed(tracer, "story", status="error")
    seed(tracer, "story")
    w = worker(snap(tracer, producing=True), "creative", "story")
    assert w["state"] == "done" and w["tries"] == 2 and w["rounds"] == 1


def test_the_revise_loop_counts_rounds(tracer):
    seed(tracer, "editing_director", attempt=1)
    seed(tracer, "editing_director", attempt=2)
    s = snap(tracer, producing=True)
    assert worker(s, "edit", "editing_director")["rounds"] == 2
    assert s["totals"]["rounds"] == 2


# ----------------------------------------------------------- stale rows

def test_a_row_killed_mid_node_is_interrupted_not_working(tracer):
    long_ago = time.time() - 10 * 3600
    seed(tracer, "render", status="running", started=long_ago)
    s = snap(tracer)                      # nobody here is running this job
    assert worker(s, "render", "render")["state"] == "interrupted"
    assert team(s, "render")["state"] == "interrupted"
    assert s["status"] == "stopped"


def test_a_stale_row_does_not_inflate_the_run_time(tracer):
    long_ago = time.time() - 10 * 3600
    seed(tracer, "render", status="running", started=long_ago)
    # counted to its last sign of life, not to "now"
    assert snap(tracer)["totals"]["wall_s"] < 60


def test_a_slow_but_live_node_is_still_working(tracer):
    """A render may be quiet for many minutes. Only its own deadline makes
    it stale."""
    seed(tracer, "render", status="running", started=time.time() - 20 * 60)
    s = snap(tracer)
    assert worker(s, "render", "render")["state"] == "working"
    assert s["status"] == "running"


def test_a_job_that_ended_marks_the_unrun_workers_as_not_run(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "learning")
    s = snap(tracer)
    assert s["status"] == "done"
    assert worker(s, "publish", "title")["state"] == "not_run"


# ---------------------------------------------------------------- checker

def test_a_checker_is_pending_until_the_team_has_finished(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision", status="running")
    c = team(snap(tracer, producing=True), "footage")["checker"]
    assert c["state"] == "pending" and c["checks"] == []


def test_a_checker_passes_clean_work(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision")
    store = FakeStore(ClipAnalysis={"payload": {"clips": [
        {"usability": {"usable": True}}, {"usability": {"usable": True}}]}})
    c = team(snap(tracer, store, producing=True), "footage")["checker"]
    assert c["state"] == "pass", c
    assert any(k["name"] == "clips_found" and "2 clips" in k["detail"]
               for k in c["checks"])


def test_a_fallback_is_a_warning_never_a_pass(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision", fallback="measurements only (no model output)")
    store = FakeStore(ClipAnalysis={"payload": {"clips": [
        {"usability": {"usable": True}}]}})
    s = snap(tracer, store, producing=True)
    c = team(s, "footage")["checker"]
    assert c["state"] == "warn"
    nd = next(k for k in c["checks"] if k["name"] == "not_degraded")
    assert "Vision used a fallback" in nd["detail"]
    assert team(s, "footage")["degraded"] is True


def test_a_degraded_artifact_is_a_warning(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision")
    store = FakeStore(
        ClipAnalysis={"payload": {"clips": [{}]}, "degraded": False},
        VisionAnalysis={"payload": {}, "degraded": True})
    c = team(snap(tracer, store, producing=True), "footage")["checker"]
    assert c["state"] == "warn"
    assert any("Vision is marked degraded" in k["detail"]
               for k in c["checks"] if k["name"] == "not_degraded")


def test_a_checker_fails_when_no_clips_were_found(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision")
    store = FakeStore(ClipAnalysis={"payload": {"clips": []}})
    c = team(snap(tracer, store, producing=True), "footage")["checker"]
    assert c["state"] == "fail"


def test_a_checker_fails_when_the_artifact_is_missing(tracer):
    seed(tracer, "clip_analysis")
    seed(tracer, "vision")
    c = team(snap(tracer, FakeStore(), producing=True), "footage")["checker"]
    assert c["state"] == "fail"
    assert any(k["name"] == "clips_found" and k["state"] == "fail"
               for k in c["checks"])


def test_a_worker_that_recorded_no_artifact_fails_the_check(tracer):
    seed(tracer, "story", outputs=())
    seed(tracer, "content_strategy")
    seed(tracer, "research")
    c = team(snap(tracer, producing=True), "creative")["checker"]
    assert c["state"] == "fail"
    assert any(k["name"] == "outputs_recorded" and "Story" in k["detail"]
               for k in c["checks"])


def test_the_render_check_looks_for_the_video_on_disk(tracer, tmp_path):
    seed(tracer, "render")
    real = tmp_path / "out.mp4"
    real.write_bytes(b"x")
    ok = FakeStore(RenderResult={"payload": {"video_path": str(real)}})
    gone = FakeStore(RenderResult={"payload": {
        "video_path": str(tmp_path / "missing.mp4")}})
    c1 = team(snap(tracer, ok, producing=True), "render")["checker"]
    c2 = team(snap(tracer, gone, producing=True), "render")["checker"]
    assert c1["state"] == "pass"
    assert c2["state"] == "warn"
    assert "not found on this machine" in c2["checks"][-1]["detail"]


def test_the_quality_check_reads_the_critics_verdict(tracer):
    seed(tracer, "visual_review")        # always runs before the critic
    seed(tracer, "critic")
    passed = FakeStore(CriticResult={"payload": {
        "verdict": "PASS", "overall_score": 91.0}})
    revise = FakeStore(CriticResult={"payload": {
        "verdict": "REVISE", "overall_score": 52.0}})
    assert team(snap(tracer, passed, producing=True),
                "quality")["checker"]["state"] == "pass"
    assert team(snap(tracer, revise, producing=True),
                "quality")["checker"]["state"] == "warn"


# ------------------------------------------------------------------- feed

def test_the_feed_is_in_time_order_and_carries_real_tool_calls(tracer):
    seed(tracer, "clip_analysis",
         tools=[("pipeline.probe", {"stage": "probe"}),
                ("pipeline.measure", {"clips": 2})],
         notes=["deadline 3600s"])
    seed(tracer, "vision")
    feed = snap(tracer, producing=True)["feed"]
    times = [e["t"] for e in feed]
    assert times == sorted(times)
    texts = [e["text"] for e in feed]
    assert any(t.startswith("pipeline.probe") for t in texts)
    assert "started" in texts and "finished" in texts
    assert {e["kind"] for e in feed} >= {"start", "note", "tool", "ok"}


def test_a_finished_team_adds_a_lead_line_to_the_feed(tracer):
    seed(tracer, "animation")
    seed(tracer, "transition")
    seed(tracer, "caption")
    seed(tracer, "editing_director")
    feed = snap(tracer, producing=True)["feed"]
    lead = [e for e in feed if e["kind"] == "lead"]
    assert lead and lead[0]["worker"] == "Edit Lead"
    assert "Edit complete" in lead[0]["text"]


def test_a_failure_appears_in_the_feed_as_an_error(tracer):
    seed(tracer, "vision", status="error")
    kinds = [e["kind"] for e in snap(tracer)["feed"]]
    assert "error" in kinds


def test_an_unknown_node_does_not_break_the_snapshot(tracer):
    seed(tracer, "some_future_node")
    seed(tracer, "clip_analysis")
    s = snap(tracer, producing=True)
    assert any(e["team"] == "other" for e in s["feed"])
    assert worker(s, "footage", "clip_analysis")["state"] == "done"


def test_the_feed_is_capped(tracer):
    for _ in range(mission.FEED_LIMIT):
        seed(tracer, "story", tools=[("t", {}), ("t", {})])
    assert len(snap(tracer, producing=True)["feed"]) == mission.FEED_LIMIT


# -------------------------------------------------------------- signature

def test_the_signature_changes_when_anything_visible_changes(tracer):
    span = seed(tracer, "story", status="running")
    a = mission.signature(tracer.for_job(JOB))
    assert a == mission.signature(tracer.for_job(JOB))      # stable
    span.tool("story.draft", beats=6)
    b = mission.signature(tracer.for_job(JOB))
    span.ok()
    c = mission.signature(tracer.for_job(JOB))
    assert len({a, b, c}) == 3


def test_the_snapshot_is_json_serialisable(tracer):
    seed(tracer, "clip_analysis", tools=[("p", {"path": "x", "n": 3})])
    json.dumps(snap(tracer, producing=True))        # no default=: it must
                                                    # already be plain data


# -------------------------------------------------------------------- API

@pytest.fixture
def app(tmp_path, monkeypatch):
    """The real router over a seeded store, with nothing else mounted."""
    from fastapi import FastAPI
    from agents import api, store as store_mod

    db = tmp_path / "state.db"
    tr = TraceRecorder(db)
    seed(tr, "clip_analysis")
    seed(tr, "vision")
    seed(tr, "learning")

    monkeypatch.setattr(api, "resolve_job_db", lambda jid, *a, **k: (db, "shared"))
    monkeypatch.setattr(api.compat, "is_agent_job", lambda jid, *a, **k: jid == JOB)
    monkeypatch.setattr(store_mod, "shared_db_path", lambda *a, **k: db)
    a = FastAPI()
    a.include_router(api.router)
    return a


def test_api_teams_describes_the_hierarchy(app):
    from fastapi.testclient import TestClient
    d = TestClient(app).get("/api/v2/mission/teams").json()
    assert d["counts"]["teams"] == 9
    assert d["chief"]["label"] == "Chief"


def test_api_lists_runs_with_their_status(app):
    from fastapi.testclient import TestClient
    jobs = TestClient(app).get("/api/v2/mission/jobs").json()["jobs"]
    assert [j["id"] for j in jobs] == [JOB]
    assert jobs[0]["status"] == "done" and jobs[0]["agents"] == 3


def test_api_snapshot_for_a_job(app):
    from fastapi.testclient import TestClient
    r = TestClient(app).get(f"/api/v2/jobs/{JOB}/mission")
    assert r.status_code == 200
    s = r.json()
    assert s["job_id"] == JOB and s["status"] == "done"
    assert len(s["teams"]) == 9


def test_api_refuses_a_job_that_is_not_an_agent_job(app):
    from fastapi.testclient import TestClient
    assert TestClient(app).get(
        "/api/v2/jobs/nope/mission").status_code == 404
    assert TestClient(app).get(
        "/api/v2/jobs/nope/mission/stream").status_code == 404


def test_api_stream_sends_a_snapshot_then_ends_for_a_finished_job(app):
    from fastapi.testclient import TestClient
    with TestClient(app).stream(
            "GET", f"/api/v2/jobs/{JOB}/mission/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert "event: snapshot" in body and "event: end" in body
    data = next(l for l in body.splitlines() if l.startswith("data: {\"job"))
    assert json.loads(data[6:])["status"] == "done"


def test_the_mission_page_is_served_by_the_real_app():
    from fastapi.testclient import TestClient
    import main
    r = TestClient(main.app).get("/mission")
    assert r.status_code == 200
    assert "Mission Control" in r.text
    assert "/static/mission.js" in r.text


def test_the_page_assets_exist_and_use_no_innerhtml():
    """Every server string must go in through textContent. innerHTML with
    trace text would let a tool name or an error message inject markup."""
    web = ROOT / "frontend"
    for name in ("mission.html", "mission.css", "mission.js"):
        assert (web / name).exists(), name
    js = (web / "mission.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js
