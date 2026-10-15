"""
Additive /api/v2 surface for agent-owned jobs.

Everything here is NEW. No existing route changes behaviour, and the legacy
app runs unchanged if this router is never mounted - `server/main.py` includes
it inside a try/except precisely so a missing agent dependency cannot stop the
existing application from starting.

WHY A SEPARATE SURFACE
----------------------
The legacy endpoints describe a job that is a thread inside the web server.
An agent job is a LangGraph run that usually lives in a different process
entirely (the CLI), owns its state in the shared artifact store, and has real
process-tree cancellation through the scheduler. None of that fits the legacy
vocabulary, and bending the legacy routes to cover it would change behaviour
for ordinary builds. So the agent-shaped questions get agent-shaped routes,
and the legacy routes keep serving the projection.

CANCELLATION ACROSS PROCESSES
-----------------------------
The scheduler can only tree-kill processes ITS OWN runtime owns. When the web
server did not start the job it has no such runtime, so cancellation is a
two-part mechanism:

  1. a marker file both processes can see (compat.request_cancel)
  2. the running job's CancelWatcher polls it once a second and performs the
     REAL kill through its own scheduler

so a cancel issued from the browser reaches an ffmpeg started by the CLI,
within about a second. When the job IS running in this process, the kill also
happens immediately and synchronously.
"""

from __future__ import annotations

import shutil
import threading
import time
import traceback
from pathlib import Path

import pydantic
from fastapi import APIRouter, HTTPException

from . import compat
from .artifacts import (SCHEMA_VERSION, ClipAnalysisSet, CriticResult,
                        EditPlan, JobSettings, RenderResult)
from .base import agent_registry
from .store import ArtifactStore, resolve_job_db, shared_store
from .tools.llm_tools import LLMClient
from .trace import TraceRecorder

router = APIRouter(prefix="/api/v2", tags=["agents"])


def _require_agent_job(job_id: str) -> None:
    if not compat.is_agent_job(job_id):
        raise HTTPException(
            404, f"{job_id} is not an agent-built job; use the existing "
                 f"/api/jobs endpoints for legacy builds")


# --------------------------------------------------------------- listing

@router.get("/jobs")
def list_agent_jobs():
    """Agent-built jobs only. The legacy /api/jobs still lists everything."""
    out = []
    for jid in compat.agent_jobs():
        db, source = resolve_job_db(jid)
        store = ArtifactStore(db)
        render = store.get(jid, "RenderResult", RenderResult)
        critic = store.get(jid, "CriticResult", CriticResult)
        out.append({
            "id": jid,
            "store": source,
            "attempts": len(store.all_revisions(jid, "RenderResult",
                                                RenderResult)),
            "verdict": critic.payload.verdict if critic else None,
            "score": critic.payload.overall_score if critic else None,
            "video": render.payload.video_path if render else None,
            "projected": (Path(compat.job_paths(jid)[1]) / "job.json").exists(),
        })
    return {"jobs": out}


def _saved_settings(job_id: str) -> dict | None:
    """The JobSettings a run was started with, if the runner wrote them."""
    try:
        from . import runner
        _src, build = runner.job_paths(job_id)
        f = build / "agent_settings.json"
        if not f.exists():
            return None
        import json
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:                               # noqa: BLE001
        return None


# ---------------------------------------------------------------- status

@router.get("/jobs/{job_id}/status")
def agent_status(job_id: str):
    """What the agent layer knows, unprojected and untranslated."""
    failed = _FAILED.get(job_id)
    if failed is not None and not compat.is_agent_job(job_id):
        # died before it wrote anything, so there is no job to describe -
        # report the failure itself rather than a 404 the UI reads as "not
        # started yet" and keeps polling forever
        return {"job_id": job_id, "status": "failed", "failed": failed,
                "settings": _saved_settings(job_id)}
    _require_agent_job(job_id)
    db, source = resolve_job_db(job_id)
    store = ArtifactStore(db)
    tracer = TraceRecorder(db)

    clips = store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    plan = store.get(job_id, "EditPlan", EditPlan)
    render = store.get(job_id, "RenderResult", RenderResult)
    critic = store.get(job_id, "CriticResult", CriticResult)
    rows = tracer.for_job(job_id)

    running = [r for r in rows if r["status"] == "running"]
    return {
        "job_id": job_id,
        "store": source,
        "cancel_requested": compat.cancel_requested(job_id),
        # Whether a run is in flight RIGHT NOW, asked of the thread that
        # would be doing it rather than inferred from the artifacts.
        #
        # A client could only guess before, and guessed wrong in the one
        # case that matters. Refining re-runs the graph over the same job,
        # and the previous run's CriticResult stays on disk until the new
        # one replaces it - so the moment after a refine started, `/status`
        # still answered "PASS" and the UI flipped straight back to the
        # finished video. It looked as though Send had done nothing, so the
        # person pressed it again and got "already being produced".
        #
        # `current_node` cannot stand in for this: a run killed mid-node
        # leaves its trace row saying "running" for ever.
        #
        # This answers for THIS process, which is the one the browser is
        # talking to and the one whose `produce` would refuse a second
        # run. A job being produced by some other server would read false
        # here - correctly, since this server could not report on it.
        "producing": _is_running(job_id),
        "current_node": running[-1]["node"] if running else None,
        "artifacts": {
            "ClipAnalysis": clips.artifact_id if clips else None,
            "EditPlan": plan.artifact_id if plan else None,
            "RenderResult": render.artifact_id if render else None,
            "CriticResult": critic.artifact_id if critic else None,
        },
        "revisions": {
            "EditPlan": store.next_revision(job_id, "EditPlan"),
            "RenderResult": store.next_revision(job_id, "RenderResult"),
        },
        "verdict": critic.payload.verdict if critic else None,
        "score": critic.payload.overall_score if critic else None,
        "degraded": bool(plan.degraded) if plan else None,
        "fallback_used": plan.fallback_used if plan else None,
        # What this job was actually produced with. Without it a client
        # cannot refine a run it did not start itself: "make the opening
        # faster" would arrive with every other choice reset to default,
        # silently throwing away the mood, length and quality the person
        # picked. Read from the job's own saved settings, never guessed.
        "settings": _saved_settings(job_id),
        "failed": failed,
        "totals": tracer.totals(job_id),
    }


@router.get("/jobs/{job_id}/trace")
def agent_trace(job_id: str):
    _require_agent_job(job_id)
    db, _ = resolve_job_db(job_id)
    return {"job_id": job_id, "trace": TraceRecorder(db).for_job(job_id)}


# ---------------------------------------------------------- cancellation

@router.get("/jobs/{job_id}/artifacts/{kind}")
def agent_artifact(job_id: str, kind: str):
    """One agent's finished output, as it was stored.

    `/status` reports only artifact IDs and `/trace` only what each node
    DID, so a client could see that Story ran and not a word of what it
    wrote. The result timeline said "Nothing recorded" under Story for a
    run whose six beats had shaped the entire cut - the data was in the
    store the whole time with no way to ask for it.

    Returned verbatim, with the envelope that says how far to trust it:
    `degraded` and `fallback_used` travel with the payload so a caller
    cannot present a deterministic answer as a reasoned one.
    """
    _require_agent_job(job_id)
    db, _source = resolve_job_db(job_id)
    row = ArtifactStore(db).get_raw(job_id, kind)
    if row is None:
        raise HTTPException(
            404, f"{job_id} has no {kind} artifact; it may not have run yet")
    return row


@router.get("/jobs/{job_id}/published")
def get_published_video(job_id: str):
    """Which YouTube video this build became, and how it is doing.

    Both halves, because either alone is useless: the reference without
    the numbers is a link, and the numbers without the reference belong to
    nobody. `matched` says plainly whether the exported row was found, so
    an operator who has recorded a link but not yet dropped an export is
    told that rather than shown zeroes.
    """
    _require_agent_job(job_id)
    from . import runner
    from .tools import analytics_store as AS
    _src, build = runner.job_paths(job_id)

    ref = AS.load_reference(build)
    if ref is None:
        return {"job_id": job_id, "video_id": None, "matched": False,
                "why": "no video has been recorded for this production"}

    export = AS.read_export(AS.export_dir())
    row = export.by_id(ref["video_id"])
    out = {"job_id": job_id, "video_id": ref["video_id"],
           "url": ref.get("url", ""), "recorded_at": ref.get("recorded_at"),
           "matched": row is not None,
           "export_files": export.files, "export_skipped": export.skipped}
    if row is None:
        out["why"] = (
            f"{ref['video_id']} is not in the analytics export"
            + (f" ({', '.join(export.files)})" if export.files else
               "; no CSV has been dropped in data/analytics yet"))
        return out
    out["metrics"] = {
        "views": row.views, "watch_minutes": row.watch_minutes,
        "likes": row.likes, "subscribers_gained": row.subs_gained,
        "window_days": row.window_days, "first_day": row.first_day,
        "last_day": row.last_day, "source_file": row.source_file}
    return out


@router.post("/jobs/{job_id}/published")
def set_published_video(job_id: str, body: dict | None = None):
    """Record the YouTube video this build became.

    Nothing in a build can know this: the upload happens outside the
    system entirely, so the one link between an edit and its reception has
    to be given once, by hand. Without it the Learning agent has nothing
    to learn FROM - a view count that cannot be attributed to a decision
    teaches nothing.

    A link in any shape is accepted; anything that is not a video id is
    refused rather than stored, because a wrong id would quietly credit
    one video's numbers to another video's decisions.
    """
    _require_agent_job(job_id)
    from . import runner
    from .tools import analytics_store as AS

    raw = str((body or {}).get("url") or (body or {}).get("video_id") or "")
    video_id = AS.video_id_from(raw)
    if video_id is None:
        raise HTTPException(
            422, "that is not a YouTube video link or id; paste the watch, "
                 "share or Studio URL, or the 11-character id itself")

    _src, build = runner.job_paths(job_id)
    if not build.exists():
        raise HTTPException(404, f"{job_id} has no build folder")
    rec = AS.save_reference(build, video_id, raw.strip())
    return {"job_id": job_id, **rec}


@router.delete("/jobs/{job_id}/published")
def clear_published_video(job_id: str):
    """Withdraw a reference - a link recorded against the wrong build is
    worse than none, so taking it back must be as easy as giving it."""
    _require_agent_job(job_id)
    from . import runner
    from .tools import analytics_store as AS
    _src, build = runner.job_paths(job_id)
    return {"job_id": job_id, "cleared": AS.clear_reference(build)}


@router.post("/jobs/{job_id}/learning/approve")
def approve_learning(job_id: str):
    """Grant the one gate that no amount of data opens.

    Six of the seven gates are arithmetic over stored reports and open
    themselves when the evidence is there. This one does not, and must
    not: it is the step at which a person looks at a measured difference
    and decides whether to act on it. A system that approved its own
    conclusions would have six checks and no judgement.

    The approval is for THIS job's report. It does not apply anywhere
    else and it is not remembered across a restart, because an approval
    that outlives the evidence it was given for is not an approval.
    """
    _require_agent_job(job_id)
    from .learning import approve as _approve
    fresh = _approve(job_id)
    return {"job_id": job_id, "approved": True, "already": not fresh,
            "note": ("the Learning report is rebuilt on the next run of "
                     "that node; approval alone changes nothing on disk")}


@router.post("/jobs/{job_id}/cancel")
def cancel_agent_job(job_id: str):
    """Cancel an agent job, killing its ffmpeg process tree.

    Reports honestly which mechanism did the work, because they differ: an
    in-process kill is immediate, a cross-process one takes up to the
    watcher's poll interval.
    """
    _require_agent_job(job_id)

    requested = compat.request_cancel(job_id)
    if not requested:
        raise HTTPException(500, "could not write the cancellation request")

    killed = 0
    mechanism = "requested"
    try:
        from .graph import runtime_for
        rt = runtime_for(job_id)
        report = rt.scheduler.cancel(job_id)
        killed = report["processes_killed"]
        mechanism = "in-process"
    except Exception:                                  # noqa: BLE001
        # not running in this process - the marker is the whole mechanism and
        # the owning process will act on it at its next poll
        mechanism = "cross-process (marker)"

    return {
        "job_id": job_id,
        "cancelled": True,
        "mechanism": mechanism,
        "processes_killed": killed,
        "note": ("the job's own scheduler performs the kill; a job running in "
                 "another process picks the request up within about a second"),
    }


@router.delete("/jobs/{job_id}/cancel")
def clear_agent_cancel(job_id: str):
    """Withdraw a cancellation request that has not been acted on yet."""
    _require_agent_job(job_id)
    compat.clear_cancel_request(job_id)
    return {"job_id": job_id, "cancel_requested": False}


# ------------------------------------------------------------- produce

# One run per job at a time. The graph writes into outputs/<id>; two runs
# over the same folder would fight over the manifest and the segments.
_RUNNING: dict[str, threading.Thread] = {}

# A production that dies before its first artifact leaves NOTHING behind:
# no agent-job marker, so `/status` 404s, and no trace rows, so there is
# nothing to show. Swallowing that exception made a dead run and a slow run
# look identical - the UI span indefinitely on "starting" while the real
# cause (ffprobe not on PATH) sat in a thread nobody could see. This holds
# the reason until the next produce for the same job replaces it.
_FAILED: dict[str, dict] = {}


def _refresh_legacy_record(job_id: str) -> None:
    """Drop the stale in-memory Job so the legacy routes re-read job.json.

    `POST /api/jobs` puts a Job in `jobs._JOBS` and the agent path then
    rewrites outputs/<id>/job.json on disk, behind that object's back. The
    legacy routes read the in-memory copy, which still says `status: draft`
    and `output: None` - so `/api/jobs/{id}/download` answered 409 for the
    whole life of the server process even once the projection had landed,
    and the result page's <video>, which streams from that same route,
    played nothing. Restarting the server "fixed" it, which is what made the
    cause hard to see.

    `jobs.get()` restores from disk whenever the id is absent, so evicting
    is enough; nothing has to be rebuilt here. Swallowed deliberately - a
    verified video must not be reported as failed because a cache could not
    be cleared.
    """
    try:
        import jobs as jobs_mod
        with jobs_mod._LOCK:
            jobs_mod._JOBS.pop(job_id, None)
    except Exception:                               # noqa: BLE001
        pass
_RUN_LOCK = threading.Lock()


def _is_running(job_id: str) -> bool:
    t = _RUNNING.get(job_id)
    return bool(t and t.is_alive())


@router.post("/jobs/{job_id}/produce", status_code=202)
def produce(job_id: str, body: dict | None = None):
    """Start a real agent run over clips already uploaded for this job.

    The browser had no way to do this: runs started only from
    `python -m agents.runner run`, and the legacy `/api/jobs/{id}/start`
    refuses agent-built jobs on purpose. Without this route a "create my
    video" button could only drive the old pipeline while the UI implied
    the agents were working, which is the one thing it must not do.

    The body is whatever the person asked for. It is validated into
    `JobSettings` by pydantic, so an unsupported style, mood or resolution
    is rejected here rather than being quietly dropped later. Returns 202:
    the run takes minutes and is followed with `/status` and `/trace`.
    """
    from . import runner                       # local: keeps import cost off
    src, _build = runner.job_paths(job_id)     # uploads/<id>, outputs/<id>
    if not src.exists() or not any(src.iterdir()):
        raise HTTPException(
            409, f"no clips uploaded for {job_id}; upload them first via "
                 f"POST /api/jobs/{job_id}/clips")

    with _RUN_LOCK:
        if _is_running(job_id):
            raise HTTPException(409, f"{job_id} is already being produced")

        try:
            settings = JobSettings(**(body or {}))
        except pydantic.ValidationError as exc:
            # name the field the person actually got wrong
            first = exc.errors()[0]
            where = ".".join(str(x) for x in first.get("loc", ())) or "body"
            raise HTTPException(422, f"{where}: {first.get('msg', 'invalid')}")

        job = runner.make_job(src=src, job_id=job_id, settings=settings)
        # persist what was asked for. The CLI already does this; the route
        # did not, so `/status` had no settings to report and a refine
        # started from a reloaded page lost every choice the person made.
        runner.save_settings(job)

        # The browser reached here via `POST /api/jobs`, which already wrote
        # a legacy job.json. `compat.project()` refuses to overwrite one at
        # the end of the run, so the finished video was never published into
        # it: `output` stayed null and `/api/jobs/{id}/download` answered 409
        # for every video made through the UI. Claiming the empty placeholder
        # now - and only while it is still empty - lets that projection land.
        if not compat.claim_placeholder(job_id):
            raise HTTPException(
                409, f"{job_id} already holds a build from the legacy "
                     f"pipeline; create a new job rather than overwriting it")

        _FAILED.pop(job_id, None)           # a retry starts clean

        def _go():
            try:
                runner.run_job(job)
            except Exception as exc:            # noqa: BLE001
                # Still must not take the server down - but it must not
                # vanish either. `/status` reads this so the person is told
                # the run stopped and why, instead of watching a spinner.
                _FAILED[job_id] = {
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "traceback": traceback.format_exc()[-2000:],
                    "at": time.time(),
                }
            finally:
                _refresh_legacy_record(job_id)

        t = threading.Thread(target=_go, name=f"produce-{job_id}", daemon=True)
        _RUNNING[job_id] = t
        t.start()

    return {"job_id": job_id, "status": "running",
            "settings": settings.model_dump(mode="json"),
            "follow": {"status": f"/api/v2/jobs/{job_id}/status",
                       "trace": f"/api/v2/jobs/{job_id}/trace"}}


@router.delete("/jobs/{job_id}")
def delete_agent_job(job_id: str):
    """Remove a production completely: its files AND its artifacts.

    The legacy `DELETE /api/jobs/{id}` removes the two folders and the
    in-memory record, but the agent artifacts and the trace live in the
    SHARED store, so they survive it - the job keeps appearing in
    `/api/v2/jobs`, which lists from that store, and so in "My videos". A
    half-deleted production that still shows up is worse than none.

    Refuses while the job is being produced: the running graph would write
    its next artifact straight back into the store and leave the folders
    half-rebuilt. Cancel it first.

    This cannot be undone, so the caller is expected to have asked.
    """
    from . import runner
    with _RUN_LOCK:
        if _is_running(job_id):
            raise HTTPException(
                409, f"{job_id} is being produced right now; cancel it "
                     f"first with POST /api/v2/jobs/{job_id}/cancel")

    db, _source = resolve_job_db(job_id)
    artifacts = traces = 0
    try:
        artifacts = ArtifactStore(db).purge_job(job_id)
        traces = TraceRecorder(db).purge_job(job_id)
    except Exception:                                   # noqa: BLE001
        pass            # the folders still go; a stale row is not worth 500

    src, build = runner.job_paths(job_id)
    removed = []
    for d in (build, src):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)

    _FAILED.pop(job_id, None)
    _refresh_legacy_record(job_id)       # the legacy list reads from disk

    return {"deleted": job_id, "artifacts": artifacts, "trace_rows": traces,
            "folders": len(removed)}


@router.get("/produce/options")
def produce_options():
    """What the engine can actually honour, so the UI cannot offer more.

    Derived from the JobSettings literals, which are themselves checked
    against `extras.MOODS` by a parity test. A UI built from this list
    cannot drift into offering a style or a mood that does not exist.
    """
    import typing
    def opts(name):
        ann = JobSettings.model_fields[name].annotation
        return [a for a in typing.get_args(ann) if isinstance(a, str)]
    return {
        "style": opts("style"), "pacing": opts("pacing"),
        "music": opts("music"), "captions": opts("captions"),
        "voiceover": opts("voiceover"), "framing": opts("framing"),
        "resolution": opts("resolution"),
        "quality": opts("quality"), "orientation": opts("orientation"),
        "notes": {
            "voiceover": "text only - nothing in this project synthesises a "
                         "voice, so 'text' writes narration rather than "
                         "speaking it",
            "music": "these are the moods the music engine actually plays",
        },
    }


# ----------------------------------------------------------- projection

@router.post("/jobs/{job_id}/project")
def project_agent_job(job_id: str):
    """Re-publish a job into the legacy compatibility layer.

    Normally automatic at the end of a run; exposed for the case where a job
    finished before this was wired up, or a projection was deleted.
    """
    _require_agent_job(job_id)
    try:
        payload = compat.project(job_id)
    except compat.NotAnAgentJob as exc:
        raise HTTPException(409, str(exc))
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    return {"job_id": job_id, "status": payload["status"],
            "output": (payload.get("output") or {}).get("file")}


@router.get("/health")
def agent_health():
    """Is the agent layer present, what does it hold, and can it reason?

    The agent list and the counts are DERIVED from the package itself, not
    typed in here. The previous version of this endpoint returned a literal
    four-id list and a literal phase number, and kept returning them while
    the system grew to eighteen agents - a hardcoded answer cannot go stale
    loudly, so it goes stale silently. `reasoning_enabled` is reported for
    the same reason: a reader must be able to tell a model-backed run from
    the deterministic fallbacks without reading the artifacts.
    """
    store = shared_store()
    registry = agent_registry()
    llm = LLMClient().describe()
    return {
        "ok": True,
        "shared_store": store.db_path,
        "jobs_in_store": len(store.jobs()),
        "agent_jobs_on_disk": compat.agent_jobs(),
        "agents": list(registry),
        "agent_detail": list(registry.values()),
        "agent_count": len(registry),
        "reasoning_agents": sum(1 for a in registry.values() if a["uses_llm"]),
        "reasoning_enabled": bool(llm["enabled"]),
        "model": llm.get("model"),
        "provider": llm.get("provider"),
        "why_no_model": llm.get("why_disabled"),
        "schema_version": SCHEMA_VERSION,
    }


__all__ = ["router"]
