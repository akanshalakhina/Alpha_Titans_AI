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

from pathlib import Path

from fastapi import APIRouter, HTTPException

from . import compat
from .artifacts import (SCHEMA_VERSION, ClipAnalysisSet, CriticResult,
                        EditPlan, RenderResult)
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


# ---------------------------------------------------------------- status

@router.get("/jobs/{job_id}/status")
def agent_status(job_id: str):
    """What the agent layer knows, unprojected and untranslated."""
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
        "totals": tracer.totals(job_id),
    }


@router.get("/jobs/{job_id}/trace")
def agent_trace(job_id: str):
    _require_agent_job(job_id)
    db, _ = resolve_job_db(job_id)
    return {"job_id": job_id, "trace": TraceRecorder(db).for_job(job_id)}


# ---------------------------------------------------------- cancellation

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
