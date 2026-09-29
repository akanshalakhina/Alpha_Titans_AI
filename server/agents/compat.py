"""
Compatibility facade: make an agent-built job visible to the existing app.

THE PROBLEM THIS SOLVES
-----------------------
Phase 1 produces real, verified videos in outputs/<job_id>/ - but the existing
application cannot see them. `jobs.listing()` walks the outputs folder and
calls `jobs._restore(job_id)`, which returns None when there is no `job.json`.
The agent runner writes `agent_settings.json`, not `job.json`, so every
agent-built video is invisible in the drawer and undownloadable through
`/api/jobs/<id>/download`.

HOW IT SOLVES IT
----------------
By writing the one file `_restore` looks for. Nothing in server/jobs.py,
server/main.py or the frontend is touched or needs to be: `_restore` reads a
file off disk, so producing that file is the whole integration.

    JobState + RenderResult + CriticResult + trace  ->  outputs/<id>/job.json

WHAT IS PROJECTED AND WHAT IS NOT
---------------------------------
`output` is NOT rebuilt. `extras.master_stage` already writes
outputs/<id>/output.json containing exactly the keys the frontend reads, and
that file is the verified record of what was actually produced - so it is
reused verbatim rather than re-derived, with only attempt provenance added.

The ten legacy stages are projected HONESTLY rather than all marked done:

    probe    done     clip_analysis ran build_video.py probe
    measure  done     clip_analysis ran build_video.py measure
    vision   skipped  there is no Vision Agent in Phase 1
    plan     done     editing_director selected windows
    direct   done     editing_director produced the EditPlan
    sheets   skipped  the agent path does not build contact sheets
    cut      done     rendering ran build_video.py cut
    join     done     rendering ran build_video.py join
    music    skipped  there is no Music Agent in Phase 1
    master   done     rendering ran extras.master_stage

`Job.progress()` counts both "done" and "skipped", so this reads 100% without
claiming work that never happened.

ONE JUDGEMENT CALL, STATED
--------------------------
The legacy status vocabulary is draft|running|done|error|cancelled. It has no
"produced, but did not pass review". A graph run that ESCALATED did make a
playable file, but its own critic refused it. That is projected as `error`
with the reason attached, rather than `done` - conservative rather than
overselling. `/download` is unaffected either way, because it checks `output`
and not `status`.

USAGE
-----
    python -m agents.compat --job-id phase1demo
    python -m agents.compat --all
    python -m agents.compat --job-id phase1demo --dry-run

NOT DONE HERE, ON PURPOSE: the runner does not call this automatically at the
end of a build. That hook is three lines in runner.run_job, but it is a change
to the agent runner rather than to this facade, so it is left for the step
that actually wants it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .artifacts import (ClipAnalysisSet, CriticResult, EditPlan,
                        JobSettings, RenderResult)
from .store import (ArtifactStore, resolve_job_db, shared_db_path)
from .tools.pipeline_bridge import adapter_mod, config_mod
from .trace import TraceRecorder

STATE_DB = "agent_state.db"
SETTINGS_FILE = "agent_settings.json"
JOB_FILE = "job.json"
EDIT_PLAN_FILE = "edit_plan.json"
CANCEL_FILE = "CANCEL.request"
MARKER = "agents/phase1"

# legacy stage id -> (status, which agent did the equivalent work)
STAGE_PROJECTION: dict[str, tuple[str, str]] = {
    "probe":   ("done",    "clip_analysis"),
    "measure": ("done",    "clip_analysis"),
    "vision":  ("skipped", ""),
    "plan":    ("done",    "editing_director"),
    "direct":  ("done",    "editing_director"),
    "sheets":  ("skipped", ""),
    "cut":     ("done",    "rendering"),
    "join":    ("done",    "rendering"),
    "music":   ("skipped", ""),
    "master":  ("done",    "rendering"),
}

# which trace node establishes each projected stage's timing
STAGE_NODE = {
    "probe": "clip_analysis", "measure": "clip_analysis",
    "plan": "editing_director", "direct": "editing_director",
    "cut": "render", "join": "render", "master": "render",
}


class NotAnAgentJob(Exception):
    """Raised when a folder holds no agent build to project."""


def _is_our_projection(target: Path) -> bool:
    """True only when this facade wrote the file.

    An unreadable or unparseable job.json is treated as NOT ours, so the
    cautious branch is the default: a file we cannot identify is left alone.
    """
    try:
        existing = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (existing.get("settings") or {}).get("_produced_by") == MARKER


# --------------------------------------------------------------- locating

def job_paths(job_id: str, root: Path | None = None) -> tuple[Path, Path]:
    if root is not None:
        return root / "uploads" / job_id, root / "outputs" / job_id
    return (Path(config_mod.UPLOADS_DIR) / job_id,
            Path(config_mod.OUTPUTS_DIR) / job_id)


def is_agent_job(job_id: str, root: Path | None = None) -> bool:
    """Was this job built by the agent path?

    The marker used to be "does outputs/<job>/agent_state.db exist". With one
    shared store that file no longer exists for new jobs, so the question is
    now asked of the store itself - which is also the more truthful question,
    since a job is an agent build when it has agent artifacts, not when it has
    a particular file.
    """
    db, _ = resolve_job_db(job_id, root)
    if not Path(db).exists():
        return False
    try:
        return ArtifactStore(db).has_job(job_id)
    except Exception:                            # pragma: no cover
        return False


def agent_jobs(root: Path | None = None) -> list[str]:
    """Every agent-built job that still has a folder under outputs/.

    Union of the shared store and any job that has not been migrated yet, so
    this keeps reporting pre-migration jobs.
    """
    outputs = (root / "outputs") if root is not None else Path(config_mod.OUTPUTS_DIR)
    if not outputs.exists():
        return []
    # `_agents` holds the shared store, not a job
    on_disk = {d.name for d in outputs.iterdir()
               if d.is_dir() and not d.name.startswith("_")}

    found: set[str] = set()
    shared = shared_db_path(root)
    if shared.exists():
        try:
            found |= set(ArtifactStore(shared).jobs())
        except Exception:                        # pragma: no cover
            pass
    for name in on_disk:
        if (outputs / name / STATE_DB).exists():
            found.add(name)                      # not migrated yet
    return sorted(found & on_disk)


# --------------------------------------------------------------- reading

def _read_settings(build: Path) -> JobSettings:
    p = build / SETTINGS_FILE
    if p.exists():
        try:
            return JobSettings(**json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return JobSettings()


def _created_at(build: Path, started: float | None) -> float:
    """Best available creation time for the job folder.

    The drawer sorts on this, so it has to be the job's beginning rather than
    any later milestone. Candidates are taken in order of directness and the
    earliest is used.
    """
    candidates: list[float] = []
    for name in (SETTINGS_FILE, "manifest.json"):
        p = build / name
        if p.exists():
            try:
                candidates.append(p.stat().st_mtime)
            except OSError:
                pass
    if started is not None:
        candidates.append(started)
    return min(candidates) if candidates else time.time()


def _graph_status(job_id: str, build: Path) -> dict:
    """The graph's own end state, if the checkpoint is readable.

    Best-effort on purpose: a missing or unreadable checkpoint must not stop
    a finished video being published, so the verdict falls back to the
    CriticResult artifact.
    """
    cp = build / "checkpoints.db"
    if not cp.exists():
        return {}
    try:
        from .graph import build_graph
        graph, conn = build_graph(cp)
        try:
            snap = graph.get_state({"configurable": {"thread_id": job_id}})
            return dict(snap.values or {})
        finally:
            conn.close()
    except Exception:
        return {}


# ------------------------------------------------------------ projection

def _project_settings(s: JobSettings, graph: dict) -> dict:
    """Agent settings -> the legacy settings dict the UI reads.

    Knobs Phase 1 genuinely does not have are filled with what the build
    ACTUALLY did, not with a plausible-looking default: there is no Music
    Agent so the bed really is none, no speed control so it really was 1.0x,
    and every cut really was a plain fade.
    """
    return {
        "orientation": s.orientation,
        "resolution": s.resolution,
        "quality": s.quality,
        "target_seconds": s.target_seconds,
        "title": s.title,
        "transition_duration": s.transition_duration,
        # Phase 1 has no agent for these; these are what the render did.
        "speed": "normal",
        "transition_style": "cut",
        "mood": "auto",
        "music_source": "none",
        "keep_audio": "none",
        "sequence": "hook",
        "_has_script": False,
        # provenance, so nobody mistakes this folder for a legacy build
        "_produced_by": MARKER,
        "_agent_attempts": graph.get("attempt"),
        "_agent_revisions": graph.get("revision_count"),
    }


def _project_stages(trace_rows: list[dict], live: bool = False) -> dict:
    """The ten legacy stages, with real timings where a node supplies one."""
    by_node: dict[str, dict] = {}
    for r in trace_rows:
        if r["status"] not in ("ok", "skipped"):
            continue
        cur = by_node.setdefault(r["node"], {"started": r["started_at"],
                                             "finished": r["ended_at"],
                                             "lines": 0})
        cur["started"] = min(cur["started"], r["started_at"])
        if r["ended_at"]:
            cur["finished"] = max(cur["finished"] or 0, r["ended_at"])
        cur["lines"] += len(r["tool_calls"])

    running_nodes = {r["node"] for r in trace_rows if r["status"] == "running"}

    stages: dict[str, dict] = {}
    for sid, (status, _agent) in STAGE_PROJECTION.items():
        node = STAGE_NODE.get(sid)
        t = by_node.get(node) if node else None
        if live:
            # mid-flight: a stage whose node has not run yet is pending, and
            # the node executing right now is running. Reporting the finished
            # projection's statuses here would show a half-built job as
            # complete.
            if node and node in running_nodes:
                status = "running"
            elif t is None:
                status = "pending" if status == "done" else status
        stages[sid] = {
            "id": sid,
            "status": status,
            "started": t["started"] if t else None,
            "finished": t["finished"] if t else None,
            "lines": t["lines"] if t else 0,
        }
    return stages


def _project_output(build: Path, render: RenderResult | None,
                    attempts: int) -> dict | None:
    """Reuse master_stage's own output.json; never re-derive it.

    That file is the verified record of what ffmpeg actually produced, and it
    already carries exactly the keys the frontend reads. Only provenance is
    added. If it is missing or its file has gone, the archived attempt is used
    so `/download` still resolves.
    """
    meta: dict | None = None
    p = build / "output.json"
    if p.exists():
        try:
            meta = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            meta = None

    if meta and not Path(meta.get("path", "")).exists():
        meta = None                       # canonical file has gone

    if meta is None and render is not None:
        archived = Path(render.video_path)
        if not archived.exists():
            return None
        t = render.technical
        meta = {
            "file": archived.name, "path": str(archived),
            "thumbnail": (Path(render.thumbnail_path).name
                          if render.thumbnail_path else None),
            "duration": t.duration, "frames": t.frames,
            "frames_expected": t.frames_expected, "frames_ok": t.frames_ok,
            "width": t.width, "height": t.height, "fps": t.fps,
            "video_codec": t.video_codec, "audio_codec": t.audio_codec,
            "size": t.size, "lufs": t.lufs, "black_frames": t.black_frames,
            "speed": 1.0, "segments": len(render.segments),
            "music": "silent", "keep_audio": "none",
            "has_source_audio": False,
            "orientation": ("vertical" if (t.height or 0) > (t.width or 0)
                            else "horizontal"),
            "resolution": f"{t.width}x{t.height}",
        }

    if meta is None:
        return None

    meta = dict(meta)
    meta["_produced_by"] = MARKER
    if render is not None:
        meta["attempt"] = render.attempt
        meta["attempts"] = attempts
        meta["attempt_file"] = Path(render.video_path).name
    return meta


def _project_status(graph: dict, critic: CriticResult | None,
                    has_output: bool) -> tuple[str, str | None]:
    """Graph outcome -> the legacy status vocabulary.

    An escalated run produced a playable file its own critic refused. The
    legacy vocabulary has no word for that, so it is reported as an error with
    the reason rather than as a clean success.
    """
    status = graph.get("status")
    verdict = graph.get("verdict") or (critic.verdict if critic else None)

    if not has_output:
        return "error", (graph.get("error")
                         or "the agent build produced no output file")
    if status == "done" or verdict == "PASS":
        return "done", None
    if status == "escalated" or verdict == "ESCALATE":
        reason = graph.get("error") or (critic.summary if critic else "")
        return "error", ("the agent build produced a video but its critic did "
                         "not pass it: " + (reason or "escalated"))
    if status == "running":
        # a build that was interrupted and never resumed
        return "error", "the agent build was interrupted and has not resumed"
    return "error", (graph.get("error") or "the agent build did not complete")


# ------------------------------------------------------------------- api

def build_job_json(job_id: str, root: Path | None = None) -> dict:
    """Assemble the legacy job.json for an agent-built job. Writes nothing."""
    _, build = job_paths(job_id, root)
    # the shared store, or this job's own database when it predates the move
    db, db_source = resolve_job_db(job_id, root)
    if not Path(db).exists():
        raise NotAnAgentJob(f"{build} holds no agent build (no artifact store)")

    store = ArtifactStore(db)
    if not store.has_job(job_id):
        raise NotAnAgentJob(
            f"no agent artifacts for {job_id} in {db_source} store {db}")

    render = store.get(job_id, "RenderResult", RenderResult)
    critic = store.get(job_id, "CriticResult", CriticResult)
    if render is None and not (build / "output.json").exists():
        raise NotAnAgentJob(f"job {job_id} has no finished render to publish")

    trace_rows = TraceRecorder(db).for_job(job_id)
    graph = _graph_status(job_id, build)
    settings = _read_settings(build)

    attempts = len(store.all_revisions(job_id, "RenderResult", RenderResult))
    output = _project_output(build, render.payload if render else None, attempts)
    status, error = _project_status(graph, critic.payload if critic else None,
                                    output is not None)

    started = min((r["started_at"] for r in trace_rows), default=None)
    finished = max((r["ended_at"] for r in trace_rows if r["ended_at"]),
                   default=None)
    # When the JOB was created, not when analysis finished. agent_settings.json
    # is written immediately after the folder pair is made, so its mtime is
    # the closest honest stand-in; the artifact's produced_at is the end of
    # clip_analysis and would date the job ~20s after it actually started.
    created = _created_at(build, started)

    return {
        "id": job_id,
        "settings": _project_settings(settings, graph),
        "created": created,
        "status": status,
        "error": error,
        "started": started,
        "finished": finished,
        "stages": _project_stages(trace_rows),
        "output": output,
        # Phase 1 has no script, captions or voice note.
        "captions": [],
        "voice": None,
        "script": "",
    }


# ------------------------------------------------------- edit plan (A)

# The legacy UI groups decisions by the five agent ids /api/health advertises
# (vision, editor, animation, merge, music) and renders `names[d.agent] || ''`.
# Phase 1's four agents map onto the two of those that describe the work they
# actually do; nothing is invented for agents that do not exist yet.
LEGACY_AGENT = {
    "clip_analysis": "editor",        # legacy editor owns probe + measure
    "editing_director": "editor",     # legacy editor owns plan + direct
    "rendering": "merge",             # legacy merge owns cut + master
    "critic": "merge",                # legacy merge owns "assemble and verify"
}

# The legacy renderer understands exactly three buckets: "vision" -> "saw it",
# "measurement" -> "decided", anything else -> "yours". Mapping is done only
# where it is TRUE:
#   policy / fallback are rules applied to measured values, so "decided" is
#   honest; `llm` has no legacy equivalent at all and would be badged "yours",
#   which would credit the operator for a model's choice - so its raw value is
#   kept and the true source is written into `why` where the UI always shows it.
_DECIDED_BY = {
    "measurement": "measurement",
    "policy": "measurement",
    "fallback": "measurement",
    "human": "you",
    "llm": "llm",
}
_NEEDS_SOURCE_IN_TEXT = {"llm", "fallback"}


def _project_decision(d: dict, produced_by: str) -> dict:
    raw = d.get("decided_by", "measurement")
    why = d.get("why", "")
    if raw in _NEEDS_SOURCE_IN_TEXT:
        why = f"[decided by the {raw}] {why}"
    return {
        "agent": LEGACY_AGENT.get(produced_by, "editor"),
        "what": d.get("what", ""),
        "chose": d.get("chose", ""),
        "why": why,
        "decided_by": _DECIDED_BY.get(raw, raw),
        # extra keys the legacy UI ignores but /plan returns, so the real
        # provenance is never lost in the projection
        "decided_by_detail": raw,
        "confidence": d.get("confidence"),
        "produced_by": produced_by,
    }


def build_edit_plan(job_id: str, root: Path | None = None) -> dict | None:
    """Project the typed artifacts into the legacy edit_plan.json shape.

    The data already exists - every agent artifact carries a `decisions` list
    with exactly the what/chose/why/decided_by fields the UI renders. This is
    a re-shaping, not a re-derivation, and it invents nothing.

    Decisions are emitted in production order (analysis, then the edit, then
    the render, then the critique) so the panel reads as the build happened.
    """
    db, _ = resolve_job_db(job_id, root)
    if not Path(db).exists():
        return None
    store = ArtifactStore(db)
    if not store.has_job(job_id):
        return None

    plan_art = store.get(job_id, "EditPlan", EditPlan)
    if plan_art is None:
        return None

    decisions: list[dict] = []
    for kind, model in (("ClipAnalysis", ClipAnalysisSet),
                        ("EditPlan", EditPlan),
                        ("RenderResult", RenderResult),
                        ("CriticResult", CriticResult)):
        art = store.get(job_id, kind, model)
        if art is None:
            continue
        for d in art.decisions:
            decisions.append(_project_decision(d.model_dump(mode="json"),
                                               art.produced_by))

    plan = plan_art.payload
    clips = store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    motions = sorted(c.motion.mean_ydif
                     for c in (clips.payload.usable_clips if clips else []))
    median = motions[len(motions) // 2] if motions else 0.0

    return {
        # Phase 1 has no Vision Agent; say so rather than leaving it ambiguous
        "vision": {"used": False, "model": None, "provider": None,
                   "clips_seen": 0, "duplicates": [], "hook": None,
                   "story": ""},
        "median_motion": round(median, 3),
        "busy_threshold": round(max(median, 0.05), 3),
        "segments": len(plan.segments),
        "transitions": ["fade"] * max(len(plan.segments) - 1, 0),
        "decisions": decisions,
        "_produced_by": MARKER,
        "_edit_plan_revision": plan_art.revision,
        "_degraded": plan_art.degraded,
        "_fallback_used": plan_art.fallback_used,
    }


def project_edit_plan(job_id: str, root: Path | None = None,
                      dry_run: bool = False) -> dict | None:
    """Write outputs/<job_id>/edit_plan.json, which /api/jobs/{id}/plan reads.

    Refuses to overwrite a plan the legacy director wrote, for the same reason
    `project` refuses to overwrite a legacy job.json.
    """
    _, build = job_paths(job_id, root)
    target = build / EDIT_PLAN_FILE
    if target.exists():
        try:
            existing = json.loads(target.read_text())
            if existing.get("_produced_by") != MARKER:
                raise FileExistsError(
                    f"{target} was written by the legacy director; refusing "
                    f"to overwrite it")
        except json.JSONDecodeError:
            pass

    payload = build_edit_plan(job_id, root)
    if payload is None:
        return None
    if not dry_run:
        target.write_text(json.dumps(payload, indent=2))
    return payload


# ------------------------------------------------ live heartbeat (C)

def project_running(job_id: str, root: Path | None = None,
                    node: str | None = None) -> dict | None:
    """Project a RUNNING agent job so the existing UI can follow it.

    Called at node boundaries, not per line: the goal is that /api/jobs/{id}
    reports meaningful stages and /events emits real transitions, NOT that
    ffmpeg's stdout is streamed. That needs an event bus and is deliberately
    out of scope.

    Returns None - writing nothing - when the job already has a TERMINAL
    projection. A finished build must never be dragged back to "running" by a
    late heartbeat.
    """
    _, build = job_paths(job_id, root)
    target = build / JOB_FILE
    if target.exists():
        if not _is_our_projection(target):
            return None                    # never touch a legacy job.json
        try:
            existing = json.loads(target.read_text())
            if existing.get("status") in ("done", "error", "cancelled"):
                return None                # terminal always wins
        except (OSError, json.JSONDecodeError):
            pass

    db, _src = resolve_job_db(job_id, root)
    if not Path(db).exists():
        return None
    trace_rows = TraceRecorder(db).for_job(job_id)
    if not trace_rows:
        return None

    settings = _read_settings(build)
    started = min((r["started_at"] for r in trace_rows), default=None)

    payload = {
        "id": job_id,
        "settings": _project_settings(settings, {}),
        "created": _created_at(build, started),
        "status": "running",
        "error": None,
        "started": started,
        "finished": None,
        "stages": _project_stages(trace_rows, live=True),
        "output": None,
        "captions": [],
        "voice": None,
        "script": "",
    }
    if node:
        payload["settings"]["_agent_node"] = node
    try:
        target.write_text(json.dumps(payload, indent=2))
    except OSError:
        return None
    return payload


# --------------------------------------------- cancellation marker (D)

def finalize(job_id: str, root: Path | None = None, *,
             status: str = "error", error: str | None = None) -> dict | None:
    """Force a live projection to a terminal one.

    Needed for a job that STOPS WITHOUT PRODUCING A RENDER - cancelled during
    analysis, or crashed before the first attempt. `project()` refuses those
    (there is no finished output to publish), so without this the last
    heartbeat's "running" would stand forever and the existing UI would show a
    build that never ends.

    Only ever touches this facade's own projection, and only when it is
    currently non-terminal.
    """
    _, build = job_paths(job_id, root)
    target = build / JOB_FILE
    if not target.exists() or not _is_our_projection(target):
        return None
    try:
        payload = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") in ("done", "error", "cancelled"):
        return payload                      # already terminal; leave it alone

    payload["status"] = status
    payload["error"] = error or "the agent build stopped before producing a video"
    payload["finished"] = time.time()
    for st in payload.get("stages", {}).values():
        if st.get("status") == "running":
            st["status"] = "error"
    try:
        target.write_text(json.dumps(payload, indent=2))
    except OSError:
        return None
    return payload


def cancel_marker(job_id: str, root: Path | None = None) -> Path:
    _, build = job_paths(job_id, root)
    return build / CANCEL_FILE


def request_cancel(job_id: str, root: Path | None = None) -> bool:
    """Ask a job to stop, from ANY process.

    The scheduler can only tree-kill processes its own runtime owns, and a job
    is usually started by the CLI rather than by the web server. A marker file
    is the smallest thing both processes can see; the running job polls it and
    performs the real kill through its own scheduler.
    """
    p = cancel_marker(job_id, root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(time.time()))
        return True
    except OSError:
        return False


def cancel_requested(job_id: str, root: Path | None = None) -> bool:
    return cancel_marker(job_id, root).exists()


def clear_cancel_request(job_id: str, root: Path | None = None) -> None:
    try:
        cancel_marker(job_id, root).unlink(missing_ok=True)
    except OSError:
        pass


# --------------------------------------------------------------- write

def project(job_id: str, root: Path | None = None, *,
            overwrite: bool = True, dry_run: bool = False) -> dict:
    """Write outputs/<job_id>/job.json so the existing app can see the build.

    Refuses to clobber a legacy job.json unless asked: a folder built by
    server/jobs.py owns its own state file, and overwriting it would corrupt
    a real build's record.

    Also projects edit_plan.json, so the UI's decisions panel works for agent
    jobs the same way it does for legacy ones.
    """
    _, build = job_paths(job_id, root)
    target = build / JOB_FILE

    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists")
    if target.exists() and not _is_our_projection(target):
        # FileExistsError subclasses OSError, so this deliberately sits
        # outside the try that reads the file - raising it in there would
        # have it swallowed by the same handler that tolerates an unreadable
        # file, and a real build's state would be silently overwritten.
        raise FileExistsError(
            f"{target} was written by the legacy pipeline, not by the agent "
            f"path; refusing to overwrite it")

    payload = build_job_json(job_id, root)
    if not dry_run:
        target.write_text(json.dumps(payload, indent=2))
        try:
            project_edit_plan(job_id, root)
        except (FileExistsError, OSError):
            pass                  # a legacy plan wins; job.json still lands
    return payload


def project_all(root: Path | None = None, dry_run: bool = False) -> dict[str, str]:
    """Publish every agent job found. Returns {job_id: outcome}."""
    out: dict[str, str] = {}
    for jid in agent_jobs(root):
        try:
            p = project(jid, root, dry_run=dry_run)
            out[jid] = f"ok ({p['status']})"
        except (NotAnAgentJob, FileExistsError) as exc:
            out[jid] = f"skipped: {exc}"
        except Exception as exc:                        # noqa: BLE001
            out[jid] = f"error: {type(exc).__name__}: {exc}"
    return out


# ------------------------------------------------------------------- cli

def _cli() -> int:
    ap = argparse.ArgumentParser(
        prog="agents.compat",
        description="Publish an agent-built job into the existing app's "
                    "job listing and download endpoint.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-id")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--root", help="override the app root for uploads/outputs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written, write nothing")
    args = ap.parse_args()
    root = Path(args.root).resolve() if args.root else None

    if args.all:
        for jid, outcome in project_all(root, args.dry_run).items():
            print(f"{jid:16} {outcome}")
        return 0

    try:
        payload = project(args.job_id, root, dry_run=args.dry_run)
    except (NotAnAgentJob, FileExistsError) as exc:
        print(f"cannot publish {args.job_id}: {exc}", file=sys.stderr)
        return 1

    _, build = job_paths(args.job_id, root)
    print(f"{'would write' if args.dry_run else 'wrote'} {build / JOB_FILE}")
    print(f"  status   {payload['status']}")
    if payload["error"]:
        print(f"  error    {payload['error']}")
    o = payload["output"] or {}
    print(f"  title    {payload['settings']['title']}")
    print(f"  output   {o.get('file')}  {o.get('duration')}s  "
          f"{o.get('resolution')}  frames_ok={o.get('frames_ok')}")
    print(f"  download {o.get('path')}")
    done = sum(1 for s in payload["stages"].values() if s["status"] == "done")
    skip = sum(1 for s in payload["stages"].values() if s["status"] == "skipped")
    print(f"  stages   {done} done, {skip} skipped (not run in Phase 1)")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(_cli())
