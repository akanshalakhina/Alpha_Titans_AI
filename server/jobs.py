"""
Job store and runner.

One job = one build folder = one run of the existing pipeline. Stage order is
exactly the pipeline's own order, with the caption write slotted in where the
pipeline expects it (after `plan`, before `cut`) and the two finishing stages
(`music`, `master`) last.

Everything the UI displays comes from either the stage exit codes, the stage
stdout, or manifest.json. Nothing is simulated.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path

import adapter
import director
import extras
import vision
from adapter import STAGES, STAGE_IDS, STAGE_WEIGHT, TOTAL_WEIGHT

_LOCK = threading.RLock()
_JOBS: dict[str, "Job"] = {}


class Job:
    def __init__(self, job_id: str, settings: dict):
        self.id = job_id
        self.settings = settings
        self.created = time.time()
        self.status = "draft"          # draft|running|done|error|cancelled
        self.error: str | None = None
        self.started: float | None = None
        self.finished: float | None = None
        self.current: str | None = None
        self.stages = {
            sid: {"id": sid, "label": lbl, "detail": det, "weight": w,
                  "status": "pending", "started": None, "finished": None,
                  "lines": 0}
            for sid, lbl, det, w in STAGES
        }
        self.log: list[dict] = []
        self.output: dict | None = None
        self.plan: dict | None = None
        self.vision: dict | None = None
        self.captions: list[str] = []
        self.voice: dict | None = None
        self.script: str = ""
        self._seq = 0
        self._cancel = threading.Event()

    # ------------------------------------------------------------ logging

    def emit(self, kind: str, **payload):
        with _LOCK:
            self._seq += 1
            evt = {"seq": self._seq, "t": round(time.time() - (self.started or self.created), 2),
                   "kind": kind, **payload}
            self.log.append(evt)
            if len(self.log) > 4000:
                del self.log[:1000]
            return evt

    def line(self, stage: str, text: str):
        st = self.stages.get(stage)
        if st:
            st["lines"] += 1
        self.emit("line", stage=stage, text=text[:500])

    # ----------------------------------------------------------- progress

    def progress(self) -> float:
        done = 0.0
        for sid in STAGE_IDS:
            s = self.stages[sid]
            if s["status"] in ("done", "skipped"):
                done += s["weight"]
            elif s["status"] == "running":
                done += s["weight"] * 0.45
        return round(100.0 * done / TOTAL_WEIGHT, 1)

    def snapshot(self) -> dict:
        m = adapter.read_manifest(self.id) or {}
        clips = m.get("clips") or []
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "current": self.current,
            "progress": self.progress(),
            "settings": self.settings,
            "stages": [self.stages[s] for s in STAGE_IDS],
            "script": self.script,
            "captions": self.captions,
            "voice": self.voice,
            "output": self.output,
            "edit_plan": self.plan or director.read_plan(self.id),
            "vision": self.vision or vision.read(self.id),
            "analysis": {
                "orientation": m.get("orientation"),
                "canvas": m.get("canvas"),
                "target": m.get("target"),
                "planned_total": m.get("planned_total"),
                "total_duration": m.get("total_duration"),
                "transitions": m.get("transitions"),
                "dropped": m.get("dropped"),
                "rejected": m.get("rejected"),
                "clips": [
                    {k: c.get(k) for k in
                     ("name", "duration", "display", "fps", "codec",
                      "orientation", "mean_ydif", "max_ydif", "freeze_share",
                      "slowmo")}
                    for c in clips
                ],
                "segments": m.get("segments"),
            },
        }


# ----------------------------------------------------------------- store

def create(settings: dict) -> Job:
    job_id = uuid.uuid4().hex[:12]
    adapter.src_dir(job_id).mkdir(parents=True, exist_ok=True)
    adapter.build_dir(job_id).mkdir(parents=True, exist_ok=True)
    adapter.voice_dir(job_id).mkdir(parents=True, exist_ok=True)
    job = Job(job_id, settings)
    with _LOCK:
        _JOBS[job_id] = job
    _persist(job)
    return job


def get(job_id: str) -> Job | None:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job:
        return job
    return _restore(job_id)


def listing(limit: int = 30) -> list[dict]:
    out = []
    with _LOCK:
        jobs = list(_JOBS.values())
    seen = {j.id for j in jobs}
    # pick up jobs from previous server runs
    if adapter.OUTPUTS.exists():
        for d in adapter.OUTPUTS.iterdir():
            if d.is_dir() and d.name not in seen:
                j = _restore(d.name)
                if j:
                    jobs.append(j)
    for j in sorted(jobs, key=lambda x: x.created, reverse=True)[:limit]:
        m = adapter.read_manifest(j.id) or {}
        # A draft with nothing in it is an artefact of opening the page, not a
        # build anyone made. Listing it would only be noise.
        if j.status == "draft" and not _has_clips(j.id):
            continue
        out.append({
            "id": j.id, "status": j.status, "created": j.created,
            "title": j.settings.get("title") or "Untitled build",
            "progress": j.progress(),
            "clips": len(m.get("clips") or []),
            "duration": (j.output or {}).get("duration"),
            "orientation": m.get("orientation"),
            "thumbnail": (j.output or {}).get("thumbnail"),
        })
    return out


def _has_clips(job_id: str) -> bool:
    d = adapter.src_dir(job_id)
    return d.exists() and any(p.suffix.lower() in adapter.VIDEO_EXT
                              for p in d.iterdir())


def _state_path(job_id: str) -> Path:
    return adapter.job_dir(job_id) / "job.json"


def _persist(job: Job) -> None:
    try:
        with open(_state_path(job.id), "w") as f:
            json.dump({
                "id": job.id, "settings": job.settings, "created": job.created,
                "status": job.status, "error": job.error,
                "started": job.started, "finished": job.finished,
                "stages": job.stages, "output": job.output,
                "captions": job.captions, "voice": job.voice,
                "script": job.script,
            }, f, indent=2)
    except OSError:
        pass


def _restore(job_id: str) -> Job | None:
    p = _state_path(job_id)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    job = Job(job_id, d.get("settings") or {})
    job.created = d.get("created") or job.created
    job.status = d.get("status") or "draft"
    job.error = d.get("error")
    job.started = d.get("started")
    job.finished = d.get("finished")
    job.output = d.get("output")
    job.captions = d.get("captions") or []
    job.voice = d.get("voice")
    job.script = d.get("script") or ""
    for sid, st in (d.get("stages") or {}).items():
        if sid in job.stages:
            job.stages[sid].update(st)
    # a job that was mid-run when the server stopped is not running now
    if job.status == "running":
        job.status = "error"
        job.error = job.error or "server restarted while this build was running"
        for st in job.stages.values():
            if st["status"] == "running":
                st["status"] = "error"
    with _LOCK:
        _JOBS[job_id] = job
    return job


# ---------------------------------------------------------------- running

def start(job: Job) -> None:
    if job.status == "running":
        return
    # a cancel leaves the stop flag raised; clearing it here is what lets an
    # operator cancel a build and then start another one from the same job
    job._cancel.clear()
    clips = [p for p in adapter.src_dir(job.id).iterdir()
             if p.suffix.lower() in adapter.VIDEO_EXT] \
        if adapter.src_dir(job.id).exists() else []
    if not clips:
        raise ValueError("Add at least one clip before starting a build.")
    job.status = "running"
    job.error = None
    job.started = time.time()
    for st in job.stages.values():
        st.update(status="pending", started=None, finished=None, lines=0)
    job.emit("status", status="running")
    t = threading.Thread(target=_run, args=(job,), daemon=True)
    t.start()


def cancel(job: Job) -> None:
    job._cancel.set()
    job.emit("line", stage=job.current or "probe",
             text="cancellation requested; stopping after the current stage")


def _run(job: Job) -> None:
    try:
        for sid, label, _detail, _w in STAGES:
            if job._cancel.is_set():
                job.status = "cancelled"
                job.emit("status", status="cancelled")
                _persist(job)
                return

            st = job.stages[sid]
            st.update(status="running", started=time.time())
            job.current = sid
            job.emit("stage", stage=sid, status="running", label=label)

            if sid == "vision":
                # real frames -> a real vision model. Returns None and logs
                # why when no key is configured; the build continues either way.
                job.vision = vision.analyse(job.id, job.settings,
                                            lambda ln: job.line(sid, ln))
                _persist(job)
            elif sid == "direct":
                # decides what the planner does not, and writes edit_plan.json;
                # its choices go into job.settings so later stages consume them
                job.plan = director.direct(job.id, job.settings,
                                           lambda ln: job.line(sid, ln))
                job.captions = extras.write_captions(
                    job.id, job.script, job.settings.get("vision_captions"))
                filled = len([c for c in job.captions if c])
                job.line(sid, "captions written for %d segments (%d non-empty)"
                              % (len(job.captions), filled))
                _persist(job)
            elif sid == "music":
                # the score is written against the real transition marks the
                # join stage recorded, rescaled by the speed setting
                extras.music_stage(job.id, job.settings,
                                   lambda ln: job.line(sid, ln))
            elif sid == "master":
                job.output = extras.master_stage(
                    job.id, job.settings, lambda ln: job.line(sid, ln))
            else:
                rc = adapter.run_stage(job.id, sid, job.settings,
                                       lambda ln: job.line(sid, ln))
                if rc != 0:
                    raise RuntimeError(
                        f"stage '{sid}' exited {rc} - see the log for the "
                        f"pipeline's own error")
                # captions are written after `direct`, not after `plan`:
                # the director is what decides whether the vision pass may
                # fill any blanks, and cut needs the file before it runs.

            # the vision stage completes whether or not a model was reachable;
            # calling that "done" would put a green tick on a pass that never
            # looked at anything, so a pass that produced nothing is "skipped"
            final = "skipped" if (sid == "vision" and not job.vision) else "done"
            st.update(status=final, finished=time.time())
            job.emit("stage", stage=sid, status=final, label=label,
                     progress=job.progress())
            _persist(job)

        job.status = "done"
        job.finished = time.time()
        job.current = None
        job.emit("status", status="done", output=job.output)
    except Exception as exc:                      # noqa: BLE001 - reported to UI
        job.status = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.finished = time.time()
        if job.current and job.current in job.stages:
            job.stages[job.current].update(status="error", finished=time.time())
        job.emit("status", status="error", error=job.error)
        job.emit("line", stage=job.current or "probe",
                 text=traceback.format_exc()[-1200:])
    finally:
        _persist(job)
