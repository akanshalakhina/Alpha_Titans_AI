"""
Alpha Titans - production API.

A thin HTTP surface over the existing pipeline. Every endpoint either moves a
file into place, reads manifest.json, or runs a pipeline stage. There is no
mock data anywhere in this server: if the pipeline cannot do something, the
endpoint says so rather than inventing a result.

Run (from the application folder):
    ./run.sh                     # or: python3 -m uvicorn main:app --app-dir server
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

import adapter
import config
import director
import extras
import jobs
import vision

WEB = config.FRONTEND_DIR
MAX_CLIP_BYTES = config.MAX_CLIP_BYTES

config.ensure_dirs()

app = FastAPI(title="Alpha Titans Content Engine", version="1.0")

if config.CORS_ORIGINS:
    # Only needed when the frontend is served from somewhere other than this
    # server. Same-origin (the default) needs none of this.
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_credentials=True, allow_methods=["*"],
                       allow_headers=["*"])


# ------------------------------------------------------------------ helpers

def safe_name(name: str) -> str:
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
    return name or "file"


def need(job_id: str) -> jobs.Job:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"No job {job_id}")
    return job


def resolve_music(settings: dict) -> None:
    """Turn a library track name into the absolute path the music stage expects.

    Both /start and /regenerate need this. It lived only in /start at first,
    so a regenerate with an uploaded track silently produced a silent video -
    the verification caught it as -70 LUFS.
    """
    if settings.get("music_source") != "upload":
        settings.pop("music_file", None)
        return
    track = settings.get("music_track")
    if not track:
        return
    path = adapter.MUSIC_LIB / safe_name(track)
    if path.exists():
        settings["music_file"] = str(path)
    else:
        settings.pop("music_file", None)


def agent_owned(job_id: str) -> bool:
    """Was this job built by the agent layer rather than by this pipeline?

    COMPATIBILITY SAFETY GUARD. The agent layer publishes a job.json so its
    videos appear in the listing and can be downloaded. That also makes the
    job reachable by `/start`, which would run THIS pipeline over the agent's
    folder and overwrite its manifest, captions, segments and output - an
    agent build destroyed by a button that looks harmless.

    Deliberately a plain file read of the marker the projection writes: no
    import of the agent package, so the legacy application still starts and
    behaves identically when the agent dependencies are not installed.
    """
    p = adapter.build_dir(job_id) / "job.json"
    try:
        with open(p) as f:
            data = json.load(f)
        produced = (data.get("settings") or {}).get("_produced_by") or ""
        return produced.startswith("agents/")
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def serve_within(root: Path, name: str) -> FileResponse:
    """Serve a file from a directory, refusing anything that escapes it."""
    target = (root / name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "path outside the job folder")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, f"{name} not found")
    return FileResponse(target)


# -------------------------------------------------------------------- meta

@app.get("/api/health")
def health():
    pre = adapter.preflight()
    return {
        "ok": pre["pipeline"] and pre["ffmpeg"],
        "capabilities": pre,
        "stages": [{"id": s, "label": l, "detail": d, "weight": w}
                   for s, l, d, w in adapter.STAGES],
        "options": {
            "orientation": ["auto", "horizontal", "vertical"],
            "resolution": list(adapter.RESOLUTIONS),
            "quality": list(adapter.QUALITY),
            "speed": list(adapter.SPEEDS) + ["custom"],
            # "auto" means the director decides from the measurements
            "transition_style": ["auto"] + list(adapter.TRANSITIONS),
            "mood": ["auto"] + list(extras.MOODS),
            "music_source": ["generate", "upload", "none"],
            "keep_audio": list(adapter.KEEP_AUDIO),
            "sequence": list(adapter.SEQUENCES),
        },
        "agents": adapter.agents_manifest(),
        "stage_agent": adapter.STAGE_AGENT,
        "vision": vision.describe(),
        "keep_audio_detail": adapter.KEEP_AUDIO,
        "sequence_detail": adapter.SEQUENCES,
        # Surfaced so the UI can be honest about it rather than showing a
        # control that silently does nothing.
        "notes": {
            "transcription": (
                "Offline speech-to-text engine (faster-whisper) is ready. Voice notes will be transcribed."
                if pre.get("transcription")
                else "No offline speech-to-text engine is installed in this "
                     "environment, so voice notes are recorded, stored and played "
                     "back but not machine-transcribed."
            ),
        },
        "limits": {
            "clip_mb": config.MAX_CLIP_MB,
            "track_mb": config.MAX_TRACK_MB,
            "voice_mb": config.MAX_VOICE_MB,
        },
    }


# -------------------------------------------------------------------- jobs

@app.get("/api/jobs")
def list_jobs():
    return {"jobs": jobs.listing()}


@app.post("/api/jobs")
async def create_job(payload: dict | None = None):
    settings = dict(payload or {})
    job = jobs.create(settings)
    return {"id": job.id, "job": job.snapshot()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return need(job_id).snapshot()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    need(job_id)
    # a job now owns two folders: its uploads and its build output
    shutil.rmtree(adapter.build_dir(job_id), ignore_errors=True)
    shutil.rmtree(adapter.src_dir(job_id), ignore_errors=True)
    jobs._JOBS.pop(job_id, None)
    return {"deleted": job_id}


@app.patch("/api/jobs/{job_id}")
async def patch_job(job_id: str, payload: dict):
    job = need(job_id)
    job.settings.update(payload or {})
    jobs._persist(job)
    return job.snapshot()


# ------------------------------------------------------------------- clips

@app.get("/api/jobs/{job_id}/clips")
def get_clips(job_id: str):
    need(job_id)
    d = adapter.src_dir(job_id)
    out = []
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in adapter.VIDEO_EXT:
                out.append({"name": p.name, **adapter.media_summary(p)})
    return {"clips": out}


@app.post("/api/jobs/{job_id}/clips")
async def upload_clips(job_id: str, files: list[UploadFile] = File(...)):
    need(job_id)
    d = adapter.src_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in d.iterdir()
                      if p.suffix.lower() in adapter.VIDEO_EXT)
    idx = len(existing)
    saved, rejected = [], []
    for f in files:
        name = safe_name(f.filename or "clip.mp4")
        if not name.lower().endswith(adapter.VIDEO_EXT):
            rejected.append({"name": name, "why": "not a video container"})
            continue
        idx += 1
        # numeric prefix keeps the pipeline's sorted glob in the operator's order
        target = d / f"{idx:02d}_{name}"
        size = 0
        with open(target, "wb") as out:
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > MAX_CLIP_BYTES:
                    out.close()
                    target.unlink(missing_ok=True)
                    rejected.append({"name": name,
                                     "why": f"over {config.MAX_CLIP_MB} MB"})
                    break
                out.write(chunk)
        if not target.exists():
            idx -= 1
            continue
        info = adapter.media_summary(target)
        if not info.get("duration"):
            target.unlink(missing_ok=True)
            idx -= 1
            rejected.append({"name": name, "why": "ffprobe could not read it"})
            continue
        saved.append({"name": target.name, **info})
    return {"saved": saved, "rejected": rejected}


@app.delete("/api/jobs/{job_id}/clips/{name}")
def delete_clip(job_id: str, name: str):
    need(job_id)
    target = (adapter.src_dir(job_id) / safe_name(name)).resolve()
    try:
        target.relative_to(adapter.src_dir(job_id).resolve())
    except ValueError:
        raise HTTPException(400, "path outside the job folder")
    if not target.exists():
        raise HTTPException(404, "clip not found")
    target.unlink()
    return {"removed": target.name}


@app.post("/api/jobs/{job_id}/clips/order")
def reorder_clips(job_id: str, payload: dict):
    """Rewrite the numeric prefixes so the pipeline's sorted glob matches the
    order the operator dragged them into."""
    need(job_id)
    order = payload.get("order") or []
    d = adapter.src_dir(job_id)
    present = {p.name for p in d.iterdir() if p.suffix.lower() in adapter.VIDEO_EXT}
    if set(order) != present:
        raise HTTPException(400, "order must list exactly the current clips")
    def _rename(a, b):
        # Windows refuses to rename a file another process (a preview still
        # streaming it, a poster ffmpeg) has open. It is usually released
        # within a moment, so retry briefly before giving up.
        for attempt in range(12):
            try:
                a.rename(b)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(0.25)

    tmp, done = [], []
    try:
        for i, name in enumerate(order, 1):
            src = d / name
            stem = re.sub(r"^\d{2,3}_", "", name)
            holding = d / f"tmp_{i:03d}_{stem}"
            _rename(src, holding)
            done.append((holding, src))
            tmp.append((holding, f"{i:02d}_{stem}"))
        final = []
        for holding, newname in tmp:
            _rename(holding, d / newname)
            final.append(newname)
    except OSError as e:
        # put back whatever was already moved so the bin is never left half-renamed
        for holding, original in reversed(done):
            try:
                if holding.exists():
                    holding.rename(original)
            except OSError:
                pass
        raise HTTPException(409, "a clip is still in use (a preview may be playing) - "
                                 f"try the reorder again in a moment ({e.__class__.__name__})")
    return {"order": final}


@app.get("/api/jobs/{job_id}/poster/{name}")
def clip_poster(job_id: str, name: str):
    """A real frame from the clip, extracted once and cached beside it.

    The browser will not reliably paint a frame for a <video> that has only
    been asked for metadata, so the card looked black until this existed.
    """
    need(job_id)
    src = (adapter.src_dir(job_id) / safe_name(name)).resolve()
    try:
        src.relative_to(adapter.src_dir(job_id).resolve())
    except ValueError:
        raise HTTPException(400, "path outside the job folder")
    if not src.exists():
        raise HTTPException(404, "clip not found")
    cache = adapter.poster_dir(job_id)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / (src.stem + ".jpg")
    if not out.exists():
        info = adapter.media_summary(src)
        at = max(0.1, min((info.get("duration") or 1) * 0.25, 5.0))
        extras.run([config.FFMPEG, "-nostdin", "-v", "error", "-y", "-ss",
                    f"{at:.2f}", "-i", str(src), "-frames:v", "1",
                    "-vf", "scale=320:-2", str(out)])
    if not out.exists():
        raise HTTPException(500, "could not extract a frame")
    return FileResponse(out, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/clip/{name}")
def stream_clip(job_id: str, name: str):
    need(job_id)
    return serve_within(adapter.src_dir(job_id), safe_name(name))


# ------------------------------------------------------------ script/voice

@app.post("/api/jobs/{job_id}/script")
async def set_script(job_id: str, payload: dict):
    job = need(job_id)
    job.script = (payload.get("text") or "")[:20000]
    jobs._persist(job)
    m = adapter.read_manifest(job_id) or {}
    n = len(m.get("segments") or [])
    preview = extras.split_script(job.script, n) if n else []
    return {"script": job.script, "segments": n, "preview": preview}


@app.post("/api/jobs/{job_id}/script-file")
async def upload_script(job_id: str, file: UploadFile = File(...)):
    job = need(job_id)
    raw = await file.read()
    if len(raw) > 512 * 1024:
        raise HTTPException(413, "script file over 512 KB")
    try:
        job.script = raw.decode("utf-8", "replace")[:20000]
    except Exception:
        raise HTTPException(400, "could not read that file as text")
    jobs._persist(job)
    return {"script": job.script, "filename": safe_name(file.filename or "")}


@app.post("/api/jobs/{job_id}/voice")
async def upload_voice(job_id: str, file: UploadFile = File(...)):
    job = need(job_id)
    raw = await file.read()
    if len(raw) > config.MAX_VOICE_MB * 1024 * 1024:
        raise HTTPException(413, f"voice note over {config.MAX_VOICE_MB} MB")
    job.voice = extras.store_voice_note(job_id, file.filename or "voice.webm", raw)
    jobs._persist(job)
    return job.voice


@app.get("/api/jobs/{job_id}/voice-file")
def voice_file(job_id: str):
    job = need(job_id)
    if not job.voice:
        raise HTTPException(404, "no voice note on this job")
    return serve_within(adapter.voice_dir(job_id), job.voice["filename"])


# ------------------------------------------------------------ music library

@app.get("/api/music")
def music_library():
    adapter.MUSIC_LIB.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(adapter.MUSIC_LIB.iterdir()):
        if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"):
            info = adapter.ffprobe_json(p).get("format", {})
            try:
                dur = round(float(info.get("duration") or 0), 2)
            except (TypeError, ValueError):
                dur = 0
            meta_path = p.with_suffix(p.suffix + ".json")
            tags = {}
            if meta_path.exists():
                try:
                    tags = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    tags = {}
            out.append({"name": p.name, "duration": dur,
                        "size": p.stat().st_size, "mood": tags.get("mood")})
    return {"tracks": out}


@app.post("/api/music")
async def add_music(file: UploadFile = File(...), mood: str = Form("")):
    adapter.MUSIC_LIB.mkdir(parents=True, exist_ok=True)
    name = safe_name(file.filename or "track.wav")
    if not name.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")):
        raise HTTPException(400, "not an audio file")
    target = adapter.MUSIC_LIB / name
    raw = await file.read()
    if len(raw) > config.MAX_TRACK_MB * 1024 * 1024:
        raise HTTPException(413, f"track over {config.MAX_TRACK_MB} MB")
    target.write_bytes(raw)
    info = adapter.ffprobe_json(target)
    if not info.get("streams"):
        target.unlink(missing_ok=True)
        raise HTTPException(400, "ffprobe could not read that audio")
    if mood:
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps({"mood": mood}))
    dur = round(float(info.get("format", {}).get("duration") or 0), 2)
    return {"name": name, "duration": dur, "size": len(raw), "mood": mood or None}


@app.delete("/api/music/{name}")
def del_music(name: str):
    p = (adapter.MUSIC_LIB / safe_name(name)).resolve()
    try:
        p.relative_to(adapter.MUSIC_LIB.resolve())
    except ValueError:
        raise HTTPException(400, "bad path")
    if not p.exists():
        raise HTTPException(404, "not found")
    p.unlink()
    p.with_suffix(p.suffix + ".json").unlink(missing_ok=True)
    return {"removed": name}


@app.get("/api/music/{name}")
def get_music(name: str):
    return serve_within(adapter.MUSIC_LIB, safe_name(name))


# ------------------------------------------------------------------- build

@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, payload: dict | None = None):
    job = need(job_id)
    if agent_owned(job_id):
        # see agent_owned(): running this pipeline over an agent build's
        # folder would overwrite its manifest, segments and output
        raise HTTPException(
            409, "This video was built by the agent pipeline. Starting it "
                 "here would overwrite that build. Use the agent runner, or "
                 "create a new video from these clips.")
    if payload:
        job.settings.update(payload)
    resolve_music(job.settings)
    # a fresh run must re-decide rather than inherit the last run's choices
    job.settings.pop("transition_list", None)
    job.settings["_has_script"] = bool((job.script or "").strip())
    try:
        jobs.start(job)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = need(job_id)
    if agent_owned(job_id):
        # jobs.cancel() sets a flag on an in-process thread. An agent job has
        # no such thread here, so this used to return 200 having done nothing.
        # Say so, and point at the endpoint that can actually kill it.
        raise HTTPException(
            409, "This video is being built by the agent pipeline; cancel it "
                 "with POST /api/v2/jobs/%s/cancel, which stops the real "
                 "encoder process." % job_id)
    jobs.cancel(job)
    return {"status": job.status}


@app.post("/api/jobs/{job_id}/regenerate")
async def regenerate(job_id: str, payload: dict | None = None):
    """New build from the same clips and script with changed settings.

    The source clips are hard-linked where possible so a regenerate costs no
    extra disk, and the original build stays on disk for comparison.
    """
    old = need(job_id)
    settings = dict(old.settings)
    settings.update(payload or {})
    resolve_music(settings)
    new = jobs.create(settings)
    new.script = old.script
    new.voice = old.voice
    src_old, src_new = adapter.src_dir(job_id), adapter.src_dir(new.id)
    for p in sorted(src_old.iterdir()):
        if p.suffix.lower() in adapter.VIDEO_EXT:
            try:
                os.link(p, src_new / p.name)
            except OSError:
                shutil.copy2(p, src_new / p.name)
    if old.voice:
        try:
            adapter.voice_dir(new.id).mkdir(parents=True, exist_ok=True)
            shutil.copy2(adapter.voice_dir(job_id) / old.voice["filename"],
                         adapter.voice_dir(new.id) / old.voice["filename"])
        except OSError:
            new.voice = None
    jobs._persist(new)
    jobs.start(new)
    return {"id": new.id, "job": new.snapshot()}


# ----------------------------------------------------------------- outputs

@app.get("/api/jobs/{job_id}/file/{name}")
def build_file(job_id: str, name: str):
    need(job_id)
    return serve_within(adapter.build_dir(job_id), safe_name(name))


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = need(job_id)
    if not job.output:
        raise HTTPException(409, "this build has no finished output yet")
    path = Path(job.output["path"])
    if not path.exists():
        raise HTTPException(404, "output file is missing from disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/jobs/{job_id}/vision")
def vision_analysis(job_id: str):
    """What the vision model saw, per clip. 404 when it did not run."""
    need(job_id)
    v = vision.read(job_id)
    if not v:
        raise HTTPException(404, "no vision analysis for this build")
    return JSONResponse(v)


@app.get("/api/jobs/{job_id}/plan")
def edit_plan(job_id: str):
    """The edit plan: every decision and the reason behind it."""
    need(job_id)
    plan = director.read_plan(job_id)
    if not plan:
        raise HTTPException(404, "no edit plan yet - run the build")
    return JSONResponse(plan)


@app.get("/api/jobs/{job_id}/notes")
def notes(job_id: str):
    need(job_id)
    p = adapter.build_dir(job_id) / "output.json"
    if not p.exists():
        raise HTTPException(404, "no output metadata yet")
    return JSONResponse(json.loads(p.read_text()))


# ------------------------------------------------------------------ events

@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str, after: int = 0):
    """Server-sent events: real stage transitions and real pipeline stdout."""
    job = need(job_id)

    async def stream():
        cursor = after
        idle = 0
        yield f"data: {json.dumps({'kind': 'snapshot', 'job': job.snapshot()})}\n\n"
        while True:
            fresh = [e for e in job.log if e["seq"] > cursor]
            if fresh:
                cursor = fresh[-1]["seq"]
                for e in fresh:
                    yield f"data: {json.dumps(e)}\n\n"
                idle = 0
            else:
                idle += 1
            if job.status in ("done", "error", "cancelled") and not fresh:
                yield ("data: " + json.dumps(
                    {"kind": "final", "job": job.snapshot()}) + "\n\n")
                return
            if idle > 900:                      # ~7.5 min with no activity
                yield "data: {\"kind\": \"idle\"}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------- web

@app.get("/")
def index():
    f = WEB / "index.html"
    if not f.exists():
        return HTMLResponse(
            "<h1>Alpha Titans</h1><p>frontend/index.html missing at "
            f"{WEB}</p>", status_code=500)
    # always revalidate, so a new frontend is never mixed with a cached old page
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


# The agent layer is optional: if its dependencies are not installed the
# existing application must still start and behave exactly as before.
try:
    from agents.api import router as _agent_router
    app.include_router(_agent_router)
except Exception:                       # noqa: BLE001 - optional component
    pass


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    # `python3 server/main.py` works as well as the uvicorn CLI
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
