"""HTTP routes for the cloud path, under /api/cloud.

Two audiences, two kinds of credential:

  BROWSER  `Authorization: Bearer <AT_CLOUD_API_TOKEN>`. Uploads, status,
           cancel. The token is held in the tab's sessionStorage and sent in
           a header - never in a URL, where it would land in logs.
  WORKER   `Authorization: Bearer <Google ID token>` naming the worker VM's
           service account, verified per request. Claim, heartbeat,
           complete, fail, idle. Nothing here is reachable without it.

Every error reaches a browser as a CATEGORY and a fixed sentence written for
a person. The cloud provider's own message, which can name buckets, folders,
accounts and request IDs, stays in the server log, redacted.
"""

from __future__ import annotations

import hmac
import logging
import re
import secrets
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from pydantic import BaseModel, Field

from . import estimate as ESTIMATE
from . import queue as Q
from .gcp import CHUNK, CHUNK_ALIGN, drive_link
from .paths import check_job_id, safe_name, video_ext
from .sanitize import CloudError, public_message, redact

log = logging.getLogger("alpha.cloud.api")

MAX_CHUNK = 2 * CHUNK                                  # 16 MiB
TICKET_S = 60                    # a download link lives this long
VIEW_S = 30 * 60                 # a player link, reusable, this long
_RANGE_REQ = re.compile(r"^bytes=\d*-\d*$")
_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_WORKER_ID = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")
_STATUS = {"invalid": 400, "auth": 401, "permission": 403, "not_found": 404,
           "lease_expired": 409, "verification": 409, "quota": 507,
           "rate_limited": 429, "daily_cap": 429, "transient": 503,
           "timeout": 504,
           "worker_unavailable": 503, "internal": 500}


# ------------------------------------------------------------------ models

class FileSpec(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    mime: str = Field(min_length=1, max_length=100)


class CreateUpload(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=200,
                                 pattern=r"^[A-Za-z0-9._:-]+$")
    files: list[FileSpec] = Field(min_length=1, max_length=64)
    settings: dict = Field(default_factory=dict)


class DirectFile(BaseModel):
    idx: int = Field(ge=0)
    drive_file_id: str = Field(min_length=10, max_length=200,
                               pattern=r"^[A-Za-z0-9_-]+$")


class CompleteUpload(BaseModel):
    files: list[DirectFile] = Field(default_factory=list)
    # the person's final choices, sent with "Create my video": the clips
    # start uploading as soon as they are picked, before the options are
    settings: dict | None = None


class WorkerHello(BaseModel):
    worker_id: str
    ready: bool
    problems: list[str] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)


class WorkerOnly(BaseModel):
    worker_id: str


class WorkerBeat(BaseModel):
    worker_id: str
    job_id: str
    claim_token: int
    state: str | None = None
    stage: str | None = Field(default=None, max_length=80)
    progress: dict | None = None


class WorkerComplete(BaseModel):
    worker_id: str
    job_id: str
    claim_token: int
    outputs: list[dict]
    trace: dict | None = None


class WorkerFail(BaseModel):
    worker_id: str
    job_id: str
    claim_token: int
    category: str = Field(max_length=40)
    message: str = Field(default="", max_length=4000)
    retryable: bool | None = None


# ------------------------------------------------------------------- views

def job_view(job: dict, vm: dict | None = None,
             events: list | None = None, link=drive_link) -> dict:
    """What a browser may see about a job. No session URIs, no raw errors,
    no worker internals beyond a state word."""
    cat = job.get("error_category")
    out = {
        "job_id": job["job_id"],
        "title": (job.get("settings") or {}).get("title"),
        "state": job["state"],
        "stage": job.get("stage"),
        "progress": job.get("progress") or {},
        "attempt": job["attempt"],
        "max_attempts": job["max_attempts"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "retry_at": job["not_before"] if job["state"] == Q.RETRYING
        else None,
        "cancel_requested": bool(job.get("cancel_requested")),
        "error": ({"category": cat, "message": public_message(cat)}
                  if cat else None),
        "sources": [{"idx": s["idx"], "name": s["name"], "size": s["size"],
                     "received": s["received"],
                     "verified": bool(s["verified"])}
                    for s in job.get("sources", [])],
        # a store with no public link (a private bucket) is downloaded
        # through the control plane, with the same token as everything else
        "outputs": [{"kind": o.get("kind"), "name": o.get("name"),
                     "size": o.get("size"),
                     "link": (link(o["drive_file_id"])
                              or f"/api/cloud/jobs/{job['job_id']}"
                                 f"/outputs/{i}")}
                    for i, o in enumerate(job.get("outputs", []))
                    if o.get("drive_file_id")],
        "has_trace": bool(job.get("has_trace")),
    }
    if vm is not None:
        out["worker"] = {"state": vm["state"]}
    if events is not None:
        out["events"] = [{"at": e["at"], "from": e["frm"],
                          "to": e["to_state"]} for e in events[-50:]]
    return out


def _error(exc: CloudError) -> JSONResponse:
    log.warning("cloud request refused: %s", redact(str(exc)))
    return JSONResponse(exc.public(),
                        status_code=_STATUS.get(exc.category, 500))


def _bearer(request: Request) -> str:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h[:7].lower() == "bearer " else ""


# ------------------------------------------------------------------ router

def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/cloud", tags=["cloud"])
    cfg, queue, vm = ctx.cfg, ctx.queue, ctx.vm
    locks: dict[tuple, threading.Lock] = {}
    locks_guard = threading.Lock()

    def source_lock(job_id: str, idx: int) -> threading.Lock:
        with locks_guard:
            return locks.setdefault((job_id, idx), threading.Lock())

    def operator(request: Request) -> None:
        if not cfg.api_token:
            return                     # loopback only; mount() enforces it
        tok = _bearer(request)
        if not tok or not hmac.compare_digest(tok.encode(),
                                              cfg.api_token.encode()):
            raise CloudError("auth", "missing or wrong operator token")

    def is_admin(request: Request) -> bool:
        tok = _bearer(request)
        return bool(cfg.api_token and tok and hmac.compare_digest(
            tok.encode(), cfg.api_token.encode()))

    def visitor(request: Request) -> None:
        """Routes about ONE job. With open access anyone holding the link
        may use them - the job id, 48 random bits, is the key to that job
        and nothing else. Otherwise the operator token, as before."""
        if cfg.open_access:
            return
        operator(request)

    hist_cache: dict = {"at": None, "data": None}

    def history() -> dict:
        """Typical step timings from recent finished jobs, recomputed at
        most every two minutes."""
        now = queue.clock()
        if hist_cache["data"] is None or now - hist_cache["at"] > 120:
            finished = []
            for j in queue.recent(60):
                if j["state"] != Q.COMPLETED:
                    continue
                try:
                    rows = (queue.trace(j["job_id"]) or {}).get("trace") or []
                except CloudError:
                    rows = []
                finished.append((j, queue.events(j["job_id"]), rows))
                if len(finished) >= ESTIMATE.HISTORY_JOBS:
                    break
            hist_cache.update(at=now, data=ESTIMATE.learn(finished))
        return hist_cache["data"]

    def worker(request: Request, worker_id: str) -> None:
        if not _WORKER_ID.match(worker_id or ""):
            raise CloudError("invalid", "malformed worker id")
        tok = _bearer(request)
        if not tok:
            raise CloudError("auth", "no worker credential")
        ctx.worker_auth(tok)           # raises CloudError("auth")

    def link(file_id: str):
        f = getattr(ctx.drive, "link", None)
        return f(file_id) if f else drive_link(file_id)

    def view(job, vm_row=None, events=None):
        return job_view(job, vm_row, events, link=link)

    def guarded(fn):
        """Run `fn` and turn any CloudError into a safe response."""
        try:
            return fn()
        except CloudError as exc:
            return _error(exc)
        except HTTPException:
            raise
        except Exception as exc:                      # noqa: BLE001
            log.exception("cloud route failed")
            return _error(CloudError("internal", str(exc)))

    # ------------------------------------------------------------ browser

    @r.get("/config")
    def config_view():
        return {"enabled": True, "upload_mode": cfg.upload_mode,
                "chunk_bytes": CHUNK, "chunk_align": CHUNK_ALIGN,
                "max_upload_bytes": cfg.max_upload_bytes,
                "max_files": cfg.max_files_per_job,
                "auth_required": bool(cfg.api_token) and not cfg.open_access,
                "open_access": cfg.open_access}

    @r.post("/uploads")
    def create_upload(body: CreateUpload, request: Request):
        def go():
            visitor(request)
            if len(body.files) > cfg.max_files_per_job:
                raise CloudError("invalid", "too many files")
            files = []
            for f in body.files:
                if not video_ext(f.name) or not f.mime.startswith("video/"):
                    raise CloudError("invalid", f"{safe_name(f.name)} is not "
                                                f"a supported video")
                if f.size > cfg.max_upload_bytes:
                    raise CloudError("invalid", "file too large")
                files.append({"name": safe_name(f.name), "size": f.size,
                              "mime": f.mime})
            settings = ctx.validate_settings(body.settings)
            capped = cfg.open_access and not is_admin(request)
            job, created = queue.create(
                body.idempotency_key, files, settings,
                cap_per_day=cfg.max_jobs_per_day if capped else None)
            origin = request.headers.get("origin")
            targets = []
            for s in job["sources"]:
                t = {"idx": s["idx"], "received": s["received"],
                     "verified": bool(s["verified"])}
                if job["state"] == Q.UPLOADING and not s["verified"]:
                    row = queue.session(job["job_id"], s["idx"])
                    uri = row["session_uri"]
                    if not uri:
                        uri = ctx.drive.start_session(
                            f"{job['job_id']}_{s['idx'] + 1:02d}_{s['name']}",
                            s["mime"], s["size"], cfg.drive_input_folder,
                            origin if cfg.upload_mode == "direct" else None)
                        queue.set_session(job["job_id"], s["idx"], uri)
                    if cfg.upload_mode == "direct":
                        t["mode"], t["session_uri"] = "direct", uri
                    else:
                        t["mode"] = "relay"
                        t["url"] = (f"/api/cloud/uploads/{job['job_id']}"
                                    f"/{s['idx']}")
                targets.append(t)
            return {"job": view(job), "created": created,
                    "files": targets}
        return guarded(go)

    def _sync(job_id: str, idx: int, src: dict) -> tuple[bool, int, dict]:
        """Ask Drive what it holds - the server's view is the truth."""
        done, nxt, meta = ctx.drive.query(src["session_uri"], src["size"])
        if not done:
            queue.record_received(job_id, idx, nxt)
        return done, nxt, meta

    def _finish_source(job_id: str, idx: int, src: dict, meta: dict) -> dict:
        size = int(meta.get("size") or -1)
        if size != src["size"]:
            raise CloudError("verification", f"Drive holds {size} bytes, "
                                             f"{src['size']} were declared")
        queue.mark_source_uploaded(job_id, idx, meta["id"],
                                   meta.get("md5Checksum"), size)
        return {"received": size, "complete": True}

    @r.put("/uploads/{job_id}/{idx}")
    async def relay_chunk(job_id: str, idx: int, request: Request):
        try:
            visitor(request)
            check_job_id(job_id)
            if cfg.upload_mode != "relay":
                raise CloudError("invalid", "relay uploads are disabled")
            m = _RANGE.match(request.headers.get("content-range", ""))
            if not m:
                raise CloudError("invalid", "Content-Range is required")
            start, end, total = (int(x) for x in m.groups())
            data = bytearray()
            async for part in request.stream():
                data += part
                if len(data) > MAX_CHUNK:
                    raise CloudError("invalid", "chunk too large")
            if len(data) != end - start + 1:
                raise CloudError("invalid", "chunk length does not match "
                                            "Content-Range")
        except CloudError as exc:
            return _error(exc)

        def go():
            with source_lock(job_id, idx):
                job = queue.get(job_id)
                if job["state"] != Q.UPLOADING:
                    raise CloudError("invalid", "this job is not accepting "
                                                "uploads")
                src = queue.session(job_id, idx)
                if src["verified"]:
                    return {"received": src["size"], "complete": True}
                if total != src["size"] or end >= total:
                    raise CloudError("invalid", "size differs from the "
                                                "declared size")
                if not src["session_uri"]:
                    raise CloudError("invalid", "no upload session")
                if start != src["received"]:
                    done, nxt, meta = _sync(job_id, idx, src)
                    if done:
                        return _finish_source(job_id, idx, src, meta)
                    if start != nxt:
                        return JSONResponse({"received": nxt,
                                             "complete": False},
                                            status_code=409)
                try:
                    done, nxt, meta = ctx.drive.put_chunk(
                        src["session_uri"], bytes(data), start, total)
                except CloudError as exc:
                    if exc.category == "verification":
                        # the session expired; open a fresh one from zero
                        uri = ctx.drive.start_session(
                            f"{job_id}_{idx + 1:02d}_{src['name']}",
                            src["mime"], src["size"],
                            cfg.drive_input_folder)
                        queue.set_session(job_id, idx, uri)
                        queue.record_received(job_id, idx, 0)
                        return JSONResponse({"received": 0,
                                             "complete": False,
                                             "restarted": True},
                                            status_code=409)
                    raise
                if done:
                    return _finish_source(job_id, idx, src, meta)
                queue.record_received(job_id, idx, nxt)
                return {"received": nxt, "complete": False}

        return await run_in_threadpool(guarded, go)

    @r.get("/uploads/{job_id}/{idx}")
    def upload_offset(job_id: str, idx: int, request: Request):
        def go():
            visitor(request)
            check_job_id(job_id)
            src = queue.session(job_id, idx)
            if src["verified"]:
                return {"received": src["size"], "complete": True}
            if cfg.upload_mode == "relay" and src["session_uri"]:
                done, nxt, meta = _sync(job_id, idx, src)
                if done:
                    return _finish_source(job_id, idx, src, meta)
                return {"received": nxt, "complete": False}
            return {"received": src["received"], "complete": False}
        return guarded(go)

    @r.post("/uploads/{job_id}/complete")
    def complete_upload(job_id: str, request: Request,
                        body: CompleteUpload | None = None):
        def go():
            visitor(request)
            check_job_id(job_id)
            job = queue.get(job_id)
            if (job["state"] == Q.UPLOADING and body
                    and body.settings is not None):
                queue.update_settings(job_id,
                                      ctx.validate_settings(body.settings))
            if job["state"] == Q.UPLOADING and body and body.files:
                if cfg.upload_mode != "direct":
                    raise CloudError("invalid", "file ids are only accepted "
                                                "in direct mode")
                for f in body.files:
                    src = queue.session(job_id, f.idx)
                    meta = ctx.drive.get(f.drive_file_id)
                    if cfg.drive_input_folder not in (meta.get("parents")
                                                      or []):
                        raise CloudError("invalid", "that file is not in the "
                                                    "upload folder")
                    if meta.get("trashed"):
                        raise CloudError("invalid", "that file is trashed")
                    _finish_source(job_id, f.idx, src, meta)
            job, newly = queue.queue_if_uploaded(job_id)
            start = None
            if newly:
                try:
                    queue.set_output_folder(job_id, ctx.drive.create_folder(
                        job_id, cfg.drive_output_folder))
                except CloudError as exc:
                    # not fatal: the worker finds-or-creates it too
                    log.warning("output folder for %s: %s", job_id,
                                exc.category)
                start = vm.request_start(f"job {job_id} queued")
                job = queue.get(job_id)
            return {"job": view(job, queue.vm()), "worker_start": start}
        return guarded(go)

    @r.get("/jobs")
    def list_jobs(request: Request, limit: int = 30, ids: str = ""):
        def go():
            v = queue.vm()
            if is_admin(request) or not cfg.open_access:
                operator(request)
                jobs = queue.recent(max(1, min(100, limit)))
            else:
                # without the admin token, only the jobs this browser
                # made - it names them; nobody can list anyone else's
                jobs = []
                for jid in [x.strip() for x in ids.split(",") if x][:50]:
                    try:
                        check_job_id(jid)
                        jobs.append(queue.get(jid))
                    except CloudError:
                        continue
                jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return {"jobs": [view(j) for j in jobs],
                    "worker": {"state": v["state"]}}
        return guarded(go)

    @r.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        def go():
            visitor(request)
            check_job_id(job_id)
            job, vm_row, ev = queue.get(job_id), queue.vm(),                 queue.events(job_id)
            out = view(job, vm_row, ev)
            out["estimate"] = ESTIMATE.estimate(
                job, ev, vm_row["state"], history(), queue.clock())
            return out
        return guarded(go)

    @r.get("/jobs/{job_id}/trace")
    def get_trace(job_id: str, request: Request):
        def go():
            visitor(request)
            check_job_id(job_id)
            t = queue.trace(job_id)
            if t is None:
                raise CloudError("not_found", "no trace yet")
            return t
        return guarded(go)

    def _output(job_id: str, n: int) -> dict:
        check_job_id(job_id)
        outs = queue.get(job_id).get("outputs") or []
        if not 0 <= n < len(outs) or not outs[n].get("drive_file_id"):
            raise CloudError("not_found", "no such output")
        if getattr(ctx.drive, "open_media", None) is None:
            raise CloudError("not_found", "this store has direct links")
        return outs[n]

    def _stream(out: dict, n: int, rng: str | None = None,
                inline: bool = False):
        if rng and not _RANGE_REQ.match(rng):
            rng = None                     # anything odd: send it whole
        try:
            status, headers, body = ctx.drive.open_media(
                out["drive_file_id"], rng)
        except TypeError:                  # a store without Range support
            status, headers, body = ctx.drive.open_media(out["drive_file_id"])

        def chunks():
            try:
                while True:
                    block = body.read(1024 * 1024)
                    if not block:
                        break
                    yield block
            finally:
                body.close()

        name = safe_name(out.get("name") or f"output-{n}")
        extra = {k.title(): headers[k] for k in ("content-length",
                                                 "content-range")
                 if headers.get(k)}
        return StreamingResponse(
            chunks(), status_code=206 if status == 206 else 200,
            media_type=headers.get("content-type",
                                   "application/octet-stream"),
            headers={"Content-Disposition":
                     f'{"inline" if inline else "attachment"}; '
                     f'filename="{name}"',
                     "Accept-Ranges": "bytes", "Cache-Control": "no-store",
                     **extra})

    @r.get("/jobs/{job_id}/outputs/{n}")
    def get_output(job_id: str, n: int, request: Request):
        """One finished output, for a caller that can send the token in a
        header (a script, curl). Browsers use a download ticket instead -
        see `output_ticket`."""
        try:
            visitor(request)
            return _stream(_output(job_id, n), n)
        except CloudError as exc:
            return _error(exc)

    # A browser downloads a big file properly only when its OWN download
    # manager fetches it: progress, written straight to disk, nothing held
    # in the tab's memory. That manager cannot add an Authorization header,
    # and the token must never go into a URL. So the page trades the token
    # for a ticket: random, good for ONE download of ONE output, for
    # TICKET_S seconds. Seen in a log afterwards, it is already spent.
    # (Pulling 37 MB through fetch() into a Blob first is what failed in
    # the browser with "Failed to fetch" while the same bytes arrived
    # whole through curl.)
    #
    # A VIEW ticket is the second kind, for the page's own <video> and <img>:
    # a player asks for the same file many times (seeking sends Range
    # requests), so it is reusable - but only for VIEW_S, only for that
    # one output, and it is served inline rather than as a download.
    tickets: dict[str, tuple[str, int, float, bool]] = {}
    tickets_guard = threading.Lock()

    @r.post("/jobs/{job_id}/outputs/{n}/ticket")
    def output_ticket(job_id: str, n: int, request: Request,
                      mode: str = "download"):
        def go():
            visitor(request)
            if mode not in ("download", "view"):
                raise CloudError("invalid", "mode is download or view")
            _output(job_id, n)                    # exists, and streamable
            tok = secrets.token_urlsafe(24)
            now = ctx.queue.clock()
            life = VIEW_S if mode == "view" else TICKET_S
            with tickets_guard:
                for k in [k for k, v in tickets.items() if v[2] < now]:
                    tickets.pop(k, None)
                tickets[tok] = (job_id, n, now + life, mode == "view")
            return {"url": f"/api/cloud/download/{tok}",
                    "expires_in": life}
        return guarded(go)

    @r.get("/download/{ticket}")
    def download(ticket: str, request: Request):
        with tickets_guard:
            hit = tickets.get(ticket)
            if hit is not None and not hit[3]:
                tickets.pop(ticket, None)         # a download: single use
        try:
            if hit is None or hit[2] < ctx.queue.clock():
                raise CloudError("not_found", "download link expired or "
                                              "already used")
            return _stream(_output(hit[0], hit[1]), hit[1],
                           rng=request.headers.get("range") if hit[3]
                           else None, inline=hit[3])
        except CloudError as exc:
            return _error(exc)

    @r.get("/options")
    def options():
        """What the pipeline can honour, from JobSettings itself - the same
        lists the studio's Create screen offers, so the two cannot drift."""
        import typing
        try:
            from agents.artifacts import JobSettings
        except Exception:                         # noqa: BLE001
            return _error(CloudError("internal", "agent layer missing"))

        def opts(name):
            ann = JobSettings.model_fields[name].annotation
            return [a for a in typing.get_args(ann) if isinstance(a, str)]
        return {k: opts(k) for k in ("style", "pacing", "music", "captions",
                                     "voiceover", "framing", "resolution",
                                     "quality", "orientation")}

    @r.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request):
        def go():
            visitor(request)
            check_job_id(job_id)
            return view(queue.cancel(job_id), queue.vm())
        return guarded(go)

    @r.post("/reconcile")
    def reconcile_now(request: Request):
        def go():
            operator(request)
            return ctx.reconciler.tick()
        return guarded(go)

    # ------------------------------------------------------------- worker

    @r.post("/worker/hello")
    def worker_hello(body: WorkerHello, request: Request):
        def go():
            worker(request, body.worker_id)
            log.info("worker %s hello ready=%s %s", body.worker_id,
                     body.ready, redact(body.info, 300))
            return vm.hello(body.worker_id, body.ready,
                            [redact(p, 200) for p in body.problems[:10]])
        return guarded(go)

    @r.post("/worker/claim")
    def worker_claim(body: WorkerOnly, request: Request):
        def go():
            worker(request, body.worker_id)
            vm.seen(body.worker_id)
            if not vm.may_claim():
                return Response(status_code=204)
            c = queue.claim(body.worker_id)
            if c is None:
                return Response(status_code=204)
            vm.seen(body.worker_id, activity=True)
            return {"job_id": c.job_id, "claim_token": c.claim_token,
                    "attempt": c.attempt, "settings": c.settings,
                    "sources": c.sources,
                    "output_folder_id": c.output_folder_id,
                    "output_root": cfg.drive_output_folder,
                    "lease_s": cfg.lease_s, "heartbeat_s": cfg.heartbeat_s,
                    "max_job_s": cfg.max_job_s}
        return guarded(go)

    @r.post("/worker/heartbeat")
    def worker_heartbeat(body: WorkerBeat, request: Request):
        def go():
            worker(request, body.worker_id)
            check_job_id(body.job_id)
            out = queue.heartbeat(body.job_id, body.worker_id,
                                  body.claim_token, body.state, body.stage,
                                  body.progress)
            vm.seen(body.worker_id, activity=True)
            return out
        return guarded(go)

    @r.post("/worker/complete")
    def worker_complete(body: WorkerComplete, request: Request):
        def go():
            worker(request, body.worker_id)
            check_job_id(body.job_id)
            job = queue.complete(body.job_id, body.worker_id,
                                 body.claim_token, body.outputs, body.trace)
            vm.seen(body.worker_id, activity=True)
            return {"state": job["state"]}
        return guarded(go)

    @r.post("/worker/fail")
    def worker_fail(body: WorkerFail, request: Request):
        def go():
            worker(request, body.worker_id)
            check_job_id(body.job_id)
            job = queue.fail(body.job_id, body.worker_id, body.claim_token,
                             body.category, body.message, body.retryable)
            vm.seen(body.worker_id, activity=True)
            return {"state": job["state"]}
        return guarded(go)

    @r.post("/worker/idle")
    def worker_idle(body: WorkerOnly, request: Request):
        def go():
            worker(request, body.worker_id)
            return {"action": vm.idle(body.worker_id)}
        return guarded(go)

    return r


def page_route(web_dir):
    """`/cloud` -> frontend/cloud.html. The page asks for the operator token
    itself, so serving it needs none."""
    r = APIRouter()

    @r.get("/cloud", include_in_schema=False)
    def cloud_page():
        f = web_dir / "cloud.html"
        if not f.exists():
            raise HTTPException(404, "cloud.html is missing")
        return FileResponse(f, headers={"Cache-Control": "no-cache"})

    return r
