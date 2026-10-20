"""The data plane. Runs on the worker VM, started by systemd at boot.

    cd server && python -m cloud.worker

  1. Check the host can do the work: ffmpeg, ffprobe, the pipeline imports,
     free disk, and a GPU if one is required. Say hello.
  2. Drain the queue: claim -> download from Drive -> run the existing
     pipeline (as a subprocess, under a deadline) -> upload the deliverables
     -> verify them in Drive -> complete. Then claim again.
  3. Nothing left: ask the control plane. Only when IT says `shutdown`,
     exit with code 75, and the systemd unit powers the machine off.

EXIT CODES
----------
    75  nothing to do: the unit's ExecStopPost powers the VM off
     3  this host cannot work: systemd retries, and gives up by powering off
     0  stopped by a signal (systemctl stop, a shutdown already under way):
        nothing more is done, and in particular nothing powers off

The worker holds no Compute permission. It cannot start or stop anything,
itself included, except by exiting 75.

WHAT IT NEVER DOES
------------------
* complete a job whose deliverables are not verified in Drive;
* report a job it no longer owns - a 409 on heartbeat means another worker
  has it, so this one stops quietly;
* run anything a request supplied. The pipeline command is fixed; the only
  inputs are the job id (12 hex characters, checked) and settings that
  `JobSettings` validates again in the subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .config import CloudConfig
from .gcp import Drive, Http, md5_of, token_source
from .paths import check_job_id, safe_name, source_filename, video_ext, within
from .queue import DOWNLOADING, PROCESSING, UPLOADING_RESULTS, LeaseLost
from .sanitize import CloudError, redact

log = logging.getLogger("alpha.cloud.worker")

EXIT_OK = 0
EXIT_NOT_READY = 3
EXIT_POWEROFF = 75
SERVER_DIR = Path(__file__).resolve().parents[1]
MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".wav", ".m4a",
                  ".aac", ".ts"}


class Stopped(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------- control plane

class ControlPlane:
    """The worker's only way to touch the queue: authenticated HTTP."""

    def __init__(self, base_url: str, auth_header, http: Http | None = None,
                 worker_id: str = ""):
        self.base = base_url.rstrip("/")
        self.auth_header = auth_header
        self.http = http or Http(timeout=30)
        self.worker_id = worker_id

    def _post(self, path: str, payload: dict):
        body = json.dumps({"worker_id": self.worker_id, **payload}).encode()
        r = self.http.request("POST", f"{self.base}/api/cloud/worker/{path}",
                              {"Content-Type": "application/json",
                               "Authorization": self.auth_header()}, body,
                              timeout=30)
        if r.status == 204:
            return None
        data = r.json()
        if r.status == 409 and data.get("category") == "lease_expired":
            raise LeaseLost(payload.get("job_id", "?"))
        if r.status >= 400:
            raise CloudError(data.get("category", "internal"),
                             f"control plane {path}: HTTP {r.status}",
                             r.status)
        return data

    def hello(self, ready: bool, problems: list, info: dict) -> dict:
        return self._post("hello", {"ready": ready, "problems": problems,
                                    "info": info})

    def claim(self) -> dict | None:
        return self._post("claim", {})

    def heartbeat(self, job_id, token, state, stage, progress) -> dict:
        return self._post("heartbeat", {"job_id": job_id,
                                        "claim_token": token, "state": state,
                                        "stage": stage, "progress": progress})

    def complete(self, job_id, token, outputs, trace) -> dict:
        return self._post("complete", {"job_id": job_id,
                                       "claim_token": token,
                                       "outputs": outputs, "trace": trace})

    def fail(self, job_id, token, category, message, retryable) -> dict:
        return self._post("fail", {"job_id": job_id, "claim_token": token,
                                   "category": category,
                                   "message": redact(message, 2000),
                                   "retryable": retryable})

    def idle(self) -> str:
        return (self._post("idle", {}) or {}).get("action", "wait")


# --------------------------------------------------------------- heartbeat

class Heartbeat:
    """Keeps the lease alive and carries progress, on its own thread.

    A heartbeat the control plane refuses with `lease lost` means the job
    belongs to someone else now, and the worker must stop. A heartbeat that
    cannot get through at all is tolerated until the lease would have run
    out anyway - at which point the control plane will have re-queued the
    job, and this worker no longer owns it either."""

    def __init__(self, cp: ControlPlane, job_id: str, token: int,
                 interval: float, lease_s: float, clock=time.time,
                 sleep=time.sleep):
        self.cp, self.job_id, self.token = cp, job_id, token
        self.interval, self.lease_s = interval, lease_s
        self.clock, self.sleep = clock, sleep
        self.state, self.stage = DOWNLOADING, DOWNLOADING
        self.extra: dict = {}
        self.progress_fn = None
        self.stop_reason: str | None = None
        self._last_ok = clock()
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _progress(self) -> dict:
        p = dict(self.extra)
        if self.progress_fn is not None and self.state == PROCESSING:
            try:
                p.update(self.progress_fn() or {})
            except Exception:                         # noqa: BLE001
                pass                       # progress is a nicety, not a duty
        return p

    def beat(self, strict: bool = False) -> None:
        with self._lock:
            progress = self._progress()
            stage = progress.get("current_node") or self.stage
            try:
                resp = self.cp.heartbeat(self.job_id, self.token, self.state,
                                         stage, progress) or {}
                self._last_ok = self.clock()
                if resp.get("cancel"):
                    self.stop_reason = self.stop_reason or "cancelled"
            except LeaseLost:
                self.stop_reason = "lease_lost"
                if strict:
                    raise
            except CloudError:
                if self.clock() - self._last_ok > self.lease_s:
                    self.stop_reason = "lease_lost"
                    if strict:
                        raise LeaseLost(self.job_id) from None
                elif strict:
                    raise

    def set(self, state: str, stage: str | None = None,
            attempts: int = 5) -> None:
        """Move to `state` NOW, synchronously. `complete` is refused unless
        the control plane has seen `uploading_results`, so this must land
        before the worker goes on; it is retried, but not past the lease."""
        self.state, self.stage = state, stage or state
        for i in range(attempts):
            try:
                self.beat(strict=True)
                return
            except LeaseLost:
                raise
            except CloudError:
                if i == attempts - 1:
                    raise
                self.sleep(min(10.0, 2.0 ** i))

    def start(self) -> None:
        def loop():
            while not self._halt.wait(self.interval):
                self.beat()
        self._thread = threading.Thread(target=loop, name="heartbeat",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()

    def should_stop(self) -> bool:
        return self.stop_reason is not None


# ---------------------------------------------------------------- pipeline

def _descendants(pid: int) -> list[int]:
    """Every process under `pid`, read from /proc. Collected BEFORE any is
    killed, so reparenting cannot hide one."""
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def kill_tree(proc: subprocess.Popen) -> None:
    """The pipeline and everything it started - ffmpeg included, which the
    media pool runs in process groups of its own."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=30, check=False)
    else:
        victims = _descendants(proc.pid) if Path("/proc").exists() else []
        for pid in [proc.pid, *victims]:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:                 # pragma: no cover
        pass


class SubprocessPipeline:
    """`python -m cloud.run_one` under a deadline, killable at any moment."""

    def __init__(self, server_dir: Path = SERVER_DIR,
                 python: str = sys.executable, clock=time.time,
                 sleep=time.sleep):
        self.server_dir, self.python = Path(server_dir), python
        self.clock, self.sleep = clock, sleep

    def run(self, job_id: str, settings: dict, timeout_s: float,
            should_stop, workdir: Path) -> dict:
        check_job_id(job_id)
        workdir.mkdir(parents=True, exist_ok=True)
        spec, result = workdir / "cloud_spec.json", workdir / "cloud_result.json"
        spec.write_text(json.dumps({"settings": settings}), encoding="utf-8")
        result.unlink(missing_ok=True)
        cmd = [self.python, "-m", "cloud.run_one", "--job-id", job_id,
               "--spec", str(spec), "--result", str(result)]
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
              if os.name == "nt" else {"start_new_session": True})
        with open(workdir / "cloud_pipeline.log", "ab") as out:
            proc = subprocess.Popen(cmd, cwd=self.server_dir, stdout=out,
                                    stderr=subprocess.STDOUT, **kw)
            started = self.clock()
            while proc.poll() is None:
                if should_stop():
                    kill_tree(proc)
                    return {"ok": False, "stopped": True}
                if self.clock() - started > timeout_s:
                    kill_tree(proc)
                    return {"ok": False, "timeout": True}
                self.sleep(2.0)
        try:
            out = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out = {"ok": False, "error": f"the pipeline exited "
                                         f"{proc.returncode} without a result"}
        out["returncode"] = proc.returncode
        return out


# ----------------------------------------------------------- local readers

def local_progress(job_id: str, outputs_dir: Path) -> dict:
    """Which graph nodes have run, read from the worker's own trace rows -
    the same data `/api/v2/jobs/{id}/trace` serves. Not tracked twice."""
    from agents.trace import shared_tracer
    rows = shared_tracer(outputs=outputs_dir).for_job(job_id)
    done: list[str] = []
    for r in rows:
        if r["status"] == "ok" and r["node"] not in done:
            done.append(r["node"])
    running = [r for r in rows if r["status"] == "running"]
    return {"nodes_done": done,
            "current_node": running[-1]["node"] if running else None,
            # when it started, so the page can tell how far along it is
            "current_started_at": (running[-1].get("started_at")
                                   if running else None)}


def local_trace(job_id: str, outputs_dir: Path) -> dict:
    """The job's trace, for the dashboard once the VM is gone."""
    from agents.trace import shared_tracer
    keep = ("node", "agent", "attempt", "status", "started_at", "ended_at",
            "tokens_in", "tokens_out", "fallback_used", "error", "notes")
    rows = []
    for r in shared_tracer(outputs=outputs_dir).for_job(job_id):
        row = {k: r.get(k) for k in keep}
        for k in ("fallback_used", "error"):
            row[k] = redact(row[k], 300) if row[k] else row[k]
        row["notes"] = [redact(n, 200) for n in (r.get("notes") or [])][:8]
        rows.append(row)
    return {"job_id": job_id, "trace": rows}


# ------------------------------------------------------------- host checks

def gpu_info() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,"
                              "driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20,
                             check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip()[:200] if out.returncode == 0 else None


def host_checks(cfg: CloudConfig, outputs_dir: Path) -> tuple[list, dict]:
    problems, info = [], {}
    for key, name in (("AT_FFMPEG", "ffmpeg"), ("AT_FFPROBE", "ffprobe")):
        exe = os.environ.get(key, name)
        try:
            r = subprocess.run([exe, "-version"], capture_output=True,
                               text=True, timeout=20, check=False)
            ok = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            info[name] = r.stdout.splitlines()[0][:80] if r.stdout else "ok"
        else:
            problems.append(f"{name} is not runnable")
    try:
        import agents.runner                          # noqa: F401
        info["pipeline"] = "importable"
    except Exception as exc:                          # noqa: BLE001
        problems.append(f"the pipeline cannot be imported "
                        f"({type(exc).__name__})")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(outputs_dir).free
    info["free_gb"] = round(free / 1024 ** 3, 1)
    if free < cfg.min_free_bytes:
        problems.append(f"only {info['free_gb']} GB free on the work disk")
    gpu = gpu_info()
    info["gpu"] = gpu or "none"
    if cfg.require_gpu and not gpu:
        problems.append("no GPU is visible to nvidia-smi")
    return problems, info


def worker_identity() -> str:
    """Stable for one boot, different after a reboot: a restarted worker
    process keeps its activation, a rebooted VM starts a new one."""
    host = socket.gethostname()[:40]
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = uuid.uuid4().hex
    return re.sub(r"[^A-Za-z0-9._:-]", "-", f"{host}:{boot[:8]}")


# ------------------------------------------------------------------ worker

class Worker:
    def __init__(self, cfg: CloudConfig, cp: ControlPlane, drive, pipeline,
                 *, uploads_dir: Path, outputs_dir: Path, worker_id: str,
                 clock=time.time, sleep=time.sleep, progress_fn=None,
                 trace_fn=None, checks=None):
        self.cfg, self.cp, self.drive, self.pipeline = cfg, cp, drive, pipeline
        self.uploads_dir = Path(uploads_dir)
        self.outputs_dir = Path(outputs_dir)
        self.worker_id = worker_id
        self.clock, self.sleep = clock, sleep
        self.progress_fn = progress_fn
        self.trace_fn = trace_fn
        self.checks = checks or (lambda: host_checks(cfg, self.outputs_dir))
        self._stopping = False
        self.outcomes: list[tuple[str, str]] = []

    def request_stop(self) -> None:
        self._stopping = True

    # ------------------------------------------------------------ the loop

    def _hello(self, ready: bool, problems: list, info: dict) -> bool:
        deadline = self.clock() + self.cfg.startup_timeout_s
        while True:
            try:
                self.cp.hello(ready, problems, info)
                return True
            except CloudError as exc:
                log.warning("control plane unreachable at hello: %s",
                            exc.category)
                if self.clock() > deadline or self._stopping:
                    return False
                self.sleep(min(30.0, self.cfg.poll_s))

    def run(self) -> int:
        problems, info = self.checks()
        ready = not problems
        log.info("worker %s ready=%s %s", self.worker_id, ready, info)
        if not self._hello(ready, problems, info):
            # Nothing can be done without the control plane, and a VM that
            # waits for it is a VM that bills. It starts us again when it
            # has work.
            log.error("control plane never answered; powering off")
            return EXIT_POWEROFF
        if not ready:
            for p in problems:
                log.error("not ready: %s", p)
            return EXIT_NOT_READY

        unreachable_since: float | None = None
        while not self._stopping:
            try:
                claim = self.cp.claim()
                action = None if claim else self.cp.idle()
                unreachable_since = None
            except CloudError as exc:
                unreachable_since = unreachable_since or self.clock()
                gone = self.clock() - unreachable_since
                log.warning("control plane unreachable for %.0fs: %s", gone,
                            exc.category)
                if gone > max(self.cfg.idle_grace_s, 600.0):
                    log.error("control plane gone too long; powering off")
                    return EXIT_POWEROFF
                self.sleep(self.cfg.poll_s)
                continue
            if claim:
                self.outcomes.append((claim["job_id"], self.process(claim)))
                continue
            if action == "claim":
                continue
            if action == "shutdown":
                log.info("nothing left to do; powering off")
                return EXIT_POWEROFF
            self.sleep(self.cfg.poll_s)
        return EXIT_OK

    # ------------------------------------------------------------- one job

    def _check(self, beat: Heartbeat) -> None:
        if self._stopping:
            raise Stopped("shutdown")
        if beat.stop_reason:
            raise Stopped(beat.stop_reason)

    def process(self, claim: dict) -> str:
        job_id = check_job_id(claim["job_id"])
        token = int(claim["claim_token"])
        beat = Heartbeat(self.cp, job_id, token,
                         float(claim.get("heartbeat_s",
                                         self.cfg.heartbeat_s)),
                         float(claim.get("lease_s", self.cfg.lease_s)),
                         clock=self.clock, sleep=self.sleep)
        if self.progress_fn is not None:
            beat.progress_fn = lambda: self.progress_fn(job_id)
        beat.start()
        try:
            self.download(claim, beat)
            self._check(beat)
            beat.set(PROCESSING)
            self._check(beat)
            build = within(self.outputs_dir, job_id)
            timeout = float(claim.get("max_job_s", self.cfg.max_job_s))
            result = self.pipeline.run(
                job_id, claim.get("settings") or {}, timeout,
                lambda: self._stopping or beat.should_stop(), build)
            self._check(beat)
            if result.get("timeout"):
                raise CloudError("timeout", f"the pipeline ran past "
                                            f"{timeout:.0f}s and was stopped")
            if not result.get("ok"):
                raise CloudError("pipeline", result.get("error")
                                 or "the pipeline delivered no video")
            beat.set(UPLOADING_RESULTS)
            # a cancel that arrived while the pipeline ran is seen on this
            # beat; nothing is uploaded for a job nobody wants any more
            self._check(beat)
            outputs = self.upload_outputs(claim, result, beat)
            self._check(beat)
            trace = None
            if self.trace_fn is not None:
                try:
                    trace = self.trace_fn(job_id)
                except Exception:                     # noqa: BLE001
                    trace = None
            # what the critic concluded, for the finished screen - read from
            # the pipeline's own result, never inferred here
            trace = {**(trace or {}), "summary": {
                k: result.get(k) for k in ("verdict", "score", "attempts",
                                           "mode")}}
            beat.stop()
            self.cp.complete(job_id, token, outputs, trace)
            if not self.cfg.keep_media:
                self.cleanup(job_id)
            return "completed"
        except Stopped as s:
            if s.reason == "lease_lost":
                log.warning("%s: lease lost; leaving it to its new owner",
                            job_id)
                return "abandoned"
            if s.reason == "cancelled":
                self._report(job_id, token, "cancelled", "cancelled by user",
                             False)
                return "cancelled"
            self._report(job_id, token, "transient", "the worker is shutting "
                                                     "down", True)
            return "interrupted"
        except LeaseLost:
            return "abandoned"
        except CloudError as exc:
            retry = exc.retryable
            if exc.category == "timeout":
                # one more try, not max_attempts more: a job that ran out of
                # time once will probably run out again
                retry = int(claim.get("attempt", 1)) <= 1
            self._report(job_id, token, exc.category, exc.detail, retry)
            return "failed"
        except Exception as exc:                      # noqa: BLE001
            log.exception("%s: worker error", job_id)
            self._report(job_id, token, "internal",
                         f"{type(exc).__name__}: {exc}", None)
            return "failed"
        finally:
            beat.stop()

    def _report(self, job_id, token, category, message, retryable) -> None:
        for i in range(3):
            try:
                self.cp.fail(job_id, token, category, message, retryable)
                return
            except LeaseLost:
                return
            except CloudError:
                self.sleep(2.0 ** i)
        log.error("%s: could not report failure; its lease will expire",
                  job_id)

    # ----------------------------------------------------------- transfers

    def download(self, claim: dict, beat: Heartbeat) -> list[Path]:
        job_id = claim["job_id"]
        ws = within(self.uploads_dir, job_id)
        ws.mkdir(parents=True, exist_ok=True)
        sources = sorted(claim.get("sources") or [], key=lambda s: s["idx"])
        if not sources:
            raise CloudError("invalid", "the job has no source files")
        need = sum(int(s["size"]) for s in sources) * 4 \
            + self.cfg.min_free_bytes
        free = shutil.disk_usage(ws).free
        if free < need:
            raise CloudError("quota", f"worker disk: {free} bytes free, "
                                      f"{need} needed")
        paths = []
        for n, s in enumerate(sources):
            if not s.get("drive_file_id"):
                raise CloudError("invalid", "a source has no Drive file")
            dest = within(ws, source_filename(s["idx"], s["name"]))
            if not video_ext(dest.name):
                raise CloudError("invalid", "unsupported source file type")
            beat.extra = {"downloading": n + 1, "of": len(sources)}
            self.drive.download(s["drive_file_id"], dest, int(s["size"]),
                                s.get("md5"))
            paths.append(dest)
            self._check(beat)
        beat.extra = {"downloaded": len(paths)}
        return paths

    def upload_outputs(self, claim: dict, result: dict,
                       beat: Heartbeat) -> list[dict]:
        job_id = claim["job_id"]
        build = within(self.outputs_dir, job_id)
        folder = claim.get("output_folder_id") or self.drive.create_folder(
            job_id, claim["output_root"])
        items = [("video", result.get("video_path"), "video/mp4"),
                 ("thumbnail", result.get("thumbnail_path"), "image/jpeg"),
                 ("youtube", build / "youtube.txt", "text/plain"),
                 ("report", build / "output.json", "application/json"),
                 ("plan", build / "edit_plan.json", "application/json"),
                 ("log", build / "run.log", "text/plain")]
        outputs = []
        for kind, raw, mime in items:
            path = Path(raw).resolve() if raw else None
            if path is None or not path.is_file():
                if kind == "video":
                    raise CloudError("pipeline", "no video was produced")
                continue
            try:
                path.relative_to(build)
            except ValueError:
                # deliverables come from this job's own folder, never from
                # wherever a result file happens to point
                raise CloudError("internal", "a deliverable is outside the "
                                             "job's folder") from None
            name = (safe_name(path.name) if kind == "video"
                    else f"{job_id}_{safe_name(path.name)}")
            size, md5 = path.stat().st_size, md5_of(path)
            same = [f for f in self.drive.find(folder, name)
                    if int(f.get("size") or -1) == size
                    and (f.get("md5Checksum") or "").lower() == md5]
            # an earlier attempt that died after uploading this exact file
            # already did the work; uploading it twice would only duplicate it
            meta = same[0] if same else self.drive.upload(path, name, mime,
                                                          folder)
            outputs.append({"kind": kind, "name": name, "size": size,
                            "md5": md5, "drive_file_id": meta["id"],
                            "verified": True})
            beat.extra = {"uploaded": len(outputs)}
            self._check(beat)
        return outputs

    def cleanup(self, job_id: str) -> None:
        """After the control plane has ACKNOWLEDGED completion: the source
        copies and the heavy render files go; logs, plans, checkpoints and
        the shared caches stay, so the job can still be explained."""
        ws = within(self.uploads_dir, job_id)
        shutil.rmtree(ws, ignore_errors=True)
        build = within(self.outputs_dir, job_id)
        for sub in ("parts", "attempts"):
            shutil.rmtree(build / sub, ignore_errors=True)
        if build.exists():
            for p in build.iterdir():
                if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES:
                    p.unlink(missing_ok=True)


# -------------------------------------------------------------------- main

def _store(cfg: CloudConfig, http, tokens):
    if cfg.storage == "gcs":
        from .gcs import GcsStore
        return GcsStore(http, tokens, cfg.gcs_bucket)
    return Drive(http, tokens, cfg.shared_drive)


class _Redacting(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), 2000)
        record.args = ()
        return True


def main(argv=None) -> int:
    handler = logging.StreamHandler()
    handler.addFilter(_Redacting())
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s")
    cfg = CloudConfig.from_env()
    problems = cfg.problems("worker")
    if problems:
        for p in problems:
            log.error("worker cannot start: %s", p)
        return EXIT_NOT_READY

    import config as app_config                       # server/config.py
    tokens = token_source(cfg.token_source)
    http = Http()
    if cfg.worker_auth == "google_id_token":
        def auth():
            return "Bearer " + tokens.id_token(cfg.public_url)
    else:
        def auth():
            return "Bearer " + cfg.worker_token
    wid = worker_identity()
    outputs = Path(app_config.OUTPUTS_DIR)
    worker = Worker(
        cfg, ControlPlane(cfg.public_url, auth, http, wid),
        _store(cfg, http, tokens), SubprocessPipeline(),
        uploads_dir=Path(app_config.UPLOADS_DIR), outputs_dir=outputs,
        worker_id=wid,
        progress_fn=lambda j: local_progress(j, outputs),
        trace_fn=lambda j: local_trace(j, outputs))
    signal.signal(signal.SIGTERM, lambda *_: worker.request_stop())
    return worker.run()


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(main())
