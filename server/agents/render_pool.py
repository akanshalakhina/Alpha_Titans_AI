"""
Bounded media pool - and the only owner of an ffmpeg process.

TWO RULES THIS ENFORCES
-----------------------
1. ffmpeg never executes inside a LangGraph node. Graph nodes are short and
   checkpointable; a four-minute encode on the graph's own call stack blocks
   its bookkeeping and, on a replayed node, would run twice.
2. Every process this package starts is OWNED: the pool holds the Popen
   handle, can poll it, wait on it, terminate it, kill its whole tree and
   report its exit status. Nothing can be orphaned.

WHY TREE-KILL, NOT KILL
-----------------------
The process this pool starts is `python build_video.py <stage>`. THAT process
spawns ffmpeg. So ffmpeg is a GRANDCHILD, and killing our direct child leaves
it running, holding the CPU and the output file. Only a tree kill reaches it:

    Windows   taskkill /F /T /PID   (/T = the whole tree)
    POSIX     os.killpg on the session leader we created with start_new_session

Processes are therefore always started in their own group/session, which is
what makes the tree addressable in the first place.

WHAT CHANGED IN PHASE 2 STEP 3
------------------------------
Previously the pool submitted opaque callables and the Popen handle was a
local inside the callable - unreachable once the call was in flight. So a
timeout could only abandon the WAIT while ffmpeg kept running. `run_process`
inverts that: the pool creates the process, so the deadline is enforceable and
cancellation is real.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

IS_WINDOWS = os.name == "nt"


def default_workers() -> int:
    """Half the cores, capped at 4.

    An x264 encode saturates whatever it is given, so four concurrent renders
    are not four times faster - they are four times slower each, plus context
    switching. The cap is a throughput decision, not a timidity one.
    """
    try:
        cpu = os.cpu_count() or 2
    except NotImplementedError:              # pragma: no cover
        cpu = 2
    return max(1, min(cpu // 2, 4))


class ProcessTimeout(RuntimeError):
    """A managed process exceeded its deadline and was killed."""


class ProcessCancelled(RuntimeError):
    """A managed process was cancelled and killed."""


# --------------------------------------------------------------- process

@dataclass
class ManagedProcess:
    """One subprocess the pool owns for its whole life."""

    popen: subprocess.Popen
    cmd: list[str]
    job_id: str | None = None
    label: str = ""
    started_at: float = field(default_factory=time.time)
    killed: bool = False
    kill_reason: str = ""

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def returncode(self) -> int | None:
        return self.popen.returncode

    def poll(self) -> int | None:
        return self.popen.poll()

    @property
    def running(self) -> bool:
        return self.popen.poll() is None

    def wait(self, timeout: float | None = None) -> int:
        return self.popen.wait(timeout=timeout)

    def elapsed(self) -> float:
        return round(time.time() - self.started_at, 2)

    # ------------------------------------------------------------ killing

    def terminate(self) -> None:
        """Ask nicely. Does NOT reach a grandchild ffmpeg on its own."""
        if self.running:
            try:
                self.popen.terminate()
            except OSError:                          # pragma: no cover
                pass

    def kill_tree(self, reason: str = "") -> bool:
        """Kill this process AND everything it spawned. Returns True if it
        was alive and is now gone.

        This is the method that actually reaches ffmpeg, because ffmpeg is a
        grandchild of what we started.
        """
        if not self.running:
            return False
        self.killed = True
        self.kill_reason = reason
        try:
            if IS_WINDOWS:
                # /T walks the tree, /F does not ask
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.pid)],
                               capture_output=True, timeout=30)
            else:                                    # pragma: no cover
                try:
                    os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            pass
        # taskkill is asynchronous; make sure the handle is really reaped so
        # the caller cannot observe a zombie
        try:
            self.popen.wait(timeout=15)
        except subprocess.TimeoutExpired:            # pragma: no cover
            try:
                self.popen.kill()
            except OSError:
                pass
        return True

    def describe(self) -> dict:
        return {"pid": self.pid, "job_id": self.job_id, "label": self.label,
                "cmd": " ".join(str(c) for c in self.cmd[:4]),
                "running": self.running, "returncode": self.returncode,
                "elapsed_s": self.elapsed(), "killed": self.killed,
                "kill_reason": self.kill_reason}


# ------------------------------------------------------------------ pool

class MediaPool:
    """The only place in this package that is allowed to start ffmpeg."""

    def __init__(self, max_workers: int | None = None, name: str = "media"):
        self.max_workers = max_workers or default_workers()
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers,
                                        thread_name_prefix=f"at-{name}")
        self._lock = threading.RLock()
        self._submitted = 0
        self._completed = 0
        self._closed = False
        self._processes: dict[int, ManagedProcess] = {}
        self._cancelled_jobs: set[str] = set()

    # ------------------------------------------------------------ submit

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        if self._closed:
            raise RuntimeError("media pool is closed")
        with self._lock:
            self._submitted += 1
        fut = self._pool.submit(fn, *args, **kwargs)
        fut.add_done_callback(self._done)
        return fut

    def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Submit and wait, on the caller's thread.

        Still used by short, non-process work. Long process work goes through
        `run_process`, which the pool owns and can kill.
        """
        return self.submit(fn, *args, **kwargs).result()

    def run_timeout(self, timeout_s: float | None, fn: Callable[..., Any],
                    *args, **kwargs) -> Any:
        """Submit and wait with a deadline on the WAIT only.

        Kept for non-process callables. For anything that starts a process use
        `run_process`, whose deadline kills the process rather than merely
        giving up on it.
        """
        return self.submit(fn, *args, **kwargs).result(timeout=timeout_s)

    def map_bounded(self, fn: Callable[..., Any],
                    items: list[Any]) -> list[Any]:
        futures = [self.submit(fn, it) for it in items]
        return [f.result() for f in futures]

    def _done(self, _fut: Future) -> None:
        with self._lock:
            self._completed += 1

    # ----------------------------------------------------- owned process

    def run_process(self, cmd: list[str], *, env: dict | None = None,
                    on_line: Callable[[str], None] | None = None,
                    timeout_s: float | None = None,
                    job_id: str | None = None,
                    label: str = "") -> tuple[int, str]:
        """Start a process the pool OWNS, stream it, and enforce its deadline.

        Runs on the CALLING thread. That is deliberate: callers are already
        executing on a pool worker, so submitting again would consume a second
        slot and can deadlock a saturated pool. What the pool provides here is
        ownership and the ability to kill, not another queue hop.

        On timeout or cancellation the whole process TREE is killed, so the
        ffmpeg grandchild dies with it and nothing is orphaned.
        """
        if job_id is not None and self.is_cancelled(job_id):
            raise ProcessCancelled(f"job {job_id} was cancelled before "
                                   f"{label or 'the process'} started")

        full_env = dict(os.environ)
        full_env.update(env or {})

        popen = subprocess.Popen(
            [str(c) for c in cmd], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=full_env,
            # Own group/session, so the tree is addressable for a kill.
            **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
               if IS_WINDOWS else {"start_new_session": True}))

        managed = ManagedProcess(popen=popen, cmd=list(cmd), job_id=job_id,
                                 label=label)
        with self._lock:
            self._processes[managed.pid] = managed

        watchdog = None
        if timeout_s:
            watchdog = threading.Timer(
                timeout_s, managed.kill_tree,
                kwargs={"reason": f"timeout after {timeout_s}s"})
            watchdog.daemon = True
            watchdog.start()

        lines: list[str] = []
        try:
            assert popen.stdout is not None
            for line in popen.stdout:
                line = line.rstrip()
                if line:
                    lines.append(line)
                    if on_line:
                        on_line(line)
            rc = popen.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()
            with self._lock:
                self._processes.pop(managed.pid, None)

        out = "\n".join(lines)
        if managed.killed:
            if "cancel" in managed.kill_reason:
                raise ProcessCancelled(
                    f"{label or 'process'} cancelled: {managed.kill_reason}")
            raise ProcessTimeout(
                f"{label or 'process'} exceeded its deadline "
                f"({managed.kill_reason}) after {managed.elapsed()}s "
                f"and its process tree was killed")
        return rc, out

    # ------------------------------------------------------ cancellation

    def kill_job_processes(self, job_id: str, reason: str) -> int:
        """Kill this job's live processes WITHOUT banning the job.

        A deadline uses this rather than `cancel_job`: blowing a timeout means
        this attempt is over, not that the job may never run again. Setting
        the persistent flag and clearing it afterwards would also race - the
        future resolves the moment the process dies, so a caller could observe
        the job as cancelled before the clear ran.
        """
        with self._lock:
            mine = [p for p in self._processes.values() if p.job_id == job_id]
        return sum(1 for p in mine if p.kill_tree(reason=reason))

    def cancel_job(self, job_id: str) -> int:
        """Kill every process this job owns AND refuse further work for it.

        The job id is remembered, so a process that has not started yet is
        refused rather than racing into existence after the cancel. Use
        `kill_job_processes` when the intent is to stop the current attempt
        rather than to ban the job.
        """
        with self._lock:
            self._cancelled_jobs.add(job_id)
        return self.kill_job_processes(job_id, f"cancelled job {job_id}")

    def clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancelled_jobs.discard(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled_jobs

    def kill_all(self, reason: str = "shutdown") -> int:
        with self._lock:
            mine = list(self._processes.values())
        return sum(1 for p in mine if p.kill_tree(reason=reason))

    # ------------------------------------------------------------- stats

    @property
    def processes(self) -> list[ManagedProcess]:
        with self._lock:
            return list(self._processes.values())

    def active_processes(self, job_id: str | None = None) -> list[dict]:
        return [p.describe() for p in self.processes
                if job_id is None or p.job_id == job_id]

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"max_workers": self.max_workers,
                    "submitted": self._submitted,
                    "completed": self._completed,
                    "live_processes": len(self._processes)}

    def close(self, wait: bool = True) -> None:
        self._closed = True
        # never leave a process behind when the pool goes away
        self.kill_all(reason="pool closed")
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "MediaPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_GLOBAL: MediaPool | None = None
_GLOBAL_LOCK = threading.Lock()


def global_pool() -> MediaPool:
    """Process-wide pool, so two concurrent jobs share one concurrency bound."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None or _GLOBAL._closed:
            _GLOBAL = MediaPool(name="global")
        return _GLOBAL


__all__ = ["MediaPool", "ManagedProcess", "global_pool", "default_workers",
           "ProcessTimeout", "ProcessCancelled", "IS_WINDOWS"]
