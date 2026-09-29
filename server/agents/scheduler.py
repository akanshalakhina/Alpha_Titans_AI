
"""
Job-level render scheduler.

The smallest thing that satisfies the architecture's requirement, and no more:
submit render work, bound the concurrency, know what is running, cancel it,
time it out, and report how it ended. No broker, no worker daemon, no
distributed queue - this is one operator on one machine.

WHAT IT ADDS OVER CALLING THE POOL DIRECTLY
-------------------------------------------
The pool bounds concurrency and owns processes. The scheduler adds the JOB as
a first-class unit on top of that:

  * work is tagged with its job id, so `cancel(job_id)` reaches every process
    that job started - including ones queued but not yet running
  * a deadline belongs to the work item, and blowing it kills the process tree
    rather than merely abandoning the wait
  * `running()` / `pending()` answer "what is this machine doing right now",
    which nothing could answer before

AWAITABLE, SO THE GRAPH NODE DOES NOT SIT BLOCKED
-------------------------------------------------
`submit` returns a `ScheduledWork` wrapping a plain `concurrent.futures.Future`.
An async caller awaits it through `asyncio.wrap_future`, so the event loop stays
free for the whole render instead of one node monopolising it.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

from .render_pool import MediaPool, ProcessCancelled, ProcessTimeout


class WorkCancelled(RuntimeError):
    """The work item was cancelled before or during execution."""


@dataclass
class ScheduledWork:
    """A submitted unit of render work."""

    work_id: str
    job_id: str
    label: str
    future: Future
    timeout_s: float | None = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancelled: bool = False
    error: str | None = None
    timer: Any = None          # the deadline timer, if any

    @property
    def running(self) -> bool:
        return self.started_at is not None and self.finished_at is None

    @property
    def pending(self) -> bool:
        return self.started_at is None and not self.future.done()

    @property
    def done(self) -> bool:
        return self.future.done()

    def wait(self, timeout: float | None = None) -> Any:
        return self.future.result(timeout=timeout)

    def __await__(self):
        """Await the work without blocking the event loop."""
        return asyncio.wrap_future(self.future).__await__()

    def describe(self) -> dict:
        return {"work_id": self.work_id, "job_id": self.job_id,
                "label": self.label, "pending": self.pending,
                "running": self.running, "done": self.done,
                "cancelled": self.cancelled, "error": self.error,
                "wall_s": round((self.finished_at or time.time())
                                - (self.started_at or self.submitted_at), 2)}


class RenderScheduler:
    """Job-aware front end to one bounded MediaPool."""

    def __init__(self, pool: MediaPool | None = None,
                 max_workers: int | None = None,
                 default_timeout_s: float | None = None):
        self.pool = pool or MediaPool(max_workers=max_workers,
                                      name="scheduler")
        self._owns_pool = pool is None
        self.default_timeout_s = default_timeout_s
        self._lock = threading.RLock()
        self._work: dict[str, ScheduledWork] = {}

    # ------------------------------------------------------------ submit

    def submit(self, job_id: str, fn: Callable[..., Any], *args,
               label: str = "", timeout_s: float | None = None,
               **kwargs) -> ScheduledWork:
        """Queue work for a job. Returns immediately.

        The deadline is enforced around the whole work item; the pool enforces
        its own on each individual process. Both end in a tree kill, so an
        overrun never leaves ffmpeg behind.
        """
        timeout_s = timeout_s if timeout_s is not None else self.default_timeout_s
        work_id = f"w_{uuid.uuid4().hex[:10]}"

        # The work item is built BEFORE anything is submitted. On a free pool
        # the worker starts the instant submit() is called, so a body that
        # looked the item up afterwards raced and lost - it could run before
        # the item existed. `_body` only touches fields that exist at
        # construction, so closing over it directly is safe.
        work = ScheduledWork(work_id=work_id, job_id=job_id, label=label,
                             future=Future(), timeout_s=timeout_s)

        def _body():
            work.started_at = time.time()
            if self.pool.is_cancelled(job_id):
                work.cancelled = True
                work.finished_at = time.time()
                raise WorkCancelled(f"job {job_id} was cancelled while "
                                    f"{label or 'work'} was queued")
            try:
                return fn(*args, **kwargs)
            except (ProcessCancelled, WorkCancelled) as exc:
                work.cancelled = True
                work.error = str(exc)
                raise
            except Exception as exc:                  # noqa: BLE001
                work.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                work.finished_at = time.time()

        with self._lock:
            self._work[work_id] = work
        work.future = self.pool.submit(_body)

        if timeout_s:
            t = threading.Timer(timeout_s, self._expire, args=(work_id,))
            t.daemon = True
            t.start()
            work.timer = t
        return work

    def run(self, job_id: str, fn: Callable[..., Any], *args,
            label: str = "", timeout_s: float | None = None,
            **kwargs) -> Any:
        """Submit and wait. For synchronous callers (the non-render nodes)."""
        return self.submit(job_id, fn, *args, label=label,
                           timeout_s=timeout_s, **kwargs).wait()

    async def run_async(self, job_id: str, fn: Callable[..., Any], *args,
                        label: str = "", timeout_s: float | None = None,
                        **kwargs) -> Any:
        """Submit and await, without blocking the event loop."""
        return await self.submit(job_id, fn, *args, label=label,
                                 timeout_s=timeout_s, **kwargs)

    # ------------------------------------------------------------ expiry

    def _expire(self, work_id: str) -> None:
        with self._lock:
            work = self._work.get(work_id)
        if work is None or work.done:
            return
        # kill, but do NOT ban the job: a deadline ends this attempt, it does
        # not mean the job may never run again
        killed = self.pool.kill_job_processes(
            work.job_id, f"deadline {work.timeout_s}s on {work.label or 'work'}")
        work.error = (f"timed out after {work.timeout_s}s; killed {killed} "
                      f"process tree(s)")

    # ------------------------------------------------------ cancellation

    def cancel(self, job_id: str) -> dict:
        """Cancel a job: kill its live processes, drop anything still queued."""
        killed = self.pool.cancel_job(job_id)
        dropped = 0
        with self._lock:
            mine = [w for w in self._work.values() if w.job_id == job_id]
        for w in mine:
            if w.pending and w.future.cancel():
                w.cancelled = True
                w.finished_at = time.time()
                dropped += 1
            elif w.running:
                w.cancelled = True
        return {"job_id": job_id, "processes_killed": killed,
                "queued_dropped": dropped}

    def clear_cancel(self, job_id: str) -> None:
        self.pool.clear_cancel(job_id)

    # ------------------------------------------------------------- state

    def running(self) -> list[dict]:
        with self._lock:
            return [w.describe() for w in self._work.values() if w.running]

    def pending(self) -> list[dict]:
        with self._lock:
            return [w.describe() for w in self._work.values() if w.pending]

    def running_jobs(self) -> set[str]:
        with self._lock:
            return {w.job_id for w in self._work.values() if w.running}

    def for_job(self, job_id: str) -> list[dict]:
        with self._lock:
            return [w.describe() for w in self._work.values()
                    if w.job_id == job_id]

    @property
    def stats(self) -> dict:
        with self._lock:
            work = list(self._work.values())
        return {"max_workers": self.pool.max_workers,
                "submitted": len(work),
                "running": sum(1 for w in work if w.running),
                "pending": sum(1 for w in work if w.pending),
                "cancelled": sum(1 for w in work if w.cancelled),
                "failed": sum(1 for w in work if w.error and not w.cancelled),
                "live_processes": len(self.pool.processes)}

    def close(self, wait: bool = True) -> None:
        if self._owns_pool:
            self.pool.close(wait=wait)

    def __enter__(self) -> "RenderScheduler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["RenderScheduler", "ScheduledWork", "WorkCancelled",
           "ProcessTimeout", "ProcessCancelled"]
