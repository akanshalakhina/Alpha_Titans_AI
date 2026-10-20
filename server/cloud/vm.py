"""The worker VM's lifecycle, as decided by the control plane.

The control plane is the only thing that STARTS the worker, and it stops it
only as a safety net. In normal operation the worker powers itself off when
the control plane tells it there is nothing left to do (see `idle`). The
worker has no Compute permission at all.

SINGLE FLIGHT
-------------
`request_start` decides inside one `BEGIN IMMEDIATE` transaction whether a
start is needed, and records that it is starting BEFORE making the Compute
call - outside the transaction, so a slow API never holds the queue's lock.
Ten uploads finishing in the same second produce one `instances.start`. If
the process dies between the commit and the call, the reconciler finds a VM
row saying `starting` beside a VM that is down, and starts it again.

FLAPPING
--------
Jobs are committed before a start is requested, so a worker that boots always
has something to find. Starts are spaced by `min_start_interval_s`, backed
off exponentially after failures, and capped per hour, after which the start
circuit opens and stays visibly open until the hour rolls over.
"""

from __future__ import annotations

import logging
import time

from .queue import (VM_READY, VM_START_FAILED, VM_STARTING, VM_STOPPED,
                    VM_STOPPING, JobQueue)
from .sanitize import CloudError

log = logging.getLogger("alpha.cloud.vm")

RUNNING_LIKE = frozenset({"PROVISIONING", "STAGING", "RUNNING", "REPAIRING"})
DOWN = frozenset({"TERMINATED", "STOPPED", "SUSPENDED"})
GOING_DOWN = frozenset({"STOPPING", "SUSPENDING"})


class VmManager:
    def __init__(self, queue: JobQueue, compute, cfg, clock=time.time):
        self.queue = queue
        self.compute = compute               # None: VM control not configured
        self.cfg = cfg
        self.clock = clock

    # --------------------------------------------------------------- start

    def start_backoff(self, failures: int) -> float:
        if not failures:
            return 0.0
        return min(3600.0, max(self.cfg.min_start_interval_s, 60.0)
                   * (2 ** (failures - 1)))

    def request_start(self, reason: str) -> str:
        """Ask for the worker VM, at most once however often this is
        called. Returns what happened, for the caller and the trace:
        started, already_active, stopping, throttled, circuit_open,
        failed or not_configured."""
        if self.compute is None:
            return "not_configured"
        now = self.clock()
        with self.queue.vm_tx() as (con, vm):
            st = vm["state"]
            if st in (VM_STARTING, VM_READY):
                return "already_active"
            if st == VM_STOPPING:
                # the reconciler starts it again once it is down
                return "stopping"
            wait = max(self.cfg.min_start_interval_s,
                       self.start_backoff(vm["start_failures"]))
            if vm["last_start_at"] and now - vm["last_start_at"] < wait:
                return "throttled"
            recent = [t for t in vm["starts"] if now - t < 3600]
            if len(recent) >= self.cfg.max_starts_per_hour:
                self.queue.vm_update(con, last_error=(
                    f"start circuit open: {len(recent)} starts in the last "
                    f"hour"))
                return "circuit_open"
            self.queue.vm_update(con, state=VM_STARTING, op_started_at=now,
                                 last_start_at=now, starts=recent + [now],
                                 ready_at=None, worker_id=None,
                                 activation_at=None, stop_requested_at=None)
        log.info("starting the worker VM (%s)", reason)
        self.queue.mark_starting()
        try:
            self.compute.start()
        except CloudError as exc:
            self._start_failed(f"start failed ({exc.category}): {exc.detail}")
            return "failed"
        return "started"

    def _start_failed(self, why: str) -> None:
        with self.queue.vm_tx() as (con, vm):
            self.queue.vm_update(con, state=VM_START_FAILED,
                                 start_failures=vm["start_failures"] + 1,
                                 last_error=why)
        self.queue.unmark_starting(f"worker start failed: {why[:120]}")
        log.warning("worker start failed: %s", why)

    def _stop(self, why: str) -> bool:
        log.warning("stopping the worker VM: %s", why)
        try:
            self.compute.stop()
        except CloudError as exc:
            self.queue.vm_update(last_error=f"stop failed: {exc.detail}")
            return False
        # Keep the CAUSE. "The worker is not ready" with the worker's own
        # reason ("ffmpeg is not runnable") thrown away tells an operator
        # that something is wrong and nothing about what.
        before = self.queue.vm()["last_error"]
        self.queue.vm_update(
            state=VM_STOPPING, stop_requested_at=self.clock(),
            last_error=f"{why} - {before}" if before and before != why
            else why)
        return True

    # ------------------------------------------------- what the worker says

    def hello(self, worker_id: str, ready: bool, problems: list[str]) -> dict:
        now = self.clock()
        with self.queue.vm_tx() as (con, vm):
            if ready:
                same = vm["worker_id"] == worker_id and vm["activation_at"]
                self.queue.vm_update(
                    con, state=VM_READY, worker_id=worker_id, ready_at=now,
                    last_seen_at=now, last_activity_at=now,
                    activation_at=vm["activation_at"] if same else now,
                    start_failures=0, last_error=None)
            else:
                self.queue.vm_update(
                    con, state=VM_START_FAILED, worker_id=worker_id,
                    last_seen_at=now,
                    start_failures=vm["start_failures"] + 1,
                    last_error="worker not ready: " + "; ".join(problems)[:300])
        if not ready:
            self.queue.unmark_starting("the worker reported it is not ready")
        return {"accepted": bool(ready)}

    def seen(self, worker_id: str, activity: bool = False) -> None:
        """Any authenticated call from a worker. A worker that is talking
        to us is proof the VM came up, whatever the row says."""
        now = self.clock()
        with self.queue.vm_tx() as (con, vm):
            cols = {"last_seen_at": now}
            if activity:
                cols["last_activity_at"] = now
            if vm["state"] == VM_STARTING:
                cols.update(state=VM_READY, ready_at=now,
                            worker_id=worker_id,
                            activation_at=vm["activation_at"] or now)
            self.queue.vm_update(con, **cols)

    def may_claim(self) -> bool:
        """False once this activation has run its course, or once the
        worker has been told to shut down. It finishes what it holds, powers
        off, and a fresh activation picks up whatever is left."""
        vm = self.queue.vm()
        if vm["state"] == VM_STOPPING:
            return False
        started = vm["activation_at"]
        return not (started and self.clock() - started
                    > self.cfg.max_activation_s)

    def idle(self, worker_id: str) -> str:
        """The worker found nothing to claim. `claim`, `wait` or `shutdown`.

        Decided inside ONE transaction, with the counts read in it: either a
        job is visible and the worker is told to claim it, or none is and the
        VM is marked `stopping` before anything else can queue a job. A job
        queued a moment later sees `stopping`, and the reconciler starts the
        VM again once it is down. Nothing falls between the two."""
        now = self.clock()
        with self.queue.vm_tx() as (con, vm):
            c = self.queue.counts(con)
            over = bool(vm["activation_at"]) and (
                now - vm["activation_at"] > self.cfg.max_activation_s)
            if c["claimable"] and not over and vm["state"] != VM_STOPPING:
                self.queue.vm_update(con, last_seen_at=now)
                return "claim"
            if c["active"]:
                self.queue.vm_update(con, last_seen_at=now)
                return "wait"
            marks = [t for t in (vm["last_activity_at"], vm["ready_at"],
                                 vm["activation_at"]) if t]
            since = max(marks) if marks else now
            if over or now - since >= self.cfg.idle_grace_s:
                self.queue.vm_update(con, state=VM_STOPPING,
                                     stop_requested_at=now, last_seen_at=now)
                log.info("worker %s told to shut down (%s)", worker_id,
                         "activation limit" if over else "idle")
                return "shutdown"
            self.queue.vm_update(con, last_seen_at=now)
            return "wait"

    # ------------------------------------------------------------ reconcile

    def reconcile(self) -> dict:
        """Compare what the control plane believes with what Compute Engine
        reports, and repair the difference. Safe to call at any time and as
        often as wanted: every action is guarded by the VM row's state."""
        if self.compute is None:
            return {"vm": "not_configured", **self.queue.counts()}
        try:
            real = self.compute.status()
        except CloudError as exc:
            self.queue.vm_update(last_error=f"status failed: {exc.detail}")
            return {"vm": "unknown", "error": exc.category,
                    **self.queue.counts()}
        now = self.clock()
        vm = self.queue.vm()
        c = self.queue.counts()
        st = vm["state"]
        actions: list[str] = []

        if real in DOWN:
            if st in (VM_READY, VM_STOPPING):
                self.queue.vm_update(state=VM_STOPPED, worker_id=None,
                                     activation_at=None,
                                     stop_requested_at=None)
                actions.append("vm_stopped")
                st = VM_STOPPED
            elif st == VM_STARTING and now - (vm["op_started_at"] or now) > 90:
                self._start_failed("the VM did not start")
                actions.append("start_did_not_take")
                st = VM_START_FAILED
            if c["claimable"] and st in (VM_STOPPED, VM_START_FAILED):
                # covers an upload whose own start request was lost, and
                # every retry whose backoff has now passed
                r = self.request_start(f"reconcile: {c['claimable']} job(s) "
                                       f"waiting")
                actions.append(f"start:{r}")

        elif real in RUNNING_LIKE:
            silent = now - (vm["last_seen_at"] or 0)
            if (st == VM_STARTING and now - (vm["op_started_at"] or now)
                    > self.cfg.startup_timeout_s):
                self._start_failed(f"no worker reported ready within "
                                   f"{self.cfg.startup_timeout_s:.0f}s")
                if self._stop("the worker never became ready"):
                    actions.append("stopped_unready_vm")
            elif st == VM_START_FAILED:
                # a booted VM whose worker cannot work is pure cost
                if self._stop("the worker is not ready"):
                    actions.append("stopped_unready_vm")
            elif (st == VM_STOPPING and now - (vm["stop_requested_at"] or now)
                  > 300):
                if self._stop("asked to shut down, still running"):
                    actions.append("forced_stop")
            elif st == VM_READY:
                quiet = (self.cfg.idle_grace_s + 3 * self.cfg.heartbeat_s
                         + 60)
                if not c["active"] and not c["claimable"] and silent > quiet:
                    if self._stop("running with no work and no worker "
                                  "contact"):
                        actions.append("stopped_orphan")
                elif (not c["active"] and c["claimable"]
                      and silent > self.cfg.startup_timeout_s):
                    self._start_failed("the worker stopped responding")
                    if self._stop("work is waiting but the worker is "
                                  "silent"):
                        actions.append("stopped_silent_worker")
            elif st == VM_STOPPED:
                # Running, yet we never started it: somebody else did. Leave
                # it, say so. If its worker comes up it will say hello and be
                # managed from then on.
                if silent > self.cfg.heartbeat_s * 3:
                    self.queue.vm_update(last_error=(
                        "the VM is running but was not started by the "
                        "control plane; leaving it alone"))
        # GOING_DOWN: wait for it to finish

        return {"vm": real, "state": self.queue.vm()["state"],
                "actions": actions, **c}
