"""Periodic reconciliation: the RECOVERY path, never the trigger.

Jobs are triggered by the queue. An upload completes, the job is queued and a
start is requested in that same request. This loop exists for everything that
path cannot cover:

  * a worker that died mid-job          -> its lease expires, the job retries
  * a start request that was lost       -> queued work beside a stopped VM
  * a retry whose backoff has passed    -> the VM is started for it
  * a VM that booted but never worked   -> stopped, start retried with backoff
  * a VM left running with nothing to do -> stopped
  * an upload abandoned half-way        -> failed, visibly
  * a job that waited too long          -> failed, visibly

It runs as a thread inside the control plane, which is always on, so it
costs nothing. `tick()` is also exposed on an authenticated route, so an
external scheduler could drive it - but that is optional, because systemd
restarts the control plane process, and with it this thread.
"""

from __future__ import annotations

import logging
import threading
import time

from .sanitize import redact

log = logging.getLogger("alpha.cloud.reconcile")


class Reconciler:
    def __init__(self, queue, vm, cfg, clock=time.time):
        self.queue, self.vm, self.cfg, self.clock = queue, vm, cfg, clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last: dict = {}

    def tick(self) -> dict:
        """One pass. Serialised: two overlapping ticks would only race each
        other for the same rows, so the second waits."""
        with self._lock:
            report: dict = {"at": self.clock()}
            try:
                report["leases_expired"] = self.queue.expire_leases()
                report["expired"] = self.queue.expire_stale(
                    self.cfg.upload_timeout_s, self.cfg.queue_timeout_s)
                report["vm"] = self.vm.reconcile()
            except Exception as exc:                  # noqa: BLE001
                # never let one bad pass kill the loop
                report["error"] = redact(f"{type(exc).__name__}: {exc}")
                log.exception("reconcile pass failed")
            self.last = report
            moved = report.get("leases_expired") or report.get("expired")
            acts = (report.get("vm") or {}).get("actions")
            if moved or acts:
                log.info("reconcile: %s", report)
            return report

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(self.cfg.reconcile_s):
                self.tick()

        self._thread = threading.Thread(target=loop, name="cloud-reconcile",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
