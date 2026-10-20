"""The worker VM's lifecycle and the reconciler.

Brief, section 13: tests 3, 5, 9, 10 and 11 live here (the worker-side
halves of 9 and 10 are in test_worker.py). Compute Engine is a fake; no test
starts or stops a real machine.
"""

from __future__ import annotations

import threading

import pytest

from cloud import queue as Q
from cloud.reconcile import Reconciler

from test_queue import queued


@pytest.fixture
def rec(queue, vm, cfg, clock):
    return Reconciler(queue, vm, cfg, clock=clock)


# ---------------------------------------------- 3. one start, however asked

def test_repeated_start_requests_start_the_vm_once(vm, compute, queue):
    queued(queue)
    results = [vm.request_start("upload") for _ in range(10)]
    assert compute.starts() == 1
    assert results[0] == "started"
    assert set(results[1:]) == {"already_active"}


def test_concurrent_start_requests_start_the_vm_once(vm, compute, queue):
    """Ten uploads completing in the same instant, on ten threads."""
    queued(queue)
    compute.start_delay = 0.05           # hold the API call open
    barrier = threading.Barrier(10)
    results = []

    def go():
        barrier.wait()
        results.append(vm.request_start("upload"))

    threads = [threading.Thread(target=go) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert compute.starts() == 1, results
    assert results.count("started") == 1


def test_starting_marks_the_waiting_jobs(vm, queue):
    jid = queued(queue)
    vm.request_start("upload")
    assert queue.get(jid)["state"] == Q.STARTING_VM


def test_starts_are_spaced_and_capped(vm, compute, queue, clock, cfg):
    queued(queue)
    for _ in range(cfg.max_starts_per_hour):
        vm.request_start("x")
        queue.vm_update(state=Q.VM_STOPPED)          # pretend it stopped
        assert vm.request_start("too soon") == "throttled"
        clock.advance(cfg.min_start_interval_s + 1)
    assert vm.request_start("one too many") == "circuit_open"
    assert compute.starts() == cfg.max_starts_per_hour
    assert "circuit open" in queue.vm()["last_error"]


def test_nothing_starts_without_compute_configured(queue, cfg, clock):
    from cloud.vm import VmManager
    assert VmManager(queue, None, cfg, clock).request_start("x") == \
        "not_configured"


# ------------------------------------------------- 5. a failed start retries

def test_a_failed_start_puts_jobs_back_and_records_why(vm, compute, queue):
    jid = queued(queue)
    compute.fail_start = 1
    assert vm.request_start("upload") == "failed"
    row = queue.vm()
    assert row["state"] == Q.VM_START_FAILED and row["start_failures"] == 1
    assert "ya29" not in row["last_error"], "redacted before storage"
    assert queue.get(jid)["state"] == Q.QUEUED


def test_the_reconciler_retries_a_failed_start_with_backoff(vm, compute,
                                                            queue, rec,
                                                            clock):
    queued(queue)
    compute.fail_start = 1
    vm.request_start("upload")
    rec.tick()
    assert compute.starts() == 1, "backing off, not hammering"
    clock.advance(vm.start_backoff(1) + 1)
    rec.tick()
    assert compute.starts() == 2
    assert queue.vm()["state"] == Q.VM_STARTING


def test_a_vm_that_boots_but_never_reports_ready_is_stopped(vm, compute,
                                                            queue, rec,
                                                            clock, cfg):
    jid = queued(queue)
    vm.request_start("upload")                 # VM now "RUNNING", no hello
    clock.advance(cfg.startup_timeout_s + 1)
    report = rec.tick()
    assert "stopped_unready_vm" in report["vm"]["actions"]
    assert compute.calls[-1] == "stop"
    assert queue.vm()["state"] == Q.VM_STOPPING
    assert queue.get(jid)["state"] == Q.QUEUED


def test_a_worker_that_says_it_is_not_ready_gets_its_vm_stopped(vm,
                                                                compute,
                                                                queue, rec):
    queued(queue)
    vm.request_start("upload")
    vm.hello("w1", False, ["ffmpeg is not runnable"])
    rec.tick()
    assert compute.calls[-1] == "stop"
    assert "ffmpeg" in queue.vm()["last_error"]


def test_a_start_that_never_took_is_noticed(vm, compute, queue, rec, clock):
    """The control plane committed `starting`, then died before the
    Compute call. The reconciler sees a VM that is down and starts it."""
    queued(queue)
    queue.vm_update(state=Q.VM_STARTING, op_started_at=clock(),
                    last_start_at=clock() - 10_000)
    clock.advance(120)
    rec.tick()                         # notices: start did not take
    clock.advance(vm.start_backoff(1) + 1)
    rec.tick()
    assert compute.starts() == 1


# ------------------------------------------ 11. a lost start is recovered

def test_a_queued_job_whose_start_request_was_lost_is_found(queue, rec,
                                                           compute):
    """The upload completed, the job was queued, and the immediate start
    request never happened. Reconciliation alone gets it a worker."""
    jid = queued(queue)
    assert compute.starts() == 0
    report = rec.tick()
    assert compute.starts() == 1
    assert "start:started" in report["vm"]["actions"]
    assert queue.get(jid)["state"] == Q.STARTING_VM


def test_a_retry_whose_backoff_passed_gets_a_worker(queue, rec, compute,
                                                    clock):
    jid = queued(queue)
    c = queue.claim("w")
    queue.fail(jid, "w", c.claim_token, "transient", "x")
    rec.tick()
    assert compute.starts() == 0, "still inside its backoff"
    clock.advance(queue.get(jid)["not_before"] - clock() + 1)
    rec.tick()
    assert compute.starts() == 1


def test_nothing_waiting_means_nothing_started(rec, compute):
    rec.tick()
    assert compute.calls == []


# ---------------------------------------------- 9 and 10, control-plane side

def _ready(vm, queue, clock):
    vm.request_start("x")
    vm.hello("w1", True, [])


def test_idle_says_claim_when_work_exists(vm, queue, clock):
    _ready(vm, queue, clock)
    queued(queue, "key-later-01")
    assert vm.idle("w1") == "claim"


def test_idle_never_says_shutdown_while_a_job_is_active(vm, queue, clock,
                                                        cfg):
    _ready(vm, queue, clock)
    jid = queued(queue)
    queue.claim("w1")
    clock.advance(cfg.idle_grace_s * 10)
    assert vm.idle("w1") == "wait"
    assert queue.vm()["state"] == Q.VM_READY


def test_idle_waits_out_the_grace_then_says_shutdown(vm, queue, clock, cfg):
    _ready(vm, queue, clock)
    clock.advance(cfg.idle_grace_s - 5)
    assert vm.idle("w1") == "wait"
    clock.advance(10)
    assert vm.idle("w1") == "shutdown"
    assert queue.vm()["state"] == Q.VM_STOPPING


def test_a_job_queued_after_shutdown_is_not_lost(vm, queue, compute, rec,
                                                 clock, cfg):
    """The race the single transaction exists for: idle said shutdown,
    then a job arrived. It must get a worker once the VM is down."""
    _ready(vm, queue, clock)
    clock.advance(cfg.idle_grace_s + 1)
    assert vm.idle("w1") == "shutdown"
    jid = queued(queue, "key-late-0001")
    assert vm.request_start("upload") == "stopping"
    compute.set("TERMINATED")                      # the worker powered off
    clock.advance(cfg.min_start_interval_s + 1)
    rec.tick()
    assert compute.starts() == 2
    assert queue.get(jid)["state"] == Q.STARTING_VM


def test_a_long_activation_is_wound_down(vm, queue, clock, cfg):
    _ready(vm, queue, clock)
    queued(queue)
    clock.advance(cfg.max_activation_s + 1)
    assert vm.may_claim() is False
    assert vm.idle("w1") == "shutdown"


# --------------------------------------------------- orphans and strays

def test_a_running_vm_with_no_work_and_no_worker_is_stopped(vm, queue,
                                                            compute, rec,
                                                            clock, cfg):
    _ready(vm, queue, clock)
    clock.advance(cfg.idle_grace_s + 3 * cfg.heartbeat_s + 61)
    report = rec.tick()
    assert "stopped_orphan" in report["vm"]["actions"]


def test_a_vm_still_running_after_shutdown_was_agreed_is_stopped(
        vm, queue, compute, rec, clock, cfg):
    _ready(vm, queue, clock)
    clock.advance(cfg.idle_grace_s + 1)
    vm.idle("w1")
    clock.advance(301)
    report = rec.tick()
    assert "forced_stop" in report["vm"]["actions"]


def test_a_vm_started_by_someone_else_is_left_alone(queue, compute, rec,
                                                    clock):
    compute.set("RUNNING")
    clock.advance(1000)
    rec.tick()
    assert "stop" not in compute.calls
    assert "not started by the control plane" in queue.vm()["last_error"]


def test_the_vm_row_follows_the_real_vm_down(vm, queue, compute, rec,
                                             clock, cfg):
    _ready(vm, queue, clock)
    clock.advance(cfg.idle_grace_s + 1)
    vm.idle("w1")
    compute.set("TERMINATED")
    rec.tick()
    assert queue.vm()["state"] == Q.VM_STOPPED


def test_an_unreachable_compute_api_breaks_nothing(queue, rec, compute):
    queued(queue)
    compute.fail_status = True
    report = rec.tick()
    assert report["vm"]["vm"] == "unknown"
    assert "error" not in report


def test_reconcile_also_expires_leases(queue, rec, clock):
    jid = queued(queue)
    queue.claim("w")
    clock.advance(queue.lease_s + 1)
    assert rec.tick()["leases_expired"] == [jid]
