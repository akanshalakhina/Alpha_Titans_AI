"""
Job-level scheduler: bounded concurrency, queuing, cancellation, timeouts.

These use deterministic doubles (sleeps and counters) rather than real
encodes, because what is under test is the scheduling contract - how many run
at once, what a cancel reaches, what a deadline does. The real-process
behaviour it sits on is covered by test_process_ownership.py.
"""

import asyncio
import os
import sys
import threading
import time

import pytest

from agents.render_pool import MediaPool, ProcessCancelled, ProcessTimeout
from agents.scheduler import RenderScheduler, WorkCancelled

FFMPEG = os.environ.get("AT_FFMPEG", "ffmpeg")


@pytest.fixture
def sched():
    s = RenderScheduler(max_workers=2)
    yield s
    s.close()


class ConcurrencyProbe:
    """Records the high-water mark of simultaneous execution."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.ran = 0

    def work(self, seconds=0.3):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(seconds)
        with self.lock:
            self.now -= 1
            self.ran += 1
        return "ok"


# ------------------------------------------------ 6. bounded concurrency

def test_concurrency_is_bounded_by_the_pool(sched):
    probe = ConcurrencyProbe()
    work = [sched.submit("job%d" % i, probe.work, label="w%d" % i)
            for i in range(6)]
    for w in work:
        w.wait(timeout=60)
    assert probe.ran == 6
    assert probe.peak <= 2, f"ran {probe.peak} at once, bound is 2"


def test_a_single_worker_serialises_everything():
    s = RenderScheduler(max_workers=1)
    try:
        probe = ConcurrencyProbe()
        work = [s.submit("j", probe.work, label=str(i)) for i in range(4)]
        for w in work:
            w.wait(timeout=60)
        assert probe.peak == 1
    finally:
        s.close()


# ------------------------------------------------- 7. multiple jobs queue

def test_several_jobs_queue_and_all_complete(sched):
    results = {}
    work = {jid: sched.submit(jid, lambda j=jid: f"done-{j}", label="r")
            for jid in ("a", "b", "c", "d")}
    for jid, w in work.items():
        results[jid] = w.wait(timeout=60)
    assert results == {"a": "done-a", "b": "done-b",
                       "c": "done-c", "d": "done-d"}
    assert sched.stats["submitted"] == 4
    assert sched.stats["running"] == 0
    assert sched.stats["pending"] == 0


def test_running_and_pending_are_observable(sched):
    gate = threading.Event()
    started = threading.Event()

    def blocked():
        started.set()
        gate.wait(30)
        return "ok"

    w1 = sched.submit("jobA", blocked, label="blocker")
    assert started.wait(20)
    assert sched.running_jobs() == {"jobA"}
    assert [r["job_id"] for r in sched.running()] == ["jobA"]
    gate.set()
    assert w1.wait(timeout=60) == "ok"
    assert sched.running() == []


def test_work_is_attributable_to_its_job(sched):
    sched.submit("jobA", lambda: "x", label="one").wait(timeout=30)
    sched.submit("jobB", lambda: "y", label="two").wait(timeout=30)
    assert [w["label"] for w in sched.for_job("jobA")] == ["one"]
    assert [w["label"] for w in sched.for_job("jobB")] == ["two"]


# ----------------------------------------------------- 5. cancellation

def test_cancel_drops_queued_work_for_that_job():
    s = RenderScheduler(max_workers=1)
    try:
        gate = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            gate.wait(30)
            return "first"

        first = s.submit("jobA", blocker, label="running")
        assert started.wait(20)
        queued = s.submit("jobA", lambda: "second", label="queued")
        assert queued.pending

        report = s.cancel("jobA")
        assert report["queued_dropped"] == 1
        assert queued.cancelled

        gate.set()
        first.wait(timeout=60)
    finally:
        s.close()


def test_cancel_marks_running_work_and_kills_its_processes(sched, tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time; print('GO', flush=True); time.sleep(60)")
    started = threading.Event()

    def run_it():
        return sched.pool.run_process(
            [sys.executable, str(script)],
            on_line=lambda l: started.set(), job_id="jobA", label="proc")

    w = sched.submit("jobA", run_it, label="render")
    assert started.wait(30)

    report = sched.cancel("jobA")
    assert report["processes_killed"] == 1

    with pytest.raises(ProcessCancelled):
        w.wait(timeout=60)
    assert w.cancelled
    assert sched.pool.active_processes() == []


def test_cancel_refuses_work_submitted_after_it(sched):
    sched.cancel("jobA")
    w = sched.submit("jobA", lambda: "should not run", label="late")
    with pytest.raises(WorkCancelled):
        w.wait(timeout=30)
    assert w.cancelled


def test_cancel_does_not_touch_other_jobs(sched):
    sched.cancel("jobA")
    assert sched.submit("jobB", lambda: "fine").wait(timeout=30) == "fine"


# ---------------------------------------------------------- 3. timeouts

def test_a_deadline_kills_the_process_not_just_the_wait(sched, tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time; print('GO', flush=True); time.sleep(60)")
    started = threading.Event()

    def run_it():
        return sched.pool.run_process(
            [sys.executable, str(script)],
            on_line=lambda l: started.set(), job_id="jobT",
            label="proc", timeout_s=2)

    w = sched.submit("jobT", run_it, label="render", timeout_s=30)
    with pytest.raises(ProcessTimeout):
        w.wait(timeout=60)
    assert sched.pool.active_processes() == []


def test_a_scheduler_deadline_kills_without_banning_the_job(sched, tmp_path):
    """The work item's own deadline, independent of the process deadline.

    It must raise a TIMEOUT rather than a cancellation, and it must not leave
    the job flagged as cancelled: blowing a deadline ends this attempt, it
    does not mean the job may never run again.
    """
    script = tmp_path / "sleep.py"
    script.write_text("import time; print('GO', flush=True); time.sleep(60)")
    started = threading.Event()

    def run_it():
        return sched.pool.run_process(
            [sys.executable, str(script)],
            on_line=lambda l: started.set(), job_id="jobS", label="proc")

    w = sched.submit("jobS", run_it, label="render", timeout_s=3)
    assert started.wait(30)
    with pytest.raises(ProcessTimeout, match="deadline"):
        w.wait(timeout=60)
    assert sched.pool.active_processes() == []
    assert not sched.pool.is_cancelled("jobS")
    # and the job can be submitted again straight away
    assert sched.submit("jobS", lambda: "ok").wait(timeout=30) == "ok"


# --------------------------------------------------- 8. error reporting

def test_a_failure_is_reported_with_its_type(sched):
    def boom():
        raise ValueError("render exploded")

    w = sched.submit("jobA", boom, label="render")
    with pytest.raises(ValueError, match="render exploded"):
        w.wait(timeout=30)
    assert "ValueError: render exploded" in w.error
    assert not w.cancelled
    assert sched.stats["failed"] == 1


def test_a_cancel_is_not_counted_as_a_failure(sched):
    sched.cancel("jobA")
    w = sched.submit("jobA", lambda: "x")
    with pytest.raises(WorkCancelled):
        w.wait(timeout=30)
    assert sched.stats["cancelled"] == 1
    assert sched.stats["failed"] == 0


# ------------------------------------------------------ async awaiting

def test_work_can_be_awaited_without_blocking_the_loop(sched):
    """The property node_render depends on: the event loop stays free."""
    ticks = {"n": 0}

    async def main():
        async def ticker():
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.02)

        t = asyncio.create_task(ticker())
        result = await sched.submit("jobA", time.sleep, 0.6, label="slow")
        t.cancel()
        return result

    asyncio.run(main())
    assert ticks["n"] > 5, (
        f"the loop only ticked {ticks['n']} times - it was blocked")


def test_run_async_returns_the_value(sched):
    async def main():
        return await sched.run_async("jobA", lambda: 42, label="calc")

    assert asyncio.run(main()) == 42


# ------------------------------------------- the scheduler shares one pool

def test_a_shared_pool_means_one_concurrency_bound():
    pool = MediaPool(max_workers=2, name="shared")
    s = RenderScheduler(pool=pool)
    try:
        assert s.pool is pool
        probe = ConcurrencyProbe()
        work = [s.submit("j%d" % i, probe.work) for i in range(5)]
        for w in work:
            w.wait(timeout=60)
        assert probe.peak <= 2
    finally:
        s.close()
        pool.close()


def test_closing_a_borrowed_pool_is_left_to_its_owner():
    pool = MediaPool(max_workers=1, name="borrowed")
    s = RenderScheduler(pool=pool)
    s.close()
    assert not pool._closed, "the scheduler closed a pool it does not own"
    pool.close()
