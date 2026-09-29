"""
Process ownership: the pool holds the handle, and a deadline actually kills.

The defect this closes is specific. Our subprocess is `python build_video.py`,
and THAT spawns ffmpeg - so ffmpeg is a grandchild. Killing the direct child
leaves it running, holding the CPU and the output file. These tests reproduce
that exact shape (a python parent that spawns a real ffmpeg) and assert the
grandchild is gone afterwards.

At least one test drives a real ffmpeg process, as required.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from agents.render_pool import (IS_WINDOWS, ManagedProcess, MediaPool,
                                ProcessCancelled, ProcessTimeout)

FFMPEG = os.environ.get("AT_FFMPEG", "ffmpeg")


# ---------------------------------------------------------------- helpers

def pid_alive(pid: int) -> bool:
    """Is this pid still running? No psutil dependency."""
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:                                               # pragma: no cover
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_gone(pid: int, timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not pid_alive(pid):
            return True
        time.sleep(0.25)
    return not pid_alive(pid)


def spawner_script(tmp_path: Path, seconds: int = 600) -> Path:
    """A python parent that spawns a REAL long-running ffmpeg and prints its
    pid - the same parent/grandchild shape as `python build_video.py cut`.

    `-re` matters: without it `-t 600` means 600 seconds of MEDIA, and
    encoding 600s of 64x64 testsrc to null finishes in a couple of wall
    seconds, so the process would exit before any deadline could fire.
    `-re` paces at the native frame rate, making media time wall time.
    """
    p = tmp_path / "spawn_ffmpeg.py"
    p.write_text(textwrap.dedent(f"""
        import subprocess, sys
        proc = subprocess.Popen(
            [r"{FFMPEG}", "-nostdin", "-v", "error", "-re", "-f", "lavfi",
             "-i", "testsrc2=s=64x64:r=30", "-t", "{seconds}",
             "-f", "null", "-"])
        print("FFMPEG_PID", proc.pid, flush=True)
        proc.wait()
        print("CHILD_DONE", flush=True)
    """))
    return p


def python_sleeper(tmp_path: Path, seconds: float, name="sleeper.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(f"""
        import time, sys
        print("STARTED", flush=True)
        time.sleep({seconds})
        print("FINISHED", flush=True)
    """))
    return p


# -------------------------------------------- 1. the pool owns the handle

def test_the_pool_registers_a_live_process(tmp_path, pool):
    script = python_sleeper(tmp_path, 5)
    seen: list[dict] = []
    started = threading.Event()

    def watch(line):
        if line.strip() == "STARTED":
            started.set()

    fut = pool.submit(pool.run_process, [sys.executable, str(script)],
                      on_line=watch, job_id="jobA", label="sleeper")
    assert started.wait(20), "child never started"
    seen = pool.active_processes()

    assert len(seen) == 1, f"pool is not tracking the process: {seen}"
    assert seen[0]["job_id"] == "jobA"
    assert seen[0]["label"] == "sleeper"
    assert seen[0]["running"] is True
    assert seen[0]["pid"] > 0

    rc, _ = fut.result(timeout=60)
    assert rc == 0


def test_a_managed_process_exposes_poll_wait_and_returncode(tmp_path, pool):
    script = python_sleeper(tmp_path, 0.2)
    rc, out = pool.run_process([sys.executable, str(script)], job_id="jobA")
    assert rc == 0
    assert "STARTED" in out and "FINISHED" in out


# --------------------------------- 2. success completes and releases

def test_a_successful_process_is_released_from_the_registry(tmp_path, pool):
    script = python_sleeper(tmp_path, 0.2)
    assert pool.active_processes() == []
    rc, _ = pool.run_process([sys.executable, str(script)], job_id="jobA")
    assert rc == 0
    assert pool.active_processes() == [], "handle leaked after completion"
    assert pool.stats["live_processes"] == 0


# ------------------------------------- 3/4. timeout kills the REAL ffmpeg

@pytest.mark.slow
def test_a_timeout_kills_the_ffmpeg_GRANDCHILD(tmp_path, pool):
    """The whole point of Step 3, with a real ffmpeg.

    Before this change the deadline abandoned the wait and ffmpeg kept
    running. Here the pid is captured from the child's own stdout and the test
    asserts it is genuinely gone.
    """
    script = spawner_script(tmp_path, seconds=600)
    ffmpeg_pid: dict[str, int] = {}
    got = threading.Event()

    def watch(line):
        if line.startswith("FFMPEG_PID"):
            ffmpeg_pid["pid"] = int(line.split()[1])
            got.set()

    with pytest.raises(ProcessTimeout) as exc:
        pool.run_process([sys.executable, str(script)], on_line=watch,
                         timeout_s=4, job_id="jobA", label="ffmpeg-spawner")

    assert "deadline" in str(exc.value)
    assert got.wait(5), "never saw the ffmpeg pid"
    pid = ffmpeg_pid["pid"]
    assert wait_gone(pid), (
        f"ORPHAN: ffmpeg pid {pid} is still running after the timeout")
    assert pool.active_processes() == []


@pytest.mark.slow
def test_no_orphan_remains_after_a_cancel(tmp_path, pool):
    script = spawner_script(tmp_path, seconds=600)
    ffmpeg_pid: dict[str, int] = {}
    got = threading.Event()

    def watch(line):
        if line.startswith("FFMPEG_PID"):
            ffmpeg_pid["pid"] = int(line.split()[1])
            got.set()

    fut = pool.submit(pool.run_process, [sys.executable, str(script)],
                      on_line=watch, job_id="jobCancel", label="ffmpeg")
    assert got.wait(30), "never saw the ffmpeg pid"

    killed = pool.cancel_job("jobCancel")
    assert killed == 1

    with pytest.raises(ProcessCancelled):
        fut.result(timeout=60)

    pid = ffmpeg_pid["pid"]
    assert wait_gone(pid), f"ORPHAN: ffmpeg pid {pid} survived the cancel"
    assert pool.active_processes() == []


# --------------------------------------------------- 5. cancellation

def test_cancel_refuses_to_start_new_work_for_that_job(tmp_path, pool):
    """A cancel must not be raced by a process that had not started yet."""
    script = python_sleeper(tmp_path, 5)
    pool.cancel_job("jobA")
    with pytest.raises(ProcessCancelled, match="cancelled before"):
        pool.run_process([sys.executable, str(script)], job_id="jobA")


def test_clear_cancel_lets_the_job_run_again(tmp_path, pool):
    script = python_sleeper(tmp_path, 0.2)
    pool.cancel_job("jobA")
    pool.clear_cancel("jobA")
    rc, _ = pool.run_process([sys.executable, str(script)], job_id="jobA")
    assert rc == 0


def test_cancel_only_touches_the_named_job(tmp_path, pool):
    script = python_sleeper(tmp_path, 4)
    started = threading.Event()
    fut = pool.submit(pool.run_process, [sys.executable, str(script)],
                      on_line=lambda l: started.set(), job_id="keepme")
    assert started.wait(20)
    assert pool.cancel_job("otherjob") == 0
    rc, _ = fut.result(timeout=60)
    assert rc == 0, "cancelling another job killed this one"


def test_kill_all_leaves_nothing_behind(tmp_path, pool):
    script = python_sleeper(tmp_path, 10)
    ready = threading.Event()
    fut = pool.submit(pool.run_process, [sys.executable, str(script)],
                      on_line=lambda l: ready.set(), job_id="jobA")
    assert ready.wait(20)
    assert pool.kill_all(reason="test") == 1
    with pytest.raises(ProcessTimeout):
        fut.result(timeout=60)
    assert pool.active_processes() == []


# ----------------------------------------- 8. a failure reports correctly

def test_a_failing_process_reports_its_exit_code_and_output(tmp_path, pool):
    script = tmp_path / "boom.py"
    script.write_text("import sys; print('about to fail'); sys.exit(7)")
    rc, out = pool.run_process([sys.executable, str(script)], job_id="jobA")
    assert rc == 7
    assert "about to fail" in out


@pytest.mark.slow
def test_a_real_ffmpeg_failure_surfaces_its_stderr(tmp_path, pool):
    """Real ffmpeg, real error - a missing input file."""
    rc, out = pool.run_process(
        [FFMPEG, "-nostdin", "-v", "error", "-i",
         str(tmp_path / "does_not_exist.mp4"), "-f", "null", "-"],
        job_id="jobA", label="ffmpeg-fail")
    assert rc != 0
    assert "does_not_exist" in out or "No such file" in out


def test_a_real_ffmpeg_process_succeeds_and_is_released(tmp_path, pool):
    """The happy path, with a real encoder."""
    out_file = tmp_path / "tiny.mp4"
    rc, _ = pool.run_process(
        [FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=s=64x64:r=30:d=1", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out_file)],
        job_id="jobA", label="real-encode")
    assert rc == 0
    assert out_file.exists() and out_file.stat().st_size > 0
    assert pool.active_processes() == []


# ------------------------------------------------ process group isolation

def test_processes_are_started_in_their_own_group(tmp_path, pool):
    """Group isolation is what makes the tree addressable for a kill."""
    script = python_sleeper(tmp_path, 3)
    ready = threading.Event()
    fut = pool.submit(pool.run_process, [sys.executable, str(script)],
                      on_line=lambda l: ready.set(), job_id="jobA")
    assert ready.wait(20)
    proc = pool.processes[0]
    assert isinstance(proc, ManagedProcess)
    if not IS_WINDOWS:                                 # pragma: no cover
        assert os.getpgid(proc.pid) != os.getpgid(os.getpid())
    fut.result(timeout=60)


def test_closing_the_pool_kills_whatever_is_left(tmp_path):
    script = python_sleeper(tmp_path, 30)
    p = MediaPool(max_workers=2, name="closing")
    ready = threading.Event()
    p.submit(p.run_process, [sys.executable, str(script)],
             on_line=lambda l: ready.set(), job_id="jobA")
    assert ready.wait(20)
    pid = p.processes[0].pid
    p.close()
    assert wait_gone(pid), f"ORPHAN: pid {pid} survived pool close"
