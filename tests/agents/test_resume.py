"""
Checkpoint and resume - with a REAL process kill.

This does not simulate an interruption. A separate Python process is started,
allowed to get past the analysis and planning nodes, and then killed with its
whole process tree. A different process - the test process - then resumes the
same job from the SQLite checkpoint and finishes it.

The assertion that matters is not merely "it finished": it is that the nodes
which had already completed were NOT run again. Today, a server restart during
a build marks the job as an error and loses everything (server/jobs.py:229).
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agents.artifacts import CriticResult, EditPlan, JobSettings, RenderResult
from agents.runner import peek_state, resume_job
from agents.store import ArtifactStore
from agents.trace import TraceRecorder

ROOT = Path(__file__).resolve().parents[2]
CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"
JOB_ID = "resumejob"


def _spawn(src: Path, root: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "server") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # EMPTY, not popped. Popping does not work: `config._load_dotenv` runs
    # `os.environ.setdefault` over the real .env, so a key removed here is
    # put straight back in the child and the job makes live model calls.
    # This test is about the checkpoint surviving a kill, not about
    # reasoning, and a live run is at the mercy of the provider's rate
    # limit - once a refused call started waiting as long as Groq asked
    # for, the child could no longer reach `editing_director` inside the
    # 180s below and the test failed for a reason it does not test.
    # `setdefault` leaves an existing empty value alone, and `_env` reads
    # "" as absent, so the child runs its deterministic fallbacks offline.
    env["AT_LLM_API_KEY"] = ""
    env["AT_VISION_API_KEY"] = ""
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        [sys.executable, "-m", "agents.runner", "run",
         "--src", str(src), "--job-id", JOB_ID, "--root", str(root),
         "--orientation", "horizontal", "--resolution", "720p",
         "--quality", "draft", "--target", "10"],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, creationflags=flags)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child AND any ffmpeg it started, like a real crash would."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:                                              # pragma: no cover
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:                  # pragma: no cover
        pass


def _shared_db(root: Path) -> Path:
    """The shared store the spawned child will write to."""
    from agents.store import shared_db_path
    return shared_db_path(outputs=root / "outputs")


def _completed_nodes(db: Path, job_id: str) -> list[str]:
    if not db.exists():
        return []
    try:
        return [r["node"] for r in TraceRecorder(db).for_job(job_id)
                if r["status"] == "ok"]
    except Exception:
        return []


@pytest.mark.slow
def test_a_killed_job_resumes_from_its_checkpoint(agent_root, calm_clip,
                                                  lively_clip):
    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)

    build = agent_root / "outputs" / JOB_ID
    db = _shared_db(agent_root)

    # ---- 1. start a real job in its own process ----------------------
    proc = _spawn(src, agent_root)

    # ---- 2. let it get past analysis and planning, then kill it -------
    deadline = time.time() + 180
    killed_after = None
    while time.time() < deadline:
        done = _completed_nodes(db, JOB_ID)
        if "editing_director" in done:
            killed_after = list(done)
            break
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"child exited before the kill point:\n{out[-3000:]}")
        time.sleep(0.25)

    assert killed_after is not None, "job never reached the editing_director node"
    _kill_tree(proc)
    assert proc.poll() is not None, "the child process survived the kill"

    assert "clip_analysis" in killed_after
    assert "editing_director" in killed_after
    # the render had not finished, so the job is genuinely incomplete
    assert "critic" not in killed_after

    # ---- 3. the checkpoint survived the kill --------------------------
    assert (build / "checkpoints.db").exists(), "no checkpoint was written"
    snap = peek_state(JOB_ID, agent_root)
    assert snap["values"].get("job_id") == JOB_ID
    assert snap["values"].get("clip_analysis_id"), "analysis id not checkpointed"
    assert snap["values"].get("edit_plan_id"), "edit plan id not checkpointed"
    assert snap["next"], "the checkpoint does not know what to do next"

    before = TraceRecorder(db).for_job(JOB_ID)
    analysis_runs_before = sum(1 for r in before
                               if r["node"] == "clip_analysis" and r["status"] == "ok")
    assert analysis_runs_before == 1

    # ---- 4. resume, in THIS process (a different one) -----------------
    final = resume_job(JOB_ID, root=agent_root,
                       settings=JobSettings(orientation="horizontal",
                                            resolution="720p",
                                            quality="draft",
                                            target_seconds=10.0))

    assert final["status"] in ("done", "escalated"), final
    assert final["history"], "resumed run produced no attempt"

    # ---- 5. the completed work was NOT redone -------------------------
    after = TraceRecorder(db).for_job(JOB_ID)
    analysis_runs_after = sum(1 for r in after
                              if r["node"] == "clip_analysis" and r["status"] == "ok")
    assert analysis_runs_after == 1, (
        "clip_analysis ran again after the resume - the checkpoint was not "
        "honoured")

    # ---- 6. it really produced a video --------------------------------
    video = Path(final["history"][-1]["video"])
    assert video.exists() and video.stat().st_size > 1000
    assert video.read_bytes()[4:8] == b"ftyp"

    # ---- 7. the artifacts from before the kill are the same ones ------
    store = ArtifactStore(db)
    plan = store.get(JOB_ID, "EditPlan", EditPlan)
    assert plan.artifact_id == snap["values"]["edit_plan_id"], (
        "the resumed run re-planned instead of reusing the checkpointed plan")
    assert store.get(JOB_ID, "RenderResult", RenderResult) is not None
    assert store.get(JOB_ID, "CriticResult", CriticResult) is not None


@pytest.mark.slow
def test_resume_without_a_checkpoint_is_refused(agent_root):
    with pytest.raises(FileNotFoundError):
        resume_job("nosuchjob", root=agent_root)
