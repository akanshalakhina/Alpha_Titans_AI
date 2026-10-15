"""A build the server was killed during must be continuable, and must
say that it can be.

WHAT THIS GUARDS
----------------
A production runs in a thread of the server process. Close the server
mid-build - a crash, a restart, a Ctrl+C - and the thread dies while the
artifacts, the trace and the LangGraph checkpoint all survive on disk.

Nothing noticed. `/status` answered with no verdict and nothing
producing, which the UI drew as a run still in progress, so the person
watched a spinner for a build that had stopped minutes earlier. It
happened three times in one afternoon.

`runner.resume_job` could always finish these - `test_resume.py` proves
it across a real process kill - but only from a terminal. These tests
pin the two halves that were missing: noticing, and offering.

Re-running from the start was never the answer. The reasoning nodes
would spend a second set of tokens against a daily quota and could
reach different answers, leaving the second half of the video arguing
with the first.
"""

from __future__ import annotations

import pathlib
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import api as A                                  # noqa: E402

JOB = "halfbuilt01"


class _Rows:
    """Just enough TraceRecorder/ArtifactStore for the question asked."""

    def __init__(self, rows, critic=None):
        self._rows, self._critic = rows, critic

    def for_job(self, _job):
        return self._rows

    def get(self, _job, kind, _model=None):
        return self._critic if kind == "CriticResult" else None

    def next_revision(self, _job, _kind):
        return 0

    def totals(self, _job):
        return {}


@pytest.fixture
def client(tmp_path, monkeypatch):
    build = tmp_path / "outputs" / JOB
    build.mkdir(parents=True)

    from agents import runner
    monkeypatch.setattr(runner, "job_paths",
                        lambda j, *a, **k: (tmp_path / "uploads" / j, build))
    monkeypatch.setattr(A, "_require_agent_job", lambda j: None)
    monkeypatch.setattr(A, "resolve_job_db", lambda j, *a, **k: (":memory:", "x"))

    state = {"rows": [], "critic": None, "resumed": []}
    monkeypatch.setattr(A, "TraceRecorder", lambda db: _Rows(state["rows"]))
    monkeypatch.setattr(A, "ArtifactStore",
                        lambda db: _Rows(state["rows"], state["critic"]))
    monkeypatch.setattr(runner, "resume_job",
                        lambda job_id, *a, **k: state["resumed"].append(job_id))

    app = FastAPI()
    app.include_router(A.router)
    c = TestClient(app)
    c.build, c.state = build, state
    A._RUNNING.pop(JOB, None)
    A._FAILED.pop(JOB, None)
    return c


def _checkpoint(client):
    from agents import runner
    (client.build / runner.CHECKPOINT_DB).write_bytes(b"sqlite-ish")


def _row(node, status="ok"):
    return {"node": node, "status": status, "seq": 1}


# ============================================== noticing it was left open

def test_a_half_built_job_is_reported_as_interrupted(client):
    _checkpoint(client)
    client.state["rows"] = [_row("clip_analysis"), _row("story"),
                            _row("render", "running")]
    got = A._interrupted(JOB)
    assert got and got["resumable"] is True
    assert got["nodes_done"] == 2 and got["stopped_at"] == "render"
    assert "checkpoint survived" in got["why"]


def test_the_last_finished_node_is_named_when_none_was_running(client):
    _checkpoint(client)
    client.state["rows"] = [_row("clip_analysis"), _row("vision")]
    assert A._interrupted(JOB)["stopped_at"] == "vision"


def test_status_carries_it(client):
    _checkpoint(client)
    client.state["rows"] = [_row("story")]
    body = client.get(f"/api/v2/jobs/{JOB}/status").json()
    assert body["interrupted"]["resumable"] is True
    assert body["producing"] is False


# ------------------------------------------------- and when it is not one

def test_a_finished_job_is_not_interrupted(client):
    _checkpoint(client)
    client.state["rows"] = [_row("critic")]
    client.state["critic"] = object()              # a verdict exists
    assert A._interrupted(JOB) is None


def test_a_job_with_no_checkpoint_is_not_interrupted(client):
    client.state["rows"] = [_row("story")]
    assert A._interrupted(JOB) is None, "it never ran under the graph"


def test_a_job_that_never_ran_a_node_is_not_interrupted(client):
    _checkpoint(client)
    assert A._interrupted(JOB) is None


def test_a_running_job_is_not_interrupted(client, monkeypatch):
    _checkpoint(client)
    client.state["rows"] = [_row("story")]
    monkeypatch.setattr(A, "_is_running", lambda j: True)
    assert A._interrupted(JOB) is None, "it is building right now"


def test_a_job_that_failed_loudly_is_not_called_interrupted(client):
    """A crash with a reason is a different thing from a lost thread,
    and the person needs to be told which."""
    _checkpoint(client)
    client.state["rows"] = [_row("story")]
    A._FAILED[JOB] = {"error": "ffprobe not found", "at": time.time()}
    try:
        assert A._interrupted(JOB) is None
    finally:
        A._FAILED.pop(JOB, None)


# ===================================================== offering to finish

def test_resuming_continues_from_the_checkpoint(client):
    _checkpoint(client)
    client.state["rows"] = [_row("clip_analysis"), _row("story")]

    r = client.post(f"/api/v2/jobs/{JOB}/resume")
    assert r.status_code == 202
    assert r.json()["resumed_from"]["nodes_done"] == 2

    for _ in range(50):
        if client.state["resumed"]:
            break
        time.sleep(0.02)
    assert client.state["resumed"] == [JOB], "runner.resume_job ran"


def test_there_is_nothing_to_resume_on_a_finished_job(client):
    _checkpoint(client)
    client.state["rows"] = [_row("critic")]
    client.state["critic"] = object()
    r = client.post(f"/api/v2/jobs/{JOB}/resume")
    assert r.status_code == 409 and "nothing to resume" in r.json()["detail"]


def test_a_job_with_no_checkpoint_cannot_be_resumed(client):
    client.state["rows"] = [_row("story")]
    assert client.post(f"/api/v2/jobs/{JOB}/resume").status_code == 409


def test_a_running_job_is_not_resumed_on_top_of_itself(client, monkeypatch):
    _checkpoint(client)
    client.state["rows"] = [_row("story")]
    monkeypatch.setattr(A, "_is_running", lambda j: True)
    r = client.post(f"/api/v2/jobs/{JOB}/resume")
    assert r.status_code == 409 and "already being produced" in \
        r.json()["detail"]


def test_a_failure_during_resume_is_recorded_not_swallowed(client,
                                                           monkeypatch):
    _checkpoint(client)
    client.state["rows"] = [_row("story")]
    from agents import runner

    def _boom(job_id, *a, **k):
        raise RuntimeError("the checkpoint is corrupt")
    monkeypatch.setattr(runner, "resume_job", _boom)

    assert client.post(f"/api/v2/jobs/{JOB}/resume").status_code == 202
    for _ in range(50):
        if A._FAILED.get(JOB):
            break
        time.sleep(0.02)
    try:
        assert "the checkpoint is corrupt" in A._FAILED[JOB]["error"]
    finally:
        A._FAILED.pop(JOB, None)
