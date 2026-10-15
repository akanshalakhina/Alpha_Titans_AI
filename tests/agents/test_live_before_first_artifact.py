"""A job that has started tracing is already an agent job.

THE BUG THIS PINS
-----------------
`is_agent_job` asked the artifact store "do you have rows for this job?" -
and an artifact only exists once the FIRST agent has FINISHED. Measuring an
eight-minute clip takes minutes, so for all of that time `/status` and
`/trace` answered 404 and the activity view showed every node greyed out as
"queued" with

    0 agents ran  .  0ms  .  0 tool calls

while `clip_analysis` was demonstrably running, its probe tool calls already
written to the trace. The person saw a dead screen and assumed nothing was
happening.

The trace is written when a node STARTS, so it is the earlier and more
truthful signal of "this is an agent build".
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import compat                                 # noqa: E402
from agents.store import ArtifactStore                    # noqa: E402
from agents.trace import TraceRecorder                    # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    ArtifactStore(str(path))                 # create the schema
    TraceRecorder(str(path))
    monkeypatch.setattr(compat, "resolve_job_db",
                        lambda jid, root=None: (str(path), "test"))
    return str(path)


def test_a_job_with_nothing_at_all_is_not_an_agent_job(db):
    assert compat.is_agent_job("never-started") is False


def test_a_running_first_node_already_counts(db):
    """The 404 window this closes: tracing has begun, nothing has finished."""
    rec = TraceRecorder(db)
    with rec.node("job1", "clip_analysis", "clip_analysis") as span:
        span.tool("pipeline.probe", stage="probe")
        # still inside the node - nothing has been written to the store yet
        assert ArtifactStore(db).has_job("job1") is False
        assert compat.is_agent_job("job1") is True


def test_it_stays_true_once_the_node_finishes(db):
    rec = TraceRecorder(db)
    with rec.node("job2", "clip_analysis", "clip_analysis"):
        pass
    assert compat.is_agent_job("job2") is True


def test_one_job_tracing_does_not_make_another_one_an_agent_job(db):
    rec = TraceRecorder(db)
    with rec.node("mine", "clip_analysis", "clip_analysis"):
        pass
    assert compat.is_agent_job("mine") is True
    assert compat.is_agent_job("someone-else") is False


def test_an_artifact_alone_still_counts(db):
    """The original signal must keep working on its own."""
    rec_free = pathlib.Path(db)
    assert rec_free.exists()
    # no trace rows for this id at all, only the artifact path matters here
    assert compat.is_agent_job("no-trace-no-artifact") is False


def test_a_missing_database_is_not_an_agent_job(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "resolve_job_db",
                        lambda jid, root=None: (str(tmp_path / "gone.db"),
                                                "test"))
    assert compat.is_agent_job("anything") is False
