"""Artifact store: immutability, revisions, input-hash caching. Trace store."""

import pytest

from agents.artifacts import Artifact, Cost, Decision, EditPlan, Segment
from agents.store import ArtifactStore
from agents.trace import TraceRecorder


def seg(index=1, clip_id="a.mp4", start=0.0, end=4.0):
    return Segment(index=index, clip_id=clip_id, start=start, end=end,
                   duration=round(end - start, 2), orientation="horizontal")


def plan(total=4.0):
    return EditPlan(segments=[seg()], total_duration=total,
                    target_seconds=20.0, canvas=(1920, 1080))


def art(job="j1", revision=0, inputs_hash="h1", payload=None, **kw):
    return Artifact[EditPlan](
        job_id=job, kind="EditPlan", revision=revision,
        produced_by="editing_director", inputs_hash=inputs_hash,
        payload=payload or plan(), **kw)


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "state.db")


# ----------------------------------------------------------- persistence

def test_put_then_get_roundtrips_payload_and_envelope(store):
    store.put(art(decisions=[Decision(what="w", chose="c", why="y",
                                      decided_by="llm", confidence=0.8)],
                  confidence=0.77, degraded=True, fallback_used="fb",
                  cost=Cost(tokens_in=10, tokens_out=5, usd=0.01)))
    got = store.get("j1", "EditPlan", EditPlan)
    assert got is not None
    assert got.payload.segments[0].clip_id == "a.mp4"
    assert got.confidence == 0.77
    assert got.degraded is True
    assert got.fallback_used == "fb"
    assert got.cost.tokens_in == 10
    assert got.decisions[0].decided_by == "llm"


def test_artifacts_are_immutable_in_the_store(store):
    store.put(art(revision=0))
    with pytest.raises(ValueError) as e:
        store.put(art(revision=0, payload=plan(total=99.0)))
    assert "immutable" in str(e.value)


def test_revisions_accumulate_and_get_returns_the_latest(store):
    store.put(art(revision=0, payload=plan(total=4.0)))
    store.put(art(revision=1, payload=plan(total=8.0)))
    assert store.get("j1", "EditPlan", EditPlan).payload.total_duration == 8.0
    assert store.get("j1", "EditPlan", EditPlan,
                     revision=0).payload.total_duration == 4.0
    assert len(store.all_revisions("j1", "EditPlan", EditPlan)) == 2


def test_next_revision_counts_up_from_nothing(store):
    assert store.next_revision("j1", "EditPlan") == 0
    store.put(art(revision=0))
    assert store.next_revision("j1", "EditPlan") == 1


def test_get_by_inputs_is_the_cache_lookup(store):
    store.put(art(revision=0, inputs_hash="abc"))
    assert store.get_by_inputs("j1", "EditPlan", "abc", EditPlan) is not None
    assert store.get_by_inputs("j1", "EditPlan", "zzz", EditPlan) is None


def test_jobs_are_isolated_from_each_other(store):
    store.put(art(job="j1", revision=0))
    assert store.get("j2", "EditPlan", EditPlan) is None


def test_manifest_lists_every_artifact(store):
    store.put(art(revision=0))
    store.put(art(revision=1))
    rows = store.manifest("j1")
    assert len(rows) == 2
    assert {r["revision"] for r in rows} == {0, 1}


# ---------------------------------------------------------------- traces

def test_trace_records_a_successful_node(tmp_path):
    t = TraceRecorder(tmp_path / "state.db")
    with t.node("j1", "render", "rendering", attempt=1,
                input_artifacts=["art_1"]) as span:
        span.tool("pipeline.cut", stage="cut")
        span.cache(2)
        span.output("art_2")
    rows = t.for_job("j1")
    assert len(rows) == 1
    r = rows[0]
    assert r["status"] == "ok"
    assert r["node"] == "render" and r["agent"] == "rendering"
    assert r["input_artifacts"] == ["art_1"]
    assert r["output_artifacts"] == ["art_2"]
    assert r["tool_calls"][0]["tool"] == "pipeline.cut"
    assert r["cache_hits"] == 2
    assert r["ended_at"] is not None


def test_trace_records_a_failure_and_reraises(tmp_path):
    t = TraceRecorder(tmp_path / "state.db")
    with pytest.raises(RuntimeError):
        with t.node("j1", "render", "rendering") as span:
            span.tool("pipeline.cut")
            raise RuntimeError("ffmpeg exploded")
    r = t.for_job("j1")[0]
    assert r["status"] == "error"
    assert "ffmpeg exploded" in r["error"]
    assert r["ended_at"] is not None


def test_trace_records_llm_cost_and_fallback(tmp_path):
    t = TraceRecorder(tmp_path / "state.db")
    with t.node("j1", "editing_director", "editing_director") as span:
        span.llm(1200, 300, 0.008)
        span.fallback("deterministic_planner (no LLM configured)")
        span.retry()
    r = t.for_job("j1")[0]
    assert r["tokens_in"] == 1200 and r["tokens_out"] == 300
    assert r["usd"] == 0.008
    assert r["fallback_used"].startswith("deterministic_planner")
    assert r["retry_count"] == 1


def test_trace_sequence_is_stable_and_totals_add_up(tmp_path):
    t = TraceRecorder(tmp_path / "state.db")
    for node in ("clip_analysis", "editing_director", "render"):
        with t.node("j1", node, node) as span:
            span.tool("x")
    rows = t.for_job("j1")
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert [r["node"] for r in rows] == ["clip_analysis", "editing_director",
                                         "render"]
    totals = t.totals("j1")
    assert totals["nodes"] == 3 and totals["tool_calls"] == 3


def test_trace_table_renders(tmp_path):
    t = TraceRecorder(tmp_path / "state.db")
    with t.node("j1", "critic", "critic") as span:
        span.tool("verify.frame_count")
    table = t.render_table("j1")
    assert "critic" in table and "seq" in table
