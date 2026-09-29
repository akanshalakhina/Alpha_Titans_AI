"""
The orchestrator end to end, on real clips through the real pipeline.

The LLM is injected; everything else - probe, measure, plan, cut, join, mux,
and the critic's inspection of the finished file - is real ffmpeg work.
"""

import shutil

import pytest

from agents.artifacts import (ClipAnalysisSet, CriticResult, EditPlan,
                              JobSettings, RenderResult)
from agents.graph import route_after_critic
from agents.render_pool import MediaPool
from agents.runner import make_job, run_job
from agents.store import ArtifactStore
from conftest import llm_returning_plan

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


def _job(root, calm_clip, lively_clip, job_id, **settings):
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    return make_job(src=src, job_id=job_id, root=root,
                    settings=JobSettings(orientation="horizontal",
                                         resolution="720p", quality="draft",
                                         target_seconds=10.0, **settings),
                    max_revisions=settings.pop("max_revisions", 2))


def _store(job):
    """The shared store that sits beside this job's build folder."""
    from agents.store import store_for_build_dir
    return store_for_build_dir(job.build_dir)


# ------------------------------------------------------------- routing

def test_routing_sends_a_pass_to_finish():
    assert route_after_critic({"verdict": "PASS"}) == "finish"


def test_routing_sends_a_revise_with_a_request_back_to_the_director():
    assert route_after_critic({"verdict": "REVISE",
                               "revision_request": {"x": 1}}) == "revise"


def test_routing_escalates_a_revise_with_nothing_to_act_on():
    assert route_after_critic({"verdict": "REVISE",
                               "revision_request": None}) == "escalate"


def test_routing_escalates_an_escalate():
    assert route_after_critic({"verdict": "ESCALATE"}) == "escalate"


def test_the_graph_has_the_cycle():
    """critic -> editing_director must be a real edge, or there is no loop."""
    import tempfile
    from pathlib import Path

    from agents.graph import build_graph
    graph, conn = build_graph(Path(tempfile.mkdtemp()) / "cp.db")
    try:
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("critic", "editing_director") in edges
        # Phase 3 put Vision between analysis and the director: the model may
        # only pick from windows the measurement pass produced, so there is
        # nothing to choose from until clip_analysis has run.
        assert ("clip_analysis", "vision") in edges
        assert ("clip_analysis", "research") in edges
        assert ("clip_analysis", "editing_director") not in edges
        # ContentStrategy is the JOIN. Vision and Research now meet at the
        # strategy, not at the director - "what is there" and "what the world
        # says" are answered before "what are we saying", which is answered
        # before "how do we cut it".
        assert ("vision", "content_strategy") in edges
        assert ("research", "content_strategy") in edges
        assert ("content_strategy", "story") in edges
        assert ("story", "voiceover") in edges
        assert ("voiceover", "music") in edges
        assert ("music", "animation") in edges
        assert ("animation", "editing_director") in edges
        assert ("music", "editing_director") not in edges
        assert ("voiceover", "editing_director") not in edges
        assert ("story", "editing_director") not in edges
        assert ("content_strategy", "editing_director") not in edges
        # the edges those two replaced must be GONE, or the director would be
        # briefed and un-briefed on the same run
        assert ("vision", "editing_director") not in edges
        assert ("research", "editing_director") not in edges
        assert ("editing_director", "render") in edges
        assert ("render", "critic") in edges
    finally:
        conn.close()


# -------------------------------------------------------------- the runs

@pytest.mark.slow
def test_a_strong_hook_passes_on_the_first_attempt(agent_root, calm_clip,
                                                   lively_clip, pool):
    job = _job(agent_root, calm_clip, lively_clip, "gpass")
    llm = llm_returning_plan(
        [{"clip_id": LIVELY, "target_seconds": 5.0, "reason": "open strong"},
         {"clip_id": CALM, "target_seconds": 5.0, "reason": "settle"}])

    final = run_job(job, llm=llm, pool=pool)

    assert final["status"] == "done"
    assert final["verdict"] == "PASS"
    assert final["attempt"] == 1
    assert final["revision_count"] == 0
    assert len(final["history"]) == 1

    from pathlib import Path
    video = Path(final["history"][0]["video"])
    assert video.exists() and video.stat().st_size > 1000
    assert video.read_bytes()[4:8] == b"ftyp", "not a real mp4"


@pytest.mark.slow
def test_a_weak_hook_triggers_a_real_revision_that_fixes_it(
        agent_root, calm_clip, lively_clip, pool):
    """The full loop: REVISE -> new EditPlan -> re-render -> PASS.

    The model is held constant - it proposes the calm clip first on BOTH
    passes. What changes the outcome is the critic's constraint being enforced
    deterministically, which is exactly the property under test.
    """
    job = _job(agent_root, calm_clip, lively_clip, "grevise")
    llm = llm_returning_plan(
        [{"clip_id": CALM, "target_seconds": 5.0, "reason": "slow open"},
         {"clip_id": LIVELY, "target_seconds": 5.0, "reason": "payoff"}])

    final = run_job(job, llm=llm, pool=pool)

    assert final["revision_count"] == 1, final.get("history")
    assert final["attempt"] == 2
    assert final["status"] == "done"
    assert final["verdict"] == "PASS"

    history = final["history"]
    assert len(history) == 2
    assert history[0]["verdict"] == "REVISE"
    assert history[1]["verdict"] == "PASS"
    assert history[1]["score"] > history[0]["score"], "revision did not improve"

    store = _store(job)

    # two EditPlan revisions, and they genuinely differ
    plans = store.all_revisions(job.job_id, "EditPlan", EditPlan)
    assert len(plans) == 2
    assert plans[0].payload.opening_clip == CALM
    assert plans[1].payload.opening_clip == LIVELY
    assert plans[1].payload.constraints_applied

    # two renders, both real files, and they are different files
    renders = store.all_revisions(job.job_id, "RenderResult", RenderResult)
    assert len(renders) == 2
    from pathlib import Path
    for r in renders:
        assert Path(r.payload.video_path).exists()

    # the critic compared them
    critics = store.all_revisions(job.job_id, "CriticResult", CriticResult)
    assert critics[1].payload.compared_to_previous["better"] is True


@pytest.mark.slow
def test_the_revision_request_reaches_the_director_in_the_trace(
        agent_root, calm_clip, lively_clip, pool):
    job = _job(agent_root, calm_clip, lively_clip, "gtrace")
    llm = llm_returning_plan(
        [{"clip_id": CALM, "target_seconds": 5.0, "reason": "slow"},
         {"clip_id": LIVELY, "target_seconds": 5.0, "reason": "payoff"}])
    run_job(job, llm=llm, pool=pool)

    from pathlib import Path

    from agents.trace import TraceRecorder
    from agents.trace import shared_tracer
    tracer = shared_tracer(outputs=Path(job.build_dir).resolve().parent)
    rows = tracer.for_job(job.job_id)

    # the director ran twice, and the second run saw the constraints
    director_rows = [r for r in rows if r["node"] == "editing_director"]
    assert len(director_rows) == 2
    notes = " ".join(n for r in director_rows for n in r["notes"])
    assert "constraints:" in notes
    assert "opening_must_be_liveliest" in notes

    # every node is represented, with real timings
    nodes = [r["node"] for r in rows]
    for expected in ("clip_analysis", "editing_director", "render", "critic"):
        assert expected in nodes
    assert all(r["ended_at"] and r["ended_at"] >= r["started_at"] for r in rows)


@pytest.mark.slow
def test_max_revisions_is_enforced_and_escalates(agent_root, calm_clip,
                                                 lively_clip, pool):
    """An impossible threshold must stop at the cap, not loop forever."""
    job = _job(agent_root, calm_clip, lively_clip, "gmax")
    llm = llm_returning_plan(
        [{"clip_id": CALM, "target_seconds": 4.0, "reason": "slow"},
         {"clip_id": LIVELY, "target_seconds": 4.0, "reason": "payoff"}])

    final = run_job(job, llm=llm, pool=pool, pass_threshold=101.0)

    assert final["status"] == "escalated"
    assert final["verdict"] == "ESCALATE"
    assert final["revision_count"] <= 2, "exceeded the revision cap"
    assert len(final["history"]) <= 3, "rendered more times than the cap allows"
    assert final["history"][-1]["verdict"] == "ESCALATE"


@pytest.mark.slow
def test_a_revision_does_not_rerun_the_upstream_understanding(
        agent_root, calm_clip, lively_clip, pool):
    """The point of putting the strategy upstream of the cycle.

    A revision re-plans the edit and re-renders. It must NOT re-look at the
    footage, re-research the subject or re-decide what the video is about -
    none of those inputs changed, and doing so would make every revision cost
    a full pass over everything.
    """
    from agents.artifacts import (Animation, ContentStrategy, Music,
                                  ResearchReport, Story, VisionAnalysis,
                                  Voiceover)

    job = _job(agent_root, calm_clip, lively_clip, "gnorerun")
    llm = llm_returning_plan(
        [{"clip_id": CALM, "target_seconds": 5.0, "reason": "slow open"},
         {"clip_id": LIVELY, "target_seconds": 5.0, "reason": "payoff"}])

    final = run_job(job, llm=llm, pool=pool)
    assert final["revision_count"] == 1, "no revision happened to test"

    store = _store(job)
    for kind, model in (("VisionAnalysis", VisionAnalysis),
                        ("ResearchReport", ResearchReport),
                        ("ContentStrategy", ContentStrategy),
                        ("Story", Story),
                        ("Voiceover", Voiceover),
                        ("Music", Music),
                        ("Animation", Animation)):
        revs = store.all_revisions(job.job_id, kind, model)
        assert len(revs) == 1, f"{kind} re-ran on a revision ({len(revs)}x)"

    # while the agents downstream of the cycle genuinely did run twice
    assert len(store.all_revisions(job.job_id, "EditPlan", EditPlan)) == 2
    assert len(store.all_revisions(job.job_id, "RenderResult",
                                   RenderResult)) == 2

    from agents.trace import shared_tracer
    from pathlib import Path
    tracer = shared_tracer(outputs=Path(job.build_dir).resolve().parent)
    nodes = [r["node"] for r in tracer.for_job(job.job_id)]
    assert nodes.count("content_strategy") == 1
    assert nodes.count("story") == 1
    assert nodes.count("voiceover") == 1
    assert nodes.count("music") == 1
    assert nodes.count("animation") == 1
    assert nodes.count("editing_director") == 2


@pytest.mark.slow
def test_the_strategy_joins_vision_and_research_before_the_director(
        agent_root, calm_clip, lively_clip, pool):
    """Ordering, observed on a real run rather than asserted from the edges."""
    from pathlib import Path

    from agents.trace import shared_tracer

    job = _job(agent_root, calm_clip, lively_clip, "gjoin")
    llm = llm_returning_plan(
        [{"clip_id": LIVELY, "target_seconds": 5.0, "reason": "open strong"},
         {"clip_id": CALM, "target_seconds": 5.0, "reason": "settle"}])
    run_job(job, llm=llm, pool=pool)

    tracer = shared_tracer(outputs=Path(job.build_dir).resolve().parent)
    rows = tracer.for_job(job.job_id)
    started = {}
    for r in rows:
        started.setdefault(r["node"], r["started_at"])

    for upstream in ("vision", "research"):
        assert upstream in started, f"{upstream} never ran"
    assert "content_strategy" in started

    # the strategy began only after both siblings had finished
    ends = {r["node"]: r["ended_at"] for r in rows}
    assert started["content_strategy"] >= ends["vision"]
    assert started["content_strategy"] >= ends["research"]
    assert "story" in started
    assert started["story"] >= ends["content_strategy"]
    assert "voiceover" in started
    assert started["voiceover"] >= ends["story"]
    assert "music" in started
    assert started["music"] >= ends["voiceover"]
    assert "animation" in started
    assert started["animation"] >= ends["music"]
    assert started["editing_director"] >= ends["animation"]


@pytest.mark.slow
def test_every_artifact_is_produced_and_attributed(agent_root, calm_clip,
                                                    lively_clip, pool):
    job = _job(agent_root, calm_clip, lively_clip, "gart")
    llm = llm_returning_plan(
        [{"clip_id": LIVELY, "target_seconds": 5.0, "reason": "open strong"},
         {"clip_id": CALM, "target_seconds": 5.0, "reason": "settle"}])
    run_job(job, llm=llm, pool=pool)

    store = _store(job)
    rows = store.manifest(job.job_id)
    kinds = {r["kind"] for r in rows}
    # VisionAnalysis, ResearchReport and ContentStrategy joined the set in
    # Phase 3. All three are produced on EVERY run, including this one with
    # neither an LLM nor a search provider configured - Vision carrying the
    # measured candidate windows, Research reporting insufficient signal, and
    # the strategy ranking the clips deterministically. An agent that silently
    # produced nothing would be indistinguishable from one that never ran.
    assert kinds == {"ClipAnalysis", "VisionAnalysis", "ResearchReport",
                     "ContentStrategy", "Story", "Voiceover", "Music",
                     "Animation", "EditPlan", "RenderResult", "CriticResult"}

    producers = {r["kind"]: r["produced_by"] for r in rows}
    assert producers["ClipAnalysis"] == "clip_analysis"
    assert producers["VisionAnalysis"] == "vision"
    assert producers["ResearchReport"] == "research"
    assert producers["ContentStrategy"] == "content_strategy"
    assert producers["Story"] == "story"
    assert producers["Voiceover"] == "voiceover"
    assert producers["Music"] == "music"
    assert producers["Animation"] == "animation"
    assert producers["EditPlan"] == "editing_director"
    assert producers["RenderResult"] == "rendering"
    assert producers["CriticResult"] == "critic"

    for r in rows:
        assert r["inputs_hash"], f"{r['kind']} has no inputs_hash"
        assert 0.0 <= r["confidence"] <= 1.0
