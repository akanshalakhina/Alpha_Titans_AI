"""
ClipAnalysis agent - against REAL clips and the REAL pipeline stages.

Nothing is mocked here. ffmpeg probes and measures actual files, and the
assertions are about the numbers it produced.
"""

import pytest

from agents.artifacts import ClipAnalysisSet
from agents.clip_analysis import ClipAnalysisAgent


def test_clip_analysis_runs_the_real_pipeline(analyzed):
    art, ctx = analyzed
    assert art.kind == "ClipAnalysis"
    assert art.produced_by == "clip_analysis"
    assert art.inputs_hash
    cs: ClipAnalysisSet = art.payload
    assert len(cs.clips) == 2
    assert {c.clip_id for c in cs.clips} == {"01_calm.mp4", "02_lively.mp4"}


def test_every_clip_carries_real_measurements(analyzed):
    art, _ = analyzed
    for c in art.payload.clips:
        assert c.motion.measured, f"{c.clip_id} was not measured"
        assert c.motion.ydif_series, "no YDIF series"
        assert len(c.motion.ydif_series) > 20
        assert c.technical.duration > 0
        assert c.technical.fps > 0
        assert c.technical.codec == "h264"


def test_the_lively_clip_measures_livelier_than_the_calm_one(analyzed):
    """The fixture's premise, verified rather than assumed."""
    art, _ = analyzed
    calm = art.payload.by_id("01_calm.mp4")
    lively = art.payload.by_id("02_lively.mp4")
    assert lively.motion.mean_ydif > calm.motion.mean_ydif * 2, (
        f"calm={calm.motion.mean_ydif} lively={lively.motion.mean_ydif}")


def test_both_clips_survive_the_usability_floor(analyzed):
    """Calm is not the same as dead - the pipeline's own rule, checked."""
    art, _ = analyzed
    assert len(art.payload.usable_clips) == 2
    for c in art.payload.clips:
        assert c.usability.usable, c.usability.reason


def test_decisions_are_recorded_with_reasons(analyzed):
    art, _ = analyzed
    kinds = {d.what for d in art.decisions}
    assert "Clips readable" in kinds
    assert "Usable footage" in kinds
    assert "Output shape" in kinds
    for d in art.decisions:
        assert d.why, f"{d.what} has no reason"
        assert d.decided_by in ("llm", "measurement", "policy", "human",
                                "fallback")


def test_trace_shows_the_real_tools_that_ran(analyzed):
    _, ctx = analyzed
    rows = ctx.tracer.for_job(ctx.job.job_id)
    tools = {t["tool"] for r in rows for t in r["tool_calls"]}
    assert "pipeline.probe" in tools
    assert "pipeline.measure" in tools
    assert "motion.ydif_series" in tools
    assert "motion.freeze_runs" in tools


def test_rerunning_hits_the_input_hash_cache(analyzed, tracer_span):
    """Identical inputs must not re-measure: that is what makes a revision
    cheap."""
    art, ctx = analyzed
    agent = ClipAnalysisAgent()
    with tracer_span(ctx, "clip_analysis") as span:
        again = agent.execute(ctx, span, revision=1)
    assert again.artifact_id == art.artifact_id, "expected the cached artifact"
    rows = ctx.tracer.for_job(ctx.job.job_id)
    assert any(r["status"] == "skipped" for r in rows)


def test_canvas_follows_the_requested_resolution(analyzed):
    art, _ = analyzed
    # the fixture asks for 720p horizontal
    assert art.payload.canvas == (1280, 720)
