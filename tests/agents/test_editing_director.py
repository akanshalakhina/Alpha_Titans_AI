"""
EditingDirector - the reasoning agent, and every way its reasoning is rejected.

The LLM is injected, so none of this needs an API key or a network. Everything
else is real: the deterministic fallback runs the production `plan` stage
against a genuinely probed and measured manifest, and every timestamp is
resolved by the pipeline's own `best_window`.
"""

import json

import pytest

from agents.artifacts import EditPlan, RevisionRequest
from agents.editing_director import EditingDirectorAgent
from conftest import (llm_always_cut_off, llm_cut_off_then,
                      llm_failing, llm_returning, llm_returning_plan)


def run(ctx, clip_set, tracer_span, revision=0, request=None):
    agent = EditingDirectorAgent()
    with tracer_span(ctx, "editing_director") as span:
        return agent.execute(ctx, span, clip_set=clip_set, revision=revision,
                             revision_request=request)


# ------------------------------------------------------- the happy path

def test_valid_llm_plan_is_used_and_not_degraded(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 5.0,
          "reason": "busiest material, opens strong"},
         {"clip_id": "01_calm.mp4", "target_seconds": 4.0,
          "reason": "lets the eye settle"}],
        strategy="open on motion, settle after")

    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is False
    assert art.fallback_used is None
    assert art.payload.opening_clip == "02_lively.mp4"
    assert [s.clip_id for s in art.payload.segments] == ["02_lively.mp4",
                                                         "01_calm.mp4"]
    assert any(d.decided_by == "llm" for d in art.decisions)
    assert art.cost.tokens_in > 0, "token usage must reach the artifact"


def test_the_model_never_chooses_timestamps(director_ctx, tracer_span):
    """Lengths come from the model; in/out points come from best_window."""
    from agents.tools import select_tools as ST

    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 4.0, "reason": "x"}])
    art = run(ctx, clip_set, tracer_span)

    seg = art.payload.segments[0]
    clip = clip_set.by_id("02_lively.mp4")
    expect_a, expect_b, _ = ST.best_window(clip, seg.duration)
    assert (seg.start, seg.end) == (expect_a, expect_b), (
        "in/out points must be the deterministic window search's answer")


def test_tool_calls_prove_the_deterministic_layer_ran(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 4.0, "reason": "x"}])
    run(ctx, clip_set, tracer_span)
    rows = ctx.tracer.for_job(ctx.job.job_id)
    tools = {t["tool"] for r in rows for t in r["tool_calls"]}
    assert "llm.complete_json" in tools
    assert "select.best_window" in tools


# --------------------------------------------------- rejection -> fallback

@pytest.mark.parametrize("reply,because", [
    ("this is not json at all", "no usable JSON"),
    ("{ broken json", "no usable JSON"),
    ('["a", "list", "not", "an", "object"]', "not a JSON object"),
    ('{"segments": []}', "no segments"),
    ('{"segments": "nope"}', "no segments"),
    ('{"no_segments_key": 1}', "no segments"),
])
def test_malformed_llm_output_falls_back(director_ctx, tracer_span, reply,
                                          because):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning(reply)
    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True
    assert art.fallback_used and "deterministic_planner" in art.fallback_used
    assert art.payload.segments, "the fallback must still produce a plan"


def test_a_hallucinated_clip_id_is_rejected(director_ctx, tracer_span):
    """The agent must NEVER invent or accept a clip that was not uploaded."""
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "99_does_not_exist.mp4", "target_seconds": 5.0,
          "reason": "invented"}])
    art = run(ctx, clip_set, tracer_span)

    assert art.degraded is True
    assert "never uploaded" in art.fallback_used
    real = {c.clip_id for c in clip_set.clips}
    assert all(s.clip_id in real for s in art.payload.segments)


def test_a_hallucinated_clip_among_valid_ones_still_rejects_the_reply(
        director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 5.0, "reason": "ok"},
         {"clip_id": "ghost.mp4", "target_seconds": 5.0, "reason": "bad"}])
    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True
    assert "never uploaded" in art.fallback_used


@pytest.mark.parametrize("bad", [-4.0, 0.0, "abc", None, float("nan"),
                                 float("inf")])
def test_invalid_timestamps_are_rejected(director_ctx, tracer_span, bad):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning(json.dumps({
        "strategy": "s", "opening_rationale": "r",
        "segments": [{"clip_id": "02_lively.mp4", "target_seconds": bad,
                      "reason": "x"}]}))
    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True
    assert "deterministic_planner" in art.fallback_used


def test_a_repeated_clip_becomes_two_different_moments(director_ctx,
                                                      tracer_span):
    """A repeat used to be refused outright, which is why two uploads could
    only ever make one cut. It is allowed now - but the second use must be a
    DIFFERENT stretch of the clip, never the same frames again."""
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 4.0, "reason": "a"},
         {"clip_id": "02_lively.mp4", "target_seconds": 4.0, "reason": "b"}])
    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is False, art.fallback_used
    segs = art.payload.segments
    assert segs, "the plan must not come back empty"
    assert {s.clip_id for s in segs} == {"02_lively.mp4"}

    # The guarantee is not "two segments" - a short clip may have no second
    # window of that length free, and the extra use is then DROPPED rather
    # than handed the same frames again. What must always hold is that no
    # two segments share any footage.
    spans = sorted((s.start, s.end) for s in segs)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 >= b1, f"{(a1, b1)} and {(a2, b2)} overlap"


def test_a_transport_failure_falls_back(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_failing("HTTPError: 503")
    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True
    assert "503" in art.fallback_used


def test_no_llm_configured_is_reported_clearly_not_hidden(director_ctx,
                                                          tracer_span):
    """The state of this machine today. It must be visible in the artifact,
    the fallback field, the decisions and the trace."""
    from agents.tools.llm_tools import LLMClient

    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()                       # no key, no transport
    assert ctx.llm.available() is False

    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True
    assert "AT_LLM_API_KEY" in art.fallback_used
    assert art.confidence < 0.6
    fb = [d for d in art.decisions if d.decided_by == "fallback"]
    assert fb and "deterministic" in fb[0].chose
    rows = ctx.tracer.for_job(ctx.job.job_id)
    assert any(r["fallback_used"] for r in rows)


def test_the_fallback_is_the_production_planner(director_ctx, tracer_span):
    """Not a reimplementation: the real `plan` stage writes the manifest."""
    from agents.tools.analysis_tools import read_manifest
    from agents.tools.llm_tools import LLMClient

    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    art = run(ctx, clip_set, tracer_span)

    m = read_manifest(ctx.job.build_dir)
    assert m and m.get("segments"), "the pipeline's plan stage did not run"
    assert m.get("sequence"), "the production planner records its ordering"
    assert len(art.payload.segments) == len(m["segments"])


# ----------------------------------------------------------- revisions

def _weak_first_plan(ctx, clip_set, tracer_span):
    """A plan that deliberately opens on the calm clip."""
    ctx.llm = llm_returning_plan(
        [{"clip_id": "01_calm.mp4", "target_seconds": 5.0, "reason": "slow open"},
         {"clip_id": "02_lively.mp4", "target_seconds": 5.0, "reason": "payoff"}])
    return run(ctx, clip_set, tracer_span)


def test_revision_constraints_change_the_plan(director_ctx, tracer_span):
    """The revised director must consume the constraints, not replay itself."""
    ctx, clip_set = director_ctx
    first = _weak_first_plan(ctx, clip_set, tracer_span)
    assert first.payload.opening_clip == "01_calm.mp4"

    request = RevisionRequest(
        from_attempt=1, target_agent="editing_director",
        issue="the opening measures below the video's own median",
        severity="high",
        constraints={"opening_must_be_liveliest": True,
                     "opening_max_duration": 4.0,
                     "opening_min_mean_ydif": 3.0})

    second = run(ctx, clip_set, tracer_span, revision=1, request=request)

    assert second.payload.opening_clip == "02_lively.mp4", "hook not moved"
    assert second.payload.segments[0].duration <= 4.01, "cap not applied"
    assert second.payload.constraints_applied
    assert [s.cache_key() for s in first.payload.segments] != \
           [s.cache_key() for s in second.payload.segments]


def test_revision_constraints_work_with_no_llm_at_all(director_ctx,
                                                      tracer_span):
    """A revision has to mean something on a machine with no API key."""
    from agents.tools.llm_tools import LLMClient

    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    first = run(ctx, clip_set, tracer_span)

    request = RevisionRequest(
        from_attempt=1, target_agent="editing_director", issue="weak hook",
        severity="high",
        constraints={"opening_clip": "01_calm.mp4",
                     "opening_max_duration": 3.0})
    second = run(ctx, clip_set, tracer_span, revision=1, request=request)

    assert second.payload.opening_clip == "01_calm.mp4"
    assert second.payload.segments[0].duration <= 3.01
    applied = [d for d in second.decisions if d.what == "Revision constraint"]
    assert applied, "no constraint was recorded as applied"


def test_an_unsatisfiable_constraint_is_reported_not_faked(director_ctx,
                                                            tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "01_calm.mp4", "target_seconds": 4.0, "reason": "x"}])
    request = RevisionRequest(
        from_attempt=1, target_agent="editing_director", issue="weak hook",
        severity="high",
        constraints={"opening_min_mean_ydif": 999.0})   # impossible
    art = run(ctx, clip_set, tracer_span, revision=1, request=request)

    unmet = [d for d in art.decisions if d.what == "Revision constraint UNMET"]
    assert unmet, "an impossible constraint must be reported as unmet"
    assert "does not contain" in unmet[0].why or "no window" in unmet[0].why


def test_operator_pin_overrides_everything(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"force_opening_clip": "01_calm.mp4"})})
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 4.0, "reason": "x"},
         {"clip_id": "01_calm.mp4", "target_seconds": 4.0, "reason": "y"}])
    art = run(ctx, clip_set, tracer_span)
    assert art.payload.opening_clip == "01_calm.mp4"
    assert any(d.decided_by == "human" for d in art.decisions)


# ------------------------------------------------------------ clamping

def test_over_long_requests_are_clamped_to_the_pipeline_bounds(director_ctx,
                                                                tracer_span):
    from agents.tools.pipeline_bridge import build_video

    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 300.0,
          "reason": "far too long"}])
    art = run(ctx, clip_set, tracer_span)
    seg = art.payload.segments[0]
    assert seg.duration <= build_video.MAX_SEG
    assert seg.end <= clip_set.by_id("02_lively.mp4").technical.duration + 0.01


def test_under_short_requests_are_raised_to_the_minimum(director_ctx,
                                                         tracer_span):
    from agents.tools.pipeline_bridge import build_video

    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 0.2,
          "reason": "far too short"}])
    art = run(ctx, clip_set, tracer_span)
    assert art.payload.segments[0].duration >= build_video.MIN_SEG - 0.01


def test_plan_is_schema_valid_and_indices_are_contiguous(director_ctx,
                                                          tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_returning_plan(
        [{"clip_id": "02_lively.mp4", "target_seconds": 4.0, "reason": "a"},
         {"clip_id": "01_calm.mp4", "target_seconds": 4.0, "reason": "b"}])
    art = run(ctx, clip_set, tracer_span)
    assert isinstance(art.payload, EditPlan)
    assert [s.index for s in art.payload.segments] == [1, 2]
    assert art.payload.total_duration > 0


# ============================== a reply that ran out of room, not of sense
#
# Run 96912f55fc9a, 16 Oct. The director degraded with:
#
#     deterministic_planner (the model reply was rejected
#                            (no segments in the reply))
#
# and the critic scored the result 69.9 instead of 100. That reason is
# what a model sending a wrong answer looks like, so that is what was
# investigated. The trace said otherwise:
#
#     tokens_out: 1600      against    max_tokens=1600
#
# The model had not sent a wrong answer. It had been cut off at the
# ceiling, mid-array, and the repair layer recovered the last complete
# segment object from inside the envelope it could no longer close. A
# single segment, no `segments` key, valid JSON - and indistinguishable,
# by inspection, from a model that meant to send exactly that.
#
# The fix is in `llm_tools.complete_json`: a cut-off reply is asked again
# with twice the room before anything is concluded about its content.
# These tests hold the director's end of that - the reasoning must
# survive, and when it cannot the reason given must be the true one.

def test_a_cut_off_plan_is_asked_again_and_the_reasoning_survives(
        director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = llm_cut_off_then(
        [{"clip_id": "02_lively.mp4", "target_seconds": 5.0, "reason": "a"},
         {"clip_id": "01_calm.mp4", "target_seconds": 4.0, "reason": "b"}])

    art = run(ctx, clip_set, tracer_span)

    assert art.degraded is False, "a budget that ran out is not a bad answer"
    assert art.fallback_used is None
    assert [s.clip_id for s in art.payload.segments] == ["02_lively.mp4",
                                                         "01_calm.mp4"]
    assert any(d.decided_by == "llm" for d in art.decisions)
    assert ctx.llm.budgets == [2600, 4000], "asked again with more room"


def test_the_wider_retry_is_charged_for(director_ctx, tracer_span):
    """Both calls were made and both cost tokens; the artifact must say
    so, or the per-run token figures quietly understate themselves."""
    ctx, clip_set = director_ctx
    ctx.llm = llm_cut_off_then(
        [{"clip_id": "02_lively.mp4", "target_seconds": 5.0, "reason": "a"}])
    art = run(ctx, clip_set, tracer_span)
    assert art.cost.tokens_out >= 2600 + 4000, art.cost.tokens_out


def test_a_reply_that_never_finishes_says_it_was_cut_off(director_ctx,
                                                          tracer_span):
    """The honesty requirement. The fallback is correct here - what must
    not happen again is blaming the model's judgement for the ceiling."""
    ctx, clip_set = director_ctx
    ctx.llm = llm_always_cut_off(
        [{"clip_id": "02_lively.mp4", "target_seconds": 5.0, "reason": "a"}])

    art = run(ctx, clip_set, tracer_span)

    assert art.degraded is True
    assert art.payload.segments, "the fallback must still produce a plan"
    assert "cut off" in art.fallback_used, art.fallback_used
    assert "no segments in the reply" not in art.fallback_used


def test_a_salvaged_fragment_is_never_planned_from(director_ctx, tracer_span):
    """One segment recovered from a cut-off reply must not become the
    whole video. Three clips were offered; the fragment names one."""
    ctx, clip_set = director_ctx
    ctx.llm = llm_always_cut_off(
        [{"clip_id": "02_lively.mp4", "target_seconds": 5.0, "reason": "a"},
         {"clip_id": "01_calm.mp4", "target_seconds": 4.0, "reason": "b"}])

    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True, (
        "planning from a fragment would look like a reasoned one-clip cut")
