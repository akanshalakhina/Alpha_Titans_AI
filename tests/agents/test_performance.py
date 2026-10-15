"""
PerformanceAnalysis Agent.

Deterministic - there is no model to inject, and no process or connection to
make. The properties most of these defend:

  * nothing in the project can read a video's performance: no analytics or
    statistics code, no record of an uploaded video, and no publication - the
    agent reads all three, and parity tests fail the moment one changes
  * nothing it produces can carry a figure: no field for views, CTR,
    retention, likes, revenue, a URL or a video id, and every metric is
    listed as UNMEASURED
  * pre-publication data - the critic's score, a "ready" plan, Research's
    evidence, the title's `published` flag - is named as EXCLUDED and never
    quoted, so not one of its numbers reaches the report
  * the unavailable state is a declared fallback: degraded, confidence 0.0,
    with provenance, decisions, caching and a trace like any other agent
"""

import json
import socket
import time
import typing

import pytest
from pydantic import ValidationError

from agents.artifacts import (AnalyticsProvider, CreativeCheck, CriticResult,
                              EvidenceItem, MeasuredMetric,
                              PerformanceAnalysis,
                              PerformanceGap, PerformanceMetric,
                              PerformanceStatus, PublishAsset, PublishCheck,
                              Publishing, ResearchReport, Title,
                              UnmeasuredMetric)
from agents.performance import (NEEDS, PerformanceAnalysisAgent,
                                _ANALYTICS_MARKERS, _analytics, _code,
                                _scan, _video_references)
from agents.publishing import CRITIC_PASSED, VIDEO_ON_DISK

METRICS = typing.get_args(PerformanceMetric)


# ------------------------------------------------------------- helpers

@pytest.fixture
def perf_ctx(director_ctx):
    """A real store and tracer, private to the test."""
    ctx, _cs = director_ctx
    return ctx


def a_plan(status="ready") -> Publishing:
    """A publishing plan as the Publishing agent leaves it - validated by
    the real schema, so it can only be a state that agent can produce."""
    video = PublishAsset(kind="video", path="/b/ALPHA_TITANS_horizontal.mp4",
                         file_name="ALPHA_TITANS_horizontal.mp4", size=13_000,
                         width=1920, height=1080, codec="h264",
                         duration=23.5)
    thumb = PublishAsset(kind="thumbnail", path="/b/thumbnail_agent_r00.jpg",
                         file_name="thumbnail_agent_r00.jpg", size=60_000,
                         width=1920, height=1080, codec="mjpeg")
    ok = PublishCheck(check=VIDEO_ON_DISK, passed=True, blocking=True)
    checks = {"ready": [ok],
              "needs_review": [ok, PublishCheck(check=CRITIC_PASSED,
                                                passed=False,
                                                source="policy")],
              "blocked": [PublishCheck(check=VIDEO_ON_DISK, passed=False,
                                       blocking=True)]}[status]
    return Publishing(status=status, video=video, thumbnail=thumb,
                      thumbnail_from="thumbnail_agent", title="LEGO Gears",
                      title_from="title_agent", checks=checks)


def a_critic(score=97.3) -> CriticResult:
    return CriticResult(
        attempt=1, verdict="PASS", overall_score=score, pass_threshold=65.0,
        creative=[CreativeCheck(dimension="hook", score=score,
                                evidence="first 3s measures 9.1 mean YDIF",
                                confidence=1.0)],
        summary=f"hook scores {score} against a 65 threshold")


def a_research() -> ResearchReport:
    """Evidence that LOOKS like performance - other people's view counts."""
    return ResearchReport(
        domain="LEGO Technic", provider="youtube", live=True,
        evidence=[EvidenceItem(
            title="Best LEGO gearbox - 1.2M views",
            url="https://www.youtube.com/watch?v=abc123XYZ",
            snippet="1,234,567 views - 48K likes", source="youtube",
            query="lego technic", retrieved_at=time.time())],
        audience_signals=["builds average an 8.4% click-through rate"],
        insufficient_signal=False, confidence=0.7)


def a_title() -> Title:
    return Title(text="LEGO Gears", source="model", published=True)


def run(ctx, tracer_span, publishing="ready", critic=None, research=None,
        title=None, revision=0):
    with tracer_span(ctx, "performance_analysis") as span:
        return PerformanceAnalysisAgent().execute(
            ctx, span,
            publishing=a_plan(publishing) if isinstance(publishing, str)
            else publishing,
            critic=critic, research=research, title=title,
            revision=revision)


def everything(ctx, tracer_span, **kw):
    return run(ctx, tracer_span, critic=a_critic(), research=a_research(),
               title=a_title(), **kw)


# -------------------- 1. what the project can actually do

def test_the_only_analytics_path_is_a_file_the_operator_exports():
    """There is still no API this project can call for statistics.

    This test used to assert that performance could not be measured at
    all. It can now - but only from a CSV the operator exported from
    their own warehouse and dropped in `data/analytics`, because no
    Databricks or BigQuery driver is installed and reaching YouTube
    Analytics would need the channel owner's OAuth consent. That is the
    whole reason the provider is called `youtube_export` and not
    `youtube`: these numbers are as fresh as that file, and nothing in
    the code can refresh them.
    """
    assert _analytics()["analytics"] == [],         "a live analytics path exists - the provider literal must say so"
    assert typing.get_args(AnalyticsProvider) == ("none", "youtube_export")
    assert typing.get_args(PerformanceStatus) == ("unavailable", "measured")

    from pathlib import Path as _P
    from agents.tools import analytics_store as AS
    src = _P(AS.__file__).read_text(encoding="utf-8")
    for reaching_out in ("urllib", "requests", "httpx", "socket"):
        assert reaching_out not in src,             f"the export reader must not reach the network ({reaching_out})"


def test_nothing_publishes_so_nothing_can_have_performed():
    """The publication the report would read is structurally impossible."""
    assert typing.get_args(
        Publishing.model_fields["published"].annotation) == (False,)


def test_nothing_records_which_video_was_uploaded():
    assert _video_references() == [], \
        "a field for an uploaded video now exists - read it, or say why not"


def test_the_only_youtube_code_is_a_search_with_no_statistics():
    from pathlib import Path
    assert _analytics()["search"] == ["server/agents/tools/search_tools.py"]
    root = Path(__file__).resolve().parents[2]
    code = _code((root / "server/agents/tools/search_tools.py").read_text())
    assert "'part': 'snippet'" in code and "statistics" not in code


def test_the_scan_reads_code_not_prose():
    prose = ('"""would need youtubeAnalytics reports().query and '
             'yt-analytics.readonly"""\nx = 1  # videos().list\n')
    code = ("from googleapiclient.discovery import build\n"
            "yt = build('youtubeAnalytics', 'v2')\n"
            "yt.reports().query(ids='channel==MINE', metrics='views')\n")
    assert not [m for m in _ANALYTICS_MARKERS if m in _code(prose)]
    assert {"youtubeAnalytics", "reports().query", "googleapiclient"} <= \
        {m for m in _ANALYTICS_MARKERS if m in _code(code)}


@pytest.mark.parametrize("src", [
    "URL = 'https://www.googleapis.com/youtube/v3/videos?part=statistics'\n",
    "SCOPES = ['https://www.googleapis.com/auth/yt-analytics.readonly']\n",
    "yt.videos().list(part='statistics', id=vid).execute()\n",
    "HOST = 'https://youtubereporting.googleapis.com/v1/jobs'\n",
])
def test_the_scan_would_catch_a_real_analytics_path(tmp_path, src):
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "stats.py").write_text(src)
    found = _scan(str(tmp_path), (("server/stats.py", 1),))
    assert found[1] == ("server/stats.py",)


def test_a_reference_field_would_be_found(monkeypatch):
    import agents.performance as perf
    from pydantic import BaseModel

    class Settings(BaseModel):
        title: str = ""
        youtube_url: str = ""
    monkeypatch.setattr(perf, "_REFERENCE_MODELS", (Settings,))
    assert perf._video_references() == ["Settings.youtube_url"]


# ----------------------- 2. unavailable, with the measured reasons

def test_a_ready_plan_still_has_no_performance(perf_ctx, tracer_span):
    art = run(perf_ctx, tracer_span)
    p = art.payload
    assert p.status == "unavailable" and p.provider == "none"
    assert p.published is False and p.publishing_status == "ready"
    assert p.reasons == ["not_published", "no_video_reference",
                         "no_analytics_provider"]
    assert "published=False" in p.gaps[0].evidence
    assert sorted(m.metric for m in p.unmeasured) == sorted(METRICS)
    assert all(m.needs for m in p.unmeasured)
    steps = " ".join(p.next_steps)
    assert "Publish it by hand" in steps and "YouTube Studio" in steps


def test_the_unavailable_state_is_a_declared_fallback(perf_ctx, tracer_span):
    art = run(perf_ctx, tracer_span)
    assert art.degraded is True and art.confidence == 0.0
    assert art.fallback_used.startswith("no performance data (")
    assert "has not been published" in art.fallback_used
    assert art.produced_by == "performance_analysis"
    assert art.kind == "PerformanceAnalysis" and art.inputs_hash


def test_no_publishing_plan_is_its_own_reason(perf_ctx, tracer_span):
    p = run(perf_ctx, tracer_span, publishing=None).payload
    assert p.status == "unavailable" and p.publishing_status is None
    assert p.reasons[0] == "no_publishing_plan"
    assert "not_published" not in p.reasons
    assert p.next_steps[0].startswith("Make the publishing plan first")


@pytest.mark.parametrize("status", ["needs_review", "blocked"])
def test_an_unready_plan_is_resolved_before_anything_else(
        perf_ctx, tracer_span, status):
    p = run(perf_ctx, tracer_span, publishing=status).payload
    assert p.status == "unavailable" and p.publishing_status == status
    assert p.next_steps[0].startswith(f"Resolve the publishing plan first "
                                      f"- it is {status}")
    failed = CRITIC_PASSED if status == "needs_review" else VIDEO_ON_DISK
    assert failed in p.next_steps[0]


# ----------------------------------- 3. the schema cannot hold a figure

def _unavailable(**kw):
    base = dict(publishing_status="ready",
                gaps=[PerformanceGap(reason="not_published", evidence="x")],
                unmeasured=[UnmeasuredMetric(metric=m, needs="x")
                            for m in METRICS])
    return PerformanceAnalysis(**{**base, **kw})


@pytest.mark.parametrize("kw", [
    {"views": 1200}, {"view_count": 1200}, {"ctr": 0.084},
    {"click_through_rate": 0.084}, {"retention": 0.51},
    {"audience_retention": [1.0, 0.6]}, {"likes": 48}, {"revenue": 3.2},
    {"url": "https://youtu.be/abc123XYZ"}, {"metrics": {"views": 1}},
    {"provider": "youtube_analytics"}, {"status": "insufficient_data"},
    {"status": "available"},
])
def test_the_schema_cannot_carry_a_figure_or_a_claim(kw):
    _unavailable()                                 # the base itself is valid
    with pytest.raises(ValidationError):
        _unavailable(**kw)


def test_an_unavailable_report_may_not_carry_a_number():
    """The one thing that must never happen: a report saying it knows
    nothing while holding a figure everything downstream would believe."""
    with pytest.raises(ValidationError):
        _unavailable(measured=[MeasuredMetric(metric="views", value=1200,
                                              source="oct.csv")])


def test_recording_the_upload_is_not_a_performance_claim():
    """`published` and `video_id` may be set the moment the operator
    pastes the link - before any export exists. Saying WHICH video this
    became is not the same as saying how it did, so this is allowed and
    the report stays unavailable."""
    p = _unavailable(published=True, video_id="dQw4w9WgXcQ")
    assert p.status == "unavailable" and p.measured == []


@pytest.mark.parametrize("kw", [
    {},                                             # no metric at all
    {"measured": [MeasuredMetric(metric="views", value=1, source="x.csv")]},
    {"measured": [MeasuredMetric(metric="views", value=1, source="x.csv")],
     "video_id": "dQw4w9WgXcQ"},                    # not marked published
])
def test_a_measured_report_must_say_what_and_whose(kw):
    """An empty "measured" is the worst of both: it reads as real
    performance and contains nothing, and Learning would count it."""
    with pytest.raises(ValidationError):
        PerformanceAnalysis(status="measured", provider="youtube_export",
                            **kw)


def test_a_complete_measured_report_is_accepted():
    p = PerformanceAnalysis(
        status="measured", provider="youtube_export", published=True,
        video_id="dQw4w9WgXcQ", window_days=30,
        measured=[MeasuredMetric(metric="views", value=10432,
                                 source="oct.csv")])
    assert p.measured[0].value == 10432 and p.measured[0].source == "oct.csv"


@pytest.mark.parametrize("kw", [{"value": 1200}, {"views": 1200}])
def test_an_unmeasured_metric_has_nowhere_to_put_a_value(kw):
    with pytest.raises(ValidationError):
        UnmeasuredMetric(metric="views", needs="x", **kw)


def test_unavailable_must_say_why_and_name_every_metric():
    every = [UnmeasuredMetric(metric=m, needs="x") for m in METRICS]
    gap = PerformanceGap(reason="not_published", evidence="x")
    with pytest.raises(ValidationError):            # no reason
        PerformanceAnalysis(publishing_status="ready", unmeasured=every)
    with pytest.raises(ValidationError):            # a metric left out
        PerformanceAnalysis(publishing_status="ready", gaps=[gap],
                            unmeasured=every[1:])
    with pytest.raises(ValidationError):            # a metric twice
        PerformanceAnalysis(publishing_status="ready", gaps=[gap],
                            unmeasured=every + every[:1])
    with pytest.raises(ValidationError):            # a reason twice
        PerformanceAnalysis(publishing_status="ready", gaps=[gap, gap],
                            unmeasured=every)


def test_the_publication_reason_matches_the_plan():
    every = [UnmeasuredMetric(metric=m, needs="x") for m in METRICS]
    no_plan = PerformanceGap(reason="no_publishing_plan", evidence="x")
    unpub = PerformanceGap(reason="not_published", evidence="x")
    for kw in ({"gaps": [no_plan], "publishing_status": "ready"},
               {"gaps": [unpub], "publishing_status": None},
               {"gaps": [no_plan, unpub], "publishing_status": None},
               {"gaps": [PerformanceGap(reason="no_analytics_provider",
                                        evidence="x")],
                "publishing_status": "ready"}):
        with pytest.raises(ValidationError):
            PerformanceAnalysis(unmeasured=every, **kw)


def test_the_agent_is_deterministic_by_design():
    assert PerformanceAnalysisAgent.uses_llm is False
    assert PerformanceAnalysisAgent.agency == "deterministic"
    assert set(NEEDS) == set(METRICS)


# --------------------- 4. pre-publication data is not performance data

def test_pre_publication_signals_are_named_and_excluded(perf_ctx,
                                                        tracer_span):
    p = everything(perf_ctx, tracer_span).payload
    assert p.status == "unavailable"
    assert [e.artifact for e in p.excluded_signals] == [
        "CriticResult", "Publishing", "ResearchReport", "Title"]
    why = {e.artifact: e.why for e in p.excluded_signals}
    assert "not audience retention" in why["CriticResult"]
    assert "not received well" in why["Publishing"]
    assert "never this one" in why["ResearchReport"]
    assert "not that a video was published" in why["Title"]


def test_not_one_pre_publication_number_reaches_the_report(perf_ctx,
                                                           tracer_span):
    """The critic said 97.3, Research's evidence says 1.2M views and 8.4%
    CTR - and the report, decisions and all, says none of it."""
    art = everything(perf_ctx, tracer_span)
    # the payload and the decisions - the envelope's own timestamps are not
    # the report, and a clock reading can contain any digits at all
    text = art.payload.model_dump_json() + json.dumps(
        [d.model_dump() for d in art.decisions])
    for leaked in ("97.3", "9.1", "1.2M", "1,234,567", "48K", "8.4%",
                   "abc123XYZ", "youtube.com"):
        assert leaked not in text, leaked
    assert not any(ch.isdigit() for e in art.payload.excluded_signals
                   for ch in e.signal + e.why)


def test_a_critic_score_never_changes_the_outcome(perf_ctx, tracer_span,
                                                  monkeypatch):
    # made twice for real - the cache would make this trivially true
    monkeypatch.setattr(perf_ctx.store, "get_by_inputs",
                        lambda *a, **kw: None)
    lo = run(perf_ctx, tracer_span, critic=a_critic(12.0))
    hi = run(perf_ctx, tracer_span, critic=a_critic(99.9), revision=1)
    assert lo.artifact_id != hi.artifact_id
    assert lo.payload == hi.payload and lo.decisions == hi.decisions, \
        "a pre-publication score moved the performance report"


def test_research_without_content_is_not_listed(perf_ctx, tracer_span):
    empty = ResearchReport(domain="x", insufficient_signal=True)
    p = run(perf_ctx, tracer_span, research=empty).payload
    assert "ResearchReport" not in [e.artifact for e in p.excluded_signals]


def test_the_titles_published_flag_is_not_a_publication(perf_ctx,
                                                       tracer_span):
    p = run(perf_ctx, tracer_span, title=a_title()).payload
    assert p.published is False and "not_published" in p.reasons


# --------------------------- 5. no model, no process, no network

class _Refuse:
    def __init__(self, what):
        self.what = what

    def __getattr__(self, name):
        raise AssertionError(f"the agent touched the {self.what} ({name})")


def test_it_never_calls_a_model_or_starts_a_process(perf_ctx, tracer_span):
    perf_ctx.llm = _Refuse("LLM")
    perf_ctx.pool = _Refuse("media pool")
    p = everything(perf_ctx, tracer_span).payload
    assert p.status == "unavailable"


def test_it_never_touches_the_network(perf_ctx, tracer_span, monkeypatch):
    attempts = []

    def refuse(self, *a, **kw):
        attempts.append(a)
        raise AssertionError("the performance agent opened a connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    p = everything(perf_ctx, tracer_span).payload
    assert attempts == [] and p.status == "unavailable"


# ---------------------------------------------------------- 6. caching

def test_identical_inputs_reuse_the_report(perf_ctx, tracer_span):
    a = everything(perf_ctx, tracer_span)
    b = everything(perf_ctx, tracer_span, revision=1)
    assert a.artifact_id == b.artifact_id
    rows = [r for r in perf_ctx.tracer.for_job(perf_ctx.job.job_id)
            if r["node"] == "performance_analysis"]
    assert rows[-1]["cache_hits"] == 1 and rows[-1]["status"] == "skipped"


@pytest.mark.parametrize("change", [
    {"publishing": "needs_review"}, {"publishing": None},
    {"critic": None}, {"research": None}, {"title": None}])
def test_the_report_is_remade_when_an_input_changes(perf_ctx, tracer_span,
                                                    change):
    a = everything(perf_ctx, tracer_span)
    kw = {"publishing": "ready", "critic": a_critic(),
          "research": a_research(), "title": a_title(), **change}
    b = run(perf_ctx, tracer_span, revision=1, **kw)
    assert a.inputs_hash != b.inputs_hash


def test_a_score_alone_does_not_bust_the_cache(perf_ctx, tracer_span):
    """The score is never read, so it cannot be part of the identity."""
    a = run(perf_ctx, tracer_span, critic=a_critic(12.0))
    b = run(perf_ctx, tracer_span, critic=a_critic(99.9), revision=1)
    assert a.artifact_id == b.artifact_id


# --------------------------------------------- 7. trace and decisions

def test_the_trace_records_what_was_read_and_the_fallback(perf_ctx,
                                                          tracer_span):
    with tracer_span(perf_ctx, "performance_analysis") as span:
        art = PerformanceAnalysisAgent().execute(
            perf_ctx, span, publishing=a_plan(), critic=a_critic())
        assert art.trace_id == span.trace_id
    rows = [r for r in perf_ctx.tracer.for_job(perf_ctx.job.job_id)
            if r["node"] == "performance_analysis"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert tools == ["performance.scan_analytics",
                     "performance.find_video_reference",
                     "performance.read_publication",
                     "performance.read_analytics_export"]
    read = rows[-1]["tool_calls"][2]
    assert read["plan"] is True and read["published"] is False
    assert rows[-1]["fallback_used"] == art.fallback_used
    assert art.artifact_id in rows[-1]["output_artifacts"]
    assert rows[-1]["tokens_in"] == 0 and rows[-1]["tokens_out"] == 0


def test_decisions_are_measured_policy_or_the_declared_fallback(
        perf_ctx, tracer_span):
    art = everything(perf_ctx, tracer_span)
    assert "llm" not in {d.decided_by for d in art.decisions}
    by = {d.what: d for d in art.decisions}
    assert by["Analytics provider"].decided_by == "measurement"
    assert by["Analytics provider"].chose.startswith("none")
    assert by["Publication"].decided_by == "measurement"
    assert by["Publication"].chose == "not published"
    assert by["Performance data"].decided_by == "fallback"
    assert by["Performance data"].chose.startswith("unavailable")
    assert by["Pre-publication signals"].decided_by == "policy"
    assert "critic" in by["Pre-publication signals"].chose
    assert all(d.confidence == 1.0 for d in art.decisions)


def test_the_capabilities_and_limitations_are_stated(perf_ctx, tracer_span):
    p = everything(perf_ctx, tracer_span).payload
    caps = " ".join(p.platform_capabilities)
    assert "search_tools.py" in caps and "no statistics" in caps
    assert "no YouTube Analytics, video statistics or channel access" in caps
    assert "uploaded video's id or URL: none" in caps
    lims = " ".join(p.limitations)
    assert "nothing is estimated, predicted or modelled" in lims
    assert "set aside, not converted" in lims


# ------------------------------------------ 8. reported through the app

def test_the_projection_has_no_place_for_a_figure(perf_ctx, tracer_span):
    from agents.compat import _project_performance
    p = everything(perf_ctx, tracer_span).payload
    out = _project_performance(p)
    assert out["status"] == "unavailable" and out["available"] is False
    assert out["published"] is False and out["provider"] == "none"
    assert out["reasons"] == p.reasons
    assert sorted(out["unmeasured"]) == sorted(METRICS)
    assert out["excluded"] == [e.signal for e in p.excluded_signals]
    assert not set(out) & (set(METRICS) | {"url", "video_id", "metrics"})
    json.dumps(out)
