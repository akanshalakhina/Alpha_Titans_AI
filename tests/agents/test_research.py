"""
Research Agent.

Both the search provider and the LLM are injected - no test needs a key, a
network or a paid API. The property most of these defend: the model may only
synthesise evidence that a provider actually returned. An uncited claim is
dropped, and "we don't know" is a valid, expected answer.
"""

import json
import time

import pytest

from agents.artifacts import (ContentOpportunity, EvidenceItem, ResearchReport,
                              TrendItem, VisionAnalysis, VisionClip)
from agents.research_agent import ResearchAgent
from agents.tools import search_tools as ST
from agents.tools.llm_tools import LLMClient, LLMReply
from agents.tools.search_tools import SearchClient, SearchHit, SearchOutcome

URL_A = "https://example.test/a"
URL_B = "https://example.test/b"


# ------------------------------------------------------------- helpers

def hits(n=2, dated=True):
    out = []
    for i, u in enumerate((URL_A, URL_B)[:n]):
        out.append(SearchHit(
            title=f"Result {i}", url=u, snippet=f"snippet {i}",
            source="fake", published_at="2026-09-01" if dated else None))
    return out


def search_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=400, tokens_out=200, usd=0.004)
    return LLMClient(transport=transport)


def patch_search(monkeypatch, outcome_fn):
    """Point the agent's SearchClient at an injected transport."""
    real = SearchClient

    def factory(*a, **kw):
        return real(transport=outcome_fn)

    monkeypatch.setattr("agents.research_agent.SearchClient", factory)


def ok_results(q, n):
    return SearchOutcome(query=q, provider="fake", hits=hits())


GOOD = {
    "trends": [{"label": "automation", "observation": "tools are spreading",
                "evidence_urls": [URL_A], "confidence": 0.8}],
    "topics": ["automated editing"],
    "keywords": ["ai editor", "auto cut"],
    "audience_signals": ["creators want speed"],
    "content_opportunities": [{"idea": "show the workflow",
                               "rationale": "the evidence shows interest",
                               "evidence_urls": [URL_B], "confidence": 0.7}],
    "limitations": ["only two sources"],
    "confidence": 0.7,
}


def run(ctx, tracer_span, vision=None, revision=0):
    agent = ResearchAgent()
    with tracer_span(ctx, "research") as span:
        return agent.execute(ctx, span, vision=vision, revision=revision)


@pytest.fixture
def research_ctx(director_ctx):
    """A context with a declared domain, so research has a subject."""
    ctx, _clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "ai video editing"})})
    return ctx


# ------------------------------------------------ 1. a valid response

def test_a_valid_response_produces_a_sourced_report(research_ctx, tracer_span,
                                                    monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)

    art = run(ctx, tracer_span)
    p = art.payload
    assert art.kind == "ResearchReport" and art.produced_by == "research"
    assert p.domain == "ai video editing"
    assert p.evidence, "no evidence stored"
    assert p.trends and p.trends[0].label == "automation"
    assert p.content_opportunities
    assert p.insufficient_signal is False
    assert 0.0 < p.confidence <= 1.0
    assert art.cost.tokens_in > 0


def test_every_evidence_item_carries_provenance(research_ctx, tracer_span,
                                                monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    for e in art.payload.evidence:
        assert e.url.startswith("https://")
        assert e.title and e.source and e.query
        assert e.retrieved_at > 0, "we must know when WE fetched it"


# --------------------------------- 9. fabricated / missing source data

def test_an_uncited_trend_is_discarded(research_ctx, tracer_span,
                                       monkeypatch):
    """The enforcement point for "do not fabricate"."""
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm({**GOOD, "trends": [
        {"label": "invented", "observation": "no source for this",
         "evidence_urls": [], "confidence": 0.9}]})
    art = run(ctx, tracer_span)
    assert all(t.label != "invented" for t in art.payload.trends)


def test_a_citation_to_a_url_we_never_retrieved_is_discarded(research_ctx,
                                                             tracer_span,
                                                             monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm({**GOOD, "trends": [
        {"label": "hallucinated", "observation": "cites a made-up source",
         "evidence_urls": ["https://never-retrieved.test/x"],
         "confidence": 0.95}]})
    art = run(ctx, tracer_span)
    assert all(t.label != "hallucinated" for t in art.payload.trends)


def test_undated_evidence_is_flagged_not_assumed_current(research_ctx,
                                                         tracer_span,
                                                         monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, lambda q, n: SearchOutcome(
        query=q, provider="fake", hits=hits(dated=False)))
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    p = art.payload
    assert p.undated_count == len(p.evidence)
    assert any("no publication date" in l for l in p.limitations)
    assert p.confidence <= 0.3, "undated evidence must cap confidence"


def test_a_publication_date_is_never_invented():
    e = EvidenceItem(title="t", url=URL_A, source="fake", query="q",
                     retrieved_at=time.time())
    assert e.published_at is None and e.dated is False


# ------------------------------------------ 2. malformed model replies

@pytest.mark.parametrize("reply", ["not json", "{ broken", "[1,2]", "null",
                                   '"a string"'])
def test_a_malformed_reply_keeps_the_evidence_and_degrades(research_ctx,
                                                           tracer_span,
                                                           monkeypatch, reply):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(reply)
    art = run(ctx, tracer_span)
    assert art.degraded is True
    assert art.payload.insufficient_signal is True
    assert art.payload.evidence, "real evidence must survive a bad synthesis"
    assert "evidence only" in art.fallback_used


def test_an_invalid_confidence_is_clamped(research_ctx, tracer_span,
                                          monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    for bad in (5.0, -3, "high", None, float("nan"), float("inf")):
        ctx.llm = search_llm({**GOOD, "confidence": bad})
        art = run(ctx, tracer_span,
                  revision=ctx.store.next_revision(ctx.job.job_id,
                                                   "ResearchReport"))
        assert 0.0 <= art.payload.confidence <= 1.0


def test_oversized_lists_and_text_are_capped(research_ctx, tracer_span,
                                             monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm({**GOOD,
                          "topics": ["t" * 500] * 100,
                          "keywords": ["k" * 500] * 100,
                          "audience_signals": ["s" * 900] * 50})
    art = run(ctx, tracer_span)
    p = art.payload
    assert len(p.topics) <= 10 and all(len(t) <= 120 for t in p.topics)
    assert len(p.keywords) <= 10 and all(len(k) <= 60 for k in p.keywords)
    assert len(p.audience_signals) <= 10


# ------------------------------------------------- 3. missing domain

def test_no_domain_means_no_invented_topic(director_ctx, tracer_span,
                                           monkeypatch):
    ctx, _ = director_ctx
    assert ctx.job.settings.domain == ""
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)

    art = run(ctx, tracer_span)
    p = art.payload
    assert p.insufficient_signal is True
    assert art.degraded is True
    assert p.evidence == [], "nothing should have been searched"
    assert p.trends == []
    assert any("domain" in l for l in p.limitations)


# ------------------------------- 4/5/6. empty, failure, timeout

def test_no_results_reports_insufficient_signal(research_ctx, tracer_span,
                                                monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, lambda q, n: SearchOutcome(query=q,
                                                         provider="fake"))
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    assert art.payload.insufficient_signal is True
    assert art.degraded is True
    assert art.payload.trends == []


def test_a_provider_failure_degrades_without_raising(research_ctx,
                                                     tracer_span, monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, lambda q, n: SearchOutcome(
        query=q, provider="fake", error="HTTPError: 503"))
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    assert art.degraded is True
    assert art.payload.insufficient_signal is True


def test_a_provider_timeout_degrades_without_raising(research_ctx,
                                                     tracer_span, monkeypatch):
    ctx = research_ctx

    def timeout(q, n):
        return SearchOutcome(query=q, provider="fake",
                             error="TimeoutError: timed out")
    patch_search(monkeypatch, timeout)
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    assert art.degraded is True
    assert "insufficient signal" in art.fallback_used


def test_no_provider_configured_is_a_supported_state(research_ctx,
                                                     tracer_span):
    """The state of this machine: no AT_SEARCH_API_KEY."""
    ctx = research_ctx
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)          # real SearchClient, no key
    assert art.payload.live is False
    assert art.payload.insufficient_signal is True
    assert art.degraded is True
    assert art.payload.trends == [], "no provider must mean no invented trends"


def test_no_model_keeps_the_evidence(research_ctx, tracer_span, monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = LLMClient()                 # no key
    art = run(ctx, tracer_span)
    assert art.payload.evidence, "retrieved sources must survive"
    assert art.payload.trends == []
    assert art.degraded is True
    assert "evidence only" in art.fallback_used


# ------------------------------------- 10/11. caching and freshness

def test_identical_inputs_reuse_the_report(research_ctx, tracer_span,
                                           monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    first = run(ctx, tracer_span)
    again = run(ctx, tracer_span, revision=1)
    assert again.artifact_id == first.artifact_id
    rows = ctx.tracer.for_job(ctx.job.job_id)
    assert any(r["status"] == "skipped" for r in rows)


def test_the_cache_key_moves_with_the_freshness_window(research_ctx,
                                                       tracer_span,
                                                       monkeypatch):
    """Research must not be reused forever - the TTL enters the hash."""
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    monkeypatch.setenv("AT_RESEARCH_TTL", "1")
    first = run(ctx, tracer_span)
    time.sleep(1.2)                       # cross the bucket boundary
    second = run(ctx, tracer_span, revision=1)
    assert second.inputs_hash != first.inputs_hash, "stale research was reused"


def test_a_different_domain_is_different_research(research_ctx, tracer_span,
                                                  monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    first = run(ctx, tracer_span)
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": "baking"})})
    second = run(ctx, tracer_span, revision=1)
    assert second.inputs_hash != first.inputs_hash
    assert second.payload.domain == "baking"


# --------------------------------------- 12/13. trace and decisions

def test_the_trace_shows_the_searches_that_ran(research_ctx, tracer_span,
                                               monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    run(ctx, tracer_span)
    rows = ctx.tracer.for_job(ctx.job.job_id)
    tools = {t["tool"] for r in rows for t in r["tool_calls"]}
    assert "search.query" in tools, "the searches were not traced"
    assert "llm.complete_json" in tools


def test_decisions_use_the_existing_structure(research_ctx, tracer_span,
                                              monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    assert art.decisions
    allowed = {"llm", "measurement", "policy", "human", "fallback"}
    for d in art.decisions:
        assert d.decided_by in allowed, "an incompatible decided_by was used"
        assert d.what and d.why and 0.0 <= d.confidence <= 1.0
    assert any(d.what == "Research scope" for d in art.decisions)


def test_the_artifact_keeps_the_full_envelope(research_ctx, tracer_span,
                                              monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    assert art.artifact_id and art.job_id and art.trace_id and art.inputs_hash
    assert art.produced_by == "research" and art.kind == "ResearchReport"
    assert art.schema_version >= 1 and isinstance(art.revision, int)
    assert art.cost is not None


# ----------------------------------------- 16/17. safety and injection

def test_hostile_urls_never_enter_the_evidence():
    for bad in ("javascript:alert(1)", "file:///etc/passwd", "data:text/html,x",
                "ftp://x.test/a", "not a url", "", None):
        assert ST._clean_url(bad) is None
    assert ST._clean_url("https://ok.test/x") == "https://ok.test/x"


def test_a_result_with_no_usable_url_is_dropped():
    assert ST._hit("title", "javascript:x", "s", "fake") is None
    assert ST._hit("", "https://ok.test/x", "s", "fake") is None


def test_malicious_strings_stay_inert_data(research_ctx, tracer_span,
                                           monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, lambda q, n: SearchOutcome(
        query=q, provider="fake", hits=[SearchHit(
            title="; rm -rf / #", url=URL_A,
            snippet="$(whoami) `id` ../../etc/passwd DROP TABLE artifacts;--",
            source="fake")]))
    ctx.llm = search_llm({**GOOD, "trends": [
        {"label": "'; DROP TABLE artifacts;--",
         "observation": "$(rm -rf /) ../../secret",
         "evidence_urls": [URL_A], "confidence": 0.5}],
        "keywords": ["-vf scale=1:1", "| nc attacker 4444"]})
    art = run(ctx, tracer_span)
    for t in art.payload.trends:
        assert isinstance(t.label, str) and isinstance(t.observation, str)
    for k in art.payload.keywords:
        assert isinstance(k, str)
    for e in art.payload.evidence:
        assert e.url.startswith("https://")


def test_the_report_cannot_reach_rendering(research_ctx, tracer_span,
                                           monkeypatch):
    """Research is information. It must not carry anything executable."""
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    fields = set(ResearchReport.model_fields)
    for forbidden in ("command", "path", "filtergraph", "filter", "sql",
                      "exec", "script", "url_to_fetch"):
        assert forbidden not in fields


# ------------------------------------------------ 18. domain neutrality

@pytest.mark.parametrize("domain", [
    "LEGO Technic builds", "sourdough baking", "trail running",
    "python programming", "personal finance", "competitive gaming",
])
def test_research_works_for_any_domain(research_ctx, tracer_span, monkeypatch,
                                       domain):
    ctx = research_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": domain})})
    seen: list = []
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD, record=seen)
    art = run(ctx, tracer_span)
    assert art.payload.domain == domain
    assert any(domain in q for q in art.payload.research_queries)
    assert seen and domain in seen[0]


def test_queries_are_built_deterministically_not_by_the_model(research_ctx,
                                                              tracer_span,
                                                              monkeypatch):
    """The model does not choose what to search for."""
    from datetime import datetime, timezone
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    art = run(ctx, tracer_span)
    year = str(datetime.now(timezone.utc).year)
    assert any(year in q for q in art.payload.research_queries), (
        "the real current year must reach the query")
    assert all("ai video editing" in q for q in art.payload.research_queries)


def test_vision_subjects_ground_a_query_when_available(research_ctx,
                                                       tracer_span,
                                                       monkeypatch):
    ctx = research_ctx
    patch_search(monkeypatch, ok_results)
    ctx.llm = search_llm(GOOD)
    v = VisionAnalysis(live=True, clips={"a.mp4": VisionClip(
        clip_id="a.mp4", summary="timeline editing",
        subjects=["timeline", "keyboard"])})
    art = run(ctx, tracer_span, vision=v)
    joined = " ".join(art.payload.research_queries)
    assert "timeline" in joined, "vision subjects did not reach the query"


# --------------------------------------------- 15. graph integration

def test_research_is_a_parallel_branch_that_joins_at_the_strategy():
    """The join moved from the director to ContentStrategy.

    Renamed with the edge it asserts, because a test called "joins at the
    director" that no longer checks the director would be a worse lie than a
    failing one. Everything else here is as it was - the fan-out, research's
    independence from vision, and the untouched revision cycle.
    """
    import tempfile
    from pathlib import Path

    from agents.graph import build_graph
    graph, conn = build_graph(Path(tempfile.mkdtemp()) / "cp.db")
    try:
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        # fan out
        assert ("clip_analysis", "vision") in edges
        assert ("clip_analysis", "research") in edges
        # join - now at the strategy, which then briefs the director
        assert ("vision", "content_strategy") in edges
        assert ("research", "content_strategy") in edges
        assert ("content_strategy", "story") in edges
        assert ("story", "voiceover") in edges
        assert ("voiceover", "music") in edges
        assert ("music", "animation") in edges
        assert ("animation", "editing_director") in edges
        assert ("music", "editing_director") not in edges
        assert ("voiceover", "editing_director") not in edges
        assert ("story", "editing_director") not in edges
        assert ("content_strategy", "editing_director") not in edges
        assert ("research", "editing_director") not in edges
        # research is independent of vision
        assert ("vision", "research") not in edges
        assert ("research", "vision") not in edges
        # the revision cycle is untouched
        assert ("critic", "editing_director") in edges
    finally:
        conn.close()


def test_research_has_a_retry_and_deadline_policy():
    from agents.graph import POLICIES
    p = POLICIES["research"]
    assert p.attempts >= 2 and p.timeout_s and p.timeout_s > 0


def test_parallel_nodes_write_disjoint_state_keys():
    """LangGraph raises InvalidUpdateError if two parallel nodes write the
    same key - verified against this graph shape before adopting it."""
    import inspect

    from agents import graph as G
    vsrc = inspect.getsource(G.node_vision)
    rsrc = inspect.getsource(G.node_research)
    assert '"vision_id"' in vsrc and '"research_id"' in rsrc
    assert '"research_id"' not in vsrc and '"vision_id"' not in rsrc


def test_a_research_failure_does_not_stop_the_graph(monkeypatch):
    from agents import graph as G

    def boom(*a, **kw):
        raise RuntimeError("research exploded")

    monkeypatch.setattr(G, "_run_node", boom)

    class FakeStore:
        def get(self, *a, **kw):
            return None

    class FakeRT:
        store = FakeStore()
        log: list = []

    monkeypatch.setattr(G, "runtime_for", lambda _j: FakeRT())
    out = G.node_research({"job_id": "j", "attempt": 1})
    assert out["research_id"] is None
    assert "research exploded" in out["research_failed"]
    assert FakeRT.log
