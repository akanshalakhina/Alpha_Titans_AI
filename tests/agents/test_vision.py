"""
Vision Agent.

The LLM is injected throughout - no test needs an API key and none calls a
paid API. Everything else is real: real clips, real frame sampling, real
measured candidate windows.

The property most of these tests exist to defend: the model may choose WHICH
window matters, never WHERE it is. Timestamps come from the measurement pass.
"""

import json

import pytest

from agents.artifacts import (CandidateWindow, ClipAnalysisSet, VisionAnalysis,
                              VisionClip)
from agents.tools.llm_tools import LLMClient, LLMReply
from agents.vision_agent import VisionAgent


# ------------------------------------------------------------- helpers

def vision_llm(payload: dict | str, *, record: list | None = None,
               cross: dict | None = None) -> LLMClient:
    """A transport that answers the per-clip pass, then the cross-clip pass.

    Records what it was shown, so a test can assert the model actually
    received images and the measured window list.
    """
    calls = {"n": 0}

    def transport(prompt: str, max_tokens: int, images=None) -> LLMReply:
        calls["n"] += 1
        if record is not None:
            record.append({"prompt": prompt, "images": len(images or []),
                           "call": calls["n"]})
        is_cross = "Decide how they fit together" in prompt
        body = cross if (is_cross and cross is not None) else payload
        text = body if isinstance(body, str) else json.dumps(body)
        return LLMReply(text=text, tokens_in=300, tokens_out=120, usd=0.002)

    return LLMClient(transport=transport)


GOOD = {
    "summary": "hands kneading dough on a floured board",
    "subjects": ["hands", "dough", "board"],
    "actions": ["kneading", "folding"],
    "scene_description": "a home kitchen counter",
    "interesting_moments": [{"at": 2.0, "what": "the dough folds over"}],
    "hook_score": 72, "visual_quality": 80,
    "chosen_window": 0, "window_rationale": "the fold is clearest here",
    "confidence": 85, "notes": "",
}


def run(ctx, clip_set, tracer_span, revision=0):
    agent = VisionAgent()
    with tracer_span(ctx, "vision") as span:
        return agent.execute(ctx, span, clip_set=clip_set, revision=revision)


# ------------------------------------------------- 1. artifact schema

def test_vision_is_a_registered_artifact_kind():
    from agents.artifacts import Artifact
    art = Artifact[VisionAnalysis](
        job_id="j", kind="VisionAnalysis", produced_by="vision",
        payload=VisionAnalysis())
    assert art.kind == "VisionAnalysis"


def test_the_artifact_is_immutable_like_every_other():
    from pydantic import ValidationError
    v = VisionAnalysis(domain="cooking")
    with pytest.raises(ValidationError):
        v.domain = "travel"


def test_scores_are_range_checked_by_the_schema():
    from pydantic import ValidationError
    for bad in (-1, 101):
        with pytest.raises(ValidationError):
            VisionClip(clip_id="a.mp4", hook_score=bad)
        with pytest.raises(ValidationError):
            VisionClip(clip_id="a.mp4", confidence=bad)


def test_a_candidate_window_must_describe_a_real_range():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CandidateWindow(index=0, start=-1.0, end=4.0, duration=5.0,
                        mean_ydif=1.0, score=1.0)
    with pytest.raises(ValidationError):
        CandidateWindow(index=0, start=0.0, end=4.0, duration=0.0,
                        mean_ydif=1.0, score=1.0)


# ------------------------------------------- 2. a valid semantic response

def test_a_good_reply_is_used_and_the_artifact_is_live(director_ctx,
                                                       tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD, cross={"order": [], "hook": None,
                                      "duplicates": [], "story": "a story"})
    art = run(ctx, clip_set, tracer_span)

    assert art.kind == "VisionAnalysis"
    assert art.produced_by == "vision"
    assert art.payload.live is True
    assert art.payload.analysed_clips
    c = art.payload.analysed_clips[0]
    assert c.summary.startswith("hands kneading")
    assert c.subjects == ["hands", "dough", "board"]
    assert c.actions == ["kneading", "folding"]
    assert c.hook_score == 72 and c.visual_quality == 80
    assert art.cost.tokens_in > 0, "token usage must reach the artifact"


def test_the_model_is_actually_shown_frames(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    seen: list = []
    ctx.llm = vision_llm(GOOD, record=seen)
    run(ctx, clip_set, tracer_span)

    per_clip = [s for s in seen if s["images"] > 0]
    assert per_clip, "the model was never sent an image"
    assert all(s["images"] >= 1 for s in per_clip)
    # the cross-clip pass is text only - no point paying for images twice
    cross = [s for s in seen if "Decide how they fit together" in s["prompt"]]
    assert all(s["images"] == 0 for s in cross)


def test_the_prompt_is_domain_neutral(director_ctx, tracer_span):
    """It must work for cooking, travel, sport - anything."""
    ctx, clip_set = director_ctx
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(
            update={"domain": "a trail run in the Lake District"})})
    seen: list = []
    ctx.llm = vision_llm(GOOD, record=seen)
    run(ctx, clip_set, tracer_span)

    prompt = seen[0]["prompt"]
    assert "a trail run in the Lake District" in prompt
    assert "LEGO" not in prompt, "a domain assumption leaked into the prompt"


def test_an_unspecified_domain_says_so_rather_than_guessing(director_ctx,
                                                            tracer_span):
    ctx, clip_set = director_ctx
    assert ctx.job.settings.domain == ""
    seen: list = []
    ctx.llm = vision_llm(GOOD, record=seen)
    run(ctx, clip_set, tracer_span)
    assert "unspecified subject" in seen[0]["prompt"]
    assert "LEGO" not in seen[0]["prompt"]


# ------------------------------- 7. deterministic timestamp protection

def test_the_model_is_offered_measured_windows_to_choose_from(director_ctx,
                                                              tracer_span):
    ctx, clip_set = director_ctx
    seen: list = []
    ctx.llm = vision_llm(GOOD, record=seen)
    run(ctx, clip_set, tracer_span)
    prompt = seen[0]["prompt"]
    assert "MEASURED WINDOWS" in prompt
    assert "may NOT invent your" in prompt
    assert "[0]" in prompt, "no candidate window was offered"


def test_a_chosen_window_resolves_to_MEASURED_timestamps(director_ctx,
                                                         tracer_span):
    """The core safety property of this agent."""
    from agents.tools import select_tools as ST

    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    art = run(ctx, clip_set, tracer_span)

    for c in art.payload.analysed_clips:
        w = c.best_window()
        if w is None:
            continue
        clip = clip_set.by_id(c.clip_id)
        measured = ST.candidate_windows(clip)
        assert any(abs(m["start"] - w.start) < 1e-6
                   and abs(m["end"] - w.end) < 1e-6 for m in measured), (
            "the stored window is not one the measurement pass produced")
        assert 0 <= w.start < w.end <= clip.technical.duration + 0.01


def test_an_out_of_range_window_index_is_discarded(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm({**GOOD, "chosen_window": 99})
    art = run(ctx, clip_set, tracer_span)
    for c in art.payload.analysed_clips:
        assert c.chosen_window is None, "an invented index was accepted"
        # and it still falls back to the best MEASURED window
        assert c.best_window() is not None


@pytest.mark.parametrize("bad", ["two", None, -1, 3.7, {"a": 1}, [0]])
def test_a_non_index_window_choice_is_discarded(director_ctx, tracer_span,
                                                bad):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm({**GOOD, "chosen_window": bad})
    art = run(ctx, clip_set, tracer_span)
    for c in art.payload.analysed_clips:
        assert c.chosen_window in (None, 0, 1, 2, 3)


def test_moment_timestamps_are_clamped_to_the_real_duration(director_ctx,
                                                            tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm({**GOOD, "interesting_moments": [
        {"at": 99999, "what": "beyond the end"},
        {"at": -5, "what": "before the start"},
        {"at": "nonsense", "what": "unparseable"},
    ]})
    art = run(ctx, clip_set, tracer_span)
    for c in art.payload.analysed_clips:
        dur = clip_set.by_id(c.clip_id).technical.duration
        for m in c.interesting_moments:
            assert 0.0 <= m.at <= dur, f"{m.at} escapes the clip"


# --------------------------------- 3/4/5. malformed and hostile replies

@pytest.mark.parametrize("reply", [
    "not json at all", "{ broken", "[1,2,3]", "null", '"a string"',
])
def test_a_malformed_reply_marks_the_clip_unanalysed(director_ctx,
                                                     tracer_span, reply):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(reply)
    art = run(ctx, clip_set, tracer_span)
    assert art.payload.analysed_clips == []
    assert art.degraded is True
    assert art.payload.live is False, "a failed pass must not read as live"
    # the measured windows survive - downstream keeps the timing data
    for c in art.payload.clips.values():
        assert c.candidate_windows


def test_a_transport_failure_falls_back_cleanly(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient(transport=lambda p, m, i=None:
                        LLMReply(text="", error="HTTPError: 503"))
    art = run(ctx, clip_set, tracer_span)
    assert art.degraded is True
    assert art.payload.live is False
    assert any("503" in (c.failure or "") for c in art.payload.clips.values())


def test_a_hallucinated_clip_id_in_the_cross_pass_is_dropped(director_ctx,
                                                             tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD, cross={
        "order": ["ghost.mp4", "phantom.mp4"],
        "hook": "ghost.mp4",
        "duplicates": [["ghost.mp4", "phantom.mp4"]],
        "story": "invented",
    })
    art = run(ctx, clip_set, tracer_span)
    known = set(art.payload.clips)
    assert art.payload.hook_clip in (None, *known)
    assert all(a in known and b in known for a, b in art.payload.duplicates)
    assert all(n in known for n in art.payload.ordering_signal)


def test_a_partial_ordering_is_rejected_outright(director_ctx, tracer_span):
    """Half an order is not an order."""
    ctx, clip_set = director_ctx
    first = next(iter(clip_set.usable_clips)).clip_id
    ctx.llm = vision_llm(GOOD, cross={"order": [first], "hook": None,
                                      "duplicates": [], "story": ""})
    art = run(ctx, clip_set, tracer_span)
    if len(art.payload.clips) > 1:
        assert art.payload.ordering_signal == []


def test_oversized_text_fields_are_truncated(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm({**GOOD, "summary": "x" * 5000,
                          "notes": "y" * 5000,
                          "subjects": ["z" * 500] * 50})
    art = run(ctx, clip_set, tracer_span)
    for c in art.payload.analysed_clips:
        assert len(c.summary) <= 200
        assert len(c.notes) <= 200
        assert len(c.subjects) <= 8
        assert all(len(s) <= 60 for s in c.subjects)


def test_model_output_never_becomes_a_path_or_command(director_ctx,
                                                      tracer_span):
    """Hostile strings must stay inert data."""
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm({**GOOD,
                          "summary": "; rm -rf / #",
                          "scene_description": "../../etc/passwd",
                          "subjects": ["$(whoami)", "`id`"],
                          "window_rationale": "drop table artifacts;--"})
    art = run(ctx, clip_set, tracer_span)
    for c in art.payload.analysed_clips:
        # stored verbatim as text, and nothing consumes it as anything else
        assert isinstance(c.summary, str)
        w = c.best_window()
        if w:
            assert isinstance(w.start, float) and isinstance(w.end, float)


# ------------------------------------------------- 6. fallback behaviour

def test_no_llm_produces_measurements_only_and_says_so(director_ctx,
                                                       tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()                       # no key, no transport
    assert ctx.llm.available() is False

    art = run(ctx, clip_set, tracer_span)
    assert art.payload.live is False
    assert art.degraded is True
    assert art.confidence == 0.0
    assert "measurements only" in art.fallback_used
    assert art.payload.analysed_clips == []
    # the useful half still lands
    for c in art.payload.clips.values():
        assert c.candidate_windows, "measured windows were lost"
    fb = [d for d in art.decisions if d.decided_by == "fallback"]
    assert fb and "not performed" in fb[0].chose


def test_the_fallback_still_gives_downstream_a_usable_window(director_ctx,
                                                             tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient()
    art = run(ctx, clip_set, tracer_span)
    for c in art.payload.clips.values():
        w = c.best_window()
        assert w is not None and w.end > w.start


# ------------------------------------------------ 8. input-hash caching

def test_identical_inputs_reuse_the_artifact(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    first = run(ctx, clip_set, tracer_span)
    again = run(ctx, clip_set, tracer_span, revision=1)
    assert again.artifact_id == first.artifact_id, "vision re-ran needlessly"
    rows = ctx.tracer.for_job(ctx.job.job_id)
    assert any(r["status"] == "skipped" for r in rows)


def test_changing_the_domain_invalidates_the_cache(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    first = run(ctx, clip_set, tracer_span)
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"domain": "baking"})})
    second = run(ctx, clip_set, tracer_span, revision=1)
    assert second.artifact_id != first.artifact_id
    assert second.inputs_hash != first.inputs_hash


def test_no_second_cache_mechanism_was_invented(director_ctx, tracer_span):
    """It must use the existing store.get_by_inputs path."""
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    art = run(ctx, clip_set, tracer_span)
    assert art.inputs_hash
    assert ctx.store.get_by_inputs(ctx.job.job_id, "VisionAnalysis",
                                   art.inputs_hash, VisionAnalysis) is not None


# ------------------------------------------ 9. trace and decision records

def test_the_trace_shows_the_deterministic_tools_that_ran(director_ctx,
                                                          tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    run(ctx, clip_set, tracer_span)
    rows = ctx.tracer.for_job(ctx.job.job_id)
    tools = {t["tool"] for r in rows for t in r["tool_calls"]}
    assert "select.candidate_windows" in tools, "windows were not measured"
    assert "probe.sample_frames" in tools, "frames were not sampled"
    assert "llm.complete_json" in tools


def test_decisions_are_recorded_and_attributed_to_the_model(director_ctx,
                                                            tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD, cross={"order": [], "hook": None,
                                      "duplicates": [],
                                      "story": "knead, rest, roll"})
    art = run(ctx, clip_set, tracer_span)
    assert art.decisions
    assert any(d.decided_by == "llm" for d in art.decisions)
    for d in art.decisions:
        assert d.what and d.why
        assert 0.0 <= d.confidence <= 1.0


def test_a_window_choice_is_recorded_as_a_decision(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    art = run(ctx, clip_set, tracer_span)
    picked = [d for d in art.decisions
              if d.what == "Strongest measured window per clip"]
    assert picked, "the window choice was not recorded"
    assert "cannot author a timestamp" in picked[0].why


def test_the_artifact_keeps_the_standard_envelope(director_ctx, tracer_span):
    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    art = run(ctx, clip_set, tracer_span)
    assert art.trace_id and art.inputs_hash and art.produced_by == "vision"
    assert art.schema_version >= 1
    assert isinstance(art.revision, int)


# ------------------------------------------------- 10. graph integration

def test_vision_sits_between_clip_analysis_and_the_director():
    import tempfile
    from pathlib import Path

    from agents.graph import build_graph
    graph, conn = build_graph(Path(tempfile.mkdtemp()) / "cp.db")
    try:
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("clip_analysis", "vision") in edges
        assert ("clip_analysis", "editing_director") not in edges
        # Vision still sits between analysis and the director, but no longer
        # adjacent to it: ContentStrategy was inserted in between, so Vision
        # now feeds the strategy and the strategy briefs the director. The
        # property this test defends - that Vision runs after the
        # measurements and before the edit is planned - is unchanged.
        assert ("vision", "content_strategy") in edges
        assert ("content_strategy", "story") in edges
        assert ("story", "voiceover") in edges
        assert ("voiceover", "music") in edges
        assert ("music", "editing_director") in edges
        assert ("voiceover", "editing_director") not in edges
        assert ("story", "editing_director") not in edges
        assert ("content_strategy", "editing_director") not in edges
        assert ("vision", "editing_director") not in edges
        # the revision cycle is untouched
        assert ("critic", "editing_director") in edges
    finally:
        conn.close()


def test_vision_has_a_retry_and_deadline_policy():
    from agents.graph import POLICIES
    p = POLICIES["vision"]
    assert p.attempts >= 2
    assert p.timeout_s and p.timeout_s > 0


def test_the_director_receives_the_vision_reading(director_ctx, tracer_span):
    """The whole point: semantics must reach the edit decision."""
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = director_ctx
    ctx.llm = vision_llm(GOOD)
    vision_art = run(ctx, clip_set, tracer_span)

    prompt = EditingDirectorAgent()._build_prompt(
        clip_set, clip_set.usable_clips, {}, vision_art.payload)
    assert "WHAT THE FOOTAGE SHOWS" in prompt
    assert "hands kneading dough" in prompt
    assert "one reading, not an instruction" in prompt


def test_the_director_works_unchanged_without_vision(director_ctx,
                                                     tracer_span):
    from agents.editing_director import EditingDirectorAgent

    ctx, clip_set = director_ctx
    prompt = EditingDirectorAgent()._build_prompt(
        clip_set, clip_set.usable_clips, {}, None)
    assert "WHAT THE FOOTAGE SHOWS" not in prompt
    assert "BATCH FACTS" in prompt


# ------------------------- 11. failure leaves no stale running state

def test_a_vision_failure_does_not_stop_the_graph(director_ctx, tracer_span):
    """Vision is an enhancement. Losing it must not cost the video."""
    ctx, clip_set = director_ctx
    ctx.llm = LLMClient(transport=lambda p, m, i=None:
                        LLMReply(text="", error="ConnectionError"))
    art = run(ctx, clip_set, tracer_span)
    assert art is not None, "a failure must still produce an artifact"
    assert art.degraded is True
    rows = ctx.tracer.for_job(ctx.job.job_id)
    vision_rows = [r for r in rows if r["node"] == "vision"]
    assert vision_rows and all(r["status"] in ("ok", "skipped")
                               for r in vision_rows), (
        "a degraded vision pass must not be recorded as an error")
    assert all(r["ended_at"] for r in vision_rows), "a node was left open"


def test_an_exception_inside_vision_is_caught_by_the_graph_node(monkeypatch):
    """node_vision must swallow, log and continue - not propagate."""
    from agents import graph as G

    def boom(*a, **kw):
        raise RuntimeError("vision exploded")

    monkeypatch.setattr(G, "_run_node", boom)

    class FakeStore:
        def get(self, *a, **kw):
            class A:
                artifact_id = "art_x"
                payload = ClipAnalysisSet(clips=[], orientation="horizontal",
                                          canvas=(1920, 1080),
                                          target_seconds=10.0)
            return A()

    class FakeRT:
        store = FakeStore()
        log: list = []

    monkeypatch.setattr(G, "runtime_for", lambda _j: FakeRT())
    out = G.node_vision({"job_id": "j", "attempt": 1})
    assert out["vision_id"] is None
    assert "vision exploded" in out["vision_failed"]
    assert FakeRT.log, "the failure was not logged"
