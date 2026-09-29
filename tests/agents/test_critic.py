"""
Critic - against REAL rendered MP4 files.

Nothing is mocked. Two real videos are built with ffmpeg: one that opens on
calm material and one that opens on lively material. The critic measures the
files themselves - decoding them, counting frames, and running signalstats on
the output - and the assertions are about what it measured.
"""

import subprocess

import pytest

from agents.artifacts import (ClipAnalysisSet, CriticResult, EditPlan,
                              RenderResult, RenderTechnical, Segment)
from agents.critic import CriticAgent
from agents.tools import verify_tools as VT
from conftest import FFMPEG, make_clip_analysis

CANVAS = (320, 240)


def _render_pair(out, first, second, fps=30):
    """Concatenate two clips into one real MP4 with an audio stream.

    Mirrors what extras.master_stage produces (h264 + aac, +faststart), so the
    critic is looking at the same shape of file a real build emits.
    """
    if out.exists():
        return out
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-y",
         "-i", str(first), "-i", str(second),
         "-f", "lavfi", "-i",
         "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-filter_complex",
         f"[0:v]scale={CANVAS[0]}:{CANVAS[1]},fps={fps},setsar=1[a];"
         f"[1:v]scale={CANVAS[0]}:{CANVAS[1]},fps={fps},setsar=1[b];"
         f"[a][b]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-shortest",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", str(out)],
        check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def weak_hook_video(media_root, calm_clip, lively_clip):
    """Opens calm, ends lively - a genuinely weak hook."""
    return _render_pair(media_root / "weak_hook.mp4", calm_clip, lively_clip)


@pytest.fixture(scope="session")
def strong_hook_video(media_root, calm_clip, lively_clip):
    """Opens lively - a genuinely strong hook."""
    return _render_pair(media_root / "strong_hook.mp4", lively_clip, calm_clip)


def _artifacts_for(path, order):
    probe = VT.probe_final(path)
    segs = [
        Segment(index=1, clip_id=order[0], start=0.0, end=6.0, duration=6.0,
                orientation="horizontal"),
        Segment(index=2, clip_id=order[1], start=0.0, end=6.0, duration=6.0,
                orientation="horizontal"),
    ]
    plan = EditPlan(segments=segs, total_duration=probe["duration"],
                    target_seconds=12.0, canvas=CANVAS)
    render = RenderResult(
        attempt=1, video_path=str(path),
        technical=RenderTechnical(
            duration=probe["duration"], width=probe["width"],
            height=probe["height"], fps=probe["fps"],
            video_codec=probe["video_codec"],
            audio_codec=probe["audio_codec"]))
    clip_set = ClipAnalysisSet(
        clips=[make_clip_analysis("01_calm.mp4", 6.0, 1.0),
               make_clip_analysis("02_lively.mp4", 6.0, 6.0)],
        orientation="horizontal", canvas=CANVAS, target_seconds=12.0)
    return render, plan, clip_set


def _judge(ctx, span_maker, path, order, **kw):
    render, plan, clip_set = _artifacts_for(path, order)
    agent = CriticAgent()
    # artifacts are immutable, so a second judgement in the same test writes
    # the next revision rather than overwriting the first
    revision = ctx.store.next_revision(ctx.job.job_id, "CriticResult")
    with span_maker(ctx, "critic") as span:
        return agent.execute(
            ctx, span, render=render, plan=plan, clip_set=clip_set,
            attempt=kw.pop("attempt", 1),
            revision_count=kw.pop("revision_count", 0),
            max_revisions=kw.pop("max_revisions", 2),
            revision=revision, **kw)


# --------------------------------------------------------- it looks at the file

def test_critic_measures_the_actual_file_not_the_plan(director_ctx,
                                                       tracer_span,
                                                       weak_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, weak_hook_video,
                    ["01_calm.mp4", "02_lively.mp4"])
    rows = ctx.tracer.for_job(ctx.job.job_id)
    tools = {t["tool"] for r in rows for t in r["tool_calls"]}
    # every one of these decodes the rendered mp4
    assert "verify.decode_check" in tools
    assert "verify.frame_count" in tools
    assert "verify.black_frames" in tools
    assert "verify.output_motion_profile" in tools
    assert art.payload.technical, "no technical checks recorded"


def test_all_seven_technical_checks_are_present(director_ctx, tracer_span,
                                                 strong_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, strong_hook_video,
                    ["02_lively.mp4", "01_calm.mp4"])
    names = {t.check for t in art.payload.technical}
    assert names == {"decode_valid", "frame_count", "resolution", "fps",
                     "duration", "black_frames", "audio_stream", "loudness"}


def test_a_sound_render_passes_its_technical_checks(director_ctx, tracer_span,
                                                     strong_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, strong_hook_video,
                    ["02_lively.mp4", "01_calm.mp4"])
    failed = [t.check for t in art.payload.technical
              if not t.passed and t.severity == "blocking"]
    assert not failed, f"blocking technical failures: {failed}"


def test_frame_count_is_counted_not_trusted(director_ctx, tracer_span,
                                             strong_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, strong_hook_video,
                    ["02_lively.mp4", "01_calm.mp4"])
    fc = next(t for t in art.payload.technical if t.check == "frame_count")
    assert "decoded" in str(fc.value) and fc.passed


# ------------------------------------------------------------ the verdicts

def test_a_strong_hook_passes(director_ctx, tracer_span, strong_hook_video):
    ctx, _ = director_ctx
    art, request = _judge(ctx, tracer_span, strong_hook_video,
                          ["02_lively.mp4", "01_calm.mp4"])
    assert art.payload.verdict == "PASS", art.payload.summary
    assert art.payload.overall_score >= art.payload.pass_threshold
    assert request is None


def test_a_weak_hook_is_revised(director_ctx, tracer_span, weak_hook_video):
    ctx, _ = director_ctx
    art, request = _judge(ctx, tracer_span, weak_hook_video,
                          ["01_calm.mp4", "02_lively.mp4"])
    assert art.payload.verdict == "REVISE", art.payload.summary
    assert art.payload.overall_score < art.payload.pass_threshold
    assert request is not None


def test_the_hook_score_is_evidenced_by_real_measurement(director_ctx,
                                                          tracer_span,
                                                          weak_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, weak_hook_video,
                    ["01_calm.mp4", "02_lively.mp4"])
    hook = next(c for c in art.payload.creative
                if c.dimension == "first_3s_hook")
    assert hook.measured
    assert "RENDERED file" in hook.evidence
    assert "mean YDIF" in hook.evidence
    assert 0.0 <= hook.score <= 100.0
    assert 0.0 < hook.confidence <= 1.0


def test_weak_and_strong_actually_score_differently(director_ctx, tracer_span,
                                                     weak_hook_video,
                                                     strong_hook_video):
    """The premise of the whole loop, measured rather than assumed."""
    ctx, _ = director_ctx
    weak, _ = _judge(ctx, tracer_span, weak_hook_video,
                     ["01_calm.mp4", "02_lively.mp4"])
    strong, _ = _judge(ctx, tracer_span, strong_hook_video,
                       ["02_lively.mp4", "01_calm.mp4"])
    assert strong.payload.overall_score > weak.payload.overall_score + 20


# ---------------------------------------------------------- the revision

def test_the_revision_request_is_machine_readable_and_routed(director_ctx,
                                                              tracer_span,
                                                              weak_hook_video):
    ctx, _ = director_ctx
    _, request = _judge(ctx, tracer_span, weak_hook_video,
                        ["01_calm.mp4", "02_lively.mp4"])
    assert request.target_agent == "editing_director"
    assert request.severity == "high"
    assert "opening_min_mean_ydif" in request.constraints
    assert "opening_max_duration" in request.constraints
    assert request.constraints["opening_must_be_liveliest"] is True
    # it names a clip that actually exists, never an invented one
    if "opening_clip" in request.constraints:
        assert request.constraints["opening_clip"] == "02_lively.mp4"
    assert request.issue and request.rationale


def test_max_revisions_forces_escalate(director_ctx, tracer_span,
                                        weak_hook_video):
    ctx, _ = director_ctx
    art, request = _judge(ctx, tracer_span, weak_hook_video,
                          ["01_calm.mp4", "02_lively.mp4"],
                          revision_count=2, max_revisions=2)
    assert art.payload.verdict == "ESCALATE"
    assert request is None
    assert "maximum" in art.payload.summary


def test_comparison_to_the_previous_attempt_is_recorded(director_ctx,
                                                         tracer_span,
                                                         weak_hook_video,
                                                         strong_hook_video):
    ctx, _ = director_ctx
    first, _ = _judge(ctx, tracer_span, weak_hook_video,
                      ["01_calm.mp4", "02_lively.mp4"])
    second, _ = _judge(ctx, tracer_span, strong_hook_video,
                       ["02_lively.mp4", "01_calm.mp4"], attempt=2,
                       revision_count=1, previous=first.payload)
    cmp = second.payload.compared_to_previous
    assert cmp is not None
    assert cmp["previous_attempt"] == 1
    assert cmp["better"] is True
    assert cmp["delta"] > 0


# ------------------------------------------------------------- honesty

def test_unscored_creative_dimensions_are_declared_not_invented(
        director_ctx, tracer_span, strong_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, strong_hook_video,
                    ["02_lively.mp4", "01_calm.mp4"])
    assert art.payload.unreviewed_creative is True
    assert len(art.payload.creative) == 1, (
        "Phase 1 measures one dimension; the other nine must be absent, "
        "not present with invented scores")
    coverage = next(d for d in art.decisions if d.what == "Creative coverage")
    assert "1 of 10" in coverage.chose


def test_loudness_is_measured_but_not_a_gate_while_audio_is_silent(
        director_ctx, tracer_span, strong_hook_video):
    ctx, _ = director_ctx
    art, _ = _judge(ctx, tracer_span, strong_hook_video,
                    ["02_lively.mp4", "01_calm.mp4"])
    lo = next(t for t in art.payload.technical if t.check == "loudness")
    assert lo.severity == "low"
    assert lo.passed
    assert "Phase 4" in lo.threshold


def test_hook_scoring_handles_an_unmeasurable_file():
    """A file signalstats cannot read must score 0 with zero confidence,
    not raise and not pass."""
    score, evidence, conf = VT.score_hook({"measured": False, "why": "boom"})
    assert score == 0.0 and conf == 0.0 and "boom" in evidence


def test_hook_scoring_handles_a_wholly_static_video():
    score, evidence, conf = VT.score_hook({
        "measured": True, "overall_median": 0.0, "hook_mean": 0.0,
        "hook_seconds": 3.0, "rate": 10.0, "samples_in_hook": 30})
    assert score == 0.0
    assert "static" in evidence
