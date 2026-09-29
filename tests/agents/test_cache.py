"""
Segment cache reuse - the thing that makes a revision cheap rather than ruinous.

Real renders throughout. The property under test is that a revision which
changes ONE segment re-renders that segment and reuses the other, and that a
stale segment is never silently served.
"""

import shutil
from pathlib import Path

import pytest

from agents.artifacts import EditPlan, JobSettings, Segment
from agents.base import AgentContext
from agents.clip_analysis import ClipAnalysisAgent
from agents.rendering import RenderingAgent
from agents.runner import make_job
from agents.store import ArtifactStore, shared_store
from agents.tools import render_tools as RT
from agents.tools.llm_tools import LLMClient
from agents.trace import TraceRecorder, shared_tracer

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


@pytest.fixture
def render_ctx(agent_root, calm_clip, lively_clip, pool):
    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    job = make_job(src=src, job_id="cachejob", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=10.0))
    outputs = Path(job.build_dir).resolve().parent
    ctx = AgentContext(job=job, store=shared_store(outputs=outputs),
                       tracer=shared_tracer(outputs=outputs),
                       pool=pool, llm=LLMClient())
    with ctx.tracer.node(job.job_id, "clip_analysis", "clip_analysis") as span:
        ClipAnalysisAgent().execute(ctx, span, revision=0)
    return ctx


def _plan(canvas, first_window=(0.0, 4.0), second_window=(0.0, 4.0)):
    return EditPlan(
        segments=[
            Segment(index=1, clip_id=LIVELY, start=first_window[0],
                    end=first_window[1],
                    duration=round(first_window[1] - first_window[0], 2),
                    orientation="horizontal"),
            Segment(index=2, clip_id=CALM, start=second_window[0],
                    end=second_window[1],
                    duration=round(second_window[1] - second_window[0], 2),
                    orientation="horizontal"),
        ],
        total_duration=7.6, target_seconds=10.0, canvas=canvas)


def _render(ctx, plan, attempt):
    agent = RenderingAgent()
    rev = ctx.store.next_revision(ctx.job.job_id, "RenderResult")
    with ctx.tracer.node(ctx.job.job_id, "render", "rendering",
                         attempt=attempt) as span:
        return agent.execute(ctx, span, plan=plan, attempt=attempt,
                             revision=rev)


# ------------------------------------------------------------ first pass

@pytest.mark.slow
def test_a_first_render_has_no_cache_hits(render_ctx):
    ctx = render_ctx
    art = _render(ctx, _plan((1280, 720)), 1)
    assert art.payload.cache_hits == 0
    assert all(s.rendered for s in art.payload.segments)
    assert Path(art.payload.video_path).exists()


# --------------------------------------------------- the reuse that matters

@pytest.mark.slow
def test_changing_only_segment_one_reuses_segment_two(render_ctx):
    ctx = render_ctx
    first = _render(ctx, _plan((1280, 720), (0.0, 4.0), (0.0, 4.0)), 1)
    assert first.payload.cache_hits == 0

    # a revision that moves only the opening window
    second = _render(ctx, _plan((1280, 720), (1.6, 5.6), (0.0, 4.0)), 2)

    hits = {s.index: s.cache_hit for s in second.payload.segments}
    assert hits[1] is False, "segment 1 changed and must be re-rendered"
    assert hits[2] is True, "segment 2 is identical and must be reused"
    assert second.payload.cache_hits == 1


@pytest.mark.slow
def test_an_identical_replan_reuses_everything(render_ctx):
    ctx = render_ctx
    _render(ctx, _plan((1280, 720)), 1)
    again = _render(ctx, _plan((1280, 720)), 2)
    assert again.payload.cache_hits == 2
    assert not any(s.rendered for s in again.payload.segments)


@pytest.mark.slow
def test_the_reused_segment_file_is_byte_identical(render_ctx):
    """Reuse must mean the same file, not a re-encode that happens to match."""
    import hashlib

    ctx = render_ctx
    _render(ctx, _plan((1280, 720), (0.0, 4.0), (0.0, 4.0)), 1)
    p2 = Path(ctx.job.build_dir) / "parts" / "p02.mp4"
    before = hashlib.sha256(p2.read_bytes()).hexdigest()

    _render(ctx, _plan((1280, 720), (1.6, 5.6), (0.0, 4.0)), 2)
    assert hashlib.sha256(p2.read_bytes()).hexdigest() == before


@pytest.mark.slow
def test_a_stale_segment_is_never_served(render_ctx):
    """The hazard the content key exists to prevent.

    The existing stage_cut caches on the file path alone, so without a content
    key a changed segment 1 would be served from the previous render and the
    revision would silently do nothing.
    """
    import hashlib

    ctx = render_ctx
    first = _render(ctx, _plan((1280, 720), (0.0, 4.0), (0.0, 4.0)), 1)
    p1 = Path(ctx.job.build_dir) / "parts" / "p01.mp4"
    stale = hashlib.sha256(p1.read_bytes()).hexdigest()

    second = _render(ctx, _plan((1280, 720), (2.0, 6.0), (0.0, 4.0)), 2)
    assert hashlib.sha256(p1.read_bytes()).hexdigest() != stale, (
        "segment 1 changed but the stale render was served")

    # both attempts survive as separate files, which is what lets the critic
    # compare them and ship the better one
    a1, a2 = Path(first.payload.video_path), Path(second.payload.video_path)
    assert a1 != a2 and a1.exists() and a2.exists()
    assert hashlib.sha256(a1.read_bytes()).hexdigest() != \
           hashlib.sha256(a2.read_bytes()).hexdigest()


@pytest.mark.slow
def test_the_joined_file_is_invalidated_when_a_segment_changes(render_ctx):
    """stage_join skips when the total duration matches, so two different
    cuts of the same length would reuse a stale join."""
    ctx = render_ctx
    _render(ctx, _plan((1280, 720), (0.0, 4.0), (0.0, 4.0)), 1)
    joined = Path(ctx.job.build_dir) / "_joined.mp4"
    before = joined.stat().st_mtime

    # same durations, different content - the duration guard alone would skip
    _render(ctx, _plan((1280, 720), (2.0, 6.0), (0.0, 4.0)), 2)
    assert joined.stat().st_mtime != before, "_joined.mp4 was not rebuilt"


# --------------------------------------------------------- the key itself

def test_plan_cache_reports_a_miss_when_no_key_was_written(tmp_path):
    """A part rendered by the legacy path carries no key and is not trusted."""
    from agents.artifacts import RenderSpec, SegmentRenderSpec

    build = tmp_path / "build"
    (build / "parts").mkdir(parents=True)
    spec = RenderSpec(
        job_id="j", build_dir=str(build), src_dir=str(tmp_path),
        canvas=(1280, 720), crf=24,
        segments=[SegmentRenderSpec(index=1, clip_id="a.mp4", start=0, end=4,
                                    duration=4, caption="", cache_key="k1",
                                    orientation="horizontal")])
    (build / "parts" / "p01.mp4").write_bytes(b"not a real video")
    assert RT.plan_cache(spec) == {1: False}


def test_invalidate_stale_removes_the_part_and_the_join(tmp_path):
    from agents.artifacts import RenderSpec, SegmentRenderSpec

    build = tmp_path / "build"
    (build / "parts").mkdir(parents=True)
    (build / "parts" / "p01.mp4").write_bytes(b"x")
    (build / "parts" / "p01.key").write_text("old-key")
    (build / "_joined.mp4").write_bytes(b"y")

    spec = RenderSpec(
        job_id="j", build_dir=str(build), src_dir=str(tmp_path),
        canvas=(1280, 720), crf=24,
        segments=[SegmentRenderSpec(index=1, clip_id="a.mp4", start=0, end=4,
                                    duration=4, caption="", cache_key="new-key",
                                    orientation="horizontal")])
    reused, stale = RT.invalidate_stale(spec)
    assert reused == [] and stale == [1]
    assert not (build / "parts" / "p01.mp4").exists()
    assert not (build / "_joined.mp4").exists()
