"""
Rendering Agent.

A deterministic service with zero creative authority. It compiles the typed
EditPlan into a RenderSpec, projects that onto manifest.json (the interface the
existing stages read), and runs the UNMODIFIED pipeline:

    build_video.py cut  ->  build_video.py join  ->  extras.master_stage

Every one of those runs on the bounded media pool, never inside the graph node.

Its one real decision is which segments may be reused. That is not a rename of
`good_video()`: the existing cache is keyed on the file path, so after a
revision that changes segment 1's in/out points it would happily serve the
stale render. Correct reuse needs a content key, which is computed here and
enforced before `cut` runs.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path

from .artifacts import (Artifact, Cost, Decision, EditPlan, RenderResult,
                        RenderSpec, RenderTechnical, SegmentRenderInfo)
from .base import Agent, AgentContext
from .tools import render_tools as RT
from .tools import verify_tools as VT
from .tools.pipeline_bridge import adapter_mod
from .trace import TraceSpan


@contextlib.contextmanager
def _outputs_root(build_dir: str):
    """Point the existing adapter at this job's outputs root.

    `extras.master_stage` resolves its own paths through `adapter.build_dir`,
    which reads adapter's module-level OUTPUTS. Tests and alternate roots need
    that to follow the job rather than the process-wide default, and this is
    the least invasive way to do it - adapter.py itself is not touched.
    """
    parent = Path(build_dir).resolve().parent
    previous = adapter_mod.OUTPUTS
    adapter_mod.OUTPUTS = parent
    try:
        yield
    finally:
        adapter_mod.OUTPUTS = previous


class RenderingAgent(Agent):
    id = "rendering"
    produces = "RenderResult"
    uses_llm = False
    agency = "deterministic"

    def execute(self, ctx: AgentContext, span: TraceSpan, plan: EditPlan,
                attempt: int, revision: int = 0,
                edit_plan_id: str = "",
                stage_timeout_s: float | None = None
                ) -> Artifact[RenderResult]:
        started = time.time()
        job = ctx.job

        spec = RT.compile_render_spec(
            job_id=job.job_id, build_dir=job.build_dir, src_dir=job.src_dir,
            plan=plan, quality=job.settings.quality,
            xfade_duration=job.settings.transition_duration,
            title=job.settings.title)
        inputs_hash = self.hash_inputs(spec.model_dump(mode="json"))

        # ---- projection: typed plan -> the interface ffmpeg reads ---------
        span.tool("render.project_manifest", segments=len(spec.segments))
        RT.project_manifest(spec)

        # ---- cache: what may honestly be reused --------------------------
        reused, stale = RT.invalidate_stale(spec)
        span.tool("render.plan_cache", reused=reused, invalidated=stale)
        span.cache(len(reused))
        ctx.emit("cache: %d segment(s) reused (%s), %d re-rendering (%s)"
                 % (len(reused), reused or "-", len(stale), stale or "-"))

        decisions = [Decision(
            what="Segment cache",
            chose=f"{len(reused)} reused, {len(stale)} re-rendered",
            why=("a segment is reused only when its rendered file exists, "
                 "passes the pipeline's own good_video check, AND its stored "
                 "content key still matches the clip, in-point, out-point and "
                 "caption it is meant to represent; anything else is deleted "
                 "so the existing cut stage re-renders it"),
            decided_by="policy", confidence=1.0)]

        warnings: list[str] = []

        # ---- the existing render path, on the pool -----------------------
        seg_started = time.time()
        span.tool("pipeline.cut", stage="cut", env=spec.env_overrides())
        rc, out = ctx.pool.run(RT.run_cut, spec, ctx.emit,
                               pool=ctx.pool, timeout_s=stage_timeout_s)
        if rc != 0:
            raise RuntimeError(f"cut failed (exit {rc}): {out[-500:]}")
        RT.write_cache_keys(spec)
        seg_wall = round(time.time() - seg_started, 2)

        span.tool("pipeline.join", stage="join", cuts=len(spec.segments) - 1)
        rc, out = ctx.pool.run(RT.run_join, spec, ctx.emit,
                               pool=ctx.pool, timeout_s=stage_timeout_s)
        if rc != 0:
            raise RuntimeError(f"join failed (exit {rc}): {out[-500:]}")

        span.tool("extras.master_stage", stage="master")
        with _outputs_root(job.build_dir):
            meta = ctx.pool.run(RT.run_master, job.job_id, spec, ctx.emit)

        canonical = meta.get("path") or ""
        if not canonical or not Path(canonical).exists():
            raise RuntimeError("master stage reported no output file on disk")

        # ---- archive this attempt ----------------------------------------
        # master_stage always writes the same filename, so without this the
        # next attempt overwrites this one and the critic loses the ability to
        # compare two renders - or to ship attempt 1 when attempt 2 is worse.
        attempts_dir = Path(job.build_dir) / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        archived = attempts_dir / f"attempt_{attempt:02d}.mp4"
        shutil.copy2(canonical, archived)
        video_path = str(archived)
        span.tool("render.archive_attempt", attempt=attempt,
                  path=archived.name)

        # ---- what was actually produced ----------------------------------
        probe = VT.probe_final(video_path)
        technical = RenderTechnical(
            duration=probe["duration"],
            frames=meta.get("frames"),
            frames_expected=meta.get("frames_expected"),
            frames_ok=bool(meta.get("frames_ok")),
            width=probe["width"], height=probe["height"], fps=probe["fps"],
            video_codec=probe["video_codec"], audio_codec=probe["audio_codec"],
            size=probe["size"], lufs=meta.get("lufs"),
            black_frames=bool(meta.get("black_frames")),
        )
        span.tool("verify.probe_final", duration=probe["duration"],
                  resolution=f"{probe['width']}x{probe['height']}")

        seg_info = [
            SegmentRenderInfo(
                index=s.index,
                path=str(Path(job.build_dir) / "parts" / f"p{s.index:02d}.mp4"),
                cache_hit=s.index in reused,
                rendered=s.index in stale,
                wall_s=0.0 if s.index in reused else seg_wall / max(len(stale), 1),
            )
            for s in spec.segments
        ]

        result = RenderResult(
            attempt=attempt, video_path=video_path,
            canonical_path=canonical,
            thumbnail_path=(str(Path(job.build_dir) / meta["thumbnail"])
                            if meta.get("thumbnail") else None),
            technical=technical, segments=seg_info,
            wall_s=round(time.time() - started, 2), warnings=warnings)

        ctx.emit("rendered attempt %d: %s  %.2fs  %sx%s  frames=%s"
                 % (attempt, Path(video_path).name, technical.duration,
                    technical.width, technical.height, technical.frames))

        return self.wrap(ctx, result, revision=revision,
                         inputs_hash=inputs_hash, span=span,
                         decisions=decisions, confidence=1.0,
                         cost=Cost(wall_s=result.wall_s))


__all__ = ["RenderingAgent"]
