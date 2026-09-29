"""
ClipAnalysis Agent.

Establishes ground truth: what footage physically exists and how much movement
is in it. It wraps the existing `probe` and `measure` stages rather than
reimplementing them, so the YDIF measurement, the freezedetect pass, the iOS
duplicate rule and the orientation choice are byte-for-byte production code.

WHAT MAKES THIS AN AGENT RATHER THAN A RENAMED FUNCTION
-------------------------------------------------------
`stage_measure` measures everything identically and reports to stdout. This
agent additionally:

  * decides whether each clip is USABLE, per clip, with a confidence, using
    the pipeline's own constants - a judgement the current system makes
    invisibly inside stage_plan and only prints
  * decides whether the whole batch's motion signal is trustworthy enough for
    motion-weighted budgeting, and says so when it is not
  * flags clips whose signal is AMBIGUOUS (lots of frames changing, but very
    little total change - a talking head, a screen recording, a locked-off
    product shot) for semantic escalation, which is what the Vision Agent will
    consume in Phase 3

It is classified `hybrid` and in Phase 1 makes zero LLM calls. That is stated
rather than hidden.
"""

from __future__ import annotations

import time

from .artifacts import (Artifact, ClipAnalysisSet, Cost, Decision,
                        content_hash)
from .base import Agent, AgentContext
from .tools import analysis_tools as AT
from .tools.pipeline_bridge import build_video
from .trace import TraceSpan

# A clip where nearly every frame differs but the total change is tiny is not
# dead footage - it is a small subject on a still background. Motion alone
# cannot tell that apart from a genuinely dull shot, so it is flagged for the
# Vision Agent instead of being judged here.
AMBIGUOUS_LIVE_SHARE = 0.80
AMBIGUOUS_MEAN_YDIF = 1.0


class ClipAnalysisAgent(Agent):
    id = "clip_analysis"
    produces = "ClipAnalysis"
    uses_llm = False
    agency = "hybrid"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                revision: int = 0) -> Artifact[ClipAnalysisSet]:
        started = time.time()
        job = ctx.job
        settings = job.settings

        inputs_hash = self._inputs_hash(ctx)

        # Cache: identical clips and identical orientation means the existing
        # analysis still holds. A revision re-plans the edit; it does not
        # re-measure the footage.
        cached = ctx.store.get_by_inputs(job.job_id, "ClipAnalysis",
                                         inputs_hash, ClipAnalysisSet)
        if cached is not None:
            span.note("inputs unchanged - reusing ClipAnalysis "
                      f"rev{cached.revision}")
            span.cache(len(cached.payload.clips))
            span.skipped("cache hit")
            return cached

        decisions: list[Decision] = []

        # ---- probe: the existing stage, unmodified, on the pool -----------
        span.tool("pipeline.probe", stage="probe", src=job.src_dir)
        rc, out = ctx.pool.run(
            AT.run_pipeline_stage, "probe", job.build_dir, job.src_dir,
            {"AT_CANVAS": "%dx%d" % AT.canvas_for(
                settings.orientation if settings.orientation != "auto"
                else "horizontal", settings.resolution)},
            ["--orient", settings.orientation], ctx.emit,
            pool=ctx.pool, job_id=job.job_id)
        if rc != 0:
            raise RuntimeError(f"probe failed (exit {rc}): {out[-400:]}")

        manifest = AT.read_manifest(job.build_dir) or {}
        n_clips = len(manifest.get("clips") or [])
        if not n_clips:
            raise RuntimeError("probe found no readable clips")

        # ---- measure: the existing stage, unmodified, on the pool ---------
        span.tool("pipeline.measure", stage="measure", clips=n_clips)
        rc, out = ctx.pool.run(AT.run_pipeline_stage, "measure",
                               job.build_dir, None, None, None, ctx.emit,
                               pool=ctx.pool, job_id=job.job_id)
        if rc != 0:
            # Measurement failing is survivable: ffprobe metadata still
            # exists, so the build continues with duration-proportional
            # budgeting instead of motion-weighted, and says so.
            span.fallback("probe-only (signalstats unavailable)")
            decisions.append(Decision(
                what="Motion measurement",
                chose="skipped - ffprobe metadata only",
                why=(f"the measure stage exited {rc}, so no YDIF series "
                     f"exists; the editor will budget by duration rather "
                     f"than by motion and its confidence is reduced"),
                decided_by="fallback", confidence=0.3))

        manifest = AT.read_manifest(job.build_dir) or manifest
        orientation = manifest.get("orientation") or "horizontal"
        canvas = AT.canvas_for(orientation, settings.resolution)

        clip_set = AT.manifest_to_clip_set(
            manifest, job.src_dir,
            target_seconds=settings.target_seconds or manifest.get("target"))
        clip_set = clip_set.model_copy(update={"canvas": canvas})

        span.tool("probe.ffprobe_json", clips=len(clip_set.clips))
        span.tool("motion.ydif_series",
                  measured=sum(1 for c in clip_set.clips if c.motion.measured))
        span.tool("motion.freeze_runs",
                  total_runs=sum(len(c.motion.freeze_runs)
                                 for c in clip_set.clips))

        decisions.extend(self._decisions(clip_set, manifest))
        confidence = self._confidence(clip_set)

        for c in clip_set.clips:
            ctx.emit("  %-28s mean %6.2f  live %5.1f%%  frozen %5.1f%%  %s"
                     % (c.clip_id, c.motion.mean_ydif,
                        100 * c.motion.live_share, 100 * c.motion.freeze_share,
                        "USABLE" if c.usability.usable else
                        f"REJECTED - {c.usability.reason}"))

        return self.wrap(
            ctx, clip_set, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=decisions, confidence=confidence,
            degraded=any(not c.motion.measured for c in clip_set.clips),
            fallback_used=span.fallback_used,
            cost=Cost(wall_s=round(time.time() - started, 2)))

    # ------------------------------------------------------------ helpers

    def _inputs_hash(self, ctx: AgentContext) -> str:
        """Hash the actual source files, not just their names.

        Name plus size plus mtime is enough to notice a clip being swapped for
        a different file of the same name, which a name-only hash would miss
        and would then serve a stale analysis for.
        """
        from pathlib import Path
        src = Path(ctx.job.src_dir)
        files = []
        if src.exists():
            for p in sorted(src.iterdir()):
                if p.suffix.lower() in build_video.VIDEO_EXT:
                    st = p.stat()
                    files.append([p.name, st.st_size, round(st.st_mtime, 2)])
        return content_hash({
            "files": files,
            "orientation": ctx.job.settings.orientation,
            "resolution": ctx.job.settings.resolution,
        })

    def _decisions(self, clip_set: ClipAnalysisSet,
                   manifest: dict) -> list[Decision]:
        out: list[Decision] = []
        usable = clip_set.usable_clips
        rejected = [c for c in clip_set.clips if not c.usability.usable]

        out.append(Decision(
            what="Clips readable",
            chose=f"{len(clip_set.clips)} of "
                  f"{len(clip_set.clips) + len(clip_set.unreadable)}",
            why=("ffprobe read a positive duration from each; "
                 + (", ".join(clip_set.unreadable) + " could not be read"
                    if clip_set.unreadable else "none were unreadable")),
            decided_by="measurement", confidence=1.0))

        if clip_set.dropped:
            out.append(Decision(
                what="Duplicate takes",
                chose=f"{len(clip_set.dropped)} dropped",
                why="; ".join(f"{k}: {v}" for k, v in clip_set.dropped.items()),
                decided_by="policy", confidence=1.0))

        out.append(Decision(
            what="Usable footage",
            chose=f"{len(usable)} of {len(clip_set.clips)}",
            why=("judged per clip against the pipeline's own floors "
                 f"(mean YDIF > {build_video.ABS_DEAD_YDIF} or live share "
                 f">= {build_video.LIVE_SHARE_FLOOR}); "
                 + ("; ".join(f"{c.clip_id} - {c.usability.reason}"
                              for c in rejected)
                    if rejected else "nothing was rejected")),
            decided_by="measurement", confidence=1.0))

        out.append(Decision(
            what="Output shape",
            chose=f"{clip_set.orientation} {clip_set.canvas[0]}x{clip_set.canvas[1]}",
            why=("chosen from the majority orientation by total seconds of "
                 "source footage (%.1fs vertical vs %.1fs horizontal)"
                 % ((manifest.get("totals") or {}).get("vertical_seconds", 0),
                    (manifest.get("totals") or {}).get("horizontal_seconds", 0))),
            decided_by="measurement", confidence=1.0))

        ambiguous = [c.clip_id for c in usable
                     if c.motion.live_share >= AMBIGUOUS_LIVE_SHARE
                     and c.motion.mean_ydif <= AMBIGUOUS_MEAN_YDIF]
        if ambiguous:
            out.append(Decision(
                what="Escalate to vision",
                chose=f"{len(ambiguous)} clip(s) flagged",
                why=("nearly every frame differs but total change is small, "
                     "which describes a small subject on a still background "
                     "(a talking head, a screen recording, a locked-off "
                     "product shot) as easily as a dull shot; motion alone "
                     "cannot tell those apart, so these are flagged for "
                     "semantic analysis rather than judged here: "
                     + ", ".join(ambiguous)),
                decided_by="policy", confidence=0.8))
        return out

    def _confidence(self, clip_set: ClipAnalysisSet) -> float:
        if not clip_set.clips:
            return 0.0
        return round(sum(c.usability.confidence for c in clip_set.clips)
                     / len(clip_set.clips), 3)


__all__ = ["ClipAnalysisAgent"]
