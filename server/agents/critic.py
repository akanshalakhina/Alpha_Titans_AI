"""
Critic / QA Agent.

It judges the ACTUAL RENDERED MP4. Every figure below is measured from the
file on disk: the container is probed, the stream is fully decoded, and the
output's own YDIF series is measured with the same filter chain the pipeline
uses on source clips. Nothing here reads the EditPlan to decide whether the
video is good - the plan is consulted only to route a revision at the agent
that can fix what was measured.

PHASE 1 SCOPE, STATED PLAINLY
-----------------------------
Technical:  frame count, resolution, fps, duration, loudness, black frames,
            decode validity.
Creative:   ONE dimension - the first-3-second hook - scored from the
            rendered file's own motion against its own median.

The other nine creative dimensions in the architecture (pacing, dead sections,
visual variety, transition quality, animation, captions, music sync, story
coherence, payoff) are NOT implemented here and are not faked. A dimension
that was not measured is absent from the result rather than present with an
invented number.

No LLM is used in Phase 1. The creative check is a measurement, not a
judgement, so `unreviewed_creative` is reported honestly: the nine unscored
dimensions mean this critic has looked at one narrow thing well rather than
everything badly.
"""

from __future__ import annotations

import time

from .artifacts import (Artifact, ClipAnalysisSet, Cost, CreativeCheck,
                        CriticResult, Decision, EditPlan, RenderResult,
                        RevisionRequest, TechnicalCheck)
from .base import Agent, AgentContext
from .tools import verify_tools as VT
from .trace import TraceSpan

DEFAULT_PASS_THRESHOLD = 65.0

# Opening cap applied by a hook revision. Long openings on flat footage are the
# common failure; 8s leaves room for a real shot while forcing a tighter start.
HOOK_OPENING_CAP_S = 8.0


class CriticAgent(Agent):
    id = "critic"
    produces = "CriticResult"
    uses_llm = False           # Phase 1: measurement only. Phase 5 adds vision.
    agency = "deterministic"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                render: RenderResult, plan: EditPlan,
                clip_set: ClipAnalysisSet, attempt: int,
                revision_count: int, max_revisions: int,
                previous: CriticResult | None = None,
                previously_tried: list[dict] | None = None,
                pass_threshold: float = DEFAULT_PASS_THRESHOLD,
                revision: int = 0,
                render_id: str = "") -> tuple[Artifact[CriticResult],
                                              RevisionRequest | None]:
        started = time.time()
        path = render.video_path
        inputs_hash = self.hash_inputs(path, attempt, pass_threshold)

        technical = self._technical(ctx, span, path, plan, render)
        creative, profile = self._creative(ctx, span, path)

        blocking = [t for t in technical
                    if not t.passed and t.severity == "blocking"]
        overall = self._overall(creative)

        # A blocking technical failure short-circuits: there is no point
        # scoring the hook of a file with corrupt frames.
        if blocking:
            verdict = ("REVISE" if revision_count < max_revisions
                       else "ESCALATE")
            summary = ("blocking technical failure: "
                       + "; ".join(t.check for t in blocking))
        elif overall >= pass_threshold:
            verdict, summary = "PASS", (
                "technical checks passed and the hook scores %.1f against a "
                "%.0f threshold" % (overall, pass_threshold))
        elif revision_count >= max_revisions:
            verdict, summary = "ESCALATE", (
                "the hook still scores %.1f after %d revision(s); the maximum "
                "is %d, so this goes to a human rather than looping"
                % (overall, revision_count, max_revisions))
        else:
            verdict, summary = "REVISE", (
                "the hook scores %.1f against a %.0f threshold"
                % (overall, pass_threshold))

        compared = None
        if previous is not None:
            delta = round(overall - previous.overall_score, 1)
            compared = {"previous_attempt": previous.attempt,
                        "previous_score": previous.overall_score,
                        "delta": delta, "better": delta > 0}
            span.note(f"attempt {attempt} vs {previous.attempt}: "
                      f"{delta:+.1f} points")

        result = CriticResult(
            attempt=attempt, verdict=verdict, overall_score=overall,
            pass_threshold=pass_threshold, technical=technical,
            creative=creative, compared_to_previous=compared,
            unreviewed_creative=True, summary=summary)

        request = None
        if verdict == "REVISE":
            request = self._revision_request(
                attempt, creative, plan, clip_set, profile,
                previously_tried or [])

        decisions = [
            Decision(
                what="Verdict", chose=verdict, why=summary,
                decided_by="measurement",
                confidence=min((c.confidence for c in creative), default=1.0)),
            Decision(
                what="Technical checks",
                chose=f"{sum(1 for t in technical if t.passed)}/"
                      f"{len(technical)} passed",
                why="; ".join(f"{t.check}={t.value}" for t in technical),
                decided_by="measurement", confidence=1.0),
            Decision(
                what="Creative coverage",
                chose="1 of 10 dimensions measured",
                why=("Phase 1 measures the first-3-second hook only. Pacing, "
                     "dead sections, visual variety, transition quality, "
                     "animation, captions, music sync, story coherence and "
                     "payoff are not implemented and are reported as "
                     "unreviewed rather than scored"),
                decided_by="policy", confidence=1.0),
        ]

        ctx.emit("critic: %s  score %.1f/%.0f  %s"
                 % (verdict, overall, pass_threshold, summary))
        for t in technical:
            ctx.emit("  [%s] %-16s %s" % ("ok" if t.passed else "FAIL",
                                          t.check, t.value))
        for c in creative:
            ctx.emit("  [%.0f] %-16s %s" % (c.score, c.dimension, c.evidence))

        art = self.wrap(ctx, result, revision=revision,
                        inputs_hash=inputs_hash, span=span,
                        decisions=decisions,
                        confidence=min((c.confidence for c in creative),
                                       default=1.0),
                        cost=Cost(wall_s=round(time.time() - started, 2)))
        return art, request

    # ----------------------------------------------------------- technical

    def _technical(self, ctx: AgentContext, span: TraceSpan, path: str,
                   plan: EditPlan, render: RenderResult) -> list[TechnicalCheck]:
        out: list[TechnicalCheck] = []

        span.tool("verify.probe_final", path=path)
        probe = ctx.pool.run(VT.probe_final, path)

        # --- decode validity (blocking) ---------------------------------
        span.tool("verify.decode_check")
        decode_ok, decode_err = ctx.pool.run(VT.decode_check, path)
        out.append(TechnicalCheck(
            check="decode_valid", value="clean" if decode_ok else decode_err,
            threshold="a full decode pass emits no errors", passed=decode_ok,
            severity="blocking", routes_to="rendering"))

        # --- frame count (blocking) -------------------------------------
        span.tool("verify.frame_count")
        frames = ctx.pool.run(VT.frame_count, path)
        expected = int(round(probe["duration"] * 30))
        frames_ok = frames is not None and abs(frames - expected) <= 2
        out.append(TechnicalCheck(
            check="frame_count", value=f"{frames} decoded vs {expected} expected",
            threshold="within 2 frames", passed=frames_ok,
            severity="blocking", routes_to="rendering"))

        # --- resolution (blocking) --------------------------------------
        want = tuple(plan.canvas)
        got = (probe["width"], probe["height"])
        out.append(TechnicalCheck(
            check="resolution", value=f"{got[0]}x{got[1]}",
            threshold=f"{want[0]}x{want[1]}", passed=got == want,
            severity="blocking", routes_to="rendering"))

        # --- fps ---------------------------------------------------------
        out.append(TechnicalCheck(
            check="fps", value=probe["fps"], threshold="30/1",
            passed=probe["fps"] == "30/1", severity="high",
            routes_to="rendering"))

        # --- duration ----------------------------------------------------
        drift = abs(probe["duration"] - plan.total_duration)
        out.append(TechnicalCheck(
            check="duration", value=f"{probe['duration']:.2f}s vs "
                                    f"{plan.total_duration:.2f}s planned",
            threshold="within 0.5s", passed=drift <= 0.5, severity="high",
            routes_to="editing_director"))

        # --- black frames (blocking) -------------------------------------
        span.tool("verify.black_frames")
        has_black, black_detail = ctx.pool.run(VT.black_frames, path)
        out.append(TechnicalCheck(
            check="black_frames", value=black_detail or "none",
            threshold="no run of 0.3s or longer", passed=not has_black,
            severity="blocking", routes_to="rendering"))

        # --- audio stream present ----------------------------------------
        out.append(TechnicalCheck(
            check="audio_stream", value=probe["audio_codec"] or "missing",
            threshold="an audio stream exists", passed=probe["has_audio"],
            severity="blocking", routes_to="rendering"))

        # --- loudness -----------------------------------------------------
        # Phase 1 has no Music Agent, so the master stage muxes silence. The
        # figure is still measured and reported, but failing a build for the
        # loudness of a bed this phase deliberately does not produce would be
        # dishonest, so the check is informational until Phase 4.
        span.tool("verify.loudness")
        lufs = render.technical.lufs
        if lufs is None:
            span.tool("verify.loudness", recomputed=True)
            lufs = ctx.pool.run(VT.loudness, path)
        out.append(TechnicalCheck(
            check="loudness", value=(f"{lufs} LUFS" if lufs is not None
                                     else "unmeasurable"),
            threshold="-16 +/-1.5 LUFS once a music bed exists (Phase 4); "
                      "informational while this phase renders silence",
            passed=True, severity="low", routes_to="music"))

        return out

    # ------------------------------------------------------------ creative

    def _creative(self, ctx: AgentContext, span: TraceSpan,
                  path: str) -> tuple[list[CreativeCheck], dict]:
        span.tool("verify.output_motion_profile", path=path, window_s=3.0)
        profile = ctx.pool.run(VT.output_motion_profile, path, 10.0, 3.0)
        score, evidence, confidence = VT.score_hook(profile)
        span.tool("verify.score_hook", score=score,
                  hook_mean=profile.get("hook_mean"),
                  overall_median=profile.get("overall_median"))

        return [CreativeCheck(
            dimension="first_3s_hook", score=score, evidence=evidence,
            confidence=confidence, severity="high",
            routes_to="editing_director",
            measured=bool(profile.get("measured")))], profile

    @staticmethod
    def _overall(creative: list[CreativeCheck]) -> float:
        measured = [c for c in creative if c.measured]
        if not measured:
            return 0.0
        return round(sum(c.score for c in measured) / len(measured), 1)

    # ------------------------------------------------------------ revision

    def _revision_request(self, attempt: int, creative: list[CreativeCheck],
                          plan: EditPlan, clip_set: ClipAnalysisSet,
                          profile: dict,
                          previously_tried: list[dict]) -> RevisionRequest:
        """Turn a measured failure into machine-readable constraints.

        The constraints are expressed in the SOURCE motion domain, because that
        is what the EditingDirector's window search can actually act on. The
        floor is the median of the usable clips' own mean motion: "do not open
        on something quieter than your typical clip" is a bar the footage can
        be checked against, unlike a number carried over from the encoded
        output's scale.
        """
        hook = next((c for c in creative if c.dimension == "first_3s_hook"), None)
        motions = sorted(c.motion.mean_ydif for c in clip_set.usable_clips)
        median_source = motions[len(motions) // 2] if motions else 0.0

        present = {s.clip_id for s in plan.segments}
        liveliest = max(
            (c for c in clip_set.usable_clips if c.clip_id in present),
            key=lambda c: c.motion.mean_ydif, default=None)

        constraints: dict = {
            "opening_min_mean_ydif": round(median_source, 3),
            "opening_max_duration": HOOK_OPENING_CAP_S,
            "opening_must_be_liveliest": True,
        }
        if liveliest is not None and liveliest.clip_id != plan.opening_clip:
            constraints["opening_clip"] = liveliest.clip_id

        return RevisionRequest(
            from_attempt=attempt,
            target_agent="editing_director",
            issue=("the opening 3 seconds of the rendered file measure "
                   "%.2f mean motion against a whole-video median of %.2f "
                   "(ratio %.2f); the video opens quieter than it typically is"
                   % (profile.get("hook_mean", 0.0),
                      profile.get("overall_median", 0.0),
                      (profile.get("hook_mean", 0.0)
                       / profile["overall_median"])
                      if profile.get("overall_median") else 0.0)),
            severity="high",
            constraints=constraints,
            rationale=(hook.evidence if hook else "")
                      + "; re-plan so the cut opens on its liveliest available "
                        "material and does not linger there",
            previously_tried=list(previously_tried),
        )


__all__ = ["CriticAgent", "DEFAULT_PASS_THRESHOLD"]
