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
                        RevisionRequest, TechnicalCheck, VisualReview)
from .base import Agent, AgentContext
from .tools import verify_tools as VT
from .trace import TraceSpan

DEFAULT_PASS_THRESHOLD = 65.0

# Opening cap applied by a hook revision. Long openings on flat footage are the
# common failure; 8s leaves room for a real shot while forcing a tighter start.
HOOK_OPENING_CAP_S = 8.0


# Loudness. ebur128 reports its floor (about -70) for digital silence, and
# the master stage normalises a real bed to -16, so the two are nowhere near
# each other and the floor does not need to be tight.
TARGET_LUFS = -16.0
LUFS_TOLERANCE = 1.5
SILENCE_LUFS = -50.0


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
                render_id: str = "",
                visual: VisualReview | None = None
                ) -> tuple[Artifact[CriticResult], RevisionRequest | None]:
        started = time.time()
        path = render.video_path
        inputs_hash = self.hash_inputs(
            path, attempt, pass_threshold,
            visual.model_dump(mode="json") if visual else None)

        technical = self._technical(ctx, span, path, plan, render)
        creative, profile = self._creative(ctx, span, path)
        shots = self._shots(visual)
        if shots is not None:
            creative.append(shots)
        weak = self._weak_windows(visual)

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
        elif (overall >= pass_threshold and weak
              and revision_count < max_revisions):
            verdict, summary = "REVISE", (
                "the score is %.1f, but the visual review found %d weak "
                "shot(s) that do not show the subject; they are re-cut from "
                "other moments" % (overall, len(weak)))
        elif overall >= pass_threshold:
            verdict, summary = "PASS", (
                "technical checks passed and the score is %.1f against a "
                "%.0f threshold" % (overall, pass_threshold)
                + ("; %d weak shot(s) remain after %d revision(s) - the "
                   "footage has nothing clearer" % (len(weak), revision_count)
                   if weak else ""))
        elif revision_count >= max_revisions:
            verdict, summary = "ESCALATE", (
                "the score is still %.1f after %d revision(s); the maximum "
                "is %d, so this goes to a human rather than looping"
                % (overall, revision_count, max_revisions))
        else:
            verdict, summary = "REVISE", (
                "the score is %.1f against a %.0f threshold"
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
            if blocking or overall < pass_threshold:
                request = self._revision_request(
                    attempt, creative, plan, clip_set, profile,
                    previously_tried or [])
            request = self._with_replacements(
                request, attempt, weak, visual, previously_tried or [])

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
                chose=("2 of 10 dimensions measured" if shots is not None
                       else "1 of 10 dimensions measured"),
                why=("the first-3-second hook is measured"
                     + (", and the Visual Reviewer scored every shot for "
                        "showing the subject" if shots is not None else
                        " only")
                     + ". Pacing, dead sections, visual variety, transition "
                       "quality, animation, music sync, story coherence and "
                       "payoff are not implemented and are reported as "
                       "unreviewed rather than scored"),
                decided_by="policy", confidence=1.0),
        ]
        if weak:
            decisions.append(Decision(
                what="Weak shots",
                chose=", ".join(f"#{w['index']} {w['clip_id']} "
                                f"{w['start']:.1f}-{w['end']:.1f}s"
                                for w in weak),
                why=("the Visual Reviewer scored them below %.0f/10 for "
                     "showing the subject" % visual.weak_below),
                decided_by="llm", confidence=0.7))

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
        # Judged against what the render was ASKED for, which the
        # RenderResult now carries. A video muxed silent on purpose
        # (music_source="none", keep_audio="none") is correct and passes
        # informationally. A video that was supposed to carry a bed and came
        # out inaudible is a real defect, and until 2026-10-02 it scored
        # 100/100: this check returned passed=True unconditionally, so every
        # agent build shipped silent with the critic's blessing.
        span.tool("verify.loudness")
        lufs = render.technical.lufs
        if lufs is None:
            span.tool("verify.loudness", recomputed=True)
            lufs = ctx.pool.run(VT.loudness, path)

        wanted = render.wanted_audio
        silent = lufs is None or lufs <= SILENCE_LUFS
        span.tool("verify.loudness", wanted_audio=wanted, lufs=lufs,
                  silent=silent)
        if not wanted:
            out.append(TechnicalCheck(
                check="loudness",
                value=(f"{lufs} LUFS" if lufs is not None else "unmeasurable"),
                threshold="no bed was requested for this build, so loudness "
                          "is reported but not judged",
                passed=True, severity="low", routes_to="music"))
        elif silent:
            out.append(TechnicalCheck(
                check="loudness",
                value=(f"{lufs} LUFS - inaudible" if lufs is not None
                       else "unmeasurable"),
                threshold=(f"a bed was requested ({render.music_source}"
                           + (f"/{render.mood}" if render.mood else "")
                           + f", keep_audio={render.keep_audio}) so the "
                           f"track must measure above {SILENCE_LUFS} LUFS"),
                passed=False, severity="blocking", routes_to="music"))
        else:
            ok = abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE
            out.append(TechnicalCheck(
                check="loudness", value=f"{lufs} LUFS",
                threshold=f"{TARGET_LUFS} +/-{LUFS_TOLERANCE} LUFS",
                passed=True, severity="low", routes_to="music")
                if ok else TechnicalCheck(
                check="loudness", value=f"{lufs} LUFS",
                threshold=(f"{TARGET_LUFS} +/-{LUFS_TOLERANCE} LUFS; audible "
                           f"but off target, which the loudnorm pass should "
                           f"have corrected"),
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
    def _shots(visual: VisualReview | None) -> CreativeCheck | None:
        """The Visual Reviewer's scores as a creative dimension, or None
        when nobody looked - never a made-up number."""
        if visual is None or not visual.reviewed or not visual.shots:
            return None
        score = round((visual.mean_score or 0.0) * 10, 1)
        return CreativeCheck(
            dimension="shots_show_subject", score=score,
            evidence=visual.summary, confidence=0.7, severity="high",
            routes_to="editing_director", measured=True)

    @staticmethod
    def _weak_windows(visual: VisualReview | None) -> list[dict]:
        if visual is None or not visual.reviewed:
            return []
        weak = set(visual.weak)
        return [{"index": x.index, "clip_id": x.clip_id,
                 "start": round(x.start, 2), "end": round(x.end, 2),
                 "score": x.score}
                for x in visual.shots if x.index in weak]

    @staticmethod
    def _with_replacements(request: RevisionRequest | None, attempt: int,
                           weak: list[dict], visual: VisualReview | None,
                           previously_tried: list[dict]
                           ) -> RevisionRequest | None:
        """Add "do not use these moments again" to the revision.

        Windows rejected on EARLIER attempts are carried too, so a third
        cut cannot drift back to a shot the reviewer turned down in the
        first one."""
        if not weak:
            return request
        rejected = []
        for t in previously_tried:
            rejected += (t.get("constraints") or {}).get("replace_windows", [])
        rejected += [{k: w[k] for k in ("clip_id", "start", "end")}
                     for w in weak]
        seen, unique = set(), []
        for w in rejected:
            key = (w["clip_id"], round(float(w["start"]), 2),
                   round(float(w["end"]), 2))
            if key not in seen:
                seen.add(key)
                unique.append({"clip_id": key[0], "start": key[1],
                               "end": key[2]})
        issue = ("the visual review scored %s below %.0f/10: the subject is "
                 "not clearly visible" % (
                     ", ".join(f"shot {w['index']}" for w in weak),
                     visual.weak_below if visual else 4.0))
        if request is None:
            return RevisionRequest(
                from_attempt=attempt, target_agent="editing_director",
                issue=issue, severity="high",
                constraints={"replace_windows": unique},
                rationale=("re-cut those shots from other moments of the same "
                           "footage; never the rejected windows"),
                previously_tried=list(previously_tried))
        return request.model_copy(update={
            "issue": request.issue + "; " + issue,
            "constraints": dict(request.constraints,
                                replace_windows=unique)})

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
