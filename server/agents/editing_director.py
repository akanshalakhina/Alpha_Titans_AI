"""
EditingDirector Agent - the first real reasoning agent.

THE DIVISION OF AUTHORITY
-------------------------
The LLM decides:   which clips make the cut, what order they run in, which one
                   opens, and roughly how long each should run.
Deterministic code decides:  the exact in/out frames (`best_window` over the
                   measured YDIF series), and whether every number the model
                   produced is legal.

The model never sees or writes a timestamp that is used directly, and never
writes an ffmpeg anything. Its output is a set of intentions; this module turns
intentions into an EditPlan and rejects the ones that do not survive contact
with the measurements.

REJECTION IS A FIRST-CLASS PATH, NOT AN ERROR
---------------------------------------------
The model is treated as untrusted input throughout. Any of the following
rejects the whole reply and triggers the deterministic planner:

  * the reply is not JSON, or not the expected shape
  * it names a clip that was not uploaded          <- never invent clip ids
  * it names the same clip twice
  * it proposes a non-finite, zero or negative duration
  * it selects no clips, or only clips the analysis marked unusable
  * the resulting plan fails EditPlan schema validation

In every one of those cases the artifact is written with degraded=True and
fallback_used set, so nobody can mistake a fallback for reasoning.

REVISIONS
---------
A RevisionRequest carries machine-readable `constraints`. They are applied by
deterministic code AFTER whichever planner ran, so a revision changes the plan
whether or not an LLM is configured - and the constraint that could not be
satisfied is recorded as unmet rather than quietly dropped.
"""

from __future__ import annotations

import math
import time

from .artifacts import (Artifact, ClipAnalysis, ClipAnalysisSet, Cost,
                        ContentStrategy, Decision, EditPlan,
                        Animation, Music, RevisionRequest, Segment,
                        Story, Transition, VisionAnalysis, Voiceover)
from .base import Agent, AgentContext
from .tools import select_tools as ST
from .tools.llm_tools import LLMReply
from .trace import TraceSpan

PROMPT = """You are the editing director for a short video assembled from the clips below.

Every figure here was measured from the actual footage:
  mean_motion   average frame-to-frame luma difference (YDIF). Higher = busier.
  live_share    share of frames that changed at all. High with low mean_motion
                means a small subject on a still background.
  frozen        share of the clip where the picture holds still.
  duration      the clip's real length in seconds.

CLIPS
{clips}

BATCH FACTS
  output shape      {orientation} {canvas}
  target length     {target:.0f}s
  median motion     {median:.2f}
{strategy}{constraints}
YOUR TASK
Choose which clips to use, what order they run in, and roughly how long each
should be on screen. Open on whatever will hold a viewer.

RULES
- Use ONLY these exact clip ids: {names}
- Use each clip at most once.
- target_seconds must be between {min_seg} and {max_seg}, and never longer
  than the clip itself.
- You do NOT choose timestamps. A deterministic window search picks the exact
  in/out points inside each clip from the measured motion. Give lengths only.
- If a clip adds nothing, leave it out and say why.

Reply with ONLY a JSON object, no prose around it:

{{
  "strategy": "<one sentence on the approach you took>",
  "opening_rationale": "<why the first clip opens>",
  "segments": [
    {{"clip_id": "<exact id>", "target_seconds": <number>, "reason": "<why this clip, here, for this long>"}}
  ],
  "rejected": [
    {{"clip_id": "<exact id>", "why": "<why it is not in the cut>"}}
  ]
}}"""


class EditingDirectorAgent(Agent):
    id = "editing_director"
    produces = "EditPlan"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                clip_set: ClipAnalysisSet, revision: int = 0,
                revision_request: RevisionRequest | None = None,
                clip_analysis_id: str = "",
                vision: VisionAnalysis | None = None,
                strategy: ContentStrategy | None = None,
                story: Story | None = None,
                voiceover: Voiceover | None = None,
                music: Music | None = None,
                animation: Animation | None = None,
                transition: Transition | None = None
                ) -> Artifact[EditPlan]:
        started = time.time()
        constraints = dict(revision_request.constraints) if revision_request else {}

        inputs_hash = self.hash_inputs(
            [c.clip_id for c in clip_set.clips],
            clip_set.target_seconds, clip_set.canvas,
            constraints,
            ctx.job.settings.force_opening_clip,
            # a new semantic reading is a genuinely different input
            sorted((c.clip_id, c.summary, c.hook_score)
                   for c in vision.analysed_clips) if vision else None,
            # a different brief is a different planning problem, so a
            # changed strategy must not reuse the previous plan
            (strategy.central_topic, strategy.core_angle, strategy.pacing,
             tuple(strategy.ordered_clips), strategy.hook.clip_id)
            if strategy else None,
            # a different narrative is a different planning problem: the
            # order the story is TOLD in changes which clip opens and what
            # follows it, so a changed story must not reuse the old plan
            (story.logline, tuple(story.clip_order),
             tuple((b.role, b.clip_id) for b in story.beats))
            if story else None,
            # narration changes how much screen time a beat needs to
            # carry its words, so a rewritten script is a different
            # planning problem and must not reuse the previous plan
            (voiceover.total_words,
             tuple((sg.beat_index, sg.text) for sg in voiceover.segments))
            if voiceover else None,
            # mood and energy shape how long a beat wants to breathe, so
            # a changed score is a different planning problem
            (music.mood, music.tempo_feel, tuple(music.energy_shape))
            if music else None,
            # emphasis and caption intent change what a beat has to show,
            # so a changed look is a different planning problem
            (tuple(animation.treatments),
             tuple((d.beat_index, d.emphasis, d.caption_intent)
                   for d in animation.directions))
            if animation else None,
            # the join sequence feeds AT_XFADES, so a changed set of
            # transitions genuinely changes the rendered output
            tuple(transition.xfade_sequence) if transition else None,
        )

        usable = clip_set.usable_clips
        if not usable:
            raise RuntimeError("no usable clips - nothing to edit")

        decisions: list[Decision] = []
        cost = Cost()
        degraded = False
        fallback: str | None = None

        # ---- 1. reasoning, if a model is available -----------------------
        proposal, reject_reason, reply = self._ask_llm(
            ctx, span, clip_set, usable, constraints, vision, strategy,
            story, voiceover, music, animation, transition)

        if reply is not None:
            cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                        usd=reply.usd)
            span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        # ---- 2. build segments, falling back when the model is unusable ---
        if proposal is not None:
            segments, notes = self._segments_from_proposal(
                span, clip_set, proposal, constraints)
            decisions.append(Decision(
                what="Editing strategy",
                chose=proposal.get("strategy", "")[:200] or "(none given)",
                why=proposal.get("opening_rationale", "")[:400] or "(none given)",
                decided_by="llm",
                confidence=0.75))
            for r in (proposal.get("rejected") or [])[:8]:
                if isinstance(r, dict) and r.get("clip_id") in {c.clip_id for c in clip_set.clips}:
                    decisions.append(Decision(
                        what="Clip left out", chose=str(r["clip_id"]),
                        why=str(r.get("why", ""))[:200], decided_by="llm",
                        confidence=0.7))
        else:
            degraded = True
            fallback = f"deterministic_planner ({reject_reason})"
            span.fallback(fallback)
            span.tool("select.deterministic_plan", reason=reject_reason)
            segments = ctx.pool.run(
                ST.deterministic_plan, ctx.job.build_dir, clip_set,
                clip_set.target_seconds, "hook", ctx.emit,
                pool=ctx.pool, job_id=ctx.job.job_id)
            notes = []
            decisions.append(Decision(
                what="Editing strategy",
                chose="deterministic planner (sqrt-weighted motion budget)",
                why=(f"{reject_reason}; the production planner in "
                     f"pipeline/build_video.py ran instead - the same code "
                     f"that has produced every build so far - and this "
                     f"artifact is marked degraded so it is not mistaken "
                     f"for a reasoned edit"),
                decided_by="fallback", confidence=0.5))

        # ---- 3. apply revision constraints deterministically --------------
        if constraints:
            segments, applied, unmet = self._apply_constraints(
                span, clip_set, segments, constraints)
            for line in applied:
                decisions.append(Decision(
                    what="Revision constraint", chose=line["chose"],
                    why=line["why"], decided_by="policy", confidence=1.0))
            for line in unmet:
                decisions.append(Decision(
                    what="Revision constraint UNMET", chose=line["chose"],
                    why=line["why"], decided_by="measurement", confidence=1.0))
                span.note("constraint unmet: " + line["chose"])

        # ---- 4. operator override always wins -----------------------------
        forced = ctx.job.settings.force_opening_clip
        if forced and any(s.clip_id == forced for s in segments):
            segments = self._move_to_front(segments, forced)
            decisions.append(Decision(
                what="Opening clip", chose=forced,
                why="the operator pinned this clip to open the cut",
                decided_by="human", confidence=1.0))

        segments = self._renumber(segments)
        for n in notes:
            span.note(n)

        # ---- 5. validate, or fall back ------------------------------------
        xfade = ctx.job.settings.transition_duration
        try:
            plan = EditPlan(
                segments=segments,
                total_duration=ST.total_with_crossfades(
                    [s.duration for s in segments], xfade),
                target_seconds=clip_set.target_seconds,
                canvas=clip_set.canvas,
                strategy=(proposal or {}).get("strategy", "")[:300]
                         if proposal else "deterministic motion-weighted plan",
                opening_rationale=(proposal or {}).get("opening_rationale", "")[:400]
                                  if proposal else
                                  "liveliest clip opens (pipeline default)",
                rejected=[{"clip_id": c.clip_id, "why": c.usability.reason}
                          for c in clip_set.clips if not c.usability.usable],
                constraints_applied=sorted(constraints.keys()),
                # carried so it reaches compile_render_spec -> AT_XFADES.
                # Every name is already a closed literal on the Transition
                # artifact, and EditPlan re-filters on the way in.
                xfade_list=(list(transition.xfade_sequence)
                            if transition else []),
            )
        except Exception as exc:
            # The plan itself failed schema validation. Last resort: the
            # production planner with no constraints at all.
            span.fallback("deterministic_planner (assembled plan failed validation)")
            span.note(f"EditPlan validation error: {exc}")
            segments = self._renumber(ctx.pool.run(
                ST.deterministic_plan, ctx.job.build_dir, clip_set,
                clip_set.target_seconds, "hook", ctx.emit,
                pool=ctx.pool, job_id=ctx.job.job_id))
            plan = EditPlan(
                segments=segments,
                total_duration=ST.total_with_crossfades(
                    [s.duration for s in segments], xfade),
                target_seconds=clip_set.target_seconds,
                canvas=clip_set.canvas,
                strategy="deterministic motion-weighted plan",
                opening_rationale="liveliest clip opens (pipeline default)",
                xfade_list=(list(transition.xfade_sequence)
                            if transition else []),
            )
            degraded = True
            fallback = "deterministic_planner (assembled plan failed validation)"

        if strategy is not None:
            decisions.append(Decision(
                what="Brief from ContentStrategy",
                chose=(strategy.central_topic
                       or "no topic") + f" / pacing {strategy.pacing}",
                why=("the strategy says what the video should communicate; "
                     "this plan says how it is cut. The brief carries no "
                     "timestamps, so every in/out point below is still the "
                     "deterministic window search"
                     + ("" if strategy.live else
                        " - and this brief was assembled, not reasoned")),
                decided_by="policy",
                confidence=round(strategy.confidence, 2)))

        if story is not None and story.beats:
            decisions.append(Decision(
                what="Narrative from the Story agent",
                chose=" -> ".join(f"{b.role}:{b.clip_id}"
                                  for b in story.beats[:5]),
                why=("the order the story is told in; this plan decides how "
                     "it is cut. The narrative carries no timestamps, so "
                     "every in/out point below is still the deterministic "
                     "window search"
                     + ("" if story.live else
                        " - and this sequence was assembled, not authored")),
                decided_by="policy",
                confidence=round(story.confidence, 2)))

        if voiceover is not None and voiceover.spoken_segments:
            decisions.append(Decision(
                what="Narration from the Voiceover agent",
                chose=f"{len(voiceover.spoken_segments)} line(s), "
                      f"{voiceover.total_words} words over beats "
                      + ", ".join(str(i)
                                  for i in voiceover.beats_covered[:6]),
                why=("what is said over the cut; word counts indicate "
                     "which beats need room, but carry no duration - "
                     "every in/out point below is still the "
                     "deterministic window search"
                     + ("" if voiceover.live else
                        " - and no narration was actually written")),
                decided_by="policy",
                confidence=round(voiceover.confidence, 2)))

        if music is not None and music.sections:
            decisions.append(Decision(
                what="Score direction from the Music agent",
                chose=f"{music.mood} via {music.engine or 'no engine'}, "
                      + " -> ".join(music.energy_shape[:6]),
                why=("the intended feel and where it lifts; the mood is "
                     "one the engine actually supports, and no section "
                     "carries a tempo or a length - those are fitted to "
                     "the real transition marks after this cut exists"
                     + ("" if music.live else
                        " - and this mood was chosen deterministically")),
                decided_by="policy",
                confidence=round(music.confidence, 2)))

        if animation is not None and animation.directions:
            decisions.append(Decision(
                what="Look brief from the Animation agent",
                chose=", ".join(animation.treatments[:6])
                      + (f" | {len(animation.captioned_beats)} beat(s) "
                         f"want text"
                         if animation.captioned_beats else ""),
                why=("framing is derived from orientation and is what "
                     "the renderer will do regardless; the emphasis is "
                     "advisory and carries no duration - every in/out "
                     "point below is still the deterministic window "
                     "search"
                     + ("" if animation.live else
                        " - and no emphasis was judged")),
                decided_by="policy",
                confidence=round(animation.confidence, 2)))

        if transition is not None and transition.choices:
            decisions.append(Decision(
                what="Join brief from the Transition agent",
                chose=" -> ".join(transition.xfade_sequence[:6]),
                why=("which crossfade sits at each junction; the "
                     "renderer applies one at every junction at a single "
                     "global length either way, so this changes the look "
                     "of the joins and nothing about the timing"
                     + ("" if transition.live else
                        " - and none was chosen for its effect")),
                decided_by="policy",
                confidence=round(transition.confidence, 2)))

        decisions.extend(self._plan_decisions(plan, clip_set))

        ctx.emit("edit plan: %d segments, %.1fs total, opens on %s"
                 % (len(plan.segments), plan.total_duration, plan.opening_clip))
        for s in plan.segments:
            ctx.emit("  %d. %-28s %.1f-%.1fs (%.1fs)"
                     % (s.index, s.clip_id, s.start, s.end, s.duration))

        confidence = 0.5 if degraded else 0.78
        return self.wrap(ctx, plan, revision=revision, inputs_hash=inputs_hash,
                         span=span, decisions=decisions, confidence=confidence,
                         degraded=degraded, fallback_used=fallback,
                         cost=cost.model_copy(update={
                             "wall_s": round(time.time() - started, 2)}))

    # ------------------------------------------------------------- the LLM

    def _ask_llm(self, ctx: AgentContext, span: TraceSpan,
                 clip_set: ClipAnalysisSet, usable: list[ClipAnalysis],
                 constraints: dict,
                 vision: VisionAnalysis | None = None,
                 strategy: ContentStrategy | None = None,
                 story: Story | None = None,
                 voiceover: Voiceover | None = None,
                 music: Music | None = None,
                 animation: Animation | None = None,
                 transition: Transition | None = None
                 ) -> tuple[dict | None, str, LLMReply | None]:
        """Returns (validated_proposal | None, reject_reason, reply | None)."""
        if not ctx.llm.available():
            return None, ctx.llm.describe()["why_disabled"] or "no LLM configured", None

        prompt = self._build_prompt(clip_set, usable, constraints, vision,
                                    strategy, story, voiceover, music,
                                    animation, transition)
        span.tool("llm.complete_json", provider=ctx.llm.describe().get("provider"),
                  model=ctx.llm.describe().get("model"),
                  prompt_chars=len(prompt))

        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1600)
        if not reply.ok:
            return None, f"the model call failed ({reply.error})", reply
        if raw is None:
            return None, "the model reply contained no usable JSON", reply

        ok, why = self._validate(raw, clip_set)
        if not ok:
            span.note(f"LLM output rejected: {why}")
            return None, f"the model reply was rejected ({why})", reply
        return raw, "", reply

    def _build_prompt(self, clip_set: ClipAnalysisSet,
                      usable: list[ClipAnalysis], constraints: dict,
                      vision: VisionAnalysis | None = None,
                      strategy: ContentStrategy | None = None,
                      story: Story | None = None,
                      voiceover: Voiceover | None = None,
                      music: Music | None = None,
                      animation: Animation | None = None,
                      transition: Transition | None = None) -> str:
        from .tools.pipeline_bridge import build_video

        rows = "\n".join(
            "  %-30s duration %6.1fs  mean_motion %6.2f  live_share %4.0f%%  "
            "frozen %3.0f%%  %s"
            % (c.clip_id, c.technical.duration, c.motion.mean_ydif,
               100 * c.motion.live_share, 100 * c.motion.freeze_share,
               c.technical.orientation)
            for c in usable)

        motions = sorted(c.motion.mean_ydif for c in usable)
        median = motions[len(motions) // 2] if motions else 0.0

        # what the Vision agent saw, when it ran. Motion says how MUCH is
        # moving; this says WHAT is moving, which is the only basis for
        # choosing an opener on meaning rather than on energy.
        vtext = ""
        if vision is not None and vision.analysed_clips:
            vlines = []
            for c in vision.analysed_clips:
                bits = [f"  {c.clip_id}: {c.summary or 'unclear'}"]
                if c.subjects:
                    bits.append(f"     subjects: {', '.join(c.subjects)}")
                if c.actions:
                    bits.append(f"     action:   {', '.join(c.actions)}")
                bits.append(f"     hook {c.hook_score:.0f}/100, "
                            f"visual quality {c.visual_quality:.0f}/100, "
                            f"confidence {c.confidence:.0f}/100")
                if c.interesting_moments:
                    bits.append("     moments:  " + "; ".join(
                        f"{m.at:.1f}s {m.what}"
                        for m in c.interesting_moments[:3]))
                vlines.append("\n".join(bits))
            vtext = "\n\nWHAT THE FOOTAGE SHOWS\n" + "\n".join(vlines)
            if vision.hook_clip:
                vtext += f"\n\n  suggested opener: {vision.hook_clip}"
            if vision.duplicates:
                vtext += ("\n  possibly redundant: " + "; ".join(
                    f"{a} ~ {b}" for a, b in vision.duplicates))
            if vision.story:
                vtext += f"\n  story reading: {vision.story}"
            vtext += ("\n\nThis is one reading, not an instruction. "
                      "The measurements above are facts; disagree with "
                      "the reading if the numbers warrant it.\n")

        # The brief from ContentStrategy: what this video is meant to SAY.
        # It is upstream intent, not an edit - it names no timestamps and
        # cannot, so following it still leaves every in/out point to the
        # deterministic window search below.
        stext = ""
        if strategy is not None:
            bits = []
            if strategy.central_topic:
                bits.append(f"  topic     {strategy.central_topic}")
            if strategy.core_angle:
                bits.append(f"  angle     {strategy.core_angle}")
            if strategy.audience:
                bits.append(f"  audience  {strategy.audience}")
            if strategy.objective:
                bits.append(f"  objective {strategy.objective}")
            bits.append(f"  pacing    {strategy.pacing}"
                        + (f" - {strategy.pacing_rationale}"
                           if strategy.pacing_rationale else ""))
            if strategy.narrative_direction:
                bits.append(f"  shape     {strategy.narrative_direction}")
            if strategy.hook.clip_id or strategy.hook.description:
                bits.append("  hook      %s%s"
                            % (strategy.hook.clip_id or
                               strategy.hook.description,
                               f" ({strategy.hook.why})"
                               if strategy.hook.why else ""))
            if strategy.grounded_points:
                bits.append("  points to land:")
                bits += [f"    - {k.point}"
                         for k in strategy.grounded_points[:5]]
            if strategy.clip_priorities:
                bits.append("  clip priority (importance, NOT edit order):")
                bits += ["    %d. %-28s %s%s"
                         % (c.priority, c.clip_id, c.role,
                            f" - {c.why}" if c.why else "")
                         for c in sorted(strategy.clip_priorities,
                                         key=lambda c: c.priority)[:8]]
            caveat = ("\n  This brief was assembled deterministically, not "
                      "reasoned - weigh it lightly."
                      if not strategy.live else
                      "\n  The brief says WHAT to communicate. You still "
                      "decide which clips carry it,\n  in what order, and "
                      "for how long; the measurements above outrank it.")
            stext = ("\nTHE BRIEF\n" + "\n".join(bits)
                     + caveat + "\n")

        # The narrative from the Story agent: the order the video is TOLD
        # in. It names no timestamps and cannot, so following it still leaves
        # every in/out point to the deterministic window search below.
        ntext = ""
        if story is not None and story.beats:
            # NOT `rows`: that name already holds the clip table above, and
            # shadowing it here silently fed a list into the prompt format
            beat_rows = []
            for b in story.beats:
                line = "    %d. %-9s %-28s" % (b.index, b.role, b.clip_id)
                if b.narrative:
                    line += " " + b.narrative[:80]
                beat_rows.append(line)
            head = []
            if story.logline:
                head.append(f"  logline   {story.logline}")
            for label, val in (("hook", story.hook), ("setup", story.setup),
                               ("builds", story.progression),
                               ("payoff", story.payoff),
                               ("ending", story.ending)):
                if val:
                    head.append(f"  {label:<10}{val}")
            caveat = ("\n  This sequence was assembled "
                      "deterministically, not authored - weigh it lightly."
                      if not story.live else
                      "\n  This is the order the story should be "
                      "TOLD in. You still choose\n  the cut, the "
                      "trims and every in/out point; the measurements above "
                      "outrank it.")
            ntext = ("\nTHE NARRATIVE\n"
                     + ("\n".join(head) + "\n" if head else "")
                     + "  beats:\n" + "\n".join(beat_rows) + caveat + "\n")

        # The narration from the Voiceover agent. It tells the director
        # which beats carry words - a beat with a long line needs room to
        # breathe - WITHOUT stating any duration, which it cannot express.
        vtext2 = ""
        if voiceover is not None and voiceover.spoken_segments:
            vo_rows = []
            for sg in voiceover.spoken_segments:
                vo_rows.append(
                    "    beat %d (%s, %d words): %s"
                    % (sg.beat_index, sg.tone, sg.word_count,
                       sg.text[:100]))
            caveat2 = (
                "\n  Word counts show which beats have more to say, so those "
                "can be given\n  room to breathe. They are NOT durations - "
                "you still choose every\n  in/out point from the measurements "
                "above."
                if voiceover.live else
                "\n  No narration was actually written; these are "
                "placeholders.")
            vtext2 = ("\nTHE NARRATION\n  voice: %s\n" % voiceover.voice
                      + "\n".join(vo_rows) + caveat2 + "\n")

        # The score direction. It tells the director the intended feel and
        # which beats rise or fall, WITHOUT any tempo, length or timing -
        # those are fitted to the real transition marks after the cut exists.
        mtext = ""
        if music is not None and music.sections:
            m_rows = ["    beat %d: %s%s"
                      % (sec.beat_index, sec.energy,
                         " (under narration)" if sec.under_narration else "")
                      for sec in music.sections]
            m_head = ["  mood: %s" % music.mood]
            if music.tempo_feel:
                m_head.append("  feel: %s" % music.tempo_feel)
            if music.energy_arc:
                m_head.append("  arc:  %s" % music.energy_arc)
            m_caveat = ("\n  Energy tells you where the piece lifts, so those "
                        "beats can be given\n  room. It is NOT a tempo or a "
                        "length - you still choose every in/out\n  point from "
                        "the measurements above."
                        if music.live else
                        "\n  This mood was chosen deterministically from "
                        "measured motion.")
            mtext = ("\nTHE SCORE\n" + "\n".join(m_head) + "\n"
                     + "\n".join(m_rows) + m_caveat + "\n")

        # The look brief. Framing is already decided by orientation, so what
        # reaches the director is the EMPHASIS - which beats need to show a
        # subject clearly, and which want room for words.
        atext = ""
        if animation is not None and animation.directions:
            a_rows = []
            for d in animation.supported_directions:
                bits = [d.treatment]
                if d.emphasis != "none":
                    bits.append("emphasis: " + d.emphasis)
                if d.caption_intent != "none":
                    bits.append("wants text: " + d.caption_intent)
                a_rows.append("    beat %d: %s" % (d.beat_index,
                                                   ", ".join(bits)))
            a_head = []
            if animation.visual_style:
                a_head.append("  style: %s" % animation.visual_style)
            a_caveat = ("\n  A beat that has to show a subject clearly, or "
                        "carry text, needs a\n  little longer on screen. This "
                        "is not a duration - you still choose\n  every in/out "
                        "point from the measurements above."
                        if animation.live else
                        "\n  Framing only; no emphasis was judged.")
            atext = ("\nTHE LOOK\n"
                     + ("\n".join(a_head) + "\n" if a_head else "")
                     + "\n".join(a_rows) + a_caveat + "\n")

        # The join brief. Which crossfade sits at each junction - the
        # renderer applies one at EVERY junction regardless, so this says
        # which, never whether or for how long.
        ttext = ""
        if transition is not None and transition.choices:
            t_rows = ["    beat %d -> %d: %s%s"
                      % (c.from_beat, c.to_beat, c.kind,
                         f" ({c.intent})" if c.intent else "")
                      for c in transition.choices]
            t_head = []
            if transition.style:
                t_head.append("  feel: %s" % transition.style)
            t_caveat = ("\n  Every junction crossfades at one global length, "
                        "so this is which\n  transition, never whether or how "
                        "long. It does not affect your\n  in/out points."
                        if transition.live else
                        "\n  Plain crossfades throughout; none was chosen for "
                        "its effect.")
            ttext = ("\nTHE JOINS\n"
                     + ("\n".join(t_head) + "\n" if t_head else "")
                     + "\n".join(t_rows) + t_caveat + "\n")

        ctext = ""
        if constraints:
            lines = "\n".join(f"  - {k}: {v}" for k, v in constraints.items())
            ctext = ("\nMANDATORY CONSTRAINTS FROM A PREVIOUS REVIEW\n"
                     "A render of an earlier plan was reviewed and rejected. "
                     "These constraints are\nnot advice - a deterministic "
                     "check enforces them after you reply:\n" + lines + "\n")

        return PROMPT.format(
            clips=rows + vtext, orientation=clip_set.orientation,
            canvas="%dx%d" % clip_set.canvas,
            target=clip_set.target_seconds, median=median,
            strategy=stext + ntext + vtext2 + mtext + atext + ttext,
            constraints=ctext,
            names=", ".join(c.clip_id for c in usable),
            min_seg=build_video.MIN_SEG, max_seg=build_video.MAX_SEG)

    # ---------------------------------------------------------- validation

    def _validate(self, raw, clip_set: ClipAnalysisSet) -> tuple[bool, str]:
        """Treat the model as untrusted input. Every branch here is a real
        failure mode that has to route to the fallback, not raise."""
        if not isinstance(raw, dict):
            return False, "reply was not a JSON object"

        segs = raw.get("segments")
        if not isinstance(segs, list) or not segs:
            return False, "no segments in the reply"

        known = {c.clip_id for c in clip_set.clips}
        usable = {c.clip_id for c in clip_set.usable_clips}
        seen: set[str] = set()

        for i, s in enumerate(segs):
            if not isinstance(s, dict):
                return False, f"segment {i + 1} was not an object"
            cid = s.get("clip_id")
            if not isinstance(cid, str) or not cid:
                return False, f"segment {i + 1} has no clip_id"
            if cid not in known:
                return False, (f"segment {i + 1} names {cid!r}, which was "
                               f"never uploaded")
            if cid in seen:
                return False, f"{cid!r} appears more than once"
            seen.add(cid)

            ts = s.get("target_seconds")
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                return False, f"segment {i + 1} has a non-numeric target_seconds"
            if not math.isfinite(ts) or ts <= 0:
                return False, (f"segment {i + 1} has an impossible "
                               f"target_seconds of {ts}")

        if not (seen & usable):
            return False, "every selected clip was one the analysis rejected"
        return True, ""

    # ------------------------------------------------------------ segments

    def _segments_from_proposal(self, span: TraceSpan,
                                clip_set: ClipAnalysisSet, proposal: dict,
                                constraints: dict) -> tuple[list[Segment], list[str]]:
        """Turn validated intentions into exact frames, deterministically."""
        segments: list[Segment] = []
        notes: list[str] = []
        min_open = constraints.get("opening_min_mean_ydif")

        for i, s in enumerate(proposal["segments"]):
            clip = clip_set.by_id(s["clip_id"])
            if clip is None or not clip.usability.usable:
                notes.append(f"{s['clip_id']}: dropped - the analysis marked "
                             f"it unusable")
                continue

            want = float(s["target_seconds"])
            span.tool("select.best_window", clip=clip.clip_id,
                      requested_length=round(want, 2))
            start, end, score, wnotes = ST.resolve_window(
                clip, want,
                min_mean_ydif=min_open if i == 0 else None)
            notes.extend(wnotes)

            segments.append(Segment(
                index=len(segments) + 1, clip_id=clip.clip_id,
                start=start, end=end, duration=round(end - start, 2),
                score=score, orientation=clip.technical.orientation,
                caption="",
                reason=str(s.get("reason", ""))[:200] or
                       "highest-scoring window of the requested length",
            ))
        if not segments:
            raise RuntimeError("the model selected only unusable clips")
        return segments, notes

    # --------------------------------------------------------- constraints

    def _apply_constraints(self, span: TraceSpan, clip_set: ClipAnalysisSet,
                           segments: list[Segment], constraints: dict
                           ) -> tuple[list[Segment], list[dict], list[dict]]:
        """Enforce a RevisionRequest. Deterministic, and honest when it fails.

        This runs whether the plan came from the model or the fallback, which
        is what makes a revision meaningful with no LLM configured.
        """
        applied: list[dict] = []
        unmet: list[dict] = []

        # --- which clip opens -------------------------------------------
        want_open = constraints.get("opening_clip")
        if constraints.get("opening_must_be_liveliest") and not want_open:
            present = [s.clip_id for s in segments]
            lively = sorted(
                (c for c in clip_set.usable_clips if c.clip_id in present),
                key=lambda c: -c.motion.mean_ydif)
            want_open = lively[0].clip_id if lively else None

        avoid = constraints.get("avoid_opening_clip")
        if want_open and any(s.clip_id == want_open for s in segments):
            if segments[0].clip_id != want_open:
                segments = self._move_to_front(segments, want_open)
                applied.append({
                    "chose": f"opens on {want_open}",
                    "why": ("the previous render's opening measured below the "
                            "video's own median motion, so the liveliest "
                            "available clip was moved to the front")})
            else:
                unmet.append({
                    "chose": f"opening already was {want_open}",
                    "why": ("the constraint asked for a clip that was already "
                            "opening the cut, so the running order could not "
                            "improve the hook any further")})
        elif avoid and segments[0].clip_id == avoid and len(segments) > 1:
            segments = segments[1:] + segments[:1]
            applied.append({
                "chose": f"moved {avoid} out of the opening slot",
                "why": "the review rejected it as an opener"})

        # --- how long the opening runs -----------------------------------
        cap = constraints.get("opening_max_duration")
        if cap and segments and segments[0].duration > float(cap) + 0.01:
            clip = clip_set.by_id(segments[0].clip_id)
            if clip is not None:
                span.tool("select.best_window", clip=clip.clip_id,
                          requested_length=float(cap), reason="opening cap")
                start, end, score, _ = ST.resolve_window(
                    clip, float(cap),
                    min_mean_ydif=constraints.get("opening_min_mean_ydif"))
                was = segments[0].duration
                segments[0] = segments[0].model_copy(update={
                    "start": start, "end": end,
                    "duration": round(end - start, 2), "score": score})
                applied.append({
                    "chose": f"opening cut from {was:.1f}s to "
                             f"{segments[0].duration:.1f}s",
                    "why": ("a long opening on flat footage is what the review "
                            "flagged; the window was re-searched at the "
                            "shorter length")})

        # --- did the opening actually get livelier? ----------------------
        floor = constraints.get("opening_min_mean_ydif")
        if floor is not None and segments:
            clip = clip_set.by_id(segments[0].clip_id)
            if clip is not None:
                got = ST.mean_ydif_over(clip, segments[0].start, segments[0].end)
                span.tool("select.window_score", clip=clip.clip_id,
                          opening_mean_ydif=got, required=float(floor))
                if got + 1e-6 < float(floor):
                    unmet.append({
                        "chose": f"opening reaches {got:.2f} of a required "
                                 f"{float(floor):.2f}",
                        "why": ("no window in any available clip is livelier "
                                "than this; the footage does not contain a "
                                "stronger hook, so the constraint is reported "
                                "unmet rather than faked")})
                else:
                    applied.append({
                        "chose": f"opening now measures {got:.2f} mean motion",
                        "why": (f"clears the {float(floor):.2f} floor the "
                                f"review required")})
        return segments, applied, unmet

    # --------------------------------------------------------------- utils

    @staticmethod
    def _move_to_front(segments: list[Segment], clip_id: str) -> list[Segment]:
        head = [s for s in segments if s.clip_id == clip_id]
        tail = [s for s in segments if s.clip_id != clip_id]
        return head + tail

    @staticmethod
    def _renumber(segments: list[Segment]) -> list[Segment]:
        return [s.model_copy(update={"index": i})
                for i, s in enumerate(segments, 1)]

    def _plan_decisions(self, plan: EditPlan,
                        clip_set: ClipAnalysisSet) -> list[Decision]:
        return [
            Decision(
                what="Clips in the cut",
                chose=f"{len(plan.segments)} of {len(clip_set.clips)}",
                why=", ".join(s.clip_id for s in plan.segments),
                decided_by="llm" if plan.strategy and
                           "deterministic" not in plan.strategy else "measurement",
                confidence=0.8),
            Decision(
                what="Trim points",
                chose=", ".join(f"{s.clip_id} {s.start:.1f}-{s.end:.1f}"
                                for s in plan.segments[:4])
                      + ("  ..." if len(plan.segments) > 4 else ""),
                why=("chosen by the pipeline's own window search: the "
                     "highest-scoring window of the allotted length, stepped "
                     "0.2s at a time and penalised where freezedetect found "
                     "the picture holding still. The model does not pick "
                     "timestamps."),
                decided_by="measurement", confidence=1.0),
            Decision(
                what="Pacing",
                chose=(f"{len(plan.segments)} segments, "
                       f"{min(s.duration for s in plan.segments):.1f}-"
                       f"{max(s.duration for s in plan.segments):.1f}s each"),
                why=f"finished length {plan.total_duration:.1f}s against a "
                    f"{plan.target_seconds:.0f}s target",
                decided_by="measurement", confidence=1.0),
        ]


__all__ = ["EditingDirectorAgent"]
