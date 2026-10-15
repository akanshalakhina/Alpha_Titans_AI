"""
Story Agent - the narrative shape of the video.

WHERE IT SITS
-------------
    ContentStrategy  (what we are saying)
            |
            v
        Story        (how it lands as a sequence)   <- this agent
            |
            v
    EditingDirector  (how it is cut)

Three different questions. ContentStrategy answers "what is this video about,
for whom, and why". The director answers "which clip runs when, and for how
long". Neither answers "what is the hook, what sets it up, what pays it off" -
so the director was inferring a narrative from a topic and a pacing word. This
agent is that missing layer.

WHAT IT MAY NOT DO
------------------
No timestamps, no durations, no in/out points, no transitions, no filter
strings, no paths, no commands. `StoryBeat` forbids extra fields, so a reply
carrying `start`/`end`/`duration`/`at` fails validation rather than being
quietly ignored - the guard is structural, not a promise. A beat names a clip
and, optionally, an INDEX into moments Vision already measured. Timing stays
with the deterministic window search downstream.

GROUNDING, ENFORCED
-------------------
Three rules, all applied AFTER the model replies:

  * a beat's clip must be one Vision actually ANALYSED. A beat about footage
    nobody looked at is a beat about nothing, so it is dropped outright -
    unlike a key point, a beat carries a clip into the edit, and an invented
    one would put a real clip in a real cut for an invented reason.
  * a beat citing research must cite a url that is actually in the
    ResearchReport. An invented citation is not evidence; it is stripped and
    the beat is downgraded rather than dropped, because the narrative thought
    may still be sound.
  * `support` is derived from what SURVIVED, never from what the model
    claimed.

DEGRADATION
-----------
No model, no strategy, no vision, an unusable reply - each produces a valid
story assembled from what is actually on hand, clearly marked as assembled
rather than authored. Inventing a three-act structure with no footage
understanding behind it would be worse than the gap it papers over.
"""

from __future__ import annotations

import math
import time

from .artifacts import (Artifact, ClipAnalysisSet, ContentStrategy, Cost,
                        Decision, ResearchReport, Story, StoryBeat,
                        VisionAnalysis)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "story-prompt-v1"
MAX_BEATS = 10
MAX_LIST = 8

ROLES = ("hook", "setup", "develop", "turn", "payoff", "ending")
SUPPORTS = ("vision", "research", "both", "none")

PROMPT = """You are the story editor for a short video.

The strategy below already decided WHAT this video is about. Your job is the
shape it takes: what opens it, what sets it up, what builds, what pays off,
and how it ends. You do NOT decide how it is cut - someone else owns that.

{strategy}
{vision}
{research}
YOUR TASK
Turn the brief into a narrative sequence over the clips available.

RULES - these are checked after you reply, not taken on trust
- Use ONLY these clip ids: {names}
- A beat whose clip was not analysed above will be DROPPED. Do not invent
  footage, and do not use a clip you were told nothing about.
- `moment_ref` is an INDEX into the numbered moments listed for that clip,
  or null. It is not a time. There is no field for a time, and a reply
  containing `start`, `end`, `duration` or `at` is rejected outright.
- Do NOT state statistics, view counts, dates or events. If the evidence
  above does not say it, you do not know it. A beat citing a url not listed
  above will have that citation stripped.
- A clip MAY appear in more than one beat if the narrative genuinely returns
  to it. Do not pad: fewer honest beats beat more invented ones.
- Order the beats as they should run. Open with a `hook`; end with an
  `ending`.

Reply with ONLY a JSON object, no prose around it:

{{
  "logline": "<the whole video in one sentence>",
  "hook": "<what grabs someone in the first seconds, described>",
  "setup": "<what the viewer needs to understand early>",
  "progression": "<how it builds from setup to payoff>",
  "payoff": "<what the viewer came for>",
  "ending": "<how it closes>",
  "themes": ["<what it is really about, beyond the surface>"],
  "beats": [
    {{"role": "hook|setup|develop|turn|payoff|ending",
      "clip_id": "<exact id>", "moment_ref": <index or null>,
      "narrative": "<what happens here, in the telling>",
      "why": "<why this beat sits here>",
      "support": "vision|research|both|none",
      "evidence_urls": ["<url from the evidence above>"]}}
  ],
  "risks": ["<what could make this narrative fall flat>"],
  "limitations": ["<what you could not determine from what you were given>"],
  "confidence": <0.0-1.0 - be low when you were given little to work with>
}}"""


# ------------------------------------------------------------- sanitising

def _num(raw, lo: float, hi: float, fallback: float = 0.0) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    return fallback if not math.isfinite(v) else max(lo, min(hi, v))


def _text(raw, limit: int = 200) -> str:
    if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
        return ""
    return " ".join(str(raw).split())[:limit]


def _texts(raw, limit: int = MAX_LIST, each: int = 200) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw[:limit]:
        t = _text(x, each)
        if t:
            out.append(t)
    return out


def _one_of(raw, allowed: tuple, fallback: str) -> str:
    v = raw if isinstance(raw, str) else ""
    return v if v in allowed else fallback


class StoryAgent(Agent):
    id = "story"
    produces = "Story"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                clip_set: ClipAnalysisSet,
                strategy: ContentStrategy | None = None,
                vision: VisionAnalysis | None = None,
                research: ResearchReport | None = None,
                revision: int = 0) -> Artifact[Story]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        # What we ACTUALLY have, not what was passed in. A strategy flagged
        # insufficient is not a brief, and a vision artifact with nothing
        # analysed is not a reading of the footage.
        s_ok = bool(strategy is not None and not strategy.insufficient_signal)
        v_ok = bool(vision is not None and vision.analysed_clips)
        r_ok = bool(research is not None and not research.insufficient_signal
                    and research.evidence)

        names = [c.clip_id for c in clip_set.usable_clips]

        # ---- cache: keyed on all three upstreams -----------------------
        # No time bucket: a story goes stale when the brief or the footage
        # reading changes, never on the clock.
        inputs_hash = self.hash_inputs(
            domain, sorted(names), PROMPT_VERSION, ldesc.get("model"),
            ldesc.get("provider"),
            self._strategy_digest(strategy),
            self._vision_digest(vision),
            self._research_digest(research),
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Story",
                                         inputs_hash, Story)
        if cached is not None:
            span.cache(1)
            span.note("reusing the story: strategy, vision, research, domain "
                      "and prompt are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, clip_set, strategy, vision, research, domain,
                revision, started, inputs_hash, s_ok, v_ok, r_ok,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(clip_set, strategy, vision, research,
                                    names)
        span.tool("llm.complete_json", purpose="story",
                  strategy=s_ok, vision=v_ok, research=r_ok,
                  prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=3000)
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"story reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, clip_set, strategy, vision, research, domain,
                revision, started, inputs_hash, s_ok, v_ok, r_ok,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, domain, vision, research, names,
                                        s_ok, v_ok, r_ok, ldesc)
        for note in notes:
            span.note(note)

        ctx.emit("story: %s | %d beat(s), %d grounded | confidence %.2f"
                 % (payload.logline or "no logline", len(payload.beats),
                    len(payload.grounded_beats), payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, s_ok, v_ok, r_ok),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, clip_set: ClipAnalysisSet,
                      strategy: ContentStrategy | None,
                      vision: VisionAnalysis | None,
                      research: ResearchReport | None,
                      names: list[str]) -> str:
        """Each upstream digested deliberately - never a raw JSON dump.

        Blind JSON pasting is how an agent ends up reasoning over fields it
        does not understand and citing ids that were never real.
        """
        return PROMPT.format(
            strategy=self._strategy_section(strategy),
            vision=self._vision_section(vision, clip_set),
            research=self._research_section(research),
            names=", ".join(names) or "(none)")

    @staticmethod
    def _strategy_section(strategy: ContentStrategy | None) -> str:
        if strategy is None or strategy.insufficient_signal:
            why = ("no strategy was produced for this job" if strategy is None
                   else "; ".join(strategy.limitations[:2])
                   or "the strategy reported insufficient signal")
            return ("THE BRIEF\n"
                    f"  None available ({why}).\n"
                    "  Build the narrative from the footage alone, and say so "
                    "in `limitations`.\n")
        bits = ["THE BRIEF"]
        for label, val in (("topic", strategy.central_topic),
                           ("angle", strategy.core_angle),
                           ("audience", strategy.audience),
                           ("objective", strategy.objective),
                           ("shape", strategy.narrative_direction),
                           ("engagement", strategy.engagement_strategy)):
            if val:
                bits.append(f"  {label:<11}{val}")
        bits.append(f"  {'pacing':<11}{strategy.pacing}"
                    + (f" - {strategy.pacing_rationale}"
                       if strategy.pacing_rationale else ""))
        if strategy.hook.clip_id or strategy.hook.description:
            bits.append("  %-11s%s%s" % (
                "hook idea", strategy.hook.clip_id or strategy.hook.description,
                f" ({strategy.hook.why})" if strategy.hook.why else ""))
        if strategy.grounded_points:
            bits.append("  points the video must land:")
            bits += [f"    - {k.point}" for k in strategy.grounded_points[:6]]
        if strategy.clip_priorities:
            bits.append("  clip importance (NOT running order):")
            bits += ["    %d. %-28s %s" % (c.priority, c.clip_id, c.role)
                     for c in sorted(strategy.clip_priorities,
                                     key=lambda c: c.priority)[:8]]
        if not strategy.live:
            bits.append("  NOTE: this brief was assembled deterministically, "
                        "not reasoned - weigh it lightly.")
        return "\n".join(bits) + "\n"

    @staticmethod
    def _vision_section(vision: VisionAnalysis | None,
                        clip_set: ClipAnalysisSet) -> str:
        if vision is None or not vision.analysed_clips:
            rows = "\n".join("  %s  (%.1fs)"
                             % (c.clip_id, c.technical.duration)
                             for c in clip_set.usable_clips)
            return ("\nTHE FOOTAGE\n"
                    "  No semantic analysis is available, so nothing is known "
                    "about what is\n  actually in frame. Do NOT describe "
                    "content you were not told about.\n"
                    "  The clips are:\n" + rows +
                    "\n  Say in `limitations` that the story was built without "
                    "visual understanding.\n")
        lines = ["\nTHE FOOTAGE"]
        for c in vision.analysed_clips:
            lines.append(f"  {c.clip_id}: {c.summary or 'unclear'}")
            if c.subjects:
                lines.append(f"     subjects: {', '.join(c.subjects)}")
            if c.actions:
                lines.append(f"     action:   {', '.join(c.actions)}")
            if c.interesting_moments:
                lines.append("     moments (reference these BY INDEX):")
                lines += ["       [%d] %s" % (i, m.what)
                          for i, m in enumerate(c.interesting_moments[:5])]
            else:
                lines.append("     moments:  none measured - use null")
        if vision.story:
            lines.append(f"\n  one reading of the sequence: {vision.story}")
        if vision.hook_clip:
            lines.append(f"  strongest opener by that reading: "
                         f"{vision.hook_clip}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _research_section(research: ResearchReport | None) -> str:
        if research is None or research.insufficient_signal \
                or not research.evidence:
            return ("\nOUTSIDE EVIDENCE\n"
                    "  None available. Do NOT state trends, statistics or "
                    "audience numbers -\n  you do not have them.\n")
        rows = ["\nOUTSIDE EVIDENCE",
                "  Cite these urls and no others."]
        for e in research.evidence[:8]:
            rows.append("    - %s\n      url: %s" % (e.title, e.url))
        if research.trends:
            rows.append("  patterns read from it:")
            rows += ["    - %s: %s" % (t.label, t.observation)
                     for t in research.trends[:4]]
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, domain: str,
                  vision: VisionAnalysis | None,
                  research: ResearchReport | None, names: list[str],
                  s_ok: bool, v_ok: bool, r_ok: bool, ldesc: dict
                  ) -> tuple[Story, list[str]]:
        """Everything the model returned is untrusted input."""
        known_urls = {e.url for e in research.evidence} if research else set()
        analysed = ({c.clip_id: c for c in vision.analysed_clips}
                    if vision else {})
        valid = set(names)

        beats, notes = self._beats(raw.get("beats"), known_urls, analysed,
                                   valid)

        research_basis = sorted({u for b in beats for u in b.evidence_urls})
        vision_basis = sorted({b.clip_id for b in beats})

        limitations = _texts(raw.get("limitations"))
        if not s_ok:
            limitations.append(
                "written without a usable content strategy: the narrative was "
                "shaped from the footage alone, with no brief about "
                "audience or angle")
        if not v_ok:
            limitations.append(
                "written without visual understanding: no semantic analysis of "
                "the footage was available, so no beat describes what is "
                "actually in frame")
        if not r_ok:
            limitations.append(
                "written without outside evidence: nothing here reflects what "
                "audiences are currently watching")
        if notes:
            limitations.append("; ".join(notes))

        grounded = [b for b in beats if b.support != "none"]
        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        # A model's self-assessment cannot outrun what it was given.
        if not v_ok and not s_ok:
            confidence = min(confidence, 0.25)
        elif not v_ok or not s_ok:
            confidence = min(confidence, 0.6)
        if not grounded:
            confidence = min(confidence, 0.3)

        logline = _text(raw.get("logline"), 240)
        payload = Story(
            domain=domain,
            story_type="reasoned" if (s_ok or v_ok) else "degraded",
            logline=logline,
            hook=_text(raw.get("hook"), 300),
            setup=_text(raw.get("setup"), 300),
            progression=_text(raw.get("progression"), 400),
            payoff=_text(raw.get("payoff"), 300),
            ending=_text(raw.get("ending"), 300),
            beats=beats, themes=_texts(raw.get("themes"), each=120),
            strategy_available=s_ok, vision_available=v_ok,
            research_available=r_ok,
            research_basis=research_basis[:MAX_LIST],
            vision_basis=vision_basis[:MAX_BEATS],
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not (logline and beats),
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    @staticmethod
    def _beats(raw, known_urls: set, analysed: dict, valid: set
               ) -> tuple[list[StoryBeat], list[str]]:
        """Build the beat list, dropping what cannot be grounded.

        A beat is DROPPED (not downgraded) when its clip was not analysed,
        because unlike a key point a beat carries a clip into the edit - an
        invented one would put real footage in a real cut for a reason nobody
        verified. A bad CITATION only downgrades, because the narrative
        thought behind it may still be sound.
        """
        out: list[StoryBeat] = []
        bad_clip = bad_cite = bad_ref = 0
        if not isinstance(raw, list):
            return out, []

        for item in raw[:MAX_BEATS]:
            if not isinstance(item, dict):
                continue
            cid = item.get("clip_id")
            if not isinstance(cid, str) or cid not in valid \
                    or cid not in analysed:
                bad_clip += 1
                continue                       # unverifiable footage => dropped

            # a moment reference is an INDEX, only into moments that exist
            ref = item.get("moment_ref")
            moment_ref = None
            if ref is not None:
                if isinstance(ref, int) and not isinstance(ref, bool) \
                        and 0 <= ref < len(analysed[cid].interesting_moments):
                    moment_ref = ref
                else:
                    bad_ref += 1

            urls = [u for u in (item.get("evidence_urls") or [])
                    if isinstance(u, str)][:4]
            real_urls = [u for u in urls if u in known_urls]
            if len(real_urls) < len(urls):
                bad_cite += 1

            # the clip is always one Vision analysed (anything else was
            # dropped above), so every surviving beat has visual support;
            # a real citation adds research on top of it
            support = "both" if real_urls else "vision"

            out.append(StoryBeat(
                index=len(out) + 1,            # renumbered: 1..n, contiguous
                role=_one_of(item.get("role"), ROLES, "develop"),
                clip_id=cid, moment_ref=moment_ref,
                narrative=_text(item.get("narrative"), 300),
                why=_text(item.get("why"), 240),
                support=support, evidence_urls=real_urls))

        notes = []
        if bad_clip:
            notes.append(f"{bad_clip} beat(s) named a clip that was never "
                         f"analysed and were dropped")
        if bad_cite:
            notes.append(f"{bad_cite} beat(s) cited a url that was never "
                         f"retrieved")
        if bad_ref:
            notes.append(f"{bad_ref} beat(s) gave a moment reference that "
                         f"does not exist")
        return out, notes

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan,
                       clip_set: ClipAnalysisSet,
                       strategy: ContentStrategy | None,
                       vision: VisionAnalysis | None,
                       research: ResearchReport | None, domain: str,
                       revision: int, started: float, inputs_hash: str,
                       s_ok: bool, v_ok: bool, r_ok: bool, why: str,
                       cost: Cost | None = None) -> Artifact[Story]:
        """A real sequence from what is on hand, and no more than that.

        Deliberately plain prose. Writing a logline and a three-act arc here
        would put authorship nobody performed into an artifact that declares
        itself assembled, which is worse than the gap it papers over.
        """
        span.fallback(f"deterministic story ({why})")

        usable = clip_set.usable_clips
        # order from whatever is actually known: the strategy's priorities
        # when there is one, the semantic hook reading next, motion last
        if s_ok and strategy is not None and strategy.clip_priorities:
            order = [c for c in strategy.ordered_clips
                     if c in {u.clip_id for u in usable}]
            basis = "ContentStrategy clip priorities"
        elif v_ok and vision is not None:
            order = [c.clip_id for c in sorted(
                vision.analysed_clips, key=lambda c: -c.hook_score)]
            basis = "Vision hook scores"
        else:
            order = [c.clip_id for c in sorted(
                usable, key=lambda c: -c.motion.mean_ydif)]
            basis = "measured motion (mean YDIF)"

        analysed = ({c.clip_id: c for c in vision.analysed_clips}
                    if vision else {})
        beats: list[StoryBeat] = []
        for i, cid in enumerate(order[:MAX_BEATS]):
            if i == 0:
                role = "hook"
            elif i == len(order[:MAX_BEATS]) - 1 and len(order) > 1:
                role = "ending"
            else:
                role = "develop"
            vclip = analysed.get(cid)
            beats.append(StoryBeat(
                index=len(beats) + 1, role=role, clip_id=cid,
                moment_ref=None,
                narrative=(vclip.summary if vclip and vclip.summary else ""),
                why=f"position {i + 1} by {basis}",
                support="vision" if vclip and vclip.summary else "none",
                evidence_urls=[]))

        logline = ""
        if s_ok and strategy is not None and strategy.central_topic:
            logline = strategy.central_topic
        elif domain:
            logline = domain
        elif v_ok and vision is not None and vision.story:
            logline = vision.story

        limitations = [f"{why}; this story was assembled from the available "
                       f"evidence, not authored"]
        if not s_ok:
            limitations.append("no usable content strategy was available "
                               "(absent, or it reported insufficient "
                               "signal), so the narrative has no brief "
                               "behind it")
        if not v_ok:
            limitations.append("no semantic analysis of the footage was "
                               "available, so no beat describes what is "
                               "actually in frame")
        if not r_ok:
            limitations.append("no outside evidence was available")

        minimal = not (beats or logline)
        payload = Story(
            domain=domain,
            story_type="minimal" if minimal else "degraded",
            logline=_text(logline, 240),
            hook="", setup="",
            progression=f"clips in order of {basis}; no arc was authored",
            payoff="", ending="",
            beats=beats,
            themes=[],
            strategy_available=s_ok, vision_available=v_ok,
            research_available=r_ok,
            research_basis=[], vision_basis=[b.clip_id for b in beats],
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        ctx.emit("story: %s (%s) - %d beat(s) ordered by %s"
                 % (payload.story_type, why, len(beats), basis))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, s_ok, v_ok, r_ok, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic story ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- digests

    @staticmethod
    def _strategy_digest(strategy: ContentStrategy | None):
        """What about the brief would change the story if it changed."""
        if strategy is None:
            return None
        return {
            "topic": strategy.central_topic, "angle": strategy.core_angle,
            "audience": strategy.audience, "objective": strategy.objective,
            "pacing": strategy.pacing,
            "direction": strategy.narrative_direction,
            "hook": strategy.hook.clip_id,
            "points": sorted(k.point for k in strategy.grounded_points),
            "order": list(strategy.ordered_clips),
            "insufficient": strategy.insufficient_signal,
        }

    @staticmethod
    def _vision_digest(vision: VisionAnalysis | None):
        if vision is None:
            return None
        return sorted((c.clip_id, c.summary, tuple(c.subjects),
                       tuple(m.what for m in c.interesting_moments))
                      for c in vision.analysed_clips)

    @staticmethod
    def _research_digest(research: ResearchReport | None):
        if research is None:
            return None
        return {"urls": sorted(e.url for e in research.evidence),
                "trends": sorted(t.label for t in research.trends),
                "insufficient": research.insufficient_signal}

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Story, s_ok: bool, v_ok: bool, r_ok: bool,
                   why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("ContentStrategy", s_ok),
                                    ("Vision", v_ok), ("Research", r_ok))
                    if ok] or ["neither"])
        out = [Decision(
            what="Evidence the story was built from",
            chose=" + ".join(sources),
            why=("recorded explicitly so a narrative written without a brief "
                 "or without a reading of the footage cannot later be read as "
                 "though it had both"),
            decided_by="measurement", confidence=1.0)]

        if not p.live:
            out.append(Decision(
                what="Narrative", chose=f"{p.story_type} (deterministic)",
                why=(f"{why}; beats ordered from the evidence on hand rather "
                     f"than authored - saying so plainly beats presenting "
                     f"assembly as storytelling"),
                decided_by="fallback", confidence=1.0))
            return out

        if p.logline:
            out.append(Decision(
                what="Logline", chose=p.logline[:160],
                why=(f"hook: {p.hook or 'none stated'}"[:200]),
                decided_by="llm", confidence=p.confidence))
        if p.beats:
            g = len(p.grounded_beats)
            roles = " -> ".join(b.role for b in p.beats[:6])
            out.append(Decision(
                what="Beat sequence",
                chose=f"{len(p.beats)} beat(s): {roles}",
                why=("every beat names a clip Vision actually analysed - one "
                     "that did not was dropped, because a beat carries a clip "
                     "into the edit; no beat expresses a time"),
                decided_by="llm",
                confidence=round(g / len(p.beats), 2)))
            out.append(Decision(
                what="Narrative order",
                chose=" -> ".join(p.clip_order[:6]),
                why=("the order the story should be TOLD in - the "
                     "EditingDirector still chooses the cut and every in/out "
                     "point, which this cannot express"),
                decided_by="llm", confidence=p.confidence))
        if p.payoff:
            out.append(Decision(
                what="Payoff", chose=p.payoff[:160],
                why=(p.ending or "no ending stated")[:200],
                decided_by="llm", confidence=p.confidence))
        if p.limitations:
            out.append(Decision(
                what="What this story cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["StoryAgent"]
