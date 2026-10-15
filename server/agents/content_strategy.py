"""
ContentStrategy Agent - what the video should COMMUNICATE.

WHERE IT SITS
-------------
    Vision  (what is visually there)  --+
                                        +--> ContentStrategy --> EditingDirector
    Research (what the world says)    --+

Vision and Research produce UNDERSTANDING. The EditingDirector produces an
EDIT. Between them there was nothing deciding what the video is actually
about, so the director was inferring a message from motion statistics. This
agent is that missing layer, and it is the only one allowed to answer "what
are we saying, to whom, and why".

WHAT IT MAY NOT DO
------------------
No timestamps, no filters, no commands, no paths, no title, no thumbnail, no
voiceover, no publishing. `ClipPriority` forbids extra fields, so a reply
carrying `start`/`end`/`duration` fails validation rather than being quietly
ignored - the guard is structural, not a promise. The director still owns the
edit; this hands it an intent to plan against.

GROUNDING, ENFORCED
-------------------
Two rules do the work, and both run AFTER the model replies:

  * a key point claiming RESEARCH support must cite a url that is actually in
    the ResearchReport. An invented citation is not evidence.
  * a key point claiming VISION support must name a clip Vision actually
    analysed. An unsupported visual claim is a claim about footage nobody has
    looked at.

A claim that fails either is downgraded to `support="none"` and counted, not
silently kept. The counts land in `limitations`, so a strategy built on thin
grounding says so.

DEGRADATION
-----------
Vision missing, Research missing, both missing, no model at all - each
produces a valid strategy that states what it lacked. The pipeline never
stops here, and a degraded strategy never dresses itself up as a reasoned one.
"""

from __future__ import annotations

import math
import time

from .artifacts import (Artifact, ClipAnalysisSet, ClipPriority,
                        ContentStrategy, Cost, Decision, Hook, KeyPoint,
                        ResearchReport, VisionAnalysis)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "strategy-prompt-v1"
MAX_POINTS = 8
MAX_LIST = 8
MAX_PRIORITIES = 12

PACING = ("fast", "moderate", "slow", "varied")
ROLES = ("hook", "develop", "support", "payoff", "cutaway")
SUPPORTS = ("vision", "research", "both", "none")

PROMPT = """You are the content strategist for a short video.

Your job is to decide WHAT this video should communicate - the audience, the
angle, the points worth making. You do NOT decide how it is cut. Someone else
owns the edit, and they are good at it.

SUBJECT
{subject}
{brief}
{vision}
{research}
YOUR TASK
Decide what this video is actually about and who it is for.

RULES - these are checked after you reply, not taken on trust
- Use ONLY these clip ids: {names}
- Every key point must say what supports it:
    "vision"   you can point at a clip described above
    "research" you can cite a url from the evidence above
    "both"     both of the above
    "none"     you are reasoning generally, with nothing specific behind it
  A "research" point whose url is not in the evidence above will be
  DOWNGRADED. Do not cite a url that is not listed.
- Do NOT state statistics, view counts, audience numbers, dates or events.
  If the evidence above does not say it, you do not know it.
- Do NOT give timestamps, in/out points or durations. There is no field for
  them and a reply containing them is rejected.
- "none" is an honest answer. A thin strategy that is true beats a rich one
  that is invented.

Reply with ONLY a JSON object, no prose around it:

{{
  "audience": "<who this is for, specifically>",
  "objective": "<what the video should achieve>",
  "central_topic": "<the one thing it is about>",
  "core_angle": "<the particular take on that topic>",
  "hook": {{"clip_id": "<exact id, or null>", "description": "<what opens it>",
           "why": "<why that holds a viewer>"}},
  "narrative_direction": "<how it should unfold, in one or two sentences>",
  "key_points": [
    {{"point": "<what to communicate>", "support": "vision|research|both|none",
      "evidence_urls": ["<url from the evidence above>"],
      "clip_ids": ["<exact id>"]}}
  ],
  "clip_priorities": [
    {{"clip_id": "<exact id>", "priority": <1 = most important>,
      "role": "hook|develop|support|payoff|cutaway",
      "why": "<what this clip contributes to the message>"}}
  ],
  "pacing": "fast|moderate|slow|varied",
  "pacing_rationale": "<why that pace suits this material>",
  "engagement_strategy": "<what keeps someone watching>",
  "risks": ["<what could make this video fall flat>"],
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


class ContentStrategyAgent(Agent):
    id = "content_strategy"
    produces = "ContentStrategy"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                clip_set: ClipAnalysisSet,
                vision: VisionAnalysis | None = None,
                research: ResearchReport | None = None,
                revision: int = 0,
                vision_hash: str = "", research_hash: str = "",
                ) -> Artifact[ContentStrategy]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        # What we ACTUALLY have, as opposed to what was passed in. A vision
        # artifact with nothing analysed in it is not vision, and a research
        # report flagged insufficient is not research; treating either as
        # present is how a degraded strategy ends up looking confident.
        v_ok = bool(vision is not None and vision.analysed_clips)
        r_ok = bool(research is not None and not research.insufficient_signal
                    and research.evidence)

        names = [c.clip_id for c in clip_set.usable_clips]

        # ---- cache: keyed on BOTH upstreams ----------------------------
        # Deliberately no time bucket. Research goes stale on the clock; a
        # strategy goes stale when what it was derived from changes.
        inputs_hash = self.hash_inputs(
            domain, ctx.job.settings.brief(), sorted(names), PROMPT_VERSION,
            ldesc.get("model"),
            ldesc.get("provider"),
            vision_hash or self._vision_digest(vision),
            research_hash or self._research_digest(research),
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "ContentStrategy",
                                         inputs_hash, ContentStrategy)
        if cached is not None:
            span.cache(1)
            span.note("reusing the strategy: vision, research, domain and "
                      "prompt are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        # ---- nothing to reason with ------------------------------------
        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, clip_set, vision, research, domain, revision,
                started, inputs_hash, v_ok, r_ok,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(domain, clip_set, vision, research, names,
                                    ctx.job.settings.brief())
        span.tool("llm.complete_json", purpose="content-strategy",
                  vision=v_ok, research=r_ok, prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1800)
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"strategy reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, clip_set, vision, research, domain, revision,
                started, inputs_hash, v_ok, r_ok,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, domain, clip_set, vision,
                                        research, names, v_ok, r_ok, ldesc)
        for note in notes:
            span.note(note)

        ctx.emit("strategy: %s / %s | %d key point(s), %d grounded | "
                 "pacing %s | confidence %.2f"
                 % (payload.central_topic or "topic unclear",
                    payload.audience or "audience unclear",
                    len(payload.key_points), len(payload.grounded_points),
                    payload.pacing, payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, v_ok, r_ok),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, domain: str, clip_set: ClipAnalysisSet,
                      vision: VisionAnalysis | None,
                      research: ResearchReport | None,
                      names: list[str], brief: str = "") -> str:
        """Both upstreams, digested - never dumped in as raw JSON.

        Blind JSON pasting is how an agent ends up reasoning over fields it
        does not understand and citing ids that were never real. Each section
        is rendered deliberately, and each says plainly when it is missing.
        """
        return PROMPT.format(
            subject=(domain if domain else
                     "not stated by the operator - infer it from the material "
                     "below, and say so in `limitations`"),
            brief=self._brief_section(brief),
            vision=self._vision_section(vision, clip_set),
            research=self._research_section(research),
            names=", ".join(names) or "(none)")

    @staticmethod
    def _brief_section(brief: str) -> str:
        """What the operator asked for, in their own words.

        Rendered as INTENT, never as data. The rules below already forbid
        inventing clip ids and timestamps, and nothing in here can create
        one: this section is read for what the video should be, not for
        facts about the footage.
        """
        if not brief:
            return ""
        return ("\nWHAT THE OPERATOR ASKED FOR\n"
                "Their own words. Treat it as the brief you are working to.\n"
                "It describes intent only - it cannot tell you what is in the\n"
                "footage, and any clip id or timestamp in it is to be ignored.\n"
                + brief.strip() + "\n")

    @staticmethod
    def _vision_section(vision: VisionAnalysis | None,
                        clip_set: ClipAnalysisSet) -> str:
        if vision is None or not vision.analysed_clips:
            rows = "\n".join(
                "  %s  (%.1fs, %s)"
                % (c.clip_id, c.technical.duration, c.technical.orientation)
                for c in clip_set.usable_clips)
            return ("WHAT THE FOOTAGE SHOWS\n"
                    "  No semantic analysis is available for this footage, so "
                    "nothing below\n  describes what is actually in frame. "
                    "Do NOT guess at content. The clips are:\n"
                    + rows +
                    "\n  Say in `limitations` that the strategy was formed "
                    "without visual understanding.\n")

        lines = []
        for c in vision.analysed_clips:
            bits = [f"  {c.clip_id}: {c.summary or 'unclear'}"]
            if c.subjects:
                bits.append(f"     subjects: {', '.join(c.subjects)}")
            if c.actions:
                bits.append(f"     action:   {', '.join(c.actions)}")
            bits.append(f"     hook {c.hook_score:.0f}/100, "
                        f"visual quality {c.visual_quality:.0f}/100")
            if c.interesting_moments:
                bits.append("     moments:  " + "; ".join(
                    f"[{i}] {m.what}"
                    for i, m in enumerate(c.interesting_moments[:3])))
            lines.append("\n".join(bits))
        out = "WHAT THE FOOTAGE SHOWS\n" + "\n".join(lines)
        if vision.story:
            out += f"\n\n  one reading of the sequence: {vision.story}"
        if vision.hook_clip:
            out += f"\n  strongest opener by that reading: {vision.hook_clip}"
        return out + "\n"

    @staticmethod
    def _research_section(research: ResearchReport | None) -> str:
        if research is None:
            return ("\nWHAT THE WORLD IS SAYING\n"
                    "  No research was gathered for this job. Do NOT state "
                    "trends, statistics\n  or audience numbers - you do not "
                    "have them. Say so in `limitations`.\n")
        if research.insufficient_signal or not research.evidence:
            why = "; ".join(research.limitations[:2]) or "no evidence retrieved"
            return ("\nWHAT THE WORLD IS SAYING\n"
                    f"  Research returned insufficient signal ({why}).\n"
                    "  Do NOT state trends, statistics or audience numbers - "
                    "you do not have them.\n  Say so in `limitations`.\n")

        rows = ["\nWHAT THE WORLD IS SAYING",
                "  Retrieved from %s. Cite these urls and no others."
                % research.provider, "", "  EVIDENCE"]
        for e in research.evidence[:12]:
            rows.append("    - %s\n      url: %s\n      date: %s"
                        % (e.title, e.url,
                           e.published_at or "UNKNOWN - treat as undated"))
        if research.trends:
            rows.append("\n  TRENDS READ FROM THAT EVIDENCE")
            rows += ["    - %s: %s" % (t.label, t.observation)
                     for t in research.trends[:6]]
        if research.content_opportunities:
            rows.append("\n  OPPORTUNITIES SUGGESTED "
                        "(yours to accept or reject)")
            rows += ["    - %s" % o.idea
                     for o in research.content_opportunities[:6]]
        if research.audience_signals:
            rows.append("\n  AUDIENCE SIGNALS")
            rows += ["    - %s" % a for a in research.audience_signals[:6]]
        if research.limitations:
            rows.append("\n  WHAT THIS RESEARCH CANNOT TELL YOU")
            rows += ["    - %s" % lim for lim in research.limitations[:4]]
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, domain: str, clip_set: ClipAnalysisSet,
                  vision: VisionAnalysis | None,
                  research: ResearchReport | None, names: list[str],
                  v_ok: bool, r_ok: bool, ldesc: dict
                  ) -> tuple[ContentStrategy, list[str]]:
        """Everything the model returned is untrusted input.

        Ids are clamped to clips that exist, citations to urls that were
        actually retrieved, enums to their declared values, numbers to their
        ranges. What survives is data; none of it is ever a path, a command or
        a time.
        """
        known_urls = {e.url for e in research.evidence} if research else set()
        analysed = ({c.clip_id for c in vision.analysed_clips}
                    if vision else set())
        valid = set(names)

        key_points, kp_notes = self._key_points(
            raw.get("key_points"), known_urls, analysed, valid)
        priorities, pr_notes = self._priorities(raw.get("clip_priorities"),
                                                valid, vision)
        notes = kp_notes + pr_notes
        hook = self._hook(raw.get("hook"), valid)

        # cited grounding, reduced to what actually survived validation
        research_basis = sorted({u for k in key_points
                                 for u in k.evidence_urls})
        vision_basis = sorted({c for k in key_points for c in k.clip_ids}
                              | {p.clip_id for p in priorities})

        limitations = _texts(raw.get("limitations"))
        if not v_ok:
            limitations.append(
                "formed without visual understanding: no semantic analysis of "
                "the footage was available, so nothing here describes what is "
                "actually in frame")
        if not r_ok:
            limitations.append(
                "formed without current research: no external evidence was "
                "available, so nothing here reflects what audiences are "
                "currently watching or searching for")
        if kp_notes:
            limitations.append(
                "; ".join(kp_notes) + " - those claims are retained but "
                "marked unsupported")

        grounded = [k for k in key_points if k.support != "none"]
        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        # A model's self-assessment cannot outrun what it was given. These
        # caps are the difference between a confident strategy and a
        # justified one.
        if not v_ok and not r_ok:
            confidence = min(confidence, 0.25)
        elif not v_ok or not r_ok:
            confidence = min(confidence, 0.6)
        if not grounded:
            confidence = min(confidence, 0.3)

        topic = _text(raw.get("central_topic"), 160)
        payload = ContentStrategy(
            domain=domain,
            strategy_type="reasoned" if (v_ok or r_ok) else "degraded",
            audience=_text(raw.get("audience"), 200),
            objective=_text(raw.get("objective"), 200),
            central_topic=topic,
            core_angle=_text(raw.get("core_angle"), 240),
            hook=hook,
            narrative_direction=_text(raw.get("narrative_direction"), 400),
            key_points=key_points, clip_priorities=priorities,
            pacing=_one_of(raw.get("pacing"), PACING, "moderate"),
            pacing_rationale=_text(raw.get("pacing_rationale"), 240),
            engagement_strategy=_text(raw.get("engagement_strategy"), 300),
            vision_available=v_ok, research_available=r_ok,
            research_basis=research_basis[:MAX_LIST],
            vision_basis=vision_basis[:MAX_PRIORITIES],
            content_opportunities=[
                o.idea for o in (research.content_opportunities[:MAX_LIST]
                                 if research else [])],
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not (topic and grounded),
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    @staticmethod
    def _key_points(raw, known_urls: set, analysed: set, valid: set
                    ) -> tuple[list[KeyPoint], list[str]]:
        """Downgrade any claim that cannot show its working.

        Dropping the point outright would hide what the model actually
        reasoned; keeping it at face value would launder an invention into an
        artifact. Marking it `none` keeps the thought and removes the
        authority - and `support` is derived from what SURVIVED, never from
        what the model claimed.
        """
        out: list[KeyPoint] = []
        bad_cite = bad_clip = 0
        if not isinstance(raw, list):
            return out, []
        for item in raw[:MAX_POINTS]:
            if not isinstance(item, dict):
                continue
            text = _text(item.get("point"), 240)
            if not text:
                continue

            urls = [u for u in (item.get("evidence_urls") or [])
                    if isinstance(u, str)][:4]
            real_urls = [u for u in urls if u in known_urls]
            if len(real_urls) < len(urls):
                bad_cite += 1

            clips = [c for c in (item.get("clip_ids") or [])
                     if isinstance(c, str)][:4]
            # a clip must both exist AND have been looked at, or a "visual"
            # claim is a claim about footage nobody analysed
            real_clips = [c for c in clips if c in valid and c in analysed]
            if len(real_clips) < len(clips):
                bad_clip += 1

            if real_urls and real_clips:
                support = "both"
            elif real_urls:
                support = "research"
            elif real_clips:
                support = "vision"
            else:
                support = "none"

            out.append(KeyPoint(point=text, support=support,
                                evidence_urls=real_urls, clip_ids=real_clips))

        notes = []
        if bad_cite:
            notes.append(f"{bad_cite} key point(s) cited a url that was never "
                         f"retrieved")
        if bad_clip:
            notes.append(f"{bad_clip} key point(s) referenced a clip that was "
                         f"never analysed")
        return out, notes

    @staticmethod
    def _priorities(raw, valid: set, vision: VisionAnalysis | None
                    ) -> tuple[list[ClipPriority], list[str]]:
        out: list[ClipPriority] = []
        seen: set[str] = set()
        invented = 0
        if not isinstance(raw, list):
            return out, []
        for i, item in enumerate(raw[:MAX_PRIORITIES]):
            if not isinstance(item, dict):
                continue
            cid = item.get("clip_id")
            if not isinstance(cid, str) or cid not in valid:
                invented += 1
                continue
            if cid in seen:
                continue
            seen.add(cid)

            # a moment reference is an INDEX, and only into moments that exist
            ref = item.get("moment_ref")
            moment_ref = None
            if isinstance(ref, int) and not isinstance(ref, bool) and ref >= 0:
                vclip = vision.for_clip(cid) if vision else None
                if vclip is not None and ref < len(vclip.interesting_moments):
                    moment_ref = ref

            try:
                pr = int(item.get("priority", i + 1))
            except (TypeError, ValueError):
                pr = i + 1
            out.append(ClipPriority(
                clip_id=cid, priority=max(1, min(99, pr)),
                role=_one_of(item.get("role"), ROLES, "support"),
                why=_text(item.get("why"), 200), moment_ref=moment_ref))
        notes = ([f"{invented} clip priority(ies) named a clip that does not "
                  f"exist"] if invented else [])
        return out, notes

    @staticmethod
    def _hook(raw, valid: set) -> Hook:
        if not isinstance(raw, dict):
            return Hook()
        cid = raw.get("clip_id")
        return Hook(
            clip_id=cid if isinstance(cid, str) and cid in valid else None,
            description=_text(raw.get("description"), 240),
            why=_text(raw.get("why"), 240))

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan,
                       clip_set: ClipAnalysisSet,
                       vision: VisionAnalysis | None,
                       research: ResearchReport | None, domain: str,
                       revision: int, started: float, inputs_hash: str,
                       v_ok: bool, r_ok: bool, why: str,
                       cost: Cost | None = None) -> Artifact[ContentStrategy]:
        """A real strategy from the evidence on hand, and no more than that.

        Deliberately plain. Inventing an audience or a narrative arc here
        would put reasoning nobody performed into an artifact that declares
        itself degraded, which is worse than the gap it papers over.
        """
        span.fallback(f"deterministic strategy ({why})")

        usable = clip_set.usable_clips
        # priority order from whatever is actually known: the semantic hook
        # reading when there is one, measured motion when there is not
        if v_ok and vision is not None:
            ranked = sorted(
                usable,
                key=lambda c: -(vision.for_clip(c.clip_id).hook_score
                                if vision.for_clip(c.clip_id) else 0.0))
            basis = "Vision hook scores"
        else:
            ranked = sorted(usable, key=lambda c: -c.motion.mean_ydif)
            basis = "measured motion (mean YDIF)"

        priorities = [
            ClipPriority(clip_id=c.clip_id, priority=i + 1,
                         role="hook" if i == 0 else "support",
                         why=f"ranked {i + 1} of {len(ranked)} by {basis}")
            for i, c in enumerate(ranked[:MAX_PRIORITIES])]

        key_points: list[KeyPoint] = []
        if v_ok and vision is not None:
            for c in vision.analysed_clips[:MAX_POINTS]:
                if c.summary:
                    key_points.append(KeyPoint(
                        point=c.summary, support="vision",
                        clip_ids=[c.clip_id]))
        if r_ok and research is not None:
            for t in research.trends[:3]:
                urls = [u for u in t.evidence_urls
                        if research.evidence_for(u) is not None]
                if urls:
                    key_points.append(KeyPoint(
                        point=t.observation or t.label, support="research",
                        evidence_urls=urls[:4]))

        topic = domain or (vision.story if (v_ok and vision) else "")
        limitations = [f"{why}; this strategy was assembled deterministically "
                       f"from the available evidence, not reasoned"]
        if not v_ok:
            limitations.append("no semantic analysis of the footage was "
                               "available, so nothing here describes what is "
                               "actually in frame")
        if not r_ok:
            limitations.append("no external research was available, so "
                               "nothing here reflects current audience "
                               "interest")
        if not domain:
            limitations.append("no domain was declared for this job, so the "
                               "subject is unstated rather than inferred")

        minimal = not (key_points or topic)
        payload = ContentStrategy(
            domain=domain,
            strategy_type="minimal" if minimal else "degraded",
            central_topic=_text(topic, 160),
            narrative_direction=(
                "clips in order of %s; no narrative was authored" % basis),
            hook=Hook(clip_id=priorities[0].clip_id if priorities else None,
                      description="",
                      why=f"highest ranked by {basis}" if priorities else ""),
            key_points=key_points, clip_priorities=priorities,
            pacing="moderate",
            pacing_rationale="default pace; no strategist ran to choose one",
            vision_available=v_ok, research_available=r_ok,
            research_basis=sorted({u for k in key_points
                                   for u in k.evidence_urls})[:MAX_LIST],
            vision_basis=[p.clip_id for p in priorities][:MAX_PRIORITIES],
            content_opportunities=[
                o.idea for o in (research.content_opportunities[:MAX_LIST]
                                 if research else [])],
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        ctx.emit("strategy: %s (%s) - %d clip(s) prioritised by %s"
                 % (payload.strategy_type, why, len(priorities), basis))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, v_ok, r_ok, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic strategy ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- digests

    @staticmethod
    def _vision_digest(vision: VisionAnalysis | None):
        """What about Vision would change the strategy if it changed."""
        if vision is None:
            return None
        return sorted((c.clip_id, c.summary, tuple(c.subjects),
                       tuple(c.actions), round(c.hook_score, 1))
                      for c in vision.analysed_clips)

    @staticmethod
    def _research_digest(research: ResearchReport | None):
        if research is None:
            return None
        return {
            "urls": sorted(e.url for e in research.evidence),
            "trends": sorted(t.label for t in research.trends),
            "topics": sorted(research.topics),
            "opportunities": sorted(o.idea
                                    for o in research.content_opportunities),
            "insufficient": research.insufficient_signal,
        }

    # -------------------------------------------------------- decisions

    def _decisions(self, p: ContentStrategy, v_ok: bool, r_ok: bool,
                   why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("Vision", v_ok), ("Research", r_ok)) if ok]
                   or ["neither"])
        out = [Decision(
            what="Evidence the strategy was formed from",
            chose=" + ".join(sources),
            why=("recorded explicitly so a strategy built without one of them "
                 "cannot later be read as though it had both"),
            decided_by="measurement", confidence=1.0)]

        if not p.live:
            out.append(Decision(
                what="Strategy", chose=f"{p.strategy_type} (deterministic)",
                why=(f"{why}; ranked and summarised from the evidence on hand "
                     f"rather than reasoned - stating that plainly beats "
                     f"presenting assembly as judgement"),
                decided_by="fallback", confidence=1.0))
            return out

        if p.central_topic:
            out.append(Decision(
                what="Central topic and angle",
                chose=f"{p.central_topic} - {p.core_angle}"[:200],
                why=f"for {p.audience or 'an unstated audience'}: "
                    f"{p.objective or 'no objective stated'}"[:300],
                decided_by="llm", confidence=p.confidence))
        if p.hook.clip_id or p.hook.description:
            out.append(Decision(
                what="Recommended hook",
                chose=(p.hook.clip_id or p.hook.description)[:160],
                why=(p.hook.why or "no reason given")[:200] +
                    " - a recommendation; the EditingDirector still chooses "
                    "the opener and the exact in/out points",
                decided_by="llm", confidence=p.confidence))
        if p.key_points:
            g = len(p.grounded_points)
            out.append(Decision(
                what="Key points to communicate",
                chose=f"{len(p.key_points)} point(s), {g} grounded in evidence",
                why=("a point citing research must name a url that was "
                     "actually retrieved, and one citing the footage must "
                     "name a clip Vision analysed; the rest are marked "
                     "unsupported"),
                decided_by="llm",
                confidence=round(g / len(p.key_points), 2)))
        if p.clip_priorities:
            out.append(Decision(
                what="Clip priorities",
                chose="; ".join(f"{c.clip_id}={c.role}"
                                for c in p.clip_priorities[:4]),
                why=("importance to the message, not edit order - no "
                     "timestamps are expressed or implied here"),
                decided_by="llm", confidence=p.confidence))
        out.append(Decision(
            what="Pacing", chose=p.pacing,
            why=(p.pacing_rationale or "no rationale given")[:200] +
                " - an intent for the director, not a duration",
            decided_by="llm", confidence=p.confidence))
        if p.limitations:
            out.append(Decision(
                what="What this strategy cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["ContentStrategyAgent"]
