"""
Vision Agent - semantic understanding of the footage.

WHAT IT IS FOR
--------------
ClipAnalysis can tell you a clip measures 9.7 mean YDIF between 4.2s and 7.8s.
It cannot tell you that stretch is the moment the dish comes out of the oven,
the summit comes into view, or the product is first switched on. Vision adds
that layer, and nothing else in the system can.

DOMAIN NEUTRAL, DELIBERATELY
----------------------------
The prompt describes whatever the operator says the footage is - a recipe, a
trail run, a match, an unboxing - or says plainly that the subject is
unspecified. There are no LEGO assumptions, no fixed shot taxonomy and no
enumerated "kinds". The existing server/vision.py hardcodes "a LEGO build" in
its prompts; this agent reuses that module's genuinely neutral machinery
(frame sampling, the defensive JSON parser) and leaves its prompts behind.

THE RULE THIS AGENT EXISTS TO ENFORCE
-------------------------------------
The model may NOT author a timestamp. Deterministic code scans each clip's
measured YDIF series and offers a short list of candidate windows; the model
returns an INDEX into that list. So "which stretch is the strongest reveal" is
a judgement the model makes, while "where exactly that stretch is" stays a
measurement. Every other number the model returns - moment offsets, scores -
is clamped to the clip's real duration and to 0-100 before it is stored.

Everything the model returns is untrusted input. Clip ids are clamped to ones
actually uploaded, and no model output ever becomes a path, an argument or a
filter string.

FALLBACK
--------
With no API key the agent still produces a VisionAnalysis: the measured
candidate windows, no semantics, `live=False` and `degraded=True`. Downstream
agents get the windows and simply have no interpretation to use - which is the
state the system was in before this agent existed, made explicit rather than
implied.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from .artifacts import (Artifact, CandidateWindow, ClipAnalysis,
                        ClipAnalysisSet, Cost, Decision, InterestingMoment,
                        VisionAnalysis, VisionClip)
from .base import Agent, AgentContext
from .tools import select_tools as ST
from .tools.pipeline_bridge import SERVER_DIR  # noqa: F401  (sys.path)
from .trace import TraceSpan

import vision as _legacy_vision    # existing module, UNMODIFIED

# Frames are the cost driver, so the cap is explicit and configurable. The env
# name is the one the legacy vision layer already documents.
DEFAULT_FRAMES = 4
MAX_FRAMES = 8
DEFAULT_MAX_CLIPS = 12

CLIP_PROMPT = """You are analysing frames from ONE clip of {subject}.

{n} frames are shown, evenly spaced across a clip that is {dur:.1f} seconds long.

Describe only what you can actually see. "unclear" is far more useful to me
than a confident guess.

MEASURED WINDOWS
These stretches were measured from the footage itself. You may NOT invent your
own timestamps - choose one of these by its index, or use null if none of them
captures the most interesting moment:

{windows}

Reply with ONLY a JSON object, no prose around it:

{{
  "summary": "<one sentence on what this clip shows>",
  "subjects": ["<the main things visible>"],
  "actions": ["<what is being done, if anything>"],
  "scene_description": "<the setting, in a few words>",
  "interesting_moments": [
    {{"at": <seconds from clip start, 0-{dur:.1f}>, "what": "<what happens>"}}
  ],
  "hook_score": <0-100, how well this clip would OPEN a short video>,
  "visual_quality": <0-100, framing, focus, lighting, steadiness>,
  "chosen_window": <index from the list above, or null>,
  "window_rationale": "<why that window, in a few words>",
  "confidence": <0-100, how sure you are about `summary`>,
  "notes": "<anything else worth knowing, or empty>"
}}"""

CROSS_PROMPT = """Here are descriptions of {n} clips from one shoot of {subject},
each already analysed.

{summaries}

Decide how they fit together as ONE short video.

Reply with ONLY a JSON object:

{{
  "order": ["<clip>", "..."],
  "hook": "<the clip that should open, or null>",
  "duplicates": [["<clip>", "<clip that covers the same thing>"]],
  "story": "<one sentence describing the sequence you chose>"
}}

Use ONLY these exact clip names: {names}
If two clips genuinely show different things, do NOT call them duplicates.
If no clip is a clear opener, set "hook" to null."""


def _subject(domain: str) -> str:
    """How the prompt refers to the footage. Never assumes a subject."""
    domain = (domain or "").strip()
    return domain if domain else "an unspecified subject - work it out from the frames"


def _frames_per_clip() -> int:
    try:
        return max(1, min(MAX_FRAMES,
                          int(os.environ.get("AT_VISION_FRAMES", DEFAULT_FRAMES))))
    except (TypeError, ValueError):
        return DEFAULT_FRAMES


def _max_clips() -> int:
    try:
        return max(1, int(os.environ.get("AT_VISION_MAX_CLIPS",
                                         DEFAULT_MAX_CLIPS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CLIPS


def _num(raw, lo: float, hi: float, fallback: float = 0.0) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    return max(lo, min(hi, v))


def _str_list(raw, limit: int = 8, each: int = 60) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x)[:each] for x in raw[:limit] if isinstance(x, (str, int, float))]


class VisionAgent(Agent):
    id = "vision"
    produces = "VisionAnalysis"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                clip_set: ClipAnalysisSet,
                revision: int = 0) -> Artifact[VisionAnalysis]:
        started = time.time()
        domain = ctx.job.settings.domain
        describe = ctx.llm.describe()

        inputs_hash = self.hash_inputs(
            [(c.clip_id, c.technical.duration, c.motion.mean_ydif)
             for c in clip_set.clips],
            domain, describe.get("provider"), describe.get("model"),
            _frames_per_clip(), "vision-prompt-v1",
        )

        # Existing cache convention: same footage, same domain, same model and
        # same prompt version means the previous answer still stands.
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "VisionAnalysis",
                                         inputs_hash, VisionAnalysis)
        if cached is not None:
            span.note(f"inputs unchanged - reusing VisionAnalysis "
                      f"rev{cached.revision}")
            span.cache(len(cached.payload.clips))
            span.skipped("cache hit")
            return cached

        usable = clip_set.usable_clips[:_max_clips()]
        if not usable:
            raise RuntimeError("no usable clips to analyse")

        # ---- deterministic first: windows come from the measurements -------
        windows: dict[str, list[dict]] = {}
        for c in usable:
            span.tool("select.candidate_windows", clip=c.clip_id)
            windows[c.clip_id] = ST.candidate_windows(c)

        if not ctx.llm.available():
            return self._fallback(ctx, span, clip_set, usable, windows,
                                  inputs_hash, revision, started,
                                  describe.get("why_disabled") or
                                  "no LLM configured")

        # ---- semantic pass, one call per clip ------------------------------
        clips: dict[str, VisionClip] = {}
        failures: list[str] = []
        tok_in = tok_out = 0
        usd = 0.0

        for c in usable:
            got, cost = self._analyse_clip(ctx, span, c, windows[c.clip_id],
                                           domain)
            tok_in += cost[0]
            tok_out += cost[1]
            usd += cost[2]
            clips[c.clip_id] = got
            if got.failure:
                failures.append(f"{c.clip_id}: {got.failure}")

        analysed = [c for c in clips.values() if c.analysed]
        if not analysed:
            span.note("every clip failed - falling back to measurements only")
            return self._fallback(ctx, span, clip_set, usable, windows,
                                  inputs_hash, revision, started,
                                  "the model produced nothing usable",
                                  partial=clips, failures=failures,
                                  cost=(tok_in, tok_out, usd))

        # ---- cross-clip pass: summaries only, no images --------------------
        cross, cross_cost = self._analyse_across(ctx, span, analysed, domain)
        tok_in += cross_cost[0]
        tok_out += cross_cost[1]
        usd += cross_cost[2]

        payload = VisionAnalysis(
            domain=domain, provider=describe.get("provider"),
            model=describe.get("model"), frames_per_clip=_frames_per_clip(),
            live=True, clips=clips, failures=failures, **cross)

        decisions = self._decisions(payload, live=True)
        for c in analysed:
            ctx.emit("  %-26s %s (hook %.0f, quality %.0f, conf %.0f)"
                     % (c.clip_id, (c.summary or "")[:44], c.hook_score,
                        c.visual_quality, c.confidence))

        confidence = round(
            sum(c.confidence for c in analysed) / len(analysed) / 100.0, 3)
        return self.wrap(ctx, payload, revision=revision,
                         inputs_hash=inputs_hash, span=span,
                         decisions=decisions, confidence=confidence,
                         degraded=bool(failures),
                         fallback_used=None,
                         cost=Cost(tokens_in=tok_in, tokens_out=tok_out,
                                   usd=round(usd, 6),
                                   wall_s=round(time.time() - started, 2)))

    # --------------------------------------------------------- per clip

    def _analyse_clip(self, ctx: AgentContext, span: TraceSpan,
                      clip: ClipAnalysis, windows: list[dict],
                      domain: str) -> tuple[VisionClip, tuple[int, int, float]]:
        cw = [CandidateWindow(**w) for w in windows]
        dur = clip.technical.duration

        span.tool("probe.sample_frames", clip=clip.clip_id,
                  frames=_frames_per_clip())
        frames = _legacy_vision.sample_frames(Path(clip.path), dur,
                                              _frames_per_clip())
        if not frames:
            return (VisionClip(clip_id=clip.clip_id, analysed=False,
                               candidate_windows=cw,
                               failure="could not sample frames"),
                    (0, 0, 0.0))

        listing = "\n".join(
            "  [%d] %.1f-%.1fs (%.1fs), measured motion %.2f"
            % (w["index"], w["start"], w["end"], w["duration"], w["mean_ydif"])
            for w in windows) or "  (none - this clip could not be measured)"

        prompt = CLIP_PROMPT.format(subject=_subject(domain), n=len(frames),
                                    dur=dur, windows=listing)
        span.tool("llm.complete_json", clip=clip.clip_id, images=len(frames),
                  prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1200,
                                           images=frames)
        cost = (reply.tokens_in, reply.tokens_out, reply.usd)
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok:
            return (VisionClip(clip_id=clip.clip_id, analysed=False,
                               candidate_windows=cw,
                               failure=f"call failed: {reply.error}"), cost)
        if not isinstance(raw, dict):
            return (VisionClip(clip_id=clip.clip_id, analysed=False,
                               candidate_windows=cw,
                               failure="reply was not a JSON object"), cost)

        return self._clean(clip, raw, cw), cost

    def _clean(self, clip: ClipAnalysis, raw: dict,
               cw: list[CandidateWindow]) -> VisionClip:
        """Range-check and clamp one reply. The model is untrusted input."""
        dur = clip.technical.duration

        moments = []
        for m in (raw.get("interesting_moments") or [])[:6]:
            if not isinstance(m, dict):
                continue
            at = _num(m.get("at"), 0.0, dur, -1.0)
            if at < 0:
                continue                       # unparseable, drop it
            moments.append(InterestingMoment(at=round(at, 2),
                                             what=str(m.get("what") or "")[:120]))

        # the ONLY way the model can point at a time range: an index we offered
        chosen = raw.get("chosen_window")
        valid = {w.index for w in cw}
        if not isinstance(chosen, (int, float)) or int(chosen) not in valid:
            chosen = None
        else:
            chosen = int(chosen)

        return VisionClip(
            clip_id=clip.clip_id, analysed=True,
            summary=str(raw.get("summary") or "")[:200],
            subjects=_str_list(raw.get("subjects")),
            actions=_str_list(raw.get("actions")),
            scene_description=str(raw.get("scene_description") or "")[:200],
            interesting_moments=moments,
            hook_score=_num(raw.get("hook_score"), 0, 100),
            visual_quality=_num(raw.get("visual_quality"), 0, 100),
            candidate_windows=cw,
            chosen_window=chosen,
            window_rationale=str(raw.get("window_rationale") or "")[:200],
            confidence=_num(raw.get("confidence"), 0, 100),
            notes=str(raw.get("notes") or "")[:200],
        )

    # ------------------------------------------------------- across clips

    def _analyse_across(self, ctx: AgentContext, span: TraceSpan,
                        analysed: list[VisionClip],
                        domain: str) -> tuple[dict, tuple[int, int, float]]:
        known = {c.clip_id for c in analysed}
        if len(analysed) < 2:
            return ({"ordering_signal": [c.clip_id for c in analysed],
                     "hook_clip": analysed[0].clip_id if analysed else None,
                     "duplicates": [], "story": ""}, (0, 0, 0.0))

        summaries = "\n".join(
            "- %s: %s | subjects: %s | hook %.0f | quality %.0f | conf %.0f"
            % (c.clip_id, c.summary or "unclear", ", ".join(c.subjects) or "-",
               c.hook_score, c.visual_quality, c.confidence)
            for c in analysed)
        prompt = CROSS_PROMPT.format(n=len(analysed), subject=_subject(domain),
                                     summaries=summaries,
                                     names=", ".join(sorted(known)))
        span.tool("llm.complete_json", pass_="cross-clip", images=0,
                  clips=len(analysed))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=800)
        cost = (reply.tokens_in, reply.tokens_out, reply.usd)
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note("cross-clip pass unusable - per-clip analysis kept")
            return ({"ordering_signal": [], "hook_clip": None,
                     "duplicates": [], "story": ""}, cost)

        # every id clamped to one we actually analysed
        order = [n for n in (raw.get("order") or []) if n in known]
        if len(order) != len(known):
            order = []                         # a partial order is not an order
        hook = raw.get("hook")
        dups = []
        for pair in (raw.get("duplicates") or [])[:8]:
            if (isinstance(pair, (list, tuple)) and len(pair) >= 2
                    and pair[0] in known and pair[1] in known
                    and pair[0] != pair[1]):
                dups.append((pair[0], pair[1]))
        return ({"ordering_signal": order,
                 "hook_clip": hook if hook in known else None,
                 "duplicates": dups,
                 "story": str(raw.get("story") or "")[:300]}, cost)

    # ---------------------------------------------------------- fallback

    def _fallback(self, ctx: AgentContext, span: TraceSpan,
                  clip_set: ClipAnalysisSet, usable: list[ClipAnalysis],
                  windows: dict, inputs_hash: str, revision: int,
                  started: float, why: str, partial: dict | None = None,
                  failures: list[str] | None = None,
                  cost: tuple = (0, 0, 0.0)) -> Artifact[VisionAnalysis]:
        """Measured windows, no semantics. Honest about what is missing."""
        span.fallback(f"measurements only ({why})")
        clips = dict(partial or {})
        for c in usable:
            if c.clip_id in clips:
                continue
            clips[c.clip_id] = VisionClip(
                clip_id=c.clip_id, analysed=False,
                candidate_windows=[CandidateWindow(**w)
                                   for w in windows[c.clip_id]],
                failure=why)

        payload = VisionAnalysis(
            domain=ctx.job.settings.domain, provider=None, model=None,
            frames_per_clip=0, live=False, clips=clips,
            failures=list(failures or []))

        ctx.emit("vision: %s - candidate windows measured for %d clip(s), "
                 "no semantic analysis" % (why, len(clips)))
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, live=False, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"measurements only ({why})",
            cost=Cost(tokens_in=cost[0], tokens_out=cost[1], usd=cost[2],
                      wall_s=round(time.time() - started, 2)))

    # --------------------------------------------------------- decisions

    def _decisions(self, payload: VisionAnalysis, live: bool,
                   why: str = "") -> list[Decision]:
        out: list[Decision] = []
        if not live:
            out.append(Decision(
                what="Semantic analysis",
                chose="not performed - measurements only",
                why=(f"{why}; the candidate windows below were still measured "
                     f"from the footage, so downstream agents keep the timing "
                     f"data and simply have no interpretation to use"),
                decided_by="fallback", confidence=1.0))
            return out

        analysed = payload.analysed_clips
        out.append(Decision(
            what="What the footage shows",
            chose=f"{len(analysed)} of {len(payload.clips)} clip(s) analysed",
            why="; ".join(f"{c.clip_id}: {c.summary}" for c in analysed[:4])
                or "nothing legible",
            decided_by="llm",
            confidence=round(
                sum(c.confidence for c in analysed) / len(analysed) / 100.0, 2)
                if analysed else 0.0))

        picked = [c for c in analysed if c.chosen_window is not None]
        if picked:
            out.append(Decision(
                what="Strongest measured window per clip",
                chose="; ".join(
                    "%s -> %.1f-%.1fs" % (c.clip_id, c.best_window().start,
                                          c.best_window().end)
                    for c in picked[:4]),
                why=("the model chose from windows the pipeline MEASURED; it "
                     "cannot author a timestamp, only pick one of these: "
                     + "; ".join(c.window_rationale for c in picked[:3]
                                 if c.window_rationale)),
                decided_by="llm", confidence=0.7))

        if payload.hook_clip:
            best = payload.for_clip(payload.hook_clip)
            out.append(Decision(
                what="Suggested opener", chose=payload.hook_clip,
                why=(f"highest hook score ({best.hook_score:.0f}) across the "
                     f"batch: {best.summary}" if best else "cross-clip pass"),
                decided_by="llm", confidence=0.65))

        if payload.duplicates:
            out.append(Decision(
                what="Redundant material",
                chose=f"{len(payload.duplicates)} pair(s)",
                why="; ".join(f"{a} ~ {b}" for a, b in payload.duplicates),
                decided_by="llm", confidence=0.6))

        if payload.story:
            out.append(Decision(
                what="Story reading", chose=payload.story[:120],
                why="the sequence the model would cut these into",
                decided_by="llm", confidence=0.6))
        return out


__all__ = ["VisionAgent"]
