"""
Animation Agent - how the cut should LOOK on the canvas.

WHERE IT SITS
-------------
    Story + ContentStrategy (+ Vision)
            |
            v
        Animation           <- this agent
            |
            v
    EditingDirector

WHAT THE ENGINE CAN ACTUALLY DO - READ THIS FIRST
-------------------------------------------------
`pipeline/build_video.py::fit_filter` implements exactly two framing
treatments and nothing else:

    fill      the clip's orientation matches the canvas -> scale to fill,
              then crop
    blur_pad  it does not -> a blurred, darkened bed of the same frame with
              the fitted foreground centred over it

plus a `drawtext` caption whose position and size follow the canvas
orientation. There is NO zoompan anywhere in this project: no Ken Burns, no
pan, no zoom, no parallax, no motion graphics of any kind.

So this agent does not direct movement, because directing movement the
renderer cannot perform would be direction in name only - the same failure the
Music agent avoids by choosing from the engine's own mood list. It directs
what the engine can honour, and records anything else asked for in
`unsupported_requests` so the gap is visible rather than implied.

Between-segment transitions (xfade) are NOT this agent's. They belong to the
Transition agent, which does not exist yet.

WHAT IS DERIVED RATHER THAN CHOSEN
----------------------------------
`treatment` follows from the clip's orientation against the canvas - the same
comparison fit_filter makes - so it is computed, not asked for. `clip_id`
comes from the referenced beat. What the model actually decides is `emphasis`
(what should draw the eye) and `caption_intent` (whether this beat wants words
at all), both closed sets.

NO ASSET IS EVER NAMED
----------------------
There is no overlay library, no lower-third pack and no template in this
project. A direction referencing one would be fiction, so none can be
expressed: the schema has no field for it.
"""

from __future__ import annotations

import math
import os
import time
import typing

from .artifacts import (Animation, AnimationDirection, Artifact,
                        ClipAnalysisSet, ContentStrategy, Cost, Decision,
                        OverlayAnchor, OverlayHold, OverlayPlacement,
                        Story, VisionAnalysis)
from .base import Agent, AgentContext
from .tools import asset_library as _AL
from .trace import TraceSpan

PROMPT_VERSION = "animation-prompt-v2"

# Shown only when the job HAS graphics. With an empty folder the model is
# never given the shape, so it has no way to express a placement and
# cannot name a file that is not there.
_OVERLAY_SHAPE = """,
      "overlay": {"asset": "<one of the files listed above>",
                  "anchor": "top-left|top-center|top-right|middle-left|\
center|middle-right|bottom-left|bottom-center|bottom-right",
                  "scale_pct": <2-60>, "hold": "full|open|close"}"""
OVERLAY_ANCHORS = typing.get_args(OverlayAnchor)
OVERLAY_HOLDS = typing.get_args(OverlayHold)
MAX_DIRECTIONS = 12
MAX_LIST = 8

EMPHASES = ("none", "subject", "context", "text")
CAPTION_INTENTS = ("none", "short_label", "key_phrase")

# Words that describe movement this renderer cannot perform. A request built
# from any of these is recorded as unsupported rather than silently accepted.
MOTION_WORDS = ("zoom", "pan", "ken burns", "kenburns", "dolly", "push in",
                "pull out", "parallax", "slide", "spin", "rotate", "shake",
                "wipe", "glitch", "tilt", "track", "orbit", "morph")

PROMPT = """You are directing how a short video LOOKS on screen.

The story is already decided. Your job is what draws the eye in each beat, and
whether a beat wants words on screen. You do NOT choose the cut, the timing,
or how anything is rendered.

{story}
{strategy}
{vision}
WHAT THIS RENDERER CAN ACTUALLY DO
  It places each clip on the canvas in one of two ways, and that choice is
  made for you by whether the clip's orientation matches the output:
{capabilities}

  It has NO zoom, NO pan, NO Ken Burns, NO parallax and NO motion graphics.
  Do not ask for movement. If you think a beat needs it, say so in
  `unsupported_requests` and it will be recorded as a gap rather than
  promised.

YOUR TASK
For each beat, say what should draw the eye and whether it wants words.

RULES - these are checked after you reply, not taken on trust
- Reference a beat by its INDEX from the list above. That is the only pointer
  you give; a direction whose `beat_index` is not listed will be DROPPED.
- `emphasis` must be one of: {emphases}
- `caption_intent` must be one of: {captions}
- Do NOT give timings, durations, positions, sizes, colours, filters, file
  paths or render settings. There is no field for them and a reply containing
  `start`, `end`, `duration`, `zoom`, `filter` or `path` is rejected outright.
{assets}

Reply with ONLY a JSON object, no prose around it:

{{
  "visual_style": "<how the piece should feel to look at, in a sentence>",
  "pacing_feel": "<how busy or calm the screen should be>",
  "directions": [
    {{"beat_index": <index from the list above>,
      "emphasis": "{emphases_pipe}",
      "caption_intent": "{captions_pipe}",
      "intent": "<why this beat is presented that way>"{overlay_shape}}}
  ],
  "unsupported_requests": ["<anything you wanted that this renderer cannot do>"],
  "risks": ["<what could make this look wrong>"],
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


def _texts(raw, limit: int = MAX_LIST, each: int = 160) -> list[str]:
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


def _capabilities() -> list[str]:
    """What fit_filter can actually do, described from the real module.

    Read rather than hardcoded so that if the renderer ever grows a treatment
    this reports it, and the parity test fails loudly if one disappears.
    """
    from .tools.pipeline_bridge import build_video
    import inspect
    try:
        src = inspect.getsource(build_video.fit_filter)
    except (OSError, TypeError):                      # pragma: no cover
        src = ""
    caps = []
    if "force_original_aspect_ratio=increase" in src:
        caps.append("fill: scale the clip to cover the canvas, then crop")
    if "gblur" in src:
        caps.append("blur_pad: a blurred, darkened bed of the same frame with "
                    "the fitted clip centred over it")
    if "drawtext" in src:
        caps.append("caption: one line of text, placed and sized by the "
                    "canvas orientation")
    return caps


def _has_motion(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in MOTION_WORDS)


class AnimationAgent(Agent):
    id = "animation"
    produces = "Animation"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                clip_set: ClipAnalysisSet | None = None,
                story: Story | None = None,
                strategy: ContentStrategy | None = None,
                vision: VisionAnalysis | None = None,
                revision: int = 0) -> Artifact[Animation]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        st_ok = bool(story is not None and story.beats)
        cs_ok = bool(strategy is not None and not strategy.insufficient_signal)
        v_ok = bool(vision is not None and vision.analysed_clips)

        caps = _capabilities()

        inputs_hash = self.hash_inputs(
            domain, PROMPT_VERSION, ldesc.get("model"), ldesc.get("provider"),
            self._story_digest(story),
            self._strategy_digest(strategy),
            self._vision_digest(vision),
            self._canvas_digest(clip_set),
            caps,
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Animation",
                                         inputs_hash, Animation)
        if cached is not None:
            span.cache(1)
            span.note("reusing the visual direction: story, brief, footage "
                      "reading and canvas are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, clip_set, story, strategy, vision, domain,
                revision, started, inputs_hash, st_ok, cs_ok, v_ok, caps,
                why=(ldesc.get("why_disabled") or "no model configured"))

        # The graphics this job can actually use, written from its own
        # assets folder. Defaults are generated once so a fresh install
        # can use overlays without the operator going to find artwork;
        # a file the operator put there themselves is never replaced.
        made = _AL.ensure_defaults(ctx.job.build_dir)
        assets = _AL.available(ctx.job.build_dir)
        if made:
            span.tool("assets.generate_defaults", created=len(made))
        span.tool("assets.offer", available=len(assets))

        prompt = self._build_prompt(
            story, strategy, vision, caps,
            assets=_AL.describe(ctx.job.build_dir),
            overlay_shape=_OVERLAY_SHAPE if assets else "")
        span.tool("llm.complete_json", purpose="animation",
                  beats=len(story.beats) if st_ok else 0,
                  capabilities=len(caps), prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1600, effort="low")
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"animation reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, clip_set, story, strategy, vision, domain,
                revision, started, inputs_hash, st_ok, cs_ok, v_ok, caps,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, domain, clip_set, story,
                                        st_ok, cs_ok, v_ok, caps, ldesc,
                                        assets)
        for note in notes:
            span.note(note)

        ctx.emit("animation: %d direction(s), %d captioned beat(s), "
                 "%d unsupported request(s) | confidence %.2f"
                 % (len(payload.directions), len(payload.captioned_beats),
                    len(payload.unsupported_requests), payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, st_ok, cs_ok, v_ok),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, story: Story | None,
                      strategy: ContentStrategy | None,
                      vision: VisionAnalysis | None,
                      caps: list[str],
                      assets: str = "", overlay_shape: str = "") -> str:
        return PROMPT.format(
            assets=assets or _AL.describe(None),
            overlay_shape=overlay_shape,
            story=self._story_section(story),
            strategy=self._strategy_section(strategy),
            vision=self._vision_section(vision),
            capabilities="\n".join("    - " + c for c in caps) or
                         "    (the renderer reported no treatments)",
            emphases=", ".join(EMPHASES),
            emphases_pipe="|".join(EMPHASES),
            captions=", ".join(CAPTION_INTENTS),
            captions_pipe="|".join(CAPTION_INTENTS))

    @staticmethod
    def _overlay_for(item: dict, allowed: set, dropped: list):
        """The graphic this direction asks for, if it exists.

        A name that is not in the folder is DROPPED and recorded, never
        passed on. A model asked for `lower_third_blue.png` would
        otherwise produce an artifact that reads as a placement and a
        video with nothing on it - the gap between the two being exactly
        the kind of quiet lie this project is built to avoid.
        """
        raw = item.get("overlay")
        if not isinstance(raw, dict):
            return None
        name = os.path.basename(str(raw.get("asset") or "").strip())
        if not name:
            return None
        if name not in allowed:
            dropped.append(name)
            return None
        try:
            return OverlayPlacement(
                asset=name,
                anchor=_one_of(raw.get("anchor"), OVERLAY_ANCHORS,
                               "top-right"),
                scale_pct=float(raw.get("scale_pct") or 18.0),
                hold=_one_of(raw.get("hold"), OVERLAY_HOLDS, "full"),
                why=_text(raw.get("why"), 160))
        except (TypeError, ValueError):
            # an out-of-range scale, say: the direction survives, the
            # graphic does not
            dropped.append(name)
            return None

    @staticmethod
    def _story_section(story: Story | None) -> str:
        if story is None or not story.beats:
            return ("THE STORY\n"
                    "  No story with beats is available, so there is nothing "
                    "to present.\n  Give overall style only and leave "
                    "`directions` empty.\n")
        rows = ["THE STORY"]
        if story.logline:
            rows.append(f"  logline   {story.logline}")
        rows.append("")
        rows.append("  BEATS - reference these by index, and only these:")
        for b in story.beats:
            rows.append("    [%d] %-9s %s"
                        % (b.index, b.role,
                           b.narrative or "(no description)"))
        if not story.live:
            rows.append("")
            rows.append("  NOTE: this sequence was assembled "
                        "deterministically, not authored.")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _strategy_section(strategy: ContentStrategy | None) -> str:
        if strategy is None or strategy.insufficient_signal:
            return ("\nTHE BRIEF\n  None available. Take the look from the "
                    "story above.\n")
        rows = ["\nTHE BRIEF"]
        for label, val in (("topic", strategy.central_topic),
                           ("angle", strategy.core_angle),
                           ("audience", strategy.audience)):
            if val:
                rows.append(f"  {label:<11}{val}")
        rows.append(f"  {'pacing':<11}{strategy.pacing}")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _vision_section(vision: VisionAnalysis | None) -> str:
        if vision is None or not vision.analysed_clips:
            return ("\nWHAT IS IN FRAME\n"
                    "  No semantic analysis is available, so nothing is known "
                    "about what the\n  footage shows. Do not describe content "
                    "you were not told about.\n")
        rows = ["\nWHAT IS IN FRAME"]
        for c in vision.analysed_clips:
            rows.append(f"  {c.clip_id}: {c.summary or 'unclear'}")
            if c.subjects:
                rows.append(f"     subjects: {', '.join(c.subjects)}")
            rows.append("     visual quality %.0f/100" % c.visual_quality)
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, domain: str,
                  clip_set: ClipAnalysisSet | None, story: Story | None,
                  st_ok: bool, cs_ok: bool, v_ok: bool,
                  caps: list[str], ldesc: dict,
                  assets: list[str] | None = None
                  ) -> tuple[Animation, list[str]]:
        """Everything the model returned is untrusted input."""
        beats = {b.index: b for b in story.beats} if st_ok and story else {}
        directions, notes = self._directions(raw.get("directions"), beats,
                                             clip_set, assets)

        # anything asking for movement is a gap, not a direction
        unsupported = _texts(raw.get("unsupported_requests"), each=160)
        motion_in_style = []
        for field in ("visual_style", "pacing_feel"):
            val = _text(raw.get(field), 300)
            if val and _has_motion(val):
                motion_in_style.append(field)
        if motion_in_style:
            unsupported.append(
                "the requested look mentions movement (%s), which this "
                "renderer cannot perform" % ", ".join(motion_in_style))
            notes.append("movement was requested in %s; recorded as "
                         "unsupported" % ", ".join(motion_in_style))

        limitations = _texts(raw.get("limitations"), each=200)
        if not st_ok:
            limitations.append(
                "no story with beats was available, so there is nothing to "
                "present and only an overall look was given")
        if not v_ok:
            limitations.append(
                "no semantic analysis of the footage was available, so the "
                "emphasis is not informed by what is actually in frame")
        if unsupported:
            limitations.append(
                "%d request(s) fall outside what this renderer can do and "
                "will not happen" % len(unsupported))
        if notes:
            limitations.append("; ".join(notes))

        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        if not st_ok:
            confidence = min(confidence, 0.3)
        if not v_ok:
            confidence = min(confidence, 0.6)
        if story is not None and not story.live:
            confidence = min(confidence, 0.5)

        payload = Animation(
            domain=domain,
            animation_type="reasoned" if st_ok else "degraded",
            visual_style=_text(raw.get("visual_style"), 300),
            pacing_feel=_text(raw.get("pacing_feel"), 200),
            directions=directions,
            engine_capabilities=caps,          # measured, never model output
            unsupported_requests=unsupported[:MAX_LIST],
            story_available=st_ok, strategy_available=cs_ok,
            vision_available=v_ok,
            beats_covered=sorted({d.beat_index for d in directions}),
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not directions,
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    def _directions(self, raw, beats: dict,
                    clip_set: ClipAnalysisSet | None,
                    assets: list[str] | None = None
                    ) -> tuple[list[AnimationDirection], list[str]]:
        """Build the per-beat directions, deriving everything derivable.

        `assets` is the allowlist of graphics this job actually has. A
        name outside it is dropped and reported: an artifact that reads
        as a placement over a video with nothing on it is the quiet kind
        of wrong this project exists to avoid.
        """
        kept: list[AnimationDirection] = []
        seen: set[int] = set()
        allowed = set(assets or ())
        dropped: list[str] = []
        bad_beat = dupe = 0
        if not isinstance(raw, list):
            return [], []

        for item in raw[:MAX_DIRECTIONS]:
            if not isinstance(item, dict):
                continue
            bi = item.get("beat_index")
            if not isinstance(bi, int) or isinstance(bi, bool) \
                    or bi not in beats:
                bad_beat += 1
                continue
            if bi in seen:
                dupe += 1
                continue                   # one direction per beat
            seen.add(bi)

            beat = beats[bi]
            intent = _text(item.get("intent"), 240)
            kept.append(AnimationDirection(
                index=1,                   # renumbered below
                beat_index=bi,
                clip_id=beat.clip_id,      # from the BEAT
                treatment=self._treatment_for(beat.clip_id, clip_set),
                emphasis=_one_of(item.get("emphasis"), EMPHASES, "none"),
                caption_intent=_one_of(item.get("caption_intent"),
                                       CAPTION_INTENTS, "none"),
                intent=intent,
                overlay=self._overlay_for(item, allowed, dropped),
                # an intent asking for movement cannot be honoured
                supported=not _has_motion(intent)))

        ordered = sorted(kept, key=lambda d: d.beat_index)
        out = [d.model_copy(update={"index": i + 1})
               for i, d in enumerate(ordered)]

        notes = []
        if bad_beat:
            notes.append(f"{bad_beat} direction(s) referenced a beat that "
                         f"does not exist and were dropped")
        if dropped:
            notes.append(f"{len(dropped)} overlay(s) named a graphic this "
                         f"job does not have and were dropped: "
                         f"{', '.join(sorted(set(dropped))[:4])}")
        if dupe:
            notes.append(f"{dupe} duplicate direction(s) for a beat already "
                         f"covered were dropped")
        unsup = [d for d in out if not d.supported]
        if unsup:
            notes.append(f"{len(unsup)} direction(s) asked for movement this "
                         f"renderer cannot perform and are marked unsupported")
        return out, notes

    @staticmethod
    def _treatment_for(clip_id: str,
                       clip_set: ClipAnalysisSet | None) -> str:
        """The same comparison `fit_filter` makes, made here.

        fit_filter asks whether the clip's orientation matches the canvas and
        picks fill or blur_pad from the answer. That is a fact about the
        footage, not a preference, so it is derived rather than chosen.
        """
        if clip_set is None:
            return "fill"
        clip = clip_set.by_id(clip_id)
        if clip is None:
            return "fill"
        cw, ch = clip_set.canvas
        canvas_orientation = "vertical" if ch > cw else "horizontal"
        return ("fill" if clip.technical.orientation == canvas_orientation
                else "blur_pad")

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan,
                       clip_set: ClipAnalysisSet | None, story: Story | None,
                       strategy: ContentStrategy | None,
                       vision: VisionAnalysis | None, domain: str,
                       revision: int, started: float, inputs_hash: str,
                       st_ok: bool, cs_ok: bool, v_ok: bool,
                       caps: list[str], why: str,
                       cost: Cost | None = None) -> Artifact[Animation]:
        """The framing the renderer would apply anyway, stated explicitly.

        Every treatment here is the one fit_filter will actually use, computed
        from the same orientation comparison. Nothing about emphasis or
        captions is invented: with no model there is no basis for either, so
        both stay at "none".
        """
        span.fallback(f"deterministic visual direction ({why})")

        directions: list[AnimationDirection] = []
        if st_ok and story is not None:
            for i, b in enumerate(story.beats[:MAX_DIRECTIONS]):
                directions.append(AnimationDirection(
                    index=i + 1, beat_index=b.index, clip_id=b.clip_id,
                    treatment=self._treatment_for(b.clip_id, clip_set),
                    emphasis="none",        # no basis to judge
                    caption_intent="none",  # no basis to judge
                    intent="", supported=True))

        limitations = [f"{why}; the framing below is what the renderer will "
                       f"apply regardless - no emphasis or caption intent was "
                       f"judged"]
        if not st_ok:
            limitations.append("no story with beats was available, so there "
                               "are no per-beat directions")
        if not v_ok:
            limitations.append("no semantic analysis of the footage was "
                               "available")

        payload = Animation(
            domain=domain,
            animation_type="minimal" if not directions else "degraded",
            visual_style="", pacing_feel="",
            directions=directions,
            engine_capabilities=caps,
            unsupported_requests=[],
            story_available=st_ok, strategy_available=cs_ok,
            vision_available=v_ok,
            beats_covered=sorted({d.beat_index for d in directions}),
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        ctx.emit("animation: %s (%s) - %d direction(s), framing only"
                 % (payload.animation_type, why, len(directions)))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, st_ok, cs_ok, v_ok, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic visual direction ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- digests

    @staticmethod
    def _story_digest(story: Story | None):
        if story is None:
            return None
        return {"logline": story.logline,
                "beats": [(b.index, b.role, b.clip_id, b.narrative)
                          for b in story.beats],
                "live": story.live}

    @staticmethod
    def _strategy_digest(strategy: ContentStrategy | None):
        if strategy is None:
            return None
        return {"topic": strategy.central_topic, "angle": strategy.core_angle,
                "pacing": strategy.pacing,
                "insufficient": strategy.insufficient_signal}

    @staticmethod
    def _vision_digest(vision: VisionAnalysis | None):
        if vision is None:
            return None
        return sorted((c.clip_id, c.summary, tuple(c.subjects),
                       round(c.visual_quality, 1))
                      for c in vision.analysed_clips)

    @staticmethod
    def _canvas_digest(clip_set: ClipAnalysisSet | None):
        """Orientation against the canvas decides every treatment, so it is
        part of this artifact's identity."""
        if clip_set is None:
            return None
        return {"canvas": list(clip_set.canvas),
                "clips": sorted((c.clip_id, c.technical.orientation)
                                for c in clip_set.clips)}

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Animation, st_ok: bool, cs_ok: bool, v_ok: bool,
                   why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("Story", st_ok),
                                    ("ContentStrategy", cs_ok),
                                    ("Vision", v_ok)) if ok] or ["neither"])
        out = [Decision(
            what="Evidence the visual direction was built from",
            chose=" + ".join(sources),
            why=("recorded explicitly so direction written without a reading "
                 "of the footage cannot later be read as though it had one"),
            decided_by="measurement", confidence=1.0)]

        out.append(Decision(
            what="What the renderer can do",
            chose=f"{len(p.engine_capabilities)} treatment(s): "
                  + ", ".join(c.split(":")[0]
                              for c in p.engine_capabilities),
            why=("read from pipeline/build_video.py::fit_filter rather than "
                 "assumed - there is no zoom, pan or motion of any kind in "
                 "this renderer, so none is directed"),
            decided_by="measurement", confidence=1.0))

        if p.directions:
            out.append(Decision(
                what="Framing per beat",
                chose=" -> ".join(p.treatments[:8]),
                why=("derived from each clip's orientation against the "
                     "canvas, which is the same comparison fit_filter makes; "
                     "this is a fact about the footage, not a preference"),
                decided_by="measurement", confidence=1.0))

        if not p.live:
            out.append(Decision(
                what="Visual direction",
                chose=f"{p.animation_type} - framing only",
                why=(f"{why}; emphasis and caption intent need judgement and "
                     f"there was none available, so neither was invented"),
                decided_by="fallback", confidence=1.0))
            return out

        if p.visual_style:
            out.append(Decision(
                what="Look", chose=p.visual_style[:160],
                why=(p.pacing_feel or "no pacing note given")[:200],
                decided_by="llm", confidence=p.confidence))
        if any(d.emphasis != "none" for d in p.directions):
            out.append(Decision(
                what="Emphasis per beat",
                chose=", ".join(f"{d.beat_index}:{d.emphasis}"
                                for d in p.directions
                                if d.emphasis != "none")[:200],
                why=("what should draw the eye - the one judgement on a "
                     "direction the model genuinely makes"),
                decided_by="llm", confidence=p.confidence))
        if p.captioned_beats:
            out.append(Decision(
                what="Beats wanting words on screen",
                chose=", ".join(str(i) for i in p.captioned_beats[:8]),
                why=("an intent only - the Caption agent does not exist yet, "
                     "so nothing writes these words today"),
                decided_by="llm", confidence=p.confidence))
        if p.unsupported_requests:
            out.append(Decision(
                what="Asked for but not possible",
                chose=f"{len(p.unsupported_requests)} request(s)",
                why=("; ".join(p.unsupported_requests[:2])[:250]
                     + " - recorded as a gap rather than promised"),
                decided_by="policy", confidence=1.0))
        if p.limitations:
            out.append(Decision(
                what="What this direction cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["AnimationAgent"]
