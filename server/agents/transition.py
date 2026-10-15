"""
Transition Agent - how one beat gives way to the next.

WHERE IT SITS
-------------
    Story + Animation (+ ContentStrategy)
            |
            v
        Transition          <- this agent
            |
            v
    EditingDirector

WHAT THE RENDERER ACTUALLY DOES
-------------------------------
`pipeline/build_video.py::stage_join` crossfades EVERY junction:

    xfade=transition=<name>:duration=T:offset=<measured>

with `<name>` cycling through the `XFADES` list and `T` coming from
`AT_XFADE_T`. Three consequences, and this agent is built around all three
rather than around what a transition agent might normally assume:

  * the vocabulary is the six names the renderer ships - fade, fadewhite,
    smoothleft, smoothup, wipeleft, wipeup - each confirmed to exist in
    ffmpeg's xfade filter. A seventh name would make the render fail.
  * there is NO hard cut. Every junction crossfades and xfade has no "none"
    option, so "cut here" is not a thing this renderer can be asked for.
  * the length is GLOBAL. `T` applies to every transition equally, so a
    per-junction duration cannot be expressed - and the schema has no field
    for one.

WHERE THE OUTPUT GOES
---------------------
`RenderSpec.xfade_list` already feeds `AT_XFADES`, which build_video reads.
So `xfade_sequence` - one name per junction, in order - is directly
actionable through a path that already exists. Nothing new had to be built to
make this reach the renderer.

WHAT IS DERIVED RATHER THAN CHOSEN
----------------------------------
Junctions come from the Story: n beats give exactly n-1 junctions, in order.
The model does not get to invent a junction, reorder them, or attach one to
footage the narrative never anchored - it chooses the KIND at each, and
nothing else.

FALLBACK
--------
With no model, every junction gets `fade` - the plainest transition the
renderer has and the one its default list uses most. That is a real choice
the engine can execute, not an invention, and it is marked as assembled.
"""

from __future__ import annotations

import math
import re
import time

from .artifacts import (Animation, Artifact, ContentStrategy, Cost, Decision,
                        Story, Transition, TransitionChoice)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "transition-prompt-v1"
MAX_CHOICES = 16
MAX_LIST = 8

# Requests this renderer cannot honour, in the words a model would use.
UNSUPPORTED_WORDS = ("cut", "hard cut", "jump cut", "none", "no transition",
                     "dissolve to black", "fade to black", "zoom", "spin",
                     "glitch", "whip", "slide in", "morph", "3d", "flip",
                     "circle", "iris", "pixelize", "dreamy", "radial")

PROMPT = """You are choosing how each moment of a short video gives way to the
next.

The story is already decided. Your job is the transition at each junction
between consecutive beats - and nothing else.

{story}
{animation}
{strategy}
WHAT THIS RENDERER CAN ACTUALLY DO
  It crossfades EVERY junction, using one of exactly these:
{kinds}

  Three things you cannot ask for, because the renderer cannot do them:
    - a hard cut. Every junction crossfades; there is no "none" option.
    - a different length for one transition. The fade length is one global
      setting applied to all of them equally.
    - anything not in the list above. A different name would make the render
      fail outright.

  If you want one of those, put it in `unsupported_requests` and it will be
  recorded as a gap rather than promised.

YOUR TASK
For each junction listed below, choose the transition.

RULES - these are checked after you reply, not taken on trust
- The junctions are fixed. Use exactly the `from_beat`/`to_beat` pairs given;
  a choice that names any other pair will be DROPPED.
- `kind` must be one of: {kind_names}
- Do NOT give durations, timings, offsets, positions or filter strings. There
  is no field for them and a reply containing `duration`, `start`, `offset`
  or `filter` is rejected outright.
- Do NOT name a clip or a file. Junctions are identified by beat, and the
  clips come from the story.

THE JUNCTIONS
{junctions}

Reply with ONLY a JSON object, no prose around it:

{{
  "style": "<the overall feel of how this piece moves, in a sentence>",
  "choices": [
    {{"from_beat": <index>, "to_beat": <index>,
      "kind": "{kind_pipe}",
      "intent": "<why that transition here>"}}
  ],
  "unsupported_requests": ["<anything you wanted that this renderer cannot do>"],
  "risks": ["<what could make this feel wrong>"],
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


def _engine_transitions() -> list[str]:
    """The names the renderer ships, parsed from its own default XFADES list.

    Read from the module SOURCE rather than the imported value, because
    XFADES honours an AT_XFADES override and the canonical vocabulary is the
    default the renderer was written with, not whatever happens to be in this
    process's environment.
    """
    import inspect

    from .tools.pipeline_bridge import build_video
    try:
        src = inspect.getsource(build_video)
    except (OSError, TypeError):                      # pragma: no cover
        return []
    m = re.search(r"XFADES = \(.*?else\s*(\[.*?\])", src, re.S)
    if not m:                                         # pragma: no cover
        return []
    try:
        return sorted(set(x for x in eval(m.group(1))       # noqa: S307
                          if isinstance(x, str)))
    except Exception:                                 # noqa: BLE001
        return []                                     # pragma: no cover


def _is_unsupported(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in UNSUPPORTED_WORDS)


class TransitionAgent(Agent):
    id = "transition"
    produces = "Transition"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                story: Story | None = None,
                animation: Animation | None = None,
                strategy: ContentStrategy | None = None,
                revision: int = 0) -> Artifact[Transition]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        st_ok = bool(story is not None and len(story.beats) >= 2)
        an_ok = bool(animation is not None and animation.directions)
        cs_ok = bool(strategy is not None and not strategy.insufficient_signal)

        kinds = _engine_transitions()
        junctions = self._junctions(story)

        inputs_hash = self.hash_inputs(
            domain, PROMPT_VERSION, ldesc.get("model"), ldesc.get("provider"),
            self._story_digest(story),
            self._animation_digest(animation),
            self._strategy_digest(strategy),
            kinds,
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Transition",
                                         inputs_hash, Transition)
        if cached is not None:
            span.cache(1)
            span.note("reusing the transitions: story, look, brief and the "
                      "renderer's vocabulary are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        # ---- nothing to transition between -----------------------------
        if not junctions:
            return self._deterministic(
                ctx, span, story, animation, strategy, domain, revision,
                started, inputs_hash, st_ok, an_ok, cs_ok, kinds, junctions,
                why=("there are fewer than two beats, so there is no junction "
                     "to transition across"))

        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, story, animation, strategy, domain, revision,
                started, inputs_hash, st_ok, an_ok, cs_ok, kinds, junctions,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(story, animation, strategy, kinds,
                                    junctions)
        span.tool("llm.complete_json", purpose="transition",
                  junctions=len(junctions), kinds=len(kinds),
                  prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1400, effort="low")
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"transition reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, story, animation, strategy, domain, revision,
                started, inputs_hash, st_ok, an_ok, cs_ok, kinds, junctions,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, domain, kinds, junctions,
                                        st_ok, an_ok, cs_ok, ldesc)
        for note in notes:
            span.note(note)

        ctx.emit("transition: %s | %d unsupported request(s) | "
                 "confidence %.2f"
                 % (" -> ".join(payload.xfade_sequence) or "none",
                    len(payload.unsupported_requests), payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, st_ok, an_ok, cs_ok),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # ---------------------------------------------------------- junctions

    @staticmethod
    def _junctions(story: Story | None) -> list[tuple[int, int]]:
        """n beats give exactly n-1 junctions, in order.

        Derived from the Story, never from the reply: the model chooses what
        happens AT a junction, not which junctions exist.
        """
        if story is None or len(story.beats) < 2:
            return []
        idx = [b.index for b in story.beats]
        return list(zip(idx, idx[1:]))

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, story: Story | None,
                      animation: Animation | None,
                      strategy: ContentStrategy | None,
                      kinds: list[str],
                      junctions: list[tuple[int, int]]) -> str:
        return PROMPT.format(
            story=self._story_section(story),
            animation=self._animation_section(animation),
            strategy=self._strategy_section(strategy),
            kinds="\n".join("    - " + k for k in kinds) or
                  "    (the renderer reported no transitions)",
            kind_names=", ".join(kinds),
            kind_pipe="|".join(kinds),
            junctions="\n".join(
                "    beat %d -> beat %d" % (a, b) for a, b in junctions))

    @staticmethod
    def _story_section(story: Story | None) -> str:
        if story is None or not story.beats:
            return ("THE STORY\n  No story is available.\n")
        rows = ["THE STORY"]
        if story.logline:
            rows.append(f"  logline   {story.logline}")
        rows.append("")
        rows.append("  BEATS:")
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
    def _animation_section(animation: Animation | None) -> str:
        if animation is None or not animation.directions:
            return ("\nTHE LOOK\n  No visual direction is available.\n")
        rows = ["\nTHE LOOK"]
        if animation.visual_style:
            rows.append(f"  style: {animation.visual_style}")
        rows.append("  framing per beat:")
        for d in animation.directions:
            bits = [d.treatment]
            if d.emphasis != "none":
                bits.append("emphasis " + d.emphasis)
            rows.append("    beat %d: %s" % (d.beat_index, ", ".join(bits)))
        rows.append("  A beat framed differently from the one before it may "
                    "want a softer join.")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _strategy_section(strategy: ContentStrategy | None) -> str:
        if strategy is None or strategy.insufficient_signal:
            return ("\nTHE BRIEF\n  None available. Take the feel from the "
                    "story above.\n")
        rows = ["\nTHE BRIEF"]
        if strategy.central_topic:
            rows.append(f"  topic     {strategy.central_topic}")
        rows.append(f"  pacing    {strategy.pacing}"
                    + (f" - {strategy.pacing_rationale}"
                       if strategy.pacing_rationale else ""))
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, domain: str, kinds: list[str],
                  junctions: list[tuple[int, int]],
                  st_ok: bool, an_ok: bool, cs_ok: bool, ldesc: dict
                  ) -> tuple[Transition, list[str]]:
        """Everything the model returned is untrusted input."""
        choices, notes = self._choices(raw.get("choices"), kinds, junctions)

        unsupported = _texts(raw.get("unsupported_requests"), each=160)
        style = _text(raw.get("style"), 300)
        if style and _is_unsupported(style):
            unsupported.append(
                "the requested feel asks for something this renderer cannot "
                "do - every junction crossfades, at one global length")
            notes.append("the overall style asked for an effect the renderer "
                         "does not have; recorded as unsupported")

        limitations = _texts(raw.get("limitations"), each=200)
        if not st_ok:
            limitations.append(
                "no story with two or more beats was available, so there is "
                "no junction to transition across")
        if not an_ok:
            limitations.append(
                "no visual direction was available, so the transitions are "
                "not informed by how each beat is framed")
        limitations.append(
            "this renderer crossfades every junction at one global length; "
            "a hard cut and a per-transition duration are not expressible")
        if unsupported:
            limitations.append(
                "%d request(s) fall outside what this renderer can do and "
                "will not happen" % len(unsupported))
        if notes:
            limitations.append("; ".join(notes))

        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        if not st_ok:
            confidence = min(confidence, 0.3)
        if not an_ok:
            confidence = min(confidence, 0.7)

        payload = Transition(
            domain=domain,
            transition_type="reasoned" if choices else "degraded",
            style=style,
            choices=choices,
            engine_transitions=kinds,        # measured, never model output
            unsupported_requests=unsupported[:MAX_LIST],
            story_available=st_ok, animation_available=an_ok,
            strategy_available=cs_ok,
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not choices,
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    @staticmethod
    def _choices(raw, kinds: list[str], junctions: list[tuple[int, int]]
                 ) -> tuple[list[TransitionChoice], list[str]]:
        """One choice per junction, in the Story's order.

        The junction list is fixed by the narrative. A reply naming a pair
        that is not in it is describing a transition between beats that never
        meet, so it is dropped. A junction the reply skipped still gets a
        transition, because the renderer will crossfade it regardless - so the
        honest thing is to state which one rather than leave a gap the
        renderer fills silently.
        """
        wanted = {(a, b): None for a, b in junctions}
        bad_pair = bad_kind = 0
        if isinstance(raw, list):
            for item in raw[:MAX_CHOICES]:
                if not isinstance(item, dict):
                    continue
                fb, tb = item.get("from_beat"), item.get("to_beat")
                if not (isinstance(fb, int) and isinstance(tb, int)) \
                        or isinstance(fb, bool) or isinstance(tb, bool) \
                        or (fb, tb) not in wanted:
                    bad_pair += 1
                    continue
                if wanted[(fb, tb)] is not None:
                    continue                 # first answer for a junction wins

                k = item.get("kind")
                intent = _text(item.get("intent"), 200)
                if not isinstance(k, str) or k not in kinds:
                    bad_kind += 1
                    k = "fade"               # the renderer's plainest
                    supported = False
                else:
                    supported = not _is_unsupported(intent)
                wanted[(fb, tb)] = (k, intent, supported)

        out: list[TransitionChoice] = []
        filled = 0
        for i, (a, b) in enumerate(junctions):
            got = wanted[(a, b)]
            if got is None:
                got, filled = ("fade", "", True), filled + 1
            kind, intent, supported = got
            out.append(TransitionChoice(
                index=i + 1, from_beat=a, to_beat=b, kind=kind,
                intent=intent, supported=supported))

        notes = []
        if bad_pair:
            notes.append(f"{bad_pair} choice(s) named a junction that does "
                         f"not exist and were dropped")
        if bad_kind:
            notes.append(f"{bad_kind} choice(s) named a transition this "
                         f"renderer does not have; fell back to fade")
        if filled:
            notes.append(f"{filled} junction(s) were not chosen for and "
                         f"default to fade, because the renderer crossfades "
                         f"them either way")
        return out, notes

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan,
                       story: Story | None, animation: Animation | None,
                       strategy: ContentStrategy | None, domain: str,
                       revision: int, started: float, inputs_hash: str,
                       st_ok: bool, an_ok: bool, cs_ok: bool,
                       kinds: list[str], junctions: list[tuple[int, int]],
                       why: str,
                       cost: Cost | None = None) -> Artifact[Transition]:
        """Plain crossfades. The renderer's own plainest option, everywhere.

        `fade` is chosen because it is the most common entry in the renderer's
        shipped default list and the least assertive thing a join can do.
        Nothing is invented: every name here is one the engine can execute.
        """
        span.fallback(f"deterministic transitions ({why})")

        choices = [TransitionChoice(index=i + 1, from_beat=a, to_beat=b,
                                    kind="fade", intent="", supported=True)
                   for i, (a, b) in enumerate(junctions)]

        limitations = [f"{why}; every junction uses the renderer's plainest "
                       f"crossfade - no transition was chosen for its effect"]
        if not st_ok:
            limitations.append("no story with two or more beats was "
                               "available")
        if not an_ok:
            limitations.append("no visual direction was available")
        limitations.append(
            "this renderer crossfades every junction at one global length; "
            "a hard cut and a per-transition duration are not expressible")

        payload = Transition(
            domain=domain,
            transition_type="minimal" if not choices else "degraded",
            style="", choices=choices,
            engine_transitions=kinds, unsupported_requests=[],
            story_available=st_ok, animation_available=an_ok,
            strategy_available=cs_ok,
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        ctx.emit("transition: %s (%s) - %d junction(s), all fade"
                 % (payload.transition_type, why, len(choices)))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, st_ok, an_ok, cs_ok, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic transitions ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- digests

    @staticmethod
    def _story_digest(story: Story | None):
        if story is None:
            return None
        return {"logline": story.logline,
                "beats": [(b.index, b.role, b.clip_id) for b in story.beats],
                "live": story.live}

    @staticmethod
    def _animation_digest(animation: Animation | None):
        if animation is None:
            return None
        return {"style": animation.visual_style,
                "framing": [(d.beat_index, d.treatment, d.emphasis)
                            for d in animation.directions]}

    @staticmethod
    def _strategy_digest(strategy: ContentStrategy | None):
        if strategy is None:
            return None
        return {"topic": strategy.central_topic, "pacing": strategy.pacing,
                "insufficient": strategy.insufficient_signal}

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Transition, st_ok: bool, an_ok: bool,
                   cs_ok: bool, why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("Story", st_ok),
                                    ("Animation", an_ok),
                                    ("ContentStrategy", cs_ok)) if ok]
                   or ["neither"])
        out = [Decision(
            what="Evidence the transitions were chosen from",
            chose=" + ".join(sources),
            why=("recorded explicitly so choices made without a reading of "
                 "how each beat is framed cannot later be read as though "
                 "they had one"),
            decided_by="measurement", confidence=1.0)]

        out.append(Decision(
            what="What the renderer can do",
            chose=f"{len(p.engine_transitions)} transition(s): "
                  + ", ".join(p.engine_transitions),
            why=("parsed from the XFADES default in build_video rather than "
                 "assumed; there is no hard cut and the length is one global "
                 "setting, so neither can be asked for"),
            decided_by="measurement", confidence=1.0))

        if p.choices:
            out.append(Decision(
                what="Junctions",
                chose=", ".join(f"{c.from_beat}->{c.to_beat}"
                                for c in p.choices[:6]),
                why=("n beats give exactly n-1 junctions, taken from the "
                     "story; the model chooses what happens at one, never "
                     "which ones exist"),
                decided_by="measurement", confidence=1.0))

        if not p.live:
            out.append(Decision(
                what="Transitions", chose=f"{p.transition_type} - all fade",
                why=(f"{why}; the renderer's plainest crossfade everywhere, "
                     f"which it can execute, rather than an effect chosen "
                     f"for a reason nobody had"),
                decided_by="fallback", confidence=1.0))
            return out

        out.append(Decision(
            what="Transition per junction",
            chose=" -> ".join(p.xfade_sequence[:8]),
            why=((p.style or "no overall feel stated")[:200]
                 + " - this list feeds AT_XFADES, which the renderer already "
                   "reads"),
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
                what="What these transitions cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["TransitionAgent"]
