"""
Caption Agent - which of the words already written also appear on screen.

WHERE IT SITS
-------------
    Story + Voiceover + Animation (+ ClipAnalysis, for the canvas)
            |
            v
        Caption             <- this agent
            |
            v
    EditingDirector

WHAT THE RENDERER ACTUALLY DOES
-------------------------------
`pipeline/build_video.py::fit_filter` draws AT MOST ONE caption per segment:

    drawtext=text='<caption>':fontsize=48|56:fontcolor=white:borderw=3:
             bordercolor=black@0.7:x=(w-text_w)/2:y=h-160|h-470:
             enable='between(t,0.25,<segment length - 0.35>)'

Four consequences, and this agent is built around all of them:

  * one position and one style. Bottom-centre, white with a dark outline,
    sized by canvas orientation. A reply asking for the top, a colour, a box
    or an animation is recorded as unsupported - it cannot happen.
  * no timing. A caption spans its segment less fixed margins, so there is no
    field for a start, an end or a duration.
  * one line, never wrapped. `max_chars` is derived from the renderer's own
    font size against the canvas width; a longer line runs off the frame.
  * `esc()` silently strips some characters. They are removed here first, so
    the artifact says exactly what will be drawn.

All of it is READ from the renderer's source rather than assumed, and parity
tests fail loudly if any of it changes.

NO WORD IS EVER WRITTEN HERE
----------------------------
Every caption is a verbatim span of text an upstream agent already produced
and validated: the beat's narration (Voiceover) or its story line (Story). The
model chooses WHICH words; the agent then finds them in that beat's source and
copies the source's own span. A phrase that is not there is dropped, so
fabricated text cannot reach the screen.

WHICH BEATS
-----------
Animation's `caption_intent` says whether a beat has room for words. When
Animation judged it, only the beats it marked are eligible and the intent is
copied from it. When it did not, the model may choose, and the artifact says
that nobody judged the look.

FALLBACK
--------
With no model, a beat Animation marked for words gets the opening clause of
its validated text, cut at a word boundary to fit - a verbatim extract, marked
as assembled. With no judged intent there is no basis to caption anything, so
nothing is captioned. Silence is a legitimate output; invented words are not.
"""

from __future__ import annotations

import inspect
import math
import re
import time
from dataclasses import dataclass

from .artifacts import (Animation, Artifact, Caption, CaptionLine,
                        ClipAnalysisSet, Cost, Decision, Story, Voiceover)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "caption-prompt-v1"
MAX_LINES = 16
MAX_LIST = 8

# Words per caption, by intent. Policy rather than measurement: a label names
# something, a phrase lands one point, and neither is a sentence to be read.
WORD_CAP = {"short_label": 5, "key_phrase": 12}

# The character budget is an ESTIMATE, and says so: a bold sans glyph advances
# about 0.6 em on average, and a 10% margin keeps the line off the frame edge.
GLYPH_EM = 0.6
SAFE_WIDTH = 0.9

# Budgeted against when the canvas is unknown: the narrower of the two the
# renderer produces, so the estimate can only err short.
FALLBACK_CANVAS = (1080, 1920)

# Keys a reply might use to ask for what the renderer cannot vary.
STYLE_KEYS = ("position", "placement", "style", "color", "colour", "font",
              "size", "fontsize", "box", "background", "animation", "effect")
TIMING_KEYS = ("start", "end", "duration", "at", "seconds", "offset")
# ...and the values that merely restate what it already does.
NOT_A_REQUEST = {"bottom_center", "bottom_centre", "bottom", "center",
                 "centre", "outlined_white", "white", "default"}

_WORD = re.compile(r"\w+")
_CLAUSE = re.compile(r"[.!?;:—]|\s[-–]\s")

PROMPT = """You are choosing which words appear ON SCREEN in a short video.

Nothing here is written from scratch. Every caption must be words copied
exactly from the narration or story line of the beat it appears on.

{beats}
WHAT THIS RENDERER CAN ACTUALLY DRAW
{engine}

  You cannot ask for a position, colour, size, box, animation or timing - the
  renderer has exactly one of each. If you want one, put it in
  `unsupported_requests` and it will be recorded as a gap rather than promised.

YOUR TASK
For each beat listed above, decide whether it carries a caption and, if so,
which of its words.

RULES - these are checked after you reply, not taken on trust
- Use ONLY the beats listed. A caption on any other beat is DROPPED.
- `text` must be words copied EXACTLY, in order, from that beat's own text.
  Words that are not there - a paraphrase, a summary, a word from another
  beat - are DROPPED. Do not invent anything.
- One line: at most {max_chars} characters. A short_label is at most
  {label_words} words; a key_phrase at most {phrase_words}.
- Leave a beat out entirely if none of its words are worth showing.
- Do NOT give times, durations, positions, colours, fonts, filter strings or
  file paths. There is no field for any of them.

Reply with ONLY a JSON object, no prose around it:

{{
  "captions": [
    {{"beat": <index>, "text": "<words copied exactly from that beat>",
      "intent": "short_label|key_phrase",
      "why": "<why these words, on this beat>"}}
  ],
  "unsupported_requests": ["<anything you wanted that this renderer cannot draw>"],
  "risks": ["<what could make these captions feel wrong>"],
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


# ------------------------------------------------------- the real renderer

def _engine_caption() -> dict:
    """What fit_filter's drawtext actually does, read from the renderer.

    From the SOURCE, the way the Transition agent reads XFADES and the
    Animation agent reads fit_filter: if the renderer ever grows a second
    position or a style option this stops matching, and the parity tests say
    so instead of the agent silently under- or over-promising.
    """
    from .tools.pipeline_bridge import build_video
    try:
        src = inspect.getsource(build_video.fit_filter)
        esc_src = inspect.getsource(build_video.esc)
    except (OSError, TypeError, AttributeError):      # pragma: no cover
        return {"drawtext": False}

    out: dict = {"drawtext": "drawtext" in src}
    m = re.search(r'y = "(h-\d+)" if ch > cw else "(h-\d+)"', src)
    out["y"] = ({"vertical": m.group(1), "horizontal": m.group(2)}
                if m else {})
    m = re.search(r"size = (\d+) if ch > cw else (\d+)", src)
    out["size"] = ({"vertical": int(m.group(1)),
                    "horizontal": int(m.group(2))} if m else {})
    m = re.search(r"size, y, ([\d.]+), max\(dur - ([\d.]+)", src)
    out["margins"] = [float(m.group(1)), float(m.group(2))] if m else []
    out["positions"] = (["bottom_center"]
                        if "x=(w-text_w)/2" in src and out["y"] else [])
    out["styles"] = (["outlined_white"]
                     if "fontcolor=white" in src and "borderw=" in src
                     else [])
    # esc() drops these outright rather than escaping them
    out["dropped"] = sorted(set(re.findall(r'\.replace\("(.)", ""\)',
                                           esc_src)))
    return out


def _engine_ok(engine: dict) -> bool:
    return bool(engine.get("drawtext") and engine.get("positions")
                and engine.get("styles") and engine.get("size"))


def _engine_options(engine: dict) -> list[str]:
    if not _engine_ok(engine):
        return []
    y, size = engine["y"], engine["size"]
    opts = [
        "position: bottom-centre only (y=%s horizontal, %s vertical)"
        % (y["horizontal"], y["vertical"]),
        "style: white text with a dark outline only, %dpx horizontal / "
        "%dpx vertical" % (size["horizontal"], size["vertical"]),
        "one line per segment, never wrapped",
    ]
    if engine.get("margins"):
        opts.append("timing: the whole segment, from %.2fs in until %.2fs "
                    "before it ends - not adjustable"
                    % tuple(engine["margins"]))
    if engine.get("dropped"):
        opts.append("characters the renderer strips: "
                    + " ".join(engine["dropped"]))
    return opts


def _max_chars(engine: dict, canvas) -> int:
    """How many characters one drawtext line holds across this canvas."""
    cw, ch = canvas or FALLBACK_CANVAS
    size = (engine.get("size") or {}).get(
        "vertical" if ch > cw else "horizontal")
    if not size or cw <= 0:
        return 1
    return max(1, int(SAFE_WIDTH * cw / (GLYPH_EM * size)))


# ---------------------------------------------------------------- words

# Typographic characters a model reaches for, and the plain ones that mean
# the same thing on screen. They are all `isprintable`, so the filter below
# kept them, and a single U+2011 in "Watch the four-blade propeller whirl"
# killed a whole render: `stage_cut` prints each caption, the child python
# had cp1252 stdout, and the print raised UnicodeEncodeError. The subprocess
# is now told to speak UTF-8 as well, but these belong in neither a log line
# nor a burned-in caption - the render font often has no glyph for them.
_TYPOGRAPHIC = {
    # True HYPHENS join one word, so they close up: "four-blade".
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    # DASHES separate clauses, so they need their own air. Closing them up
    # read as a compound word that was never written - an em dash in
    # "Hear that snap - powers on" became "snap-powers on". The spaces
    # collapse with the rest of the whitespace a few lines below.
    "–": " - ", "—": " - ", "―": " - ",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "…": "...",                                   # ellipsis
    " ": " ", " ": " ", " ": " ", "​": "",
}
_TYPO_TABLE = str.maketrans(_TYPOGRAPHIC)


def _clean(text: str, dropped) -> str:
    """What drawtext will actually show: one line, minus what esc() strips."""
    s = text.translate(_TYPO_TABLE)
    s = "".join(ch if ch.isprintable() else " " for ch in s)
    for ch in dropped:
        s = s.replace(ch, "")
    return " ".join(s.split())


def _locate(phrase: str, source: str) -> str | None:
    """The span of `source` holding exactly `phrase`'s words, in order.

    Matched on whole words, case-insensitively, so a model that changes the
    capitalisation or drops a comma still finds its phrase. What is RETURNED
    is the source's own span - its spelling, its punctuation - so the caption
    is the upstream text, never the model's rendering of it. Partial words do
    not match: "ear tra" is not in "gear train".
    """
    want = [w.lower() for w in _WORD.findall(phrase)]
    if not want:
        return None
    toks = list(_WORD.finditer(source))
    low = [t.group(0).lower() for t in toks]
    n = len(want)
    for i in range(len(low) - n + 1):
        if low[i:i + n] == want:
            return source[toks[i].start():toks[i + n - 1].end()]
    return None


def _extract(source: str, intent: str, max_chars: int, dropped) -> str:
    """The opening clause of `source`, cut at a word boundary to fit.

    A prefix of the validated text, never a paraphrase of it.
    """
    clause = next((p for p in _CLAUSE.split(source) if p.strip()), "")
    out: list[str] = []
    for w in clause.split()[:WORD_CAP.get(intent, WORD_CAP["key_phrase"])]:
        if len(_clean(" ".join(out + [w]), dropped)) > max_chars:
            break
        out.append(w)
    return _clean(" ".join(out), dropped).rstrip(",;:-–— ")


# ---------------------------------------------------------------- basis

@dataclass(frozen=True)
class _Basis:
    """Everything a caption decision is made from, computed once."""
    domain: str
    st_ok: bool
    vo_ok: bool
    an_ok: bool
    engine: dict
    options: list[str]
    max_chars: int
    eligible: dict          # beat index -> intent, or None if not judged
    sources: dict           # beat index -> [(source kind, validated text)]
    candidates: list        # eligible beat indices with words, story order
    clips: dict             # beat index -> clip id, copied from the Story

    @property
    def dropped(self) -> list[str]:
        return self.engine.get("dropped", [])


class CaptionAgent(Agent):
    id = "caption"
    produces = "Caption"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                story: Story | None = None,
                voiceover: Voiceover | None = None,
                animation: Animation | None = None,
                clip_set: ClipAnalysisSet | None = None,
                revision: int = 0) -> Artifact[Caption]:
        started = time.time()
        ldesc = ctx.llm.describe()
        b = self._basis(ctx, story, voiceover, animation, clip_set)

        inputs_hash = self.hash_inputs(
            b.domain, PROMPT_VERSION, ldesc.get("model"),
            ldesc.get("provider"),
            self._story_digest(story),
            self._voiceover_digest(voiceover),
            self._animation_digest(animation),
            b.engine, b.max_chars,
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Caption",
                                         inputs_hash, Caption)
        if cached is not None:
            span.cache(1)
            span.note("reusing the captions: story, narration, look and the "
                      "renderer's caption capability are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        if not _engine_ok(b.engine):
            return self._deterministic(
                ctx, span, b, revision, started, inputs_hash,
                why="the renderer reports no caption it can draw")
        if not b.candidates:
            return self._deterministic(
                ctx, span, b, revision, started, inputs_hash,
                why=self._why_nothing_to_caption(b))
        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, b, revision, started, inputs_hash,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(story, b)
        span.tool("llm.complete_json", purpose="caption",
                  beats=len(b.candidates), max_chars=b.max_chars,
                  prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1400, effort="low")
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"caption reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, b, revision, started, inputs_hash,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, b, ldesc)
        for note in notes:
            span.note(note)

        ctx.emit("caption: %d line(s) on beat(s) %s | %d unsupported "
                 "request(s) | confidence %.2f"
                 % (len(payload.lines),
                    ", ".join(map(str, payload.captioned_beats)) or "none",
                    len(payload.unsupported_requests), payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, b),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # -------------------------------------------------------------- basis

    def _basis(self, ctx: AgentContext, story: Story | None,
               voiceover: Voiceover | None, animation: Animation | None,
               clip_set: ClipAnalysisSet | None) -> _Basis:
        st_ok = bool(story is not None and story.beats)
        vo_ok = bool(voiceover is not None and voiceover.spoken_segments)
        # a look brief only gates captions when someone actually judged it;
        # the assembled one says "none" everywhere for want of a basis
        an_ok = bool(animation is not None and animation.live
                     and animation.directions)

        engine = _engine_caption()
        canvas = tuple(clip_set.canvas) if clip_set is not None else None
        eligible = self._eligible(story, animation, an_ok)
        sources = self._sources(story, voiceover)
        return _Basis(
            domain=(ctx.job.settings.domain or "").strip(),
            st_ok=st_ok, vo_ok=vo_ok, an_ok=an_ok,
            engine=engine, options=_engine_options(engine),
            max_chars=_max_chars(engine, canvas),
            eligible=eligible, sources=sources,
            candidates=[i for i in eligible if sources.get(i)],
            clips={bt.index: bt.clip_id for bt in story.beats}
            if story else {})

    @staticmethod
    def _eligible(story: Story | None, animation: Animation | None,
                  an_ok: bool) -> dict:
        """Beats with room for words, in story order, and the intent of each.

        When Animation judged it, its caption_intent decides: only the beats
        it marked are eligible, and the intent is copied from it. When it did
        not, every beat is open and the intent is the model's to give.
        """
        if story is None:
            return {}
        if an_ok:
            intents = {d.beat_index: d.caption_intent
                       for d in animation.directions}
            return {bt.index: intents[bt.index] for bt in story.beats
                    if intents.get(bt.index, "none") != "none"}
        return {bt.index: None for bt in story.beats}

    @staticmethod
    def _sources(story: Story | None,
                 voiceover: Voiceover | None) -> dict:
        """Per beat, the validated text a caption may be taken from.

        In order of preference: what is SAID over the beat, then what the
        beat is ABOUT. Nothing else is a source - there is no third place
        words can come from.
        """
        out: dict = {}
        for bt in (story.beats if story else []):
            srcs = []
            if voiceover is not None:
                srcs += [("voiceover", sg.text)
                         for sg in voiceover.for_beat(bt.index) if sg.spoken]
            if bt.narrative.strip():
                srcs.append(("story", bt.narrative))
            out[bt.index] = srcs
        return out

    @staticmethod
    def _why_nothing_to_caption(b: _Basis) -> str:
        if not b.st_ok:
            return "no story with beats was available, so there is nothing " \
                   "to caption"
        if b.an_ok and not b.eligible:
            return "the look brief marked no beat as wanting words"
        return ("no beat with room for text has validated words to show - "
                "its narration and story line are both empty")

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, story: Story | None, b: _Basis) -> str:
        return PROMPT.format(
            beats=self._beats_section(story, b),
            engine="\n".join("  - " + o for o in b.options),
            max_chars=b.max_chars,
            label_words=WORD_CAP["short_label"],
            phrase_words=WORD_CAP["key_phrase"])

    @staticmethod
    def _beats_section(story: Story | None, b: _Basis) -> str:
        rows = ["THE BEATS THAT HAVE ROOM FOR WORDS"]
        if not b.an_ok:
            rows.append("  No look brief judged which beats want words - "
                        "choose sparingly.")
        roles = {bt.index: bt.role for bt in story.beats} if story else {}
        for i in b.candidates:
            intent = b.eligible.get(i)
            rows.append("  [%d] %s%s" % (
                i, roles.get(i, ""),
                f" - wants a {intent} (from the look brief)" if intent
                else ""))
            for kind, src in b.sources[i]:
                rows.append("      %-9s \"%s\"" % (
                    "narration" if kind == "voiceover" else "story",
                    _text(src, 300)))
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, b: _Basis, ldesc: dict
                  ) -> tuple[Caption, list[str]]:
        """Everything the model returned is untrusted input."""
        lines, unsupported, notes = self._lines(raw.get("captions"), b)
        unsupported += _texts(raw.get("unsupported_requests"), each=160)

        limitations = _texts(raw.get("limitations"), each=200)
        if not b.an_ok:
            limitations.append(
                "no judged look brief was available, so which beats carry "
                "words was decided here rather than by the look")
        if not b.vo_ok:
            limitations.append(
                "no narration was written, so captions could only come from "
                "the story lines")
        limitations.append(self._fixed_limitation(b))
        if unsupported:
            limitations.append(
                "%d request(s) fall outside what this renderer can draw and "
                "will not happen" % len(unsupported))
        if notes:
            limitations.append("; ".join(notes))

        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        if not b.an_ok:
            confidence = min(confidence, 0.6)

        payload = Caption(
            domain=b.domain,
            caption_type="reasoned" if lines else "degraded",
            lines=lines,
            engine_options=b.options,         # measured, never model output
            max_chars=b.max_chars,            # derived, never model output
            unsupported_requests=unsupported[:MAX_LIST],
            story_available=b.st_ok, voiceover_available=b.vo_ok,
            animation_available=b.an_ok,
            beats_eligible=list(b.eligible),
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not lines,
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    @staticmethod
    def _lines(raw, b: _Basis) -> tuple[list[CaptionLine], list[str],
                                        list[str]]:
        """At most one caption per eligible beat, in the Story's order.

        A beat reference must be one the Story has AND one the look left room
        on. The words must be found in that beat's own validated text; the
        caption is then the SOURCE's span, not the reply's. Anything asking
        for a position, a style or a time is recorded and never honoured.
        """
        chosen: dict = {}
        unsupported: list[str] = []
        bad_beat = not_eligible = no_words = ungrounded = too_long = 0
        trimmed = 0
        timing = 0
        if isinstance(raw, list):
            for item in raw[:MAX_LINES]:
                if not isinstance(item, dict):
                    continue
                bi = item.get("beat")
                if not isinstance(bi, int) or isinstance(bi, bool) \
                        or bi not in b.clips:
                    bad_beat += 1
                    continue
                for k in STYLE_KEYS:
                    v = _text(item.get(k), 60)
                    if v and v.lower().replace(" ", "_").replace("-", "_") \
                            not in NOT_A_REQUEST:
                        unsupported.append(
                            f"beat {bi}: {k} '{v}' - the renderer has one "
                            f"caption position and style")
                if any(item.get(k) is not None for k in TIMING_KEYS):
                    timing += 1
                if bi not in b.eligible:
                    not_eligible += 1
                    continue
                if bi in chosen:
                    continue                 # first answer for a beat wins

                phrase = _text(item.get("text"), 300)
                if not phrase:
                    no_words += 1
                    continue
                found = next(((kind, sp) for kind, src in b.sources.get(bi, [])
                              for sp in [_locate(phrase, src)] if sp), None)
                if found is None:
                    ungrounded += 1
                    continue

                intent = b.eligible[bi] or _one_of(
                    item.get("intent"), ("short_label", "key_phrase"),
                    "key_phrase")
                text = _clean(found[1], b.dropped)
                if text and (len(text) > b.max_chars
                             or len(text.split()) > WORD_CAP[intent]):
                    # Too long used to be DROPPED outright. A model that
                    # overruns by a word then costs the segment its caption
                    # entirely: on a real six-segment cut, two of three
                    # eligible beats came back captionless and only one line
                    # reached the screen. `_extract` already takes a prefix
                    # at a word boundary for the deterministic path, so do
                    # the same here - a prefix of the located span is still
                    # the upstream's own words, never a paraphrase of them.
                    text = _extract(found[1], intent, b.max_chars, b.dropped)
                    if text:
                        trimmed += 1
                if not text or len(text) > b.max_chars \
                        or len(text.split()) > WORD_CAP[intent]:
                    too_long += 1
                    continue
                chosen[bi] = (text, found[0], intent,
                              _text(item.get("why"), 200))

        lines = [CaptionLine(index=n, beat_index=bi, clip_id=b.clips[bi],
                             text=text, source=src, intent=intent, why=why)
                 for n, (bi, (text, src, intent, why))
                 in enumerate(sorted(chosen.items()), 1)]

        notes = []
        if bad_beat:
            notes.append(f"{bad_beat} caption(s) named a beat the story does "
                         f"not have and were dropped")
        if not_eligible:
            notes.append(f"{not_eligible} caption(s) were on a beat the look "
                         f"left without words and were dropped")
        if no_words:
            notes.append(f"{no_words} caption(s) carried no text and were "
                         f"dropped")
        if ungrounded:
            notes.append(f"{ungrounded} caption(s) were not found word for "
                         f"word in their beat's validated text and were "
                         f"dropped - no caption may say something upstream "
                         f"did not")
        if trimmed:
            notes.append(f"{trimmed} caption(s) overran the line and were cut "
                         f"back to their opening words rather than dropped")
        if too_long:
            notes.append(f"{too_long} caption(s) would not fit on one line "
                         f"of {b.max_chars} characters, or ran past their "
                         f"word limit, and were dropped")
        if timing:
            notes.append(f"{timing} caption(s) carried a time or duration, "
                         f"which was ignored - the renderer fixes when a "
                         f"caption shows")
        return lines, list(dict.fromkeys(unsupported)), notes

    @staticmethod
    def _fixed_limitation(b: _Basis) -> str:
        return ("this renderer draws one line per segment at one position "
                "and style, for the whole segment; about %d characters fit on "
                "this canvas - an estimate from its font size, not a "
                "measurement of the rendered text" % b.max_chars)

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan, b: _Basis,
                       revision: int, started: float, inputs_hash: str,
                       why: str, cost: Cost | None = None
                       ) -> Artifact[Caption]:
        """Extract, never write.

        A beat the look marked for words gets the opening clause of its
        validated text, cut at a word boundary to fit: verbatim, and marked as
        assembled. With no judged intent there is no basis to put words on
        any beat, so none are - which is the honest output, not a failure.
        """
        span.fallback(f"deterministic captions ({why})")

        lines: list[CaptionLine] = []
        if b.an_ok and _engine_ok(b.engine):
            for i in b.candidates:
                intent = b.eligible[i]
                kind, src = b.sources[i][0]
                text = _extract(src, intent, b.max_chars, b.dropped)
                if text:
                    lines.append(CaptionLine(
                        index=len(lines) + 1, beat_index=i,
                        clip_id=b.clips[i], text=text, source=kind,
                        intent=intent, why=""))

        limitations = [f"{why}; " + (
            "each caption is the opening words of its beat's validated "
            "text, cut to fit - extracted, not chosen" if lines else
            "nothing is captioned - no words were invented to fill the gap")]
        if not b.an_ok:
            limitations.append("no judged look brief was available, so no "
                               "beat was marked as wanting words")
        if not b.vo_ok:
            limitations.append("no narration was written")
        if _engine_ok(b.engine):
            limitations.append(self._fixed_limitation(b))

        payload = Caption(
            domain=b.domain,
            caption_type="degraded" if lines else "minimal",
            lines=lines, engine_options=b.options, max_chars=b.max_chars,
            unsupported_requests=[],
            story_available=b.st_ok, voiceover_available=b.vo_ok,
            animation_available=b.an_ok,
            beats_eligible=list(b.eligible) if b.an_ok else [],
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        ctx.emit("caption: %s (%s) - %d line(s)"
                 % (payload.caption_type, why, len(lines)))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, b, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic captions ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- digests

    @staticmethod
    def _story_digest(story: Story | None):
        if story is None:
            return None
        return {"beats": [(bt.index, bt.role, bt.clip_id, bt.narrative)
                          for bt in story.beats],
                "live": story.live}

    @staticmethod
    def _voiceover_digest(voiceover: Voiceover | None):
        if voiceover is None:
            return None
        return [(sg.beat_index, sg.text) for sg in voiceover.segments]

    @staticmethod
    def _animation_digest(animation: Animation | None):
        if animation is None:
            return None
        return {"live": animation.live,
                "intents": [(d.beat_index, d.caption_intent)
                            for d in animation.directions]}

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Caption, b: _Basis,
                   why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("Story", b.st_ok),
                                    ("Voiceover", b.vo_ok),
                                    ("Animation", b.an_ok)) if ok]
                   or ["neither"])
        out = [Decision(
            what="Evidence the captions were chosen from",
            chose=" + ".join(sources),
            why=("recorded explicitly so captions placed without a judged "
                 "look, or taken only from story lines, cannot later be read "
                 "as though they had more"),
            decided_by="measurement", confidence=1.0)]

        out.append(Decision(
            what="What the renderer can draw",
            chose=(f"one line of about {p.max_chars} characters, "
                   f"bottom-centre, white outlined" if p.engine_options
                   else "nothing - no caption capability found"),
            why=("read from fit_filter's drawtext and esc() rather than "
                 "assumed; there is one position and one style and the "
                 "timing is fixed, so none of them can be asked for"),
            decided_by="measurement", confidence=1.0))

        out.append(Decision(
            what="Beats with room for words",
            chose=", ".join(map(str, p.beats_eligible[:8])) or "none",
            why=("the Animation agent's caption_intent; a beat it left "
                 "without words is not captioned" if b.an_ok else
                 "no judged look brief, so no beat was marked by the look"),
            decided_by="policy", confidence=1.0))

        if not p.live:
            out.append(Decision(
                what="Captions",
                chose=(f"{len(p.lines)} extracted" if p.lines else "none"),
                why=(f"{why}; " + (
                    "the opening words of each marked beat's validated text, "
                    "cut at a word boundary - nothing was written"
                    if p.lines else
                    "no words were invented to fill the gap")),
                decided_by="fallback", confidence=1.0))
            return out

        out.append(Decision(
            what="Caption per beat",
            chose=(" | ".join(f"{ln.beat_index}: {ln.text}"
                              for ln in p.lines[:4])[:250] or "none"),
            why=("each one located word for word in that beat's validated "
                 "narration or story line, and copied from there - this text "
                 "reaches drawtext through captions.json"),
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
                what="What these captions cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["CaptionAgent"]
