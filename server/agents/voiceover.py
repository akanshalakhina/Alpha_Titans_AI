"""
Voiceover Agent - what is SAID over the story.

WHERE IT SITS
-------------
    Story            (what is told, and in what order)
        |
        v
    Voiceover        (what is said over it)          <- this agent
        |
        v
    EditingDirector  (how it is cut)

Story decided the beats. This writes the words. The director still decides
every in/out point, and nothing here can express one.

THE ONE STRUCTURAL RULE
-----------------------
A segment references a beat by INDEX, and that is the only pointer the model
supplies. `clip_id` and `moment_ref` are COPIED from the referenced beat
rather than read from the reply, so a script cannot drift onto footage the
narrative never anchored - inventing a clip is not validated against, it is
impossible.

WHAT IT MAY NOT DO
------------------
No timestamps, no durations, no clip lengths, no transitions, no filter
strings, no paths, no commands, and no editing decisions. `NarrationSegment`
forbids extra fields, so a reply carrying `start`/`end`/`duration`/`at`/
`seconds` fails validation rather than being quietly ignored. `word_count` is
a computed property of the text, not a field a model can set and not a
duration.

GROUNDING
---------
A line citing research must cite a url actually in the ResearchReport;
invented citations are stripped and the line downgraded rather than dropped,
because the words may still be right. `support` is derived from what SURVIVED,
never from what the model claimed.

FALLBACK
--------
With no model the agent emits the SHAPE of a script - one placeholder per
beat, carrying no words at all - and says so plainly. Writing narration with
nothing to write it from would be invention presented as a script, which is
precisely the failure this layer exists to prevent.
"""

from __future__ import annotations

import math
import time

from .artifacts import (Artifact, ContentStrategy, Cost, Decision,
                        NarrationSegment, ResearchReport, Story, Voiceover)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "voiceover-prompt-v1"
MAX_SEGMENTS = 12
MAX_LIST = 8
MAX_WORDS_PER_SEGMENT = 60

TONES = ("neutral", "warm", "energetic", "calm", "authoritative", "playful",
         "wry")

PROMPT = """You are writing the narration for a short video.

The story below is already decided - the beats, their order, and which clip
each one sits on. Your job is the words spoken over them. You do NOT decide
how it is cut, how long anything is on screen, or what order the beats run in.

{story}
{strategy}
{research}
YOUR TASK
Write the narration, one or more lines per beat.

RULES - these are checked after you reply, not taken on trust
- Reference a beat by its INDEX from the list above. That is the only
  pointer you give. The clip and the moment come from the beat itself, so do
  NOT name a clip - you cannot put narration on footage the story did not
  already anchor.
- A segment whose `beat_index` is not in the list above will be DROPPED.
- Do NOT give timings, durations, or how long a line should take. There is no
  field for them and a reply containing `start`, `end`, `duration`, `at` or
  `seconds` is rejected outright.
- Do NOT write stage directions, cuts, transitions or camera instructions.
  Use `note` for delivery only - "slower here", "let it land".
- Do NOT state statistics, view counts, dates or events. If the evidence
  above does not say it, you do not know it. A line citing a url not listed
  above will have that citation stripped.
- Keep lines speakable: short sentences, roughly {max_words} words or fewer
  each. Silence is allowed - a beat that plays better without words should
  simply have no segment.

Reply with ONLY a JSON object, no prose around it:

{{
  "voice": "neutral|warm|energetic|calm|authoritative|playful|wry",
  "voice_rationale": "<why that delivery suits this material>",
  "audience": "<who is being spoken to>",
  "segments": [
    {{"beat_index": <index from the list above>,
      "text": "<what is spoken, verbatim>",
      "tone": "neutral|warm|energetic|calm|authoritative|playful|wry",
      "note": "<delivery note, or empty>",
      "evidence_urls": ["<url from the evidence above>"]}}
  ],
  "delivery_notes": ["<anything the performer needs to know>"],
  "risks": ["<what could make this narration fall flat>"],
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


class VoiceoverAgent(Agent):
    id = "voiceover"
    produces = "Voiceover"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                story: Story | None = None,
                strategy: ContentStrategy | None = None,
                research: ResearchReport | None = None,
                revision: int = 0) -> Artifact[Voiceover]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        # What we ACTUALLY have. A story with no beats is not a story to
        # narrate, and a strategy flagged insufficient is not a brief.
        st_ok = bool(story is not None and story.beats)
        cs_ok = bool(strategy is not None and not strategy.insufficient_signal)
        r_ok = bool(research is not None and not research.insufficient_signal
                    and research.evidence)

        # ---- cache: keyed on all three upstreams -----------------------
        # No time bucket: a script goes stale when the story or the brief
        # changes, never on the clock.
        inputs_hash = self.hash_inputs(
            domain, PROMPT_VERSION, ldesc.get("model"), ldesc.get("provider"),
            self._story_digest(story),
            self._strategy_digest(strategy),
            self._research_digest(research),
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Voiceover",
                                         inputs_hash, Voiceover)
        if cached is not None:
            span.cache(1)
            span.note("reusing the narration: story, strategy, research, "
                      "domain and prompt are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        # ---- nothing to narrate ----------------------------------------
        if not st_ok:
            return self._degraded(
                ctx, span, story, strategy, research, domain, revision,
                started, inputs_hash, st_ok, cs_ok, r_ok,
                why=("there is no story with beats to narrate, so there is "
                     "nothing for a line to sit on"))

        if not ctx.llm.available():
            return self._degraded(
                ctx, span, story, strategy, research, domain, revision,
                started, inputs_hash, st_ok, cs_ok, r_ok,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(story, strategy, research)
        span.tool("llm.complete_json", purpose="voiceover",
                  beats=len(story.beats), strategy=cs_ok, research=r_ok,
                  prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=2000)
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"voiceover reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._degraded(
                ctx, span, story, strategy, research, domain, revision,
                started, inputs_hash, st_ok, cs_ok, r_ok,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, domain, story, research,
                                        st_ok, cs_ok, r_ok, ldesc)
        for note in notes:
            span.note(note)

        ctx.emit("voiceover: %d line(s) over %d beat(s), %d word(s), "
                 "voice %s | confidence %.2f"
                 % (len(payload.spoken_segments), len(payload.beats_covered),
                    payload.total_words, payload.voice, payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, st_ok, cs_ok, r_ok),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, story: Story,
                      strategy: ContentStrategy | None,
                      research: ResearchReport | None) -> str:
        return PROMPT.format(
            story=self._story_section(story),
            strategy=self._strategy_section(strategy),
            research=self._research_section(research),
            max_words=MAX_WORDS_PER_SEGMENT)

    @staticmethod
    def _story_section(story: Story) -> str:
        rows = ["THE STORY", ""]
        if story.logline:
            rows.append(f"  logline   {story.logline}")
        for label, val in (("hook", story.hook), ("setup", story.setup),
                           ("builds", story.progression),
                           ("payoff", story.payoff),
                           ("ending", story.ending)):
            if val:
                rows.append(f"  {label:<10}{val}")
        rows.append("")
        rows.append("  BEATS - reference these by index, and only these:")
        for b in story.beats:
            line = "    [%d] %-9s %s" % (b.index, b.role,
                                         b.narrative or "(no description)")
            rows.append(line)
            if b.why:
                rows.append("         why: %s" % b.why)
        if not story.live:
            rows.append("")
            rows.append("  NOTE: this sequence was assembled "
                        "deterministically, not authored.")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _strategy_section(strategy: ContentStrategy | None) -> str:
        if strategy is None or strategy.insufficient_signal:
            return ("\nTHE BRIEF\n"
                    "  None available. Do not assume an audience or an angle "
                    "beyond what the\n  story above states.\n")
        rows = ["\nTHE BRIEF"]
        for label, val in (("topic", strategy.central_topic),
                           ("angle", strategy.core_angle),
                           ("audience", strategy.audience),
                           ("objective", strategy.objective),
                           ("engagement", strategy.engagement_strategy)):
            if val:
                rows.append(f"  {label:<11}{val}")
        rows.append(f"  {'pacing':<11}{strategy.pacing}"
                    + (f" - {strategy.pacing_rationale}"
                       if strategy.pacing_rationale else ""))
        if strategy.grounded_points:
            rows.append("  points the narration should land:")
            rows += [f"    - {k.point}" for k in strategy.grounded_points[:6]]
        return "\n".join(rows) + "\n"

    @staticmethod
    def _research_section(research: ResearchReport | None) -> str:
        if research is None or research.insufficient_signal \
                or not research.evidence:
            return ("\nOUTSIDE EVIDENCE\n"
                    "  None available. Do NOT state trends, statistics or "
                    "audience numbers -\n  you do not have them.\n")
        rows = ["\nOUTSIDE EVIDENCE", "  Cite these urls and no others."]
        for e in research.evidence[:8]:
            rows.append("    - %s\n      url: %s" % (e.title, e.url))
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, domain: str, story: Story,
                  research: ResearchReport | None,
                  st_ok: bool, cs_ok: bool, r_ok: bool, ldesc: dict
                  ) -> tuple[Voiceover, list[str]]:
        """Everything the model returned is untrusted input."""
        known_urls = {e.url for e in research.evidence} if research else set()
        beats = {b.index: b for b in story.beats}

        segments, notes = self._segments(raw.get("segments"), beats,
                                         known_urls)

        research_basis = sorted({u for s in segments for u in s.evidence_urls})
        covered = sorted({s.beat_index for s in segments if s.spoken})

        limitations = _texts(raw.get("limitations"))
        if not cs_ok:
            limitations.append(
                "written without a usable content strategy, so the narration "
                "assumes no audience beyond what the story states")
        if not r_ok:
            limitations.append(
                "written without outside evidence, so no line makes a claim "
                "about what audiences currently want")
        if not story.live:
            limitations.append(
                "the story it narrates was assembled deterministically rather "
                "than authored, so the sequence behind these lines is thin")
        if notes:
            limitations.append("; ".join(notes))

        spoken = [s for s in segments if s.spoken]
        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        # A model's self-assessment cannot outrun what it was given.
        if not cs_ok and not r_ok:
            confidence = min(confidence, 0.4)
        if not story.live:
            confidence = min(confidence, 0.5)
        if not spoken:
            confidence = min(confidence, 0.2)

        payload = Voiceover(
            domain=domain,
            voiceover_type="reasoned" if spoken else "degraded",
            voice=_one_of(raw.get("voice"), TONES, "neutral"),
            voice_rationale=_text(raw.get("voice_rationale"), 240),
            audience=_text(raw.get("audience"), 200),
            segments=segments,
            delivery_notes=_texts(raw.get("delivery_notes"), each=160),
            story_available=st_ok, strategy_available=cs_ok,
            research_available=r_ok,
            research_basis=research_basis[:MAX_LIST],
            beats_covered=covered,
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not spoken,
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    @staticmethod
    def _segments(raw, beats: dict, known_urls: set
                  ) -> tuple[list[NarrationSegment], list[str]]:
        """Build the script, dropping anything that cannot be anchored.

        `clip_id` and `moment_ref` are taken from the BEAT, never from the
        reply - that is what makes putting narration on invented footage
        impossible rather than merely detected.

        Segments are ordered by beat, because narration follows the narrative
        the Story already decided. A model that returned them out of order is
        corrected, not obeyed.
        """
        kept: list[tuple[int, NarrationSegment]] = []
        bad_beat = bad_cite = empty = 0
        if not isinstance(raw, list):
            return [], []

        for pos, item in enumerate(raw[:MAX_SEGMENTS]):
            if not isinstance(item, dict):
                continue
            bi = item.get("beat_index")
            if not isinstance(bi, int) or isinstance(bi, bool) \
                    or bi not in beats:
                bad_beat += 1
                continue                       # unanchored line => dropped

            text = _text(item.get("text"), 400)
            if not text:
                empty += 1
                continue                       # a line with no words is noise

            urls = [u for u in (item.get("evidence_urls") or [])
                    if isinstance(u, str)][:4]
            real_urls = [u for u in urls if u in known_urls]
            if len(real_urls) < len(urls):
                bad_cite += 1

            beat = beats[bi]
            kept.append((pos, NarrationSegment(
                index=1,                       # renumbered below
                beat_index=bi,
                clip_id=beat.clip_id,          # from the BEAT, not the reply
                moment_ref=beat.moment_ref,    # likewise
                text=text,
                tone=_one_of(item.get("tone"), TONES, "neutral"),
                support="both" if real_urls else "vision",
                evidence_urls=real_urls,
                note=_text(item.get("note"), 160))))

        # narration follows the narrative: stable sort by beat, keeping the
        # model's order among lines that share one
        reordered = sorted(kept, key=lambda p: (p[1].beat_index, p[0]))
        out = [s.model_copy(update={"index": i + 1})
               for i, (_pos, s) in enumerate(reordered)]

        notes = []
        if bad_beat:
            notes.append(f"{bad_beat} line(s) referenced a beat that does not "
                         f"exist and were dropped")
        if empty:
            notes.append(f"{empty} line(s) carried no words and were dropped")
        if bad_cite:
            notes.append(f"{bad_cite} line(s) cited a url that was never "
                         f"retrieved")
        model_order = [s.beat_index for _pos, s in kept]
        if model_order != sorted(model_order):
            notes.append("lines were reordered to follow the story's beats")
        return out, notes

    # ------------------------------------------------------ deterministic

    def _degraded(self, ctx: AgentContext, span: TraceSpan,
                  story: Story | None, strategy: ContentStrategy | None,
                  research: ResearchReport | None, domain: str,
                  revision: int, started: float, inputs_hash: str,
                  st_ok: bool, cs_ok: bool, r_ok: bool, why: str,
                  cost: Cost | None = None) -> Artifact[Voiceover]:
        """The SHAPE of a script, carrying no words.

        A placeholder per beat is useful - it tells the director which beats
        would carry narration. Words are not invented here under any
        circumstances: a fabricated line would be indistinguishable from a
        written one downstream, and that is the whole failure this layer
        exists to prevent.
        """
        span.fallback(f"no narration written ({why})")

        segments: list[NarrationSegment] = []
        if st_ok and story is not None:
            for i, b in enumerate(story.beats[:MAX_SEGMENTS]):
                segments.append(NarrationSegment(
                    index=i + 1, beat_index=b.index, clip_id=b.clip_id,
                    moment_ref=b.moment_ref,
                    text="",                    # never invented
                    tone="neutral", support="none", evidence_urls=[],
                    note="no line written"))

        limitations = [f"{why}; no narration was written - these segments are "
                       f"placeholders marking where lines would go, and carry "
                       f"no words"]
        if not st_ok:
            limitations.append("there was no story with beats to narrate")
        if not cs_ok:
            limitations.append("no usable content strategy was available")
        if not r_ok:
            limitations.append("no outside evidence was available")

        payload = Voiceover(
            domain=domain,
            voiceover_type="minimal" if not segments else "degraded",
            voice="neutral",
            voice_rationale="default delivery; no writer ran to choose one",
            audience="",
            segments=segments, delivery_notes=[],
            story_available=st_ok, strategy_available=cs_ok,
            research_available=r_ok,
            research_basis=[], beats_covered=[],
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        ctx.emit("voiceover: %s (%s) - %d placeholder(s), no words written"
                 % (payload.voiceover_type, why, len(segments)))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, st_ok, cs_ok, r_ok, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"no narration written ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- digests

    @staticmethod
    def _story_digest(story: Story | None):
        """What about the story would change the script if it changed."""
        if story is None:
            return None
        return {
            "logline": story.logline, "hook": story.hook,
            "payoff": story.payoff, "ending": story.ending,
            "beats": [(b.index, b.role, b.clip_id, b.narrative)
                      for b in story.beats],
            "live": story.live,
        }

    @staticmethod
    def _strategy_digest(strategy: ContentStrategy | None):
        if strategy is None:
            return None
        return {"topic": strategy.central_topic, "angle": strategy.core_angle,
                "audience": strategy.audience, "pacing": strategy.pacing,
                "points": sorted(k.point for k in strategy.grounded_points),
                "insufficient": strategy.insufficient_signal}

    @staticmethod
    def _research_digest(research: ResearchReport | None):
        if research is None:
            return None
        return {"urls": sorted(e.url for e in research.evidence),
                "insufficient": research.insufficient_signal}

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Voiceover, st_ok: bool, cs_ok: bool, r_ok: bool,
                   why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("Story", st_ok),
                                    ("ContentStrategy", cs_ok),
                                    ("Research", r_ok)) if ok] or ["neither"])
        out = [Decision(
            what="Evidence the narration was written from",
            chose=" + ".join(sources),
            why=("recorded explicitly so a script written without a brief or "
                 "without a story cannot later be read as though it had both"),
            decided_by="measurement", confidence=1.0)]

        if not p.live:
            out.append(Decision(
                what="Narration", chose=f"{p.voiceover_type} - no words",
                why=(f"{why}; placeholders mark where lines would go and "
                     f"carry no text. A fabricated line would be "
                     f"indistinguishable from a written one downstream"),
                decided_by="fallback", confidence=1.0))
            return out

        out.append(Decision(
            what="Voice", chose=p.voice,
            why=(p.voice_rationale or "no rationale given")[:200] +
                " - a delivery intent, not a direction to the editor",
            decided_by="llm", confidence=p.confidence))

        if p.segments:
            out.append(Decision(
                what="Narration written",
                chose=f"{len(p.spoken_segments)} line(s) over "
                      f"{len(p.beats_covered)} beat(s), {p.total_words} words",
                why=("every line is anchored to a beat the story already "
                     "decided; the clip and moment come from that beat, not "
                     "from the reply, and no line expresses a time"),
                decided_by="llm", confidence=p.confidence))
            out.append(Decision(
                what="Narration order",
                chose="beats " + ", ".join(str(i) for i in p.beats_covered[:8]),
                why=("lines follow the story's beat order - the "
                     "EditingDirector still chooses the cut and every in/out "
                     "point, which this cannot express"),
                decided_by="policy", confidence=1.0))
        if p.research_basis:
            out.append(Decision(
                what="Claims grounded in evidence",
                chose=f"{len(p.research_basis)} source(s) cited",
                why=("a line citing research must name a url that was "
                     "actually retrieved; invented citations were stripped"),
                decided_by="llm", confidence=p.confidence))
        if p.limitations:
            out.append(Decision(
                what="What this narration cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["VoiceoverAgent"]
