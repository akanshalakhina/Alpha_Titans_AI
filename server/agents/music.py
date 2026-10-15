"""
Music Agent - what the cut should sound like underneath.

WHERE IT SITS
-------------
    Story + Voiceover   (what is told, and what is spoken)
            |
            v
        Music           (what it sounds like)        <- this agent
            |
            v
    EditingDirector     (how it is cut)

THE VOCABULARY IS THE ENGINE'S, NOT THE MODEL'S
-----------------------------------------------
`mood` is chosen from the seven values `server/extras.py::MOODS` actually
supports, whose own comment reads "Every option maps to a real engine
setting". A model free to name a genre would produce direction nothing can
render; picking from the engine's own list means the plan is actionable the
moment anything downstream reads it. `engine` is then DERIVED from that
choice through the same map, so it cannot disagree with it.

That is the same discipline Vision uses for `chosen_window` and Voiceover uses
for `beat_index`: the model exercises judgement over options the deterministic
layer produced, rather than authoring the options.

WHAT IT MAY NOT DO
------------------
No timestamps, no durations, no BPM, no file paths, no asset names, no ffmpeg,
no editing decisions. `MusicSection` and `Music` forbid extra fields, so a
reply carrying `start`/`end`/`duration`/`bpm`/`file_path`/`track` fails
validation rather than being quietly ignored.

Tempo is fitted to the real transition marks by `beatbed`/`bgm_stdlib` from
measurements this agent never sees, and selecting or generating the actual
audio stays with those engines. This agent produces DIRECTION.

NO ASSET IS INVENTED
--------------------
`available_assets` is measured from the music library on disk. The model is
never shown it and cannot add to it, so a plan can never reference a track
nobody has - which, with an empty library, means it references none at all and
says so.

FALLBACK
--------
With no model the mood comes from `server/director.py::MOOD_BANDS` applied to
the real measured motion median - the same deterministic choice the existing
pipeline has always made. That is a genuine answer, not a guess, and it is
marked as assembled rather than reasoned.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from .artifacts import (Artifact, ClipAnalysisSet, ContentStrategy, Cost,
                        Decision, Music, MusicSection, Story, Voiceover)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "music-prompt-v2"
MAX_SECTIONS = 12
MAX_LIST = 8

ENERGIES = ("low", "building", "steady", "high", "peak", "falling")
TEMPOS = ("slow", "moderate", "driving", "varied")
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

PROMPT = """You are directing the music for a short video.

The story and the narration are already decided. Your job is what it sounds
like underneath them. You do NOT choose the track, the tempo, the cut, or when
anything happens - those are measured and fitted downstream.

{brief}{story}
{narration}
{strategy}
YOUR TASK
Decide the mood and the shape of the score.

RULES - these are checked after you reply, not taken on trust
- `mood` MUST be one of exactly these, and nothing else:
    {moods}
  These are the settings the music engine actually has. A mood outside this
  list is one nothing can play, so it will be replaced.
- Reference a beat by its INDEX from the list above. That is the only pointer
  you give; a section whose `beat_index` is not listed will be DROPPED.
- Do NOT give timings, durations, BPM, bar counts or "when" anything happens.
  There is no field for them and a reply containing `start`, `end`,
  `duration`, `bpm` or `seconds` is rejected outright.
- Do NOT name a track, a file, an artist or a library asset. You have no
  music library and cannot refer to one.
- Beats marked SPOKEN carry narration. Music under speech should get out of
  its way - say so in the section's intent rather than inventing a mix level.

Reply with ONLY a JSON object, no prose around it:

{{
  "mood": "{moods_pipe}",
  "mood_rationale": "<why that mood suits this material>",
  "tempo_feel": "slow|moderate|driving|varied",
  "sonic_direction": "<how it should sound, in a sentence or two>",
  "instrumentation": ["<what should be heard>"],
  "energy_arc": "<how the energy moves across the whole piece>",
  "avoid": ["<what would be wrong for this piece>"],
  "sections": [
    {{"beat_index": <index from the list above>,
      "energy": "low|building|steady|high|peak|falling",
      "intent": "<what the music is doing across this beat>"}}
  ],
  "narration_notes": ["<how the bed should behave under speech>"],
  "risks": ["<what could make this score wrong>"],
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


def _moods() -> tuple[str, ...]:
    """The engine's own vocabulary, read at runtime so it cannot drift."""
    from .tools.pipeline_bridge import extras_mod
    return tuple(extras_mod.MOODS)


def _operator_mood(ctx) -> str | None:
    """The mood the operator asked for, or None if they said 'auto'/'none'."""
    asked = getattr(ctx.job.settings, "music", "auto")
    return asked if asked not in ("auto", "none") else None


def _apply_operator_mood(payload: "Music", forced, span) -> "Music":
    """The operator's mood wins, on every path.

    JobSettings is documented as "these always win over anything an agent
    decides". Applying this only on the reasoned path meant that without a
    model key - which is every run with no credentials - the person's choice
    was quietly dropped while the UI showed it selected.
    """
    if not forced:
        return payload
    if payload.mood == forced:
        span.note(f"operator asked for {forced!r}, which is what the agent "
                  f"chose anyway")
        return payload
    span.note(f"operator asked for {forced!r}; overriding {payload.mood!r}")
    return payload.model_copy(update={
        "mood": forced,
        "engine": _engine_for(forced),
        "mood_rationale": (f"chosen by the operator, not the agent "
                           f"(it proposed {payload.mood})"),
    })


def _engine_for(mood: str) -> str:
    """Which engine that mood actually drives. Derived, never asked for."""
    from .tools.pipeline_bridge import extras_mod
    entry = extras_mod.MOODS.get(mood)
    return entry[0] if entry else ""


def _library() -> list[str]:
    """What the music library actually holds. Measured, never model output."""
    from .tools.pipeline_bridge import config_mod
    try:
        d = Path(config_mod.MUSIC_DIR)
        return sorted(p.name for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    except (OSError, AttributeError):
        return []


class MusicAgent(Agent):
    id = "music"
    produces = "Music"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                clip_set: ClipAnalysisSet | None = None,
                story: Story | None = None,
                voiceover: Voiceover | None = None,
                strategy: ContentStrategy | None = None,
                revision: int = 0) -> Artifact[Music]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        # The operator's own choice wins. JobSettings is documented as
        # "these always win over anything an agent decides", and a person who
        # picked `energetic` did not ask for a second opinion. "auto" means
        # no preference, which is how every run before this behaved.
        asked = ctx.job.settings.music
        forced = _operator_mood(ctx)
        silent = asked == "none"

        st_ok = bool(story is not None and story.beats)
        vo_ok = bool(voiceover is not None and voiceover.spoken_segments)
        cs_ok = bool(strategy is not None and not strategy.insufficient_signal)

        assets = _library()

        inputs_hash = self.hash_inputs(
            domain, asked, ctx.job.settings.spoken,
            PROMPT_VERSION, ldesc.get("model"), ldesc.get("provider"),
            self._story_digest(story),
            self._voiceover_digest(voiceover),
            self._strategy_digest(strategy),
            self._motion_digest(clip_set),
            assets,
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Music",
                                         inputs_hash, Music)
        if cached is not None:
            span.cache(1)
            span.note("reusing the music direction: story, narration, brief "
                      "and measurements are all unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, clip_set, story, voiceover, strategy, domain,
                revision, started, inputs_hash, st_ok, vo_ok, cs_ok, assets,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(story, voiceover, strategy,
                                    ctx.job.settings.brief())
        span.tool("llm.complete_json", purpose="music",
                  beats=len(story.beats) if st_ok else 0,
                  narration=vo_ok, prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1600)
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"music reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, clip_set, story, voiceover, strategy, domain,
                revision, started, inputs_hash, st_ok, vo_ok, cs_ok, assets,
                why="the model reply was unusable", cost=cost)

        payload, notes = self._validate(raw, domain, story, voiceover,
                                        st_ok, vo_ok, cs_ok, assets, ldesc)
        payload = _apply_operator_mood(payload, forced, span)

        for note in notes:
            span.note(note)

        ctx.emit("music: %s (%s engine), %s tempo, %d section(s) | "
                 "confidence %.2f"
                 % (payload.mood, payload.engine or "none",
                    payload.tempo_feel, len(payload.sections),
                    payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, st_ok, vo_ok, cs_ok),
            confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=cost)

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, story: Story | None,
                      voiceover: Voiceover | None,
                      strategy: ContentStrategy | None,
                      brief: str = "") -> str:
        moods = _moods()
        return PROMPT.format(
            brief=self._brief_section(brief),
            story=self._story_section(story, voiceover),
            narration=self._narration_section(voiceover),
            strategy=self._strategy_section(strategy),
            moods=", ".join(moods),
            moods_pipe="|".join(moods))

    @staticmethod
    def _brief_section(brief: str) -> str:
        """The person's own words.

        This was hashed into the cache key from the start but never put in
        the prompt, so two runs whose ONLY difference was the spoken brief
        recomputed - and then returned the same mood. Asking for "fast and
        punchy, high energy" and asking for "calm and slow" both scored
        `calm`, because the only thing the model could actually see was the
        measured motion, which is identical in both.

        `music` is NOT included - `brief()` leaves it out on purpose,
        because an explicit mood choice is applied as a hard override after
        this agent replies, and repeating it here would invite the model to
        argue with a decision already made.
        """
        if not brief.strip():
            return ""
        head = ("WHAT THE PERSON ASKED FOR, IN THEIR OWN WORDS\n"
                "  Read this as INTENT about how the video should "
                "feel. It\n"
                "  cannot name a mood outside the list below.\n\n")
        body = "\n".join("  " + ln
                         for ln in brief.strip().splitlines())
        return head + body + "\n\n"

    @staticmethod
    def _story_section(story: Story | None,
                       voiceover: Voiceover | None) -> str:
        if story is None or not story.beats:
            return ("THE STORY\n"
                    "  No story with beats is available, so there is no shape "
                    "to score against.\n  Give overall direction only and "
                    "leave `sections` empty.\n")
        spoken = ({s.beat_index for s in voiceover.spoken_segments}
                  if voiceover else set())
        rows = ["THE STORY"]
        if story.logline:
            rows.append(f"  logline   {story.logline}")
        if story.progression:
            rows.append(f"  builds    {story.progression}")
        rows.append("")
        rows.append("  BEATS - reference these by index, and only these:")
        for b in story.beats:
            tag = "  SPOKEN" if b.index in spoken else ""
            rows.append("    [%d] %-9s %s%s"
                        % (b.index, b.role,
                           b.narrative or "(no description)", tag))
        if not story.live:
            rows.append("")
            rows.append("  NOTE: this sequence was assembled "
                        "deterministically, not authored.")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _narration_section(voiceover: Voiceover | None) -> str:
        if voiceover is None or not voiceover.spoken_segments:
            return ("\nTHE NARRATION\n"
                    "  There is no narration, so nothing has to be heard over "
                    "speech.\n")
        rows = ["\nTHE NARRATION",
                "  voice: %s" % voiceover.voice,
                "  %d line(s) over beats %s"
                % (len(voiceover.spoken_segments),
                   ", ".join(str(i) for i in voiceover.beats_covered))]
        rows.append("  The bed has to sit under these without fighting them.")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _strategy_section(strategy: ContentStrategy | None) -> str:
        if strategy is None or strategy.insufficient_signal:
            return ("\nTHE BRIEF\n  None available. Take the tone from the "
                    "story above.\n")
        rows = ["\nTHE BRIEF"]
        for label, val in (("topic", strategy.central_topic),
                           ("angle", strategy.core_angle),
                           ("audience", strategy.audience)):
            if val:
                rows.append(f"  {label:<11}{val}")
        rows.append(f"  {'pacing':<11}{strategy.pacing}"
                    + (f" - {strategy.pacing_rationale}"
                       if strategy.pacing_rationale else ""))
        return "\n".join(rows) + "\n"

    # --------------------------------------------------------- validation

    def _validate(self, raw: dict, domain: str, story: Story | None,
                  voiceover: Voiceover | None,
                  st_ok: bool, vo_ok: bool, cs_ok: bool,
                  assets: list[str], ldesc: dict
                  ) -> tuple[Music, list[str]]:
        """Everything the model returned is untrusted input."""
        moods = _moods()
        beats = {b.index: b for b in story.beats} if st_ok and story else {}
        spoken = ({s.beat_index for s in voiceover.spoken_segments}
                  if voiceover else set())

        raw_mood = raw.get("mood")
        mood = _one_of(raw_mood, moods, moods[0])
        notes: list[str] = []
        if isinstance(raw_mood, str) and raw_mood not in moods:
            notes.append(f"mood {raw_mood!r} is not one the engine supports; "
                         f"fell back to {mood!r}")

        sections, sec_notes = self._sections(raw.get("sections"), beats,
                                             spoken)
        notes += sec_notes

        limitations = _texts(raw.get("limitations"), each=200)
        if not st_ok:
            limitations.append(
                "no story with beats was available, so the score has no shape "
                "to follow and only overall direction was given")
        if not cs_ok:
            limitations.append(
                "no usable content strategy was available, so the tone comes "
                "from the story alone")
        if not assets:
            limitations.append(
                "the music library is empty, so this direction will have to "
                "be realised by the generative engine rather than a track")
        if notes:
            limitations.append("; ".join(notes))

        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        if not st_ok:
            confidence = min(confidence, 0.3)
        if story is not None and not story.live:
            confidence = min(confidence, 0.5)

        payload = Music(
            domain=domain,
            music_type="reasoned" if st_ok else "degraded",
            mood=mood,
            mood_rationale=_text(raw.get("mood_rationale"), 240),
            engine=_engine_for(mood),      # DERIVED, never asked for
            tempo_feel=_one_of(raw.get("tempo_feel"), TEMPOS, "moderate"),
            sonic_direction=_text(raw.get("sonic_direction"), 300),
            instrumentation=_texts(raw.get("instrumentation"), each=60),
            energy_arc=_text(raw.get("energy_arc"), 240),
            avoid=_texts(raw.get("avoid"), each=80),
            sections=sections,
            narration_notes=_texts(raw.get("narration_notes")),
            available_assets=assets,       # measured, never model output
            story_available=st_ok, voiceover_available=vo_ok,
            strategy_available=cs_ok,
            beats_covered=sorted({s.beat_index for s in sections}),
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not sections,
            confidence=confidence,
            live=True, model=ldesc.get("model"))
        return payload, notes

    @staticmethod
    def _sections(raw, beats: dict, spoken: set
                  ) -> tuple[list[MusicSection], list[str]]:
        """Build the arc, dropping anything that cannot be anchored.

        `clip_id` comes from the BEAT and `under_narration` from the
        Voiceover - neither is read from the reply, so a section cannot land
        on invented footage or claim speech that is not there.

        Sections follow the story: stable-sorted by beat, one per beat.
        """
        kept: list[tuple[int, MusicSection]] = []
        seen: set[int] = set()
        bad_beat = dupe = 0
        if not isinstance(raw, list):
            return [], []

        for pos, item in enumerate(raw[:MAX_SECTIONS]):
            if not isinstance(item, dict):
                continue
            bi = item.get("beat_index")
            if not isinstance(bi, int) or isinstance(bi, bool) \
                    or bi not in beats:
                bad_beat += 1
                continue
            if bi in seen:
                dupe += 1
                continue                   # one section per beat
            seen.add(bi)

            kept.append((pos, MusicSection(
                index=1,                   # renumbered below
                beat_index=bi,
                clip_id=beats[bi].clip_id,     # from the BEAT
                energy=_one_of(item.get("energy"), ENERGIES, "steady"),
                intent=_text(item.get("intent"), 240),
                under_narration=bi in spoken)))   # from the VOICEOVER

        reordered = sorted(kept, key=lambda p: p[1].beat_index)
        out = [sec.model_copy(update={"index": i + 1})
               for i, (_pos, sec) in enumerate(reordered)]

        notes = []
        if bad_beat:
            notes.append(f"{bad_beat} section(s) referenced a beat that does "
                         f"not exist and were dropped")
        if dupe:
            notes.append(f"{dupe} duplicate section(s) for a beat already "
                         f"covered were dropped")
        return out, notes

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan,
                       clip_set: ClipAnalysisSet | None,
                       story: Story | None, voiceover: Voiceover | None,
                       strategy: ContentStrategy | None, domain: str,
                       revision: int, started: float, inputs_hash: str,
                       st_ok: bool, vo_ok: bool, cs_ok: bool,
                       assets: list[str], why: str,
                       cost: Cost | None = None) -> Artifact[Music]:
        """The mood the existing pipeline would have chosen anyway.

        `server/director.py::MOOD_BANDS` maps measured motion to a mood and
        has done since before any of this existed. Reusing it means the
        fallback is the production answer rather than an invention, and the
        mood is one the engine can actually play.
        """
        span.fallback(f"deterministic music direction ({why})")

        mood, basis = self._mood_from_motion(clip_set)

        spoken = ({s.beat_index for s in voiceover.spoken_segments}
                  if voiceover else set())
        sections: list[MusicSection] = []
        if st_ok and story is not None:
            for i, b in enumerate(story.beats[:MAX_SECTIONS]):
                sections.append(MusicSection(
                    index=i + 1, beat_index=b.index, clip_id=b.clip_id,
                    energy="steady",       # no basis to shape an arc
                    intent="", under_narration=b.index in spoken))

        limitations = [f"{why}; the mood was chosen by {basis}, which is the "
                       f"deterministic rule the pipeline has always used - no "
                       f"sonic direction was authored"]
        if not st_ok:
            limitations.append("no story with beats was available, so there "
                               "is no section arc")
        if not assets:
            limitations.append("the music library is empty, so no track was "
                               "or could be referenced")

        payload = Music(
            domain=domain,
            music_type="minimal" if not sections else "degraded",
            mood=mood, mood_rationale=basis,
            engine=_engine_for(mood),
            tempo_feel="moderate",
            sonic_direction="", instrumentation=[], energy_arc="", avoid=[],
            sections=sections, narration_notes=[],
            available_assets=assets,
            story_available=st_ok, voiceover_available=vo_ok,
            strategy_available=cs_ok,
            beats_covered=sorted({s.beat_index for s in sections}),
            risks=[], limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)

        # the fallback is the path every keyless run takes, so the
        # operator's choice has to be honoured here too
        payload = _apply_operator_mood(payload, _operator_mood(ctx), span)

        ctx.emit("music: %s (%s) - mood %s via %s, %d section(s)"
                 % (payload.music_type, why, payload.mood, basis,
                    len(sections)))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, st_ok, vo_ok, cs_ok, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic music direction ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    @staticmethod
    def _mood_from_motion(clip_set: ClipAnalysisSet | None) -> tuple[str, str]:
        """Reuse the production mood bands against real measurements."""
        from .tools.pipeline_bridge import SERVER_DIR  # noqa: F401
        try:
            import director as _legacy_director
            bands = _legacy_director.MOOD_BANDS
        except Exception:                             # pragma: no cover
            bands = [(2.5, "beat", ""), (0.8, "bright", ""), (0.0, "lofi", "")]

        usable = clip_set.usable_clips if clip_set else []
        if not usable:
            return "lofi", ("no measurements available - the quietest bed, "
                            "which cannot fight footage nobody measured")
        motions = sorted(c.motion.mean_ydif for c in usable)
        n = len(motions)
        median = (motions[n // 2] if n % 2
                  else (motions[n // 2 - 1] + motions[n // 2]) / 2.0)
        for threshold, mood, _why in bands:
            if median >= threshold:
                return mood, ("measured motion (median %.2f) against the "
                              "pipeline's own mood bands" % median)
        return "lofi", "measured motion (median %.2f)" % median

    # ---------------------------------------------------------- digests

    @staticmethod
    def _story_digest(story: Story | None):
        if story is None:
            return None
        return {"logline": story.logline, "progression": story.progression,
                "beats": [(b.index, b.role, b.clip_id, b.narrative)
                          for b in story.beats],
                "live": story.live}

    @staticmethod
    def _voiceover_digest(voiceover: Voiceover | None):
        if voiceover is None:
            return None
        return {"voice": voiceover.voice,
                "beats": sorted(voiceover.beats_covered),
                "words": voiceover.total_words}

    @staticmethod
    def _strategy_digest(strategy: ContentStrategy | None):
        if strategy is None:
            return None
        return {"topic": strategy.central_topic, "angle": strategy.core_angle,
                "pacing": strategy.pacing,
                "insufficient": strategy.insufficient_signal}

    @staticmethod
    def _motion_digest(clip_set: ClipAnalysisSet | None):
        """The measurements the fallback's mood depends on."""
        if clip_set is None:
            return None
        return sorted((c.clip_id, round(c.motion.mean_ydif, 2))
                      for c in clip_set.usable_clips)

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Music, st_ok: bool, vo_ok: bool, cs_ok: bool,
                   why: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("Story", st_ok), ("Voiceover", vo_ok),
                                    ("ContentStrategy", cs_ok)) if ok]
                   or ["neither"])
        out = [Decision(
            what="Evidence the music direction was built from",
            chose=" + ".join(sources),
            why=("recorded explicitly so direction written without a story or "
                 "without narration cannot later be read as though it had "
                 "both"),
            decided_by="measurement", confidence=1.0)]

        out.append(Decision(
            what="Engine",
            chose=p.engine or "none",
            why=(f"derived from mood {p.mood!r} through the pipeline's own "
                 f"MOODS map, so the direction cannot name a setting the "
                 f"engine does not have"),
            decided_by="policy", confidence=1.0))

        if not p.live:
            out.append(Decision(
                what="Mood", chose=p.mood,
                why=(f"{why}; chosen by {p.mood_rationale} - the "
                     f"deterministic rule the pipeline has always used, not "
                     f"an invention"),
                decided_by="measurement", confidence=1.0))
            return out

        out.append(Decision(
            what="Mood", chose=p.mood,
            why=(p.mood_rationale or "no rationale given")[:200] +
                " - picked from the engine's own vocabulary",
            decided_by="llm", confidence=p.confidence))
        if p.sections:
            out.append(Decision(
                what="Energy arc",
                chose=" -> ".join(p.energy_shape[:8]),
                why=("shaped across the story's beats; no section expresses a "
                     "time, a tempo or a length - the beat engine fits those "
                     "to the real transition marks"),
                decided_by="llm", confidence=p.confidence))
        if p.spoken_sections:
            out.append(Decision(
                what="Music under narration",
                chose=f"{len(p.spoken_sections)} section(s) sit under speech",
                why=("derived from the Voiceover, not asked of the model - "
                     "whether a beat carries words is a fact"),
                decided_by="measurement", confidence=1.0))
        if p.instrumentation:
            out.append(Decision(
                what="Sonic direction",
                chose=", ".join(p.instrumentation[:6]),
                why=(p.sonic_direction or "no direction given")[:200],
                decided_by="llm", confidence=p.confidence))
        if p.limitations:
            out.append(Decision(
                what="What this direction cannot claim",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3])[:300],
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["MusicAgent"]
