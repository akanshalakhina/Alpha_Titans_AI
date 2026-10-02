"""
Typed artifacts and the envelope every agent output carries.

WHY THIS EXISTS
---------------
Today six processes read and mutate one untyped `manifest.json` with no schema,
no version and no locking (see the audit). That file does not disappear - the
existing pipeline still reads it - but it stops being the source of truth. It
becomes a PROJECTION written from these artifacts immediately before the
existing `cut`/`join` stages are invoked.

THE RULES, ENFORCED HERE RATHER THAN BY CONVENTION
--------------------------------------------------
* Artifacts are immutable. `Artifact` is frozen. A revision produces a NEW
  artifact at `revision = n + 1`; it never edits `n`. That is what lets the
  critic compare two attempts and ship the better one.
* Every artifact records who produced it, from which inputs (`inputs_hash`),
  with what confidence, and whether it came from a fallback.
* Every artifact carries `decisions: [Decision]`. That shape is lifted
  verbatim from `server/director.py`, which already records
  what / chose / why / decided_by for every choice it makes. It was the best
  idea in the original codebase and it is now mandatory for all agents.

`inputs_hash` is a content hash of the upstream artifacts an agent consumed.
Identical inputs mean an agent can be skipped, which is what makes a revision
cheap: a revision that only re-plans the edit does not re-measure the clips.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Generic, Literal, TypeVar, get_args

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

SCHEMA_VERSION = 1


# --------------------------------------------------------------------- ids

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def content_hash(payload: Any) -> str:
    """Stable hash of any JSON-serialisable structure.

    `sort_keys` matters: two dicts that differ only in key order describe the
    same inputs and must produce the same hash, or caching never hits.
    """
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- decision

DecidedBy = Literal["llm", "measurement", "policy", "human", "fallback"]


class Decision(BaseModel):
    """One recorded choice, with the reason attached.

    Lifted from server/director.py's `decisions` list. The `why` field is not
    decoration: the critic routes a revision at a decision, and a human
    arguing with the edit needs to see the reasoning, not just the outcome.
    """
    model_config = ConfigDict(frozen=True)

    what: str
    chose: str
    why: str
    decided_by: DecidedBy
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ----------------------------------------------------------------- costs

class Cost(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    wall_s: float = 0.0


# -------------------------------------------------------------- envelope

ArtifactKind = Literal[
    "JobState", "ClipAnalysis", "VisionAnalysis", "ResearchReport",
    "ContentStrategy", "Story", "Voiceover", "Music", "Animation",
    "Transition", "Caption",
    "EditPlan", "RenderSpec", "RenderResult", "CriticResult",
    "RevisionRequest", "Thumbnail", "Title", "Publishing",
    "PerformanceAnalysis", "Learning",
]

T = TypeVar("T", bound=BaseModel)


class Artifact(BaseModel, Generic[T]):
    """The envelope every agent output is wrapped in.

    Frozen on purpose. If something needs to change, a new revision is
    produced and both are kept - that is the whole point of the design.
    """
    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    job_id: str
    kind: ArtifactKind
    schema_version: int = SCHEMA_VERSION
    revision: int = 0

    produced_by: str                     # agent id
    produced_at: float = Field(default_factory=time.time)
    inputs_hash: str = ""
    trace_id: str = ""

    payload: T

    decisions: list[Decision] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    degraded: bool = False
    fallback_used: str | None = None
    cost: Cost = Field(default_factory=Cost)

    def summary(self) -> str:
        flag = " DEGRADED" if self.degraded else ""
        fb = f" fallback={self.fallback_used}" if self.fallback_used else ""
        return (f"{self.kind}[rev{self.revision}] by {self.produced_by} "
                f"conf={self.confidence:.2f}{flag}{fb} id={self.artifact_id}")


# ------------------------------------------------------------- JobState

class JobSettings(BaseModel):
    """Operator settings. These always win over anything an agent decides."""
    model_config = ConfigDict(frozen=True)

    orientation: Literal["auto", "horizontal", "vertical"] = "auto"
    resolution: Literal["720p", "1080p", "1440p", "4k"] = "1080p"
    quality: Literal["draft", "standard", "high", "master"] = "high"
    target_seconds: float | None = None
    title: str = "ALPHA_TITANS"
    transition_duration: float = 0.4

    # What the footage IS, in the operator's own words - "a pasta recipe",
    # "a trail run in Snowdonia", "an unboxing". Passed to the Vision agent
    # so its prompt describes the actual subject instead of assuming one.
    # Empty means unspecified, and the prompt says so rather than guessing.
    domain: str = ""

    # Phase 1 test affordance: force a specific clip to open the cut. Used to
    # produce a genuinely weak hook so the critic's revision path can be
    # demonstrated on real footage rather than asserted.
    force_opening_clip: str | None = None


class JobState(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    src_dir: str
    build_dir: str
    settings: JobSettings = Field(default_factory=JobSettings)
    created_at: float = Field(default_factory=time.time)
    max_revisions: int = 2


# ---------------------------------------------------------- ClipAnalysis

class ClipTechnical(BaseModel):
    model_config = ConfigDict(frozen=True)

    duration: float
    coded: tuple[int, int]
    display: tuple[int, int]
    rotation: int = 0
    fps: float = 0.0
    codec: str | None = None
    has_audio: bool = False
    recorded_at: str | None = None
    orientation: Literal["vertical", "horizontal"]


class ClipMotion(BaseModel):
    """Everything the existing measure stage produces, typed.

    `ydif_series` is kept inline here rather than in a blob store because
    Phase 1 runs on small batches and `best_window` needs the whole series.
    The architecture calls for a blob ref at scale; that is a Phase 2 change
    and is noted rather than pretended.
    """
    model_config = ConfigDict(frozen=True)

    sample_fps: float
    mean_ydif: float
    max_ydif: float
    live_share: float
    freeze_runs: list[tuple[float, float]] = Field(default_factory=list)
    freeze_share: float = 0.0
    ydif_series: list[float] = Field(default_factory=list)
    measured: bool = True          # False when signalstats failed


class ClipUsability(BaseModel):
    model_config = ConfigDict(frozen=True)

    usable: bool = True
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ClipAnalysis(BaseModel):
    """One clip, fully characterised. The EditingDirector plans from these."""
    model_config = ConfigDict(frozen=True)

    clip_id: str                   # the on-disk filename; never invented
    path: str
    technical: ClipTechnical
    motion: ClipMotion
    usability: ClipUsability = Field(default_factory=ClipUsability)
    slowmo: bool = False

    def to_planner_dict(self) -> dict:
        """Shape the existing pipeline functions expect.

        `best_window` and `window_score` in pipeline/build_video.py read
        exactly these keys off a clip dict. Producing that shape here is what
        lets those functions be reused unchanged as deterministic tools.
        """
        return {
            "name": self.clip_id,
            "duration": self.technical.duration,
            "sample_fps": self.motion.sample_fps,
            "ydif": self.motion.ydif_series,
            "mean_ydif": self.motion.mean_ydif,
            "max_ydif": self.motion.max_ydif,
            "freeze_runs": [list(r) for r in self.motion.freeze_runs],
            "freeze_share": self.motion.freeze_share,
            "live_share": self.motion.live_share,
            "orientation": self.technical.orientation,
            "slowmo": self.slowmo,
            "display": list(self.technical.display),
            "fps": self.technical.fps,
            "codec": self.technical.codec,
        }


class ClipAnalysisSet(BaseModel):
    """The ClipAnalysis agent's payload: every clip plus batch-level facts."""
    model_config = ConfigDict(frozen=True)

    clips: list[ClipAnalysis]
    orientation: Literal["vertical", "horizontal"]
    canvas: tuple[int, int]
    target_seconds: float
    unreadable: list[str] = Field(default_factory=list)
    dropped: dict[str, str] = Field(default_factory=dict)

    @property
    def usable_clips(self) -> list[ClipAnalysis]:
        return [c for c in self.clips if c.usability.usable]

    def by_id(self, clip_id: str) -> ClipAnalysis | None:
        return next((c for c in self.clips if c.clip_id == clip_id), None)


# -------------------------------------------------- VisionAnalysis

class CandidateWindow(BaseModel):
    """A window MEASURED from the footage, offered to the model to choose from.

    This type is the enforcement point for the rule that the model may not
    invent timestamps. Every window here was produced by the pipeline's own
    `best_window` / motion series; the model is only ever allowed to return an
    INDEX into a list of these, never a number of its own.
    """
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    duration: float = Field(gt=0.0)
    mean_ydif: float                    # measured motion over the window
    score: float                        # freeze-penalised window score
    freeze_overlap: float = 0.0


class InterestingMoment(BaseModel):
    model_config = ConfigDict(frozen=True)

    at: float = Field(ge=0.0)           # clamped to the clip's real duration
    what: str


class VisionClip(BaseModel):
    """What a model saw in ONE clip, plus the measurements it was shown.

    Deliberately domain-neutral: `subjects`, `actions` and `scene_description`
    are free text the model fills for whatever the footage actually is - a
    kitchen, a trail, a match, a product demo. Nothing here assumes a subject.
    """
    model_config = ConfigDict(frozen=True)

    clip_id: str
    analysed: bool = True               # False when the model failed on it
    summary: str = ""
    subjects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene_description: str = ""
    interesting_moments: list[InterestingMoment] = Field(default_factory=list)
    hook_score: float = Field(default=0.0, ge=0.0, le=100.0)
    visual_quality: float = Field(default=0.0, ge=0.0, le=100.0)

    # measured, supplied TO the model
    candidate_windows: list[CandidateWindow] = Field(default_factory=list)
    # the model's pick, as an index into the list above - never a timestamp
    chosen_window: int | None = None
    window_rationale: str = ""

    confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    notes: str = ""
    failure: str | None = None

    def best_window(self) -> CandidateWindow | None:
        """The window the model picked, or the highest-scoring measured one."""
        if not self.candidate_windows:
            return None
        if self.chosen_window is not None:
            for w in self.candidate_windows:
                if w.index == self.chosen_window:
                    return w
        return max(self.candidate_windows, key=lambda w: w.score)


class VisionAnalysis(BaseModel):
    """The Vision agent's payload: per clip, plus cross-clip relationships."""
    model_config = ConfigDict(frozen=True)

    domain: str = ""                    # operator-declared, "" = unspecified
    provider: str | None = None
    model: str | None = None
    frames_per_clip: int = 0
    live: bool = False                  # True only for a real model call

    clips: dict[str, VisionClip] = Field(default_factory=dict)

    # cross-clip, every id clamped to one we actually analysed
    ordering_signal: list[str] = Field(default_factory=list)
    hook_clip: str | None = None
    duplicates: list[tuple[str, str]] = Field(default_factory=list)
    story: str = ""

    failures: list[str] = Field(default_factory=list)

    @property
    def analysed_clips(self) -> list[VisionClip]:
        return [c for c in self.clips.values() if c.analysed]

    def for_clip(self, clip_id: str) -> VisionClip | None:
        return self.clips.get(clip_id)


# -------------------------------------------------- ResearchReport

class EvidenceItem(BaseModel):
    """One retrieved result. Provenance is mandatory, never inferred.

    This is the enforcement point for "do not fabricate metrics": every claim
    the Research agent makes has to point at one of these, and each one was
    returned by a real provider for a real query. `published_at` is None when
    the provider gave no date - it is never guessed, because presenting old
    material as current is the specific failure this field exists to prevent.
    """
    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str = ""
    source: str                      # which provider returned it
    query: str                       # what was asked to get it
    retrieved_at: float              # when WE fetched it
    published_at: str | None = None  # None = the provider gave no date
    score: float | None = None

    @property
    def dated(self) -> bool:
        return self.published_at is not None


class TrendItem(BaseModel):
    """A pattern the model read OUT of the evidence, with citations."""
    model_config = ConfigDict(frozen=True)

    label: str
    observation: str
    evidence_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ContentOpportunity(BaseModel):
    """A suggestion for ContentStrategy to weigh - never a decision."""
    model_config = ConfigDict(frozen=True)

    idea: str
    rationale: str = ""
    evidence_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchReport(BaseModel):
    """What the world currently says about this video's subject.

    INFORMATION, NOT DECISIONS. Nothing here picks a story, a title, an edit
    or a publishing time; ContentStrategy will do that with this in hand.

    `insufficient_signal` is a first-class outcome. No provider, no results, a
    timeout or an unusable reply all land here with it set - which is an
    honest answer, and far more useful than a plausible invented one.
    """
    model_config = ConfigDict(frozen=True)

    domain: str = ""
    research_queries: list[str] = Field(default_factory=list)
    retrieved_at: float = Field(default_factory=time.time)

    provider: str = "none"
    live: bool = False               # True only when a real provider answered
    model: str | None = None         # the synthesising model, if any

    evidence: list[EvidenceItem] = Field(default_factory=list)
    trends: list[TrendItem] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    audience_signals: list[str] = Field(default_factory=list)
    content_opportunities: list[ContentOpportunity] = Field(default_factory=list)

    insufficient_signal: bool = True
    limitations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def dated_evidence(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.dated]

    @property
    def undated_count(self) -> int:
        return sum(1 for e in self.evidence if not e.dated)

    def evidence_for(self, url: str) -> EvidenceItem | None:
        return next((e for e in self.evidence if e.url == url), None)


# -------------------------------------------------- ContentStrategy

Pacing = Literal["fast", "moderate", "slow", "varied"]
ClipRole = Literal["hook", "develop", "support", "payoff", "cutaway"]
Support = Literal["vision", "research", "both", "none"]
StrategyType = Literal["reasoned", "degraded", "minimal"]


class KeyPoint(BaseModel):
    """One thing the video should communicate, with its grounding attached.

    `support` is not decoration. A point claiming research support must cite a
    URL that is actually in the ResearchReport, and one claiming vision support
    must name a clip Vision actually analysed. Validation downgrades or drops
    anything that cannot show its working, which is how "no hallucinated facts"
    is enforced rather than merely requested.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    point: str
    support: Support = "none"
    evidence_urls: list[str] = Field(default_factory=list)
    clip_ids: list[str] = Field(default_factory=list)


class ClipPriority(BaseModel):
    """How much a clip matters to the message - NOT where to cut it.

    `extra="forbid"` is the timestamp guard, and it is structural rather than
    advisory: a model reply carrying `start`, `end` or `duration` fails
    validation outright instead of having the field quietly ignored. Timing is
    a measurement, and it stays with the deterministic window search.

    `moment_ref` is an INDEX into the clip's `interesting_moments` - the same
    discipline Vision uses for `chosen_window`. An index can only ever point at
    something that was already measured.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    clip_id: str
    priority: int = Field(ge=1)
    role: ClipRole = "support"
    why: str = ""
    moment_ref: int | None = Field(default=None, ge=0)


class Hook(BaseModel):
    """What should open the video, described rather than timed."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    clip_id: str | None = None
    description: str = ""
    why: str = ""


class ContentStrategy(BaseModel):
    """WHAT the video should communicate. Never HOW it is cut.

    This is the decision layer between raw understanding (Vision + Research)
    and execution (EditingDirector). It reads both, decides the message, and
    hands the director an intent to plan against - it does not plan the edit,
    choose in/out points, write the voiceover, pick the title or publish.

    DEGRADATION IS EXPLICIT
    -----------------------
    `vision_available` and `research_available` record what this strategy was
    actually built from. A strategy formed without research is a legitimate
    output; one that quietly pretends it had research is not. `strategy_type`
    says which of the three it is, and `limitations` says what is missing in
    words a human reads.
    """
    model_config = ConfigDict(frozen=True)

    domain: str = ""
    strategy_type: StrategyType = "minimal"

    audience: str = ""
    objective: str = ""
    central_topic: str = ""
    core_angle: str = ""
    hook: Hook = Field(default_factory=Hook)
    narrative_direction: str = ""

    key_points: list[KeyPoint] = Field(default_factory=list)
    clip_priorities: list[ClipPriority] = Field(default_factory=list)

    pacing: Pacing = "moderate"
    pacing_rationale: str = ""
    engagement_strategy: str = ""

    # provenance: what this strategy was actually built from
    vision_available: bool = False
    research_available: bool = False
    research_basis: list[str] = Field(default_factory=list)   # cited urls
    vision_basis: list[str] = Field(default_factory=list)     # cited clip ids
    content_opportunities: list[str] = Field(default_factory=list)

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @property
    def grounded_points(self) -> list[KeyPoint]:
        return [k for k in self.key_points if k.support != "none"]

    def priority_for(self, clip_id: str) -> ClipPriority | None:
        return next((c for c in self.clip_priorities
                     if c.clip_id == clip_id), None)

    @property
    def ordered_clips(self) -> list[str]:
        """Clip ids by priority - a suggestion to the director, not an order."""
        return [c.clip_id
                for c in sorted(self.clip_priorities, key=lambda c: c.priority)]


# ------------------------------------------------------------- Story

BeatRole = Literal["hook", "setup", "develop", "turn", "payoff", "ending"]
StoryType = Literal["reasoned", "degraded", "minimal"]


class StoryBeat(BaseModel):
    """One moment in the narrative, pinned to footage that exists.

    `extra="forbid"` is the timestamp guard, and it is structural rather than
    advisory: a reply carrying `start`, `end`, `duration` or `at` fails
    validation outright instead of having the field quietly ignored. A beat
    says WHAT happens and in WHICH clip; when it happens stays a measurement,
    and that stays with the deterministic window search downstream.

    `moment_ref` is an INDEX into the clip's `interesting_moments` - the same
    discipline Vision uses for `chosen_window` and ContentStrategy uses for
    its own clip priorities. An index can only ever point at something that
    was already measured.

    `support` is derived from what survived validation, never from what the
    model claimed: a beat citing research must name a url actually in the
    ResearchReport, and its clip must be one Vision actually analysed.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1)
    role: BeatRole = "develop"
    clip_id: str
    moment_ref: int | None = Field(default=None, ge=0)
    narrative: str = ""
    why: str = ""
    support: Support = "none"
    evidence_urls: list[str] = Field(default_factory=list)


class Story(BaseModel):
    """The narrative shape of the video: what is told, and in what order.

    ContentStrategy decides WHAT the video should communicate. Story decides
    HOW that lands as a sequence - the hook, the setup, the beats between,
    the payoff and the ending. The EditingDirector then decides how it is
    CUT. None of those three is the same question, which is why this sits
    between them rather than inside either.

    WHAT IT MAY NOT DO
    ------------------
    No timestamps, no durations, no in/out points, no transitions, no filter
    strings, no shell. Every beat names a clip id and, optionally, an index
    into moments Vision already measured. Timing belongs to the deterministic
    window search, and nothing here can express it.

    DEGRADATION IS EXPLICIT
    -----------------------
    `strategy_available`, `vision_available` and `research_available` record
    what this story was actually built from. A story written without a
    semantic reading of the footage is a legitimate output; one that quietly
    pretends it had one is not.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    story_type: StoryType = "minimal"

    logline: str = ""
    hook: str = ""
    setup: str = ""
    progression: str = ""
    payoff: str = ""
    ending: str = ""

    beats: list[StoryBeat] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)

    # provenance: what this story was actually built from
    strategy_available: bool = False
    vision_available: bool = False
    research_available: bool = False
    research_basis: list[str] = Field(default_factory=list)   # cited urls
    vision_basis: list[str] = Field(default_factory=list)     # cited clip ids

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("beats")
    @classmethod
    def _indices_contiguous(cls, v):
        """Beats are an ORDER. Gaps or repeats would make the sequence
        ambiguous, so they are rejected rather than silently renumbered
        here - the agent renumbers before construction."""
        if not v:
            return v
        want = list(range(1, len(v) + 1))
        got = [b.index for b in v]
        if got != want:
            raise ValueError(f"beat indices must be 1..n in order, got {got}")
        return v

    @property
    def grounded_beats(self) -> list[StoryBeat]:
        return [b for b in self.beats if b.support != "none"]

    @property
    def clip_order(self) -> list[str]:
        """Clip ids in narrative order - a suggestion to the director, with
        no timing attached and no duplicate removed: a clip may legitimately
        appear in more than one beat."""
        return [b.clip_id for b in self.beats]

    def beat_for(self, role: str) -> "StoryBeat | None":
        return next((b for b in self.beats if b.role == role), None)


# --------------------------------------------------------- Voiceover

Tone = Literal["neutral", "warm", "energetic", "calm", "authoritative",
               "playful", "wry"]
VoiceoverType = Literal["reasoned", "degraded", "minimal"]


class NarrationSegment(BaseModel):
    """One spoken line, pinned to a beat the Story already decided.

    `extra="forbid"` is the guard, and it is structural rather than advisory:
    a reply carrying `start`, `end`, `duration`, `at` or `seconds` fails
    validation outright. There is no field here that can express WHEN a line
    is spoken or HOW LONG it lasts - that is a timing decision, and timing
    belongs to the deterministic window search downstream.

    `beat_index` is the only pointer the model supplies. `clip_id` and
    `moment_ref` are COPIED from the referenced beat rather than read from
    the reply, which makes inventing footage structurally impossible: a
    segment can only ever point at a clip the Story already anchored.

    `word_count` is a computed property, not a field. It is a measured
    property of the text - not a duration, and not something a model can
    inject.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1)
    beat_index: int = Field(ge=1)       # -> StoryBeat.index
    clip_id: str                        # copied from that beat
    moment_ref: int | None = Field(default=None, ge=0)   # copied from it too
    text: str = ""
    tone: Tone = "neutral"
    support: Support = "none"
    evidence_urls: list[str] = Field(default_factory=list)
    note: str = ""                      # delivery note, never a direction

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def spoken(self) -> bool:
        """False for a structural placeholder carrying no words."""
        return bool(self.text.strip())


class Voiceover(BaseModel):
    """The narration script, aligned to the Story's beats.

    Story decides what is told and in what order. This decides what is SAID
    over it. The EditingDirector still decides how it is cut, and nothing
    here expresses a time, a duration, a transition or a filter.

    A segment references a beat by INDEX. The clip and moment come from that
    beat, so the script cannot drift onto footage the narrative never
    anchored.

    DEGRADATION IS EXPLICIT
    -----------------------
    With no model the agent emits the SHAPE of a script - one placeholder per
    beat, carrying no words - and says so. Writing narration with nothing to
    write it from would be invention presented as a script, which is the
    failure this whole layer exists to avoid.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    voiceover_type: VoiceoverType = "minimal"

    # NOT `register`: that name shadows an attribute on pydantic's BaseModel
    # and emits a UserWarning on every import
    voice: Tone = "neutral"             # the overall delivery
    voice_rationale: str = ""
    audience: str = ""

    segments: list[NarrationSegment] = Field(default_factory=list)
    delivery_notes: list[str] = Field(default_factory=list)

    # provenance: what this script was actually built from
    story_available: bool = False
    strategy_available: bool = False
    research_available: bool = False
    research_basis: list[str] = Field(default_factory=list)   # cited urls
    beats_covered: list[int] = Field(default_factory=list)    # beat indices

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("segments")
    @classmethod
    def _indices_contiguous(cls, v):
        """Segments are an ORDER. A gap or a repeat would make the script
        ambiguous, so it is rejected rather than silently renumbered here -
        the agent renumbers before construction."""
        if not v:
            return v
        want = list(range(1, len(v) + 1))
        got = [seg.index for seg in v]
        if got != want:
            raise ValueError(
                f"segment indices must be 1..n in order, got {got}")
        return v

    @property
    def spoken_segments(self) -> list[NarrationSegment]:
        return [s for s in self.segments if s.spoken]

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.segments)

    @property
    def script(self) -> str:
        """The narration as plain text, in order. Data, never a command."""
        return "\n".join(s.text for s in self.segments if s.spoken)

    def for_beat(self, beat_index: int) -> list[NarrationSegment]:
        return [s for s in self.segments if s.beat_index == beat_index]


# ------------------------------------------------------------- Music

# The SEVEN moods the existing engine actually supports. Lifted verbatim from
# `server/extras.py::MOODS`, whose own comment reads "Every option maps to a
# real engine setting" - lofi/calm -> stdlib lofi, bright/upbeat -> stdlib
# bright, beat/energetic/cinematic -> beatbed.
#
# This is the enforcement point for "the music plan must be actionable": a
# mood outside this set is one nothing can render, so the model picks from
# the engine's own vocabulary rather than inventing a genre. A test asserts
# this Literal still matches extras.MOODS exactly, so the two cannot drift.
Mood = Literal["lofi", "calm", "bright", "upbeat", "beat", "energetic",
               "cinematic"]
Energy = Literal["low", "building", "steady", "high", "peak", "falling"]
TempoFeel = Literal["slow", "moderate", "driving", "varied"]
MusicType = Literal["reasoned", "degraded", "minimal"]


class MusicSection(BaseModel):
    """What the score does across ONE story beat.

    `extra="forbid"` is the guard, and it is structural rather than advisory:
    a reply carrying `start`, `end`, `duration`, `at`, `bpm` or `seconds`
    fails validation outright. Nothing here can say WHEN a section begins or
    HOW LONG it lasts - the real transition marks are measured from the
    finished cut, and tempo is fitted to them by the existing beat engine.

    `beat_index` is the only pointer the model supplies. `clip_id` is COPIED
    from the referenced beat, so a section cannot land on footage the story
    never anchored.

    `under_narration` is derived from the Voiceover, not asked of the model -
    whether a beat carries spoken words is a fact, not a judgement.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1)
    beat_index: int = Field(ge=1)       # -> StoryBeat.index
    clip_id: str                        # copied from that beat
    energy: Energy = "steady"
    intent: str = ""                    # what the music does here
    under_narration: bool = False       # derived from the Voiceover


class Music(BaseModel):
    """The music DIRECTION for the cut - never the audio, never the timing.

    Story says what is told; Voiceover says what is spoken; this says what it
    should sound like underneath. Choosing or generating the actual track,
    fitting tempo to the real transition marks and ducking under speech all
    stay with the existing deterministic engines (`beatbed`, `bgm_stdlib`),
    which do that from measurements this agent never sees.

    NO ASSET IS EVER NAMED HERE. `available_assets` is measured from the music
    library on disk; the model is not shown it and cannot add to it. A plan
    that referenced a track nobody has would be worse than one that referenced
    none.

    `engine` is derived from `mood` through the existing MOODS map, so it is a
    consequence of the choice rather than a second thing to get wrong.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    music_type: MusicType = "minimal"

    mood: Mood = "lofi"                 # from the engine's own vocabulary
    mood_rationale: str = ""
    engine: str = ""                    # DERIVED from mood, never asked for
    tempo_feel: TempoFeel = "moderate"

    sonic_direction: str = ""           # how it should sound, in words
    instrumentation: list[str] = Field(default_factory=list)
    energy_arc: str = ""
    avoid: list[str] = Field(default_factory=list)

    sections: list[MusicSection] = Field(default_factory=list)
    narration_notes: list[str] = Field(default_factory=list)

    # what the library actually holds, measured - never model output
    available_assets: list[str] = Field(default_factory=list)

    # provenance: what this direction was actually built from
    story_available: bool = False
    voiceover_available: bool = False
    strategy_available: bool = False
    beats_covered: list[int] = Field(default_factory=list)

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("sections")
    @classmethod
    def _indices_contiguous(cls, v):
        """Sections are an ORDER. A gap or a repeat would make the arc
        ambiguous, so it is rejected rather than silently renumbered here -
        the agent renumbers before construction."""
        if not v:
            return v
        want = list(range(1, len(v) + 1))
        got = [sec.index for sec in v]
        if got != want:
            raise ValueError(
                f"section indices must be 1..n in order, got {got}")
        return v

    @property
    def spoken_sections(self) -> list[MusicSection]:
        """Sections the narration talks over - where the bed must give way."""
        return [s for s in self.sections if s.under_narration]

    @property
    def energy_shape(self) -> list[str]:
        return [s.energy for s in self.sections]

    def section_for(self, beat_index: int) -> "MusicSection | None":
        return next((s for s in self.sections
                     if s.beat_index == beat_index), None)


# --------------------------------------------------------- Animation

# The TWO framing treatments `pipeline/build_video.py::fit_filter` actually
# implements. There is no third, and there is no motion of any kind: the
# render engine contains no zoompan, no Ken Burns, no pan and no zoom.
#
#   fill      orientations match -> scale to fill, then crop
#   blur_pad  they do not        -> blurred darkened bed, fitted foreground
#
# Which one applies is decided by orientation, not by taste, so it is DERIVED
# here rather than chosen. A test asserts both branches still exist in
# fit_filter, so this cannot quietly stop matching the engine.
Treatment = Literal["fill", "blur_pad"]

# What should draw the eye. This is judgement, and it is the one thing on a
# direction the model genuinely decides.
Emphasis = Literal["none", "subject", "context", "text"]

# Whether this beat wants words on screen. The Caption agent writes them; this
# only says whether there is a place for them, and the engine's drawtext can
# honour exactly that.
CaptionIntent = Literal["none", "short_label", "key_phrase"]

AnimationType = Literal["reasoned", "degraded", "minimal"]


class AnimationDirection(BaseModel):
    """How ONE story beat should be presented on the canvas.

    `extra="forbid"` is the guard, and it is structural: a reply carrying
    `start`, `end`, `duration`, `zoom`, `filter`, `path` or `ffmpeg` fails
    validation outright. Nothing here can express when something happens, how
    long it lasts, or how it is rendered.

    `beat_index` is the only pointer the model supplies. `clip_id` is COPIED
    from that beat and `treatment` is DERIVED from the clip's orientation
    against the canvas - the same rule `fit_filter` applies - so neither can
    be invented.

    `supported` records whether the engine can realise this direction today.
    An unsupported wish is kept and flagged rather than dropped, because it is
    useful to know what was wanted; it is never presented as something that
    will happen.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1)
    beat_index: int = Field(ge=1)       # -> StoryBeat.index
    clip_id: str                        # copied from that beat
    treatment: Treatment = "fill"       # DERIVED from orientation
    emphasis: Emphasis = "none"         # the model's judgement
    caption_intent: CaptionIntent = "none"
    intent: str = ""                    # why, in words
    supported: bool = True              # can the engine do this today?


class Animation(BaseModel):
    """How the cut should LOOK, within what the renderer can actually do.

    The engine's visual vocabulary is small and this artifact does not pretend
    otherwise. `engine_capabilities` is measured from the render module, and
    `unsupported_requests` records anything asked for that falls outside it -
    motion, zoom, parallax, anything with movement - so the gap is visible
    instead of implied.

    Between-segment transitions are NOT here. Those are xfade, and they belong
    to the Transition agent, which does not exist yet.

    NO ASSET IS EVER NAMED. There is no overlay library, no lower-third pack
    and no motion-graphics template in this project, so a direction that
    referenced one would be fiction.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    animation_type: AnimationType = "minimal"

    visual_style: str = ""              # the look, in a sentence
    pacing_feel: str = ""
    directions: list[AnimationDirection] = Field(default_factory=list)

    # what the renderer can actually do, measured - never model output
    engine_capabilities: list[str] = Field(default_factory=list)
    # what was asked for and cannot be done today
    unsupported_requests: list[str] = Field(default_factory=list)

    # provenance: what this direction was actually built from
    story_available: bool = False
    strategy_available: bool = False
    vision_available: bool = False
    beats_covered: list[int] = Field(default_factory=list)

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("directions")
    @classmethod
    def _indices_contiguous(cls, v):
        """Directions follow the beats in order. A gap or a repeat would make
        the sequence ambiguous, so it is rejected rather than silently
        renumbered here - the agent renumbers before construction."""
        if not v:
            return v
        want = list(range(1, len(v) + 1))
        got = [d.index for d in v]
        if got != want:
            raise ValueError(
                f"direction indices must be 1..n in order, got {got}")
        return v

    @property
    def supported_directions(self) -> list[AnimationDirection]:
        return [d for d in self.directions if d.supported]

    @property
    def captioned_beats(self) -> list[int]:
        """Beats that want words on screen, for the Caption agent later."""
        return [d.beat_index for d in self.directions
                if d.caption_intent != "none"]

    @property
    def treatments(self) -> list[str]:
        return [d.treatment for d in self.directions]

    def direction_for(self, beat_index: int) -> "AnimationDirection | None":
        return next((d for d in self.directions
                     if d.beat_index == beat_index), None)


# -------------------------------------------------------- Transition

# The SIX transitions this renderer ships, taken from the default XFADES list
# in pipeline/build_video.py and each confirmed to exist in ffmpeg's xfade
# filter. A name outside this set is one the render would fail on, so the
# model picks from here rather than from ffmpeg's full catalogue.
#
# There is deliberately NO "cut" and no "none": build_video crossfades EVERY
# junction, xfade has no such option, and the fade length `T` is global rather
# than per-transition. A hard cut is not expressible by this renderer and this
# artifact does not pretend otherwise.
#
# A test asserts this literal still matches the renderer's shipped default, so
# the two cannot drift.
TransitionKind = Literal["fade", "fadewhite", "smoothleft", "smoothup",
                         "wipeleft", "wipeup"]
TransitionType = Literal["reasoned", "degraded", "minimal"]


class TransitionChoice(BaseModel):
    """The transition at ONE junction between two consecutive beats.

    `extra="forbid"` is the guard, and it is structural: a reply carrying
    `duration`, `start`, `offset`, `seconds` or `filter` fails validation
    outright. Length is not expressible here because it is not expressible in
    the renderer either - `AT_XFADE_T` applies to every transition equally.

    `from_beat` and `to_beat` are the only pointers the model supplies, and
    both must be beats the Story actually has. `kind` is validated against the
    renderer's real vocabulary.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1)
    from_beat: int = Field(ge=1)        # -> StoryBeat.index
    to_beat: int = Field(ge=1)          # -> StoryBeat.index
    kind: TransitionKind = "fade"
    intent: str = ""
    supported: bool = True


class Transition(BaseModel):
    """How one beat gives way to the next.

    The renderer crossfades every junction using a cycling list of transition
    names supplied through `AT_XFADES`, which `RenderSpec.xfade_list` already
    feeds. So this artifact's `xfade_sequence` is directly actionable: it is
    the ordered list of names the render would use, one per junction.

    WHAT CANNOT BE EXPRESSED, AND IS NOT PRETENDED
    ----------------------------------------------
    * a hard cut - every junction crossfades, and xfade has no "none"
    * a per-transition length - `AT_XFADE_T` is one global value
    * anything outside the six names the renderer ships

    A request for any of those lands in `unsupported_requests` rather than
    being quietly dropped or silently promised.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    transition_type: TransitionType = "minimal"

    style: str = ""                     # the overall feel, in words
    choices: list[TransitionChoice] = Field(default_factory=list)

    # what the renderer actually supports, measured - never model output
    engine_transitions: list[str] = Field(default_factory=list)
    # what was asked for and cannot be done
    unsupported_requests: list[str] = Field(default_factory=list)

    # provenance: what this was actually built from
    story_available: bool = False
    animation_available: bool = False
    strategy_available: bool = False

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("choices")
    @classmethod
    def _indices_contiguous(cls, v):
        """Junctions run in order. A gap or a repeat would make the sequence
        ambiguous, so it is rejected rather than silently renumbered here -
        the agent renumbers before construction."""
        if not v:
            return v
        want = list(range(1, len(v) + 1))
        got = [c.index for c in v]
        if got != want:
            raise ValueError(
                f"choice indices must be 1..n in order, got {got}")
        return v

    @property
    def xfade_sequence(self) -> list[str]:
        """The ordered names the renderer would receive through AT_XFADES.

        One per junction, in order, so build_video's cycling index lands on
        exactly the intended transition at each one.
        """
        return [c.kind for c in self.choices]

    @property
    def supported_choices(self) -> list["TransitionChoice"]:
        return [c for c in self.choices if c.supported]

    def choice_between(self, from_beat: int,
                       to_beat: int) -> "TransitionChoice | None":
        return next((c for c in self.choices
                     if c.from_beat == from_beat and c.to_beat == to_beat),
                    None)


# ----------------------------------------------------------- Caption

# The ONE caption treatment `pipeline/build_video.py::fit_filter` implements:
#
#   drawtext  x=(w-text_w)/2   y=h-160 horizontal | h-470 vertical
#             fontcolor=white  borderw=3  bordercolor=black@0.7
#             fontsize 48 horizontal | 56 vertical
#             shown from 0.25s in until 0.35s before the segment ends
#
# One line per segment, never wrapped. There is no second position, no colour
# or box choice, no animation and no per-word timing - so each literal below
# has exactly one member, and a reply asking for anything else is recorded as
# unsupported rather than promised. A parity test asserts these still match
# fit_filter, so the two cannot quietly drift apart.
CaptionPosition = Literal["bottom_center"]
CaptionStyle = Literal["outlined_white"]

# Where a caption's words were FOUND - never what the model claimed. The agent
# locates the span in the beat's validated text and records which one held it.
CaptionSource = Literal["voiceover", "story"]
CaptionType = Literal["reasoned", "degraded", "minimal"]


class CaptionLine(BaseModel):
    """The words burnt onto ONE story beat's footage.

    `extra="forbid"` is the guard, and it is structural: a reply carrying
    `start`, `end`, `duration`, `font`, `color`, `filter` or `path` fails
    validation outright. When a caption appears, where it sits and how it
    looks are all fixed by the renderer, so none of them is expressible here.

    `beat_index` is the only pointer the model supplies. `clip_id` is COPIED
    from that beat, so a caption can only ever land on footage the Story
    already anchored. `text` is a verbatim span of that beat's validated
    narration or story line, located by the agent - a phrase that is not in
    the source is dropped, so inventing words is not detected, it is
    impossible. `source` records which of the two held it.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1)
    beat_index: int = Field(ge=1)       # -> StoryBeat.index
    clip_id: str                        # copied from that beat
    text: str = Field(min_length=1)     # a verbatim span of validated text
    source: CaptionSource               # DERIVED: where the span was found
    intent: CaptionIntent = "key_phrase"
    position: CaptionPosition = "bottom_center"   # the renderer's only one
    style: CaptionStyle = "outlined_white"        # likewise
    why: str = ""

    @field_validator("text")
    @classmethod
    def _one_printable_line(cls, v: str) -> str:
        """drawtext renders one line and does not wrap. A newline or control
        character here is a malformed caption, not a second line."""
        if not v.strip() or any(not ch.isprintable() for ch in v):
            raise ValueError("caption text must be one printable line")
        return v

    @field_validator("intent")
    @classmethod
    def _says_something(cls, v: str) -> str:
        if v == "none":
            raise ValueError("a caption line cannot have intent 'none'")
        return v


class Caption(BaseModel):
    """The words on screen, within what the renderer can actually draw.

    Story decides what is told, Voiceover what is said. This decides which of
    those words also appear on screen, and on which beat. It writes nothing
    new: every line is a span of text an upstream agent already produced and
    validated.

    WHERE IT GOES
    -------------
    The EditingDirector copies each line onto the segment showing the clip its
    beat names. From there it rides a path that already exists:
    `Segment.caption` -> `SegmentRenderSpec.caption` -> captions.json ->
    build_video `cut` -> drawtext.

    WHAT CANNOT BE EXPRESSED, AND IS NOT PRETENDED
    ----------------------------------------------
    * a position, colour, size, box or animation - the renderer has one style
    * any timing - each caption spans its whole segment, less fixed margins
    * more than one line, or more characters than fit across this canvas -
      `max_chars` is derived from the renderer's font size and canvas width
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    caption_type: CaptionType = "minimal"

    lines: list[CaptionLine] = Field(default_factory=list)

    # what the renderer can actually do, measured - never model output
    engine_options: list[str] = Field(default_factory=list)
    max_chars: int = Field(default=28, ge=1)
    # what was asked for and cannot be done
    unsupported_requests: list[str] = Field(default_factory=list)

    # provenance: what these captions were actually built from
    story_available: bool = False
    voiceover_available: bool = False
    animation_available: bool = False
    beats_eligible: list[int] = Field(default_factory=list)

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("lines")
    @classmethod
    def _one_per_beat_in_order(cls, v):
        """Lines are an ORDER, one per beat at most. A gap, a repeat or a
        second caption on the same beat would make it ambiguous which words
        land where, so each is rejected rather than silently repaired here -
        the agent renumbers and de-duplicates before construction."""
        if not v:
            return v
        want = list(range(1, len(v) + 1))
        got = [ln.index for ln in v]
        if got != want:
            raise ValueError(f"line indices must be 1..n in order, got {got}")
        beats = [ln.beat_index for ln in v]
        if len(set(beats)) != len(beats):
            raise ValueError(f"at most one caption per beat, got {beats}")
        return v

    @model_validator(mode="after")
    def _every_line_fits(self):
        """drawtext does not wrap: a line wider than the canvas is cut off by
        the frame edge. `max_chars` is derived from the renderer, so a line
        over it is one the render would visibly break."""
        for ln in self.lines:
            if len(ln.text) > self.max_chars:
                raise ValueError(
                    f"caption for beat {ln.beat_index} is {len(ln.text)} "
                    f"characters; one line on this canvas fits "
                    f"{self.max_chars}")
        return self

    @property
    def captioned_beats(self) -> list[int]:
        return [ln.beat_index for ln in self.lines]

    def for_beat(self, beat_index: int) -> "CaptionLine | None":
        return next((ln for ln in self.lines
                     if ln.beat_index == beat_index), None)

    def for_clip(self, clip_id: str) -> "CaptionLine | None":
        """The first caption, in story order, on a beat showing this clip."""
        return next((ln for ln in self.lines if ln.clip_id == clip_id), None)


# -------------------------------------------------------------- EditPlan

class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    clip_id: str
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    duration: float = Field(gt=0.0)
    score: float = 0.0
    orientation: Literal["vertical", "horizontal"]
    caption: str = ""
    reason: str = ""

    @field_validator("caption", mode="before")
    @classmethod
    def _caption_is_one_line(cls, v):
        """The last gate before text reaches drawtext.

        `CaptionLine.text` is already one printable line, so a caption coming
        from that agent is safe by construction. This exists for every OTHER
        way a Segment can be built - a fixture, a resumed job, a future caller
        - because drawtext does not wrap and a control character inside the
        filter graph is a malformed render. Normalised rather than rejected:
        an odd caption should cost its formatting, not the build.
        """
        if not isinstance(v, str):
            return ""
        return " ".join("".join(ch if ch.isprintable() else " "
                                for ch in v).split())

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be greater than start ({start})")
        return v

    def cache_key(self) -> str:
        """Identity of the RENDERED output of this segment.

        This is the fix for a real hazard: the existing `stage_cut` caches by
        FILE PATH (`parts/pNN.mp4`) and skips anything `good_video()` accepts.
        After a revision that changes segment 1's in/out points but keeps the
        segment count, `parts/p01.mp4` still exists and still probes fine, so
        the stale render would be silently reused. Keying on the content that
        actually determines the pixels - and invalidating on mismatch - is what
        makes cache reuse correct rather than merely fast.
        """
        return content_hash({
            "clip": self.clip_id, "start": round(self.start, 3),
            "end": round(self.end, 3), "caption": self.caption,
        })


class EditPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    segments: list[Segment]
    total_duration: float
    target_seconds: float
    canvas: tuple[int, int]
    strategy: str = ""
    opening_rationale: str = ""
    rejected: list[dict] = Field(default_factory=list)
    budget_method: str = "sqrt-weighted"
    constraints_applied: list[str] = Field(default_factory=list)

    # The join sequence the Transition agent chose, carried on the plan so it
    # reaches `compile_render_spec` -> `RenderSpec.xfade_list` -> AT_XFADES.
    # Empty means "the renderer's default", which is what every plan made
    # before the Transition agent existed will say.
    xfade_list: list[str] = Field(default_factory=list)

    @field_validator("xfade_list", mode="before")
    @classmethod
    def _only_supported_transitions(cls, v):
        """The last gate before a transition name can reach the renderer.

        `Transition.kind` is already a closed literal, so a sequence coming
        from that agent is safe by construction. This exists for every OTHER
        way an EditPlan can be built - a test fixture, a resumed job, a future
        caller - because a name ffmpeg does not know makes the render fail
        outright, and failing here with a dropped entry is far better than
        failing there with a dead build.

        `mode="before"` so this runs ahead of the list[str] type check and
        FILTERS rather than raising: one odd entry on a resumed job should
        cost that entry, not the build.
        """
        if not isinstance(v, list):
            return []
        allowed = set(get_args(TransitionKind))
        return [x for x in v if isinstance(x, str) and x in allowed]

    @field_validator("segments")
    @classmethod
    def _indices_contiguous(cls, v):
        if not v:
            raise ValueError("an edit plan must contain at least one segment")
        want = list(range(1, len(v) + 1))
        got = [s.index for s in v]
        if got != want:
            raise ValueError(f"segment indices must be 1..n in order, got {got}")
        return v

    @property
    def opening_clip(self) -> str:
        return self.segments[0].clip_id


# ------------------------------------------------------------ RenderSpec

class SegmentRenderSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    clip_id: str
    start: float
    end: float
    duration: float
    caption: str
    orientation: Literal["vertical", "horizontal"]
    cache_key: str


# The renderer's own audio vocabulary, read from the modules that
# implement it. `Mood` is already declared above and is asserted equal to
# extras.MOODS by the music parity test; these two complete the set.
MusicSource = Literal["generate", "upload", "none"]
KeepAudio = Literal["none", "mix", "only"]


class RenderSpec(BaseModel):
    """Everything ffmpeg needs, compiled - never authored by an LLM.

    The env_overrides dict is the existing AT_* bridge (adapter.env_for), made
    explicit. It was invisible coupling before; it is a checked contract now.
    """
    model_config = ConfigDict(frozen=True)

    job_id: str
    build_dir: str
    src_dir: str
    canvas: tuple[int, int]
    crf: int
    preset: str = "fast"
    fps: int = 30
    xfade_duration: float = 0.4
    xfade_list: list[str] = Field(default_factory=lambda: ["fade"])
    segments: list[SegmentRenderSpec]
    title: str = "ALPHA_TITANS"

    # ---- audio -------------------------------------------------------
    # Compiled from the Music artifact, never authored by a model. These
    # three are exactly the keys `extras.music_stage` and
    # `extras.master_stage` already read, so the typed Music decision
    # reaches ffmpeg through the interface those stages have always used
    # rather than through a second audio path.
    #
    # Before this existed the agent renderer passed music_source="none"
    # and keep_audio="none" as literals, so every agent-built video came
    # out as digital silence no matter what the Music agent decided.
    music_source: MusicSource = "none"
    mood: Mood | None = None
    keep_audio: KeepAudio = "none"

    @model_validator(mode="after")
    def _mood_required_when_generating(self) -> "RenderSpec":
        # generate_music() falls back to "beat" for an unknown mood, which
        # would silently ignore the Music agent. Refuse instead.
        if self.music_source == "generate" and self.mood is None:
            raise ValueError(
                "music_source='generate' needs a mood; it must come from "
                "the Music artifact, not from a default")
        return self

    def audio_settings(self) -> dict[str, str]:
        """The settings dict `music_stage`/`master_stage` already expect."""
        out = {"music_source": self.music_source,
               "keep_audio": self.keep_audio}
        if self.mood is not None:
            out["mood"] = self.mood
        return out

    def env_overrides(self) -> dict[str, str]:
        return {
            "AT_CANVAS": f"{self.canvas[0]}x{self.canvas[1]}",
            "AT_CRF": str(self.crf),
            "AT_PRESET": self.preset,
            "AT_XFADES": ",".join(self.xfade_list),
            "AT_XFADE_T": str(self.xfade_duration),
        }


# ---------------------------------------------------------- RenderResult

class SegmentRenderInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    path: str
    cache_hit: bool
    rendered: bool
    wall_s: float = 0.0


class RenderTechnical(BaseModel):
    model_config = ConfigDict(frozen=True)

    duration: float = 0.0
    frames: int | None = None
    frames_expected: int | None = None
    frames_ok: bool = False
    width: int | None = None
    height: int | None = None
    fps: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    size: int = 0
    lufs: float | None = None
    black_frames: bool = False
    decode_ok: bool = True
    decode_errors: str = ""


class RenderResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt: int
    # The per-attempt archive copy. `extras.master_stage` always writes to one
    # fixed filename, so without archiving, attempt 2 would destroy attempt 1 -
    # and the critic's "is this revision actually better?" comparison needs
    # both files to still exist.
    video_path: str
    # Where master_stage put it: the stable name the existing UI and the
    # /download endpoint expect, always the most recent attempt.
    canonical_path: str | None = None
    thumbnail_path: str | None = None
    technical: RenderTechnical
    segments: list[SegmentRenderInfo] = Field(default_factory=list)
    wall_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    # What the renderer was told to do about audio, carried forward from the
    # RenderSpec so the chain Music -> RenderSpec -> ffmpeg -> critic is
    # readable from the artifacts alone. The critic needs `music_source` in
    # particular: silence is a PASS when none was asked for and a FAILURE
    # when a bed was. Defaulted so artifacts written before this existed
    # still load, and they describe the silent path, which is what they were.
    music_source: MusicSource = "none"
    mood: Mood | None = None
    keep_audio: KeepAudio = "none"
    audio_track: str | None = None      # the wav music_stage actually made

    @property
    def wanted_audio(self) -> bool:
        """Was this render supposed to have an audible bed?"""
        return self.music_source != "none" or self.keep_audio != "none"

    @property
    def cache_hits(self) -> int:
        return sum(1 for s in self.segments if s.cache_hit)


# ---------------------------------------------------------- CriticResult

Verdict = Literal["PASS", "REVISE", "ESCALATE"]
Severity = Literal["blocking", "high", "medium", "low"]


class TechnicalCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    value: Any = None
    threshold: str = ""
    passed: bool
    severity: Severity = "high"
    routes_to: str = "rendering"


class CreativeCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: str
    score: float = Field(ge=0.0, le=100.0)
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity = "medium"
    routes_to: str = "editing_director"
    measured: bool = True          # False => the check could not be run


class CriticResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt: int
    verdict: Verdict
    overall_score: float = Field(ge=0.0, le=100.0)
    pass_threshold: float
    technical: list[TechnicalCheck] = Field(default_factory=list)
    creative: list[CreativeCheck] = Field(default_factory=list)
    compared_to_previous: dict | None = None
    unreviewed_creative: bool = False
    summary: str = ""

    @property
    def blocking_failures(self) -> list[TechnicalCheck]:
        return [t for t in self.technical
                if not t.passed and t.severity == "blocking"]


# ------------------------------------------------------- RevisionRequest

class RevisionRequest(BaseModel):
    """A routable instruction, not a complaint.

    `constraints` is machine-readable and is consumed as hard bounds by the
    target agent. "the pacing is off" is useless; {"opening_min_mean_ydif":
    5.7} is something an agent can actually satisfy or fail against.
    """
    model_config = ConfigDict(frozen=True)

    from_attempt: int
    target_agent: str
    issue: str
    severity: Severity
    constraints: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    previously_tried: list[dict] = Field(default_factory=list)

    def constraint_key(self) -> str:
        """Identity of this constraint set, for anti-oscillation.

        Re-issuing a constraint that has already been tried and failed means
        the loop is spinning; the orchestrator escalates instead.
        """
        return content_hash(self.constraints)


# ------------------------------------------------------------ Thumbnail

# The ONE image operation this pipeline has. `server/extras.py::master_stage`
# makes its thumbnail with
#
#     ffmpeg -ss <t> -i <final video> -frames:v 1 thumbnail.jpg
#
# - one frame of the delivered video, at the video's own resolution, as a
# JPEG. No scale, no crop, no text, no overlay, no compositing, and nothing in
# the project generates an image. So the only thing a thumbnail can decide is
# WHICH frame, and this literal has exactly one member. A parity test asserts
# master_stage still does exactly this.
ThumbnailOperation = Literal["frame_grab"]

# Where a candidate frame came from. Each is a reference to something already
# measured or validated upstream - never a time a model supplied.
#   vision_moment    an index into a clip's Vision `interesting_moments`
#   measured_window  the middle of a clip's measured best window
#   plan_segment     the middle of a segment the final EditPlan kept
ThumbnailSource = Literal["vision_moment", "measured_window", "plan_segment"]
ThumbnailType = Literal["reasoned", "degraded", "minimal"]


class ThumbnailCandidate(BaseModel):
    """One frame the delivered video actually contains, offered by INDEX.

    Built by the agent, never by the model: `clip_id` is a clip the plan
    kept, `moment_ref` and `window_ref` point into what Vision and the
    measurement pass already produced, and `segment_index` is the plan
    segment that shows it. A reference whose moment was trimmed out of the
    cut is not a candidate at all - it is not a frame of this video.

    `clip_at` and `frame_at` are DERIVED: the referenced moment's time in its
    clip, and where that lands in the delivered video using the transition
    offsets the renderer itself recorded. Neither is ever read from a reply.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    clip_id: str
    source: ThumbnailSource
    moment_ref: int | None = Field(default=None, ge=0)
    window_ref: int | None = Field(default=None, ge=0)
    segment_index: int = Field(ge=1)
    clip_at: float = Field(ge=0.0)      # derived: seconds into the clip
    frame_at: float = Field(ge=0.0)     # derived: seconds into the video
    what: str = ""                      # Vision's words for it, if any


class Thumbnail(BaseModel):
    """Which frame of the finished video represents it - and the file made.

    PACKAGING, NOT EDITING
    ----------------------
    This runs once, after the render/critic cycle has settled on the video
    that will be delivered, and changes nothing about that video. It chooses
    a frame of it - from candidates derived from Vision's moments, the
    measured windows and the final cut - and grabs that frame with the same
    single ffmpeg operation master_stage uses.

    NOTHING IS CLAIMED THAT DID NOT HAPPEN
    --------------------------------------
    `generated` is True only when the image exists on disk and ffprobe read
    real dimensions from it, and a validator makes the fields agree: no path
    without a measured image, no measured image without a path. A grab that
    failed is a thumbnail that was not made, and the artifact says so.

    WHAT CANNOT BE EXPRESSED
    ------------------------
    Text, crop, zoom, colour, arrows, faces, composites, generated imagery -
    the pipeline has none of them, and a request for any lands in
    `unsupported_requests` rather than being promised.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    thumbnail_type: ThumbnailType = "minimal"
    operation: ThumbnailOperation = "frame_grab"

    candidates: list[ThumbnailCandidate] = Field(default_factory=list)
    chosen: int | None = Field(default=None, ge=0)     # -> candidates[i]
    rationale: str = ""
    unrealisable: int = Field(default=0, ge=0)   # references not in the cut

    # what the pipeline can actually do, measured - never model output
    engine_operations: list[str] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)

    # what was actually produced, measured - never claimed
    video_path: str | None = None       # the delivered file it came from
    verdict: Verdict | None = None      # what the critic said of that file
    generated: bool = False
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    size: int = Field(default=0, ge=0)

    # provenance: what the choice was actually made from
    vision_available: bool = False
    strategy_available: bool = False
    story_available: bool = False

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("candidates")
    @classmethod
    def _indices_contiguous(cls, v):
        """Candidates are offered by index, so the indices ARE the contract:
        a gap or a repeat would let a reply's number mean two frames."""
        got = [c.index for c in v]
        if got != list(range(len(v))):
            raise ValueError(
                f"candidate indices must be 0..n-1 in order, got {got}")
        return v

    @model_validator(mode="after")
    def _claims_match_what_exists(self):
        if self.chosen is not None and self.chosen >= len(self.candidates):
            raise ValueError(f"chosen {self.chosen} is not one of the "
                             f"{len(self.candidates)} candidates")
        if self.generated:
            if not (self.image_path and self.width and self.height
                    and self.size > 0 and self.chosen is not None):
                raise ValueError("a generated thumbnail needs a chosen frame "
                                 "and a measured image on disk")
        elif self.image_path or self.width or self.height or self.size:
            raise ValueError("a thumbnail that was not generated cannot "
                             "carry an image path or measurements")
        return self

    @property
    def choice(self) -> "ThumbnailCandidate | None":
        return None if self.chosen is None else self.candidates[self.chosen]


# ---------------------------------------------------------------- Title

# There is no publishing integration in this project - nothing uploads, and
# nothing writes container metadata. A title reaches exactly two places:
#
#   * master_stage names the delivered FILE from JobSettings.title, sanitised
#     to [A-Za-z0-9._-]. That happens inside the render, before packaging, so
#     it is never changed after the fact - the file keeps its name.
#   * the app shows `settings.title` on the job card and above the player,
#     and untitled agent builds show the JobSettings placeholder.
#
# So the title is published by projecting it into that one display slot, and
# only when the operator did not name the video: operator settings win.
#
# 100 characters is YouTube's own title limit - the platform this system
# makes videos for - not something this project enforces anywhere else.
TITLE_MAX_CHARS = 100

# Where the chosen words came from.
#   operator   the operator named the video; nothing else is considered
#   model      written by the model, every content word found upstream
#   strategy   ContentStrategy's central topic, selected verbatim
#   story      the Story's logline, selected verbatim
#   domain     the operator's own description of the footage, verbatim
TitleSource = Literal["operator", "model", "strategy", "story", "domain"]
TitleType = Literal["operator", "reasoned", "degraded", "minimal"]


class Title(BaseModel):
    """The title the delivered video is published under - or none at all.

    PACKAGING, NOT EDITING
    ----------------------
    Runs once, after the render/critic cycle has settled on the video that
    will be delivered, so it describes the cut that actually exists: story
    beats on clips the cut dropped cannot lend a single word to it.

    NO WORD IS UNSUPPORTED
    ----------------------
    Every content word must appear in the validated material about THIS
    video - the strategy, the story beats that made the cut, the operator's
    domain. Research about the wider subject is context the model may read,
    but a word found ONLY there describes the world, not the footage, and is
    rejected. Superlatives and hype are rejected outright, and a number must
    be one the material contains. `grounded_terms` records where each content
    word was found, so the check can be audited rather than trusted.

    An empty `text` is a legitimate outcome: nothing supportable to say, so
    the placeholder stands rather than an invented title replacing it.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str = ""
    title_type: TitleType = "minimal"

    text: str = ""                      # "" = no supportable title
    source: TitleSource | None = None
    grounded_terms: list[str] = Field(default_factory=list)  # "word<-where"
    alternatives: list[str] = Field(default_factory=list)    # also passed
    rejected: list[str] = Field(default_factory=list)        # and why
    max_chars: int = Field(default=TITLE_MAX_CHARS, ge=1)

    operator_title: str | None = None   # what the operator set, if anything
    # will the app show this instead of the placeholder?
    published: bool = False

    unsupported_requests: list[str] = Field(default_factory=list)

    # provenance: what the title was actually built from
    strategy_available: bool = False
    story_available: bool = False
    research_available: bool = False

    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_signal: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    live: bool = False                 # True only for a real model call
    model: str | None = None

    @field_validator("text", "alternatives")
    @classmethod
    def _one_printable_line(cls, v):
        for t in ([v] if isinstance(v, str) else v):
            if any(not ch.isprintable() for ch in t) or t != t.strip():
                raise ValueError("a title is one printable, trimmed line")
        return v

    @model_validator(mode="after")
    def _claims_match(self):
        for t in [self.text, *self.alternatives]:
            if len(t) > self.max_chars:
                raise ValueError(f"title is {len(t)} characters; the limit "
                                 f"is {self.max_chars}")
        if bool(self.text) != (self.source is not None):
            raise ValueError("a title has a source exactly when it has text")
        if self.source == "operator" and self.text != self.operator_title:
            raise ValueError("an operator title is used verbatim")
        if self.published and (not self.text or self.source == "operator"):
            raise ValueError("only a title this agent supplied replaces the "
                             "placeholder - an operator title already shows")
        return self


# ----------------------------------------------------------- Publishing

# NOTHING IN THIS PROJECT CAN PUBLISH. The only YouTube code is a read-only
# search (`youtube/v3/search`, API key) feeding Research; uploading needs
# OAuth consent and `videos.insert`, and neither exists - no client library,
# no credentials, no endpoint. So each of these literals has exactly one
# member, and a parity test fails the moment an upload path appears:
#
#   published                   False - always; nothing here was uploaded
#   provider                    "none"
#   requires_operator_approval  True - publishing is an outward, public,
#                               hard-to-undo act, so it is the operator's
#   recommended_visibility      "private" - a first upload should be seen on
#                               the platform by its owner before anyone else
#
# There is deliberately NO field for an upload id, a URL, a view count or a
# publish time: `extra="forbid"` makes inventing one a validation error.
PublishStatus = Literal["ready", "needs_review", "blocked"]
PublishProvider = Literal["none"]
CheckSource = Literal["measured", "youtube_limit", "policy"]


class PublishCheck(BaseModel):
    """One readiness check, run locally against the files that exist.

    `youtube_limit` checks compare against YouTube's documented limits; they
    are run HERE, on disk - nothing is sent to YouTube to find out.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    passed: bool
    value: str = ""
    limit: str = ""
    source: CheckSource = "measured"
    blocking: bool = False             # a failure here means nothing to ship


class PublishAsset(BaseModel):
    """A deliverable that exists on disk, as measured at packaging time."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["video", "thumbnail"]
    path: str
    file_name: str
    size: int = Field(ge=0)
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    duration: float | None = None


class Publishing(BaseModel):
    """The publishing PLAN for the delivered video - never a publication.

    Packaging's last step. It gathers what was actually produced - the video
    the app delivers, the thumbnail and the title it shows - measures each,
    checks them against YouTube's documented limits locally, and states
    plainly what a person must do to publish, and that nothing has been.

    `status` is DERIVED from the checks by a validator, so it cannot say
    "ready" over a failed check: blocked when a blocking check failed,
    needs_review when any other did, ready only when none did. "ready" means
    ready for the OPERATOR to upload by hand.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: PublishStatus = "blocked"
    published: Literal[False] = False
    provider: PublishProvider = "none"
    requires_operator_approval: Literal[True] = True
    recommended_visibility: Literal["private"] = "private"

    video: PublishAsset | None = None
    thumbnail: PublishAsset | None = None
    thumbnail_from: Literal["thumbnail_agent", "master_stage"] | None = None
    title: str = ""
    title_from: Literal["title_agent", "operator"] | None = None
    verdict: Verdict | None = None      # what the critic said of the video

    checks: list[PublishCheck] = Field(default_factory=list)
    # what the project can do on a platform, measured - never model output
    platform_capabilities: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_follows_the_checks(self):
        failed = [c for c in self.checks if not c.passed]
        want = ("blocked" if any(c.blocking for c in failed) or not
                self.checks else "needs_review" if failed else "ready")
        if self.status != want:
            raise ValueError(f"status {self.status!r} contradicts the "
                             f"checks, which give {want!r}")
        if self.status == "ready" and not (self.video and self.thumbnail
                                           and self.title):
            raise ValueError("nothing is ready without a video, a "
                             "thumbnail and a title")
        if bool(self.title) != (self.title_from is not None):
            raise ValueError("a title has a source exactly when it exists")
        if bool(self.thumbnail) != (self.thumbnail_from is not None):
            raise ValueError("a thumbnail has a source exactly when it "
                             "exists")
        return self

    @property
    def blockers(self) -> list[PublishCheck]:
        return [c for c in self.checks if not c.passed and c.blocking]

    @property
    def warnings(self) -> list[PublishCheck]:
        return [c for c in self.checks if not c.passed and not c.blocking]


# --------------------------------------------------- PerformanceAnalysis

# NOTHING IN THIS PROJECT CAN READ HOW A VIDEO PERFORMED. Performance needs
# three things - a published video, a record of WHICH video it is, and access
# to its analytics - and none exists: nothing publishes (`Publishing.published`
# is the literal False), nothing records an uploaded video's id or URL, and
# there is no YouTube Analytics, statistics or channel-access code anywhere.
# So, exactly as for Publishing, each literal below has one member and parity
# tests fail the moment any of that changes:
#
#   status     "unavailable" - "insufficient_data" (published, measurable, too
#              little yet) is deliberately ABSENT: it needs a publication and
#              a provider, and adding it is part of adding them
#   provider   "none"
#   published  False
#
# There is deliberately NO field for a view count, a click-through rate,
# retention, likes, revenue, a URL or a video id: `extra="forbid"` makes
# inventing one a validation error. What it carries instead is WHY there is
# no data, WHICH metrics are therefore unmeasured and what each would need,
# and which pre-publication signals were set aside rather than passed off as
# performance.
PerformanceStatus = Literal["unavailable"]
AnalyticsProvider = Literal["none"]
PerformanceGapReason = Literal["not_published", "no_publishing_plan",
                               "no_video_reference", "no_analytics_provider"]
PerformanceMetric = Literal[
    "views", "likes", "comments", "impressions", "click_through_rate",
    "average_view_duration", "audience_retention", "watch_time",
    "subscribers_gained", "revenue"]


class PerformanceGap(BaseModel):
    """One measured reason there is no performance data."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: PerformanceGapReason
    evidence: str                       # what was read that shows it


class UnmeasuredMetric(BaseModel):
    """A metric this report does NOT have - named so its absence is stated
    rather than left to be filled in. There is no value field."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: PerformanceMetric
    needs: str                          # what would have to exist to read it


class ExcludedSignal(BaseModel):
    """A pre-publication signal that was present and deliberately NOT used
    as performance - named, never quoted, so no number of it leaks in."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: str
    artifact: ArtifactKind
    why: str


class PerformanceAnalysis(BaseModel):
    """How the delivered video performed - which, today, nobody can know.

    The last step after packaging. It reads what the project can actually
    see after publishing - the publishing record, and whatever analytics
    access the code has - and reports that performance is unavailable, with
    the measured reasons. Every metric is listed as unmeasured; every
    pre-publication signal that might be mistaken for performance (the
    critic's score, a "ready" plan, Research's evidence) is listed as
    excluded. Nothing is estimated.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: PerformanceStatus = "unavailable"
    provider: AnalyticsProvider = "none"
    published: Literal[False] = False
    # the publishing PLAN's status, when there is one - context for what the
    # operator does next, never a measure of reception
    publishing_status: PublishStatus | None = None

    gaps: list[PerformanceGap] = Field(default_factory=list)
    unmeasured: list[UnmeasuredMetric] = Field(default_factory=list)
    excluded_signals: list[ExcludedSignal] = Field(default_factory=list)
    # what the project can do on a platform, measured - never model output
    platform_capabilities: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unavailable_says_why(self):
        reasons = [g.reason for g in self.gaps]
        if not reasons:
            raise ValueError("an unavailable report must say why")
        if len(set(reasons)) != len(reasons):
            raise ValueError("each reason is stated once")
        if ("not_published" in reasons) == ("no_publishing_plan" in reasons):
            raise ValueError("exactly one of not_published and "
                             "no_publishing_plan explains the publication")
        if ("no_publishing_plan" in reasons) != (self.publishing_status
                                                 is None):
            raise ValueError("a publishing status exists exactly when a "
                             "publishing plan does")
        metrics = [m.metric for m in self.unmeasured]
        if sorted(metrics) != sorted(get_args(PerformanceMetric)):
            raise ValueError("with no performance data, every metric is "
                             "unmeasured - each listed exactly once")
        return self

    @property
    def reasons(self) -> list[str]:
        return [g.reason for g in self.gaps]


# --------------------------------------------------------------- Learning

# NOTHING CAN BE LEARNED YET. Learning joins OUTCOMES (measured performance of
# published videos) with the production DECISIONS behind them, and only
# proposes a change once the evidence clears every gate the architecture
# sets. There are no outcomes: `PerformanceAnalysis.status` has one member,
# "unavailable". So, as for Publishing and PerformanceAnalysis, the literals
# below have one member each, and parity tests fail the moment that changes:
#
#   status                 "insufficient_data"
#   learned                False
#   configuration_changed  False - and a change, when one is ever proposed,
#                          is a CONFIGURATION change awaiting a human's
#                          approval; nothing here can reach code at all
#
# There is deliberately NO field for an insight, a correlation, an effect, a
# causal claim, an experiment or a proposal: `extra="forbid"` makes inventing
# one a validation error. What it carries instead is the gate - every check,
# its threshold, where that threshold comes from, and what was observed -
# and what each input was, so a pre-publication signal can never be taken
# for an outcome.
LearningStatus = Literal["insufficient_data"]
LearningGateName = Literal[
    "sufficient_history", "valid_nonzero_data", "cohort_size", "time_window",
    "holdout_validation", "practical_effect", "human_approval"]
GateState = Literal["passed", "failed", "not_evaluated"]
ThresholdFrom = Literal["architecture", "policy"]
LearningInputRole = Literal["outcome", "decision", "pre_publication"]

# what each kind can ever be to the learning loop - an outcome is measured
# AFTER publication; everything made before it is a decision or a check
_OUTCOME_KINDS = ("PerformanceAnalysis",)
_DECISION_KINDS = ("ContentStrategy", "Story", "EditPlan", "Transition",
                   "Caption", "Music", "Animation", "Voiceover")


class LearningGate(BaseModel):
    """One check the evidence must clear before anything may be proposed."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: LearningGateName
    state: GateState
    required: str                       # the bar, in words, with its number
    observed: str                       # what was measured against it
    threshold_from: ThresholdFrom = "policy"


class LearningInput(BaseModel):
    """An artifact the agent read, and what it can be to learning."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ArtifactKind
    role: LearningInputRole
    summary: str = ""                   # categorical facts only, no figures
    why: str = ""

    @model_validator(mode="after")
    def _role_follows_the_kind(self):
        want = ("outcome" if self.kind in _OUTCOME_KINDS else
                "decision" if self.kind in _DECISION_KINDS else
                "pre_publication")
        if self.role != want:
            raise ValueError(f"{self.kind} is a {want}, never a {self.role}")
        return self


class LearningHistory(BaseModel):
    """What the shared store holds, across every job - measured, not
    assumed. A usable sample is a published video with measured,
    non-zero performance."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    jobs_in_store: int = Field(ge=0)
    performance_reports: int = Field(ge=0)
    unavailable_reports: int = Field(ge=0)
    unreadable_reports: int = Field(ge=0)
    usable_samples: int = Field(ge=0)
    samples_needed: int = Field(ge=0)


class ConfigTarget(BaseModel):
    """A piece of CONFIGURATION a future proposal could name - read from the
    code that consumes it. Code is never a target."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    where: str
    current: str


class Learning(BaseModel):
    """What the system has learned from how its videos performed - which,
    until there is performance data, is nothing.

    The last step. It reads this job's performance report, its production
    decisions and its final results, and the whole shared history; runs the
    architecture's gate; and returns `insufficient_data` with every gate's
    result and the minimum-sample gap. Nothing is learned, nothing is
    proposed, configuration is unchanged - and that is the correct answer,
    not a failure.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: LearningStatus = "insufficient_data"
    learned: Literal[False] = False
    configuration_changed: Literal[False] = False
    performance_status: PerformanceStatus | None = None

    history: LearningHistory
    gates: list[LearningGate] = Field(default_factory=list)
    inputs: list[LearningInput] = Field(default_factory=list)
    # what a proposal could ever name, measured - never model output
    configurable: list[ConfigTarget] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _the_gate_holds(self):
        names = [g.gate for g in self.gates]
        if names != list(get_args(LearningGateName)):
            raise ValueError("every gate is reported exactly once, in order")
        if all(g.state == "passed" for g in self.gates):
            raise ValueError("insufficient data means a gate did not pass")
        open_ = False
        for g in self.gates:
            if open_ and g.state == "passed":
                raise ValueError(f"{g.gate} cannot pass after an earlier gate "
                                 f"did not")
            open_ = open_ or g.state != "passed"
        if next(g for g in self.gates
                if g.gate == "human_approval").state == "passed":
            raise ValueError("there is no proposal, so nothing was approved")
        if self.history.usable_samples > self.history.performance_reports:
            raise ValueError("a usable sample is a performance report")
        return self

    @property
    def failed_gates(self) -> list[str]:
        return [g.gate for g in self.gates if g.state == "failed"]

    @property
    def unevaluated_gates(self) -> list[str]:
        return [g.gate for g in self.gates if g.state == "not_evaluated"]


__all__ = [
    "SCHEMA_VERSION", "new_id", "content_hash",
    "Decision", "DecidedBy", "Cost", "Artifact", "ArtifactKind",
    "JobSettings", "JobState",
    "ClipTechnical", "ClipMotion", "ClipUsability", "ClipAnalysis",
    "ClipAnalysisSet",
    "CandidateWindow", "InterestingMoment", "VisionClip", "VisionAnalysis",
    "EvidenceItem", "TrendItem", "ContentOpportunity", "ResearchReport",
    "KeyPoint", "ClipPriority", "Hook", "ContentStrategy",
    "Pacing", "ClipRole", "Support", "StrategyType",
    "BeatRole", "StoryType", "StoryBeat", "Story",
    "Tone", "VoiceoverType", "NarrationSegment", "Voiceover",
    "Mood", "Energy", "TempoFeel", "MusicType", "MusicSection", "Music",
    "Treatment", "Emphasis", "CaptionIntent", "AnimationType",
    "AnimationDirection", "Animation",
    "TransitionKind", "TransitionType", "TransitionChoice", "Transition",
    "CaptionPosition", "CaptionStyle", "CaptionSource", "CaptionType",
    "CaptionLine", "Caption",
    "Segment", "EditPlan",
    "SegmentRenderSpec", "RenderSpec",
    "SegmentRenderInfo", "RenderTechnical", "RenderResult",
    "TechnicalCheck", "CreativeCheck", "CriticResult", "Verdict", "Severity",
    "RevisionRequest",
    "ThumbnailOperation", "ThumbnailSource", "ThumbnailType",
    "ThumbnailCandidate", "Thumbnail",
    "TITLE_MAX_CHARS", "TitleSource", "TitleType", "Title",
    "PublishStatus", "PublishProvider", "CheckSource", "PublishCheck",
    "PublishAsset", "Publishing",
    "PerformanceStatus", "AnalyticsProvider", "PerformanceGapReason",
    "PerformanceMetric", "PerformanceGap", "UnmeasuredMetric",
    "ExcludedSignal", "PerformanceAnalysis",
    "LearningStatus", "LearningGateName", "GateState", "ThresholdFrom",
    "LearningInputRole", "LearningGate", "LearningInput", "LearningHistory",
    "ConfigTarget", "Learning",
]
