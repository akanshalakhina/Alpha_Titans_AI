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
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    "ContentStrategy", "Story", "Voiceover", "Music",
    "EditPlan", "RenderSpec", "RenderResult", "CriticResult",
    "RevisionRequest",
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
    "Segment", "EditPlan",
    "SegmentRenderSpec", "RenderSpec",
    "SegmentRenderInfo", "RenderTechnical", "RenderResult",
    "TechnicalCheck", "CreativeCheck", "CriticResult", "Verdict", "Severity",
    "RevisionRequest",
]
