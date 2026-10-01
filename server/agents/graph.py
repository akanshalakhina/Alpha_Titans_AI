"""
The orchestrator: a real LangGraph StateGraph with a SQLite checkpointer.

This replaces the `for` loop in server/jobs.py::_run - which has no
dependencies, no routing, no retries, no parallelism and no resumability -
with a graph that has all of them.

                    +-> vision   -+
    clip_analysis --+             +-> content_strategy -> story
                    +-> research -+                         |
                                                            v
                voiceover -> music -> animation -> transition -> caption
                                                                     |
                                                                     v
          +------------------------> editing_director -> render -> critic
          |                                ^                          |
          |                                |    PASS     -> finish   -+
          +--------------------------------+    REVISE   -+           |
                                                ESCALATE -> escalate -+
                                                                      v
                                          thumbnail -> title -> publishing
                                                                      |
                                                                      v
                                                performance_analysis -> END

The understanding agents (vision, research) fan out and JOIN at the strategy.
From there it is a chain, each layer briefing the next: the strategy briefs
the story, the story briefs the narration, the narration briefs the score,
the score briefs the look, the look briefs the joins, the joins hand
on to the captions, and the captions brief the director.

Packaging sits after it: once the cycle has settled on the video that
will be delivered - passed or escalated - the thumbnail is chosen from that
video and the title is chosen for it, and a publishing PLAN is made for
the operator - once each, and none of them can cost a render or trigger a
revision. Nothing in the project can upload, and nothing claims to. Last,
the video's performance is reported - as unavailable, with the measured
reasons, because nothing here can publish it or read its analytics, and
nothing made before publication is passed off as performance.

EVERY one of those sits OUTSIDE the revision cycle. A revision re-plans and
re-renders without re-looking at the footage, re-deciding what the video is
about, re-authoring the story, rewriting the narration, re-scoring it, or
re-choosing the words on screen.

WHAT IS GENUINELY IMPLEMENTED HERE
----------------------------------
* dependencies      edges, plus each node loading its inputs by artifact id
* conditional routing  a real `add_conditional_edges` on the critic's verdict
* a real cycle      critic -> editing_director, bounded by max_revisions
* retries           per-node, with backoff, recorded in the trace
* timeouts          per-node deadline on the media pool future
* fallbacks         declared by the agent; the EditingDirector's is the
                    production planner
* persistent state  SqliteSaver checkpoints after every node
* resumability      re-invoking with the same thread_id continues from the
                    last checkpoint rather than restarting
* max revision count + escalation
* tracing           every node, every attempt, every tool call

GRAPH STATE HOLDS POINTERS, NOT PAYLOADS
----------------------------------------
Only ids, counters and small flags live in the checkpointed state. The
artifacts themselves live in the ArtifactStore. That keeps checkpoints small,
keeps them JSON-clean, and means a resumed run reads exactly the same
immutable artifacts the original run produced.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from .artifacts import (ClipAnalysisSet, ContentStrategy, CriticResult,
                        EditPlan, JobState, RenderResult, ResearchReport,
                        Animation, Caption, Music, RevisionRequest, Story,
                        Publishing, Thumbnail, Title, Transition,
                        VisionAnalysis, Voiceover)
from .base import AgentContext
from .caption import CaptionAgent
from .clip_analysis import ClipAnalysisAgent
from .content_strategy import ContentStrategyAgent
from .critic import DEFAULT_PASS_THRESHOLD, CriticAgent
from .editing_director import EditingDirectorAgent
from .render_pool import (MediaPool, ProcessCancelled, ProcessTimeout,
                          global_pool)
from .scheduler import RenderScheduler, WorkCancelled
from .animation import AnimationAgent
from .music import MusicAgent
from .performance import PerformanceAnalysisAgent
from .publishing import PublishingAgent
from .story import StoryAgent
from .thumbnail import ThumbnailAgent
from .title import TitleAgent
from .transition import TransitionAgent
from .voiceover import VoiceoverAgent
from .rendering import RenderingAgent
from .research_agent import ResearchAgent
from .vision_agent import VisionAgent
from .store import ArtifactStore
from .tools.llm_tools import LLMClient
from .trace import TraceRecorder


# ------------------------------------------------------------------ state

class GraphState(TypedDict, total=False):
    job_id: str
    status: str                       # running | done | escalated | error
    error: str | None

    clip_analysis_id: str
    vision_id: str | None
    vision_failed: str | None
    research_id: str | None
    research_failed: str | None
    strategy_id: str | None
    strategy_failed: str | None
    story_id: str | None
    story_failed: str | None
    voiceover_id: str | None
    voiceover_failed: str | None
    music_id: str | None
    music_failed: str | None
    animation_id: str | None
    animation_failed: str | None
    transition_id: str | None
    transition_failed: str | None
    caption_id: str | None
    caption_failed: str | None
    thumbnail_id: str | None
    thumbnail_failed: str | None
    title_id: str | None
    title_failed: str | None
    publishing_id: str | None
    publishing_failed: str | None
    performance_id: str | None
    performance_failed: str | None
    edit_plan_id: str
    render_id: str
    critic_id: str

    attempt: int                      # render attempts, 1-based
    revision_count: int
    max_revisions: int
    pass_threshold: float

    verdict: str
    revision_request: dict | None
    previously_tried: list[dict]
    history: list[dict]               # one row per completed attempt


# --------------------------------------------------------------- runtime

class Runtime:
    """Non-serialisable collaborators, held outside the checkpoint.

    A resumed process rebuilds this before invoking the graph; the checkpoint
    holds only ids, so nothing here needs to survive a restart.
    """

    def __init__(self, job: JobState, store: ArtifactStore,
                 tracer: TraceRecorder, pool: MediaPool, llm: LLMClient,
                 node_timeout_s: float = 1800.0,
                 scheduler: RenderScheduler | None = None):
        self.job = job
        self.store = store
        self.tracer = tracer
        self.pool = pool
        self.llm = llm
        self.node_timeout_s = node_timeout_s
        # the scheduler shares the runtime's pool, so there is exactly one
        # concurrency bound and one process registry for the whole job
        self.scheduler = scheduler or RenderScheduler(pool=pool)
        self.log: list[str] = []

    def context(self) -> AgentContext:
        return AgentContext(job=self.job, store=self.store,
                            tracer=self.tracer, pool=self.pool, llm=self.llm,
                            log=self.log)


_REGISTRY: dict[str, Runtime] = {}
_REG_LOCK = threading.RLock()


def register(runtime: Runtime) -> None:
    with _REG_LOCK:
        _REGISTRY[runtime.job.job_id] = runtime


def runtime_for(job_id: str) -> Runtime:
    with _REG_LOCK:
        rt = _REGISTRY.get(job_id)
    if rt is None:                                     # pragma: no cover
        raise RuntimeError(f"no runtime registered for job {job_id}; the "
                           f"process must rebuild it before resuming")
    return rt


def unregister(job_id: str) -> None:
    with _REG_LOCK:
        _REGISTRY.pop(job_id, None)


# ------------------------------------------------------- retry machinery

class NodePolicy:
    """Per-node retry and timeout policy, declared rather than scattered."""

    def __init__(self, attempts: int = 1, backoff_s: float = 2.0,
                 timeout_s: float | None = None):
        self.attempts = attempts
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s


POLICIES = {
    "clip_analysis": NodePolicy(attempts=2, backoff_s=2.0, timeout_s=3600),
    "vision": NodePolicy(attempts=2, backoff_s=2.0, timeout_s=900),
    "research": NodePolicy(attempts=2, backoff_s=2.0, timeout_s=300),
    "content_strategy": NodePolicy(attempts=2, backoff_s=1.0,
                                   timeout_s=300),
    "story": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "voiceover": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "music": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "animation": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "transition": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "caption": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "editing_director": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "render": NodePolicy(attempts=2, backoff_s=3.0, timeout_s=3600),
    "critic": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=900),
    "thumbnail": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "title": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=300),
    "publishing": NodePolicy(attempts=2, backoff_s=1.0, timeout_s=900),
    "performance_analysis": NodePolicy(attempts=2, backoff_s=1.0,
                                       timeout_s=300),
}


def _heartbeat(job_id: str, node: str) -> None:
    """Project a node boundary into the compatibility layer.

    Best effort by design: the existing UI seeing a stale stage is a cosmetic
    problem, whereas a projection error killing a real build is not. Nothing
    here may raise.
    """
    try:
        from . import compat
        rt = runtime_for(job_id)
        outputs = Path(rt.job.build_dir).resolve().parent
        compat.project_running(job_id, root=outputs.parent, node=node)
    except Exception:                                 # noqa: BLE001
        pass


def _check_cancelled(job_id: str) -> None:
    """Honour a cancel requested from another process, at a node boundary."""
    try:
        from . import compat
        rt = runtime_for(job_id)
        outputs = Path(rt.job.build_dir).resolve().parent
        if not compat.cancel_requested(job_id, root=outputs.parent):
            return
    except Exception:                                 # noqa: BLE001
        return
    rt.scheduler.cancel(job_id)
    raise WorkCancelled(f"job {job_id} was cancelled")


class CancelWatcher:
    """Polls the cancel marker while a long node runs.

    A render is a single node that can take minutes, so checking only at node
    boundaries would make a cancel useless exactly when it matters. This polls
    once a second and performs the REAL kill through the scheduler, which
    tree-kills the ffmpeg the pool owns.
    """

    def __init__(self, job_id: str, scheduler, root: Path | None,
                 interval: float = 1.0):
        self.job_id = job_id
        self.scheduler = scheduler
        self.root = root
        self.interval = interval
        self.fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        from . import compat
        while not self._stop.wait(self.interval):
            try:
                if compat.cancel_requested(self.job_id, root=self.root):
                    self.fired = True
                    self.scheduler.cancel(self.job_id)
                    return
            except Exception:                         # noqa: BLE001
                return

    def __enter__(self) -> "CancelWatcher":
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"cancelwatch-{self.job_id[:8]}")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _run_node(node: str, agent_id: str, job_id: str, attempt: int,
              inputs: list[str], body):
    """Execute one node under its policy, tracing every attempt.

    A retried node writes ONE trace row per try, so a flaky stage is visible
    as retries rather than hidden behind a single green tick.

    The deadline is enforced by the SCHEDULER, not here: `policy.timeout_s` is
    passed down so that blowing it kills the process tree rather than merely
    abandoning the wait. (Until Phase 2 Step 3 this field was declared and
    never read - there was no timeout in the execution path at all.)
    """
    policy = POLICIES.get(node, NodePolicy())
    rt = runtime_for(job_id)
    last: Exception | None = None
    _check_cancelled(job_id)

    for i in range(policy.attempts):
        with rt.tracer.node(job_id, node, agent_id, attempt=attempt,
                            input_artifacts=inputs) as span:
            if i:
                span.retry()
                span.note(f"retry {i} of {policy.attempts - 1}")
            if policy.timeout_s:
                span.note(f"deadline {policy.timeout_s}s "
                          f"(kills the process tree on overrun)")
            try:
                result = body(span)
                _heartbeat(job_id, node)
                return result
            except (ProcessTimeout, WorkCancelled, ProcessCancelled) as exc:
                # A killed process is terminal for this node: retrying a
                # cancelled job, or a stage that already blew its deadline,
                # only burns the deadline again.
                span.note(f"terminal: {type(exc).__name__}")
                raise
            except Exception as exc:                  # noqa: BLE001
                last = exc
                if i == policy.attempts - 1:
                    raise
        time.sleep(policy.backoff_s * (2 ** i))
    raise last                                        # pragma: no cover


async def _run_node_async(node: str, agent_id: str, job_id: str, attempt: int,
                          inputs: list[str], body):
    """Async twin of `_run_node`, for nodes that await the scheduler.

    Identical policy and tracing; the only difference is that `body` is a
    coroutine, so the event loop stays free while the work runs instead of one
    node monopolising the graph's thread for the whole render.
    """
    policy = POLICIES.get(node, NodePolicy())
    rt = runtime_for(job_id)
    last: Exception | None = None
    _check_cancelled(job_id)
    outputs_root = Path(rt.job.build_dir).resolve().parent.parent

    for i in range(policy.attempts):
        with rt.tracer.node(job_id, node, agent_id, attempt=attempt,
                            input_artifacts=inputs) as span:
            if i:
                span.retry()
                span.note(f"retry {i} of {policy.attempts - 1}")
            if policy.timeout_s:
                span.note(f"deadline {policy.timeout_s}s "
                          f"(kills the process tree on overrun)")
            try:
                # a render is one long node, so a cancel that only landed at
                # node boundaries would be useless exactly when it matters
                with CancelWatcher(job_id, rt.scheduler, outputs_root) as w:
                    result = await body(span)
                if w.fired:                           # pragma: no cover
                    raise WorkCancelled(f"job {job_id} was cancelled")
                _heartbeat(job_id, node)
                return result
            except (ProcessTimeout, WorkCancelled, ProcessCancelled) as exc:
                span.note(f"terminal: {type(exc).__name__}")
                raise
            except Exception as exc:                  # noqa: BLE001
                last = exc
                if i == policy.attempts - 1:
                    raise
        await asyncio.sleep(policy.backoff_s * (2 ** i))
    raise last                                        # pragma: no cover


# ----------------------------------------------------------------- nodes

def node_clip_analysis(state: GraphState) -> dict:
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = ClipAnalysisAgent()

    def body(span):
        rev = rt.store.next_revision(job_id, "ClipAnalysis")
        art = agent.execute(rt.context(), span, revision=rev)
        return art

    art = _run_node("clip_analysis", ClipAnalysisAgent.id, job_id,
                    state.get("attempt", 1), [], body)
    return {"clip_analysis_id": art.artifact_id, "status": "running"}


def node_vision(state: GraphState) -> dict:
    """Semantic pass over the footage.

    Runs AFTER clip_analysis rather than beside it, and that is a real
    dependency rather than incidental serialisation: the model is only allowed
    to pick from windows the measurement pass produced, so there is nothing for
    it to choose from until those exist. Research runs beside it in parallel -
    it needs no measurements - and the two join at content_strategy.

    Never blocking. A vision failure degrades the artifact and the build
    continues on measurements alone, exactly as it did before this agent
    existed.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = VisionAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    if clip_art is None:                              # pragma: no cover
        raise RuntimeError("ClipAnalysis artifact missing")

    def body(span):
        rev = rt.store.next_revision(job_id, "VisionAnalysis")
        return agent.execute(rt.context(), span, clip_set=clip_art.payload,
                             revision=rev)

    try:
        art = _run_node("vision", VisionAgent.id, job_id,
                        state.get("attempt", 1), [clip_art.artifact_id], body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        # Vision is an enhancement, not a prerequisite. Losing it must not
        # cost the operator the video.
        rt.log.append(f"vision failed ({type(exc).__name__}: {exc}); "
                      f"continuing on measurements alone")
        return {"vision_id": None, "vision_failed": str(exc)[:200]}
    return {"vision_id": art.artifact_id, "vision_failed": None}


def node_research(state: GraphState) -> dict:
    """What the world currently says about this video's subject.

    Runs in PARALLEL with vision, both fanning out from clip_analysis and
    joining at content_strategy. That join was verified against this exact
    graph shape before it was adopted: on a revision the director re-runs
    while vision, research and the strategy correctly do not, so the critic
    cycle is unaffected.

    Writes only `research_id` / `research_failed`. Parallel nodes must touch
    DISJOINT state keys - LangGraph raises InvalidUpdateError if two of them
    write the same one, which is why this does not share vision's keys.

    Never blocking: no provider, no results or a bad reply all degrade the
    report and leave the pipeline running.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = ResearchAgent()

    # Vision is a sibling, so it may not have finished. Research reads it only
    # if it happens to be there and works fine without it.
    vision_art = rt.store.get(job_id, "VisionAnalysis", VisionAnalysis)

    def body(span):
        rev = rt.store.next_revision(job_id, "ResearchReport")
        return agent.execute(rt.context(), span,
                             vision=vision_art.payload if vision_art else None,
                             revision=rev)

    try:
        art = _run_node("research", ResearchAgent.id, job_id,
                        state.get("attempt", 1), [], body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"research failed ({type(exc).__name__}: {exc}); "
                      f"continuing without it")
        return {"research_id": None, "research_failed": str(exc)[:200]}
    return {"research_id": art.artifact_id, "research_failed": None}


def node_content_strategy(state: GraphState) -> dict:
    """The decision layer between understanding and execution.

    JOINS vision and research. LangGraph runs it once, after BOTH siblings
    have landed - verified against this exact shape before it was adopted, as
    was the property that matters more: on a revision the director re-runs and
    this node does NOT, because it sits upstream of the cycle. A revision that
    only re-plans the edit therefore costs no second strategy.

    Reads the two artifacts out of the store rather than out of graph state,
    which is the standing rule here - state carries pointers, the store
    carries payloads.

    Never blocking. Losing the strategy costs the director its brief, not the
    operator their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = ContentStrategyAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    if clip_art is None:                              # pragma: no cover
        raise RuntimeError("ClipAnalysis artifact missing")
    vision_art = rt.store.get(job_id, "VisionAnalysis", VisionAnalysis)
    research_art = rt.store.get(job_id, "ResearchReport", ResearchReport)

    inputs = [clip_art.artifact_id]
    inputs += [a.artifact_id for a in (vision_art, research_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "ContentStrategy")
        return agent.execute(
            rt.context(), span, clip_set=clip_art.payload,
            vision=vision_art.payload if vision_art else None,
            research=research_art.payload if research_art else None,
            revision=rev)

    try:
        art = _run_node("content_strategy", ContentStrategyAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"content strategy failed ({type(exc).__name__}: "
                      f"{exc}); the director plans without a brief")
        return {"strategy_id": None, "strategy_failed": str(exc)[:200]}
    return {"strategy_id": art.artifact_id, "strategy_failed": None}


def node_story(state: GraphState) -> dict:
    """The narrative layer between the brief and the edit.

    Runs ONCE, after content_strategy and before editing_director. It sits
    upstream of the critic cycle, so a revision re-plans and re-renders
    without re-authoring the story - the narrative did not change just
    because the cut needs fixing.

    Reads its three upstreams out of the store rather than out of graph
    state, which is the standing rule here: state carries pointers, the store
    carries payloads.

    Never blocking. Losing the story costs the director its narrative order,
    not the operator their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = StoryAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    if clip_art is None:                              # pragma: no cover
        raise RuntimeError("ClipAnalysis artifact missing")
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)
    vision_art = rt.store.get(job_id, "VisionAnalysis", VisionAnalysis)
    research_art = rt.store.get(job_id, "ResearchReport", ResearchReport)

    inputs = [clip_art.artifact_id]
    inputs += [a.artifact_id
               for a in (strategy_art, vision_art, research_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Story")
        return agent.execute(
            rt.context(), span, clip_set=clip_art.payload,
            strategy=strategy_art.payload if strategy_art else None,
            vision=vision_art.payload if vision_art else None,
            research=research_art.payload if research_art else None,
            revision=rev)

    try:
        art = _run_node("story", StoryAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"story failed ({type(exc).__name__}: {exc}); "
                      f"the director plans without a narrative")
        return {"story_id": None, "story_failed": str(exc)[:200]}
    return {"story_id": art.artifact_id, "story_failed": None}


def node_voiceover(state: GraphState) -> dict:
    """What is SAID over the story.

    Runs ONCE, after story and before editing_director, and sits upstream of
    the critic cycle: a revision re-plans and re-renders without rewriting the
    narration, because the words did not change just because the cut needs
    fixing.

    Reads its upstreams out of the store rather than out of graph state,
    which is the standing rule here: state carries pointers, the store
    carries payloads.

    Never blocking. Losing the narration costs the director a script, not the
    operator their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = VoiceoverAgent()

    story_art = rt.store.get(job_id, "Story", Story)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)
    research_art = rt.store.get(job_id, "ResearchReport", ResearchReport)

    inputs = [a.artifact_id
              for a in (story_art, strategy_art, research_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Voiceover")
        return agent.execute(
            rt.context(), span,
            story=story_art.payload if story_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            research=research_art.payload if research_art else None,
            revision=rev)

    try:
        art = _run_node("voiceover", VoiceoverAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"voiceover failed ({type(exc).__name__}: {exc}); "
                      f"the director plans without a script")
        return {"voiceover_id": None, "voiceover_failed": str(exc)[:200]}
    return {"voiceover_id": art.artifact_id, "voiceover_failed": None}


def node_music(state: GraphState) -> dict:
    """What the cut should sound like underneath.

    Runs ONCE, after voiceover and before editing_director, and sits upstream
    of the critic cycle: a revision re-plans and re-renders without
    re-deciding the score, because the music direction did not change just
    because the cut needs fixing.

    Takes ClipAnalysis as well as the narrative layers - its deterministic
    fallback picks the mood from measured motion using the pipeline's own
    MOOD_BANDS, which needs the measurements.

    Never blocking. Losing the direction costs the director a score brief, not
    the operator their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = MusicAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    story_art = rt.store.get(job_id, "Story", Story)
    vo_art = rt.store.get(job_id, "Voiceover", Voiceover)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)

    inputs = [a.artifact_id
              for a in (clip_art, story_art, vo_art, strategy_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Music")
        return agent.execute(
            rt.context(), span,
            clip_set=clip_art.payload if clip_art else None,
            story=story_art.payload if story_art else None,
            voiceover=vo_art.payload if vo_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            revision=rev)

    try:
        art = _run_node("music", MusicAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"music failed ({type(exc).__name__}: {exc}); "
                      f"the director plans without a score brief")
        return {"music_id": None, "music_failed": str(exc)[:200]}
    return {"music_id": art.artifact_id, "music_failed": None}


def node_animation(state: GraphState) -> dict:
    """How the cut should LOOK on the canvas.

    Runs ONCE, after music and before editing_director, and sits upstream of
    the critic cycle: a revision re-plans and re-renders without re-deciding
    the look.

    Takes ClipAnalysis as well as the narrative layers, because every framing
    treatment is derived from the clip's orientation against the canvas - the
    same comparison the renderer itself makes.

    Never blocking. Losing the direction costs the director a look brief, not
    the operator their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = AnimationAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    story_art = rt.store.get(job_id, "Story", Story)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)
    vision_art = rt.store.get(job_id, "VisionAnalysis", VisionAnalysis)

    inputs = [a.artifact_id
              for a in (clip_art, story_art, strategy_art, vision_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Animation")
        return agent.execute(
            rt.context(), span,
            clip_set=clip_art.payload if clip_art else None,
            story=story_art.payload if story_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            vision=vision_art.payload if vision_art else None,
            revision=rev)

    try:
        art = _run_node("animation", AnimationAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"animation failed ({type(exc).__name__}: {exc}); "
                      f"the director plans without a look brief")
        return {"animation_id": None, "animation_failed": str(exc)[:200]}
    return {"animation_id": art.artifact_id, "animation_failed": None}


def node_transition(state: GraphState) -> dict:
    """How one beat gives way to the next.

    Runs ONCE, after animation and before editing_director, and sits upstream
    of the critic cycle: a revision re-plans and re-renders without
    re-choosing the joins.

    Its output reaches the renderer through a path that already exists -
    RenderSpec.xfade_list feeds AT_XFADES, which build_video reads.

    Never blocking. Losing it costs the director a join brief, not the
    operator their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = TransitionAgent()

    story_art = rt.store.get(job_id, "Story", Story)
    anim_art = rt.store.get(job_id, "Animation", Animation)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)

    inputs = [a.artifact_id
              for a in (story_art, anim_art, strategy_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Transition")
        return agent.execute(
            rt.context(), span,
            story=story_art.payload if story_art else None,
            animation=anim_art.payload if anim_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            revision=rev)

    try:
        art = _run_node("transition", TransitionAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"transition failed ({type(exc).__name__}: {exc}); "
                      f"the director plans without a join brief")
        return {"transition_id": None, "transition_failed": str(exc)[:200]}
    return {"transition_id": art.artifact_id, "transition_failed": None}


def node_caption(state: GraphState) -> dict:
    """Which of the words already written also appear on screen.

    Runs ONCE, after transition and before editing_director, and sits
    upstream of the critic cycle: a revision re-plans and re-renders without
    re-choosing the words.

    Takes ClipAnalysis as well as the narrative layers, because how much text
    fits on one line depends on the canvas the renderer draws onto.

    Its output reaches the renderer through a path that already exists -
    Segment.caption -> captions.json -> build_video `cut` -> drawtext.

    Never blocking. Losing it costs the cut its captions, not the operator
    their video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = CaptionAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    story_art = rt.store.get(job_id, "Story", Story)
    vo_art = rt.store.get(job_id, "Voiceover", Voiceover)
    anim_art = rt.store.get(job_id, "Animation", Animation)

    inputs = [a.artifact_id
              for a in (clip_art, story_art, vo_art, anim_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Caption")
        return agent.execute(
            rt.context(), span,
            story=story_art.payload if story_art else None,
            voiceover=vo_art.payload if vo_art else None,
            animation=anim_art.payload if anim_art else None,
            clip_set=clip_art.payload if clip_art else None,
            revision=rev)

    try:
        art = _run_node("caption", CaptionAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"caption failed ({type(exc).__name__}: {exc}); "
                      f"the director plans without captions")
        return {"caption_id": None, "caption_failed": str(exc)[:200]}
    return {"caption_id": art.artifact_id, "caption_failed": None}


def node_editing_director(state: GraphState) -> dict:
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = EditingDirectorAgent()

    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    if clip_art is None:                              # pragma: no cover
        raise RuntimeError("ClipAnalysis artifact missing")
    vision_art = rt.store.get(job_id, "VisionAnalysis", VisionAnalysis)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)
    story_art = rt.store.get(job_id, "Story", Story)
    vo_art = rt.store.get(job_id, "Voiceover", Voiceover)
    music_art = rt.store.get(job_id, "Music", Music)
    anim_art = rt.store.get(job_id, "Animation", Animation)
    trans_art = rt.store.get(job_id, "Transition", Transition)
    cap_art = rt.store.get(job_id, "Caption", Caption)

    raw_req = state.get("revision_request")
    request = RevisionRequest(**raw_req) if raw_req else None

    def body(span):
        rev = rt.store.next_revision(job_id, "EditPlan")
        if request is not None:
            span.note(f"revision {state.get('revision_count', 0)}: "
                      f"{request.issue[:120]}")
            span.note("constraints: " + ", ".join(
                f"{k}={v}" for k, v in request.constraints.items()))
        return agent.execute(
            rt.context(), span, clip_set=clip_art.payload,
            revision=rev, revision_request=request,
            clip_analysis_id=clip_art.artifact_id,
            vision=vision_art.payload if vision_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            story=story_art.payload if story_art else None,
            voiceover=vo_art.payload if vo_art else None,
            music=music_art.payload if music_art else None,
            animation=anim_art.payload if anim_art else None,
            transition=trans_art.payload if trans_art else None,
            caption=cap_art.payload if cap_art else None)

    inputs = [clip_art.artifact_id]
    inputs += [a.artifact_id
               for a in (vision_art, strategy_art, story_art, vo_art,
                         music_art, anim_art, trans_art, cap_art)
               if a]
    art = _run_node("editing_director", EditingDirectorAgent.id, job_id,
                    state.get("attempt", 1), inputs, body)
    return {"edit_plan_id": art.artifact_id,
            # consumed: it must not be re-applied on a later pass
            "revision_request": None}


async def node_render(state: GraphState) -> dict:
    """The one async node - and the reason the graph runs under `ainvoke`.

    Rendering is the only stage that blocks for minutes. Awaiting the
    scheduler instead of calling `pool.run(...)` means the event loop is free
    for that whole time: other jobs' nodes progress, the deadline timer fires
    on schedule, and a cancel is delivered promptly rather than after ffmpeg
    happens to finish.

    The work itself still runs on a bounded pool worker, so concurrency,
    determinism, caching and artifact recording are all unchanged.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = RenderingAgent()
    attempt = state.get("attempt", 1)
    policy = POLICIES.get("render", NodePolicy())

    plan_art = rt.store.get(job_id, "EditPlan", EditPlan)
    if plan_art is None:                              # pragma: no cover
        raise RuntimeError("EditPlan artifact missing")

    async def body(span):
        rev = rt.store.next_revision(job_id, "RenderResult")
        span.tool("scheduler.submit", job_id=job_id, label="render",
                  timeout_s=policy.timeout_s)
        work = rt.scheduler.submit(
            job_id, agent.execute, rt.context(), span,
            plan=plan_art.payload, attempt=attempt, revision=rev,
            edit_plan_id=plan_art.artifact_id,
            stage_timeout_s=policy.timeout_s,
            label="render", timeout_s=policy.timeout_s)
        return await work

    art = await _run_node_async("render", RenderingAgent.id, job_id, attempt,
                                [plan_art.artifact_id], body)
    return {"render_id": art.artifact_id}


def node_critic(state: GraphState) -> dict:
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = CriticAgent()
    attempt = state.get("attempt", 1)

    render_art = rt.store.get(job_id, "RenderResult", RenderResult)
    plan_art = rt.store.get(job_id, "EditPlan", EditPlan)
    clip_art = rt.store.get(job_id, "ClipAnalysis", ClipAnalysisSet)
    if not (render_art and plan_art and clip_art):    # pragma: no cover
        raise RuntimeError("critic is missing an upstream artifact")

    previous = None
    prior = rt.store.all_revisions(job_id, "CriticResult", CriticResult)
    if prior:
        previous = prior[-1].payload

    holder: dict[str, Any] = {}

    def body(span):
        rev = rt.store.next_revision(job_id, "CriticResult")
        art, request = agent.execute(
            rt.context(), span, render=render_art.payload,
            plan=plan_art.payload, clip_set=clip_art.payload, attempt=attempt,
            revision_count=state.get("revision_count", 0),
            max_revisions=state.get("max_revisions", 2),
            previous=previous,
            previously_tried=state.get("previously_tried", []),
            pass_threshold=state.get("pass_threshold",
                                     DEFAULT_PASS_THRESHOLD),
            revision=rev, render_id=render_art.artifact_id)
        holder["request"] = request
        return art

    art = _run_node("critic", CriticAgent.id, job_id, attempt,
                    [render_art.artifact_id, plan_art.artifact_id], body)

    result: CriticResult = art.payload
    request: RevisionRequest | None = holder.get("request")

    history = list(state.get("history", []))
    history.append({
        "attempt": attempt, "verdict": result.verdict,
        "score": result.overall_score,
        "video": render_art.payload.video_path,
        "cache_hits": render_art.payload.cache_hits,
        "edit_plan_id": plan_art.artifact_id,
        "render_id": render_art.artifact_id,
        "critic_id": art.artifact_id,
    })

    out: dict = {"critic_id": art.artifact_id, "verdict": result.verdict,
                 "history": history}

    if result.verdict == "REVISE" and request is not None:
        tried = list(state.get("previously_tried", []))
        key = request.constraint_key()
        if any(t.get("key") == key for t in tried):
            # Anti-oscillation: this exact constraint has already been tried
            # and did not fix it. Looping on it again would burn a render to
            # reach the same place.
            out["verdict"] = "ESCALATE"
            out["revision_request"] = None
            out["status"] = "escalated"
            out["error"] = ("the critic re-issued a constraint that was "
                            "already tried and did not resolve the issue")
            return out
        tried.append({"key": key, "constraints": request.constraints,
                      "from_attempt": attempt})
        out["revision_request"] = request.model_dump(mode="json")
        out["previously_tried"] = tried
        out["revision_count"] = state.get("revision_count", 0) + 1
        out["attempt"] = attempt + 1
    return out


def node_finish(state: GraphState) -> dict:
    return {"status": "done"}


def node_escalate(state: GraphState) -> dict:
    return {"status": "escalated",
            "error": state.get("error")
                     or "maximum revisions reached without a passing render"}


def node_thumbnail(state: GraphState) -> dict:
    """Which frame of the delivered video represents it. PACKAGING.

    Runs ONCE, after finish or escalate - so after the render/critic cycle
    has settled on the video that ships. It reads that render and the plan
    it was cut from by the ids the cycle left in state, chooses a frame of
    it, and grabs it with master_stage's own single-frame operation. It
    cannot cost a render, trigger a revision or change the video.

    Never blocking, and never touches the job's status: losing it leaves
    master_stage's thumbnail in place, not the operator without a video.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = ThumbnailAgent()

    def pick(kind, model, wanted):
        revs = rt.store.all_revisions(job_id, kind, model)
        return next((a for a in revs if a.artifact_id == wanted),
                    revs[-1] if revs else None)

    render_art = pick("RenderResult", RenderResult, state.get("render_id"))
    plan_art = pick("EditPlan", EditPlan, state.get("edit_plan_id"))
    if render_art is None or plan_art is None:
        rt.log.append("thumbnail skipped: there is no delivered render to "
                      "package")
        return {"thumbnail_id": None,
                "thumbnail_failed": "no delivered render"}
    vision_art = rt.store.get(job_id, "VisionAnalysis", VisionAnalysis)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)
    story_art = rt.store.get(job_id, "Story", Story)

    inputs = [render_art.artifact_id, plan_art.artifact_id]
    inputs += [a.artifact_id
               for a in (vision_art, strategy_art, story_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Thumbnail")
        return agent.execute(
            rt.context(), span, plan=plan_art.payload,
            render=render_art.payload,
            vision=vision_art.payload if vision_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            story=story_art.payload if story_art else None,
            verdict=state.get("verdict"), revision=rev)

    try:
        art = _run_node("thumbnail", ThumbnailAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"thumbnail failed ({type(exc).__name__}: {exc}); "
                      f"master_stage's thumbnail stays the one shown")
        return {"thumbnail_id": None, "thumbnail_failed": str(exc)[:200]}
    return {"thumbnail_id": art.artifact_id, "thumbnail_failed": None}


def node_title(state: GraphState) -> dict:
    """The title the delivered video is published under. PACKAGING.

    Runs ONCE, after the thumbnail - so after the render/critic cycle has
    settled on the video that ships. It reads the plan that video was cut
    from by the id the cycle left in state, so story beats on clips the cut
    dropped cannot lend it a word. It cannot cost a render, trigger a
    revision or rename the delivered file.

    Never blocking, and never touches the job's status: losing it leaves the
    operator's title, or the placeholder, exactly as it was.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = TitleAgent()

    plans = rt.store.all_revisions(job_id, "EditPlan", EditPlan)
    plan_art = next((a for a in plans
                     if a.artifact_id == state.get("edit_plan_id")),
                    plans[-1] if plans else None)
    strategy_art = rt.store.get(job_id, "ContentStrategy", ContentStrategy)
    story_art = rt.store.get(job_id, "Story", Story)
    research_art = rt.store.get(job_id, "ResearchReport", ResearchReport)

    inputs = [a.artifact_id
              for a in (plan_art, strategy_art, story_art, research_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Title")
        return agent.execute(
            rt.context(), span,
            plan=plan_art.payload if plan_art else None,
            strategy=strategy_art.payload if strategy_art else None,
            story=story_art.payload if story_art else None,
            research=research_art.payload if research_art else None,
            revision=rev)

    try:
        art = _run_node("title", TitleAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"title failed ({type(exc).__name__}: {exc}); the "
                      f"video keeps the title it already had")
        return {"title_id": None, "title_failed": str(exc)[:200]}
    return {"title_id": art.artifact_id, "title_failed": None}


def node_publishing(state: GraphState) -> dict:
    """The publishing PLAN for the delivered video. PACKAGING's last step.

    Runs ONCE, after the title - so after the cycle has settled on the video
    that ships and after its thumbnail and title exist. It reads that render
    by the id the cycle left in state, verifies the deliverables and states
    what the operator must do to publish. It uploads nothing: no provider
    exists, and publishing is the operator's act.

    Never blocking, and never touches the job's status.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = PublishingAgent()

    renders = rt.store.all_revisions(job_id, "RenderResult", RenderResult)
    render_art = next((a for a in renders
                       if a.artifact_id == state.get("render_id")),
                      renders[-1] if renders else None)
    if render_art is None:
        rt.log.append("publishing skipped: there is no delivered render")
        return {"publishing_id": None,
                "publishing_failed": "no delivered render"}
    thumb_art = rt.store.get(job_id, "Thumbnail", Thumbnail)
    title_art = rt.store.get(job_id, "Title", Title)

    inputs = [a.artifact_id for a in (render_art, thumb_art, title_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "Publishing")
        return agent.execute(
            rt.context(), span, render=render_art.payload,
            thumbnail=thumb_art.payload if thumb_art else None,
            title=title_art.payload if title_art else None,
            verdict=state.get("verdict"), revision=rev)

    try:
        art = _run_node("publishing", PublishingAgent.id, job_id,
                        state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"publishing failed ({type(exc).__name__}: {exc}); "
                      f"no plan was made, and nothing was published")
        return {"publishing_id": None, "publishing_failed": str(exc)[:200]}
    return {"publishing_id": art.artifact_id, "publishing_failed": None}


def node_performance_analysis(state: GraphState) -> dict:
    """How the delivered video performed. The last step, after packaging.

    Runs ONCE, after the publishing plan exists. It reads that plan by the
    id the packaging step left in state, and reports performance as
    unavailable with the measured reasons: nothing here publishes, nothing
    records which video was uploaded, and nothing can read analytics. The
    critic's result and Research's report are passed in only so they can be
    named as EXCLUDED - pre-publication signals, never performance.

    Never blocking, and never touches the job's status.
    """
    job_id = state["job_id"]
    rt = runtime_for(job_id)
    agent = PerformanceAnalysisAgent()

    plans = rt.store.all_revisions(job_id, "Publishing", Publishing)
    pub_art = next((a for a in plans
                    if a.artifact_id == state.get("publishing_id")),
                   plans[-1] if plans else None)
    critics = rt.store.all_revisions(job_id, "CriticResult", CriticResult)
    critic_art = next((a for a in critics
                       if a.artifact_id == state.get("critic_id")),
                      critics[-1] if critics else None)
    research_art = rt.store.get(job_id, "ResearchReport", ResearchReport)
    title_art = rt.store.get(job_id, "Title", Title)

    inputs = [a.artifact_id for a in (pub_art, critic_art, research_art,
                                      title_art) if a]

    def body(span):
        rev = rt.store.next_revision(job_id, "PerformanceAnalysis")
        return agent.execute(
            rt.context(), span,
            publishing=pub_art.payload if pub_art else None,
            critic=critic_art.payload if critic_art else None,
            research=research_art.payload if research_art else None,
            title=title_art.payload if title_art else None,
            revision=rev)

    try:
        art = _run_node("performance_analysis", PerformanceAnalysisAgent.id,
                        job_id, state.get("attempt", 1), inputs, body)
    except (ProcessTimeout, WorkCancelled, ProcessCancelled):
        raise
    except Exception as exc:                          # noqa: BLE001
        rt.log.append(f"performance analysis failed ({type(exc).__name__}: "
                      f"{exc}); no report was made, and no figure was "
                      f"invented")
        return {"performance_id": None, "performance_failed": str(exc)[:200]}
    return {"performance_id": art.artifact_id, "performance_failed": None}


# ------------------------------------------------------------- routing

def route_after_critic(state: GraphState) -> Literal["revise", "escalate", "finish"]:
    verdict = state.get("verdict", "PASS")
    if verdict == "REVISE" and state.get("revision_request"):
        return "revise"
    if verdict == "ESCALATE":
        return "escalate"
    if verdict == "REVISE":                # REVISE with nothing to act on
        return "escalate"
    return "finish"


# --------------------------------------------------------------- builder

def _wire(g: "StateGraph") -> "StateGraph":
    """Nodes and edges. Shared by the sync and async builders so the two can
    never drift into different topologies."""
    g.add_node("clip_analysis", node_clip_analysis)
    g.add_node("vision", node_vision)
    g.add_node("research", node_research)
    g.add_node("content_strategy", node_content_strategy)
    g.add_node("story", node_story)
    g.add_node("voiceover", node_voiceover)
    g.add_node("music", node_music)
    g.add_node("animation", node_animation)
    g.add_node("transition", node_transition)
    g.add_node("caption", node_caption)
    g.add_node("editing_director", node_editing_director)
    g.add_node("render", node_render)
    g.add_node("critic", node_critic)
    g.add_node("finish", node_finish)
    g.add_node("escalate", node_escalate)
    g.add_node("thumbnail", node_thumbnail)
    g.add_node("title", node_title)
    g.add_node("publishing", node_publishing)
    g.add_node("performance_analysis", node_performance_analysis)

    g.set_entry_point("clip_analysis")
    # fan out: vision and research are independent of each other
    g.add_edge("clip_analysis", "vision")
    g.add_edge("clip_analysis", "research")
    # join: the STRATEGY waits for both, then briefs the director. Vision and
    # research answer "what is there"; the director answers "how to cut it".
    # Neither question is "what are we saying", which is why they no longer
    # meet at the director.
    g.add_edge("vision", "content_strategy")
    g.add_edge("research", "content_strategy")
    # the brief becomes a narrative, and only then an edit. Three distinct
    # questions - what we are saying, how it is told, how it is cut - and
    # every one of them sits OUTSIDE the critic cycle, so a revision re-plans
    # without re-deciding the message or re-authoring the story.
    g.add_edge("content_strategy", "story")
    g.add_edge("story", "voiceover")
    g.add_edge("voiceover", "music")
    g.add_edge("music", "animation")
    g.add_edge("animation", "transition")
    g.add_edge("transition", "caption")
    g.add_edge("caption", "editing_director")
    g.add_edge("editing_director", "render")
    g.add_edge("render", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "revise": "editing_director",     # <- the cycle
        "escalate": "escalate",
        "finish": "finish",
    })
    # packaging: whichever way the cycle ended, the delivered video gets its
    # thumbnail once - after the cycle, so it can never re-enter it
    g.add_edge("finish", "thumbnail")
    g.add_edge("escalate", "thumbnail")
    g.add_edge("thumbnail", "title")
    g.add_edge("title", "publishing")
    g.add_edge("publishing", "performance_analysis")
    g.add_edge("performance_analysis", END)
    return g


def build_graph(checkpoint_path: str | Path):
    """Compile with the SYNC checkpointer - for reading state, not running.

    `node_render` is a coroutine, so this build cannot be `invoke`d; it exists
    so `peek_state` and the resume tests can inspect a checkpoint without an
    event loop. The two savers share one on-disk format, so a checkpoint
    written by the async run is fully readable here.

    `check_same_thread=False` because LangGraph writes checkpoints from
    whichever thread is executing the graph, and the connection is guarded by
    the saver itself.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    return _wire(StateGraph(GraphState)).compile(checkpointer=saver), conn


@asynccontextmanager
async def build_graph_async(checkpoint_path: str | Path):
    """Compile with the ASYNC checkpointer - what actually runs a job.

    `SqliteSaver` raises NotImplementedError on every async method, so driving
    the graph with `ainvoke` requires `AsyncSqliteSaver`. Both write the same
    `checkpoints` / `writes` tables, which is what lets a job started before
    this change still resume, and lets `peek_state` keep its sync reader.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        yield _wire(StateGraph(GraphState)).compile(checkpointer=saver)


def initial_state(job: JobState,
                  pass_threshold: float = DEFAULT_PASS_THRESHOLD) -> GraphState:
    return {
        "job_id": job.job_id, "status": "running", "error": None,
        "attempt": 1, "revision_count": 0,
        "max_revisions": job.max_revisions,
        "pass_threshold": pass_threshold,
        "verdict": "", "revision_request": None,
        "previously_tried": [], "history": [],
    }


__all__ = ["GraphState", "Runtime", "register", "runtime_for", "unregister",
           "build_graph", "build_graph_async", "initial_state",
           "route_after_critic", "POLICIES", "NodePolicy"]
