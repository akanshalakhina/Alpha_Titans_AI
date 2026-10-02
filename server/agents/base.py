"""
The agent contract.

Every agent in the target architecture - all eighteen, not just the four in
Phase 1 - implements this shape. The uniformity is what lets the orchestrator
retry, trace, fall back and revise any of them without special-casing, and what
makes an agent's agency visible: `uses_llm` is a declared property, so a
deterministic service cannot quietly present itself as a reasoning agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .artifacts import (Artifact, ArtifactKind, Cost, Decision, JobState,
                        content_hash)
from .render_pool import MediaPool
from .store import ArtifactStore
from .tools.llm_tools import LLMClient
from .trace import TraceRecorder, TraceSpan


@dataclass
class AgentContext:
    """Everything an agent is allowed to touch. Nothing is global."""

    job: JobState
    store: ArtifactStore
    tracer: TraceRecorder
    pool: MediaPool
    llm: LLMClient
    log: list[str] = field(default_factory=list)

    def emit(self, line: str) -> None:
        self.log.append(line)


class Agent:
    """Base class. Subclasses implement `execute`."""

    id: str = "agent"
    produces: ArtifactKind = "JobState"
    uses_llm: bool = False
    agency: str = "deterministic"        # reasoning | hybrid | deterministic

    # ------------------------------------------------------------ helpers

    def wrap(self, ctx: AgentContext, payload, *, revision: int,
             inputs_hash: str, span: TraceSpan,
             decisions: list[Decision] | None = None,
             confidence: float = 1.0, degraded: bool = False,
             fallback_used: str | None = None,
             cost: Cost | None = None) -> Artifact:
        """Build the envelope and persist it.

        Doing this in one place is what guarantees no agent can write an
        artifact without a producer, an inputs hash, a confidence and its
        decisions attached.
        """
        art = Artifact(
            job_id=ctx.job.job_id, kind=self.produces, revision=revision,
            produced_by=self.id, inputs_hash=inputs_hash,
            trace_id=span.trace_id, payload=payload,
            decisions=list(decisions or []), confidence=confidence,
            degraded=degraded, fallback_used=fallback_used,
            cost=cost or Cost(),
        )
        ctx.store.put(art)
        span.output(art.artifact_id)
        if degraded:
            span.note(f"artifact marked degraded (fallback={fallback_used})")
        return art

    @staticmethod
    def hash_inputs(*parts: Any) -> str:
        return content_hash(list(parts))

    @staticmethod
    def timed() -> float:
        return time.time()


# ------------------------------------------------------------- registry

def agent_registry() -> dict[str, dict]:
    """Every concrete agent in this package, discovered by walking it.

    DERIVED, never hand-written. An earlier /api/v2/health answered from a
    literal list of four agent ids and a literal phase string; fourteen
    agents later it was still saying four, because nothing made it wrong.
    Walking the package means the only way to change this answer is to add
    or remove an agent, which is exactly when it should change.

    Imported lazily inside the function: the agent modules import `base`,
    so walking them at module scope would be a cycle.
    """
    import importlib
    import inspect
    import pkgutil

    import agents as _pkg

    out: dict[str, dict] = {}
    for mi in pkgutil.iter_modules(_pkg.__path__):
        if mi.ispkg:
            continue
        try:
            mod = importlib.import_module(f"agents.{mi.name}")
        except Exception:                      # noqa: BLE001
            # an agent whose optional deps are missing must not take the
            # health endpoint down with it
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, Agent) and obj is not Agent
                    and obj.__module__ == mod.__name__):
                out[obj.id] = {
                    "id": obj.id,
                    "produces": obj.produces,
                    "agency": getattr(obj, "agency", "unknown"),
                    "uses_llm": bool(getattr(obj, "uses_llm", False)),
                    "module": mi.name,
                }
    return dict(sorted(out.items()))


__all__ = ["Agent", "AgentContext", "agent_registry"]
