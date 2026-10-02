"""
Alpha Titans - agent layer (complete: all eighteen agents).

This package is ADDITIVE. Nothing in `pipeline/` and nothing in the existing
`server/` modules is modified by it. The legacy build path
(`server/jobs.py::_run`) keeps working exactly as before; this package is a
second, parallel path that can be driven from `server/agents/runner.py`.

All eighteen agents of the target architecture are implemented. Do not
maintain the list here - `agents.base.agent_registry()` derives it by
walking this package, and `/api/v2/health` answers from that. A list in a
docstring is a list that goes stale without anything failing.

    produced by a model   vision · research · content_strategy · story ·
                          voiceover · music · animation · transition ·
                          caption · editing_director · thumbnail · title
    measured or derived   clip_analysis · rendering · critic ·
                          publishing · performance_analysis · learning

The orchestrator is a real LangGraph StateGraph with a SQLite checkpointer,
a conditional edge and a genuine cycle (critic -> editing_director). Every
agent upstream of the editing_director sits OUTSIDE that cycle, so a
revision re-plans and re-renders without re-deciding the message.

With no API key configured every reasoning agent runs its own deterministic
fallback and says so: the artifact carries `degraded=True` and
`fallback_used`, and /api/v2/health reports `reasoning_enabled: false`.
"""

from __future__ import annotations

# The artifact envelope's schema version lives in artifacts.py, which is
# what actually stamps it. Importing it here too would create a second
# source of truth for the same number.
PHASE = 5
