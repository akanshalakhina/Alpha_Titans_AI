"""
Alpha Titans - agent layer (Phase 1 vertical slice).

This package is ADDITIVE. Nothing in `pipeline/` and nothing in the existing
`server/` modules is modified by it. The legacy build path
(`server/jobs.py::_run`) keeps working exactly as before; this package is a
second, parallel path that can be driven from `server/agents/runner.py`.

Phase 1 implements four of the eighteen agents in the target architecture:

    clip_analysis      wraps the existing probe/measure work as tools
    editing_director   the first real reasoning agent (LLM + deterministic
                       validation, deterministic planner as fallback)
    rendering          submits to a bounded render pool; never runs ffmpeg
                       inside a graph node
    critic             inspects the ACTUAL rendered MP4, technical + one
                       creative dimension (first-3-second hook)

The orchestrator is a real LangGraph StateGraph with a SQLite checkpointer,
a conditional edge and a genuine cycle (critic -> editing_director).
"""

from __future__ import annotations

SCHEMA_VERSION = 1
PHASE = 1
