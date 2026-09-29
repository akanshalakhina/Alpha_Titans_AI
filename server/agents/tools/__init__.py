"""
Deterministic tool layer.

Every tool here is a thin, typed wrapper over code that ALREADY EXISTS and is
already proven in production. Nothing in `pipeline/` is modified, and nothing
in it is reimplemented. The rule the architecture sets out is enforced here:

    the LLM reasons about WHICH clip and ROUGHLY how long;
    these tools compute the exact frames, the exact filtergraph and the exact
    encode, and they validate every number the LLM produced.

An LLM never writes an ffmpeg filter string. `fit_filter` in
pipeline/build_video.py does that, unchanged, exactly as it does today.
"""

from __future__ import annotations

from .pipeline_bridge import build_video, extras_mod, adapter_mod, config_mod

__all__ = ["build_video", "extras_mod", "adapter_mod", "config_mod"]
