"""
Clip analysis tools.

These do NOT reimplement probe or measure. They invoke the existing
`pipeline/build_video.py probe` and `... measure` stages as subprocesses -
exactly the way `server/adapter.py::run_stage` already invokes them - and then
convert the result into typed `ClipAnalysis` artifacts.

Why subprocess rather than calling the functions in-process: `stage_probe` and
`stage_measure` are monolithic (they read args, write manifest.json and print a
report). Calling them as the CLI already does means the YDIF measurement, the
freezedetect pass, the iOS duplicate rule and the orientation decision are
byte-for-byte the code that is in production today. Nothing is duplicated and
nothing can drift.

manifest.json is read here as the TRANSPORT OUT of that existing tool. It is
not the source of truth: the typed ClipAnalysisSet artifact is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..artifacts import (ClipAnalysis, ClipAnalysisSet, ClipMotion,
                         ClipTechnical, ClipUsability)
from .pipeline_bridge import BUILD_VIDEO_PY, adapter_mod, build_video


def run_pipeline_stage(stage: str, build_dir: str, src_dir: str | None = None,
                       env_overrides: dict[str, str] | None = None,
                       extra_args: list[str] | None = None,
                       on_line=None, *, pool=None, job_id: str | None = None,
                       timeout_s: float | None = None) -> tuple[int, str]:
    """Invoke one existing pipeline stage as a subprocess.

    Mirrors server/adapter.py::run_stage, including streaming stdout back line
    by line so the trace and the SSE log see the pipeline's own output.

    OWNERSHIP: when a `pool` is given, the pool creates and owns the process -
    which is what makes a deadline enforceable and a cancel real. The process
    started here is `python build_video.py <stage>`, and IT spawns ffmpeg, so
    only the pool's tree-kill reaches the actual encoder.

    Without a pool the call falls back to an unowned subprocess, so the module
    stays usable on its own (a bare script, a REPL). Every agent path passes
    one.
    """
    cmd = [sys.executable, str(BUILD_VIDEO_PY), stage, "--out", build_dir]
    if src_dir:
        cmd += ["--src", src_dir]
    cmd += list(extra_args or [])

    if pool is not None:
        return pool.run_process(cmd, env=env_overrides, on_line=on_line,
                                timeout_s=timeout_s, job_id=job_id,
                                label=f"pipeline:{stage}")

    env = dict(os.environ)
    env.update(env_overrides or {})
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env)
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            lines.append(line)
            if on_line:
                on_line(line)
    return proc.wait(), "\n".join(lines)


def read_manifest(build_dir: str) -> dict | None:
    p = Path(build_dir) / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------ conversion

def manifest_to_clip_set(manifest: dict, src_dir: str,
                         target_seconds: float | None = None) -> ClipAnalysisSet:
    """Turn the existing manifest into typed artifacts.

    Every number here was measured by the unmodified pipeline. Nothing is
    estimated and nothing is defaulted silently - a clip the measure stage did
    not reach is marked `measured=False` so the EditingDirector can see that
    its motion figure is absent rather than zero.
    """
    clips: list[ClipAnalysis] = []
    for c in manifest.get("clips") or []:
        measured = "mean_ydif" in c
        dur = float(c.get("duration") or 0.0)
        coded = tuple(c.get("coded") or [0, 0])
        display = tuple(c.get("display") or coded)

        technical = ClipTechnical(
            duration=dur,
            coded=(int(coded[0]), int(coded[1])),
            display=(int(display[0]), int(display[1])),
            rotation=int(c.get("rotation") or 0),
            fps=float(c.get("fps") or 0.0),
            codec=c.get("codec"),
            has_audio=bool(c.get("audio")),
            recorded_at=c.get("recorded"),
            orientation=c.get("orientation") or
                        ("vertical" if display[1] > display[0] else "horizontal"),
        )
        motion = ClipMotion(
            sample_fps=float(c.get("sample_fps") or 10.0),
            mean_ydif=float(c.get("mean_ydif") or 0.0),
            max_ydif=float(c.get("max_ydif") or 0.0),
            live_share=float(c.get("live_share") or 0.0),
            freeze_runs=[(float(a), float(b))
                         for a, b in (c.get("freeze_runs") or [])],
            freeze_share=float(c.get("freeze_share") or 0.0),
            ydif_series=[float(v) for v in (c.get("ydif") or [])],
            measured=measured,
        )
        clips.append(ClipAnalysis(
            clip_id=c["name"],
            path=str(Path(src_dir) / c["name"]),
            technical=technical,
            motion=motion,
            usability=_judge_usability(motion, dur),
            slowmo=bool(c.get("slowmo")),
        ))

    canvas = tuple(manifest.get("canvas") or [1920, 1080])
    return ClipAnalysisSet(
        clips=clips,
        orientation=manifest.get("orientation") or "horizontal",
        canvas=(int(canvas[0]), int(canvas[1])),
        target_seconds=float(target_seconds or manifest.get("target") or 120.0),
        unreadable=list(manifest.get("unreadable") or []),
        dropped=dict(manifest.get("dropped") or {}),
    )


def _judge_usability(motion: ClipMotion, duration: float) -> ClipUsability:
    """A per-clip usability judgement, using the pipeline's own constants.

    This is a DECISION the current system does not surface: `stage_plan`
    decides rejection in-line and only reports it in printed text. Making it a
    typed per-clip field is what lets the EditingDirector reason about a calm
    clip rather than discovering it vanished.

    The thresholds are build_video.py's own (ABS_DEAD_YDIF, LIVE_SHARE_FLOOR),
    read from the module rather than copied, so they cannot drift.
    """
    if not motion.measured:
        return ClipUsability(usable=True, reason="not measured", confidence=0.0)
    if duration <= 0:
        return ClipUsability(usable=False, reason="zero duration", confidence=1.0)

    dead = motion.mean_ydif <= build_video.ABS_DEAD_YDIF
    still = motion.live_share < build_video.LIVE_SHARE_FLOOR
    if dead and still:
        return ClipUsability(
            usable=False,
            reason=("nothing changes in it (only %.0f%% of frames differ at all)"
                    % (100 * motion.live_share)),
            confidence=1.0)

    # Confidence in the motion figure itself: a handful of samples is a weak
    # basis for a mean, so say so rather than presenting it as solid.
    n = len(motion.ydif_series)
    conf = 1.0 if n >= 100 else (0.6 if n >= 30 else 0.3 if n > 0 else 0.0)
    return ClipUsability(usable=True, reason="", confidence=conf)


def canvas_for(orientation: str, resolution: str) -> tuple[int, int]:
    """Resolution preset -> canvas, honouring orientation.

    Reuses adapter.RESOLUTIONS so the UI and the agent path cannot disagree.
    """
    w, h = adapter_mod.RESOLUTIONS.get(resolution, (1920, 1080))
    if orientation == "vertical":
        w, h = h, w
    return int(w), int(h)


__all__ = ["run_pipeline_stage", "read_manifest", "manifest_to_clip_set",
           "canvas_for"]
