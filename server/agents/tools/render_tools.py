"""
Render tools.

The existing render path is used UNMODIFIED:

    pipeline/build_video.py cut    -> parts/pNN.mp4   (fit_filter, drawtext)
    pipeline/build_video.py join   -> _joined.mp4     (xfade chain)
    server/extras.py master_stage  -> final mp4       (mux, loudnorm, thumb,
                                                       frame-count verify)

No filtergraph is built here and no LLM output ever reaches ffmpeg as a
string. The typed RenderSpec is PROJECTED into manifest.json immediately before
`cut` runs, because that is the interface the existing stages read. The
projection is one-way and disposable; the artifact remains the source of truth.

THE CACHE CORRECTNESS PROBLEM THIS FILE SOLVES
----------------------------------------------
`stage_cut` skips any segment whose `parts/pNN.mp4` passes `good_video()`. That
caching is what makes a revision cheap - but it is keyed on the FILE PATH, and
the path is just the segment index. After a revision that changes segment 1's
in/out points while keeping two segments, `parts/p01.mp4` still exists and
still probes fine, so the stale render would be silently reused and the
revision would appear to do nothing.

So a content key is written beside each part. Before `cut` runs, any part whose
key no longer matches its segment is deleted, which makes `good_video()` fail
for exactly the segments that genuinely changed and pass for the ones that did
not. `_joined.mp4` is dropped whenever anything changed, because `stage_join`
has its own skip-if-duration-matches guard and two different edits can share a
total duration.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import (ClipAnalysisSet, EditPlan, RenderSpec,
                         SegmentRenderSpec)
from .analysis_tools import read_manifest, run_pipeline_stage
from .pipeline_bridge import adapter_mod, build_video, extras_mod


# ------------------------------------------------------------- compiling

def compile_render_spec(job_id: str, build_dir: str, src_dir: str,
                        plan: EditPlan, quality: str = "high",
                        xfade_duration: float = 0.4,
                        title: str = "ALPHA_TITANS") -> RenderSpec:
    """EditPlan -> RenderSpec. Deterministic; never authored by a model.

    The transition list comes from the plan when the Transition agent chose
    one, and falls back to a plain `fade` per cut when it did not - which is
    what every plan made before that agent existed carries, so old jobs and
    resumed checkpoints behave exactly as they did.

    The names are safe by the time they arrive twice over: `Transition.kind`
    is a closed literal, and `EditPlan.xfade_list` re-filters against the
    renderer's vocabulary on the way in. Nothing else may set it.

    Length is not forced to match `n_cuts`. build_video indexes the list
    cyclically - `XFADES[(i - 1) % len(XFADES)]` - so a shorter sequence
    repeats and a longer one is simply not reached, both of which are
    harmless. Padding or truncating here would invent a choice nobody made.
    """
    n_cuts = max(len(plan.segments) - 1, 1)
    return RenderSpec(
        job_id=job_id,
        build_dir=str(build_dir),
        src_dir=str(src_dir),
        canvas=plan.canvas,
        crf=int(adapter_mod.QUALITY.get(quality, 18)),
        preset="fast",
        fps=30,
        xfade_duration=float(xfade_duration),
        xfade_list=list(plan.xfade_list) or ["fade"] * n_cuts,
        title=title,
        segments=[
            SegmentRenderSpec(
                index=s.index, clip_id=s.clip_id, start=s.start, end=s.end,
                duration=s.duration, caption=s.caption,
                orientation=s.orientation, cache_key=s.cache_key(),
            )
            for s in plan.segments
        ],
    )


# ------------------------------------------------------------ projection

def project_manifest(spec: RenderSpec) -> dict:
    """Write the compatibility projection the existing stages read.

    Reads the manifest probe/measure already wrote (for `src`, `canvas`, the
    measured clip rows) and replaces only the `segments` block. Everything the
    existing pipeline expects is present; nothing it does not expect is added.
    """
    build_dir = Path(spec.build_dir)
    m = read_manifest(str(build_dir))
    if not m:
        raise RuntimeError("no manifest to project onto - probe has not run")

    m["canvas"] = list(spec.canvas)
    m["segments"] = [
        {
            "clip": s.clip_id,
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "duration": round(s.duration, 2),
            "score": 0.0,
            "orientation": s.orientation,
            "slowmo": False,
            "mean_ydif": 0.0,
            "reason": "from the typed EditPlan",
            "index": s.index,
            "caption": s.caption,
        }
        for s in spec.segments
    ]
    m["_projected_from"] = "EditPlan"      # so nobody mistakes this for truth
    build_video.save(str(build_dir), m)

    caps = [s.caption for s in spec.segments]
    (build_dir / "captions.json").write_text(json.dumps(caps, indent=2))
    return m


# ----------------------------------------------------------- cache logic

def _key_path(build_dir: Path, index: int) -> Path:
    return build_dir / "parts" / f"p{index:02d}.key"


def _part_path(build_dir: Path, index: int) -> Path:
    return build_dir / "parts" / f"p{index:02d}.mp4"


def plan_cache(spec: RenderSpec) -> dict[int, bool]:
    """Which segments can legitimately be reused? Does not mutate anything.

    A segment is a cache hit only when the rendered file exists, passes the
    pipeline's own `good_video` check, AND its stored content key matches the
    segment it is supposed to represent.
    """
    build_dir = Path(spec.build_dir)
    hits: dict[int, bool] = {}
    for s in spec.segments:
        part, key = _part_path(build_dir, s.index), _key_path(build_dir, s.index)
        ok = part.exists() and build_video.good_video(str(part))
        if ok and key.exists():
            try:
                ok = key.read_text().strip() == s.cache_key
            except OSError:
                ok = False
        elif ok:
            ok = False          # rendered by something that left no key
        hits[s.index] = ok
    return hits


def invalidate_stale(spec: RenderSpec) -> tuple[list[int], list[int]]:
    """Delete parts that no longer match their segment.

    Returns (reused_indices, invalidated_indices). Drops `_joined.mp4` when
    anything was invalidated, because stage_join's own skip check compares
    total duration only and two different cuts can total the same.
    """
    build_dir = Path(spec.build_dir)
    (build_dir / "parts").mkdir(parents=True, exist_ok=True)
    hits = plan_cache(spec)

    reused = [i for i, ok in hits.items() if ok]
    stale = [i for i, ok in hits.items() if not ok]

    for idx in stale:
        _part_path(build_dir, idx).unlink(missing_ok=True)
        _key_path(build_dir, idx).unlink(missing_ok=True)

    if stale:
        (build_dir / "_joined.mp4").unlink(missing_ok=True)
    return sorted(reused), sorted(stale)


def write_cache_keys(spec: RenderSpec) -> None:
    """Stamp each rendered part with the content it represents."""
    build_dir = Path(spec.build_dir)
    for s in spec.segments:
        part = _part_path(build_dir, s.index)
        if part.exists():
            _key_path(build_dir, s.index).write_text(s.cache_key)


# -------------------------------------------------------------- stages

def run_cut(spec: RenderSpec, on_line=None, *, pool=None,
            timeout_s: float | None = None) -> tuple[int, str]:
    return run_pipeline_stage("cut", spec.build_dir,
                              env_overrides=spec.env_overrides(),
                              on_line=on_line, pool=pool,
                              job_id=spec.job_id, timeout_s=timeout_s)


def run_join(spec: RenderSpec, on_line=None, *, pool=None,
             timeout_s: float | None = None) -> tuple[int, str]:
    return run_pipeline_stage("join", spec.build_dir,
                              env_overrides=spec.env_overrides(),
                              on_line=on_line, pool=pool,
                              job_id=spec.job_id, timeout_s=timeout_s)


def run_master(job_id: str, spec: RenderSpec, on_line=None) -> dict:
    """Mux, loudness-normalise, thumbnail and VERIFY, via existing extras.py.

    Phase 1 has no Music Agent, so no audio track is produced and
    `master_stage` takes its silent-mux path - the same path the current app
    takes when music is set to none.
    """
    settings = {
        "quality": _quality_for_crf(spec.crf),
        "title": spec.title,
        "speed": "normal",
        "music_source": "none",
        "keep_audio": "none",
    }
    return extras_mod.master_stage(job_id, settings, on_line or (lambda _l: None))


def _quality_for_crf(crf: int) -> str:
    for name, value in adapter_mod.QUALITY.items():
        if value == crf:
            return name
    return "high"


__all__ = ["compile_render_spec", "project_manifest", "plan_cache",
           "invalidate_stale", "write_cache_keys", "run_cut", "run_join",
           "run_master"]
