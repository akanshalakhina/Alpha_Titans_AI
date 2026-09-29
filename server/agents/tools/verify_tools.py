"""
Verification tools - what the Critic measures on the ACTUAL rendered file.

Everything here runs against the finished MP4 on disk. Nothing inspects the
EditPlan. That distinction is the whole point: a critic that reads the plan is
marking its own homework, and would happily pass a plan that ffmpeg then
rendered as black frames.

The technical checks reuse server/extras.py's existing verification helpers
(`frame_count`, `loudness`) - the same functions that already gate every build
the app ships today.

The creative check measures the OUTPUT's own YDIF series with exactly the
filter chain `stage_measure` uses on source clips (scale=160, signalstats,
metadata=print), parsed with the pipeline's own `parse_meta`. So "is the hook
lively?" is answered by the same measurement that decided the cut, applied to
the thing that was actually produced.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .pipeline_bridge import build_video, config_mod, extras_mod

FFMPEG = config_mod.FFMPEG


# --------------------------------------------------------------- probing

def probe_final(path: str | Path) -> dict:
    """ffprobe the finished file. Uses adapter's existing wrapper."""
    from .pipeline_bridge import adapter_mod
    info = adapter_mod.ffprobe_json(str(path))
    v = next((s for s in info.get("streams", [])
              if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", [])
              if s.get("codec_type") == "audio"), {})
    fmt = info.get("format", {})
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "duration": round(duration, 3),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": v.get("r_frame_rate"),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name"),
        "has_audio": bool(a),
        "size": int(fmt.get("size") or 0),
    }


def frame_count(path: str | Path) -> int | None:
    """Count DECODED frames - existing extras.frame_count, unmodified.

    Counting decoded frames rather than trusting the container duration is the
    existing system's strongest verification idea and it is kept exactly.
    """
    return extras_mod.frame_count(Path(path))


def loudness(path: str | Path) -> float | None:
    """Integrated loudness via ebur128 - existing extras.loudness."""
    return extras_mod.loudness(Path(path))


def black_frames(path: str | Path) -> tuple[bool, str]:
    rc, out = extras_mod.run([FFMPEG, "-nostdin", "-v", "error", "-i", str(path),
                              "-vf", "blackdetect=d=0.3", "-f", "null", "-"])
    text = (out or "").strip()
    return bool(text), text[:400]


def decode_check(path: str | Path) -> tuple[bool, str]:
    """Full decode pass. Any error-level output means a corrupt stream."""
    rc, out = extras_mod.run([FFMPEG, "-nostdin", "-v", "error", "-i", str(path),
                              "-f", "null", "-"])
    text = (out or "").strip()
    return (rc == 0 and not text), text[:400]


# ------------------------------------------------- output motion profile

def output_motion_profile(path: str | Path, rate: float = 10.0,
                          hook_seconds: float = 3.0) -> dict:
    """Measure the finished video's own motion, the same way clips are measured.

    Returns the evidence the hook check is scored from. Every figure is
    measured; none is estimated. `samples_in_hook` is reported so a short or
    badly-sampled file produces a low-confidence result rather than a
    confident wrong one.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="at_critic_") as tmp:
        stats = Path(tmp) / "out.ydif.txt"
        rc, out = build_video.sh(
            [FFMPEG, "-nostdin", "-v", "error", "-i", str(path),
             "-vf", "scale=160:-2,fps=%g,signalstats,metadata=print:file=%s"
                    % (rate, build_video.esc_path(stats)),
             "-f", "null", "-"], check=False)
        if rc != 0 or not stats.exists():
            return {"measured": False, "why": (out or "")[-300:],
                    "series": [], "rate": rate}
        series_map = build_video.parse_meta(str(stats), "signalstats.YDIF")

    vals = [series_map[i] for i in sorted(series_map)]
    if not vals:
        return {"measured": False, "why": "signalstats produced no samples",
                "series": [], "rate": rate}

    n_hook = max(1, int(hook_seconds * rate))
    hook_vals = vals[:n_hook]
    ordered = sorted(vals)
    mid = len(ordered) // 2
    median = (ordered[mid] if len(ordered) % 2
              else (ordered[mid - 1] + ordered[mid]) / 2.0)

    return {
        "measured": True,
        "rate": rate,
        "hook_seconds": hook_seconds,
        "samples_total": len(vals),
        "samples_in_hook": len(hook_vals),
        "hook_mean": round(sum(hook_vals) / len(hook_vals), 3),
        "hook_max": round(max(hook_vals), 3),
        "overall_mean": round(sum(vals) / len(vals), 3),
        "overall_median": round(median, 3),
        "series": [round(v, 2) for v in vals],
    }


# ---------------------------------------------------------- hook scoring

HOOK_SLOPE = 70.0          # score at ratio 1.0; see the note below


def score_hook(profile: dict) -> tuple[float, str, float]:
    """Score the first-3-second hook. Returns (score, evidence, confidence).

    THE FORMULA, STATED PLAINLY SO IT CAN BE ARGUED WITH
    ----------------------------------------------------
        ratio = mean YDIF over the first 3s  /  median YDIF of the whole video
        score = min(100, 70 * ratio)

    The comparison is against the video's OWN median rather than an absolute
    figure, for the same reason build_video.py rejects clips relatively: YDIF
    scale is a property of the subject, so an absolute bar would pass a
    frenetic video with a dead opening and fail a calm one that opens on its
    liveliest moment.

    At ratio 1.0 the opening is exactly as lively as the video's typical
    moment, which scores 70 - a pass, but not a strong one. The default pass
    threshold of 65 therefore means "do not open on a stretch noticeably
    quieter than your own average". Reaching 100 needs an opening about 1.43x
    the median.

    This is a heuristic over a real measurement, not a claim about viewer
    retention. Phase 7's learning loop is what would replace the constant with
    something evidenced.
    """
    if not profile.get("measured"):
        return 0.0, f"motion could not be measured: {profile.get('why', '')}", 0.0

    median = profile["overall_median"]
    hook = profile["hook_mean"]
    if median <= 0:
        return 0.0, ("the whole video measures zero motion - it is static "
                     "throughout, so the opening cannot be judged against it"), 0.2

    ratio = hook / median
    score = max(0.0, min(100.0, HOOK_SLOPE * ratio))

    n = profile.get("samples_in_hook", 0)
    confidence = 1.0 if n >= 25 else (0.6 if n >= 10 else 0.3)

    evidence = (
        "first %.0fs of the RENDERED file measures %.2f mean YDIF against a "
        "whole-video median of %.2f (ratio %.2f); measured over %d samples at "
        "%gfps"
        % (profile["hook_seconds"], hook, median, ratio, n, profile["rate"])
    )
    return round(score, 1), evidence, confidence


__all__ = ["probe_final", "frame_count", "loudness", "black_frames",
           "decode_check", "output_motion_profile", "score_hook"]
