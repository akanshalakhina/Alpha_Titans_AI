"""
Selection and timing tools - the deterministic half of the EditingDirector.

THE DIVISION OF LABOUR THIS FILE ENFORCES
-----------------------------------------
The LLM decides WHICH clips, in WHAT ORDER, and ROUGHLY how long each should
run. It never decides exact timestamps. `best_window` does - by scanning the
measured YDIF series in 0.2s steps and penalising overlap with freezedetect
runs. That is arithmetic over real measurements, and a model guessing
timestamps from a text summary would be strictly worse at it.

So every duration the LLM proposes is passed through `resolve_window()`, which
clamps it to the pipeline's own MIN_SEG/MAX_SEG and the clip's real length,
then asks the existing `best_window` where that window actually goes.

`deterministic_plan()` is the fallback: it runs the REAL `plan` stage as a
subprocess, so the fallback is the production planner itself rather than a
reimplementation of it that could drift.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import ClipAnalysis, ClipAnalysisSet, Segment
from .analysis_tools import read_manifest, run_pipeline_stage
from .pipeline_bridge import build_video


# --------------------------------------------------------------- windows

def window_score(clip: ClipAnalysis, start: float, end: float) -> float:
    """Existing pipeline scorer, unmodified (build_video.window_score)."""
    return float(build_video.window_score(clip.to_planner_dict(), start, end))


def best_window(clip: ClipAnalysis, length: float) -> tuple[float, float, float]:
    """Existing pipeline window search, unmodified (build_video.best_window).

    Returns (start, end, score).
    """
    a, b, s = build_video.best_window(clip.to_planner_dict(), length)
    return float(a), float(b), float(s)


def clamp_length(clip: ClipAnalysis, wanted: float) -> tuple[float, list[str]]:
    """Clamp a proposed duration to what is physically and editorially legal.

    Bounds come from build_video.py itself (MIN_SEG 2.5, MAX_SEG 12.0) so they
    cannot diverge from the pipeline. Returns the clamped value and a list of
    human-readable notes describing every clamp that fired - those notes end up
    in the EditPlan's Decision records, so an operator can see that the model
    asked for 30s and the planner allowed 12.
    """
    notes: list[str] = []
    out = float(wanted)
    dur = clip.technical.duration

    if out < build_video.MIN_SEG:
        notes.append(f"{clip.clip_id}: {out:.1f}s raised to the "
                     f"{build_video.MIN_SEG}s minimum segment length")
        out = build_video.MIN_SEG
    if out > build_video.MAX_SEG:
        notes.append(f"{clip.clip_id}: {out:.1f}s cut to the "
                     f"{build_video.MAX_SEG}s maximum segment length")
        out = build_video.MAX_SEG
    if out > dur:
        notes.append(f"{clip.clip_id}: {out:.1f}s cut to the clip's real "
                     f"length of {dur:.1f}s")
        out = dur
    return round(out, 2), notes


def resolve_window(clip: ClipAnalysis, wanted_length: float,
                   min_mean_ydif: float | None = None,
                   search_within: tuple[float, float] | None = None,
                   avoid: list[tuple[float, float]] | None = None
                   ) -> tuple[float, float, float, list[str]]:
    """Turn a requested LENGTH into exact in/out points.

    `min_mean_ydif` is how a revision constraint reaches the timing layer: when
    the critic says the opening is too flat, the window search is re-run and
    the result is accepted only if it actually clears the bar. If nothing in
    the clip clears it, the best available window is returned together with a
    note saying so - the constraint is reported as unmet rather than faked.
    """
    length, notes = clamp_length(clip, wanted_length)

    if search_within:
        lo, hi = search_within
        sub = _restricted_best(clip, length, lo, hi)
        if sub:
            a, b, s = sub
            notes.append(f"{clip.clip_id}: window restricted to "
                         f"{lo:.1f}-{hi:.1f}s by a revision constraint")
            return a, b, s, notes

    if avoid:
        got = best_window_avoiding(clip, length, avoid)
        if got is None:
            notes.append(f"{clip.clip_id}: no {length:.1f}s left that does "
                         f"not repeat a moment already used")
            return 0.0, 0.0, 0.0, notes          # caller drops this segment
        a, b, s = got
    else:
        a, b, s = best_window(clip, length)

    if min_mean_ydif is not None:
        got = _mean_ydif_over(clip, a, b)
        if got < min_mean_ydif:
            alt = _highest_motion_window(clip, length)
            if alt and _mean_ydif_over(clip, alt[0], alt[1]) > got:
                a, b, s = alt
                got = _mean_ydif_over(clip, a, b)
                notes.append(
                    f"{clip.clip_id}: re-searched for the liveliest window to "
                    f"satisfy a minimum-motion constraint of {min_mean_ydif:.2f}")
            if got < min_mean_ydif:
                notes.append(
                    f"{clip.clip_id}: CONSTRAINT UNMET - best window reaches "
                    f"only {got:.2f} mean motion against a required "
                    f"{min_mean_ydif:.2f}; this clip does not contain a "
                    f"livelier stretch")
    return a, b, s, notes


def _restricted_best(clip: ClipAnalysis, length: float, lo: float, hi: float):
    """best_window over a sub-range, by measuring the clip's own series."""
    dur = clip.technical.duration
    lo = max(0.0, min(lo, dur))
    hi = max(lo, min(hi, dur))
    if hi - lo < length:
        return None
    best = None
    t = lo
    while t + length <= hi + 1e-6:
        s = window_score(clip, t, t + length)
        if best is None or s > best[2]:
            best = (round(t, 2), round(t + length, 2), round(s, 3))
        t += 0.2
    return best


def _overlaps(a: float, b: float, taken) -> bool:
    return any(a < hi and lo < b for lo, hi in taken)


def best_window_avoiding(clip: ClipAnalysis, length: float, taken):
    """The best window of `length` that does not overlap one already used.

    One clip used to yield exactly one segment, so two clips meant two
    shots and a single cut however "punchy" the brief asked for. Letting a
    clip appear more than once only helps if each appearance is a DIFFERENT
    moment - calling `best_window` again returns the same answer every
    time - so the ranges already spent are stepped over here.

    Returns None when nothing of this length is left, which the caller
    reports rather than papering over with a repeat of the same frames.
    """
    if not taken:
        return best_window(clip, length)
    dur = clip.technical.duration
    if length > dur:
        return None
    best = None
    t = 0.0
    while t + length <= dur + 1e-6:
        if not _overlaps(t, t + length, taken):
            s = window_score(clip, t, t + length)
            if best is None or s > best[2]:
                best = (round(t, 2), round(t + length, 2), round(s, 3))
        t += 0.2
    return best


def _highest_motion_window(clip: ClipAnalysis, length: float):
    """The window with the highest RAW mean motion, ignoring freeze penalty.

    `best_window` optimises a freeze-penalised score, which is the right
    default. When the critic's complaint is specifically 'the opening is not
    lively enough', raw motion is the thing being asked for.
    """
    rate = clip.motion.sample_fps
    vals = clip.motion.ydif_series
    if not vals or rate <= 0:
        return None
    dur = clip.technical.duration
    n = max(1, int(length * rate))
    best = None
    t = 0.0
    while t + length <= dur + 1e-6:
        i0 = int(t * rate)
        chunk = vals[i0:i0 + n]
        if chunk:
            m = sum(chunk) / len(chunk)
            if best is None or m > best[3]:
                best = (round(t, 2), round(t + length, 2),
                        round(window_score(clip, t, t + length), 3), m)
        t += 0.2
    return best[:3] if best else None


def _mean_ydif_over(clip: ClipAnalysis, a: float, b: float) -> float:
    rate = clip.motion.sample_fps
    vals = clip.motion.ydif_series
    if not vals or rate <= 0:
        return 0.0
    i0, i1 = int(a * rate), max(int(b * rate), int(a * rate) + 1)
    chunk = vals[i0:i1]
    return round(sum(chunk) / len(chunk), 3) if chunk else 0.0


def mean_ydif_over(clip: ClipAnalysis, a: float, b: float) -> float:
    return _mean_ydif_over(clip, a, b)


# ------------------------------------------------------------ allocation

def sqrt_weighted_budget(clips: list[ClipAnalysis],
                         target: float) -> dict[str, float]:
    """Share a runtime budget out by the square root of measured motion.

    This is build_video.py's allocation rule (stage_plan, the `weights` block).
    It is re-expressed here rather than called because `stage_plan` does
    rejection, allocation, window search, ordering and manifest writing in one
    function with no seam - but the rule itself is reproduced exactly, and the
    deterministic FALLBACK path below runs the real `stage_plan` subprocess, so
    the production planner remains the authority.

    The square root is what stops one frenetic clip eating the whole reel while
    still letting livelier shots earn more screen time.
    """
    if not clips:
        return {}
    weights = {c.clip_id: max(c.motion.mean_ydif, 0.25) ** 0.5 for c in clips}
    wsum = sum(weights.values()) or 1.0
    lengths: dict[str, float] = {}
    for c in clips:
        want = target * weights[c.clip_id] / wsum
        lengths[c.clip_id] = max(build_video.MIN_SEG,
                                 min(build_video.MAX_SEG, want,
                                     c.technical.duration))
    # grow whatever has room if the first pass undershoots (same as stage_plan)
    for _ in range(80):
        totalled = sum(lengths.values())
        if totalled >= target - 0.5:
            break
        room = [c for c in clips
                if lengths[c.clip_id] < min(build_video.MAX_SEG,
                                            c.technical.duration) - 0.05]
        if not room:
            break
        add = min(0.5, (target - totalled) / len(room))
        for c in room:
            lengths[c.clip_id] = min(build_video.MAX_SEG, c.technical.duration,
                                     lengths[c.clip_id] + add)
    return {k: round(v, 2) for k, v in lengths.items()}


def candidate_windows(clip: ClipAnalysis, lengths: list[float] | None = None,
                      top_n: int = 4) -> list[dict]:
    """Measured windows to OFFER a model, so it never invents a timestamp.

    Every window here comes out of the same scan the planner uses: stepped
    0.2s at a time over the clip's real YDIF series, scored with the freeze
    penalty. The model is shown this list and may only return an INDEX into
    it - which is what makes "the LLM decides what, deterministic code decides
    where" enforceable rather than aspirational.

    Windows are offered at several lengths so a model choosing a hook is not
    forced into one duration, and near-duplicates are dropped so the list
    stays short enough to put in a prompt.
    """
    dur = clip.technical.duration
    if dur <= 0 or not clip.motion.ydif_series:
        return []

    if lengths is None:
        lengths = [l for l in (3.0, 5.0, 8.0, build_video.MAX_SEG)
                   if l <= dur + 0.01] or [min(dur, build_video.MIN_SEG)]

    found: list[dict] = []
    for length in lengths:
        length = min(length, dur)
        if length < 0.5:
            continue
        a, b, score = best_window(clip, length)
        if any(abs(w["start"] - a) < 0.3 and abs(w["end"] - b) < 0.3
               for w in found):
            continue                      # near-duplicate of one we have
        overlap = 0.0
        for fa, fb in clip.motion.freeze_runs:
            overlap += max(0.0, min(b, fb) - max(a, fa))
        found.append({
            "start": round(a, 2), "end": round(b, 2),
            "duration": round(b - a, 2),
            "mean_ydif": _mean_ydif_over(clip, a, b),
            "score": round(score, 3),
            "freeze_overlap": round(overlap, 2),
        })

    found.sort(key=lambda w: -w["score"])
    for i, w in enumerate(found[:top_n]):
        w["index"] = i
    return found[:top_n]


def total_with_crossfades(durations: list[float], xfade: float) -> float:
    """Finished length once crossfades overlap. Same arithmetic as stage_join."""
    if not durations:
        return 0.0
    return round(sum(durations) - (len(durations) - 1) * xfade, 2)


# ------------------------------------------------------ deterministic plan

# A shot long enough to read, short enough that the top-up looks edited
# rather than like one clip left running.
TOP_UP_SHOT = 6.0
# Below this there is nothing worth cutting to.
TOP_UP_MIN = 2.0


def top_up_to_target(clip_set: ClipAnalysisSet, segments: list[Segment],
                     target_seconds: float | None) -> tuple[list[Segment],
                                                            list[str]]:
    """Add more moments until the cut is as long as the person asked for.

    THE BUG THIS FIXES
    ------------------
    `pipeline/build_video.py` gives each clip ONE window and caps it at
    `MAX_SEG = 12.0`, so a two-clip job can never exceed 24 seconds
    however long a video was asked for. Every degraded run of a 30-second
    brief came out at 23.6s, and a 25-second brief came out at 23.6s too -
    the same number, because the target was never what decided it.

    The reasoned planner does not have this problem: it takes several
    moments from one clip (30s asked, 28.0s delivered over six cuts). So
    the shortfall is not a shortage of footage, it is the fallback asking
    for one window per clip. This takes more windows from the same clips,
    stepping over what is already spent, exactly as the reasoned path
    does.

    Deliberately NOT done by raising `MAX_SEG`: one 15-second shot is a
    clip left running, not an edit, and the cap is in a protected file.

    Returns the lengthened list and a note for each addition, so a
    degraded plan still says exactly how it reached its length.
    """
    if not target_seconds or not segments:
        return segments, []
    have = sum(s.duration for s in segments)
    short = target_seconds - have
    if short < TOP_UP_MIN:
        return segments, []

    by_id = {c.clip_id: c for c in clip_set.clips}
    taken: dict[str, list[tuple[float, float]]] = {}
    for s in segments:
        taken.setdefault(s.clip_id, []).append((s.start, s.end))

    # richest footage first, so the added moments are the best left
    order = sorted((c for c in clip_set.clips if c.clip_id in by_id),
                   key=lambda c: -(c.motion.mean_ydif or 0.0))
    added: list[Segment] = []
    notes: list[str] = []
    progress = True
    while short >= TOP_UP_MIN and progress:
        progress = False
        for clip in order:
            if short < TOP_UP_MIN:
                break
            length = round(min(TOP_UP_SHOT, short), 2)
            spot = best_window_avoiding(clip, length,
                                        taken.get(clip.clip_id, []))
            if spot is None:
                continue                  # this clip has nothing left
            st, en, sc = spot
            taken.setdefault(clip.clip_id, []).append((st, en))
            added.append(Segment(
                index=1,                  # renumbered below
                clip_id=clip.clip_id, start=st, end=en,
                duration=round(en - st, 2), score=sc,
                orientation=clip_set.orientation, caption="",
                reason=(f"added to reach the {target_seconds:g}s asked for; "
                        f"best unused window in this clip")))
            short = round(short - (en - st), 2)
            progress = True

    if not added:
        return segments, [
            f"{short:.1f}s short of the {target_seconds:g}s asked for, and "
            f"no unused window of {TOP_UP_MIN:g}s or more is left in any clip"]

    # Keep each clip where the planner put it, and each clip's own moments
    # in the order they happen - a cut that jumps backwards inside one clip
    # reads as a mistake.
    first_seen = {s.clip_id: i for i, s in enumerate(reversed(segments))}
    merged = sorted(segments + added,
                    key=lambda s: (-first_seen.get(s.clip_id, 0), s.start))
    notes.append(f"{len(added)} moment(s) added to reach "
                 f"{sum(s.duration for s in merged):.1f}s of the "
                 f"{target_seconds:g}s asked for")
    return ([s.model_copy(update={"index": i + 1})
             for i, s in enumerate(merged)], notes)


def deterministic_plan(build_dir: str, clip_set: ClipAnalysisSet,
                       target_seconds: float | None = None,
                       sequence: str = "hook",
                       on_line=None, *, pool=None,
                       job_id: str | None = None) -> list[Segment]:
    """THE FALLBACK: run the real `plan` stage and read back its segments.

    This is the production planner - the one that produced every video the
    system has shipped - not a reimplementation of it. When the LLM fails,
    returns nonsense, or names a clip that does not exist, this is what runs,
    and the artifact is marked degraded so nobody mistakes it for reasoning.
    """
    args = ["--sequence", sequence]
    if target_seconds:
        args += ["--target", str(float(target_seconds))]
    rc, out = run_pipeline_stage("plan", build_dir, extra_args=args,
                                 on_line=on_line, pool=pool,
                                 job_id=job_id)
    if rc != 0:
        raise RuntimeError(f"the deterministic planner failed (exit {rc}): "
                           f"{out[-500:]}")
    m = read_manifest(build_dir) or {}
    segs = m.get("segments") or []
    if not segs:
        raise RuntimeError("the deterministic planner selected no segments")
    return [
        Segment(
            index=int(s["index"]),
            clip_id=s["clip"],
            start=float(s["start"]),
            end=float(s["end"]),
            duration=float(s["duration"]),
            score=float(s.get("score") or 0.0),
            orientation=s.get("orientation") or clip_set.orientation,
            caption="",
            reason=s.get("reason") or "best-scoring window",
        )
        for s in segs
    ]


__all__ = ["window_score", "best_window", "clamp_length", "resolve_window",
           "candidate_windows",
           "sqrt_weighted_budget", "total_with_crossfades",
           "deterministic_plan", "mean_ydif_over"]
