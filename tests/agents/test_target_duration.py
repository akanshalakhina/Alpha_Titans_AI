"""A degraded plan must still be as long as the person asked for.

THE BUG THIS PINS
-----------------
`pipeline/build_video.py` gives each clip ONE window and caps it at
`MAX_SEG = 12.0`. With two clips that is 24 seconds, so every degraded run
answered a 30-second brief with 23.6s - and a 25-second brief with 23.6s
too, the same number, because the target never entered the arithmetic.

It is not a shortage of footage. The reasoned planner takes several
moments from one clip and hits the mark: 30s asked, 28.0s delivered over
six cuts; 35s asked, 33.0s over six. Only the fallback is one-window-
per-clip, and `MAX_SEG` lives in a protected file.

So the fallback now takes further windows from the same clips, stepping
over what it has already spent - the same `best_window_avoiding` the
reasoned path uses.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import (ClipAnalysis, ClipAnalysisSet,  # noqa: E402
                              ClipMotion, ClipTechnical,
                              ClipUsability, Segment)
from agents.tools import select_tools as ST                 # noqa: E402


def _seg(i, clip, start, end):
    return Segment(index=i, clip_id=clip, start=start, end=end,
                   duration=round(end - start, 2), score=1.0,
                   orientation="horizontal")


# Real ClipAnalysis objects: `best_window_avoiding` hands the clip to
# `build_video.best_window` through `to_planner_dict()`, so a stand-in
# would only prove the stand-in works.
def _clip(cid: str, dur: float, bias: float = 0.0) -> ClipAnalysis:
    n = int(dur * 10)
    return ClipAnalysis(
        clip_id=cid, path=f"/tmp/{cid}",
        technical=ClipTechnical(duration=dur, coded=(1920, 1080),
                                display=(1920, 1080), fps=30.0,
                                orientation="horizontal"),
        motion=ClipMotion(sample_fps=10.0, mean_ydif=3.0 + bias,
                          max_ydif=9.0, live_share=1.0, freeze_runs=[],
                          ydif_series=[1.0 + (i % 50) / 10.0
                                       for i in range(n)]),
        usability=ClipUsability(usable=True))


@pytest.fixture
def two_clips():
    """One long clip and one short, as in the footage this was found on."""
    return ClipAnalysisSet(
        clips=[_clip("long.mov", 60.0, bias=1.0), _clip("short.mov", 12.0)],
        orientation="horizontal", canvas=(1920, 1080), target_seconds=30.0)


# --------------------------------------------------------- the shortfall

def test_a_short_plan_is_lengthened_to_the_target(two_clips):
    plan = [_seg(1, "long.mov", 0.0, 12.0), _seg(2, "short.mov", 0.0, 12.0)]
    out, notes = ST.top_up_to_target(two_clips, plan, 30.0)

    total = sum(s.duration for s in out)
    assert total >= 30.0 - ST.TOP_UP_MIN, f"still only {total}s"
    assert len(out) > 2, "the extra length must come from extra cuts"
    assert notes and "added" in notes[0]


def test_the_added_moments_are_new_footage(two_clips):
    plan = [_seg(1, "long.mov", 0.0, 12.0), _seg(2, "short.mov", 0.0, 12.0)]
    out, _ = ST.top_up_to_target(two_clips, plan, 30.0)

    spans = {}
    for s in out:
        for a, b in spans.setdefault(s.clip_id, []):
            assert s.end <= a + 1e-6 or s.start >= b - 1e-6, \
                f"{s.clip_id} {s.start}-{s.end} repeats {a}-{b}"
        spans[s.clip_id].append((s.start, s.end))


def test_no_added_shot_is_longer_than_one_reads_as_a_cut(two_clips):
    plan = [_seg(1, "long.mov", 0.0, 12.0)]
    out, _ = ST.top_up_to_target(two_clips, plan, 40.0)
    for s in out[1:]:
        assert s.duration <= ST.TOP_UP_SHOT + 1e-6


def test_each_clips_moments_stay_in_the_order_they_happen(two_clips):
    plan = [_seg(1, "long.mov", 30.0, 42.0), _seg(2, "short.mov", 0.0, 6.0)]
    out, _ = ST.top_up_to_target(two_clips, plan, 34.0)
    for cid in {s.clip_id for s in out}:
        starts = [s.start for s in out if s.clip_id == cid]
        assert starts == sorted(starts), f"{cid} jumps backwards"


def test_the_clips_keep_the_order_the_planner_chose(two_clips):
    plan = [_seg(1, "short.mov", 0.0, 6.0), _seg(2, "long.mov", 0.0, 12.0)]
    out, _ = ST.top_up_to_target(two_clips, plan, 34.0)
    order, seen = [], set()
    for s in out:
        if s.clip_id not in seen:
            seen.add(s.clip_id)
            order.append(s.clip_id)
    assert order == ["short.mov", "long.mov"], "the hook must stay first"


def test_segments_are_renumbered_from_one(two_clips):
    plan = [_seg(1, "long.mov", 0.0, 12.0), _seg(2, "short.mov", 0.0, 12.0)]
    out, _ = ST.top_up_to_target(two_clips, plan, 30.0)
    assert [s.index for s in out] == list(range(1, len(out) + 1))


# ------------------------------------------------------- left well alone

def test_a_plan_already_long_enough_is_untouched(two_clips):
    plan = [_seg(1, "long.mov", 0.0, 12.0), _seg(2, "long.mov", 20.0, 38.0)]
    out, notes = ST.top_up_to_target(two_clips, plan, 30.0)
    assert out == plan and notes == []


def test_no_target_means_nothing_to_aim_at(two_clips):
    plan = [_seg(1, "long.mov", 0.0, 12.0)]
    assert ST.top_up_to_target(two_clips, plan, None) == (plan, [])


def test_an_empty_plan_is_not_invented_from_nothing(two_clips):
    assert ST.top_up_to_target(two_clips, [], 30.0) == ([], [])


def test_footage_that_cannot_reach_the_target_says_so(two_clips):
    """Two clips of 60s and 12s cannot make five minutes. The shortfall is
    reported rather than padded with repeated frames."""
    plan = [_seg(1, "long.mov", 0.0, 12.0), _seg(2, "short.mov", 0.0, 12.0)]
    out, notes = ST.top_up_to_target(two_clips, plan, 300.0)

    total = sum(s.duration for s in out)
    assert total < 300.0
    assert notes, "silence here would look like the target was met"
    spans = {}
    for s in out:
        for a, b in spans.setdefault(s.clip_id, []):
            assert s.end <= a + 1e-6 or s.start >= b - 1e-6, "padded with a repeat"
        spans[s.clip_id].append((s.start, s.end))
