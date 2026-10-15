"""One clip can carry more than one cut.

THE LIMIT THIS REMOVES
----------------------
`stage_plan` takes exactly one window per clip:

    for c in usable:
        a, b, score = best_window(c, lengths[c["name"]])

and the EditingDirector's prompt said "Use each clip at most once", with the
validator refusing a repeat. So two uploads meant two shots and ONE cut, no
matter how punchy the brief asked for - "fast and punchy" produced a single
crossfade in fifteen seconds, and the Transition, Caption and Story agents
each had one junction and two beats to work with.

A repeat only helps if it is a DIFFERENT moment, and `best_window` returns
the same answer every time it is asked. `best_window_avoiding` steps over
the stretches already spent, and the director hands them back as it goes.
When nothing of that length is left, the use is DROPPED with a note - the
same frames never appear twice.

The deterministic fallback still runs the production planner unchanged, so
a build with no model behaves exactly as it always did.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import (ClipAnalysis, ClipMotion,     # noqa: E402
                              ClipTechnical, ClipUsability)
from agents.tools import select_tools as ST                # noqa: E402


# A real ClipAnalysis, because `best_window` hands the clip to
# `build_video.best_window` through `to_planner_dict()` - a stand-in would
# only prove the stand-in works.
def a_clip(dur=60.0, cid="c.mp4") -> ClipAnalysis:
    n = int(dur * 10)
    return ClipAnalysis(
        clip_id=cid, path=f"/tmp/{cid}",
        technical=ClipTechnical(duration=dur, coded=(1920, 1080),
                                display=(1920, 1080), fps=30.0,
                                orientation="horizontal"),
        motion=ClipMotion(sample_fps=10.0, mean_ydif=3.0, max_ydif=6.0,
                          live_share=1.0, freeze_runs=[],
                          ydif_series=[1.0 + (i % 50) / 10.0
                                       for i in range(n)]),
        usability=ClipUsability(usable=True),
    )


def _score(clip, a, b):
    return ST.window_score(clip, a, b)


# ------------------------------------------------- picking several windows

def test_a_second_window_is_a_different_stretch():
    clip = a_clip(60.0)
    first = ST.best_window_avoiding(clip, 5.0, [])
    second = ST.best_window_avoiding(clip, 5.0, [first[:2]])
    assert second is not None
    assert (second[0], second[1]) != (first[0], first[1])


def test_windows_never_overlap_each_other():
    clip = a_clip(60.0)
    taken = []
    for _ in range(5):
        got = ST.best_window_avoiding(clip, 5.0, taken)
        assert got is not None
        a, b = got[0], got[1]
        for lo, hi in taken:
            assert a >= hi or b <= lo, f"{(a, b)} overlaps {(lo, hi)}"
        taken.append((a, b))
    assert len(taken) == 5


def test_it_runs_out_honestly_rather_than_repeating():
    """A clip cannot carry unlimited cuts, and must say so instead of
    handing back a stretch it has already given.

    How many fit depends on where the best ones land - a 5s window taken
    from the middle of a 12s clip leaves two gaps too short to use - so the
    test asserts the PROPERTY, not a count: it keeps asking until it is
    refused, and every answer along the way is new ground."""
    clip = a_clip(12.0)
    taken = []
    for _ in range(10):
        got = ST.best_window_avoiding(clip, 5.0, taken)
        if got is None:
            break
        a, b = got[0], got[1]
        for lo, hi in taken:
            assert a >= hi or b <= lo, f"{(a, b)} repeats {(lo, hi)}"
        taken.append((a, b))
    else:
        raise AssertionError("it never ran out - it must be repeating")
    assert taken, "at least the first window should have been found"


def test_no_avoid_list_is_the_old_behaviour():
    clip = a_clip(60.0)
    assert ST.best_window_avoiding(clip, 5.0, []) == ST.best_window(clip, 5.0)
    assert ST.best_window_avoiding(clip, 5.0, None) == ST.best_window(clip, 5.0)


def test_a_window_longer_than_the_clip_is_impossible():
    assert ST.best_window_avoiding(a_clip(4.0), 9.0, [(0, 1)]) is None


# ------------------------------------------------------- resolve_window

def test_resolve_window_reports_when_nothing_is_left():
    clip = a_clip(12.0)
    a, b, _s, notes = ST.resolve_window(clip, 5.0, avoid=[(0.0, 12.0)])
    assert (a, b) == (0.0, 0.0), "an empty span tells the caller to drop it"
    assert any("not repeat a moment already used" in n for n in notes)


def test_resolve_window_without_avoid_is_untouched():
    clip = a_clip(60.0)
    a, b, _s, _n = ST.resolve_window(clip, 5.0)
    assert b > a


# --------------------------------------------- the director's own rules

def test_the_prompt_now_allows_a_repeat():
    import agents.editing_director as ED
    assert "at most once" not in ED.PROMPT
    assert "MAY be used more than once" in ED.PROMPT


def test_the_validator_no_longer_refuses_a_repeat():
    import inspect
    import agents.editing_director as ED
    src = inspect.getsource(ED)
    assert "appears more than once" not in src
