"""The Pattern Miner: lessons from the videos that exist, kept honest.

What must hold:
* a lesson compares groups of at least two videos, never one;
* a difference that one video made is `fragile` and cannot be approved;
* the label says how big the sample is;
* seconds watched are never compared across Shorts and regular videos;
* nothing reaches the planning agents until a person approves it, and a
  decision survives the next run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

from datalayers import layers as L              # noqa: E402
from datalayers import lessons as LS            # noqa: E402


def video(vid, title, views, hold, hook="statement", topic="lego"):
    return (vid, title, "2026-09-07", 1, views, hold, 1.0, 0, 0, 0.0, 1, 1,
            hook, topic, None, None)


@pytest.fixture
def db(tmp_path):
    db = L.connect(tmp_path / "l.db")
    yield db
    db.close()


def fill(db, rows):
    db.execute("DELETE FROM analytics_video_performance")
    db.executemany("INSERT INTO analytics_video_performance VALUES "
                   "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    db.commit()


def channel():
    return [video("a", "Build | Alpha Titans Shorts", 1200, 6),
            video("b", "Build | Alpha Titans Shorts", 300, 7),
            video("c", "Build | Alpha Titans Shorts", 200, 5),
            video("d", "Build | Alpha Titans", 150, 25),
            video("e", "Build | Alpha Titans", 100, 30),
            video("f", "Build | Alpha Titans", 90, 22)]


def test_a_stable_difference_becomes_a_proposed_lesson(db):
    fill(db, channel())
    LS.mine(db)
    got = LS.listing(db)
    fmt = [l for l in got if l["dimension"] == "format"]
    assert len(fmt) == 1
    l = fmt[0]
    assert (l["better"], l["worse"]) == ("Shorts", "regular videos")
    assert l["stable"] == 1 and l["status"] == "proposed"
    assert l["confidence"] == "early signal"          # 3 videos a group


def test_one_video_cannot_make_a_lesson_alone(db):
    rows = channel()[:3] + [video("z", "Build | Alpha Titans", 99, 9)]
    fill(db, rows)
    LS.mine(db)
    assert not [l for l in LS.listing(db) if l["dimension"] == "format"]


def test_a_difference_one_video_made_is_fragile_and_not_approvable(db):
    rows = [video("a", "x Shorts", 5000, 6), video("b", "x Shorts", 10, 6),
            video("c", "x Shorts", 10, 6), video("d", "x", 100, 9),
            video("e", "x", 100, 9), video("f", "x", 100, 9)]
    fill(db, rows)
    LS.mine(db)
    l = next(l for l in LS.listing(db) if l["dimension"] == "format")
    assert l["confidence"] == "fragile" and not l["stable"]
    assert "one video made it" in l["text"]
    assert LS.decide(db, l["id"], approve=True) is False


def test_seconds_watched_are_never_compared_across_formats(db):
    fill(db, channel())
    LS.mine(db)
    assert not [l for l in LS.listing(db) if l["dimension"] == "format"
                and l["metric"] == "day7_avg_view_s"]


def test_hooks_are_compared_on_both_views_and_seconds(db):
    rows = [video("a", "t", 500, 20, hook="question"),
            video("b", "t", 400, 18, hook="question"),
            video("c", "t", 100, 5, hook="statement"),
            video("d", "t", 90, 6, hook="statement")]
    fill(db, rows)
    LS.mine(db)
    hooks = {l["metric"] for l in LS.listing(db)
             if l["dimension"] == "hook_type"}
    assert hooks == {"day7_views", "day7_avg_view_s"}


@pytest.mark.parametrize("n,want", [(2, "early signal"), (4, "early signal"),
                                    (5, "medium"), (11, "medium"),
                                    (12, "strong")])
def test_the_label_is_the_sample_size(n, want):
    assert LS.label(n) == want


def test_only_approved_lessons_reach_the_planners(db):
    fill(db, channel())
    LS.mine(db)
    assert LS.guidance(db) == ""
    lid = LS.listing(db)[0]["id"]
    assert LS.decide(db, lid, approve=True)
    g = LS.guidance(db)
    assert "approved by a person" in g and "Shorts averaged" in g
    assert "[early signal]" in g
    assert LS.decide(db, lid, approve=False)
    assert LS.guidance(db) == ""


def test_a_decision_survives_the_next_run(db):
    fill(db, channel())
    LS.mine(db)
    lid = LS.listing(db)[0]["id"]
    LS.decide(db, lid, approve=True, who="Akansha")
    LS.mine(db)
    l = next(l for l in LS.listing(db) if l["id"] == lid)
    assert l["status"] == "approved" and l["decided_by"] == "Akansha"


def test_an_approved_lesson_that_stops_holding_is_asked_again(db):
    fill(db, channel())
    LS.mine(db)
    lid = LS.listing(db)[0]["id"]
    LS.decide(db, lid, approve=True)
    fill(db, [video("a", "x Shorts", 5000, 6), video("b", "x Shorts", 10, 6),
              video("c", "x Shorts", 10, 6), video("d", "x", 100, 9),
              video("e", "x", 100, 9), video("f", "x", 100, 9)])
    LS.mine(db)
    l = next(l for l in LS.listing(db) if l["id"] == lid)
    assert l["status"] == "proposed" and l["confidence"] == "fragile"
    assert LS.guidance(db) == ""


def test_the_learning_agent_hands_approved_lessons_on(db, tmp_path,
                                                      monkeypatch):
    from agents import learning as LN
    from datalayers import run as R
    fill(db, channel())
    LS.mine(db)
    LS.decide(db, LS.listing(db)[0]["id"], approve=True)
    db.close()
    monkeypatch.setattr(R, "DB_PATH", tmp_path / "l.db")
    monkeypatch.setenv("AT_CHANNEL_LESSONS", "1")
    assert "Shorts averaged" in LN.channel_guidance()
    monkeypatch.setenv("AT_CHANNEL_LESSONS", "0")
    assert LN.channel_guidance() == ""
