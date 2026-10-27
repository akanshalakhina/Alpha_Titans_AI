"""Pattern Miner: lessons from the channel's own videos, as many as exist.

WHY A SECOND WAY OF LEARNING
----------------------------
The Learning agent compares the DECISIONS behind our videos (pacing, shots)
with their results, and only once 30 linked videos exist. The channel's
videos are not linked to the runs that made them, so it has nothing to
compare - however low its bar is set. Yet the warehouse already says a lot
about each video: Short or regular, the hook its title uses, its topic.
This module learns from those, on the videos that exist today.

HOW A SMALL SAMPLE IS KEPT HONEST
---------------------------------
* A group needs at least `MIN_GROUP` videos (2) - one video is an anecdote.
* STABILITY: the difference must keep its direction when the single best
  video is removed from EACH group. In this channel one Short has a third of
  all Short views (1,181 of 3,570); a pattern that disappears without it was
  that video, not a pattern. Such a lesson is `fragile` and never offered.
* Every lesson carries its size as a label - early signal under 5 videos a
  group, medium under 12, strong at 12 or more - and the label goes into the
  prompt with it, so an agent weighs it accordingly.
* Seconds watched are not compared across Shorts and regular videos: a Short
  is shorter, so fewer seconds is not a worse video.
* Nothing is used until a person approves it. Approved lessons reach the
  Content Strategy and Editing Director agents as EVIDENCE, not orders.
"""

from __future__ import annotations

import hashlib
import os
import time

MIN_GROUP = int(os.environ.get("AT_LESSON_MIN_GROUP", "2") or 2)
METRICS = (("day7_views", "first-week views", ""),
           ("day7_avg_view_s", "average seconds watched", "s"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
  id TEXT PRIMARY KEY, dimension TEXT, metric TEXT, better TEXT, worse TEXT,
  better_mean REAL, worse_mean REAL, better_n INTEGER, worse_n INTEGER,
  lift REAL, stable INTEGER, confidence TEXT, text TEXT, created_at REAL,
  status TEXT, decided_at REAL, decided_by TEXT);
"""


def _fmt(v: float, unit: str) -> str:
    return f"{v:.1f}{unit}" if unit else f"{v:,.0f}"


def label(n: int) -> str:
    return "strong" if n >= 12 else "medium" if n >= 5 else "early signal"


def _groups(rows, dim):
    out: dict[str, list] = {}
    for r in rows:
        if dim == "format":
            key = "Shorts" if "short" in (r["title"] or "").lower() else \
                  "regular videos"
        else:
            key = (r[dim] or "").strip()
        if key:
            out.setdefault(key, []).append(r)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _without_top(vals):
    return sorted(vals)[:-1]


def compare(rows, dim: str, metric: str, unit: str, name: str) -> list[dict]:
    """Every pair of groups on one dimension: which did better, by how
    much, on how many videos, and whether it survives losing its best."""
    groups = {k: v for k, v in _groups(rows, dim).items()
              if len(v) >= MIN_GROUP}
    keys = sorted(groups)
    found = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va = [float(r[metric] or 0) for r in groups[a]]
            vb = [float(r[metric] or 0) for r in groups[b]]
            ma, mb = _mean(va), _mean(vb)
            if ma == mb:
                continue
            hi, lo = (a, b) if ma > mb else (b, a)
            vhi, vlo = (va, vb) if ma > mb else (vb, va)
            mhi, mlo = max(ma, mb), min(ma, mb)
            stable = _mean(_without_top(vhi)) > _mean(_without_top(vlo))
            n = min(len(vhi), len(vlo))
            lift = mhi / mlo if mlo else None
            conf = label(n) if stable else "fragile"
            text = (f"{hi} averaged {_fmt(mhi, unit)} {name} against "
                    f"{_fmt(mlo, unit)} for {lo}"
                    + (f" ({lift:.1f}x)" if lift else "")
                    + f", over {len(vhi)} and {len(vlo)} videos. "
                    + ("It still holds with each group's best video "
                       "removed." if stable else
                       "It does NOT hold once each group's best video is "
                       "removed - one video made it."))
            lid = hashlib.sha1(f"{dim}|{metric}|{hi}|{lo}".encode()
                               ).hexdigest()[:10]
            found.append({"id": lid, "dimension": dim, "metric": metric,
                          "better": hi, "worse": lo, "better_mean": mhi,
                          "worse_mean": mlo, "better_n": len(vhi),
                          "worse_n": len(vlo), "lift": lift,
                          "stable": stable, "confidence": conf,
                          "text": text})
    return found


def mine(db) -> list[dict]:
    """Pattern Miner's step: find every lesson the analytics layer
    supports. A decision already taken on a lesson (approved, rejected)
    is kept when the same lesson is found again; its numbers refresh."""
    db.executescript(SCHEMA)
    rows = db.execute("SELECT * FROM analytics_video_performance "
                      "WHERE mature=1").fetchall()
    found = []
    for dim in ("format", "hook_type", "topic"):
        for metric, name, unit in METRICS:
            if dim == "format" and metric == "day7_avg_view_s":
                continue            # a Short is shorter; not a fair contest
            found += compare(rows, dim, metric, unit, name)
    now = time.time()
    keep = {r["id"]: r for r in db.execute(
        "SELECT id, status, decided_at, decided_by FROM lessons")}
    db.execute("DELETE FROM lessons")
    for f in found:
        old = keep.get(f["id"])
        status = old["status"] if old and old["status"] in (
            "approved", "rejected") else "proposed"
        if not f["stable"] and status == "approved":
            status = "proposed"     # it no longer holds: ask again
        db.execute(
            "INSERT INTO lessons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f["id"], f["dimension"], f["metric"], f["better"], f["worse"],
             f["better_mean"], f["worse_mean"], f["better_n"], f["worse_n"],
             f["lift"], int(f["stable"]), f["confidence"], f["text"], now,
             status, old["decided_at"] if old else None,
             old["decided_by"] if old else None))
    db.commit()
    stable = sum(1 for f in found if f["stable"])
    return [{"name": "lessons_found", "ok": True,
             "detail": f"{len(found)} comparison(s); {stable} hold without "
                       f"their best video, {len(found) - stable} fragile"},
            {"name": "nothing_used_without_approval", "ok": True,
             "detail": "every new lesson starts as proposed"}]


def decide(db, lesson_id: str, approve: bool, who: str = "operator") -> bool:
    db.executescript(SCHEMA)
    r = db.execute("SELECT stable FROM lessons WHERE id=?",
                   (lesson_id,)).fetchone()
    if r is None or (approve and not r["stable"]):
        return False                # a fragile lesson cannot be approved
    db.execute("UPDATE lessons SET status=?, decided_at=?, decided_by=? "
               "WHERE id=?", ("approved" if approve else "rejected",
                              time.time(), who[:60], lesson_id))
    db.commit()
    return True


def listing(db) -> list[dict]:
    db.executescript(SCHEMA)
    order = {"strong": 0, "medium": 1, "early signal": 2, "fragile": 3}
    rows = [dict(r) for r in db.execute("SELECT * FROM lessons")]
    return sorted(rows, key=lambda r: (order.get(r["confidence"], 9),
                                       -(r["lift"] or 0)))


def guidance(db) -> str:
    """Approved lessons as a prompt section, or "" when there are none."""
    rows = [r for r in listing(db) if r["status"] == "approved"]
    if not rows:
        return ""
    lines = "\n".join(f"  - [{r['confidence']}] {r['text']}" for r in rows)
    return ("LESSONS FROM THIS CHANNEL'S OWN VIDEOS (approved by a person)\n"
            f"{lines}\n"
            "  Small samples: weigh each by its label. These describe past "
            "videos; use them\n  where they fit this footage, and say so in "
            "`limitations` when they do not.\n")


__all__ = ["MIN_GROUP", "compare", "decide", "guidance", "label", "listing",
           "mine"]
