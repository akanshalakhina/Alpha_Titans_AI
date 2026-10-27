"""The Data Layers team: source -> staging -> AI -> analytics.

WHAT EACH LAYER IS FOR
----------------------
  source     the warehouse rows EXACTLY as they arrived, plus when and how
             they were fetched. Append-only: a bug further down is fixed by
             rebuilding from here, never by editing here.
  staging    the same facts made trustworthy: one row per video per day,
             duplicates removed, dates as dates, and every video's numbers
             over the SAME window - its first seven days - so a video
             published on the 7th is not compared with one published on the
             16th on unequal terms. (`SUM(views)` over all time, the
             manager's first query, always favours the older video.)
  ai         meaning a model adds: the hook a title uses, its topic. Every
             row says who tagged it - a model by name, or the rules.
  analytics  one row per video, ready to answer questions, and the file the
             Performance and Learning agents read.

Each layer has a CHECKER. When it fails, the layers after it do not run:
a staging table built on a broken source would look fine and be wrong.

WHERE IT RUNS
-------------
Today on a workstation, signed in with gcloud; later on the VM, from cron,
as the VM's service account. The code is the same; only the identity
changes. Storage is one SQLite file - the standard library's - until a
BigQuery dataset of our own is granted, at which point the same tables move
there.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

SOURCE_TABLE = "myukti.d_content_analytics.youtube_video_daily"
LOCATION = "asia-south1"
DAY7 = 7                                      # the window every video gets

SOURCE_SQL = f"""
SELECT video_id, title, published_at, CAST(metric_date AS STRING) AS
       metric_date, views, likes, comments, shares, watch_time_minutes,
       avg_view_duration_seconds, subscribers_gained, loaded_at
FROM `{SOURCE_TABLE}`
"""

HOOK_TYPES = ("question", "how_to", "result_first", "list", "challenge",
              "statement")

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_batches (
  batch_id TEXT PRIMARY KEY, fetched_at REAL, origin TEXT, rows INTEGER,
  bytes_billed INTEGER, query TEXT);
CREATE TABLE IF NOT EXISTS source_youtube_video_daily (
  batch_id TEXT, video_id TEXT, title TEXT, published_at REAL,
  metric_date TEXT, views REAL, likes REAL, comments REAL, shares REAL,
  watch_time_minutes REAL, avg_view_duration_seconds REAL,
  subscribers_gained REAL, loaded_at REAL);
CREATE TABLE IF NOT EXISTS source_jobs (
  batch_id TEXT, job_id TEXT, video_id TEXT, decisions TEXT);

CREATE TABLE IF NOT EXISTS staging_video_daily (
  video_id TEXT, metric_date TEXT, title TEXT, published_date TEXT,
  views REAL, likes REAL, comments REAL, shares REAL, watch_minutes REAL,
  avg_view_s REAL, subs_gained REAL,
  PRIMARY KEY (video_id, metric_date));
CREATE TABLE IF NOT EXISTS staging_video_day7 (
  video_id TEXT PRIMARY KEY, title TEXT, published_date TEXT,
  first_day TEXT, last_day TEXT, days_tracked INTEGER, mature INTEGER,
  day7_views REAL, day7_watch_minutes REAL, day7_likes REAL,
  day7_comments REAL, day7_shares REAL, day7_subs REAL,
  day7_avg_view_s REAL, total_views REAL, total_watch_minutes REAL);

CREATE TABLE IF NOT EXISTS ai_video_tags (
  video_id TEXT PRIMARY KEY, title TEXT, hook_type TEXT, topic TEXT,
  tagged_by TEXT, tagged_at REAL);

CREATE TABLE IF NOT EXISTS analytics_video_performance (
  video_id TEXT PRIMARY KEY, title TEXT, published_date TEXT,
  mature INTEGER, day7_views REAL, day7_avg_view_s REAL,
  day7_watch_minutes REAL, day7_likes REAL, day7_subs REAL,
  like_rate REAL, rank_views INTEGER, rank_hold INTEGER, hook_type TEXT,
  topic TEXT, job_id TEXT, decisions TEXT);
CREATE TABLE IF NOT EXISTS analytics_groups (
  dimension TEXT, value TEXT, videos INTEGER, avg_day7_views REAL,
  avg_day7_view_s REAL, PRIMARY KEY (dimension, value));
CREATE TABLE IF NOT EXISTS layer_runs (
  run_id TEXT, layer TEXT, agent TEXT, started_at REAL, ended_at REAL,
  ok INTEGER, rows INTEGER, checks TEXT);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _day(ts) -> str | None:
    if ts is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc
                                         ).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _plus(day: str, n: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=n)).isoformat()


def _latest_batch(db) -> str | None:
    r = db.execute("SELECT batch_id FROM source_batches "
                   "ORDER BY fetched_at DESC LIMIT 1").fetchone()
    return r["batch_id"] if r else None


# ------------------------------------------------------------------ 1. source

COLUMNS = ("video_id", "title", "published_at", "metric_date", "views",
           "likes", "comments", "shares", "watch_time_minutes",
           "avg_view_duration_seconds", "subscribers_gained", "loaded_at")


def load_source(db, rows: list[dict], *, origin: str, bytes_billed: int = 0,
                jobs: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Source Loader: store the rows untouched, as a new batch."""
    batch = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    db.execute("INSERT INTO source_batches VALUES (?,?,?,?,?,?)",
               (batch, time.time(), origin, len(rows), bytes_billed,
                SOURCE_SQL.strip()))
    db.executemany(
        f"INSERT INTO source_youtube_video_daily VALUES "
        f"(?,{','.join('?' * len(COLUMNS))})",
        [(batch, *[r.get(c) for c in COLUMNS]) for r in rows])
    db.executemany("INSERT INTO source_jobs VALUES (?,?,?,?)",
                   [(batch, j["job_id"], j["video_id"],
                     json.dumps(j.get("decisions") or {}))
                    for j in jobs or []])
    db.commit()
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    keyless = sum(1 for r in rows if not r.get("video_id")
                  or not r.get("metric_date"))
    return batch, [
        _check("rows_arrived", len(rows) > 0, f"{len(rows)} row(s) fetched"),
        _check("columns_present", not missing,
               "all expected columns present" if not missing
               else "missing: " + ", ".join(missing)),
        _check("keys_present", keyless == 0,
               f"{keyless} row(s) without a video_id or date"),
    ]


# ---------------------------------------------------------------- 2. staging

def build_staging(db, batch: str) -> list[dict]:
    """Cleaner: one row per video per day, then each video's first week."""
    db.execute("DELETE FROM staging_video_daily")
    db.execute("DELETE FROM staging_video_day7")
    raw = db.execute(
        "SELECT * FROM source_youtube_video_daily WHERE batch_id=? "
        "AND video_id IS NOT NULL AND metric_date IS NOT NULL "
        "ORDER BY COALESCE(loaded_at, 0)", (batch,)).fetchall()
    latest: dict[tuple, sqlite3.Row] = {}
    for r in raw:                       # a later load of the same day wins
        latest[(r["video_id"], r["metric_date"])] = r
    # Likes, comments, shares and subscribers are NET changes per day: a
    # like taken back is -1 on the day it went (two such days in the real
    # table, 16 Sep 2026). They stay as they are - clamping them to zero
    # would overstate the totals. Views and watch time cannot go below
    # zero, so a negative one there is a broken row.
    impossible = 0
    for (vid, day), r in latest.items():
        nums = [float(r[c] or 0) for c in (
            "views", "likes", "comments", "shares", "watch_time_minutes",
            "subscribers_gained")]
        if nums[0] < 0 or nums[4] < 0:
            impossible += 1
        db.execute(
            "INSERT INTO staging_video_daily VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (vid, day, (r["title"] or "").strip(), _day(r["published_at"]),
             *nums[:5], float(r["avg_view_duration_seconds"] or 0),
             nums[5]))

    videos = db.execute(
        "SELECT video_id, MAX(title) title, MIN(published_date) pub, "
        "MIN(metric_date) first_day, MAX(metric_date) last_day, "
        "COUNT(*) days, SUM(views) tv, SUM(watch_minutes) tw "
        "FROM staging_video_daily GROUP BY video_id").fetchall()
    no_pub = 0
    for v in videos:
        pub = v["pub"]
        if not pub:
            no_pub += 1
            continue
        end = _plus(pub, DAY7 - 1)
        w = db.execute(
            "SELECT SUM(views) v, SUM(watch_minutes) wm, SUM(likes) l, "
            "SUM(comments) c, SUM(shares) s, SUM(subs_gained) sg "
            "FROM staging_video_daily WHERE video_id=? AND metric_date "
            "BETWEEN ? AND ?", (v["video_id"], pub, end)).fetchone()
        views, wm = w["v"] or 0.0, w["wm"] or 0.0
        db.execute(
            "INSERT INTO staging_video_day7 VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (v["video_id"], v["title"], pub, v["first_day"], v["last_day"],
             v["days"], int(v["last_day"] >= end), views, wm,
             w["l"] or 0.0, w["c"] or 0.0, w["s"] or 0.0, w["sg"] or 0.0,
             round(wm * 60.0 / views, 2) if views else 0.0,
             v["tv"] or 0.0, v["tw"] or 0.0))
    db.commit()
    n7 = db.execute("SELECT COUNT(*) FROM staging_video_day7").fetchone()[0]
    dupes = len(raw) - len(latest)
    return [
        _check("one_row_per_day", True,
               f"{len(latest)} day-row(s) kept; {dupes} duplicate(s) "
               f"resolved to the latest load"),
        _check("views_and_watch_time_not_negative", impossible == 0,
               f"{impossible} day-row(s) with negative views or watch time "
               f"(net likes/comments/subscribers may be negative)"),
        _check("every_video_has_a_publish_date", no_pub == 0,
               f"{no_pub} video(s) without a publish date"),
        _check("week_one_for_every_video", n7 == len(videos) - no_pub,
               f"{n7} of {len(videos)} video(s) have first-week numbers"),
    ]


# --------------------------------------------------------------------- 3. ai

TAG_PROMPT = """These are YouTube video titles from one channel. For each, \
say which kind of hook the title uses and its topic in two or three words.

hook_type is exactly one of: question (asks something), how_to (promises to \
show how), result_first (shows the finished thing or outcome), list (numbers \
or a ranking), challenge (a test, a dare, a time limit), statement (anything \
else).

Titles:
{titles}

Answer with JSON only: {{"videos": [{{"n": 1, "hook_type": "...", \
"topic": "..."}}, ...]}} - one entry per title, numbered as given."""


def rule_tag(title: str) -> tuple[str, str]:
    """What the words alone say, when no model is available."""
    t = (title or "").strip().lower()
    # "?" anywhere: titles end in the channel name, not the question
    if "?" in t or re.match(r"^(why|what|can|will|is|does|who)\b", t):
        return "question", ""
    if re.match(r"^how\b", t):
        return "how_to", ""
    if re.match(r"^(top\s+)?\d+\b", t):
        return "list", ""
    if re.search(r"\b(challenge|vs\.?|in \d+ (sec|min|hour))", t):
        return "challenge", ""
    return "statement", ""


def tag_titles(db, llm=None, now=None) -> list[dict]:
    """Enricher: hook type and topic for every video, each row signed."""
    vids = db.execute("SELECT video_id, title FROM staging_video_day7 "
                      "ORDER BY video_id").fetchall()
    tags: dict[str, tuple[str, str, str]] = {}
    note = "rules (no model available)"
    if llm is not None and vids and llm.available():
        listing = "\n".join(f"{i + 1}. {v['title'][:150]}"
                            for i, v in enumerate(vids))
        try:
            parsed, reply = llm.complete_json(
                TAG_PROMPT.format(titles=listing), max_tokens=1600)
            rows = (parsed or {}).get("videos") if isinstance(parsed, dict) \
                else None
            model = ((reply.meta or {}).get("model") if reply else "") or "llm"
            for r in rows or []:
                i = int(r.get("n", 0)) - 1
                h = str(r.get("hook_type", "")).strip().lower()
                if 0 <= i < len(vids) and h in HOOK_TYPES:
                    tags[vids[i]["video_id"]] = (
                        h, str(r.get("topic", ""))[:60].strip(), model)
            note = f"{model}: {len(tags)} of {len(vids)} tagged"
        except Exception as exc:                            # noqa: BLE001
            note = f"model failed ({type(exc).__name__}); rules used"
    db.execute("DELETE FROM ai_video_tags")
    for v in vids:
        hook, topic, by = tags.get(v["video_id"]) or (
            *rule_tag(v["title"]), "rules")
        db.execute("INSERT INTO ai_video_tags VALUES (?,?,?,?,?,?)",
                   (v["video_id"], v["title"], hook, topic, by,
                    now or time.time()))
    db.commit()
    bad = db.execute("SELECT COUNT(*) FROM ai_video_tags WHERE hook_type "
                     f"NOT IN ({','.join('?' * len(HOOK_TYPES))})",
                     HOOK_TYPES).fetchone()[0]
    n = db.execute("SELECT COUNT(*) FROM ai_video_tags").fetchone()[0]
    return [_check("every_video_tagged", n == len(vids),
                   f"{n} of {len(vids)} video(s) tagged - {note}"),
            _check("tags_from_the_allowed_list", bad == 0,
                   f"{bad} tag(s) outside the list")]


# -------------------------------------------------------------- 4. analytics

EXPORT_NAME = "bigquery_day7.csv"


def build_analytics(db, batch: str, export_dir: str | Path | None = None
                    ) -> list[dict]:
    """Analytics Builder: one row per video, the groups, and the file the
    Performance agent reads."""
    db.execute("DELETE FROM analytics_video_performance")
    db.execute("DELETE FROM analytics_groups")
    jobs = {r["video_id"]: r for r in db.execute(
        "SELECT * FROM source_jobs WHERE batch_id=?", (batch,))}
    rows = db.execute(
        "SELECT d.*, t.hook_type, t.topic FROM staging_video_day7 d "
        "LEFT JOIN ai_video_tags t USING (video_id)").fetchall()
    by_views = sorted(rows, key=lambda r: -r["day7_views"])
    by_hold = sorted(rows, key=lambda r: -r["day7_avg_view_s"])
    rank_v = {r["video_id"]: i + 1 for i, r in enumerate(by_views)}
    rank_h = {r["video_id"]: i + 1 for i, r in enumerate(by_hold)}
    for r in rows:
        j = jobs.get(r["video_id"])
        db.execute(
            "INSERT INTO analytics_video_performance VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["video_id"], r["title"], r["published_date"], r["mature"],
             r["day7_views"], r["day7_avg_view_s"], r["day7_watch_minutes"],
             r["day7_likes"], r["day7_subs"],
             round(r["day7_likes"] / r["day7_views"], 4)
             if r["day7_views"] else 0.0,
             rank_v[r["video_id"]], rank_h[r["video_id"]], r["hook_type"],
             r["topic"], j["job_id"] if j else None,
             j["decisions"] if j else None))
    db.execute(
        "INSERT INTO analytics_groups SELECT 'hook_type', hook_type, "
        "COUNT(*), ROUND(AVG(day7_views), 1), ROUND(AVG(day7_avg_view_s), 2) "
        "FROM analytics_video_performance GROUP BY hook_type")
    # Shorts and regular uploads live different lives (in the real table,
    # 13 Oct 2026: Shorts reach more people, regular videos are watched
    # three to five times longer), so they are never averaged together
    # without saying so. The channel marks its Shorts in the title.
    db.execute(
        "INSERT INTO analytics_groups SELECT 'format', CASE WHEN "
        "LOWER(title) LIKE '%shorts%' THEN 'short' ELSE 'regular' END f, "
        "COUNT(*), ROUND(AVG(day7_views), 1), ROUND(AVG(day7_avg_view_s), 2) "
        "FROM analytics_video_performance GROUP BY f")
    db.commit()

    written, export_note = 0, "no export folder given"
    if export_dir is not None:
        path = Path(export_dir) / EXPORT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["video_id", "title", "views", "watch_minutes",
                        "likes", "subs_gained", "first_day", "last_day",
                        "days_tracked"])
            for r in rows:
                if not r["mature"]:
                    continue        # a week not yet over is not a result
                w.writerow([r["video_id"], r["title"], int(r["day7_views"]),
                            round(r["day7_watch_minutes"], 2),
                            int(r["day7_likes"]), int(r["day7_subs"]),
                            r["published_date"],
                            _plus(r["published_date"], DAY7 - 1), DAY7])
                written += 1
        export_note = f"{written} matured video(s) written to {path.name}"
    n = db.execute("SELECT COUNT(*) FROM analytics_video_performance"
                   ).fetchone()[0]
    return [_check("one_row_per_video", n == len(rows),
                   f"{n} video(s) in the analytics table"),
            _check("linked_to_our_jobs", True,
                   f"{len(jobs)} video(s) linked to an Alpha Titans job "
                   f"(the link is recorded once per video, after upload)"),
            _check("export_for_learning", True, export_note)]


# ------------------------------------------------------------------ the run

AGENTS = (("source", "Source Loader"), ("staging", "Cleaner"),
          ("ai", "Enricher"), ("analytics", "Analytics Builder"),
          ("lessons", "Pattern Miner"))


def _mine(db):
    from datalayers import lessons
    return lessons.mine(db)


def run(db, fetch, *, llm=None, export_dir=None, jobs=None,
        emit=print) -> dict:
    """All four layers in order. A failed checker stops the rest.

    `fetch()` returns (rows, info): the warehouse query, or a stand-in in
    tests. Returns what ran and every check, for the screen and the log."""
    run_id = uuid.uuid4().hex[:10]
    done: list[dict] = []
    batch = None

    def step(layer, agent, fn):
        started = time.time()
        checks = fn()
        ok = all(c["ok"] for c in checks)
        db.execute("INSERT INTO layer_runs VALUES (?,?,?,?,?,?,?,?)",
                   (run_id, layer, agent, started, time.time(), int(ok),
                    None, json.dumps(checks)))
        db.commit()
        done.append({"layer": layer, "agent": agent, "ok": ok,
                     "checks": checks})
        emit(f"[{'ok' if ok else 'STOP'}] {layer:<9} {agent}")
        for c in checks:
            emit(f"      {'pass' if c['ok'] else 'FAIL'}  {c['name']}: "
                 f"{c['detail']}")
        return ok

    def source():
        nonlocal batch
        rows, info = fetch()
        batch, checks = load_source(db, rows, origin=SOURCE_TABLE,
                                    bytes_billed=info.get("bytes_billed", 0),
                                    jobs=jobs)
        return checks

    if not step("source", AGENTS[0][1], source):
        return {"run_id": run_id, "steps": done, "ok": False}
    for layer, agent, fn in (
            ("staging", AGENTS[1][1], lambda: build_staging(db, batch)),
            ("ai", AGENTS[2][1], lambda: tag_titles(db, llm)),
            ("analytics", AGENTS[3][1],
             lambda: build_analytics(db, batch, export_dir)),
            ("lessons", AGENTS[4][1], lambda: _mine(db))):
        if not step(layer, agent, fn):
            return {"run_id": run_id, "steps": done, "ok": False}
    return {"run_id": run_id, "steps": done, "ok": True, "batch": batch}


def report(db) -> dict:
    """What the analytics layer says, for people. Descriptive only."""
    vids = [dict(r) for r in db.execute(
        "SELECT * FROM analytics_video_performance ORDER BY rank_views")]
    groups = [dict(r) for r in db.execute(
        "SELECT * FROM analytics_groups ORDER BY avg_day7_views DESC")]
    mature = [v for v in vids if v["mature"]]
    linked = sum(1 for v in vids if v["job_id"])
    views = sum(v["day7_views"] for v in mature)
    watch = sum(v["day7_watch_minutes"] for v in mature)
    return {
        "videos": vids, "groups": groups,
        "summary": {
            "videos": len(vids), "matured": len(mature),
            "linked_to_jobs": linked,
            "day7_views_total": views,
            # weighted by views: what an average viewer actually watched
            "avg_view_s": round(watch * 60.0 / views, 1) if views else None,
            "learning_needs": 30,
        },
    }


__all__ = ["AGENTS", "DAY7", "HOOK_TYPES", "LOCATION", "SOURCE_SQL",
           "SOURCE_TABLE", "build_analytics", "build_staging", "connect",
           "load_source", "report", "rule_tag", "run", "tag_titles"]
