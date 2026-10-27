"""The Data Layers team, on rows shaped like the real warehouse table.

What must hold:
* the source layer keeps rows exactly as they came, batch by batch;
* staging keeps one row per video per day (the latest load wins), keeps a
  NET negative like as it is, and stops on impossible numbers;
* every video is judged over its first seven days, and a week not yet over
  is never exported as a result;
* the enricher signs every tag, and falls back to rules without a model;
* a failed checker stops every layer after it;
* the BigQuery reader converts by schema and never leaks the token.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

from datalayers import bigquery as BQ          # noqa: E402
from datalayers import layers as L             # noqa: E402

PUB = dt.datetime(2026, 9, 7, 12, tzinfo=dt.timezone.utc).timestamp()


def day(n):
    return (dt.date(2026, 9, 7) + dt.timedelta(days=n)).isoformat()


def row(vid, n, views, *, title="LEGO Build | Alpha Titans Shorts",
        likes=1, watch=1.0, loaded=1.0, pub=PUB):
    return {"video_id": vid, "title": title, "published_at": pub,
            "metric_date": day(n), "views": views, "likes": likes,
            "comments": 0, "shares": 0, "watch_time_minutes": watch,
            "avg_view_duration_seconds": 6.0, "subscribers_gained": 0,
            "loaded_at": loaded}


def table():
    rows = [row("aaaaaaaaaaa", n, 100, watch=10.0) for n in range(10)]
    rows.append(row("aaaaaaaaaaa", 2, 500, watch=50.0, loaded=2.0))  # reload
    rows.append(row("bbbbbbbbbbb", 0, 20, likes=-2,
                    title="How fast can it go? | Alpha Titans"))
    rows += [row("ccccccccccc", n, 5, pub=PUB + 20 * 86400) for n in
             range(20, 23)]                       # published late: week open
    return rows


@pytest.fixture
def db(tmp_path):
    return L.connect(tmp_path / "layers.db")


def fetch_of(rows):
    return lambda: (rows, {"bytes_billed": 10485760})


# --------------------------------------------------------------- the run

def test_all_four_layers_run_in_order_and_pass(db, tmp_path):
    lines = []
    out = L.run(db, fetch_of(table()), export_dir=tmp_path / "exp",
                emit=lines.append)
    assert out["ok"]
    assert [s["layer"] for s in out["steps"]] == [
        "source", "staging", "ai", "analytics", "lessons"]
    assert [s["agent"] for s in out["steps"]] == [a for _, a in L.AGENTS]
    runs = db.execute("SELECT layer, ok FROM layer_runs").fetchall()
    assert [tuple(r) for r in runs] == [("source", 1), ("staging", 1),
                                        ("ai", 1), ("analytics", 1),
                                        ("lessons", 1)]


def test_the_source_layer_keeps_rows_exactly_and_appends(db):
    L.run(db, fetch_of(table()), emit=lambda _: None)
    L.run(db, fetch_of(table()), emit=lambda _: None)
    n = db.execute("SELECT COUNT(*) FROM source_youtube_video_daily"
                   ).fetchone()[0]
    assert n == 2 * len(table())
    assert db.execute("SELECT COUNT(*) FROM source_batches").fetchone()[0] == 2


def test_staging_keeps_the_latest_load_of_a_day(db):
    L.run(db, fetch_of(table()), emit=lambda _: None)
    v = db.execute("SELECT views FROM staging_video_daily WHERE "
                   "video_id='aaaaaaaaaaa' AND metric_date=?",
                   (day(2),)).fetchone()[0]
    assert v == 500


def test_a_like_taken_back_stays_negative(db):
    L.run(db, fetch_of(table()), emit=lambda _: None)
    likes = db.execute("SELECT likes FROM staging_video_daily WHERE "
                       "video_id='bbbbbbbbbbb'").fetchone()[0]
    assert likes == -2


def test_negative_views_stop_every_layer_after_staging(db):
    bad = table() + [row("ddddddddddd", 0, -5)]
    out = L.run(db, fetch_of(bad), emit=lambda _: None)
    assert not out["ok"]
    assert [s["layer"] for s in out["steps"]] == ["source", "staging"]
    assert db.execute("SELECT COUNT(*) FROM analytics_video_performance"
                      ).fetchone()[0] == 0


def test_an_empty_source_stops_at_once(db):
    out = L.run(db, fetch_of([]), emit=lambda _: None)
    assert not out["ok"] and len(out["steps"]) == 1


# ---------------------------------------------------------------- week one

def test_every_video_is_judged_on_its_first_seven_days(db):
    L.run(db, fetch_of(table()), emit=lambda _: None)
    a = db.execute("SELECT * FROM staging_video_day7 WHERE "
                   "video_id='aaaaaaaaaaa'").fetchone()
    # days 0..6, with day 2 re-loaded as 500: 6 x 100 + 500
    assert a["day7_views"] == 1100
    assert a["total_views"] == 1400
    assert a["mature"] == 1
    assert a["day7_avg_view_s"] == pytest.approx(
        (6 * 10 + 50) * 60 / 1100, abs=0.01)


def test_an_unfinished_week_is_not_exported_as_a_result(db, tmp_path):
    L.run(db, fetch_of(table()), export_dir=tmp_path, emit=lambda _: None)
    c = db.execute("SELECT mature FROM staging_video_day7 WHERE "
                   "video_id='ccccccccccc'").fetchone()[0]
    assert c == 0
    with (tmp_path / L.EXPORT_NAME).open(encoding="utf-8") as fh:
        ids = {r["video_id"] for r in csv.DictReader(fh)}
    assert "ccccccccccc" not in ids and "aaaaaaaaaaa" in ids


def test_the_export_is_what_the_performance_agent_reads(db, tmp_path):
    from agents.tools import analytics_store as AS
    L.run(db, fetch_of(table()), export_dir=tmp_path, emit=lambda _: None)
    exp = AS.read_export(tmp_path)
    r = exp.by_id("aaaaaaaaaaa")
    assert r is not None and r.views == 1100 and r.window_days == 7
    assert not exp.skipped


# -------------------------------------------------------------------- tags

def test_without_a_model_the_rules_tag_and_say_so(db):
    L.run(db, fetch_of(table()), llm=None, emit=lambda _: None)
    t = {r["video_id"]: r for r in db.execute("SELECT * FROM ai_video_tags")}
    assert t["bbbbbbbbbbb"]["hook_type"] == "question"
    assert all(r["tagged_by"] == "rules" for r in t.values())


class FakeReply:
    meta = {"model": "test-model"}


class FakeLLM:
    def available(self):
        return True

    def complete_json(self, prompt, max_tokens=0):
        n = prompt.count("\n") and len([l for l in prompt.splitlines()
                                        if l[:1].isdigit()])
        return {"videos": [{"n": i + 1, "hook_type": "result_first",
                            "topic": "lego"} for i in range(n)]}, FakeReply()


def test_a_model_tags_and_signs_its_rows(db):
    L.run(db, fetch_of(table()), llm=FakeLLM(), emit=lambda _: None)
    rows = db.execute("SELECT hook_type, tagged_by FROM ai_video_tags"
                      ).fetchall()
    assert {tuple(r) for r in rows} == {("result_first", "test-model")}


def test_shorts_and_regular_videos_are_grouped_apart(db):
    L.run(db, fetch_of(table()), emit=lambda _: None)
    g = {r["value"]: r["videos"] for r in db.execute(
        "SELECT * FROM analytics_groups WHERE dimension='format'")}
    assert g == {"short": 2, "regular": 1}


def test_the_report_weights_watch_time_by_views(db):
    L.run(db, fetch_of(table()), emit=lambda _: None)
    s = L.report(db)["summary"]
    # only "aaa" has a finished first week: "bbb" has one day of data
    assert s["matured"] == 1 and s["linked_to_jobs"] == 0
    assert s["avg_view_s"] == pytest.approx((6 * 10 + 50) * 60 / 1100,
                                            abs=0.1)


# --------------------------------------------------------------- bigquery

class Resp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_rows_are_converted_by_their_schema_and_paged():
    pages = [
        {"jobReference": {"jobId": "j1"}, "jobComplete": True,
         "totalBytesBilled": "10485760", "pageToken": "p2",
         "schema": {"fields": [{"name": "video_id", "type": "STRING"},
                               {"name": "views", "type": "INTEGER"},
                               {"name": "published_at", "type": "TIMESTAMP"}]},
         "rows": [{"f": [{"v": "a"}, {"v": "5"}, {"v": "1.7E9"}]}]},
        {"jobComplete": True,
         "schema": {"fields": [{"name": "video_id", "type": "STRING"},
                               {"name": "views", "type": "INTEGER"},
                               {"name": "published_at", "type": "TIMESTAMP"}]},
         "rows": [{"f": [{"v": "b"}, {"v": "7"}, {"v": None}]}]},
    ]
    seen = []

    def http(req, timeout=None):
        seen.append(req)
        return Resp(pages[len(seen) - 1])

    rows, info = BQ.query("SELECT 1", "proj", "asia-south1", "tok",
                          http=http)
    assert rows == [{"video_id": "a", "views": 5, "published_at": 1.7e9},
                    {"video_id": "b", "views": 7, "published_at": None}]
    assert info["bytes_billed"] == 10485760
    assert "pageToken=p2" in seen[1].full_url


def test_an_error_names_the_problem_and_never_the_token():
    import urllib.error

    def http(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {},
            io.BytesIO(b'{"error": {"message": "Access Denied"}}'))

    with pytest.raises(BQ.BigQueryError) as e:
        BQ.query("SELECT 1", "proj", "asia-south1", "secret-token",
                 http=http)
    assert "Access Denied" in str(e.value)
    assert "secret-token" not in str(e.value)


def test_no_identity_is_refused_before_any_call():
    with pytest.raises(BQ.BigQueryError):
        BQ.query("SELECT 1", "proj", "asia-south1", "",
                 http=lambda *a, **k: pytest.fail("called"))
