"""Reading real outcomes, and refusing to invent the link to them.

WHAT THIS GUARDS
----------------
Learning has sat at 0 usable samples of 30 since it was written, because
`PerformanceAnalysis` can only say "unavailable": the project uploads
nothing and has no analytics reader. The numbers exist in the operator's
warehouse; what crosses the gap is a CSV exported from it, because no
Databricks or BigQuery driver is installed and adding one would mean a new
dependency and their warehouse credentials.

The dangerous part is not the reading, it is the JOIN. A view count
attached to the wrong build would teach Learning the opposite of the
truth, and unlike a crash it would never announce itself. So the id comes
from a link the operator records after uploading, and anything that cannot
be matched is reported unmatched rather than guessed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.tools import analytics_store as AS              # noqa: E402


HEADER = ("title,video_id,views,watch_time_minutes,likes,"
          "subscribers_gained,first_day,last_day\n")


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "analytics"
    d.mkdir()
    return d


def _csv(folder, name="export.csv", body=""):
    (folder / name).write_text(HEADER + body, encoding="utf-8")
    return folder


# ====================================================== the YouTube link

@pytest.mark.parametrize("text", [
    "dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?si=abc",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "  https://studio.youtube.com/video/dQw4w9WgXcQ/edit  ",
])
def test_every_shape_of_link_yields_the_same_id(text):
    assert AS.video_id_from(text) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("text", [
    "", "   ", "not a link", "https://example.com/watch?v=short",
    "https://www.youtube.com/@somechannel",
    "https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw",
])
def test_anything_that_is_not_an_id_is_refused(text):
    assert AS.video_id_from(text) is None, \
        "a wrong id would attach one video's numbers to another's decisions"


# ================================================== the job's own record

def test_a_recorded_video_comes_back(tmp_path):
    AS.save_reference(tmp_path, "dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
    rec = AS.load_reference(tmp_path)
    assert rec["video_id"] == "dQw4w9WgXcQ"
    assert rec["recorded_at"] > 0


def test_no_record_is_simply_absent(tmp_path):
    assert AS.load_reference(tmp_path) is None


def test_a_corrupt_record_reads_as_absent_not_as_a_crash(tmp_path):
    AS.reference_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert AS.load_reference(tmp_path) is None


def test_a_record_without_an_id_is_not_a_record(tmp_path):
    AS.reference_path(tmp_path).write_text(json.dumps({"url": "x"}),
                                           encoding="utf-8")
    assert AS.load_reference(tmp_path) is None


def test_a_record_can_be_withdrawn(tmp_path):
    AS.save_reference(tmp_path, "dQw4w9WgXcQ")
    assert AS.clear_reference(tmp_path) is True
    assert AS.load_reference(tmp_path) is None
    assert AS.clear_reference(tmp_path) is False


# ========================================================== the CSV rows

def test_the_exported_query_is_read(folder):
    _csv(folder, body="LEGO build,dQw4w9WgXcQ,10432,8120.5,310,47,"
                      "2026-09-01,2026-10-01\n")
    ex = AS.read_export(folder)

    assert len(ex.rows) == 1
    r = ex.rows[0]
    assert (r.video_id, r.title) == ("dQw4w9WgXcQ", "LEGO build")
    assert (r.views, r.watch_minutes, r.likes, r.subs_gained) == \
        (10432.0, 8120.5, 310.0, 47.0)
    assert r.source_file == "export.csv"


def test_the_window_the_export_covers_is_counted(folder):
    """Learning's time_window gate needs a span, and SUM(views) over all
    time does not carry one."""
    _csv(folder, body="a,dQw4w9WgXcQ,1,1,1,1,2026-09-01,2026-09-30\n")
    assert AS.read_export(folder).rows[0].window_days == 30


def test_an_export_with_no_dates_reports_no_window(folder):
    (folder / "x.csv").write_text("video_id,views\ndQw4w9WgXcQ,5\n",
                                  encoding="utf-8")
    assert AS.read_export(folder).rows[0].window_days == 0


def test_the_warehouses_own_column_names_are_accepted(folder):
    (folder / "raw.csv").write_text(
        "title,video_id,SUM(views),SUM(watch_time_minutes),SUM(likes),"
        "SUM(subscribers_gained)\nx,dQw4w9WgXcQ,90,12,3,1\n",
        encoding="utf-8")
    r = AS.read_export(folder).rows[0]
    assert (r.views, r.watch_minutes, r.likes, r.subs_gained) == \
        (90.0, 12.0, 3.0, 1.0)


def test_a_spreadsheets_idea_of_a_number_is_understood(folder):
    _csv(folder, body='x,dQw4w9WgXcQ,"12,430",1.5K,,7,,\n')
    r = AS.read_export(folder).rows[0]
    assert r.views == 12430.0
    assert r.watch_minutes == 1500.0
    assert r.likes == 0.0, "a blank is zero, not a crash"


def test_several_exports_are_read_together(folder):
    _csv(folder, "sep.csv", "a,dQw4w9WgXcQ,10,1,1,1,,\n")
    _csv(folder, "oct.csv", "b,aaaaaaaaaaa,20,2,2,2,,\n")
    ex = AS.read_export(folder)
    assert {r.video_id for r in ex.rows} == {"dQw4w9WgXcQ", "aaaaaaaaaaa"}
    assert sorted(ex.files) == ["oct.csv", "sep.csv"]


def test_the_later_export_of_one_video_wins(folder):
    """An export is a snapshot; the newer one has riper numbers."""
    _csv(folder, "a_sep.csv", "x,dQw4w9WgXcQ,10,1,1,1,,\n")
    _csv(folder, "b_oct.csv", "x,dQw4w9WgXcQ,99,1,1,1,,\n")
    ex = AS.read_export(folder)
    assert len(ex.rows) == 1 and ex.rows[0].views == 99.0


# ------------------------------------------------- refusing bad material

def test_the_wrong_query_is_named_not_silently_dropped(folder):
    (folder / "wrong.csv").write_text("channel,subs\nmine,42\n",
                                      encoding="utf-8")
    ex = AS.read_export(folder)
    assert ex.rows == []
    assert any("wrong.csv" in s and "video_id" in s for s in ex.skipped), \
        "exporting the wrong query is the likeliest mistake here"


def test_a_row_with_no_video_id_is_left_out(folder):
    _csv(folder, body="x,,10,1,1,1,,\nz,dQw4w9WgXcQ,20,1,1,1,,\n")
    assert [r.video_id for r in AS.read_export(folder).rows] == ["dQw4w9WgXcQ"]


def test_an_empty_or_missing_folder_is_harmless(tmp_path, folder):
    assert AS.read_export(folder).rows == []
    assert AS.read_export(tmp_path / "never-made").rows == []


def test_a_match_is_by_id_only(folder):
    _csv(folder, body="Some title,dQw4w9WgXcQ,10,1,1,1,,\n")
    ex = AS.read_export(folder)
    assert ex.by_id("dQw4w9WgXcQ") is not None
    assert ex.by_id("aaaaaaaaaaa") is None, "no fuzzy fallback to the title"
