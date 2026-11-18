"""Feature Analyst, Opening Reviewer, the feature lessons and the Why Writer.

What must hold:
* a downloaded file is matched to its video by title or id, and a file
  that matches nothing - or more than one video - is reported, not guessed;
* cuts are found in a real file, and a video is measured only once;
* "% watched" is the average view against the video's length;
* feature lessons are tried across all videos AND within Shorts and
  regular videos apart, and say which;
* the Why Writer keeps only sentences that cite a lesson it was given, and
  without a model falls back to the lessons' own words.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

from datalayers import features as FT          # noqa: E402
from datalayers import layers as L             # noqa: E402
from datalayers import lessons as LS           # noqa: E402
from datalayers import why as WY               # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def db(tmp_path):
    d = L.connect(tmp_path / "f.db")
    d.executescript(FT.SCHEMA)
    yield d
    d.close()


# ------------------------------------------------------------- matching

VIDEOS = [("IoXCt-UmhNo", "LEGO Tank Tracks Build | Alpha Titans Shorts"),
          ("yj-IG0hVGLk", "LEGO Chassis Build - Upbeat Music | Alpha Titans Shorts"),
          ("aaaaaaaaaa1", "Same Title"), ("aaaaaaaaaa2", "Same Title")]


def test_a_studio_download_is_matched_by_its_title(tmp_path):
    (tmp_path / "LEGO Tank Tracks Build ｜ Alpha Titans Shorts.mp4").write_bytes(b"x")
    found, notes = FT.match_files(tmp_path, VIDEOS)
    assert found == {"IoXCt-UmhNo": tmp_path / "LEGO Tank Tracks Build ｜ Alpha Titans Shorts.mp4"}
    assert notes == []


def test_a_video_id_in_the_name_wins(tmp_path):
    (tmp_path / "upload yj-IG0hVGLk final.mp4").write_bytes(b"x")
    found, _ = FT.match_files(tmp_path, VIDEOS)
    assert list(found) == ["yj-IG0hVGLk"]


def test_unknown_and_ambiguous_files_are_reported_not_guessed(tmp_path):
    (tmp_path / "Holiday clip.mp4").write_bytes(b"x")
    (tmp_path / "Same Title.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not a video")
    found, notes = FT.match_files(tmp_path, VIDEOS)
    assert found == {}
    assert any("no video in the warehouse" in n for n in notes)
    assert any("matches 2 videos" in n for n in notes)
    assert len(notes) == 2


# ------------------------------------------------------------ measuring

def _two_shot_clip(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=s=320x240:r=30:d=3",
         "-f", "lavfi", "-i", "smptebars=s=320x240:r=30:d=3",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is not on PATH")
def test_a_cut_between_two_shots_is_found(tmp_path):
    clip = _two_shot_clip(tmp_path / "two.mp4")
    cuts = FT.scene_cuts(clip, 6.0)
    assert cuts is not None and len(cuts) == 1 and 2.5 <= cuts[0] <= 3.5
    m = FT.measure(clip)
    assert m["shots"] == 2 and m["cuts_first5"] == 1
    assert m["duration_s"] == pytest.approx(6.0, abs=0.1) and m["vertical"] == 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is not on PATH")
def test_a_video_is_measured_once(tmp_path, db):
    folder = tmp_path / "vids"
    folder.mkdir()
    _two_shot_clip(folder / "Same Clip.mp4")
    db.execute("INSERT INTO staging_video_day7 (video_id, title, published_date, "
               "mature) VALUES ('bbbbbbbbbbb', 'Same Clip', '2026-09-07', 1)")
    first = FT.analyse_features(db, folder)
    again = FT.analyse_features(db, folder)
    assert "1 newly measured" in first[1]["detail"]
    assert "0 newly measured, 1 in total" in again[1]["detail"]


def test_no_files_says_where_to_put_them(tmp_path, db):
    checks = FT.analyse_features(db, tmp_path)
    assert "download them from YouTube Studio" in checks[0]["detail"]
    assert all(c["ok"] for c in checks)


def test_without_a_vision_model_openings_are_not_reviewed(tmp_path, db):
    checks = FT.review_openings(db, None, tmp_path)
    assert checks[0]["ok"] and "nothing to look at" in checks[0]["detail"]


# --------------------------------------------------------- the lessons

def seed(db, rows):
    """rows: (vid, title, day7_views, avg_view_s, duration, result_first)"""
    for vid, title, views, hold, dur, rf in rows:
        db.execute("INSERT INTO staging_video_day7 (video_id, title, "
                   "published_date, mature) VALUES (?,?,?,1)",
                   (vid, title, "2026-09-07"))
        db.execute("INSERT INTO analytics_video_performance (video_id, title, "
                   "published_date, mature, day7_views, day7_avg_view_s) "
                   "VALUES (?,?,?,?,?,?)",
                   (vid, title, "2026-09-07", 1, views, hold))
        db.execute("INSERT INTO video_features (video_id, file, file_sig, "
                   "duration_s, shots, opening_energy, avg_shot_s) VALUES "
                   "(?,?,?,?,?,?,?)", (vid, vid + ".mp4", "s", dur, 5, 1.0,
                                       dur / 5))
        db.execute("INSERT INTO ai_video_looks (video_id, result_first) "
                   "VALUES (?,?)", (vid, rf))
    db.commit()


CHANNEL = [("a", "A | Shorts", 900, 12, 20, 1), ("b", "B | Shorts", 700, 11, 20, 1),
           ("c", "C | Shorts", 300, 4, 20, 0), ("d", "D | Shorts", 200, 3, 20, 0),
           ("e", "E", 150, 50, 100, 1), ("f", "F", 120, 45, 100, 1),
           ("g", "G", 80, 15, 100, 0), ("h", "H", 60, 10, 100, 0)]


def test_percent_watched_is_the_average_view_against_the_length(db):
    seed(db, CHANNEL)
    rows = {r["video_id"]: r for r in LS.why_rows(db)}
    assert rows["a"]["pct_watched"] == 60.0
    assert rows["e"]["pct_watched"] == 50.0


def test_feature_lessons_are_tried_overall_and_within_each_format(db):
    seed(db, CHANNEL)
    found, tried = LS.mine_features(db)
    rf = [f for f in found if f["dimension"] == "result_first"
          and f["metric"] == "pct_watched"]
    scopes = {f["text"].split(":")[0] if f["text"].startswith("Among") else "all"
              for f in rf}
    assert scopes == {"all", "Among Shorts", "Among regular videos"}
    assert all(f["better"] == "opens on the finished build" for f in rf)
    assert all(f["stable"] for f in rf)
    assert tried >= 3


def test_mine_keeps_the_feature_lessons_with_the_title_ones(db):
    seed(db, CHANNEL)
    LS.mine(db)
    dims = {l["dimension"] for l in LS.listing(db)}
    assert "result_first" in dims and "format" in dims


# --------------------------------------------------------- the writer

class Reply:
    meta = {"model": "test-model"}


class Writer:
    def __init__(self, sentences):
        self.sentences = sentences

    def available(self):
        return True

    def complete_json(self, prompt, max_tokens=0):
        return {"sentences": self.sentences(prompt)}, Reply()


def test_the_writer_keeps_only_sentences_that_cite_a_given_lesson(db):
    seed(db, CHANNEL)
    LS.mine(db)
    lid = next(l["id"] for l in LS.listing(db) if l["stable"])

    def answer(prompt):
        assert lid in prompt
        return [f"Opening on the finished build holds viewers [{lid}].",
                "Thumbnails with faces win.",
                "Shorter titles help [0123456789]."]
    WY.write(db, Writer(answer))
    got = WY.latest(db)
    assert got["written_by"] == "test-model"
    assert got["text"] == f"Opening on the finished build holds viewers [{lid}]."
    assert got["lesson_ids"] == lid


def test_without_a_model_the_strongest_lessons_speak_for_themselves(db):
    seed(db, CHANNEL)
    LS.mine(db)
    WY.write(db, None)
    got = WY.latest(db)
    assert got["written_by"] == "rules" and got["text"]
    assert got["lesson_ids"]


def test_with_nothing_that_held_the_writer_says_so(db):
    WY.write(db, None)
    assert "No pattern has held up yet" in WY.latest(db)["text"]


def test_a_rejected_lesson_is_not_written_about(db):
    seed(db, CHANNEL)
    LS.mine(db)
    for l in LS.listing(db):
        LS.decide(db, l["id"], approve=False)
    WY.write(db, None)
    assert "No pattern has held up yet" in WY.latest(db)["text"]
