"""The same footage must not be measured twice.

WHAT THIS SAVES
---------------
`stage_measure` decodes a clip TWICE - once for `freezedetect`, once for
`signalstats`. On one eight-minute upload that took 586 seconds, half of a
twenty-minute production. Its own "already measured" check keeps a
`<clip>.ydif.txt` beside the build, but every job gets a fresh build
directory, so re-uploading the same file re-measured it from scratch: four
runs over one file produced four byte-identical results (md5 9c5be1a5...).

The measurement is a pure function of the file's bytes, so the key is a
sha256 of the file - 1.5s for 451 MB against the ten minutes it saves -
and never the name or timestamp, which say nothing about the pixels.

Measured end to end on the real footage: 586s -> 1.8s.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.tools import measure_cache as MC                # noqa: E402


FULL = {"mean_ydif": 20.34, "max_ydif": 121.3, "live_share": 0.8,
        "freeze_share": 0.05, "freeze_runs": [[2.5, 3.0]],
        "sample_fps": 10.0, "ydif": [1.0, 2.0, 3.0]}


@pytest.fixture
def dirs(tmp_path):
    src = tmp_path / "uploads"; src.mkdir()
    build = tmp_path / "build"; (build / "measure").mkdir(parents=True)
    cache = tmp_path / "cache"
    (src / "a.mp4").write_bytes(b"pretend this is video" * 100)
    (build / "measure" / "a.mp4.ydif.txt").write_text("frame:0 YDIF=1.0")
    return src, build, cache


def _manifest(**over):
    row = {"name": "a.mp4", "duration": 10.0}
    row.update(over)
    return {"clips": [row]}


# ------------------------------------------------------------ the key

def test_the_key_is_the_bytes_not_the_name(dirs):
    src, _b, _c = dirs
    same = src / "renamed.mp4"
    same.write_bytes((src / "a.mp4").read_bytes())
    assert MC.file_key(src / "a.mp4") == MC.file_key(same)


def test_different_footage_is_a_different_key(dirs):
    src, _b, _c = dirs
    (src / "b.mp4").write_bytes(b"a different video entirely")
    assert MC.file_key(src / "a.mp4") != MC.file_key(src / "b.mp4")


def test_a_missing_file_has_no_key(dirs):
    src, _b, _c = dirs
    assert MC.file_key(src / "gone.mp4") is None


# ------------------------------------------------------- store/restore

def test_a_measured_clip_comes_back_whole(dirs):
    src, build, cache = dirs
    assert MC.store(cache, src, build, _manifest(**FULL)) == ["a.mp4"]

    fresh = _manifest()                      # a new job: nothing measured
    other = build.parent / "build2"
    (other / "measure").mkdir(parents=True)
    assert MC.restore(cache, src, other, fresh) == ["a.mp4"]

    got = fresh["clips"][0]
    for f in MC.MEASURED_FIELDS:
        assert got[f] == FULL[f], f
    assert (other / "measure" / "a.mp4.ydif.txt").exists(), \
        "stage_measure looks for this file, and skips the clip when it is there"


def test_a_clip_this_build_already_measured_is_left_alone(dirs):
    src, build, cache = dirs
    MC.store(cache, src, build, _manifest(**FULL))
    already = _manifest(mean_ydif=99.0)      # this build measured it itself
    assert MC.restore(cache, src, build, already) == []
    assert already["clips"][0]["mean_ydif"] == 99.0, "not overwritten"


def test_nothing_cached_is_simply_a_miss(dirs):
    src, build, cache = dirs
    assert MC.restore(cache, src, build, _manifest()) == []


# ------------------------------------------- it must never guess wrong

def test_a_half_written_entry_is_not_a_hit(dirs):
    """The stats file is copied first and the json lands last, so a pair
    caught mid-write reads as a miss rather than a partial measurement."""
    src, build, cache = dirs
    MC.store(cache, src, build, _manifest(**FULL))
    key = MC.file_key(src / "a.mp4")
    (cache / f"{key}.json").unlink()
    assert MC.load(cache, key) is None
    assert MC.restore(cache, src, build, _manifest()) == []


def test_an_incomplete_row_is_refused(dirs):
    """A row from a narrower measure pass must not stand in for a full one."""
    src, build, cache = dirs
    partial = dict(FULL); partial.pop("ydif")
    assert MC.store(cache, src, build, _manifest(**partial)) == []


def test_corrupt_json_is_a_miss_not_a_crash(dirs):
    src, build, cache = dirs
    MC.store(cache, src, build, _manifest(**FULL))
    key = MC.file_key(src / "a.mp4")
    (cache / f"{key}.json").write_text("{not json")
    assert MC.load(cache, key) is None


def test_storing_without_a_stats_file_stores_nothing(dirs):
    src, build, cache = dirs
    (build / "measure" / "a.mp4.ydif.txt").unlink()
    assert MC.store(cache, src, build, _manifest(**FULL)) == []


def test_a_cache_that_does_not_exist_yet_is_harmless(tmp_path):
    src = tmp_path / "u"; src.mkdir()
    build = tmp_path / "b"; build.mkdir()
    assert MC.restore(tmp_path / "never-made", src, build, _manifest()) == []
