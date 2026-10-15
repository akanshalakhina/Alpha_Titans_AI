"""One copy of each clip, and not one byte of it lost.

WHY LINKING AND NOT DELETING
----------------------------
`uploads/` reached 7.8 GB and the obvious fix was a retention policy.
Measuring first showed why that would have been the wrong tool: of 7.80
GB in 76 files there were NINE distinct clips - one 430 MB file stored
twelve times, another across twenty-three copies - so 7.03 GB was the
same bytes written again.

Deleting old uploads would have taken away the ability to refine those
videos, permanently. Linking takes away only the duplication.

WHAT THESE TESTS ARE FOR
------------------------
A hardlink is one inode under several names, so a mistake here does not
look like a crash - it looks like a clip quietly becoming a different
clip. Every test below is about that: same content after, every path
still opens, nothing with different content ever joined, and a failure
halfway leaving the original in place.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _module(tmp_path):
    """The real script, pointed at a disposable uploads folder."""
    path = ROOT / "scripts" / "dedupe_uploads.py"
    spec = importlib.util.spec_from_file_location("dedupe_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = tmp_path
    mod.UPLOADS = tmp_path / "uploads"
    mod.LEDGER = tmp_path / "data" / "_ledger.json"
    mod.MIN_BYTES = 16
    return mod


@pytest.fixture
def dd(tmp_path):
    (tmp_path / "uploads").mkdir()
    return _module(tmp_path)


def _clip(dd, job: str, name: str, body: bytes):
    d = dd.UPLOADS / job
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(body)
    return p


BIG = b"the same footage, byte for byte" * 40
OTHER = b"entirely different footage here" * 40


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ========================================= it finds what is duplicated

def test_identical_clips_in_different_jobs_are_one_group(dd):
    _clip(dd, "job1", "a.mov", BIG)
    _clip(dd, "job2", "a.mov", BIG)
    assert len(dd.survey(dd.UPLOADS)["groups"]) == 1


def test_a_different_clip_is_never_grouped_with_it(dd):
    _clip(dd, "job1", "a.mov", BIG)
    _clip(dd, "job2", "b.mov", OTHER)
    assert len(dd.survey(dd.UPLOADS)["groups"]) == 2


def test_the_name_is_not_what_groups_them(dd):
    """Two files called the same thing with different content must not
    be linked, and two different names with one content must be."""
    _clip(dd, "job1", "same-name.mov", BIG)
    _clip(dd, "job2", "same-name.mov", OTHER)
    _clip(dd, "job3", "other-name.mov", BIG)
    groups = dd.survey(dd.UPLOADS)["groups"]
    assert len(groups) == 2
    assert sorted(len(v) for v in groups.values()) == [1, 2]


def test_small_files_are_left_out(dd):
    _clip(dd, "job1", "tiny.txt", b"x")
    _clip(dd, "job2", "tiny.txt", b"x")
    assert dd.survey(dd.UPLOADS)["groups"] == {}


def test_reporting_changes_nothing(dd, capsys):
    a = _clip(dd, "job1", "a.mov", BIG)
    b = _clip(dd, "job2", "a.mov", BIG)
    before = (_sha(a), _sha(b), a.stat().st_ino, b.stat().st_ino)
    assert dd.report(dd.UPLOADS) == 0
    capsys.readouterr()
    assert (_sha(a), _sha(b), a.stat().st_ino, b.stat().st_ino) == before


# ======================================== what linking does, and does not

def test_every_path_still_opens_and_reads_the_same(dd, capsys):
    paths = [_clip(dd, f"job{i}", "a.mov", BIG) for i in range(4)]
    before = [_sha(p) for p in paths]
    dd.apply(dd.UPLOADS)
    capsys.readouterr()

    assert [p.exists() for p in paths] == [True] * 4
    assert [_sha(p) for p in paths] == before, "a clip became another clip"


def test_they_really_are_one_file_afterwards(dd, capsys):
    a = _clip(dd, "job1", "a.mov", BIG)
    b = _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    assert dd._same_file(a, b), "nothing was actually saved"


def test_a_different_clip_is_never_touched(dd, capsys):
    a = _clip(dd, "job1", "a.mov", BIG)
    other = _clip(dd, "job2", "b.mov", OTHER)
    before = _sha(other)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    assert _sha(other) == before and not dd._same_file(a, other)


def test_nothing_is_deleted(dd, capsys):
    paths = [_clip(dd, f"job{i}", "a.mov", BIG) for i in range(3)]
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    assert all(p.is_file() for p in paths)
    assert len(list(dd.UPLOADS.rglob("*.mov"))) == 3


def test_no_working_file_is_left_behind(dd, capsys):
    _clip(dd, "job1", "a.mov", BIG)
    _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    assert not list(dd.UPLOADS.rglob("*.relinking")), \
        "a half-finished link must not leave a stray file"


def test_running_it_twice_is_harmless(dd, capsys):
    a = _clip(dd, "job1", "a.mov", BIG)
    b = _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    dd.apply(dd.UPLOADS)
    out = capsys.readouterr().out
    assert "linked 0 duplicate(s)" in out
    assert _sha(a) == _sha(b) == hashlib.sha256(BIG).hexdigest()


# ================================================ failing without damage

def test_a_link_that_cannot_be_made_leaves_the_original(dd, capsys,
                                                        monkeypatch):
    """The rename-then-link order exists for this: a crash between the
    two must not be able to lose a clip."""
    a = _clip(dd, "job1", "a.mov", BIG)
    b = _clip(dd, "job2", "a.mov", BIG)
    monkeypatch.setattr(dd.os, "link",
                        lambda *args: (_ for _ in ()).throw(
                            OSError("cross-device link")))
    dd.apply(dd.UPLOADS)
    out = capsys.readouterr().out

    assert b.exists() and _sha(b) == _sha(a), "the clip survived"
    assert not list(dd.UPLOADS.rglob("*.relinking"))
    assert "left alone" in out and "cross-device" in out


# ============================================================ verifying

def test_verify_reads_every_linked_file_back(dd, capsys):
    _clip(dd, "job1", "a.mov", BIG)
    _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()

    assert dd.verify(dd.UPLOADS) == 0
    assert "still reads as the clip it was" in capsys.readouterr().out


def test_verify_notices_a_linked_file_that_changed(dd, capsys):
    _clip(dd, "job1", "a.mov", BIG)
    b = _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()

    b.unlink()                                   # break the link, then differ
    b.write_bytes(OTHER)
    assert dd.verify(dd.UPLOADS) == 1
    assert "content changed" in capsys.readouterr().out


def test_verify_notices_a_linked_file_that_vanished(dd, capsys):
    _clip(dd, "job1", "a.mov", BIG)
    b = _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    b.unlink()
    assert dd.verify(dd.UPLOADS) == 1
    assert "gone" in capsys.readouterr().out


def test_verify_before_anything_was_linked(dd, capsys):
    assert dd.verify(dd.UPLOADS) == 0
    assert "nothing has been linked" in capsys.readouterr().out


def test_the_ledger_records_what_points_where(dd, capsys):
    _clip(dd, "job1", "a.mov", BIG)
    _clip(dd, "job2", "a.mov", BIG)
    dd.apply(dd.UPLOADS)
    capsys.readouterr()
    rec = json.loads(dd.LEDGER.read_text(encoding="utf-8"))
    assert len(rec) == 1
    only = next(iter(rec.values()))
    assert only["same_as"].endswith("a.mov") and only["sha256"]


def test_an_empty_uploads_folder_is_harmless(dd, capsys):
    assert dd.report(dd.UPLOADS) == 0
    assert dd.apply(dd.UPLOADS) == 0
    capsys.readouterr()


# ============================ the assumption the whole thing rests on

def test_nothing_in_the_project_writes_into_uploads():
    """A hardlink is only safe because the source footage is read-only
    after upload. If something starts writing there, writing through one
    job's path would change every other job's clip - and this test is
    what should fail first.
    """
    import re
    suspicious = re.compile(r"src_dir[^\n]*?\b(open\(|write_|unlink|rename)")
    for folder in ("server/agents", "server/agents/tools", "pipeline"):
        for py in (ROOT / folder).glob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            hit = suspicious.search(text)
            assert hit is None, f"{py.name} writes into the source dir: {hit}"
