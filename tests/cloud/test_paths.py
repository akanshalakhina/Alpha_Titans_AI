"""Names that came from a person never become paths that escape.

Brief, section 13, test 13.
"""

from __future__ import annotations

import pytest

from cloud.paths import (check_job_id, safe_name, source_filename,
                         video_ext, within)
from cloud.sanitize import CloudError


@pytest.mark.parametrize("raw,expect", [
    ("clip.mp4", "clip.mp4"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\Windows\\win.ini", "win.ini"),
    ("C:\\Users\\x\\clip.MOV", "clip.mov"),
    ("con.mp4", "_con.mp4"),
    ("LPT1.avi", "_LPT1.avi"),
    ("a\x00b.mp4", "a_b.mp4"),
    ("   .hidden.mp4  ", "hidden.mp4"),
    ("", "file"),
    ("....", "file"),
])
def test_safe_name(raw, expect):
    assert safe_name(raw) == expect


@pytest.mark.parametrize("name,ok", [
    ("a.mp4", True), ("a.MOV", True), ("a.m4v", True), ("a.mkv", True),
    ("a.avi", True), ("a.webm", False), ("a.mp4.exe", False),
    ("a", False), ("a.txt", False)])
def test_only_the_renderers_formats_are_videos(name, ok):
    assert bool(video_ext(name)) is ok


def test_within_refuses_escape(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    assert within(ws, "a.mp4") == (ws / "a.mp4").resolve()
    for bad in ("../x", "../../x", "a/../../x", str(tmp_path / "other")):
        with pytest.raises(CloudError):
            within(ws, bad)
    with pytest.raises(CloudError):
        within(ws, ".")


@pytest.mark.parametrize("jid", ["abcdef123456", "0123456789ab"])
def test_job_ids_are_ours(jid):
    assert check_job_id(jid) == jid


@pytest.mark.parametrize("jid", ["ABCDEF123456", "../abcdef1234", "abc",
                                 "abcdef1234567", "", None])
def test_anything_else_is_not_a_job_id(jid):
    with pytest.raises(CloudError):
        check_job_id(jid)


def test_source_files_are_numbered_and_cannot_collide():
    assert source_filename(0, "x.mp4") == "01_x.mp4"
    assert source_filename(1, "x.mp4") == "02_x.mp4"
    assert source_filename(0, "../../x.mp4") == "01_x.mp4"
