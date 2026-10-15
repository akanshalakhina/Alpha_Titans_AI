"""Frame counting must survive a rotated (vertical) render.

THE BUG THIS PINS
-----------------
`extras.frame_count` parses ffprobe's `-of csv=p=0` output with
`out.isdigit()`. A file carrying display-matrix side data - which is every
VERTICAL video this pipeline produces - makes ffprobe append an empty
trailing field, so the output is "240," and `isdigit()` is False. The count
came back None, the critic read that as a BLOCKING failure, and a perfectly
decodable vertical video was sent round the revision loop twice before
escalating. Observed on a real 720x1280 production: frames=None,
frames_expected=360, two REVISE verdicts, final file 240 decodable frames.

extras.py is a protected file, so `verify_tools.frame_count` retries with a
tolerant parse. These tests pin both halves: the original path still wins
when it works, and the fallback rescues the rotated case.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.tools import verify_tools as VT          # noqa: E402


def test_the_original_path_is_used_when_it_works(monkeypatch):
    """A horizontal render must behave exactly as it always did."""
    monkeypatch.setattr(VT.extras_mod, "frame_count", lambda p: 708)
    called = []
    monkeypatch.setattr(VT.extras_mod, "run",
                        lambda cmd: called.append(cmd) or (0, "should not run"))
    assert VT.frame_count("anything.mp4") == 708
    assert not called, "the fallback probe must not run when the first try works"


def test_a_trailing_comma_no_longer_loses_the_count(monkeypatch):
    """The exact ffprobe output a vertical render produces."""
    monkeypatch.setattr(VT.extras_mod, "frame_count", lambda p: None)
    monkeypatch.setattr(VT.extras_mod, "run", lambda cmd: (0, "240,\n"))
    assert VT.frame_count("vertical.mp4") == 240


def test_a_clean_number_from_the_fallback_also_parses(monkeypatch):
    monkeypatch.setattr(VT.extras_mod, "frame_count", lambda p: None)
    monkeypatch.setattr(VT.extras_mod, "run", lambda cmd: (0, "360\n"))
    assert VT.frame_count("x.mp4") == 360


def test_a_failed_probe_is_still_unmeasurable(monkeypatch):
    """A real failure must stay None rather than becoming a wrong number."""
    monkeypatch.setattr(VT.extras_mod, "frame_count", lambda p: None)
    monkeypatch.setattr(VT.extras_mod, "run", lambda cmd: (1, ""))
    assert VT.frame_count("broken.mp4") is None


def test_garbage_output_is_not_coerced(monkeypatch):
    monkeypatch.setattr(VT.extras_mod, "frame_count", lambda p: None)
    monkeypatch.setattr(VT.extras_mod, "run", lambda cmd: (0, "N/A\n"))
    assert VT.frame_count("weird.mp4") is None


def test_the_fallback_asks_ffprobe_without_csv(monkeypatch):
    """csv=p=0 is what appends the empty field, so it must not be reused."""
    monkeypatch.setattr(VT.extras_mod, "frame_count", lambda p: None)
    seen = {}
    monkeypatch.setattr(VT.extras_mod, "run",
                        lambda cmd: seen.update(cmd=[str(c) for c in cmd])
                        or (0, "240,"))
    VT.frame_count("v.mp4")
    joined = " ".join(seen["cmd"])
    assert "csv=p=0" not in joined
    assert "-count_frames" in joined
