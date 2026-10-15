"""A caption must survive the renderer's own stdout.

THE BUG THIS PINS
-----------------
`pipeline/build_video.py::stage_cut` prints every segment's caption as it
renders it. On Windows a child python gets cp1252 stdout, which cannot
ENCODE most of what a model writes, so one non-breaking hyphen (U+2011) in
the agent-written caption

    "Watch the four-blade propeller whirl"

raised UnicodeEncodeError inside that print and took the whole render with
it: `RuntimeError: cut failed (exit 1)`. A finished cut, lost to a log line.

Two defences, because either alone leaves a gap:
  * the stage is told to speak UTF-8, so ANY character a model picks can be
    printed (`analysis_tools.run_pipeline_stage`); and
  * the caption is normalised to plain ASCII punctuation before it ever
    reaches the renderer, because the render font often has no glyph for a
    non-breaking hyphen either, and nobody wants one burned into a video.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.caption import _clean                     # noqa: E402
from agents.tools import analysis_tools as AT         # noqa: E402


# ------------------------------------------------ 1. the caption itself

def test_the_exact_caption_that_killed_a_render():
    out = _clean("Watch the four‑blade propeller whirl", "")
    assert out == "Watch the four-blade propeller whirl"
    out.encode("cp1252")        # must not raise: this is what crashed


def test_a_true_hyphen_closes_up_the_word():
    """A hyphen joins one word, so it must not gain spaces."""
    for ch in "‐‑‒−":
        assert _clean(f"gear{ch}train", "") == "gear-train", repr(ch)


def test_a_dash_separates_and_keeps_its_air():
    """An em dash is punctuation between clauses, not part of a word.

    Closing it up read as a compound nobody wrote: "Hear that snap - powers
    on" came back as "snap-powers on".
    """
    for ch in "–—―":
        assert _clean(f"snap{ch}powers on", "") == "snap - powers on", repr(ch)


def test_smart_quotes_become_plain_ones():
    assert _clean("‘one’ and “two”", "") == "'one' and \"two\""


def test_an_ellipsis_becomes_three_dots():
    assert _clean("wait…", "") == "wait..."


def test_odd_spaces_collapse():
    assert _clean("a b c​d", "") == "a b cd"


def test_a_cleaned_caption_always_encodes_for_the_renderer():
    """The property that matters: whatever comes in, stdout can print it."""
    messy = ("A “smart” four‑blade rotor — spinning… "
             "at speed")
    _clean(messy, "").encode("cp1252")


def test_plain_text_is_left_alone():
    assert _clean("Mounting drive wheels", "") == "Mounting drive wheels"


# ------------------------------------------- 2. the stage speaks UTF-8

def test_the_pipeline_stage_is_told_to_speak_utf8(monkeypatch):
    seen = {}

    class Pool:
        def run_process(self, cmd, *, env=None, on_line=None, timeout_s=None,
                        job_id=None, label=""):
            seen["env"] = env
            return 0, ""

    AT.run_pipeline_stage("cut", "build", pool=Pool())
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8"


def test_a_caller_that_sets_the_encoding_itself_wins(monkeypatch):
    seen = {}

    class Pool:
        def run_process(self, cmd, *, env=None, on_line=None, timeout_s=None,
                        job_id=None, label=""):
            seen["env"] = env
            return 0, ""

    AT.run_pipeline_stage("cut", "build", env_overrides={"PYTHONIOENCODING":
                                                         "latin-1"},
                          pool=Pool())
    assert seen["env"]["PYTHONIOENCODING"] == "latin-1"


def test_other_overrides_are_not_lost(monkeypatch):
    seen = {}

    class Pool:
        def run_process(self, cmd, *, env=None, on_line=None, timeout_s=None,
                        job_id=None, label=""):
            seen["env"] = env
            return 0, ""

    AT.run_pipeline_stage("cut", "build",
                          env_overrides={"AT_CANVAS": "1280x720"}, pool=Pool())
    assert seen["env"]["AT_CANVAS"] == "1280x720"
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8"
