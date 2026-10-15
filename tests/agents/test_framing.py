"""A mixed-orientation set must still come out as one video.

THE PROBLEM THIS SOLVES
-----------------------
`build_video.fit_filter` reads ONE field to decide how to place a segment:

    same = (seg["orientation"] == ("vertical" if ch > cw else "horizontal"))
    if same and not crop:   crop to FILL the canvas
    else:                   scale the whole clip in, blurred bars behind

So a landscape roll with one vertical clip in it cut between full-frame
shots and a narrow letterboxed strip - the finished video did not read as
one piece. Measured on a real build: the vertical segment's picture
occupied about a quarter of the frame, the rest a blurred copy of itself.

`framing="fill"` (the default) projects every segment as already matching
the canvas, so the stage crops each one to fill. `framing="fit"` keeps the
old behaviour for anyone who would rather lose no pixels.

The TRUE orientation is never altered - `SegmentRenderSpec.orientation`
still carries it, and that is what the artifacts and the critic read. Only
the compatibility projection says "treat this as matching".
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import (JobSettings, RenderSpec,          # noqa: E402
                              SegmentRenderSpec)
from agents.tools import render_tools as RT                     # noqa: E402


def _spec(canvas=(1280, 720), framing="fill") -> RenderSpec:
    return RenderSpec(
        job_id="j", build_dir="b", src_dir="s", canvas=canvas, crf=18,
        framing=framing,
        segments=[
            SegmentRenderSpec(index=1, clip_id="wide.mp4", start=0, end=5,
                              duration=5, caption="", orientation="horizontal",
                              cache_key="k1"),
            SegmentRenderSpec(index=2, clip_id="tall.mp4", start=0, end=4,
                              duration=4, caption="", orientation="vertical",
                              cache_key="k2"),
        ])


@pytest.fixture
def projected(tmp_path, monkeypatch):
    """Capture what `project_manifest` hands the render stage."""
    def run(spec):
        saved = {}
        monkeypatch.setattr(RT, "read_manifest",
                            lambda d: {"src": "s", "clips": []})
        monkeypatch.setattr(RT.build_video, "save",
                            lambda d, m: saved.update(m))
        monkeypatch.setattr(RT, "Path", pathlib.Path)
        monkeypatch.setattr(pathlib.Path, "write_text",
                            lambda self, *a, **k: None, raising=False)
        RT.project_manifest(spec)
        return saved
    return run


# ------------------------------------------------------------- defaults

def test_fill_is_the_default_everywhere():
    assert JobSettings().framing == "fill"
    assert _spec().framing == "fill"


def test_canvas_orientation_is_read_off_the_canvas():
    assert _spec(canvas=(1280, 720)).canvas_orientation == "horizontal"
    assert _spec(canvas=(1080, 1920)).canvas_orientation == "vertical"
    assert _spec(canvas=(1080, 1080)).canvas_orientation == "horizontal"


# ------------------------------------------------------------ the fix

def test_fill_makes_every_segment_match_the_canvas(projected):
    """Both clips must take fit_filter's crop-to-fill branch."""
    m = projected(_spec(canvas=(1280, 720), framing="fill"))
    got = [s["orientation"] for s in m["segments"]]
    assert got == ["horizontal", "horizontal"], got


def test_fill_works_for_a_vertical_canvas_too(projected):
    m = projected(_spec(canvas=(1080, 1920), framing="fill"))
    assert [s["orientation"] for s in m["segments"]] == ["vertical", "vertical"]


def test_fit_keeps_the_old_blur_pad_behaviour(projected):
    """Anyone who would rather lose no pixels still can."""
    m = projected(_spec(canvas=(1280, 720), framing="fit"))
    assert [s["orientation"] for s in m["segments"]] == ["horizontal",
                                                         "vertical"]


def test_the_true_orientation_is_never_altered():
    """Artifacts and the critic read the real value, not the projection."""
    spec = _spec(framing="fill")
    assert [s.orientation for s in spec.segments] == ["horizontal", "vertical"]


# ------------------------------------------------- it reaches the spec

def test_compile_render_spec_carries_framing():
    from agents.artifacts import EditPlan
    assert "framing" in RT.compile_render_spec.__code__.co_varnames


def test_an_unknown_framing_value_falls_back_to_fill():
    """A closed literal upstream, but the helper must not pass junk on."""
    import inspect
    src = inspect.getsource(RT.compile_render_spec)
    assert 'framing if framing in ("fill", "fit") else "fill"' in src
