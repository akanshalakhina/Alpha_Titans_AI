"""Graphics over the cut, and an intro/outro around it.

THE LIMIT THIS REMOVES
----------------------
The renderer could crop, scale, draw text, blur and crossfade. It had no
way to put an image on the screen at all, so a brief asking for stickers
or a logo could never be honoured however well any agent reasoned about
it.

THE RULE IT MUST NOT BREAK
--------------------------
The agent layer names an asset, a beat and an ANCHOR. It never names a
pixel or a second. `OverlayPlacement` forbids extra fields, so a reply
carrying `x`, `y`, `start` or `end` fails validation outright - the model
has no vocabulary for geometry or time, and therefore cannot put a
sticker three seconds into a two-second shot.

Everything that IS a number - the margin, the scaled width, the
`between(t,a,b)` window - is computed in `build_video` from the canvas
and the segment's own measured duration.

AND THE ONE THAT MATTERS MOST
-----------------------------
A build that asks for no graphics must produce the filtergraph it always
produced, byte for byte. That is asserted below, because every other
guarantee in this file is worthless if adding the feature changed the
videos that do not use it.
"""

from __future__ import annotations

import json
import pathlib
import sys
import typing

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "pipeline"))

import build_video as BV                                     # noqa: E402
from agents.artifacts import (Animation, AnimationDirection,  # noqa: E402
                              EditPlan, OverlayAnchor, OverlayHold,
                              OverlayPlacement, Segment)
from agents.tools import render_tools as RT                   # noqa: E402

CANVAS = (1920, 1080)


def a_seg(index=1, start=0.0, end=6.0, caption="", clip="a.mp4"):
    return {"index": index, "clip": clip, "start": start, "end": end,
            "orientation": "horizontal", "caption": caption}


@pytest.fixture
def asset(tmp_path):
    p = tmp_path / "sparkle.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return p


def _ov(asset, **kw):
    base = {"path": str(asset), "anchor": "top-right", "scale_pct": 18,
            "hold": "full"}
    base.update(kw)
    return [base]


# ============================= 1. nothing asked for, nothing changed

@pytest.mark.parametrize("caption", ["", "a caption"])
def test_a_build_with_no_overlays_is_byte_identical(caption):
    """The whole feature is worthless if it altered the videos that do
    not use it."""
    seg = a_seg(caption=caption)
    before, lbefore, _ = BV.fit_filter(seg, CANVAS)
    after, lafter, ins = BV.fit_filter(seg, CANVAS, None, overlays=None)

    assert after == before and lafter == lbefore
    assert ins == [], "no extra -i input is added"
    assert "overlay=x=" not in after


def test_an_empty_overlay_list_is_the_same_as_none():
    seg = a_seg()
    assert BV.fit_filter(seg, CANVAS, None, []) == BV.fit_filter(seg, CANVAS)


def test_no_overlays_json_means_no_overlays(tmp_path):
    assert BV._read_overlays(str(tmp_path)) == {}


# ========================= 2. the filtergraph it actually generates

def test_an_overlay_is_chained_after_the_caption(asset):
    seg = a_seg(caption="hello")
    base, _, _ = BV.fit_filter(seg, CANVAS)
    graph, label, ins = BV.fit_filter(seg, CANVAS, None, _ov(asset))

    assert graph.startswith(base), "the existing chain is untouched"
    tail = graph[len(base):]
    assert "[vc][ov0]overlay=" in tail, "it builds on the captioned label"
    assert label == "[vo0]" and ins == ["-i", str(asset)]


def test_an_overlay_without_a_caption_builds_on_the_plain_label(asset):
    graph, label, _ = BV.fit_filter(a_seg(), CANVAS, None, _ov(asset))
    assert "[v][ov0]overlay=" in graph and label == "[vo0]"


def test_two_overlays_chain_onto_each_other(asset, tmp_path):
    second = tmp_path / "logo.png"
    second.write_bytes(asset.read_bytes())
    ovs = _ov(asset) + _ov(second, anchor="bottom-left")
    graph, label, ins = BV.fit_filter(a_seg(), CANVAS, None, ovs)

    assert "[vo0][ov1]overlay=" in graph, "the second sits on the first"
    assert label == "[vo1]" and len(ins) == 4


# ============================== 3. words become numbers, here not there

def test_every_anchor_gives_a_different_position(asset):
    seen = {}
    for anchor in typing.get_args(OverlayAnchor):
        graph, _, _ = BV.fit_filter(a_seg(), CANVAS, None,
                                    _ov(asset, anchor=anchor))
        expr = graph.split("overlay=")[1].split(":enable")[0]
        assert expr not in seen, f"{anchor} collides with {seen.get(expr)}"
        seen[expr] = anchor
    assert len(seen) == 9


def test_the_margin_is_a_share_of_the_canvas_not_a_constant(asset):
    wide, _, _ = BV.fit_filter(a_seg(), (1920, 1080), None, _ov(asset))
    small, _, _ = BV.fit_filter(a_seg(), (1280, 720), None, _ov(asset))
    assert "y=77" in wide, "4% of 1920"
    assert "y=51" in small, "4% of 1280"


def test_the_scale_is_a_share_of_the_canvas_not_of_the_asset(asset):
    graph, _, _ = BV.fit_filter(a_seg(), CANVAS, None,
                                _ov(asset, scale_pct=25))
    assert "scale=480:-1" in graph, "25% of 1920, aspect preserved"


def test_an_absurd_scale_is_clamped(asset):
    for pct, want in ((0.1, 2.0), (900, 60.0)):
        graph, _, _ = BV.fit_filter(a_seg(), CANVAS, None,
                                    _ov(asset, scale_pct=pct))
        assert "scale=%d:-1" % round(1920 * want / 100) in graph


@pytest.mark.parametrize("hold, lo, hi", [
    # the 0.15s edge guard applies at whichever end touches the cut, so a
    # graphic never shares a frame with the crossfade either side of it
    ("full", 0.15, 5.85), ("open", 0.15, 2.70), ("close", 3.30, 5.85)])
def test_the_hold_window_comes_from_the_segments_own_duration(
        asset, hold, lo, hi):
    graph, _, _ = BV.fit_filter(a_seg(end=6.0), CANVAS, None,
                                _ov(asset, hold=hold))
    assert "between(t\\,%.2f\\,%.2f)" % (lo, hi) in graph


def test_a_longer_segment_holds_the_graphic_longer(asset):
    short, _, _ = BV.fit_filter(a_seg(end=4.0), CANVAS, None, _ov(asset))
    long_, _, _ = BV.fit_filter(a_seg(end=12.0), CANVAS, None, _ov(asset))
    assert "3.85" in short and "11.85" in long_


def test_a_segment_too_short_to_hold_one_is_skipped(asset, capsys):
    graph, label, ins = BV.fit_filter(a_seg(end=0.2), CANVAS, None,
                                      _ov(asset, hold="close"))
    assert ins == [] and "overlay=x=" not in graph
    assert "too short" in capsys.readouterr().out


# ============================================= 4. refusing bad material

def test_a_missing_asset_is_skipped_not_fatal(tmp_path, capsys):
    graph, label, ins = BV.fit_filter(
        a_seg(), CANVAS, None, _ov(tmp_path / "nope.png"))
    assert ins == [] and "overlay=x=" not in graph
    assert "missing" in capsys.readouterr().out, \
        "losing a sticker is a disappointment; losing the video is a defect"


def test_an_unreadable_overlays_file_is_reported_and_ignored(tmp_path,
                                                             capsys):
    (tmp_path / "overlays.json").write_text("{not json")
    assert BV._read_overlays(str(tmp_path)) == {}
    assert "unreadable" in capsys.readouterr().out


def test_an_asset_name_cannot_escape_the_build(tmp_path):
    (tmp_path / "overlays.json").write_text(json.dumps(
        [{"segment_index": 1, "asset": "../../../etc/passwd"}]))
    got = BV._read_overlays(str(tmp_path))
    assert ".." not in got[1][0]["path"]
    assert got[1][0]["path"].endswith("passwd")
    assert "assets" in got[1][0]["path"], "it is forced inside assets/"


def test_a_row_with_no_segment_index_is_skipped(tmp_path, capsys):
    (tmp_path / "overlays.json").write_text(json.dumps(
        [{"asset": "x.png"}, {"segment_index": 2, "asset": "y.png"}]))
    assert list(BV._read_overlays(str(tmp_path))) == [2]


# ====================================== 5. the vocabularies cannot drift

def test_the_anchors_are_the_renderers_own():
    assert set(typing.get_args(OverlayAnchor)) == set(BV.ANCHORS)


def test_the_holds_are_the_renderers_own():
    assert set(typing.get_args(OverlayHold)) == set(BV.HOLDS)


def test_the_model_has_no_vocabulary_for_geometry_or_time():
    for bad in ({"x": 10}, {"y": 10}, {"start": 1.0}, {"end": 2.0},
                {"t": 1.0}, {"px": 3}, {"filter": "overlay"}):
        with pytest.raises(Exception):
            OverlayPlacement(asset="a.png", **bad)


def test_a_placement_is_accepted_with_words_alone():
    p = OverlayPlacement(asset="a.png", anchor="center", hold="open")
    assert p.scale_pct == 18.0 and p.why == ""


# ================================== 6. beats resolve onto segments

def _plan(*clips):
    return EditPlan(
        target_seconds=30.0, total_duration=len(clips) * 6.0,
        canvas=CANVAS,
        segments=[Segment(index=i + 1, clip_id=c, start=0.0, end=6.0,
                          duration=6.0, orientation="horizontal")
                  for i, c in enumerate(clips)])


def _animation(*directions):
    return Animation(animation_type="reasoned", directions=list(directions))


def test_a_beats_graphic_reaches_every_segment_cut_from_that_clip():
    """A clip used twice is two moments of one beat."""
    plan = _plan("a.mp4", "b.mp4", "a.mp4")
    anim = _animation(AnimationDirection(
        index=1, beat_index=1, clip_id="a.mp4",
        overlay=OverlayPlacement(asset="s.png")))
    got = RT._overlays_by_segment(plan, anim)
    assert sorted(got) == [1, 3], "both cuts from a.mp4, not b.mp4"


def test_no_animation_means_no_overlays():
    assert RT._overlays_by_segment(_plan("a.mp4"), None) == {}


def test_directions_without_a_graphic_add_nothing():
    anim = _animation(AnimationDirection(index=1, beat_index=1,
                                         clip_id="a.mp4"))
    assert RT._overlays_by_segment(_plan("a.mp4"), anim) == {}


def test_a_graphic_for_a_clip_that_was_cut_lands_nowhere():
    anim = _animation(AnimationDirection(
        index=1, beat_index=1, clip_id="dropped.mp4",
        overlay=OverlayPlacement(asset="s.png")))
    assert RT._overlays_by_segment(_plan("a.mp4"), anim) == {}


# =========================================== 7. intro and outro

def test_no_roll_in_the_assets_folder_means_none(tmp_path):
    assert BV._roll_source(str(tmp_path), "intro") is None


def test_a_roll_is_found_by_name(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "intro.mp4").write_bytes(b"x")
    (assets / "something-else.mp4").write_bytes(b"x")
    assert BV._roll_source(str(tmp_path), "intro").endswith("intro.mp4")
    assert BV._roll_source(str(tmp_path), "outro") is None


def test_a_still_image_is_not_a_roll(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "intro.png").write_bytes(b"x")
    assert BV._roll_source(str(tmp_path), "intro") is None


def test_attaching_nothing_leaves_the_cut_alone(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(BV, "sh", lambda *a, **k: called.append(a) or (0, ""))
    m = {"transitions": [5.6, 11.2], "total_duration": 17.0,
         "canvas": list(CANVAS)}
    BV._attach_rolls(str(tmp_path), m, str(tmp_path / "_joined.mp4"),
                     canvas=CANVAS)
    assert called == [], "ffmpeg was never invoked"
    assert m["transitions"] == [5.6, 11.2], "the beat marks are untouched"


# ================================ 8. the prompt is written from the folder
#
# A model cannot name a file that does not exist, and inviting it to try
# is how `lower_third_blue.png` ends up in an artifact with nothing on
# screen. So the asset paragraph is generated from the job's own folder.

from agents.tools import asset_library as AL                 # noqa: E402


def test_an_empty_folder_keeps_the_original_refusal(tmp_path):
    said = AL.describe(tmp_path)
    assert "Do NOT name an overlay" in said
    assert ".png" not in said, "nothing is offered that is not there"


def test_no_build_directory_at_all_is_the_same_refusal():
    assert "Do NOT name an overlay" in AL.describe(None)
    assert AL.available(None) == [] and AL.ensure_defaults(None) == []


def test_the_defaults_are_real_pngs(tmp_path):
    made = AL.ensure_defaults(tmp_path)
    assert sorted(made) == sorted(AL.DEFAULTS)
    for name in made:
        raw = (AL.assets_dir(tmp_path) / name).read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        import struct
        w, h, depth, colour = struct.unpack(">IIBB", raw[16:26])
        assert (w, h, depth, colour) == (AL.SIZE, AL.SIZE, 8, 6), \
            "8-bit RGBA; a graphic with no alpha is a rectangle stuck on"


def test_the_operators_own_artwork_is_never_overwritten(tmp_path):
    AL.ensure_defaults(tmp_path)
    mine = AL.assets_dir(tmp_path) / "sparkle.png"
    mine.write_bytes(b"\x89PNG\r\n\x1a\nMINE")
    assert AL.ensure_defaults(tmp_path) == [], "nothing was remade"
    assert mine.read_bytes().endswith(b"MINE")


def test_rolls_are_not_offered_as_stickers(tmp_path):
    d = AL.assets_dir(tmp_path)
    d.mkdir(parents=True)
    for n in ("intro.mp4", "outro.mp4", "intro.png", "logo.png"):
        (d / n).write_bytes(b"x")
    assert AL.available(tmp_path) == ["logo.png"], \
        "a pre-roll is a different feature with a different vocabulary"


def test_a_jpg_is_not_offered(tmp_path):
    d = AL.assets_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "photo.jpg").write_bytes(b"x")
    assert AL.available(tmp_path) == []


def test_the_filled_prompt_lists_exactly_what_is_there(tmp_path):
    AL.ensure_defaults(tmp_path)
    said = AL.describe(tmp_path)
    assert "Do NOT name an overlay" not in said
    for name in AL.DEFAULTS:
        assert name in said
    assert "the ONLY ones you may name" in said


# -------------------------------------- the allowlist is actually enforced

def _direction(asset):
    return {"beat_index": 1, "emphasis": "none", "caption_intent": "none",
            "intent": "x", "overlay": {"asset": asset}}


def _agent():
    from agents.animation import AnimationAgent
    return AnimationAgent()


def test_a_graphic_the_job_has_is_kept():
    dropped = []
    got = _agent()._overlay_for(_direction("sparkle.png"),
                                {"sparkle.png"}, dropped)
    assert got and got.asset == "sparkle.png" and dropped == []


def test_a_graphic_the_job_does_not_have_is_dropped_and_named():
    dropped = []
    assert _agent()._overlay_for(_direction("lower_third_blue.png"),
                                 {"sparkle.png"}, dropped) is None
    assert dropped == ["lower_third_blue.png"], \
        "a placement with nothing behind it is the quiet kind of wrong"


def test_an_asset_path_is_reduced_to_its_name():
    dropped = []
    assert _agent()._overlay_for(_direction("../../etc/passwd"),
                                 {"sparkle.png"}, dropped) is None
    assert dropped == ["passwd"]


def test_a_direction_with_no_overlay_is_unaffected():
    item = {"beat_index": 1, "intent": "x"}
    assert _agent()._overlay_for(item, {"sparkle.png"}, []) is None


def test_an_out_of_range_scale_loses_the_graphic_not_the_direction():
    dropped = []
    item = _direction("sparkle.png")
    item["overlay"]["scale_pct"] = 500
    assert _agent()._overlay_for(item, {"sparkle.png"}, dropped) is None
    assert dropped == ["sparkle.png"]


def test_an_unknown_anchor_falls_back_rather_than_failing():
    item = _direction("sparkle.png")
    item["overlay"]["anchor"] = "somewhere-nice"
    got = _agent()._overlay_for(item, {"sparkle.png"}, [])
    assert got.anchor == "top-right"
