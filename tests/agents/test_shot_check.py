"""Shots chosen by looking at them; music written to the frames; stock
graphics opt-in.

What must hold:
* the candidates are the pipeline's own best windows, distinct and never a
  moment already used - the model only ranks them;
* the frame judged is the slice the viewer will see (a horizontal clip
  going into a vertical video is cropped before it is shown to the model);
* any failure falls back to the busiest window, as before;
* Lyria gets frames of the cut, and a retry after a refusal drops them;
* the stock graphics are not offered unless asked for.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from agents.artifacts import ClipAnalysisSet
from agents.tools import ai_music as AM
from agents.tools import asset_library as AL
from agents.tools import select_tools as ST
from agents.tools import shot_check as SC
from conftest import make_clip_analysis


class Span:
    def __init__(self):
        self.tools, self.notes, self.llm_calls = [], [], []

    def tool(self, name, **kw):
        self.tools.append((name, kw))

    def note(self, text):
        self.notes.append(text)

    def llm(self, *a):
        self.llm_calls.append(a)


class Reply:
    tokens_in, tokens_out, usd = 300, 20, 0.0


class Seer:
    """A vision model that answers with the scores it was given."""

    def __init__(self, scores=None, raises=False):
        self.scores, self.raises, self.calls = scores, raises, []

    def complete_json(self, prompt, max_tokens=300, images=None, **kw):
        self.calls.append((prompt, len(images or [])))
        if self.raises:
            raise RuntimeError("provider down")
        return ({"scores": self.scores} if self.scores is not None
                else None), Reply()


def busy_clip(duration=60.0):
    # four separated bursts of motion, strongest first
    series = [1.0] * int(duration * 10)
    for start, level in ((50, 20.0), (200, 16.0), (350, 12.0), (500, 8.0)):
        for i in range(start, min(start + 60, len(series))):
            series[i] = level
    return make_clip_analysis("a.mp4", duration=duration, series=series)


# ------------------------------------------------------------ candidates

def test_candidates_are_distinct_and_best_first():
    got = ST.top_windows(busy_clip(), 6.0, k=4)
    assert len(got) == 4
    scores = [s for _, _, s in got]
    assert scores == sorted(scores, reverse=True)
    for i, (a, b, _) in enumerate(got):
        for c, d, _ in got[i + 1:]:
            assert b <= c or a >= d, "candidates never overlap"


def test_candidates_never_repeat_a_moment_already_used():
    used = [(5.0, 11.0)]
    for a, b, _ in ST.top_windows(busy_clip(), 6.0, used, k=4):
        assert b <= 5.0 or a >= 11.0


def test_candidates_can_be_spread_across_the_clip():
    got = ST.top_windows(busy_clip(), 6.0, k=3, spread=15.0)
    starts = sorted(a for a, _, _ in got)
    assert all(b - a >= 15.0 for a, b in zip(starts, starts[1:]))


def test_a_shot_longer_than_the_clip_has_no_candidates():
    assert ST.top_windows(busy_clip(10.0), 20.0) == []


# -------------------------------------------------------------- frames

def test_a_frame_is_a_small_jpeg_from_the_middle(lively_clip):
    f = SC.frame_jpeg(lively_clip, 3.0)
    assert f and f[:2] == b"\xff\xd8"


def test_the_frame_is_cropped_as_the_viewer_will_see_it(lively_clip):
    """A horizontal clip in a vertical video is judged as its centre slice."""
    import struct
    f = SC.frame_jpeg(lively_clip, 3.0, canvas=(1080, 1920), fill=True)
    i = 2
    while i < len(f):                       # walk to the SOF marker
        marker, length = f[i + 1], struct.unpack(">H", f[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC2):
            h, w = struct.unpack(">HH", f[i + 5:i + 9])
            assert h > w, (w, h)
            return
        i += 2 + length
    pytest.fail("no frame header found")


def _jpeg_size(f):
    import struct
    i = 2
    while i < len(f):
        marker, length = f[i + 1], struct.unpack(">H", f[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC2):
            h, w = struct.unpack(">HH", f[i + 5:i + 9])
            return w, h
        i += 2 + length
    return None


def test_a_strip_shows_three_moments_of_the_shot_side_by_side(lively_clip):
    f = SC.strip_jpeg(lively_clip, [1.0, 3.0, 5.0])
    w, h = _jpeg_size(f)
    assert 2.5 * h < w, "three frames wide, one high"


def test_a_strip_is_cropped_as_the_viewer_will_see_it(lively_clip):
    f = SC.strip_jpeg(lively_clip, [1.0, 3.0, 5.0], canvas=(1080, 1920),
                      fill=True)
    w, h = _jpeg_size(f)
    assert abs(w / h - 3 * 1080 / 1920) < 0.1


def test_an_unreadable_file_gives_no_frame(tmp_path):
    assert SC.frame_jpeg(tmp_path / "missing.mp4", 1.0) is None


# ------------------------------------------------------------ choosing

def _frames(monkeypatch):
    monkeypatch.setattr(SC, "strip_jpeg", lambda *a, **k: b"\xff\xd8" + b"x" * 800)


def test_the_best_looking_candidate_wins_over_the_busiest(monkeypatch):
    _frames(monkeypatch)
    clip, span = busy_clip(), Span()
    cands = ST.top_windows(clip, 6.0, k=4)
    monkeypatch.setattr(SC, "CANDIDATES", 4)
    cands = ST.top_windows(clip, 6.0, k=4, spread=clip.technical.duration / 8)
    got = SC.choose(Seer([2, 3, 9, 4]), span, clip, 6.0, [], "the gears",
                    (1920, 1080), "fill")
    assert got is not None
    assert (got[0], got[1]) == (cands[2][0], cands[2][1])
    assert any("chose #3" in n for n in got[3])
    assert span.llm_calls, "the request's tokens are accounted for"


def test_no_answer_falls_back_to_the_busiest_window(monkeypatch):
    _frames(monkeypatch)
    for seer in (Seer(None), Seer(raises=True), Seer([1, 2])):
        assert SC.choose(seer, Span(), busy_clip(), 6.0, [], "x",
                         (1920, 1080), "fill") is None


def test_the_model_is_told_what_the_shot_should_show(monkeypatch):
    _frames(monkeypatch)
    monkeypatch.setattr(SC, "CANDIDATES", 4)
    seer = Seer([5, 5, 5, 5])
    SC.choose(seer, Span(), busy_clip(), 6.0, [], "the red gear turning",
              (1920, 1080), "fill", about="a motorised build")
    prompt, n = seer.calls[0]
    assert "the red gear turning" in prompt and "a motorised build" in prompt
    assert "three moments" in prompt
    assert n == 4


def test_scores_are_clamped_and_must_match_the_frames():
    assert SC._scores({"scores": [12, -3, "7"]}, 3) == [10.0, 0.0, 7.0]
    assert SC._scores({"scores": [1, 2]}, 3) is None
    assert SC._scores({"scores": ["x", 1, 2]}, 3) is None


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AT_SHOT_CHECK", "0")
    assert SC.enabled() is False
    monkeypatch.setenv("AT_SHOT_CHECK", "1")
    assert SC.enabled() is True


# ------------------------------------------- inside the editing director

def test_the_director_uses_the_checked_shot_and_keeps_the_fallback():
    from agents.editing_director import EditingDirectorAgent
    clip = busy_clip()
    cs = ClipAnalysisSet(clips=[clip], orientation="horizontal",
                         canvas=(1920, 1080), target_seconds=12.0)
    proposal = {"segments": [{"clip_id": "a.mp4", "target_seconds": 6.0,
                              "reason": "open on it"},
                             {"clip_id": "a.mp4", "target_seconds": 6.0,
                              "reason": "then again"}]}
    asked = []

    def screen(c, want, taken, purpose):
        asked.append(purpose)
        return (40.0, 46.0, 1.0, ["checked"]) if purpose == "then again" else None

    segs, notes = EditingDirectorAgent()._segments_from_proposal(
        Span(), cs, proposal, {}, screen=screen)
    assert asked == ["open on it", "then again"]
    assert (segs[1].start, segs[1].end) == (40.0, 46.0)
    assert segs[0].end > segs[0].start, "the first fell back to the search"


# ------------------------------------------------------------ music

class Http:
    def __init__(self):
        self.bodies = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        if "metadata.google.internal" in url:
            raise OSError("not on a VM")
        self.bodies.append(json.loads(req.data))

        class R:
            def read(self_):
                return json.dumps({"status": "completed", "outputs": [
                    {"type": "audio", "mime_type": "audio/mpeg",
                     "data": base64.b64encode(b"ID3" + b"0" * 2000).decode()}]
                }).encode()

            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False
        return R()


def test_lyria_is_sent_the_frames_of_the_cut(tmp_path, monkeypatch):
    from test_ai_music import a_music
    monkeypatch.setenv("AT_MUSIC_PROJECT", "test-project-123")
    http = Http()
    path, plan = AM.compose(a_music(), "j", tmp_path, 28.0, [9.6], http=http,
                            token="t", images=[b"\xff\xd8one", b"\xff\xd8two"])
    parts = http.bodies[0]["input"]
    assert [p["type"] for p in parts] == ["text", "image", "image"]
    assert "frames from this video" in parts[0]["text"]
    assert plan["frames"] == 2 and path is not None


def test_the_plain_retry_sends_no_frames(tmp_path, monkeypatch):
    from test_ai_music import a_music
    p = AM.build_prompt(a_music(), 28.0, [], 90.0, 55, plain=True)
    assert "frames" not in p


# --------------------------------------------------- the stock graphics

def test_the_stock_graphics_are_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("AT_STOCK_GRAPHICS", raising=False)
    assert AL.stock_enabled() is False
    monkeypatch.setenv("AT_STOCK_GRAPHICS", "1")
    assert AL.stock_enabled() is True


def test_the_animation_agent_only_installs_them_when_asked():
    import inspect
    import agents.animation as A
    src = inspect.getsource(A)
    assert "_AL.ensure_defaults(ctx.job.build_dir)\n                if _AL.stock_enabled()" in src
