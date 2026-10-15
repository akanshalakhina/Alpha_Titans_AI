"""The score must follow what the Music agent decided, not just its mood.

THE GAP THIS CLOSES
-------------------
`extras.MOODS` maps a mood to `--style lofi` or `--style bright` and
nothing else. Seven moods therefore produced TWO scores, and the same
mood produced a byte-identical wav every single time - while the Music
agent had reasoned about a tempo, an energy arc, an instrumentation and a
per-beat section plan that reached nothing at all.

`extras.py` and all three `pipeline/*.py` are protected, but the engine
already takes `bpm`, `root_midi` and `seed`; only `style` was ever
passed. So the engine is imported as a LIBRARY, driven from the whole
artifact, and the wav goes to `music_stage` down the path it already has
for an operator's own track.

The seed is a hash of what the agent wrote, never of the clock: the same
reasoning has to give the same score, or two builds of one brief could
never be compared.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import Music, MusicSection            # noqa: E402
from agents.tools import score_tools as SC                  # noqa: E402


def a_music(**kw) -> Music:
    base = dict(music_type="reasoned", mood="upbeat", tempo_feel="driving",
                engine="stdlib", sonic_direction="punchy rhythmic pulse",
                energy_arc="rises to the payoff",
                instrumentation=["bright synth"], avoid=["slow tempos"])
    base.update(kw)
    return Music(**base)


def _sections(*energies):
    return [MusicSection(index=i + 1, beat_index=i + 1, clip_id="c.mp4",
                         energy=e, intent="x")
            for i, e in enumerate(energies)]


def _md5(p) -> str:
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# =============================================== choosing the engine's knobs

def test_the_tempo_feeling_picks_the_tempo_band():
    fast = SC.settings_for(a_music(tempo_feel="driving"), "j", [])
    slow = SC.settings_for(a_music(tempo_feel="slow"), "j", [])
    assert fast["bpm_lo"] > slow["bpm_hi"], \
        "driving and slow must not overlap at all"


def test_each_mood_gets_its_own_key():
    """Two moods at one tempo were the same tune; the root stops that."""
    roots = {m: SC.settings_for(a_music(mood=m), "j", [])["root_midi"]
             for m in ("lofi", "calm", "bright", "upbeat", "energetic")}
    assert len(set(roots.values())) >= 4, roots


def test_the_mood_still_decides_the_voicing():
    assert SC.settings_for(a_music(mood="calm"), "j", [])["style"] == "lofi"
    assert SC.settings_for(a_music(mood="upbeat"), "j", [])["style"] == "bright"


def test_the_style_map_agrees_with_the_engines_own():
    """`extras.MOODS` is what actually ran before. The two must not drift
    apart about what a mood sounds like."""
    from agents.tools.pipeline_bridge import extras_mod
    for mood, (engine, extra) in extras_mod.MOODS.items():
        if engine != "stdlib":
            continue                       # beatbed has no --style at all
        assert SC.STYLE_FOR_MOOD[mood] == extra[extra.index("--style") + 1]


def test_every_mood_and_tempo_in_the_vocabulary_is_mapped():
    import typing
    from agents.artifacts import Mood, TempoFeel
    for m in typing.get_args(Mood):
        assert m in SC.STYLE_FOR_MOOD and m in SC.ROOT_FOR_MOOD, m
    for t in typing.get_args(TempoFeel):
        assert t in SC.TEMPO_BANDS, t


# ------------------------------------------------------------- the arc

def test_a_rising_arc_sits_lower_in_the_band_than_a_peaking_one():
    rising = SC.settings_for(a_music(sections=_sections("low", "building")),
                             "j", [])
    peaking = SC.settings_for(a_music(sections=_sections("high", "peak")),
                              "j", [])
    assert rising["arc_pull"] < 0 < peaking["arc_pull"]
    assert rising["bpm_hi"] < peaking["bpm_hi"]


def test_no_sections_is_simply_the_middle_of_the_band():
    s = SC.settings_for(a_music(sections=[]), "j", [])
    assert s["arc_pull"] == 0.0
    lo, hi = SC.TEMPO_BANDS["driving"]
    assert lo <= s["bpm_lo"] < s["bpm_hi"] <= hi


# ------------------------------------------------- reproducible, not random

def test_the_same_reasoning_gives_the_same_seed():
    assert SC.settings_for(a_music(), "j", [])["seed"] == \
        SC.settings_for(a_music(), "j", [])["seed"]


@pytest.mark.parametrize("change", [
    {"mood": "calm"}, {"tempo_feel": "slow"},
    {"sonic_direction": "something else entirely"},
    {"energy_arc": "falls away"}, {"instrumentation": ["piano"]},
    {"avoid": ["drums"]}, {"sections": _sections("peak")},
])
def test_a_different_brief_gives_a_different_seed(change):
    assert SC.settings_for(a_music(**change), "j", [])["seed"] != \
        SC.settings_for(a_music(), "j", [])["seed"], \
        "this decision reached the score in name only"


def test_two_jobs_with_one_brief_still_differ():
    """Otherwise every video of the same mood shares one soundtrack."""
    assert SC.settings_for(a_music(), "job-a", [])["seed"] != \
        SC.settings_for(a_music(), "job-b", [])["seed"]


# ===================================================== rendering for real

def test_the_engine_writes_a_real_wav(tmp_path):
    out, plan = SC.compose(a_music(), "j", tmp_path / "s.wav", 4.0, [1.5])
    assert out is not None and out.stat().st_size > 1000, plan
    import wave
    with wave.open(str(out)) as w:
        # the engine's own sample rate, read off it - a number copied here
        # would silently stop matching the day the engine changed
        assert w.getnchannels() == 2
        assert w.getframerate() == SC._engine().SR
    lo, hi = SC.TEMPO_BANDS["driving"]
    assert plan["bpm"] and lo <= plan["bpm"] <= hi


def test_a_different_brief_sounds_different(tmp_path):
    a, _ = SC.compose(a_music(), "j", tmp_path / "a.wav", 4.0, [1.5])
    b, _ = SC.compose(a_music(mood="calm", tempo_feel="slow"), "j",
                      tmp_path / "b.wav", 4.0, [1.5])
    assert _md5(a) != _md5(b), "the same two bytes for every mood was the bug"


def test_the_same_brief_sounds_identical(tmp_path):
    a, _ = SC.compose(a_music(), "j", tmp_path / "a.wav", 4.0, [1.5])
    b, _ = SC.compose(a_music(), "j", tmp_path / "b.wav", 4.0, [1.5])
    assert _md5(a) == _md5(b)


# ------------------------------------------------------- failing safely

def test_a_missing_engine_falls_back_rather_than_going_silent(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(SC, "_engine", lambda: None)
    out, plan = SC.compose(a_music(), "j", tmp_path / "s.wav", 4.0, [1.5])
    assert out is None and "not found" in plan["error"]


def test_an_engine_that_throws_is_reported_not_swallowed(tmp_path,
                                                         monkeypatch):
    class Boom:
        @staticmethod
        def best_bpm(*a, **k):
            raise RuntimeError("no")

    monkeypatch.setattr(SC, "_engine", lambda: Boom)
    out, plan = SC.compose(a_music(), "j", tmp_path / "s.wav", 4.0, [1.5])
    assert out is None and "RuntimeError: no" in plan["error"]


# ------------------------------------------- the shape handed to the engine

def test_the_cut_length_accounts_for_every_crossfade():
    from agents.artifacts import SegmentRenderSpec
    from agents.tools.render_tools import _cut_shape

    class Spec:
        xfade_duration = 0.4
        segments = [SegmentRenderSpec(index=i + 1, clip_id="c", start=0.0,
                                      end=6.0, duration=6.0, caption="",
                                      orientation="horizontal",
                                      cache_key="k")
                    for i in range(5)]

    total, marks = _cut_shape(Spec())
    assert total == pytest.approx(30.0 - 4 * 0.4)      # the 28.4s observed
    assert len(marks) == 4 and marks[0] == pytest.approx(5.6)
    assert marks == sorted(marks)
