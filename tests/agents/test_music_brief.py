"""The spoken brief must reach the Music model, not just its cache key.

THE BUG THIS PINS
-----------------
`JobSettings.spoken` was hashed into the Music agent's `inputs_hash` from
the start, but `_build_prompt` was never given it. The cache key therefore
said the brief mattered while the prompt proved it did not: two productions
over the same footage, differing ONLY in the dictated brief, both missed the
cache (different key), both called the model, and both came back `calm` -
because the only thing the model could actually see was the measured motion,
which is identical in both. Observed on real footage: "Keep it calm and
slow" and "Make it fast and punchy, high energy" both scored `calm`.

`music` is deliberately absent from the brief: an explicit mood choice is
applied as a hard override after the agent replies, so repeating it in the
prompt would invite the model to argue with a settled decision.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import JobSettings          # noqa: E402
from agents.music import MusicAgent               # noqa: E402


def _prompt(brief: str) -> str:
    return MusicAgent()._build_prompt(None, None, None, brief)


def test_the_dictated_words_reach_the_model():
    p = _prompt("Make it fast and punchy. High energy.")
    assert "fast and punchy" in p
    assert "High energy" in p


def test_two_different_briefs_give_two_different_prompts():
    """The whole point: identical prompts cannot produce different moods."""
    a = _prompt("Keep it calm and slow. Let each shot breathe.")
    b = _prompt("Make it fast and punchy. Cut hard on the action.")
    assert a != b


def test_no_brief_adds_no_section():
    """A run with nothing dictated must look exactly as it did before."""
    assert _prompt("") == _prompt("   ")
    assert "WHAT THE PERSON ASKED FOR" not in _prompt("")


def test_the_brief_is_framed_as_intent_not_as_a_mood():
    p = _prompt("make it sound like a horror trailer")
    assert "INTENT" in p
    # it must not be able to smuggle in a mood the engine cannot play
    assert "cannot name a mood outside the list" in p


def test_the_settings_brief_is_what_gets_passed():
    """`brief()` renders the closed choices as words and omits `music`."""
    s = JobSettings(voice_transcript="open on the loudest moment",
                    style="cinematic", pacing="fast", music="energetic")
    b = s.brief()
    assert "open on the loudest moment" in b
    assert "cinematic" in b and "fast pacing" in b
    assert "energetic" not in b, "the mood override must not be suggested"
    p = _prompt(b)
    assert "open on the loudest moment" in p
    # `energetic` still appears further down as one of the moods the engine
    # offers - that is the list, not a suggestion. It must not appear in the
    # brief section, which is the part framed as what the person wants.
    section = p.split("WHAT THE PERSON ASKED FOR", 1)[1].split("THE STORY", 1)[0]
    assert "energetic" not in section


def test_a_multi_line_brief_stays_readable():
    p = _prompt("typed line\ndictated line")
    assert "  typed line" in p
    assert "  dictated line" in p
