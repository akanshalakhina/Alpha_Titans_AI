"""A reasoned answer must not be lost to punctuation.

WHAT THIS GUARDS
----------------
On 14 Oct the daily quota pushed the model chain onto a smaller model and
the Story node reported `story reply unusable: not a JSON object`. A
whole reasoned narrative - hook, setup, develop, payoff - was thrown away
and replaced by the deterministic fallback, because of formatting.

`vision._json_from` finds the first balanced `{...}`. It is a protected
file and it is right most of the time, with two blind spots a smaller
model walks into: it counts braces without regard for string literals,
and when the first candidate fails to parse it stops rather than trying
the next one.

Three defences, in order of preference:
  1. ask for JSON in the request itself (`response_format`), so the
     model is constrained rather than merely instructed;
  2. repair what can be recovered from the reply as written;
  3. tell the model exactly what was wrong and ask once more.

What is NOT done is guessing. A reply that cannot be parsed returns None
and the caller falls back deterministically, marked degraded.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.tools import json_repair as JR                  # noqa: E402
from agents.tools import llm_tools as LT                    # noqa: E402

GOOD = '{"beats": [{"role": "hook", "clip_id": "a.mp4"}]}'


# ============================================ what a clean reply still does

def test_a_clean_reply_is_untouched():
    value, how = JR.parse(GOOD)
    assert how == "clean" and value["beats"][0]["role"] == "hook"


def test_a_clean_array_is_untouched():
    assert JR.parse('[1, 2, 3]') == ([1, 2, 3], "clean")


# ================================================== what it recovers

def test_a_markdown_fence_is_stripped():
    value, how = JR.parse("Sure, here you go:\n```json\n" + GOOD + "\n```")
    assert value["beats"] and how in ("clean", "extracted")


def test_a_trailing_comma_is_repaired():
    value, how = JR.parse('{"a": 1, "b": 2,}')
    assert value == {"a": 1, "b": 2} and how == "trailing-comma"


def test_a_trailing_comma_in_a_list_is_repaired():
    value, _ = JR.parse('{"beats": ["hook", "payoff",]}')
    assert value == {"beats": ["hook", "payoff"]}


def test_a_brace_inside_a_string_does_not_break_the_scan():
    """`_json_from` counts braces blind to string literals, so a caption
    containing one threw its depth count off and lost the whole object."""
    value, how = JR.parse('prose first {"text": "use the { brace", "n": 1}')
    assert value == {"text": "use the { brace", "n": 1}
    assert how == "extracted"


def test_an_example_object_before_the_real_one_is_stepped_over():
    """`_json_from` stops at the first candidate that fails to parse."""
    chatty = 'For example {not valid json at all} - here is the answer: ' + GOOD
    value, _ = JR.parse(chatty)
    assert value["beats"][0]["clip_id"] == "a.mp4"


def test_prose_on_both_sides_is_ignored():
    value, _ = JR.parse("Certainly! " + GOOD + " Let me know if you need more.")
    assert value["beats"]


def test_an_escaped_quote_inside_a_string_is_handled():
    value, _ = JR.parse(r'x {"t": "she said \"go\" {", "n": 2}')
    assert value["n"] == 2


# ===================================================== what it refuses

@pytest.mark.parametrize("text", [
    "", "   ", "I'm sorry, I can't help with that.",
    "The answer is 42.", "```\nnot json\n```",
])
def test_an_unusable_reply_is_refused_not_guessed_at(text):
    value, how = JR.parse(text)
    assert value is None, "a guessed answer is worse than an honest fallback"
    assert how in ("empty", "unparseable", "not-an-object")


def test_a_bare_scalar_is_not_an_object():
    assert JR.parse("42")[0] is None
    assert JR.parse('"just a string"')[0] is None


def test_nothing_is_invented_for_a_missing_field():
    value, _ = JR.parse('{"beats": []}')
    assert value == {"beats": []}, "an empty list stays empty"


# ======================================================= the complaint

def test_the_complaint_quotes_what_came_back():
    msg = JR.complain("Sure! Here is your story in plain English.")
    assert "Sure! Here is your story" in msg
    assert "no markdown" in msg and "trailing commas" in msg


def test_the_complaint_is_bounded():
    assert len(JR.complain("x" * 5000)) < 400, "a retry prompt is not a dump"


def test_the_retry_prompt_keeps_the_original_task():
    """A bare correction with no question attached produces an apology,
    not an answer."""
    out = JR.with_complaint("Write the story beats.", "nope")
    assert out.startswith("Write the story beats.")
    assert "could not be parsed" in out


# ================================================ the client, end to end

class _Say(LT.LLMClient):
    """Answers with each scripted reply in turn, counting the calls."""

    def __init__(self, *replies):
        self._transport = None
        self._replies, self.prompts = list(replies), []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None):
        self.prompts.append(prompt)
        text = self._replies[min(len(self.prompts) - 1,
                                 len(self._replies) - 1)]
        return LT.LLMReply(text=text, meta={"model": "m"})


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(LT, "_env", lambda k, d=None:
                        "k" if k.endswith("API_KEY") else
                        ("m" if k.endswith("MODEL") else d))


def test_a_chatty_reply_is_recovered_without_a_second_call(keyed):
    c = _Say("Of course!\n```json\n" + GOOD + "\n```")
    parsed, reply = c.complete_json("go", 500)
    assert parsed["beats"] and len(c.prompts) == 1, "no retry was needed"


def test_a_repair_is_recorded_so_it_is_not_silent(keyed):
    c = _Say('{"a": 1,}')
    _parsed, reply = c.complete_json("go", 500)
    assert reply.meta["json_repair"] == "trailing-comma"


def test_an_unusable_reply_is_asked_again_with_the_reason(keyed):
    c = _Say("I think the story should be about friendship.", GOOD)
    parsed, reply = c.complete_json("Write the beats.", 500)

    assert parsed["beats"], "the second answer was accepted"
    assert len(c.prompts) == 2
    assert "Write the beats." in c.prompts[1], "the task went with it"
    assert "friendship" in c.prompts[1], "the model was told what it sent"
    assert reply.meta["json_retry"] is True


def test_it_gives_up_after_one_retry(keyed):
    """A model that cannot produce JSON twice will not on the third try,
    and the deterministic fallback is waiting."""
    c = _Say("nope", "still nope")
    parsed, _reply = c.complete_json("go", 500)
    assert parsed is None and len(c.prompts) == 2


def test_a_clean_first_answer_is_never_asked_twice(keyed):
    c = _Say(GOOD)
    parsed, reply = c.complete_json("go", 500)
    assert parsed["beats"] and len(c.prompts) == 1
    assert "json_retry" not in (reply.meta or {})
