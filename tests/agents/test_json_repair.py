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
              effort=None, json_mode=True):
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


# ================ when asking for JSON makes things worse, not better
#
# `response_format={"type":"json_object"}` was added to constrain the
# model rather than merely instruct it. On a real run it did the
# opposite for the four agents with the largest schemas - strategy,
# story, music and the editing director - all four took:
#
#   HTTP 400 - Failed to generate JSON. Please adjust your prompt.
#
# The provider REFUSES a reply it cannot make valid JSON of, instead of
# returning the malformed one. So the repair layer and the corrective
# retry above - which exist for precisely that reply - never saw it, and
# a recoverable formatting slip became an unrecoverable call. The cut
# lost its narrative to a flag meant to protect it.

import io                                                    # noqa: E402
import urllib.error                                          # noqa: E402

REFUSAL = ('{"error":{"message":"Failed to generate JSON. Please adjust '
           'your prompt. See \'failed_generation\' for more details.",'
           '"type":"invalid_request_error"}}')


def _http_error(code, body):
    return urllib.error.HTTPError("http://x", code, "bad", {},
                                  io.BytesIO(body.encode()))


VALIDATE_REFUSAL = REFUSAL.replace("generate", "validate")


@pytest.mark.parametrize("body", [
    REFUSAL,
    VALIDATE_REFUSAL,
    "FAILED TO GENERATE JSON",
    "failed to validate json",
])
def test_both_wordings_of_the_refusal_are_recognised(body):
    """It says this two ways, and matching only the first is how the
    Story agent kept failing after the other three were fixed:

        Failed to generate JSON. Please adjust your prompt.
        Failed to validate JSON. Please adjust your prompt.

    One is the model producing nothing usable, the other its output not
    fitting the schema. Both refused a reply we would rather have had.
    """
    assert LT._json_mode_refused(body), body


def test_the_validate_wording_also_earns_a_retry(keyed):
    class _Validate(LT.LLMClient):
        def __init__(self):
            self._transport = None
            self.modes = []

        def _http(self, prompt, max_tokens, images=None,
                  model_override=None, effort=None, json_mode=True):
            self.modes.append(json_mode)
            if json_mode:
                raise _http_error(400, VALIDATE_REFUSAL)
            return LT.LLMReply(text=GOOD, meta={"model": "m"})

    c = _Validate()
    parsed, _ = c.complete_json("go", 2000)
    assert c.modes == [True, False] and parsed


def test_an_ordinary_400_is_not_mistaken_for_it():
    assert not LT._json_mode_refused('{"error":{"message":"model not found"}}')
    assert not LT._json_mode_refused("")


def test_the_body_can_be_read_twice():
    """The classifier and the error message both want it, and
    `HTTPError.read()` empties the stream."""
    exc = _http_error(400, REFUSAL)
    assert LT._body_of(exc) == LT._body_of(exc) != ""


class _Refuses(LT.LLMClient):
    """400s while json_mode is on, answers once it is off."""

    def __init__(self):
        self._transport = None
        self.modes = []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None, json_mode=True):
        self.modes.append(json_mode)
        if json_mode:
            raise _http_error(400, REFUSAL)
        return LT.LLMReply(text="Sure! " + GOOD, meta={"model": "m"})


def test_it_asks_again_without_the_flag(keyed):
    c = _Refuses()
    parsed, _reply = c.complete_json("write the beats", 2000)

    assert c.modes == [True, False], "one retry, with the flag off"
    assert parsed == {"beats": [{"role": "hook", "clip_id": "a.mp4"}]}, \
        "and the repair layer then recovered the chatty reply"


class _AlwaysRefuses(LT.LLMClient):
    def __init__(self):
        self._transport = None
        self.calls = 0

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None, json_mode=True):
        self.calls += 1
        raise _http_error(400, REFUSAL)


def test_it_does_not_loop(keyed):
    """Without the flag the same refusal cannot recur, but if the
    provider sends it anyway the client must stop, not spin."""
    c = _AlwaysRefuses()
    parsed, reply = c.complete_json("go", 500)
    assert parsed is None and c.calls == 2
    assert "Failed to generate JSON" in reply.error


class _OtherError(LT.LLMClient):
    def __init__(self):
        self._transport = None
        self.calls = 0

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None, json_mode=True):
        self.calls += 1
        raise _http_error(400, '{"error":{"message":"model not found"}}')


def test_a_different_400_is_not_retried(keyed):
    c = _OtherError()
    _parsed, reply = c.complete_json("go", 500)
    assert c.calls == 1, "only the json-mode refusal earns a second ask"
    assert "model not found" in reply.error


def test_the_error_message_carries_the_providers_words(keyed):
    c = _OtherError()
    _p, reply = c.complete_json("go", 500)
    assert "HTTP Error 400" in reply.error and "model not found" in reply.error


# ====================================== when a cut-off reply is salvaged
#
# On 16 Oct the EditingDirector degraded with:
#
#     deterministic_planner (the model reply was rejected
#                            (no segments in the reply))
#
# which reads as "the model answered badly". It had not. The trace said
# `tokens_out: 1600` against `max_tokens=1600` - the reply was CUT OFF
# at the ceiling, part-way through the `segments` array.
#
# What turned that into a wrong diagnosis was this repair layer being
# too good at its job. `_candidates` scans in document order and returns
# the first value that parses; on a reply cut off inside the array the
# outer object never closes, so the scan walked PAST it and returned the
# last COMPLETE segment object it found:
#
#     {"clip_id": "01_IMG_0050.MOV", "target_seconds": 6.0, ...}
#
# A perfectly valid JSON object - and not the answer. `parsed is not
# None`, so the widen-and-retry below never fired, and the director was
# handed one segment's worth of a reply as though the model had sent it
# deliberately.
#
# The discriminator is positional and needs no guessing: a value that
# begins at the reply's first JSON character is the envelope, however
# much prose follows it. A value that begins LATER is what was left
# after the envelope was lost. The second kind is a `fragment`.

CUT_IN_ARRAY = '''{
  "strategy": "Open on the strongest build moment, then let it settle",
  "segments": [
    {"clip_id": "01.MOV", "target_seconds": 6.0, "why": "strongest motion"},
    {"clip_id": "02.MOV", "target_sec'''


def test_a_reply_cut_off_inside_an_array_is_called_a_fragment():
    value, how = JR.parse(CUT_IN_ARRAY)
    assert how == "fragment", (
        "an inner object recovered from a cut-off reply is not the answer")
    assert "segments" not in value, "the envelope really was lost"


def test_an_object_followed_by_prose_is_not_a_fragment():
    """The envelope arrived whole; the model just kept talking. Nothing
    was lost and nothing needs asking again."""
    value, how = JR.parse(GOOD + "\n\nI chose these because the first clip")
    assert how == "extracted" and value["beats"]


def test_prose_before_a_whole_object_is_not_a_fragment():
    """Preamble is not loss either - the object still starts at the
    first JSON character in the reply."""
    value, how = JR.parse("Here is the plan:\n" + GOOD)
    assert how == "extracted" and value["beats"]


def test_a_fenced_block_is_not_a_fragment():
    value, how = JR.parse("```json\n" + GOOD + "\n```")
    assert how in ("clean", "extracted") and value["beats"]


# ------------------------------------------- and what the client does with it

class _SayCut(LT.LLMClient):
    """Answers with each scripted (text, stop_reason) in turn."""

    def __init__(self, *replies):
        self._transport = None
        self._replies, self.calls = list(replies), []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None, json_mode=True):
        self.calls.append(max_tokens)
        text, stop = self._replies[min(len(self.calls) - 1,
                                       len(self._replies) - 1)]
        return LT.LLMReply(text=text, stop_reason=stop, tokens_out=max_tokens,
                           meta={"model": "m"})


def test_a_cut_off_reply_is_asked_again_with_room(keyed):
    c = _SayCut((CUT_IN_ARRAY, "length"), (GOOD, "stop"))
    parsed, reply = c.complete_json("go", 500)

    assert c.calls == [500, 1000], "the budget was doubled, once"
    assert parsed["beats"], "the second, whole answer was accepted"
    assert reply.meta["widened_from"] == 500


def test_a_whole_answer_in_a_cut_off_reply_is_not_asked_again(keyed):
    """Truncation alone is not loss. The model wrote the object and was
    cut off mid-sentence afterwards; asking again would buy nothing."""
    c = _SayCut((GOOD + "\n\nI chose these because the first", "length"))
    parsed, _reply = c.complete_json("go", 500)
    assert parsed["beats"] and c.calls == [500], "no second call"


def test_a_reply_cut_off_twice_is_reported_as_cut_off(keyed):
    """The caller must be told the budget ran out, NOT that the model
    sent a bad answer - they call for opposite responses."""
    c = _SayCut((CUT_IN_ARRAY, "length"))
    parsed, reply = c.complete_json("go", 500)

    assert parsed is None, "a fragment is not an answer"
    assert reply.error and "cut off" in reply.error
    assert "1000-token" in reply.error, "the budget it actually died at"


def test_the_widening_stops_at_the_ceiling(keyed):
    """A model stuck in a loop cannot run the bill up."""
    c = _SayCut((CUT_IN_ARRAY, "length"))
    _parsed, _reply = c.complete_json("go", LT._MAX_RETRY_TOKENS)
    assert c.calls == [LT._MAX_RETRY_TOKENS], "already at the ceiling"


def test_the_fragment_is_visible_in_the_trace(keyed):
    c = _SayCut((CUT_IN_ARRAY, "length"))
    _parsed, reply = c.complete_json("go", 500)
    assert reply.meta.get("json_repair") == "fragment"
