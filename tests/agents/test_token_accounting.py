"""What a video really costs, counted honestly.

WHY THIS MATTERS
----------------
The daily token quota is the binding constraint on how many videos a day
can be made - 200,000 per model against roughly 21,000 per video. A
budget cannot be managed from figures that leave out the waste.

And they did. Each `LLMReply` carries the usage of its OWN http call, and
`complete_json` returns only the last one, so an attempt that was
truncated, thrown away and asked again at a wider budget cost real tokens
that appeared in no trace and no total. The Title agent did this on every
single run: its 2,751 reported output tokens were missing the 1,400 it
had already spent being cut off.

The fix has two halves, and the second only works because of the first:
count the discarded attempts, then size each agent's budget from what it
is measured to need so the discard stops happening.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.tools import llm_tools as LT                    # noqa: E402

GOOD = '{"ok": true}'


class _Scripted(LT.LLMClient):
    """Replies in turn, each with its own usage, like the real transport."""

    def __init__(self, *replies):
        self._transport = None
        self._replies, self.calls = list(replies), []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None, json_mode=True):
        self.calls.append(max_tokens)
        text, stop, tin, tout = self._replies[
            min(len(self.calls) - 1, len(self._replies) - 1)]
        return LT.LLMReply(text=text, stop_reason=stop, tokens_in=tin,
                           tokens_out=tout, usd=0.001, meta={"model": "m"})


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(LT, "_env", lambda k, d=None:
                        "k" if k.endswith("API_KEY") else
                        ("m" if k.endswith("MODEL") else d))


# ============================================ the waste that was invisible

def test_a_widened_retry_reports_both_calls(keyed):
    """What Title did on every run: cut off at its cap, asked again at
    double, and only the second call was ever counted."""
    c = _Scripted(("", "length", 600, 1400),          # cut off, no JSON
                  (GOOD, "stop", 600, 2700))          # the answer
    _parsed, reply = c.complete_json("go", 1400)

    assert c.calls == [1400, 2800], "it widened once"
    assert reply.tokens_out == 1400 + 2700, "the discarded call counted"
    assert reply.tokens_in == 600 + 600
    assert reply.meta["discarded_tokens"] == 2000


def test_a_corrective_retry_reports_both_calls(keyed):
    c = _Scripted(("I think the story is about friendship.", "stop", 500, 40),
                  (GOOD, "stop", 560, 30))
    _parsed, reply = c.complete_json("go", 1000)

    assert reply.tokens_out == 70 and reply.tokens_in == 1060
    assert reply.meta["discarded_tokens"] == 540


def test_a_single_clean_call_reports_itself_and_no_more(keyed):
    c = _Scripted((GOOD, "stop", 500, 120))
    _parsed, reply = c.complete_json("go", 1000)

    assert (reply.tokens_in, reply.tokens_out) == (500, 120)
    assert "discarded_tokens" not in (reply.meta or {})


def test_the_cost_is_carried_even_when_the_retry_also_fails(keyed):
    """The tokens were spent whether or not an answer came back."""
    c = _Scripted(("not json", "stop", 400, 50), ("still not", "stop", 450, 60))
    parsed, reply = c.complete_json("go", 1000)

    assert parsed is None
    assert reply.tokens_in == 850 and reply.tokens_out == 110


def test_money_is_carried_too(keyed):
    c = _Scripted(("", "length", 100, 500), (GOOD, "stop", 100, 600))
    _parsed, reply = c.complete_json("go", 500)
    assert reply.usd == pytest.approx(0.002)


# ==================================== budgets sized from what was measured

def _budget(agent: str) -> int:
    src = (ROOT / "server" / "agents" / f"{agent}.py").read_text(
        encoding="utf-8")
    m = re.search(r"complete_json\([^)]*max_tokens=(\d+)", src, re.S)
    assert m, f"{agent}: no complete_json budget found"
    return int(m.group(1))


@pytest.mark.parametrize("agent, observed", [
    # the output each was measured producing on real footage
    ("title", 2751),
    ("story", 2054),
])
def test_a_budget_is_not_below_what_the_agent_was_measured_to_need(
        agent, observed):
    """A cap below the measured need is not a saving. The call still runs,
    the reply is still thrown away, and the widened retry pays for both."""
    assert _budget(agent) >= observed, (
        f"{agent} is capped below what it actually writes, so every run "
        f"pays for a discarded call")


def test_every_budget_leaves_room_to_widen_once():
    """Widening doubles, bounded by `_MAX_RETRY_TOKENS`. A budget over
    half that ceiling cannot widen meaningfully, so a genuine overrun has
    nowhere to go."""
    for agent in ("title", "story", "content_strategy", "editing_director",
                  "caption", "music", "animation", "transition"):
        assert _budget(agent) <= LT._MAX_RETRY_TOKENS, agent


# ============================== suppressing reasoning where it is not needed
#
# gpt-oss spends most of its output budget thinking before it writes.
# On a Title-shaped prompt, measured on this key:
#
#     openai/gpt-oss-20b   default 863 out  ->  low  82 out
#     qwen/qwen3.8-27b     default  73 out  ->  low 562 out
#     allam-2-7b           low -> HTTP 400, "not supported with this model"
#
# which is why this is per-model and not a flag set once.

class _Seen(LT.LLMClient):
    """Records the body that would go over the wire."""

    def __init__(self):
        self._transport = None
        self.bodies = []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None, json_mode=True):
        self.bodies.append({"model": model_override or LT.model(),
                            "effort": effort, "images": bool(images)})
        return LT.LLMReply(text=GOOD, meta={"model": "m"})


def test_only_the_family_it_was_measured_on_takes_the_flag():
    assert LT._takes_effort("openai/gpt-oss-20b")
    assert LT._takes_effort("openai/gpt-oss-120b")
    assert not LT._takes_effort("qwen/qwen3.8-27b"), \
        "low made qwen eight times longer, not shorter"
    assert not LT._takes_effort("allam-2-7b"), "allam rejects the flag"


def test_the_vocabulary_is_the_one_the_provider_accepts():
    assert LT.EFFORTS == ("low", "medium", "high"), \
        "`none` is rejected with HTTP 400"


def test_effort_reaches_the_request(keyed):
    c = _Seen()
    c.complete_json("go", 500, effort="low")
    assert c.bodies[0]["effort"] == "low"


def test_a_node_that_asks_for_nothing_is_unchanged(keyed):
    c = _Seen()
    c.complete_json("go", 500)
    assert c.bodies[0]["effort"] is None


@pytest.mark.parametrize("agent", ["title", "thumbnail", "transition",
                                   "animation", "caption"])
def test_the_light_nodes_ask_for_low_effort(agent):
    src = (ROOT / "server" / "agents" / f"{agent}.py").read_text(
        encoding="utf-8")
    assert 'effort="low"' in src, f"{agent} is a short, grounded choice"


@pytest.mark.parametrize("agent", ["content_strategy", "story",
                                   "editing_director", "voiceover"])
def test_the_thinking_nodes_are_left_alone(agent):
    """These decide the shape of the whole video. Cheapening their
    reasoning to save tokens would be saving on the wrong thing."""
    src = (ROOT / "server" / "agents" / f"{agent}.py").read_text(
        encoding="utf-8")
    assert 'effort="low"' not in src, agent


def test_the_flag_never_rides_along_with_images():
    """The vision path goes to another provider entirely."""
    import inspect
    src = inspect.getsource(LT.LLMClient._http)
    i = src.index("reasoning_effort")
    guard = src[max(0, i - 400):i]
    assert "not images" in guard or "_takes_effort" in guard
