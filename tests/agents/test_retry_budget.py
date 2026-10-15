"""A rate limit must be waited out for as long as the provider asks.

Groq answers 429 with `Retry-After`. That figure is measured from the
window that actually refused the call, so it beats any guess. Clamping it
to 30s made every retry ask again while the window was still shut: five
attempts spent inside one minute, and the agent degraded for a limit that
would have cleared on its own. Two productions running at once (one from
a stale checkout on another drive) degraded six agents this way -
story, music, animation, transition, editing_director and title.

What must NOT happen instead is sleeping past the agent's 300s deadline:
a killed node writes nothing at all, where a fallback at least writes a
deterministic answer marked degraded. Hence a total budget.
"""

from __future__ import annotations

import pathlib
import sys
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.tools import llm_tools as LT            # noqa: E402


def _err(code: int, retry_after: str | None):
    hdrs = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError("http://x", code, "busy", hdrs, None)


# --------------------------------------------------- the provider's number

def test_a_long_retry_after_is_honoured_not_clamped_to_30s():
    assert LT._retry_after(_err(429, "60"), 0) == 60.0


def test_an_unaffordable_retry_after_is_reported_whole():
    """So the caller can see it does not fit, rather than being handed a
    trimmed number that looks affordable and sleeps the whole budget."""
    assert LT._retry_after(_err(429, "3600"), 0) == 3600.0


def test_a_nonsense_header_falls_back_to_backoff():
    assert LT._retry_after(_err(429, "soon"), 2) == 8.0


def test_without_a_header_the_backoff_is_unchanged():
    assert [LT._retry_after(_err(503, None), i) for i in range(5)] == \
        [2.0, 4.0, 8.0, 16.0, 30.0]


# ------------------------------------------------------- the total budget

class _Client(LT.LLMClient):
    """Always refused, counting the calls and the sleeping."""

    def __init__(self, exc):
        self._transport = None          # the real __init__ is not run
        self._exc, self.calls, self.slept = exc, 0, []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None):
        self.calls += 1
        raise self._exc


@pytest.fixture
def no_real_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(LT.time, "sleep", slept.append)
    monkeypatch.setattr(LT, "_env", lambda k, d=None:
                        "k" if k.endswith("API_KEY") else
                        ("m" if k.endswith("MODEL") else d))
    return slept


def test_waiting_stops_before_the_agent_deadline(no_real_sleep):
    """Told to wait 100s each time, it waits twice and then gives up -
    300s of sleeping would see the node killed with nothing written."""
    c = _Client(_err(429, "100"))
    reply = c.complete("hi", 100)

    assert sum(no_real_sleep) <= LT._RETRY_BUDGET_S
    assert no_real_sleep == [100.0]          # a second would overrun
    assert not reply.ok
    assert "gave up after 100s of waiting" in reply.error


def test_short_waits_still_use_every_retry(no_real_sleep, monkeypatch):
    monkeypatch.setattr(LT, "_RETRIES", 5)
    c = _Client(_err(429, "5"))
    c.complete("hi", 100)
    assert c.calls == 6, "one first call plus five retries"
    assert no_real_sleep == [5.0] * 5


def test_a_spent_quota_falls_back_at_once_instead_of_sleeping(no_real_sleep):
    """An hour is longer than any agent has. Waiting even part of it only
    delays the fallback that is now certain."""
    c = _Client(_err(429, "3600"))
    reply = c.complete("hi", 100)

    assert no_real_sleep == [], "nothing was gained by waiting"
    assert c.calls == 1
    assert "gave up after 0s of waiting" in reply.error


def test_a_non_transient_code_is_not_retried(no_real_sleep):
    c = _Client(_err(401, None))
    reply = c.complete("hi", 100)
    assert c.calls == 1 and no_real_sleep == []
    assert "401" in reply.error


# ============================================================ model chain
#
# A DAILY allowance is spent per model, so a model that has run out is
# replaced, not waited for. On 14 Oct `openai/gpt-oss-120b` reached its
# 200,000 tokens and SIX agents - content_strategy, story, music,
# animation, transition and the editing_director - each burned 150s
# waiting for a window that reopens tomorrow, then fell back. Three other
# models on the same key still had quota and all answered the same prompt
# with clean JSON.

import io                                                   # noqa: E402


def _err_body(code: int, body: str, retry_after: str | None = None):
    hdrs = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(
        "http://x", code, "busy", hdrs, io.BytesIO(body.encode()))


TPD = ('{"error":{"message":"Rate limit reached for model `m-120b` in '
       'organization `org_x` service tier `on_demand` on tokens per day '
       '(TPD): Limit 200000, Used 199159, Requested 1577."}}')
TPM = ('{"error":{"message":"Rate limit reached ... on tokens per minute '
       '(TPM): Limit 8000, Used 7900, Requested 1577."}}')


def test_a_daily_refusal_is_told_apart_from_a_per_minute_one():
    assert _daily(TPD) == "TPD spent (199159/200000)"
    assert _daily(TPM) is None, "a minute DOES clear by waiting"


def _daily(body):
    return LT._daily_quota_spent(_err_body(429, body))


def test_an_unreadable_body_is_treated_as_the_survivable_kind():
    """Guessing 'give up' on a refusal we cannot read would throw away a
    retry that might well have worked."""
    assert LT._daily_quota_spent(_err(429, "30")) is None


# ------------------------------------------------------- the chain itself

class _Chain(LT.LLMClient):
    """Refuses for every model named in `out_of_quota`, answers for others."""

    def __init__(self, out_of_quota):
        self._transport = None
        self.out, self.tried = set(out_of_quota), []

    def _http(self, prompt, max_tokens, images=None, model_override=None,
              effort=None):
        used = model_override or LT.model()
        self.tried.append(used)
        if used in self.out:
            raise _err_body(429, TPD)
        return LT.LLMReply(text="{}", meta={"model": used})


@pytest.fixture
def chain_env(monkeypatch):
    vals = {"AT_LLM_API_KEY": "k", "AT_LLM_MODEL": "m-120b",
            "AT_LLM_FALLBACK_MODELS": "m-20b, m-27b"}
    monkeypatch.setattr(LT, "_env", lambda k, d=None: vals.get(k, d))
    monkeypatch.setattr(LT.time, "sleep",
                        lambda s: pytest.fail(f"slept {s}s for a spent day"))


def test_a_spent_model_is_replaced_not_waited_for(chain_env):
    c = _Chain({"m-120b"})
    reply = c.complete("hi", 100)
    assert c.tried == ["m-120b", "m-20b"], "moved on at once"
    assert reply.ok and reply.meta["model"] == "m-20b"


def test_the_trace_says_the_answer_came_from_a_stand_in(chain_env):
    c = _Chain({"m-120b"})
    spent = c.complete("hi", 100).meta["after_quota_spent"]
    assert spent == ["m-120b: TPD spent (199159/200000)"], \
        "an answer from a different model must not read as the configured one"


def test_it_walks_the_whole_chain(chain_env):
    c = _Chain({"m-120b", "m-20b"})
    assert c.complete("hi", 100).meta["model"] == "m-27b"
    assert c.tried == ["m-120b", "m-20b", "m-27b"]


def test_every_model_spent_reports_all_of_them(chain_env):
    c = _Chain({"m-120b", "m-20b", "m-27b"})
    reply = c.complete("hi", 100)
    assert not reply.ok
    for m in ("m-120b", "m-20b", "m-27b"):
        assert m in reply.error, "the person must see the whole key is spent"


def test_the_configured_model_is_used_when_it_works(chain_env):
    c = _Chain(set())
    assert c.complete("hi", 100).meta["model"] == "m-120b"
    assert c.tried == ["m-120b"], "no stand-in is tried needlessly"
    assert "after_quota_spent" not in c.complete("hi", 100).meta


def test_images_never_leave_the_vision_model(chain_env):
    """A text stand-in given base64 frames fails for a different reason;
    the vision model is chosen inside _http and must not be overridden."""
    c = _Chain({"m-120b"})
    c.complete("hi", 100, images=["ZmFrZQ=="])
    assert c.tried == ["m-120b"], "no text model was substituted"


def test_no_fallbacks_configured_behaves_as_before(monkeypatch):
    vals = {"AT_LLM_API_KEY": "k", "AT_LLM_MODEL": "m-120b"}
    monkeypatch.setattr(LT, "_env", lambda k, d=None: vals.get(k, d))
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    c = _Chain({"m-120b"})
    reply = c.complete("hi", 100)
    assert c.tried == ["m-120b"] and not reply.ok


def test_a_fallback_list_never_repeats_the_configured_model(monkeypatch):
    vals = {"AT_LLM_MODEL": "m-a", "AT_LLM_FALLBACK_MODELS": "m-a, m-b,, m-b"}
    monkeypatch.setattr(LT, "_env", lambda k, d=None: vals.get(k, d))
    assert LT.fallback_models() == ["m-b"]
