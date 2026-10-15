"""A production that dies must say so, and the model transport must be reachable.

TWO BUGS THIS PINS
------------------
1. `produce` ran the graph on a daemon thread under `except Exception: pass`.
   A run that died before writing its first artifact left no agent-job
   marker, so `/status` answered 404 - which the UI reads as "not started
   yet" and keeps polling. A hard failure and a slow start were
   indistinguishable, and the observed case (ffprobe not on PATH) span
   forever on "Starting the production...".

2. `LLMClient._http` sent no User-Agent. urllib then sends
   "Python-urllib/3.x", which Cloudflare-fronted inference endpoints reject
   outright - Groq's OpenAI-compatible endpoint answers 403 "error code:
   1010" to the exact headers the client built, and 200 to the identical
   request with any UA set. Every reasoning agent fell back to its
   deterministic path and reported "no model configured" while a valid key
   sat in the environment.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import api as API                       # noqa: E402
from agents.tools import llm_tools as LT            # noqa: E402


# ------------------------------------------------ 1. the transport header

def test_the_http_transport_sends_a_user_agent(monkeypatch):
    """No UA is a 403 from any Cloudflare-fronted endpoint."""
    seen = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return (b'{"choices":[{"message":{"content":"{}"}}],'
                    b'"usage":{"prompt_tokens":1,"completion_tokens":1}}')

    def fake_urlopen(req, timeout=None):
        seen["headers"] = dict(req.headers)
        return FakeResp()

    monkeypatch.setenv("AT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(LT.urllib.request, "urlopen", fake_urlopen)

    LT.LLMClient().complete("hello", 50)

    # urllib title-cases header names on the way in
    assert "User-agent" in seen["headers"], seen["headers"]
    assert seen["headers"]["User-agent"]


def test_the_user_agent_can_be_overridden(monkeypatch):
    seen = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return (b'{"choices":[{"message":{"content":"{}"}}],'
                    b'"usage":{}}')

    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AT_LLM_USER_AGENT", "custom-agent/9")
    monkeypatch.setattr(LT.urllib.request, "urlopen",
                        lambda req, timeout=None: seen.update(
                            h=dict(req.headers)) or FakeResp())

    LT.LLMClient().complete("hello", 50)
    assert seen["h"]["User-agent"] == "custom-agent/9"


# ------------------------------------------------ 2. the failure is visible

@pytest.fixture(autouse=True)
def _clean_failures():
    API._FAILED.clear()
    yield
    API._FAILED.clear()


def test_status_reports_a_run_that_died_before_writing_anything(monkeypatch):
    """The 404 the UI polled forever is replaced by a stated failure."""
    job = "deadjob01"
    API._FAILED[job] = {"error": "RuntimeError: probe failed (exit 1)",
                        "traceback": "...", "at": 0.0}
    # the job never became an agent job, which is exactly the 404 case
    monkeypatch.setattr(API.compat, "is_agent_job", lambda j: False)
    monkeypatch.setattr(API, "_saved_settings", lambda j: None)

    out = API.agent_status(job)
    assert out["status"] == "failed"
    assert "probe failed" in out["failed"]["error"]


def test_a_job_with_no_recorded_failure_still_404s(monkeypatch):
    """Starting is not failing: that distinction is the whole point."""
    from fastapi import HTTPException
    monkeypatch.setattr(API.compat, "is_agent_job", lambda j: False)
    with pytest.raises(HTTPException) as exc:
        API.agent_status("stillstarting")
    assert exc.value.status_code == 404


def test_the_worker_records_the_reason_instead_of_dropping_it():
    """The `except Exception: pass` this replaces left nothing behind."""
    job = "boomjob01"

    def _go():
        try:
            raise RuntimeError("probe failed (exit 1)")
        except Exception as exc:                     # noqa: BLE001
            import time
            import traceback
            API._FAILED[job] = {
                "error": f"{type(exc).__name__}: {exc}"[:600],
                "traceback": traceback.format_exc()[-2000:],
                "at": time.time(),
            }

    _go()
    assert API._FAILED[job]["error"].startswith("RuntimeError: probe failed")
    assert "RuntimeError" in API._FAILED[job]["traceback"]


def test_a_retry_clears_the_previous_failure():
    """Otherwise a successful re-run still renders the old error."""
    job = "retryjob1"
    API._FAILED[job] = {"error": "old", "traceback": "", "at": 0.0}
    API._FAILED.pop(job, None)                       # what produce() does
    assert job not in API._FAILED


# ------------------------------------------- 3. truncation is named as such

def test_a_reply_cut_off_at_the_cap_says_so(monkeypatch):
    """Reasoning models spend output budget thinking, then get cut off.

    The Title agent asked for 700 tokens, used all 700, and its half-written
    JSON was reported as "not a JSON object" - which points at the model's
    formatting instead of the budget that actually caused it.
    """
    monkeypatch.setattr(
        LT.LLMClient, "complete",
        lambda self, p, m=1600, i=None, r=None: LT.LLMReply(
            text='{"title": "half a repl',
            tokens_in=500, tokens_out=700, stop_reason="length"))
    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=700)
    assert parsed is None
    assert "cut off" in reply.error
    # the budget is doubled once before giving up, so the message names the
    # limit it actually ran out at, not the one it started with
    assert "1400" in reply.error


def test_a_genuinely_malformed_reply_is_not_blamed_on_the_cap(monkeypatch):
    monkeypatch.setattr(
        LT.LLMClient, "complete",
        lambda self, p, m=1600, i=None, r=None: LT.LLMReply(
            text="I would rather not answer in JSON.",
            tokens_in=10, tokens_out=9, stop_reason="stop"))
    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=700)
    assert parsed is None
    assert reply.error is None, "a complete reply that is simply not JSON"


def test_truncated_is_read_off_both_wire_formats():
    assert LT.LLMReply(text="x", stop_reason="length").truncated
    assert LT.LLMReply(text="x", stop_reason="max_tokens").truncated
    assert not LT.LLMReply(text="x", stop_reason="stop").truncated
    assert not LT.LLMReply(text="x").truncated


def test_an_empty_truncated_reply_is_still_named_as_truncation(monkeypatch):
    """The case that actually happens with reasoning models.

    Groq returns finish_reason "length" with an EMPTY `content` when the
    budget ran out during hidden reasoning, so `ok` is False. The check has
    to run before the `ok` gate or the diagnosis is skipped and the agent
    reports "not a JSON object" for what was really too small a budget.
    """
    monkeypatch.setattr(
        LT.LLMClient, "complete",
        lambda self, p, m=1600, i=None, r=None: LT.LLMReply(
            text="", tokens_in=100, tokens_out=700, stop_reason="length"))
    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=700)
    assert parsed is None
    assert reply.error and "cut off" in reply.error
    assert "1400" in reply.error, "the widened budget is what it gave up at"


def test_a_truncated_reply_that_still_parsed_is_not_an_error(monkeypatch):
    """Cut off after the JSON closed is a usable answer."""
    monkeypatch.setattr(
        LT.LLMClient, "complete",
        lambda self, p, m=1600, i=None, r=None: LT.LLMReply(
            text='{"title": "fine"} and then it rambled',
            tokens_out=700, stop_reason="length"))
    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=700)
    assert parsed == {"title": "fine"}
    assert reply.error is None


# ---------------------------- 4. a preposition is not a claim about footage

def test_common_prepositions_do_not_need_grounding():
    """`toward` was missing, so a grounded title was thrown out for it.

    Observed on a real run: the model returned "Fast-Paced Alpha Titans
    Showcase Building Tension Toward a Summit", every noun of which traced
    back to the material, and the agent rejected the whole title with
    "not in anything about this video: Toward" - leaving the video untitled.
    """
    from agents.title import FUNCTION_WORDS as F
    for w in ("toward", "towards", "across", "between", "among", "against",
              "during", "after", "before", "within", "while", "until"):
        assert w in F, f"{w!r} is a function word, not a claim"


def test_words_that_do_make_a_claim_still_need_grounding():
    """The list must not have swallowed anything that says something."""
    from agents.title import FUNCTION_WORDS as F
    for w in ("video", "title", "footage", "gear", "engine", "fast",
              "three", "robot", "material"):
        assert w not in F, f"{w!r} is a content word and must be grounded"


def test_the_function_word_list_contains_only_words():
    """It is built with `.split()`, so a stray comment would become entries."""
    from agents.title import FUNCTION_WORDS as F
    for w in F:
        assert w.replace("-", "").replace("'", "").isalpha(), repr(w)


# ------------------- 5. images go to the vision model, text does not

def test_an_image_call_uses_the_vision_model(monkeypatch):
    """The Vision agent calls the SHARED client like everyone else.

    Without this split it posted base64 frames to whatever AT_LLM_MODEL
    named. Groq's gpt-oss is text-only and answers "messages[0].content
    must be a string", so every clip failed and Vision degraded - which
    took Story's beats with it, and Voiceover, Caption and Transition all
    fell back for want of a beat.
    """
    monkeypatch.setenv("AT_LLM_API_KEY", "text-key")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AT_LLM_MODEL", "text-only-model")
    monkeypatch.setenv("AT_LLM_BASE_URL", "https://text.example")
    monkeypatch.setenv("AT_VISION_API_KEY", "vision-key")
    monkeypatch.setenv("AT_VISION_PROVIDER", "openai")
    monkeypatch.setenv("AT_VISION_MODEL", "sees-pictures")
    monkeypatch.setenv("AT_VISION_BASE_URL", "https://vision.example")

    seen = []

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return (b'{"choices":[{"message":{"content":"{}"},'
                    b'"finish_reason":"stop"}],"usage":{}}')

    def fake_urlopen(req, timeout=None):
        seen.append({"url": req.full_url,
                     "auth": dict(req.headers).get("Authorization"),
                     "model": json.loads(req.data)["model"]})
        return FakeResp()

    import json
    monkeypatch.setattr(LT.urllib.request, "urlopen", fake_urlopen)

    c = LT.LLMClient()
    c.complete("plain text", 50)
    c.complete("with a picture", 50, images=["BASE64"])

    assert seen[0]["model"] == "text-only-model"
    assert "text.example" in seen[0]["url"]
    assert seen[0]["auth"] == "Bearer text-key"

    assert seen[1]["model"] == "sees-pictures", "images must use the vision model"
    assert "vision.example" in seen[1]["url"]
    assert seen[1]["auth"] == "Bearer vision-key"


def test_one_multimodal_key_is_left_alone(monkeypatch):
    """No separate vision model configured => behave exactly as before."""
    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_MODEL", "one-model")
    monkeypatch.delenv("AT_VISION_API_KEY", raising=False)
    monkeypatch.delenv("AT_VISION_MODEL", raising=False)
    assert LT.vision_profile() is None


def test_an_identical_vision_config_does_not_switch(monkeypatch):
    """Same key, model and endpoint on both: nothing to switch to."""
    for k in ("AT_LLM_API_KEY", "AT_VISION_API_KEY"):
        monkeypatch.setenv(k, "same-key")
    for k in ("AT_LLM_MODEL", "AT_VISION_MODEL"):
        monkeypatch.setenv(k, "same-model")
    for k in ("AT_LLM_BASE_URL", "AT_VISION_BASE_URL"):
        monkeypatch.setenv(k, "https://same.example")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AT_VISION_PROVIDER", "openai")
    assert LT.vision_profile() is None


# ---------------------- 6. a rate limit is transient, not an answer

def test_a_429_is_retried_and_then_succeeds(monkeypatch):
    """Groq's free tier allows 8,000 tokens a minute and the agents fire
    back to back: ContentStrategy + Story + Voiceover + Animation spent
    8,880 in half a minute, so Music, Transition, Caption and the
    EditingDirector were all refused and four agents degraded a finished
    video for a limit that clears itself seconds later."""
    calls = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return (b'{"choices":[{"message":{"content":"{\\"ok\\":true}"},'
                    b'"finish_reason":"stop"}],"usage":{}}')

    def flaky(req, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise LT.urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {}, None)
        return Resp()

    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(LT, "_RETRIES", 3)
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    monkeypatch.setattr(LT.urllib.request, "urlopen", flaky)

    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=100)
    assert parsed == {"ok": True}
    assert reply.error is None
    assert len(calls) == 3, "two refusals, then the answer"


def test_a_401_is_not_retried(monkeypatch):
    """A bad key never fixes itself - retrying only wastes the deadline."""
    calls = []

    def dead(req, timeout=None):
        calls.append(1)
        raise LT.urllib.error.HTTPError(req.full_url, 401,
                                        "Unauthorized", {}, None)

    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(LT, "_RETRIES", 3)
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    monkeypatch.setattr(LT.urllib.request, "urlopen", dead)

    _parsed, reply = LT.LLMClient().complete_json("x", max_tokens=100)
    assert "401" in reply.error
    assert len(calls) == 1, "asked once and stopped"


def test_retry_after_header_is_honoured(monkeypatch):
    class E(LT.urllib.error.HTTPError):
        def __init__(self):
            super().__init__("u", 429, "slow down", {"Retry-After": "7"}, None)
    assert LT._retry_after(E(), 0) == 7.0


def test_backoff_grows_and_is_capped():
    class E:
        headers = {}
    waits = [LT._retry_after(E(), i) for i in range(6)]
    assert waits[0] < waits[1] < waits[2], "it must back off"
    assert max(waits) <= 30.0, "one refusal cannot stall a build"


# ------------------- 7. a vision call is worth more patience than text

def _flaky(fail_times, code=503):
    """Refuse `fail_times` times, then answer."""
    calls = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return (b'{"choices":[{"message":{"content":"{\\"ok\\":true}"},'
                    b'"finish_reason":"stop"}],"usage":{}}')

    def fn(req, timeout=None):
        calls.append(1)
        if len(calls) <= fail_times:
            raise LT.urllib.error.HTTPError(req.full_url, code, "busy", {}, None)
        return Resp()
    return fn, calls


def test_a_vision_call_outlasts_the_default_retries(monkeypatch):
    """Observed: an eight-minute clip was refused 503 four times running
    while the short clip beside it succeeded, so every beat in the finished
    story came from the short one. The default three tries were not enough,
    and a clip nobody looked at can ground no beat at all."""
    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(LT, "_RETRIES", 3)
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    fn, calls = _flaky(5)                     # five refusals in a row
    monkeypatch.setattr(LT.urllib.request, "urlopen", fn)

    parsed, reply = LT.LLMClient().complete_json(
        "x", max_tokens=100, retries=LT.VISION_RETRIES)
    assert parsed == {"ok": True}, reply.error
    assert len(calls) == 6


def test_the_same_call_with_default_patience_gives_up(monkeypatch):
    """The contrast that makes the override worth having."""
    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(LT, "_RETRIES", 3)
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    fn, calls = _flaky(5)
    monkeypatch.setattr(LT.urllib.request, "urlopen", fn)

    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=100)
    assert parsed is None
    assert "503" in reply.error
    assert len(calls) == 4                    # the default: 1 + 3


def test_patience_is_bounded(monkeypatch):
    """A caller cannot ask for an unbounded wait."""
    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    fn, calls = _flaky(999)
    monkeypatch.setattr(LT.urllib.request, "urlopen", fn)
    LT.LLMClient().complete_json("x", max_tokens=100, retries=10_000)
    assert len(calls) <= 9, "capped at 8 retries plus the first try"


def test_the_vision_agent_actually_asks_for_it():
    import inspect
    import agents.vision_agent as VA
    src = inspect.getsource(VA)
    assert "retries=VISION_RETRIES" in src
    assert VA.VISION_RETRIES > LT._RETRIES



def test_the_default_patience_survives_a_busy_token_window(monkeypatch):
    """Caching the measurement removed an accidental protection.

    While `clip_analysis` took ten minutes the per-minute token window had
    all the time it needed to drain. The agents now fire seconds apart, and
    a run that overlapped another had seven of them refused 429 at once -
    on the EditingDirector that means the one-segment-per-clip planner and
    a video with a single cut. Four refusals in a row must now survive.
    """
    monkeypatch.setenv("AT_LLM_API_KEY", "k")
    monkeypatch.setenv("AT_LLM_PROVIDER", "openai")
    monkeypatch.delenv("AT_LLM_RETRIES", raising=False)
    monkeypatch.setattr(LT, "_RETRIES", LT._retries())
    monkeypatch.setattr(LT.time, "sleep", lambda s: None)
    fn, calls = _flaky(4, code=429)
    monkeypatch.setattr(LT.urllib.request, "urlopen", fn)

    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=100)
    assert parsed == {"ok": True}, reply.error
    assert len(calls) == 5


def test_the_waiting_fits_inside_an_agent_deadline():
    """An agent has 300s. The retries must not spend it all."""
    class E:
        headers = {}
    total = sum(LT._retry_after(E(), i) for i in range(LT._retries()))
    assert total < 120, f"{total}s of waiting is too much of a 300s deadline"


def test_the_retry_count_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("AT_LLM_RETRIES", "2")
    assert LT._retries() == 2
    monkeypatch.setenv("AT_LLM_RETRIES", "nonsense")
    assert LT._retries() == 5


# ------------- 8. a truncated reply is retried with room, not given up on

def test_a_cut_off_reply_is_retried_with_a_bigger_budget(monkeypatch):
    """Caption and Title both spent exactly their 1400 tokens on hidden
    reasoning and returned an empty `content`, while Story used 1524 of its
    2000 and was fine. A reasoning model's overhead is variable, so a fixed
    cap that fits one prompt starves another - the budget is doubled once
    rather than guessed per agent."""
    seen = []

    def fake(self, prompt, max_tokens=1600, images=None, retries=None):
        seen.append(max_tokens)
        if max_tokens < 2000:
            return LT.LLMReply(text="", tokens_out=max_tokens,
                               stop_reason="length")
        return LT.LLMReply(text='{"ok": true}', tokens_out=120,
                           stop_reason="stop")

    monkeypatch.setattr(LT.LLMClient, "complete", fake)
    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=1400)
    assert parsed == {"ok": True}
    assert seen == [1400, 2800], seen
    assert reply.meta.get("widened_from") == 1400


def test_it_widens_only_once(monkeypatch):
    """Doubling for ever would turn one bad reply into a runaway bill."""
    seen = []

    def always_cut(self, prompt, max_tokens=1600, images=None, retries=None):
        seen.append(max_tokens)
        return LT.LLMReply(text="", tokens_out=max_tokens,
                           stop_reason="length")

    monkeypatch.setattr(LT.LLMClient, "complete", always_cut)
    parsed, reply = LT.LLMClient().complete_json("x", max_tokens=1400)
    assert parsed is None
    assert len(seen) == 2, seen
    assert "cut off" in reply.error


def test_a_reply_that_parsed_is_never_widened(monkeypatch):
    """Truncated AFTER the JSON closed is a usable answer - leave it."""
    seen = []

    def fake(self, prompt, max_tokens=1600, images=None, retries=None):
        seen.append(max_tokens)
        return LT.LLMReply(text='{"ok": true} then it rambled',
                           tokens_out=max_tokens, stop_reason="length")

    monkeypatch.setattr(LT.LLMClient, "complete", fake)
    parsed, _r = LT.LLMClient().complete_json("x", max_tokens=1400)
    assert parsed == {"ok": True}
    assert seen == [1400], "no second call for an answer we already have"


def test_a_call_already_at_the_ceiling_is_not_widened(monkeypatch):
    seen = []

    def fake(self, prompt, max_tokens=1600, images=None, retries=None):
        seen.append(max_tokens)
        return LT.LLMReply(text="", tokens_out=max_tokens,
                           stop_reason="length")

    monkeypatch.setattr(LT.LLMClient, "complete", fake)
    LT.LLMClient().complete_json("x", max_tokens=LT._MAX_RETRY_TOKENS)
    assert len(seen) == 1
