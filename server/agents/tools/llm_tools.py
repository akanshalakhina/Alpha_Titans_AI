"""
LLM client for the reasoning agents.

Deliberately built on the transport that already exists in server/vision.py -
raw urllib, provider-agnostic, no SDK - and it reuses that module's `_json_from`
parser directly, because parsing a JSON object out of a reply that may be
wrapped in prose or fences is exactly the problem that code already solves
defensively.

TESTABILITY
-----------
The client takes an optional `transport` callable. Tests inject one that
returns whatever the case needs - malformed JSON, a hallucinated clip id, a
negative timestamp - so every failure path is exercised with no API key and no
network. Real runs leave it None and go over HTTP.

CONFIGURATION
-------------
    AT_LLM_API_KEY     preferred; falls back to AT_VISION_API_KEY so an
                       existing .env keeps working unchanged
    AT_LLM_PROVIDER    anthropic (default) | openai
    AT_LLM_MODEL       model id; provider-appropriate default if unset
    AT_LLM_BASE_URL    endpoint override
    AT_LLM_TIMEOUT     seconds (default 60)

No key configured is a SUPPORTED state, not an error: the EditingDirector runs
its deterministic fallback and says so in the artifact and the trace.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .pipeline_bridge import SERVER_DIR  # noqa: F401  (ensures sys.path)

import vision as _vision                 # existing module, unmodified

from . import json_repair as _repair

DEFAULT_MODEL = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o"}
DEFAULT_BASE = {"anthropic": "https://api.anthropic.com",
                "openai": "https://api.openai.com"}

# Published per-million-token prices, used only to put an order-of-magnitude
# figure in the trace. Not billing.
PRICE_PER_MTOK = {
    "anthropic": (3.00, 15.00),
    "openai": (2.50, 10.00),
}


@dataclass
class LLMReply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    error: str | None = None
    meta: dict = field(default_factory=dict)
    # Why the model stopped. "length" means the answer was CUT OFF at
    # max_tokens, which reads downstream as malformed JSON - a very
    # different problem from a model that answered badly, and one the
    # caller can only diagnose if it is told.
    stop_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    @property
    def truncated(self) -> bool:
        return self.stop_reason in ("length", "max_tokens")


def _env(key: str, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _retries() -> int:
    """How many times a refused call is asked again.

    Three was enough while `clip_analysis` took ten minutes: the pause gave
    a per-minute token window all the time it needed to drain. Caching the
    measurement took that accidental protection away - the agents now fire
    seconds apart, and a run that overlapped another one had
    ContentStrategy, Story, Music, Animation, Transition, Caption and the
    EditingDirector all refused 429 and fall back, which on the
    EditingDirector means the one-segment-per-clip planner and a video with
    a single cut in it.

    Five fits comfortably: with the backoff below that is about 60 seconds
    of waiting against each agent's 300-second deadline.
    """
    try:
        return max(0, min(8, int(_env("AT_LLM_RETRIES", 5))))
    except (TypeError, ValueError):
        return 5


_RETRIES = _retries()

# A refused vision call loses a whole clip from the story, so it is worth
# far more waiting than a text call: with the backoff below, six retries
# spend about 90 seconds against the Vision agent's 900-second deadline.
# Observed need: an eight-minute clip was refused 503 four times running
# while the short clip beside it succeeded, and every beat in the finished
# story then came from the short one.
VISION_RETRIES = 6

# The ceiling a truncated call may be retried up to. High enough that a
# reasoning model can think and still answer, low enough that a model stuck
# in a loop cannot run the bill up.
_MAX_RETRY_TOKENS = 4000


# The total a single `complete()` may spend before it gives up - across
# every retry AND every stand-in model, counting the time inside each
# attempt as well as the pauses between them.
#
# It used to count only the pauses, and to start again for each model in
# the chain. Neither was enough. On 16 Oct a studio upload sat on step 1
# and never moved: `content_strategy` began and nothing could stop it,
# because `POLICIES[...].timeout_s` is enforced by the SCHEDULER over
# media-pool work and an in-process HTTP call is not pool work. With the
# configured chain that came to
#
#     4 models x (6 attempts x 90s in flight + 150s waiting) = 46 minutes
#
# of one agent, inside a node labelled 300s. A request that hangs for its
# full timeout has spent the agent's time just as surely as a sleep has,
# so both are counted and the total is shared by the whole chain.
#
# The original reason for the budget is unchanged:
#
# Capping each individual wait at 30s threw away the one number that is
# actually known to be right. Groq answers a rate limit with `Retry-After`
# saying how long its token window really needs - sometimes 60s or more
# when two runs are competing - and clamping that to 30s meant every
# retry asked again while the window was still shut, spent all five
# attempts inside a minute, and degraded the agent for a limit that would
# have cleared had we simply waited as long as we were told to.
#
# So the provider's number is honoured, and a budget bounds the total
# instead. 150s against each agent's 300s deadline leaves room for the
# call itself plus the deterministic fallback: being killed on the
# deadline is worse than falling back, because a killed node writes
# nothing at all.
_RETRY_BUDGET_S = 150.0


def _gave_up(waited: float, in_flight: float) -> str:
    """How the budget went, in the words that fit what happened.

    Waiting and hanging are both the agent's time, and they want
    different fixes: one says the provider is rate-limiting us, the
    other says it is unwell. The message has to tell them apart, or the
    next person reads "gave up after 0s of waiting" on a call that
    occupied four minutes and concludes nothing was tried.
    """
    if in_flight < 1.0:
        return f"gave up after {waited:.0f}s of waiting"
    if waited < 1.0:
        return (f"gave up after {in_flight:.0f}s of calls that did not "
                f"come back")
    return (f"gave up after {waited + in_flight:.0f}s "
            f"({waited:.0f}s waiting, {in_flight:.0f}s in calls)")


# Which models take `reasoning_effort`, and are helped by it.
#
# Measured on one Title-shaped prompt, same key, same day:
#
#     openai/gpt-oss-20b   default 863 out  ->  low  82 out
#     qwen/qwen3.8-27b     default  73 out  ->  low 562 out
#     allam-2-7b           low -> HTTP 400, "not supported with this model"
#
# So this is NOT a flag to set globally. On gpt-oss it removes ninety per
# cent of the output on a task that needs no deep thought; on qwen it
# makes the reply eight times longer than leaving it alone; on allam it
# fails the call outright. Only the family it was measured to help is
# listed, and an unlisted model simply runs as before.
_LOW_EFFORT_MODELS = ("openai/gpt-oss",)

# Groq rejects anything else outright, so the vocabulary is closed here
# rather than discovered at runtime.
EFFORTS = ("low", "medium", "high")


def _takes_effort(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _LOW_EFFORT_MODELS)


# What Groq answers when `response_format=json_object` is set and the
# model cannot satisfy it. The reply is NOT returned - the call fails -
# so there is nothing for the repair layer to work on.
# It says this two ways, and matching only the first is how the Story
# agent kept failing after the other three were fixed:
#
#     Failed to generate JSON. Please adjust your prompt.
#     Failed to validate JSON. Please adjust your prompt.
#
# One is the model producing nothing usable, the other its output not
# fitting the schema. Both are the same thing here - the flag refused a
# reply we would rather have had and tried to repair.
_JSON_MODE_REFUSED = re.compile(r"failed to (?:generate|validate) json",
                                re.I)


def _body_of(exc) -> str:
    """The response body, read once and remembered.

    `HTTPError.read()` empties the stream, and two different callers now
    want the same body - the classifier and the message. Caching it on
    the exception means neither can take it from the other.
    """
    cached = getattr(exc, "_at_body", None)
    if cached is not None:
        return cached
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:                                   # noqa: BLE001
        body = ""
    try:
        exc._at_body = body
    except Exception:                                   # noqa: BLE001
        pass
    return body


def _json_mode_refused(body: str) -> bool:
    return bool(_JSON_MODE_REFUSED.search(body or ""))


def _http_error(exc) -> str:
    """The provider's own words, not just the status line.

    `HTTPError: HTTP Error 400: Bad Request` says nothing a person can
    act on, and that is exactly what the Story agent reported when a
    change to the request body started being rejected - the reason was
    sitting in the response the whole time and nobody read it.

    A 429's body is already parsed by `_daily_quota_spent`, which
    consumes it; this is for every other code, where the body is the
    only explanation there is.
    """
    base = f"{type(exc).__name__}: {exc}"
    body = _body_of(exc)
    if not body:
        return base
    try:
        parsed = json.loads(body)
        msg = (parsed.get("error") or {}).get("message") or body
    except (ValueError, AttributeError):
        msg = body
    return f"{base} - {' '.join(str(msg).split())[:400]}"


def _carry_cost(kept: LLMReply, discarded: LLMReply) -> None:
    """Add a thrown-away attempt's tokens onto the reply that is kept.

    Each `LLMReply` carries the usage of ITS OWN http call, and
    `complete_json` returns only the last one - so a reply that was
    truncated, discarded and asked again at a wider budget cost real
    tokens that appeared in no trace and no total. The Title agent does
    this on every run, and its 2,751 reported output tokens were missing
    the 1,400 it had already spent being cut off.

    Under-reporting spend is not a cosmetic fault here: the daily quota
    is the binding constraint on how many videos a day can be made, and
    a budget cannot be managed from figures that leave out the waste.
    """
    kept.tokens_in += discarded.tokens_in
    kept.tokens_out += discarded.tokens_out
    kept.usd += discarded.usd
    spent = discarded.tokens_in + discarded.tokens_out
    if spent:
        kept.meta = {**(kept.meta or {}),
                     "discarded_tokens": spent + (kept.meta or {})
                     .get("discarded_tokens", 0)}


def _retry_after(exc, attempt: int) -> float:
    """How long to wait before asking again.

    Providers say so in `Retry-After` when they know; Groq does on a rate
    limit, and that figure beats any guess of ours - it is measured from
    the window that actually refused us. Without it, back off 2s, 4s, 8s -
    long enough for a per-minute token window to free up, short enough to
    stay inside an agent's deadline.
    """
    hdr = ""
    try:
        hdr = (exc.headers.get("Retry-After") or "").strip()
    except Exception:                                   # noqa: BLE001
        hdr = ""
    if hdr:
        try:
            # Returned WHOLE, not clipped to the budget. A provider asking
            # for an hour is telling us the quota is spent, not that it
            # wants us back in 150 seconds; clipping it would spend the
            # entire budget on one hopeless sleep and still fall back. The
            # caller refuses any wait it cannot afford, so an impossible
            # number gives up at once and leaves the deadline to the
            # deterministic answer. Bounded only against a nonsense header.
            return max(0.5, min(86_400.0, float(hdr)))
        except ValueError:
            pass
    return min(30.0, 2.0 * (2 ** attempt))


def api_key() -> str | None:
    return _env("AT_LLM_API_KEY") or _env("AT_VISION_API_KEY")


def provider() -> str:
    p = (_env("AT_LLM_PROVIDER") or _env("AT_VISION_PROVIDER", "anthropic") or "").lower()
    return p if p in DEFAULT_MODEL else "anthropic"


# A provider's DAILY allowance is spent per MODEL, so a model that has run
# out is replaced, never waited for. Comma-separated, tried in order after
# `AT_LLM_MODEL`.
def fallback_models() -> list[str]:
    seen, out = {model()}, []
    for m in (x.strip() for x in (_env("AT_LLM_FALLBACK_MODELS") or "").split(",")):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


_SPENT = re.compile(
    r"on (?:tokens|requests) per day \(([TR]PD)\): Limit (\d+), Used (\d+)")


def _daily_quota_spent(exc) -> str | None:
    """Whether this 429 is the DAY's allowance rather than the minute's.

    The difference decides whether waiting can possibly work, and the
    provider only says which in the BODY. On 14 Oct every reasoning agent
    degraded on 429 while `x-ratelimit-remaining-tokens` read 7,923 of
    8,000 and a small probe answered 200 - because that header describes
    the per-minute bucket and says nothing about the day:

        Rate limit reached for model `openai/gpt-oss-120b` ... on
        tokens per day (TPD): Limit 200000, Used 199159, Requested 1577

    Small calls still fitted in the last ~840 tokens, so Caption and
    Voiceover finished while Story and the EditingDirector could not,
    which is what made a hard daily stop look intermittent. Each agent
    then spent its whole retry budget waiting for a window that does not
    reopen until tomorrow.

    Returns a short description to carry into the trace, or None when the
    refusal is the ordinary per-minute kind that waiting does clear.
    """
    try:
        body = exc.read().decode("utf-8", "replace")[:2000]
    except Exception:                                   # noqa: BLE001
        return None
    m = _SPENT.search(body)
    if m:
        return f"{m.group(1)} spent ({m.group(3)}/{m.group(2)})"
    return "daily quota spent" if "per day" in body.lower() else None


def model() -> str:
    return _env("AT_LLM_MODEL") or _env("AT_VISION_MODEL") or DEFAULT_MODEL[provider()]


def base_url() -> str:
    return (_env("AT_LLM_BASE_URL") or _env("AT_VISION_BASE_URL")
            or DEFAULT_BASE[provider()]).rstrip("/")


def vision_profile() -> dict | None:
    """The model to use for a call that carries IMAGES, when it differs.

    The Vision agent calls this shared client like every other agent, so
    without this it sent base64 frames to whatever `AT_LLM_MODEL` names. A
    text-only model rejects them outright - Groq's gpt-oss answers
    "messages[0].content must be a string" - so every clip failed, Vision
    degraded to measurements, and the cascade took Story's beats with it:
    Voiceover, Caption and Transition all fell back for want of a beat.
    Meanwhile a perfectly good vision model sat configured in AT_VISION_*,
    which only the legacy `server/vision.py` ever read.

    Returns None when no separate vision model is configured, so a single
    multimodal key keeps behaving exactly as it did.
    """
    key, mdl = _env("AT_VISION_API_KEY"), _env("AT_VISION_MODEL")
    if not (key and mdl):
        return None
    prov = (_env("AT_VISION_PROVIDER") or "openai").lower()
    if prov not in DEFAULT_MODEL:
        prov = "openai"
    base = (_env("AT_VISION_BASE_URL") or DEFAULT_BASE[prov]).rstrip("/")
    if mdl == model() and base == base_url() and key == api_key():
        return None                 # same model either way; nothing to switch
    return {"key": key, "model": mdl, "provider": prov, "base_url": base}


def timeout_s() -> float:
    try:
        return max(10.0, float(_env("AT_LLM_TIMEOUT", 60)))
    except (TypeError, ValueError):
        return 60.0


def _price(tok_in: int, tok_out: int) -> float:
    pin, pout = PRICE_PER_MTOK.get(provider(), (0.0, 0.0))
    return round(tok_in / 1e6 * pin + tok_out / 1e6 * pout, 6)


class LLMClient:
    """One text-completion call, JSON-shaped reply expected."""

    def __init__(self, transport: Callable[[str, int], LLMReply] | None = None):
        self._transport = transport

    # --------------------------------------------------------- capability

    def available(self) -> bool:
        return self._transport is not None or bool(api_key())

    def describe(self) -> dict:
        if self._transport is not None:
            return {"enabled": True, "provider": "injected-transport",
                    "model": "test", "why_disabled": None}
        if not api_key():
            # Agent-neutral on purpose: this client is shared, and naming one
            # agent here put "the EditingDirector runs its deterministic
            # planner" into the Vision agent's own artifact, which was simply
            # untrue of the agent reporting it.
            return {"enabled": False, "provider": None, "model": None,
                    "why_disabled": ("neither AT_LLM_API_KEY nor "
                                     "AT_VISION_API_KEY is set, so no agent "
                                     "can reason; each falls back to its own "
                                     "deterministic path")}
        return {"enabled": True, "provider": provider(), "model": model(),
                "base_url": base_url(), "why_disabled": None}

    # ------------------------------------------------------------- calls

    def complete(self, prompt: str, max_tokens: int = 1600,
                 images: list[str] | None = None,
                 retries: int | None = None,
                 effort: str | None = None,
                 json_mode: bool = True) -> LLMReply:
        """One call. `images` are base64 JPEGs, for the vision path.

        The injected transport is given the images too, so a test can assert
        what the model was actually shown without a key or a network.

        `retries` lets a caller buy more patience than the default. A
        refused VISION call costs a whole clip - nothing downstream can
        ground a beat in footage nobody looked at - so the Vision agent
        asks for more, and it has a 900s deadline to spend.
        """
        if self._transport is not None:
            try:
                return self._transport(prompt, max_tokens, images)
            except TypeError:
                # transports written before images existed take two arguments
                return self._transport(prompt, max_tokens)
        if not api_key():
            return LLMReply(text="", error="no API key configured")
        # 429 and 503 are TRANSIENT and routine on a free tier, not answers.
        # Groq allows 8,000 tokens a minute; the agents fire back to back, so
        # ContentStrategy + Story + Voiceover + Animation alone spent 8,880 in
        # half a minute and Music, Transition, Caption and the EditingDirector
        # were all refused - each one degrading a finished video for a limit
        # that clears itself seconds later. Gemini answers 503 the same way
        # when a model is busy. Waiting and asking again is the whole fix.
        last = None
        tries = _RETRIES if retries is None else max(0, min(8, int(retries)))
        # The configured model, then its stand-ins. Images are excluded:
        # they must go to the vision model `_http` picks, and a text model
        # given base64 frames fails differently and for a different reason.
        chain: list[str | None] = [None]
        if not images:
            chain += fallback_models()
        spent: list[str] = []
        # One budget for the whole call, outside the chain loop and
        # covering the time spent INSIDE each attempt as well as between
        # them. Both of those were missing and together they let a
        # single agent hold the graph for most of an hour - see
        # `_RETRY_BUDGET_S`.
        waited = in_flight = 0.0

        for alt in chain:
            for attempt in range(tries + 1):
                if waited + in_flight >= _RETRY_BUDGET_S:
                    break
                began = time.monotonic()
                try:
                    reply = self._http(prompt, max_tokens, images, alt,
                                       effort, json_mode)
                    if spent:
                        # the trace must show that the answer came from a
                        # stand-in, not from the model that was configured
                        reply.meta = {**(reply.meta or {}),
                                      "after_quota_spent": spent}
                    return reply
                except urllib.error.HTTPError as exc:
                    in_flight += time.monotonic() - began
                    last = exc
                    note = (_daily_quota_spent(exc) if exc.code == 429
                            else None)
                    if note:
                        # No pause inside one run reaches tomorrow. Move to
                        # the next model instead of spending this agent's
                        # whole budget learning that again.
                        spent.append(f"{alt or model()}: {note}")
                        break
                    if (exc.code == 400 and json_mode
                            and _json_mode_refused(_body_of(exc))):
                        # `response_format` makes the provider REFUSE a
                        # reply it cannot make valid JSON of, rather than
                        # return the malformed one - so the repair layer
                        # and the corrective retry, which exist for
                        # exactly that reply, never see it.
                        #
                        # Measured on real footage: with it on, the four
                        # agents with the largest schemas - strategy,
                        # story, music and the editing director - all
                        # took a 400 and fell back, and the cut lost its
                        # narrative. Asking again without it gives those
                        # defences something to work with.
                        return self.complete(prompt, max_tokens, images,
                                             retries, effort,
                                             json_mode=False)
                    if exc.code not in (429, 503) or attempt == tries:
                        return LLMReply(text="", error=_http_error(exc))
                    pause = _retry_after(exc, attempt)
                    if waited + in_flight + pause > _RETRY_BUDGET_S:
                        # Stopping here is deliberate. Sleeping past the
                        # budget would see the agent killed on its deadline
                        # with nothing written; giving up now leaves time
                        # for the deterministic fallback, and says how long
                        # we waited so the trace shows a busy provider
                        # rather than a broken one.
                        return LLMReply(
                            text="", error=f"{type(exc).__name__}: {exc} "
                                           f"({_gave_up(waited, in_flight)})")
                    time.sleep(pause)
                    waited += pause
                except (urllib.error.URLError, OSError, json.JSONDecodeError,
                        TimeoutError, KeyError, ValueError) as exc:
                    in_flight += time.monotonic() - began
                    return LLMReply(text="",
                                    error=f"{type(exc).__name__}: {exc}")

        err = f"{type(last).__name__}: {last}" if last else "no reply"
        if spent:
            err += " (" + "; ".join(spent) + ")"
        return LLMReply(text="", error=err)

    def complete_json(self, prompt: str, max_tokens: int = 1600,
                      images: list[str] | None = None,
                      retries: int | None = None,
                      effort: str | None = None) -> tuple[Any, LLMReply]:
        """Call, then pull a JSON value out of whatever came back.

        Reuses vision._json_from, which walks the reply for a balanced JSON
        object rather than trusting the model to have obeyed 'JSON only'.
        Returns (parsed_or_None, reply) - a None here is a normal, expected
        outcome that the caller must handle by falling back.
        """
        reply = self.complete(prompt, max_tokens, images, retries, effort)
        parsed, how = _repair.parse(reply.text)
        if how not in ("clean", "empty"):
            # worth seeing: a model that always needs its commas fixed is
            # a prompt problem, and nobody notices a silent repair
            reply.meta = {**(reply.meta or {}), "json_repair": how}

        # This check sits BEFORE the `ok` gate on purpose. A reasoning model
        # that runs out of budget mid-thought returns finish_reason "length"
        # with an EMPTY `content` - the tokens all went to its hidden
        # reasoning - so `ok` is False and an early return here would skip
        # the diagnosis entirely. That is what happened: the Title agent
        # asked for 700 tokens, used all 700, received "" and reported "not
        # a JSON object", blaming the model's formatting for what was
        # actually too small an output budget.
        # Cut off with nothing to show for it: ask once more with room.
        #
        # A reasoning model spends a VARIABLE amount of the output budget
        # thinking before it writes, so a fixed cap that fits one prompt
        # starves another. Caption and Title both used exactly their 1400
        # and returned an empty `content`, while Story used 1524 of 2000
        # and was fine - the same cap, two different outcomes. Guessing a
        # bigger number per agent would just move the cliff, so the budget
        # is doubled once, here, when the reply was truncated before any
        # JSON appeared. It costs a second call only on a reply that was
        # going to be thrown away.
        #
        # A cut-off reply that the repair layer SALVAGED is the same
        # problem wearing a disguise, and it cost a day to see. The
        # EditingDirector asked for 1600 and spent exactly 1600; the
        # reply stopped part-way through its `segments` array; the
        # repair scan walked past the envelope that never closed and
        # returned the last complete segment object inside it. Valid
        # JSON, `parsed is not None`, so this widening never fired - and
        # the director rejected it with "no segments in the reply",
        # which reads as a model that answered badly rather than a
        # budget that ran out. `fragment` is that case named, so the
        # same second call covers it.
        if (reply.truncated and (parsed is None or how == "fragment")
                and max_tokens < _MAX_RETRY_TOKENS):
            wider = min(max_tokens * 2, _MAX_RETRY_TOKENS)
            retried = self.complete(prompt, wider, images, retries, effort)
            _carry_cost(retried, reply)
            parsed, how = _repair.parse(retried.text)
            if how not in ("clean", "empty"):
                retried.meta = {**(retried.meta or {}), "json_repair": how}
            if parsed is not None and how != "fragment":
                retried.meta = {**(retried.meta or {}),
                                "widened_from": max_tokens}
                return parsed, retried
            reply = retried
            max_tokens = wider

        # What survived a cut is not an answer. Returning it would hand
        # the caller one segment of a plan, one beat of a story, and no
        # way of telling that from a model that meant to send exactly
        # that.
        if how == "fragment":
            parsed = None

        # A reply that arrived WHOLE and still is not JSON is a formatting
        # slip, not a budget problem, and the model can fix it if it is
        # told what went wrong. Saying "that was not JSON" alone teaches
        # nothing, so the complaint quotes what actually came back.
        #
        # Found when the daily quota pushed the chain onto a smaller
        # model and Story reported "not a JSON object": a whole reasoned
        # narrative thrown away over punctuation. One retry only - a model
        # that cannot produce JSON twice will not produce it on the third
        # try either, and the deterministic fallback is waiting.
        if (parsed is None and reply.text and not reply.truncated
                and not reply.error):
            again = self.complete(
                _repair.with_complaint(prompt, reply.text),
                max_tokens, images, retries, effort)
            _carry_cost(again, reply)
            parsed, how = _repair.parse(again.text)
            if parsed is not None:
                again.meta = {**(again.meta or {}),
                              "json_retry": True,
                              **({"json_repair": how}
                                 if how != "clean" else {})}
                return parsed, again
            reply = again

        if parsed is None and reply.truncated and not reply.error:
            reply.error = (f"the reply was cut off at the {max_tokens}-token "
                           f"output limit before the JSON was finished")
        if not reply.ok:
            return None, reply
        return parsed, reply

    # ------------------------------------------------------------- http

    def _http(self, prompt: str, max_tokens: int,
              images: list[str] | None = None,
              model_override: str | None = None,
              effort: str | None = None,
              json_mode: bool = True) -> LLMReply:
        key, prov = api_key(), provider()
        use_model, use_base = model_override or model(), base_url()
        # A call carrying images goes to the vision model when one is
        # configured separately - see `vision_profile` for why.
        vp = vision_profile() if images else None
        if vp:
            key, prov = vp["key"], vp["provider"]
            use_model, use_base = vp["model"], vp["base_url"]
        # A User-Agent is not optional in practice. urllib sends
        # "Python-urllib/3.x" by default, and Cloudflare-fronted inference
        # endpoints reject that signature outright: Groq's OpenAI-compatible
        # endpoint answers 403 "error code: 1010" to the exact headers built
        # below and 200 to the identical request with any UA set. The failure
        # looks like a dead API key, so every agent fell back to its
        # deterministic path and reported "no model configured" while a
        # perfectly valid key sat in the environment.
        headers = {"Content-Type": "application/json",
                   "User-Agent": _env("AT_LLM_USER_AGENT", "alpha-titans/1.0")}
        if prov == "anthropic":
            headers["x-api-key"] = key
            headers["anthropic-version"] = _env("AT_VISION_API_VERSION",
                                                "2023-06-01")
            path = "/v1/messages"
            content: Any = prompt
            if images:
                # same shape server/vision.py::_ask already sends, so the
                # two paths cannot drift apart
                content = [{"type": "image",
                            "source": {"type": "base64",
                                       "media_type": "image/jpeg",
                                       "data": b}} for b in images]
                content.append({"type": "text", "text": prompt})
            body = {"model": use_model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content}]}
        else:
            headers["Authorization"] = f"Bearer {key}"
            path = "/v1/chat/completions"
            content = prompt
            if images:
                content = [{"type": "image_url",
                            "image_url": {"url":
                                          f"data:image/jpeg;base64,{b}"}}
                           for b in images]
                content.append({"type": "text", "text": prompt})
            body = {"model": use_model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content}]}
            # Every caller of this client goes through `complete_json`, so
            # every reply is meant to be JSON - say so, rather than asking
            # in the prose and hoping. Checked against both models on this
            # key: gpt-oss-20b and qwen3.8-27b each honour it.
            #
            # TEXT ONLY. The vision path goes to a different provider's
            # OpenAI-compatible endpoint, which may not implement this at
            # all; a 400 there would lose a clip's analysis to a flag that
            # was only ever meant to tidy up formatting. Opt out with
            # AT_LLM_JSON_MODE=0 if a future model rejects it.
            if (json_mode and not images
                    and _env("AT_LLM_JSON_MODE", "1") != "0"):
                body["response_format"] = {"type": "json_object"}
            # Only where it was measured to help - see `_LOW_EFFORT_MODELS`.
            # A model that is not on that list runs exactly as before,
            # which matters because the fallback chain can put any of them
            # behind any agent.
            if effort in EFFORTS and _takes_effort(use_model):
                body["reasoning_effort"] = effort

        req = urllib.request.Request(use_base + path,
                                     data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s()) as r:
            data = json.loads(r.read())

        if prov == "anthropic":
            parts = data.get("content") or []
            text = "".join(p.get("text", "") for p in parts
                           if isinstance(p, dict))
            usage = data.get("usage") or {}
            tin = int(usage.get("input_tokens") or 0)
            tout = int(usage.get("output_tokens") or 0)
            stop = data.get("stop_reason")
        else:
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "")
            usage = data.get("usage") or {}
            tin = int(usage.get("prompt_tokens") or 0)
            tout = int(usage.get("completion_tokens") or 0)
            stop = choice.get("finish_reason")

        return LLMReply(text=text, tokens_in=tin, tokens_out=tout,
                        usd=_price(tin, tout), stop_reason=stop,
                        meta={"provider": prov, "model": use_model})


__all__ = ["LLMClient", "LLMReply", "api_key", "provider", "model"]
