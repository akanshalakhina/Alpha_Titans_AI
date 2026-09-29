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
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .pipeline_bridge import SERVER_DIR  # noqa: F401  (ensures sys.path)

import vision as _vision                 # existing module, unmodified

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

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def _env(key: str, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def api_key() -> str | None:
    return _env("AT_LLM_API_KEY") or _env("AT_VISION_API_KEY")


def provider() -> str:
    p = (_env("AT_LLM_PROVIDER") or _env("AT_VISION_PROVIDER", "anthropic") or "").lower()
    return p if p in DEFAULT_MODEL else "anthropic"


def model() -> str:
    return _env("AT_LLM_MODEL") or _env("AT_VISION_MODEL") or DEFAULT_MODEL[provider()]


def base_url() -> str:
    return (_env("AT_LLM_BASE_URL") or _env("AT_VISION_BASE_URL")
            or DEFAULT_BASE[provider()]).rstrip("/")


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
                 images: list[str] | None = None) -> LLMReply:
        """One call. `images` are base64 JPEGs, for the vision path.

        The injected transport is given the images too, so a test can assert
        what the model was actually shown without a key or a network.
        """
        if self._transport is not None:
            try:
                return self._transport(prompt, max_tokens, images)
            except TypeError:
                # transports written before images existed take two arguments
                return self._transport(prompt, max_tokens)
        if not api_key():
            return LLMReply(text="", error="no API key configured")
        try:
            return self._http(prompt, max_tokens, images)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, TimeoutError, KeyError, ValueError) as exc:
            return LLMReply(text="", error=f"{type(exc).__name__}: {exc}")

    def complete_json(self, prompt: str, max_tokens: int = 1600,
                      images: list[str] | None = None) -> tuple[Any, LLMReply]:
        """Call, then pull a JSON value out of whatever came back.

        Reuses vision._json_from, which walks the reply for a balanced JSON
        object rather than trusting the model to have obeyed 'JSON only'.
        Returns (parsed_or_None, reply) - a None here is a normal, expected
        outcome that the caller must handle by falling back.
        """
        reply = self.complete(prompt, max_tokens, images)
        if not reply.ok:
            return None, reply
        return _vision._json_from(reply.text), reply

    # ------------------------------------------------------------- http

    def _http(self, prompt: str, max_tokens: int,
              images: list[str] | None = None) -> LLMReply:
        key, prov = api_key(), provider()
        headers = {"Content-Type": "application/json"}
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
            body = {"model": model(), "max_tokens": max_tokens,
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
            body = {"model": model(), "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content}]}

        req = urllib.request.Request(base_url() + path,
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
        else:
            text = ((data.get("choices") or [{}])[0].get("message") or {}
                    ).get("content", "")
            usage = data.get("usage") or {}
            tin = int(usage.get("prompt_tokens") or 0)
            tout = int(usage.get("completion_tokens") or 0)

        return LLMReply(text=text, tokens_in=tin, tokens_out=tout,
                        usd=_price(tin, tout),
                        meta={"provider": prov, "model": model()})


__all__ = ["LLMClient", "LLMReply", "api_key", "provider", "model"]
