"""
Search provider abstraction - the deterministic half of research.

WHY THIS EXISTS
---------------
A language model does not know what is trending today, and asking it to
pretend is how you get confident, plausible, invented statistics. So the
division here is strict:

    this module          fetches real results from a real provider
    ResearchAgent/LLM    synthesises ONLY what this module returned

If this module returns nothing, the agent reports insufficient signal. It does
not fill the gap from the model's memory.

NO NEW DEPENDENCIES
-------------------
Every provider is plain HTTPS and JSON, reached with urllib - the same way
server/vision.py and llm_tools.py already talk to model APIs. The `tavily`
package happens to be installed in this environment but is NOT in
requirements.txt, so building on it would add an undeclared dependency for no
benefit: its REST endpoint is three lines of urllib.

PROVIDERS
---------
    AT_SEARCH_PROVIDER = tavily | brave | youtube | none   (default: none)
    AT_SEARCH_API_KEY  = ...        no key => provider `none`
    AT_SEARCH_TIMEOUT  = 20         seconds
    AT_SEARCH_MAX      = 8          results per query

`none` is a first-class, supported state - not an error. It returns no
results, the agent degrades honestly, and the video pipeline is unaffected.

EVERY RESULT IS UNTRUSTED
-------------------------
Provider responses are third-party text. Titles, snippets and URLs are
length-capped, URLs are scheme-checked, and nothing from a result is ever
interpolated into a command, a path or a query.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

PROVIDERS = ("tavily", "brave", "youtube", "none")

DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_RESULTS = 8
MAX_TITLE = 200
MAX_SNIPPET = 600
MAX_URL = 500
ALLOWED_SCHEMES = ("http", "https")


@dataclass
class SearchHit:
    """One result, already sanitised. Provenance is mandatory."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""                 # which provider produced it
    published_at: str | None = None  # None when the provider gave no date
    retrieved_at: float = field(default_factory=time.time)
    score: float | None = None

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet,
                "source": self.source, "published_at": self.published_at,
                "retrieved_at": self.retrieved_at, "score": self.score}


@dataclass
class SearchOutcome:
    """What a query produced, including why it produced nothing."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    provider: str = "none"
    error: str | None = None
    wall_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


# ------------------------------------------------------------- config

def _env(key: str, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def api_key() -> str | None:
    return _env("AT_SEARCH_API_KEY")


def provider() -> str:
    p = (_env("AT_SEARCH_PROVIDER", "") or "").lower()
    if p in PROVIDERS:
        return p
    # A key with no explicit provider means tavily, the only one whose free
    # tier needs nothing else. No key means no provider, which is supported.
    return "tavily" if api_key() else "none"


def timeout_s() -> float:
    try:
        return max(5.0, float(_env("AT_SEARCH_TIMEOUT", DEFAULT_TIMEOUT)))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def max_results() -> int:
    try:
        return max(1, min(20, int(_env("AT_SEARCH_MAX", DEFAULT_MAX_RESULTS))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS


# --------------------------------------------------------- sanitising

def _clean_url(raw) -> str | None:
    """A URL we are willing to store. Never fetched, only recorded."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    u = raw.strip()[:MAX_URL]
    try:
        parsed = urllib.parse.urlparse(u)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ALLOWED_SCHEMES or not parsed.netloc:
        return None
    return u


def _clean_text(raw, limit: int) -> str:
    if not isinstance(raw, (str, int, float)):
        return ""
    return " ".join(str(raw).split())[:limit]


def _hit(title, url, snippet, source, published=None, score=None
         ) -> SearchHit | None:
    u = _clean_url(url)
    t = _clean_text(title, MAX_TITLE)
    if not u or not t:
        return None                      # provenance is not optional
    pub = _clean_text(published, 40) or None
    try:
        sc = float(score) if score is not None else None
    except (TypeError, ValueError):
        sc = None
    return SearchHit(title=t, url=u, snippet=_clean_text(snippet, MAX_SNIPPET),
                     source=source, published_at=pub, score=sc)


# ------------------------------------------------------------- client

class SearchClient:
    """One query in, sanitised results out.

    `transport` is injectable for tests, exactly as LLMClient does it, so the
    whole agent can be exercised with no key and no network.
    """

    def __init__(self, transport: Callable[[str, int], SearchOutcome] | None = None):
        self._transport = transport

    # ---------------------------------------------------- capability

    def available(self) -> bool:
        return self._transport is not None or (
            provider() != "none" and bool(api_key()))

    def describe(self) -> dict:
        if self._transport is not None:
            return {"enabled": True, "provider": "injected-transport",
                    "live": False, "why_disabled": None}
        if not self.available():
            return {"enabled": False, "provider": provider(), "live": False,
                    "why_disabled": (
                        "AT_SEARCH_API_KEY is not set, so no research provider "
                        "is reachable; the agent reports insufficient signal "
                        "rather than inventing trends")}
        return {"enabled": True, "provider": provider(), "live": True,
                "why_disabled": None}

    # -------------------------------------------------------- search

    def search(self, query: str, limit: int | None = None) -> SearchOutcome:
        limit = limit or max_results()
        started = time.time()

        if self._transport is not None:
            out = self._transport(query, limit)
            out.wall_s = round(time.time() - started, 3)
            return out

        if not self.available():
            return SearchOutcome(query=query, provider=provider(),
                                 error="no search provider configured",
                                 wall_s=0.0)

        p = provider()
        try:
            if p == "tavily":
                hits = self._tavily(query, limit)
            elif p == "brave":
                hits = self._brave(query, limit)
            elif p == "youtube":
                hits = self._youtube(query, limit)
            else:                                    # pragma: no cover
                hits = []
            return SearchOutcome(query=query, hits=hits, provider=p,
                                 wall_s=round(time.time() - started, 3))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, TimeoutError, KeyError, ValueError) as exc:
            return SearchOutcome(query=query, provider=p,
                                 error=f"{type(exc).__name__}: {exc}",
                                 wall_s=round(time.time() - started, 3))

    # ------------------------------------------------------ providers

    def _get_json(self, url: str, headers: dict, data: bytes | None = None
                  ) -> dict:
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=timeout_s()) as r:
            return json.loads(r.read())

    def _tavily(self, query: str, limit: int) -> list[SearchHit]:
        # Two forms of the same credential, deliberately.
        #
        # Tavily's current API reference authenticates with
        # `Authorization: Bearer tvly-...` and shows no key in the body;
        # the body `api_key` field is the older documented form, which
        # this was written against and which some accounts still accept.
        # Nobody here has a key to test with, so sending only one form
        # would be a coin toss settled by a 401 at the worst moment -
        # the first real research run. Both cost nothing and agree.
        body = json.dumps({"api_key": api_key(), "query": query,
                           "max_results": limit,
                           "search_depth": "basic",
                           "include_answer": False}).encode()
        data = self._get_json("https://api.tavily.com/search",
                              {"Content-Type": "application/json",
                               "Authorization": f"Bearer {api_key()}"}, body)
        out = []
        for r in (data.get("results") or [])[:limit]:
            h = _hit(r.get("title"), r.get("url"), r.get("content"),
                     "tavily", r.get("published_date"), r.get("score"))
            if h:
                out.append(h)
        return out

    def _brave(self, query: str, limit: int) -> list[SearchHit]:
        url = ("https://api.search.brave.com/res/v1/web/search?"
               + urllib.parse.urlencode({"q": query, "count": limit}))
        data = self._get_json(url, {"Accept": "application/json",
                                    "X-Subscription-Token": api_key()})
        out = []
        for r in ((data.get("web") or {}).get("results") or [])[:limit]:
            h = _hit(r.get("title"), r.get("url"), r.get("description"),
                     "brave", r.get("age"))
            if h:
                out.append(h)
        return out

    def _youtube(self, query: str, limit: int) -> list[SearchHit]:
        url = ("https://www.googleapis.com/youtube/v3/search?"
               + urllib.parse.urlencode({
                   "part": "snippet", "q": query, "type": "video",
                   "maxResults": limit, "order": "relevance",
                   "key": api_key()}))
        data = self._get_json(url, {"Accept": "application/json"})
        out = []
        for r in (data.get("items") or [])[:limit]:
            sn = r.get("snippet") or {}
            vid = (r.get("id") or {}).get("videoId")
            if not vid:
                continue
            h = _hit(sn.get("title"), f"https://www.youtube.com/watch?v={vid}",
                     sn.get("description"), "youtube", sn.get("publishedAt"))
            if h:
                out.append(h)
        return out


__all__ = ["SearchClient", "SearchHit", "SearchOutcome", "PROVIDERS",
           "provider", "api_key", "max_results"]
