"""
Research Agent - what the world currently says about this video's subject.

THE ONE RULE
------------
The model does not know what is trending. Asking it to is how you get
confident, plausible, invented statistics with no source. So:

    SearchClient   fetches real results from a real provider   (deterministic)
        |
        v
    evidence       titles, urls, snippets, dates, retrieved_at
        |
        v
    the model      synthesises ONLY what is in that list        (reasoning)

Every trend and every opportunity must cite evidence URLs that came back from
the provider. A citation the model invents is dropped in validation. If the
provider returned nothing, the report says `insufficient_signal` and stops -
it does not fall back to the model's memory, because a fabricated trend is
worse than an absent one.

INFORMATION, NOT DECISIONS
--------------------------
This agent does not choose the story, the script, the edit, the title, the
thumbnail or when to publish. It hands ContentStrategy a sourced picture and
gets out of the way.

TIME
----
Research is perishable. Every evidence item carries `retrieved_at` from this
run and `published_at` only when the provider supplied one - never guessed.
Undated evidence is counted and reported as a limitation rather than quietly
treated as current.

NEVER BLOCKING
--------------
No key, a timeout, a rate limit, an empty result set or an unusable reply all
produce a valid degraded report. The video pipeline does not depend on the
internet being available.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone

from .artifacts import (Artifact, ContentOpportunity, Cost, Decision,
                        EvidenceItem, ResearchReport, TrendItem,
                        VisionAnalysis)
from .base import Agent, AgentContext
from .tools.search_tools import SearchClient, SearchHit
from .trace import TraceSpan

# Trend research goes stale fast. An hour is long enough that a revision or a
# retry reuses the same picture, short enough that tomorrow's build re-reads
# the world. Configurable, because "fast" differs by subject.
DEFAULT_TTL_S = 3600
MAX_QUERIES = 3
MAX_EVIDENCE = 24
MAX_LIST = 10
PROMPT_VERSION = "research-prompt-v1"

SYNTHESIS_PROMPT = """Today is {today}.

You are researching the subject of a short video: {subject}

Below are REAL search results retrieved just now. Synthesise ONLY what is in
them. Do not add trends, numbers or claims from your own memory - if the
evidence does not support something, leave it out.

EVIDENCE
{evidence}

Every trend and opportunity must cite `evidence_urls` taken from the list
above. A citation not in that list will be discarded.

Reply with ONLY a JSON object:

{{
  "trends": [
    {{"label": "<short name>", "observation": "<what the evidence shows>",
      "evidence_urls": ["<url from above>"], "confidence": <0.0-1.0>}}
  ],
  "topics": ["<specific topics the evidence suggests are live right now>"],
  "keywords": ["<search terms an audience is plausibly using>"],
  "audience_signals": ["<what the evidence implies about what people want>"],
  "content_opportunities": [
    {{"idea": "<an angle worth considering>", "rationale": "<why, from the evidence>",
      "evidence_urls": ["<url from above>"], "confidence": <0.0-1.0>}}
  ],
  "limitations": ["<what this evidence does NOT tell us>"],
  "confidence": <0.0-1.0, overall - be low when the evidence is thin or undated>
}}

If the evidence is too thin to say anything useful, return empty lists and a
low confidence. That is a valid and useful answer."""


def _ttl_s() -> int:
    try:
        return max(0, int(os.environ.get("AT_RESEARCH_TTL", DEFAULT_TTL_S)))
    except (TypeError, ValueError):
        return DEFAULT_TTL_S


def _num(raw, lo: float, hi: float, fallback: float = 0.0) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    return fallback if not math.isfinite(v) else max(lo, min(hi, v))


def _texts(raw, limit: int = MAX_LIST, each: int = 120) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw[:limit]:
        if isinstance(x, (str, int, float)):
            t = " ".join(str(x).split())[:each]
            if t:
                out.append(t)
    return out


class ResearchAgent(Agent):
    id = "research"
    produces = "ResearchReport"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                vision: VisionAnalysis | None = None,
                revision: int = 0) -> Artifact[ResearchReport]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        search = SearchClient()
        sdesc = search.describe()
        ldesc = ctx.llm.describe()

        # ---- no subject: refuse to invent one --------------------------
        if not domain:
            return self._degraded(
                ctx, span, domain, [], revision, started, sdesc, provider_ok=False,
                why=("no domain or topic was given for this job, so there is "
                     "nothing to research; a subject guessed from the footage "
                     "would be invention, not research"),
                limitation="set JobSettings.domain to enable research")

        queries = self._queries(domain, vision)

        # ---- freshness-aware cache -------------------------------------
        inputs_hash = self.hash_inputs(
            domain, queries, sdesc.get("provider"), ldesc.get("model"),
            PROMPT_VERSION,
            # a time bucket, so identical research is reused within the TTL
            # and genuinely re-read after it
            int(time.time() // _ttl_s()) if _ttl_s() else time.time(),
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "ResearchReport",
                                         inputs_hash, ResearchReport)
        if cached is not None:
            age = time.time() - cached.payload.retrieved_at
            span.note(f"reusing research from {age:.0f}s ago "
                      f"(TTL {_ttl_s()}s)")
            span.cache(len(cached.payload.evidence))
            span.skipped("cache hit within freshness window")
            return cached

        # ---- deterministic collection ----------------------------------
        if not search.available():
            return self._degraded(
                ctx, span, domain, queries, revision, started, sdesc,
                provider_ok=False,
                why=sdesc.get("why_disabled") or "no search provider",
                limitation=("no research provider is configured, so no "
                            "current evidence could be gathered"))

        evidence: list[EvidenceItem] = []
        errors: list[str] = []
        for q in queries:
            span.tool("search.query", query=q, provider=sdesc.get("provider"))
            out = search.search(q)
            if not out.ok:
                errors.append(f"{q}: {out.error}")
                continue
            for h in out.hits:
                if len(evidence) >= MAX_EVIDENCE:
                    break
                evidence.append(self._evidence(h, q))

        if not evidence:
            return self._degraded(
                ctx, span, domain, queries, revision, started, sdesc,
                provider_ok=True,
                why=("the provider returned no usable results"
                     + (f" ({'; '.join(errors[:2])})" if errors else "")),
                limitation="no evidence was retrieved for these queries")

        span.note(f"{len(evidence)} evidence item(s), "
                  f"{sum(1 for e in evidence if not e.dated)} undated")

        # ---- synthesis, over that evidence only ------------------------
        if not ctx.llm.available():
            return self._evidence_only(ctx, span, domain, queries, evidence,
                                       revision, started, sdesc,
                                       ldesc.get("why_disabled") or
                                       "no model configured")

        payload, cost = self._synthesise(ctx, span, domain, queries, evidence,
                                         sdesc, ldesc)
        if payload is None:
            return self._evidence_only(ctx, span, domain, queries, evidence,
                                       revision, started, sdesc,
                                       "the model reply was unusable",
                                       cost=cost)

        decisions = self._decisions(payload, live=True)
        ctx.emit("research: %d evidence, %d trend(s), %d opportunity(ies), "
                 "confidence %.2f"
                 % (len(payload.evidence), len(payload.trends),
                    len(payload.content_opportunities), payload.confidence))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=decisions, confidence=payload.confidence,
            degraded=payload.insufficient_signal, fallback_used=None,
            cost=Cost(tokens_in=cost[0], tokens_out=cost[1], usd=cost[2],
                      wall_s=round(time.time() - started, 2)))

    # ---------------------------------------------------------- queries

    def _queries(self, domain: str, vision: VisionAnalysis | None
                 ) -> list[str]:
        """Deterministic query construction - the model does not choose these.

        The year is taken from the real clock rather than from anything a
        model produced, so "recent" means recent.
        """
        year = datetime.now(timezone.utc).year
        qs = [f"{domain} trends {year}", f"popular {domain} videos {year}"]
        # one query grounded in what the footage actually shows, when known
        if vision is not None:
            subjects = [s for c in vision.analysed_clips for s in c.subjects]
            if subjects:
                top = ", ".join(dict.fromkeys(subjects))[:80]
                qs.append(f"{domain} {top}")
        return qs[:MAX_QUERIES]

    @staticmethod
    def _evidence(h: SearchHit, query: str) -> EvidenceItem:
        return EvidenceItem(
            title=h.title, url=h.url, snippet=h.snippet, source=h.source,
            query=query, retrieved_at=h.retrieved_at,
            published_at=h.published_at, score=h.score)

    # -------------------------------------------------------- synthesis

    def _synthesise(self, ctx: AgentContext, span: TraceSpan, domain: str,
                    queries: list[str], evidence: list[EvidenceItem],
                    sdesc: dict, ldesc: dict
                    ) -> tuple[ResearchReport | None, tuple]:
        listing = "\n".join(
            "  - %s\n    url: %s\n    date: %s\n    %s"
            % (e.title, e.url, e.published_at or "UNKNOWN - treat as undated",
               e.snippet[:200])
            for e in evidence)
        prompt = SYNTHESIS_PROMPT.format(
            today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            subject=domain, evidence=listing)

        span.tool("llm.complete_json", purpose="research-synthesis",
                  evidence=len(evidence), prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1600)
        cost = (reply.tokens_in, reply.tokens_out, reply.usd)
        span.llm(*cost)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"synthesis unusable: {reply.error or 'not an object'}")
            return None, cost

        known = {e.url for e in evidence}
        undated = sum(1 for e in evidence if not e.dated)

        limitations = _texts(raw.get("limitations"), each=200)
        if undated:
            limitations.append(
                f"{undated} of {len(evidence)} sources carried no publication "
                f"date, so their currency cannot be confirmed")
        if sdesc.get("provider") == "youtube":
            limitations.append("evidence is YouTube search only, which is a "
                               "platform view rather than the whole web")

        trends = self._cited(raw.get("trends"), known, TrendItem,
                             ("label", "observation"))
        opps = self._cited(raw.get("content_opportunities"), known,
                           ContentOpportunity, ("idea", "rationale"))

        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        # thin or undated evidence caps how confident the report may claim to
        # be, whatever the model said about itself
        if undated == len(evidence):
            confidence = min(confidence, 0.3)
        if len(evidence) < 3:
            confidence = min(confidence, 0.4)

        payload = ResearchReport(
            domain=domain, research_queries=queries,
            provider=sdesc.get("provider") or "none", live=bool(sdesc.get("live")),
            model=ldesc.get("model"), evidence=evidence,
            trends=trends, topics=_texts(raw.get("topics")),
            keywords=_texts(raw.get("keywords"), each=60),
            audience_signals=_texts(raw.get("audience_signals"), each=160),
            content_opportunities=opps,
            insufficient_signal=not (trends or opps),
            limitations=limitations[:MAX_LIST], confidence=confidence)
        return payload, cost

    @staticmethod
    def _cited(raw, known: set[str], model_cls, text_fields: tuple):
        """Keep only items whose citations are real. Uncited claims are dropped.

        This is where "do not fabricate" is enforced rather than requested: a
        trend the model invented has no URL from the evidence list, so it does
        not survive.
        """
        out = []
        if not isinstance(raw, list):
            return out
        for item in raw[:MAX_LIST]:
            if not isinstance(item, dict):
                continue
            urls = [u for u in (item.get("evidence_urls") or [])
                    if isinstance(u, str) and u in known][:5]
            if not urls:
                continue                       # uncited => dropped
            a = " ".join(str(item.get(text_fields[0]) or "").split())[:120]
            b = " ".join(str(item.get(text_fields[1]) or "").split())[:300]
            if not a:
                continue
            kw = {text_fields[0]: a, text_fields[1]: b,
                  "evidence_urls": urls,
                  "confidence": _num(item.get("confidence"), 0.0, 1.0)}
            out.append(model_cls(**kw))
        return out

    # --------------------------------------------------------- degraded

    def _evidence_only(self, ctx, span, domain, queries, evidence, revision,
                       started, sdesc, why, cost=(0, 0, 0.0)):
        """Real evidence, no synthesis. Still useful, and honest about it."""
        span.fallback(f"evidence only ({why})")
        undated = sum(1 for e in evidence if not e.dated)
        payload = ResearchReport(
            domain=domain, research_queries=queries,
            provider=sdesc.get("provider") or "none",
            live=bool(sdesc.get("live")), model=None, evidence=evidence,
            insufficient_signal=True,
            limitations=[f"{why}; {len(evidence)} source(s) were retrieved but "
                         f"not synthesised into trends",
                         f"{undated} source(s) carried no publication date"],
            confidence=0.0)
        ctx.emit("research: %d source(s) retrieved, not synthesised (%s)"
                 % (len(evidence), why))
        return self.wrap(
            ctx, payload, revision=revision,
            inputs_hash=self.hash_inputs(domain, queries, "evidence-only"),
            span=span, decisions=self._decisions(payload, live=False, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"evidence only ({why})",
            cost=Cost(tokens_in=cost[0], tokens_out=cost[1], usd=cost[2],
                      wall_s=round(time.time() - started, 2)))

    def _degraded(self, ctx, span, domain, queries, revision, started, sdesc,
                  provider_ok: bool, why: str, limitation: str):
        """No evidence at all. Says so; invents nothing."""
        span.fallback(f"insufficient signal ({why})")
        payload = ResearchReport(
            domain=domain, research_queries=queries,
            provider=sdesc.get("provider") or "none", live=False, model=None,
            insufficient_signal=True, limitations=[limitation], confidence=0.0)
        ctx.emit(f"research: insufficient signal - {why}")
        return self.wrap(
            ctx, payload, revision=revision,
            inputs_hash=self.hash_inputs(domain, queries, "insufficient",
                                         provider_ok),
            span=span, decisions=self._decisions(payload, live=False, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"insufficient signal ({why})",
            cost=Cost(wall_s=round(time.time() - started, 2)))

    # -------------------------------------------------------- decisions

    def _decisions(self, p: ResearchReport, live: bool,
                   why: str = "") -> list[Decision]:
        out = [Decision(
            what="Research scope",
            chose=(", ".join(p.research_queries) if p.research_queries
                   else "none - no subject given"),
            why=("queries built from the job's declared domain and the real "
                 "current year; the model does not choose what to search for"),
            decided_by="policy", confidence=1.0)]

        if not live:
            out.append(Decision(
                what="Current evidence", chose="insufficient signal",
                why=(f"{why}; reporting nothing is the honest answer - a "
                     f"trend synthesised from model memory would look like "
                     f"research without being it"),
                decided_by="fallback", confidence=1.0))
            return out

        out.append(Decision(
            what="Evidence gathered",
            chose=f"{len(p.evidence)} source(s) from {p.provider}",
            why=(f"{len(p.dated_evidence)} carried a publication date, "
                 f"{p.undated_count} did not and are treated as undated"),
            decided_by="measurement", confidence=1.0))

        if p.trends:
            out.append(Decision(
                what="Trends read from the evidence",
                chose="; ".join(t.label for t in p.trends[:4]),
                why=("each cites a source that was actually retrieved; "
                     "uncited claims were discarded in validation"),
                decided_by="llm",
                confidence=round(sum(t.confidence for t in p.trends)
                                 / len(p.trends), 2)))
        if p.content_opportunities:
            out.append(Decision(
                what="Opportunities for ContentStrategy",
                chose="; ".join(o.idea for o in p.content_opportunities[:3]),
                why=("offered as information - this agent does not choose the "
                     "story, the title or the edit"),
                decided_by="llm", confidence=0.6))
        if p.limitations:
            out.append(Decision(
                what="What this research cannot tell us",
                chose=f"{len(p.limitations)} limitation(s)",
                why="; ".join(p.limitations[:3]),
                decided_by="measurement", confidence=1.0))
        return out


__all__ = ["ResearchAgent"]
