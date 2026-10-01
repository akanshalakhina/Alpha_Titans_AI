"""
PerformanceAnalysis Agent - how the delivered video performed, and an honest
"unavailable" for as long as nobody can know.

WHERE IT SITS
-------------
    ... thumbnail -> title -> publishing -> performance_analysis -> END

After packaging. It runs ONCE, after the publishing plan exists, so it can
never cost a render, trigger a revision or change any deliverable.

WHAT PERFORMANCE DATA ACTUALLY EXISTS
-------------------------------------
None. Performance needs three things, and the project has none of them:

    a published video        nothing publishes; `Publishing.published` is the
                             literal False, and the operator's own upload is
                             never reported back
    a record of WHICH video  no setting, artifact or route holds an uploaded
                             video's id or URL
    analytics access         no YouTube Analytics, statistics or channel-access
                             code anywhere. The only YouTube call is Research's
                             read-only search, which returns OTHER creators'
                             videos and no statistics at all

This agent reads all three on every run - the publishing record from the
store, the typed records' fields and the project's source - and parity tests
fail the moment any of them changes.

WHAT IT DOES INSTEAD
--------------------
Reports `unavailable`, with the measured reasons. Lists every metric as
unmeasured, with what reading it would need. And names the pre-publication
signals that look like performance but are not - the critic's score, a
"ready" publishing plan, Research's evidence, the title's `published` flag -
as EXCLUDED, without quoting a number from any of them.

That is the deterministic fallback, and today it is the only path: the
artifact is marked degraded at confidence 0.0, exactly as Research is when it
has no signal. Nothing is estimated, predicted or modelled - a guessed view
count, once written down, is indistinguishable from a measured one.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

from .artifacts import (Artifact, Cost, CriticResult, Decision,
                        ExcludedSignal, JobSettings, PerformanceAnalysis,
                        PerformanceGap, PublishAsset, Publishing,
                        ResearchReport, Title, UnmeasuredMetric)
from .base import Agent, AgentContext
from .publishing import _code
from .trace import TraceSpan

MAX_STEPS = 6

# What would make an ANALYTICS path if it ever appeared in the project's
# CODE: the Analytics or Reporting API, a video statistics read, their
# client-library calls, the analytics OAuth scopes, or any OAuth client that
# could reach a channel at all. Matched against code with comments and
# docstrings removed, so prose about their absence (as here) is not mistaken
# for one. Split so this module's own source never matches its own scan.
_ANALYTICS_MARKERS = tuple("".join(p) for p in (
    ("youtube", "Analytics"), ("youtubeanalytics", ".googleapis"),
    ("youtube", "reporting"), ("youtube/v3/", "videos"),
    ("videos().", "list"), ("reports().", "query"), ("yt-", "analytics"),
    ("googleapi", "client"), ("google_auth_", "oauthlib"),
    ("oauth2", "client"), ("InstalledApp", "Flow")))
_SEARCH_MARKER = "youtube/v3/" + "search"

# A field of these names, on a typed record, would be somewhere an uploaded
# video's identity could be kept.
_REFERENCE_TOKENS = ("url", "video_id", "upload_id", "youtube", "watch",
                     "permalink")
_REFERENCE_MODELS = (JobSettings, Publishing, PublishAsset)

_PUBLIC = ("the uploaded video's id and a read of its public statistics - "
           "this project records no id and makes no such read")
_OWNER = ("the channel owner's analytics, which need the owner's OAuth "
          "consent - nothing in this project has channel access")
NEEDS: dict[str, str] = {
    "views": _PUBLIC, "likes": _PUBLIC, "comments": _PUBLIC,
    "impressions": _OWNER, "click_through_rate": _OWNER,
    "average_view_duration": _OWNER, "audience_retention": _OWNER,
    "watch_time": _OWNER, "subscribers_gained": _OWNER,
    "revenue": ("the channel owner's analytics on a monetised channel, "
                "with the owner's OAuth consent - nothing in this project "
                "has channel access"),
}


# ------------------------------------------------------- the real project

def _analytics() -> dict:
    """What this project can read back from a video platform, from its source.

    Scans the code of every python file under server/ and pipeline/ (this
    module excepted, since it names what it is looking for). Returns which
    files search YouTube, and which - if any - could read analytics.
    """
    root = Path(__file__).resolve().parents[2]
    me = Path(__file__).resolve()
    files = []
    for base in (root / "server", root / "pipeline"):
        for py in sorted(base.rglob("*.py")):
            if py.resolve() == me or "__pycache__" in py.parts:
                continue
            try:
                files.append((py.relative_to(root).as_posix(),
                              py.stat().st_mtime_ns))
            except OSError:                           # pragma: no cover
                continue
    found = _scan(str(root), tuple(files))
    return {"search": list(found[0]), "analytics": list(found[1])}


@functools.lru_cache(maxsize=8)
def _scan(root: str, files: tuple) -> tuple[tuple, tuple]:
    """Keyed on every file's modification time, so an unchanged project is
    not re-parsed - and a changed one always is."""
    search, analytics = [], []
    for rel, _mtime in files:
        try:
            src = _code((Path(root) / rel).read_text(encoding="utf-8",
                                                     errors="ignore"))
        except OSError:                               # pragma: no cover
            continue
        if _SEARCH_MARKER in src:
            search.append(rel)
        if any(m in src for m in _ANALYTICS_MARKERS):
            analytics.append(rel)
    return tuple(search), tuple(analytics)


def _video_references() -> list[str]:
    """Every field, on the records an upload would be noted in, that could
    hold an uploaded video's identity. Read from the schemas themselves."""
    return [f"{m.__name__}.{name}" for m in _REFERENCE_MODELS
            for name in m.model_fields
            if any(t in name.lower() for t in _REFERENCE_TOKENS)]


def _capabilities(scan: dict, refs: list[str]) -> list[str]:
    out = []
    if scan["search"]:
        out.append("YouTube Data API search - read-only, used by Research ("
                   + ", ".join(scan["search"]) + "); it returns other "
                   "creators' videos and no statistics")
    if scan["analytics"]:
        out.append("an analytics or channel-access path appears in "
                   + ", ".join(scan["analytics"])
                   + " - this agent does not use it")
    else:
        out.append("no YouTube Analytics, video statistics or channel access "
                   "anywhere in the project")
    out.append("a field that could hold an uploaded video's id or URL: "
               + (", ".join(refs) if refs else "none"))
    return out


# ------------------------------------------------- pre-publication signals

def _excluded(publishing: Publishing | None, critic: CriticResult | None,
              research: ResearchReport | None,
              title: Title | None) -> list[ExcludedSignal]:
    """What this job HAS that could pass for performance, and is not.

    Named and explained, never quoted: not one number from any of them is
    carried into the report."""
    out = []
    if critic is not None:
        out.append(ExcludedSignal(
            signal="the critic's verdict and scores", artifact="CriticResult",
            why=("measured on the rendered file before anyone watched it - "
                 "the hook score is a motion heuristic, not audience "
                 "retention")))
    if publishing is not None:
        out.append(ExcludedSignal(
            signal="the publishing plan's readiness checks",
            artifact="Publishing",
            why=("checks of files on disk against YouTube's documented "
                 "limits - \"ready\" means ready to upload, not received "
                 "well")))
    if research is not None and (research.evidence or research.trends
                                 or research.audience_signals):
        out.append(ExcludedSignal(
            signal="Research's evidence, trends and audience signals",
            artifact="ResearchReport",
            why=("gathered about the subject before this video existed - "
                 "other people's videos and pages, never this one")))
    if title is not None and title.published:
        out.append(ExcludedSignal(
            signal="the title's published flag", artifact="Title",
            why=("means the app shows the Title agent's title in place of "
                 "the placeholder - not that a video was published")))
    return out


class PerformanceAnalysisAgent(Agent):
    id = "performance_analysis"
    produces = "PerformanceAnalysis"
    uses_llm = False
    agency = "deterministic"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                publishing: Publishing | None = None,
                critic: CriticResult | None = None,
                research: ResearchReport | None = None,
                title: Title | None = None,
                revision: int = 0) -> Artifact[PerformanceAnalysis]:
        started = time.time()

        # ---- what the project can see after publishing, read now ----------
        scan = _analytics()
        span.tool("performance.scan_analytics",
                  analytics=len(scan["analytics"]),
                  search=len(scan["search"]))
        refs = _video_references()
        span.tool("performance.find_video_reference", fields=len(refs))
        span.tool("performance.read_publication",
                  plan=publishing is not None,
                  published=bool(publishing and publishing.published),
                  status=publishing.status if publishing else None)
        excluded = _excluded(publishing, critic, research, title)

        inputs_hash = self.hash_inputs(
            ctx.job.job_id,
            (publishing.status, publishing.published,
             [c.check for c in publishing.checks if not c.passed])
            if publishing else None,
            scan, refs, [(e.artifact, e.signal) for e in excluded],
            sorted(NEEDS))
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "PerformanceAnalysis",
                                         inputs_hash, PerformanceAnalysis)
        if cached is not None:
            span.cache(1)
            span.note("reusing the performance report: the publication, the "
                      "project's analytics access and the signals set aside "
                      "are unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        gaps = self._gaps(publishing, scan, refs)
        payload = PerformanceAnalysis(
            publishing_status=publishing.status if publishing else None,
            gaps=gaps,
            unmeasured=[UnmeasuredMetric(metric=m, needs=n)
                        for m, n in NEEDS.items()],
            excluded_signals=excluded,
            platform_capabilities=_capabilities(scan, refs),
            next_steps=self._next_steps(publishing, excluded),
            limitations=self._limitations(scan, excluded))

        why = self._why(payload)
        span.fallback(f"no performance data ({why})")
        ctx.emit(f"performance: unavailable - {why}; nothing was estimated")
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, publishing, why),
            confidence=0.0, degraded=True,
            fallback_used=f"no performance data ({why})",
            cost=Cost(wall_s=round(time.time() - started, 2)))

    # -------------------------------------------------------------- gaps

    @staticmethod
    def _gaps(publishing: Publishing | None, scan: dict,
              refs: list[str]) -> list[PerformanceGap]:
        if publishing is None:
            out = [PerformanceGap(
                reason="no_publishing_plan",
                evidence=("no publishing plan exists for this job, so "
                          "nothing records a publication - and nothing in "
                          "this project can publish"))]
        else:
            out = [PerformanceGap(
                reason="not_published",
                evidence=(f"the publishing plan (status {publishing.status}) "
                          f"records published=False: nothing in this "
                          f"project can upload, and an upload made by hand "
                          f"is never reported back"))]
        if not refs:
            out.append(PerformanceGap(
                reason="no_video_reference",
                evidence=("no setting or record has a field for an uploaded "
                          "video's id or URL ("
                          + ", ".join(m.__name__ for m in _REFERENCE_MODELS)
                          + " read), so there is no video to look up")))
        out.append(PerformanceGap(
            reason="no_analytics_provider",
            evidence=(("no YouTube Analytics, statistics or channel-access "
                       "code in server/ or pipeline/ (scanned on this run)")
                      if not scan["analytics"] else
                      ("an analytics path appears in "
                       + ", ".join(scan["analytics"])
                       + ", but nothing here is wired to it"))))
        return out

    @staticmethod
    def _why(p: PerformanceAnalysis) -> str:
        parts = {"not_published": "the video has not been published",
                 "no_publishing_plan": "no publishing plan exists",
                 "no_video_reference": "nothing records which video it would "
                                       "be",
                 "no_analytics_provider": "nothing in this project can read "
                                          "analytics"}
        return "; ".join(parts[r] for r in p.reasons)

    # ------------------------------------------------------- next steps

    @staticmethod
    def _next_steps(publishing: Publishing | None,
                    excluded: list[ExcludedSignal]) -> list[str]:
        if publishing is None:
            steps = ["Make the publishing plan first - this job has none, so "
                     "nothing records what would be published"]
        elif publishing.status != "ready":
            open_ = [c.check for c in publishing.checks if not c.passed]
            steps = [f"Resolve the publishing plan first - it is "
                     f"{publishing.status}"
                     + (": " + "; ".join(open_) if open_ else "")]
        else:
            steps = ["Publish it by hand, following the publishing plan - "
                     "performance only exists once people can watch it"]
        steps.append("Read how it performs in YouTube Studio's analytics for "
                     "the uploaded video - nothing in this project can read "
                     "it, and this report will stay unavailable")
        if excluded:
            steps.append("Do not read the critic's score or a \"ready\" plan "
                         "as a forecast - they check the file, not an "
                         "audience")
        return [s[:300] for s in steps][:MAX_STEPS]

    @staticmethod
    def _limitations(scan: dict, excluded) -> list[str]:
        out = ["performance needs a published video, a record of which video "
               "it is, and access to its analytics - this report is "
               "unavailable until all three exist",
               "nothing is estimated, predicted or modelled: no view count, "
               "click-through rate, retention, likes or revenue is reported, "
               "because none was measured"]
        if scan["search"]:
            out.append("Research's YouTube search returns other creators' "
                       "videos without statistics - it says nothing about "
                       "this one")
        if excluded:
            out.append("pre-publication signals were set aside, not "
                       "converted: " + ", ".join(e.signal for e in excluded))
        return out

    # -------------------------------------------------------- decisions

    def _decisions(self, p: PerformanceAnalysis,
                   publishing: Publishing | None, why: str) -> list[Decision]:
        gap = {g.reason: g for g in p.gaps}
        out = [Decision(
            what="Analytics provider",
            chose="none - nothing in this project can read a video's "
                  "performance",
            why="; ".join(p.platform_capabilities)[:300] + " - read from "
                "the project's source and schemas on this run, not assumed",
            decided_by="measurement", confidence=1.0)]

        pub = gap.get("not_published") or gap["no_publishing_plan"]
        out.append(Decision(
            what="Publication",
            chose=("not published" if publishing is not None
                   else "no publishing plan"),
            why=pub.evidence,
            decided_by="measurement", confidence=1.0))

        out.append(Decision(
            what="Performance data",
            chose="unavailable - " + ", ".join(m.metric
                                               for m in p.unmeasured)
                  + " are all unmeasured",
            why=(f"{why}; reporting nothing is the honest answer - a number "
                 f"here would be invented, and once written down it would "
                 f"look exactly like a measurement"),
            decided_by="fallback", confidence=1.0))

        out.append(Decision(
            what="Pre-publication signals",
            chose=("excluded: " + ", ".join(e.signal
                                            for e in p.excluded_signals)
                   if p.excluded_signals else "none present"),
            why=("each was produced before anyone could watch the video - "
                 "a check of the file or of the subject, never of how this "
                 "video was received - so none is reported as performance"),
            decided_by="policy", confidence=1.0))
        return out


__all__ = ["PerformanceAnalysisAgent"]
