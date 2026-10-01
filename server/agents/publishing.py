"""
Publishing Agent - the plan for publishing the delivered video. Never a
publication.

WHERE IT SITS
-------------
    ... critic -> finish | escalate -> thumbnail -> title -> publishing -> END

PACKAGING's last step. It runs ONCE, after the render/critic cycle has settled
on the video that will be delivered and after its thumbnail and title exist,
so it can never cost a render, trigger a revision or change any deliverable.

WHAT PUBLISHING ACTUALLY EXISTS
-------------------------------
None. The only YouTube code in the project is a read-only search
(`youtube/v3/search`, by API key) that feeds Research. Uploading needs OAuth
consent and `videos.insert`; there is no client library, no credential and no
endpoint for either. This agent reads that from the source on every run, and a
parity test fails the moment an upload path appears.

So it does not publish, and nothing it produces can say it did: `published`
is the literal False, `provider` the literal "none", and there is no field an
upload id, a URL or a metric could go in.

WHAT IT DOES INSTEAD
--------------------
Gathers what was actually produced - the video the app's /download serves,
the thumbnail and the title the app shows, selected by the SAME rules the app
uses - measures each, decodes the video end to end, and checks everything
against YouTube's documented limits locally. The status follows from those
checks and cannot contradict them:

    blocked       there is no sound video to publish
    needs_review  there is, but a person must decide something first
    ready         ready for the OPERATOR to upload by hand

Every outcome ends in the same place: publishing is the operator's act. It is
public and hard to undo, so the plan recommends uploading as Private first and
says, plainly, that nothing has been uploaded.

Deterministic by design: there is nothing here for a model to judge.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

from .artifacts import (TITLE_MAX_CHARS, Artifact, Cost, Decision,
                        JobSettings, PublishAsset, PublishCheck, Publishing,
                        RenderResult, Thumbnail, Title)
from .base import Agent, AgentContext
from .trace import TraceSpan

DECODE_TIMEOUT_S = 900.0
MAX_STEPS = 10

# YouTube's documented limits. Checked HERE, on the files on disk - nothing
# is sent to YouTube to find out.
YT_TITLE_MAX = TITLE_MAX_CHARS             # 100 characters
YT_TITLE_FORBIDDEN = "<>"                  # rejected in a video title
YT_THUMB_MAX_BYTES = 2 * 1024 * 1024       # custom thumbnails: 2 MB
YT_THUMB_MIN_WIDTH = 640                   # custom thumbnails: >= 640 px wide
YT_THUMB_CODECS = ("mjpeg", "png", "gif")  # JPG, PNG or GIF, as ffprobe names

# What would make an UPLOAD path if it ever appeared in the project's CODE -
# an upload endpoint, the client library's insert call, or an OAuth client.
# Matched against code with comments and docstrings removed, so describing
# the absence of an upload path (as this module does) is not mistaken for
# one. Split so this module's own source never matches its own scan.
_UPLOAD_MARKERS = tuple("".join(p) for p in (
    ("youtube/v3/", "videos"), ("upload/", "youtube"),
    ("videos().", "insert"), ("googleapis.com/", "upload"),
    ("googleapi", "client"), ("google_auth_", "oauthlib"),
    ("oauth2", "client"), ("InstalledApp", "Flow")))
_SEARCH_MARKER = "youtube/v3/" + "search"

PLACEHOLDER = JobSettings.model_fields["title"].default

# the check names, so tests and the critic can refer to them exactly
VIDEO_ON_DISK = "the delivered video is on disk"
VIDEO_READABLE = "it has a readable video stream"
VIDEO_DECODES = "it decodes end to end"
CRITIC_PASSED = "the critic passed it"
TITLE_PRESENT = "it has a title"
TITLE_LENGTH = "the title fits YouTube's limit"
TITLE_CHARS = "the title has no characters YouTube rejects"
THUMB_PRESENT = "it has a thumbnail"
THUMB_FORMAT = "the thumbnail is a JPG, PNG or GIF"
THUMB_SIZE = "the thumbnail is under YouTube's size limit"
THUMB_WIDTH = "the thumbnail is wide enough for YouTube"


# ------------------------------------------------------- the real project

def _code(src: str) -> str:
    """A module's CODE: comments dropped by the parser, docstrings removed,
    then unparsed. Prose about an upload path is not an upload path."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:                               # pragma: no cover
        return src
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _platform() -> dict:
    """What this project can do on a video platform, read from its source.

    Scans the code of every python file under server/ and pipeline/ (this
    module excepted, since it names what it is looking for). Returns which
    files search YouTube, and which - if any - could upload.
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
    return {"search": list(found[0]), "upload": list(found[1])}


@functools.lru_cache(maxsize=8)
def _scan(root: str, files: tuple) -> tuple[tuple, tuple]:
    """Keyed on every file's modification time, so an unchanged project is
    not re-parsed - and a changed one always is."""
    search, upload = [], []
    for rel, _mtime in files:
        try:
            src = _code((Path(root) / rel).read_text(encoding="utf-8",
                                                     errors="ignore"))
        except OSError:                               # pragma: no cover
            continue
        if _SEARCH_MARKER in src:
            search.append(rel)
        if any(m in src for m in _UPLOAD_MARKERS):
            upload.append(rel)
    return tuple(search), tuple(upload)


def _capabilities(platform: dict) -> list[str]:
    out = []
    if platform["search"]:
        out.append("YouTube Data API search - read-only, by API key, used by "
                   "Research (" + ", ".join(platform["search"]) + ")")
    if platform["upload"]:
        out.append("an upload path appears in " + ", ".join(platform["upload"])
                   + " - this agent does not use it")
    else:
        out.append("no upload, OAuth or channel access anywhere in the "
                   "project")
    return out


def _ffmpeg() -> str:
    from .tools.pipeline_bridge import config_mod
    return config_mod.FFMPEG


def _identity(path: str | None):
    """A file as it is right now - so a cached plan is reused only for the
    very same bytes on disk."""
    if not path:
        return None
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return (str(p), None)
    return (str(p), st.st_size, st.st_mtime_ns)


def _measure(kind: str, path: str | None) -> PublishAsset | None:
    """The file as it actually is: on disk, non-empty, and what ffprobe reads
    from it. Nothing is reported for a file that is not there."""
    from .tools.verify_tools import probe_final
    if not path:
        return None
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    info = probe_final(p)
    return PublishAsset(
        kind=kind, path=str(p), file_name=p.name, size=p.stat().st_size,
        width=info.get("width"), height=info.get("height"),
        codec=info.get("video_codec"),
        duration=info.get("duration") if kind == "video" else None)


class PublishingAgent(Agent):
    id = "publishing"
    produces = "Publishing"
    uses_llm = False
    agency = "deterministic"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                render: RenderResult,
                thumbnail: Thumbnail | None = None,
                title: Title | None = None,
                verdict: str | None = None,
                revision: int = 0) -> Artifact[Publishing]:
        from . import compat

        started = time.time()
        verdict = verdict if verdict in ("PASS", "REVISE", "ESCALATE") \
            else None
        platform = _platform()
        span.tool("publishing.scan_platform",
                  search=len(platform["search"]),
                  upload=len(platform["upload"]))

        # ---- the deliverables, chosen by the rules the app itself uses ----
        build = Path(ctx.job.build_dir)
        video_path = next((p for p in (render.canonical_path,
                                       render.video_path)
                           if p and Path(p).exists()), None)
        thumb_name = compat._packaged_thumbnail(build, render, thumbnail)
        if thumb_name is not None:
            thumb_path, thumb_from = str(build / thumb_name), \
                "thumbnail_agent"
        elif render.thumbnail_path and Path(render.thumbnail_path).exists():
            thumb_path, thumb_from = render.thumbnail_path, "master_stage"
        else:
            thumb_path, thumb_from = None, None

        op = (ctx.job.settings.title or "").strip()
        if op and op != PLACEHOLDER:
            title_text, title_from = op, "operator"
        else:
            agent_title = compat._published_title(ctx.job.settings, title)
            title_text, title_from = ((agent_title, "title_agent")
                                      if agent_title else ("", None))

        inputs_hash = self.hash_inputs(
            ctx.job.job_id, _identity(video_path), _identity(thumb_path),
            thumb_from, title_text, title_from, verdict, platform,
            YT_TITLE_MAX, YT_THUMB_MAX_BYTES, YT_THUMB_MIN_WIDTH)
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Publishing",
                                         inputs_hash, Publishing)
        if cached is not None:
            span.cache(1)
            span.note("reusing the publishing plan: the delivered files, the "
                      "title and the verdict are byte-for-byte unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        # ---- measure ------------------------------------------------------
        video = _measure("video", video_path)
        thumb = _measure("thumbnail", thumb_path)
        decode_ok, decode_log = self._decode(ctx, span, video)

        checks = self._checks(video, decode_ok, decode_log, verdict,
                              title_text, thumb)
        failed = [c for c in checks if not c.passed]
        status = ("blocked" if any(c.blocking for c in failed) else
                  "needs_review" if failed else "ready")

        payload = Publishing(
            status=status,
            video=video, thumbnail=thumb,
            thumbnail_from=thumb_from if thumb else None,
            title=title_text, title_from=title_from,
            verdict=verdict, checks=checks,
            platform_capabilities=_capabilities(platform),
            next_steps=self._next_steps(ctx.job.job_id, status, video,
                                        thumb, title_text, failed),
            limitations=self._limitations(platform, thumb_from, title_from))

        ctx.emit("publishing: %s - %d check(s), %d failed - NOT published "
                 "(no provider; the operator publishes)"
                 % (status, len(checks), len(failed)))
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, platform),
            confidence=1.0, degraded=False, fallback_used=None,
            cost=Cost(wall_s=round(time.time() - started, 2)))

    # ------------------------------------------------------------ decode

    def _decode(self, ctx: AgentContext, span: TraceSpan,
                video: PublishAsset | None) -> tuple[bool, str]:
        """Decode every frame of the delivered file, now. The process belongs
        to the media pool, so a deadline or a cancel kills it like a render
        stage and nothing outlives the build."""
        if video is None or not video.codec:
            return False, ""
        cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-i", video.path,
               "-f", "null", "-"]
        span.tool("publishing.decode_check", file=video.file_name)
        try:
            rc, log = ctx.pool.run_process(
                cmd, timeout_s=DECODE_TIMEOUT_S, job_id=ctx.job.job_id,
                label="publishing:decode")
        except Exception as exc:                      # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"[:200]
        text = (log or "").strip()
        return rc == 0 and not text, text[:200]

    # ------------------------------------------------------------ checks

    @staticmethod
    def _checks(video: PublishAsset | None, decode_ok: bool,
                decode_log: str, verdict: str | None, title: str,
                thumb: PublishAsset | None) -> list[PublishCheck]:
        c = [PublishCheck(
            check=VIDEO_ON_DISK, passed=video is not None,
            value=(f"{video.file_name}, {video.size} bytes" if video
                   else "missing"),
            limit="the file the app's /download serves", blocking=True)]
        if video is not None:
            c.append(PublishCheck(
                check=VIDEO_READABLE, passed=bool(video.codec and video.width),
                value=(f"{video.codec} {video.width}x{video.height}, "
                       f"{video.duration:.2f}s" if video.codec else
                       "no video stream"), blocking=True))
            c.append(PublishCheck(
                check=VIDEO_DECODES, passed=decode_ok,
                value="no errors" if decode_ok else (decode_log
                                                     or "did not decode"),
                limit="a full decode with no error output", blocking=True))
        c.append(PublishCheck(
            check=CRITIC_PASSED, passed=verdict == "PASS",
            value=verdict or "no verdict", limit="PASS", source="policy"))

        c.append(PublishCheck(
            check=TITLE_PRESENT, passed=bool(title),
            value=(repr(title[:80]) if title else
                   f"none - the placeholder {PLACEHOLDER} is not a title"),
            source="policy"))
        if title:
            c.append(PublishCheck(
                check=TITLE_LENGTH, passed=len(title) <= YT_TITLE_MAX,
                value=f"{len(title)} characters",
                limit=f"at most {YT_TITLE_MAX}", source="youtube_limit"))
            bad = sorted({ch for ch in title if ch in YT_TITLE_FORBIDDEN})
            c.append(PublishCheck(
                check=TITLE_CHARS, passed=not bad,
                value="none" if not bad else " ".join(bad),
                limit="no " + " or ".join(YT_TITLE_FORBIDDEN),
                source="youtube_limit"))

        c.append(PublishCheck(
            check=THUMB_PRESENT, passed=thumb is not None,
            value=thumb.file_name if thumb else
            "none - YouTube would pick a frame itself", source="policy"))
        if thumb is not None:
            c.append(PublishCheck(
                check=THUMB_FORMAT, passed=thumb.codec in YT_THUMB_CODECS,
                value=str(thumb.codec), limit="JPG, PNG or GIF",
                source="youtube_limit"))
            c.append(PublishCheck(
                check=THUMB_SIZE, passed=thumb.size <= YT_THUMB_MAX_BYTES,
                value=f"{thumb.size} bytes",
                limit=f"at most {YT_THUMB_MAX_BYTES} bytes (2 MB)",
                source="youtube_limit"))
            c.append(PublishCheck(
                check=THUMB_WIDTH,
                passed=(thumb.width or 0) >= YT_THUMB_MIN_WIDTH,
                value=f"{thumb.width}x{thumb.height}",
                limit=f"at least {YT_THUMB_MIN_WIDTH} px wide",
                source="youtube_limit"))
        return c

    # ------------------------------------------------------- next steps

    @staticmethod
    def _next_steps(job_id: str, status: str, video, thumb, title: str,
                    failed: list[PublishCheck]) -> list[str]:
        if status == "blocked":
            return (["Nothing can be published: "
                     + "; ".join(f"{c.check} ({c.value})"
                                 for c in failed if c.blocking)[:300],
                     "Fix the render and build again - this plan does not "
                     "work around a missing or broken video"])[:MAX_STEPS]
        steps = ["Review the video in the app - nothing has left this "
                 "machine"]
        steps += [f"Resolve first: {c.check} ({c.value}"
                  + (f"; needs {c.limit}" if c.limit else "") + ")"
                  for c in failed]
        steps.append(f"Download it from /api/jobs/{job_id}/download "
                     f"({video.file_name})")
        steps.append(f'Use the title "{title}"' if title else
                     "Name the video first - it has no title yet")
        if thumb is not None:
            steps.append(f"Upload {thumb.file_name} as its custom thumbnail")
        steps.append("Upload it yourself in YouTube Studio as Private, check "
                     "it there, then choose who can see it. Nothing in this "
                     "project can upload, and nothing has been uploaded.")
        return [s[:300] for s in steps][:MAX_STEPS]

    @staticmethod
    def _limitations(platform: dict, thumb_from, title_from) -> list[str]:
        out = ["no publishing provider exists, so this is a plan for a person "
               "to carry out - no upload, URL, view count or publish time "
               "exists to report"]
        if platform["upload"]:
            out.append("an upload path appeared in the project; it is not "
                       "used here and the operator's approval would still "
                       "be required")
        if thumb_from == "master_stage":
            out.append("the Thumbnail agent's image was not usable for this "
                       "render, so master_stage's frame is in the plan")
        if title_from is None:
            out.append("there is no title to publish under")
        out.append("YouTube's limits are checked locally against its "
                   "documented values; YouTube itself was not asked")
        return out

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Publishing, platform: dict) -> list[Decision]:
        out = [Decision(
            what="Publishing provider",
            chose="none - nothing in this project can upload",
            why="; ".join(p.platform_capabilities)[:300] + " - read from "
                "the project's source on this run, not assumed",
            decided_by="measurement", confidence=1.0)]

        v = p.video
        out.append(Decision(
            what="Delivered video",
            chose=(f"{v.file_name} {v.codec} {v.width}x{v.height} "
                   f"{v.duration:.2f}s" if v and v.codec else
                   v.file_name if v else "missing"),
            why=("the file the app's /download serves, decoded end to end "
                 "at packaging time through the media pool"),
            decided_by="measurement", confidence=1.0))

        out.append(Decision(
            what="Thumbnail in the plan",
            chose=(f"{p.thumbnail.file_name} ({p.thumbnail_from})"
                   if p.thumbnail else "none"),
            why=("the image the app shows for this video, chosen by the same "
                 "rule: the Thumbnail agent's frame when it was generated "
                 "from this very render, master_stage's otherwise"),
            decided_by="policy", confidence=1.0))

        out.append(Decision(
            what="Title in the plan",
            chose=(p.title[:200] if p.title else "none"),
            why=("the operator named the video; operator settings win"
                 if p.title_from == "operator" else
                 "the Title agent's title, which the app shows in place of "
                 "the placeholder" if p.title_from == "title_agent" else
                 "the operator named nothing and no supportable title was "
                 "found, so there is nothing to publish it under"),
            decided_by="human" if p.title_from == "operator" else "policy",
            confidence=1.0))

        failed = [c for c in p.checks if not c.passed]
        out.append(Decision(
            what="Readiness",
            chose=f"{p.status}: {len(p.checks)} check(s), {len(failed)} "
                  f"failed",
            why=("; ".join(f"{c.check} ({c.value})" for c in failed)[:300]
                 or "every check passed"),
            decided_by="measurement", confidence=1.0))

        out.append(Decision(
            what="Operator approval",
            chose="required; recommended visibility: private",
            why=("publishing is public and hard to undo, so it is the "
                 "operator's act and never this agent's - a first upload "
                 "should be seen on the platform by its owner before anyone "
                 "else"),
            decided_by="policy", confidence=1.0))
        return out


__all__ = ["PublishingAgent"]
