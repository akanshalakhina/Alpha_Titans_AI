"""
Thumbnail Agent - which frame of the finished video represents it.

WHERE IT SITS
-------------
    ... editing_director -> render -> critic --PASS-----> finish   --+
                               ^          |                          +--> thumbnail -> END
                               +-REVISE---+--ESCALATE--> escalate --+

PACKAGING, not editing. It runs ONCE, after the render/critic cycle has
settled on the video that will be delivered, so it can never cost a render,
trigger a revision or change a single frame of the video itself.

WHAT THE PIPELINE ACTUALLY DOES
-------------------------------
`server/extras.py::master_stage` makes the thumbnail with exactly one
operation:

    ffmpeg -ss <t> -i <final video> -frames:v 1 thumbnail.jpg

One frame of the delivered video, at the video's own resolution, as a JPEG.
No scale, no crop, no text, no overlay, and nothing anywhere in the project
generates an image. So the only real decision is WHICH frame - and that one
has never been made: master_stage takes "the middle of the strongest planned
segment" by `score`, but the agent path projects every segment with score 0.0,
so it lands on the middle of segment 1 every time.

This agent makes that decision and realises it with the SAME operation. It
reads both facts from the source rather than assuming them, and parity tests
fail loudly if either changes.

EVERY CANDIDATE IS A FRAME THE VIDEO CONTAINS
---------------------------------------------
Candidates are built here, never by the model, from three kinds of reference:

    vision_moment    an index into a clip's Vision `interesting_moments`
    measured_window  the middle of a clip's measured best window
    plan_segment     the middle of a segment the final cut kept

A reference is a candidate only if it falls inside a kept segment, clear of
the crossfades. Its position in the delivered video is DERIVED from the
transition offsets the renderer itself recorded. The model sees the list and
answers with an index - it cannot name a time, a clip or a path.

NOTHING IS CLAIMED THAT DID NOT HAPPEN
--------------------------------------
The frame is grabbed through the media pool, then probed. `generated` is set
only when the file exists and ffprobe reads real dimensions from it. A failed
grab is reported as a thumbnail that was not made, and the existing
master_stage thumbnail stays the one that is shown.

FALLBACK
--------
With no model: the Story's hook moment if it made the cut, otherwise the
middle of the strongest planned segment by its measured window score - the
rule master_stage was written to apply.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import time
from pathlib import Path

from .artifacts import (Artifact, ContentStrategy, Cost, Decision, EditPlan,
                        RenderResult, Story, Thumbnail, ThumbnailCandidate,
                        VisionAnalysis)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "thumbnail-prompt-v1"
MAX_CANDIDATES = 24
MAX_CHOICES = 3
MAX_LIST = 8
GRAB_TIMEOUT_S = 120.0

# Seconds kept clear of a segment's edges, beyond the crossfade itself, so a
# candidate is one clean frame of one clip rather than a blend of two.
EDGE_PAD = 0.1
# Two references closer than this inside one segment are the same frame.
SAME_FRAME_S = 0.5

# What a reply might ask to do to the frame - none of which exists here.
TREATMENT_KEYS = ("text", "title", "overlay", "crop", "zoom", "filter",
                  "color", "colour", "style", "treatment", "edit")
TIMING_KEYS = ("at", "time", "timestamp", "seconds", "start", "frame_at",
               "clip_at", "path", "file", "clip_id")

PROMPT = """You are choosing the THUMBNAIL for a finished short video: the one
frame that will represent it.

The video is already cut and rendered. Every option below is a frame that
video actually contains - you choose WHICH, and nothing else.

{brief}{story}
THE FRAMES YOU CAN CHOOSE FROM
{candidates}

WHAT THIS PIPELINE CAN ACTUALLY DO
  It grabs ONE frame of the finished video, at the video's own size, as a
  JPEG. That is all. There is no text, title, crop, zoom, colour change,
  arrow, face cut-out, collage or generated image. If you want any of those,
  put it in `unsupported_requests` and it will be recorded as a gap rather
  than promised.

RULES - these are checked after you reply, not taken on trust
- Choose by `index` from the list above. Any other number is DROPPED.
- Do NOT give a time, a timestamp, a clip name or a file path. There is no
  field for any of them; each frame's position comes from the list.
- Rank up to three, best first. The first valid one is used.

Reply with ONLY a JSON object, no prose around it:

{{
  "choices": [<index>, <index>, <index>],
  "rationale": "<why the first one sells this video>",
  "unsupported_requests": ["<anything you wanted done to the frame that this pipeline cannot do>"],
  "risks": ["<what could make this thumbnail mislead or underperform>"],
  "limitations": ["<what you could not judge from what you were given>"],
  "confidence": <0.0-1.0 - be low when you were given little to work with>
}}"""


# ------------------------------------------------------------- sanitising

def _num(raw, lo: float, hi: float, fallback: float = 0.0) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    return fallback if not math.isfinite(v) else max(lo, min(hi, v))


def _text(raw, limit: int = 200) -> str:
    if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
        return ""
    return " ".join(str(raw).split())[:limit]


def _texts(raw, limit: int = MAX_LIST, each: int = 160) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw[:limit]:
        t = _text(x, each)
        if t:
            out.append(t)
    return out


def _is_index(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ------------------------------------------------------- the real pipeline

def _engine_thumbnail() -> dict:
    """What master_stage actually does to make a thumbnail, from its source.

    Read rather than assumed, the way Transition reads XFADES and Caption
    reads drawtext: if the thumbnail call ever grows a filter, a crop or an
    overlay - or stops being a single frame - this reports it, and the
    parity tests fail instead of the agent quietly mis-describing it.
    """
    from .tools import render_tools as RT
    from .tools.pipeline_bridge import extras_mod
    try:
        src = inspect.getsource(extras_mod.master_stage)
        proj = inspect.getsource(RT.project_manifest)
    except (OSError, TypeError, AttributeError):      # pragma: no cover
        return {"frame_grab": False}
    m = re.search(r'thumb = bd / "([^"]+)"\s*\n\s*run\(\[(.*?)\], on_line\)',
                  src, re.S)
    if not m:
        return {"frame_grab": False}
    call = m.group(2)
    return {
        "file": m.group(1),
        "format": Path(m.group(1)).suffix.lstrip(".").lower(),
        "frame_grab": '"-frames:v", "1"' in call and '"-ss"' in call,
        "filters": any(f in call for f in ('"-vf"', '"-filter_complex"',
                                           "drawtext", "scale", "crop")),
        # master_stage's own choice, and why it never varies on this path
        "rule_by_score": 'key=lambda s: s.get("score", 0)' in src,
        "scores_projected_zero": '"score": 0.0' in proj,
    }


def _engine_ok(engine: dict) -> bool:
    return bool(engine.get("frame_grab") and not engine.get("filters")
                and engine.get("format") in ("jpg", "jpeg"))


def _engine_operations(engine: dict) -> list[str]:
    if not _engine_ok(engine):
        return []
    return ["frame_grab: one frame of the delivered video, at its own "
            "resolution, as a JPEG (the operation extras.master_stage uses)",
            "no scale, crop, text, overlay or compositing on the thumbnail",
            "no image generation anywhere in the pipeline"]


def _ffmpeg() -> str:
    from .tools.pipeline_bridge import config_mod
    return config_mod.FFMPEG


# ------------------------------------------------------------ the timeline

def _timeline(build_dir: str, plan: EditPlan, xfade: float
              ) -> tuple[list[float], float, bool]:
    """Where each segment starts in the delivered video, and the retime.

    The starts are the transition offsets `build_video join` RECORDED in the
    manifest for this very render - not recomputed - provided the manifest
    still describes this plan. `factor` is the retime master_stage applied,
    from its own _audio.json. Returns (starts, factor, measured).
    """
    from .tools.analysis_tools import read_manifest
    n = len(plan.segments)
    m = read_manifest(build_dir) or {}
    offs = m.get("transitions")
    segs = m.get("segments") or []
    same_plan = [s.get("clip") for s in segs] == [s.clip_id
                                                  for s in plan.segments]
    if (same_plan and isinstance(offs, list) and len(offs) == n - 1
            and all(isinstance(x, (int, float)) for x in offs)):
        starts, measured = [0.0] + [float(x) for x in offs], True
    else:
        starts, acc = [0.0], 0.0
        for s in plan.segments[:-1]:
            acc += s.duration - xfade
            starts.append(round(acc, 3))
        measured = False

    factor = 1.0
    try:
        a = json.loads((Path(build_dir) / "_audio.json").read_text())
        f = float(a.get("factor") or 1.0)
        if math.isfinite(f) and f > 0:
            factor = f
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return starts, factor, measured


class ThumbnailAgent(Agent):
    id = "thumbnail"
    produces = "Thumbnail"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                plan: EditPlan, render: RenderResult,
                vision: VisionAnalysis | None = None,
                strategy: ContentStrategy | None = None,
                story: Story | None = None,
                verdict: str | None = None,
                revision: int = 0) -> Artifact[Thumbnail]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()
        verdict = verdict if verdict in ("PASS", "REVISE", "ESCALATE") \
            else None

        v_ok = bool(vision is not None and vision.analysed_clips)
        cs_ok = bool(strategy is not None and not strategy.insufficient_signal)
        st_ok = bool(story is not None and story.beats)

        engine = _engine_thumbnail()
        xfade = float(ctx.job.settings.transition_duration)
        starts, factor, offsets_measured = _timeline(ctx.job.build_dir, plan,
                                                     xfade)
        candidates, unrealisable = self._candidates(
            plan, vision, starts, factor, xfade,
            render.technical.duration or 0.0)

        inputs_hash = self.hash_inputs(
            domain, PROMPT_VERSION, ldesc.get("model"), ldesc.get("provider"),
            self._vision_digest(vision), self._strategy_digest(strategy),
            self._story_digest(story),
            [(s.clip_id, s.start, s.end, s.score) for s in plan.segments],
            (render.video_path, render.technical.duration,
             render.technical.size),
            starts, factor, engine, verdict,
        )
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Thumbnail",
                                         inputs_hash, Thumbnail)
        # a cached image that has since gone is not a thumbnail any more
        if cached is not None and (not cached.payload.generated or Path(
                cached.payload.image_path).exists()):
            span.cache(1)
            span.note("reusing the thumbnail: the delivered video and every "
                      "reference it was chosen from are unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        basis = dict(domain=domain, v_ok=v_ok, cs_ok=cs_ok, st_ok=st_ok,
                     engine=engine, candidates=candidates,
                     unrealisable=unrealisable, verdict=verdict,
                     offsets_measured=offsets_measured, render=render)

        # ---- choose ----------------------------------------------------
        cost = None
        choice = None
        if ctx.llm.available() and len(candidates) > 1:
            choice, cost = self._ask(ctx, span, story, strategy, vision,
                                     plan, basis, started, ldesc)
        if choice is None:
            why = ("only one frame to choose from" if len(candidates) <= 1
                   else basis.get("llm_why")
                   or ldesc.get("why_disabled") or "no model configured")
            chosen, rationale = self._deterministic_choice(candidates, plan,
                                                           story)
            span.fallback(f"deterministic thumbnail ({why})")
            live, model, raw = False, None, {}
            fallback = f"deterministic thumbnail ({why})"
        else:
            chosen, rationale, raw = choice
            live, model, fallback = True, ldesc.get("model"), None

        # ---- realise ---------------------------------------------------
        made = self._grab(ctx, span, candidates[chosen], render, engine,
                          revision)

        payload = self._payload(basis, chosen, rationale, raw, made, live,
                                model, fallback)
        ctx.emit("thumbnail: [%d] %s on %s (segment %d) - %s"
                 % (chosen, candidates[chosen].source,
                    candidates[chosen].clip_id,
                    candidates[chosen].segment_index,
                    f"{Path(payload.image_path).name} {payload.width}x"
                    f"{payload.height}" if payload.generated
                    else "NOT generated: " + made["why"]))

        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, basis, made, fallback),
            confidence=payload.confidence,
            degraded=(not live) or (not payload.generated),
            fallback_used=fallback,
            cost=(cost or Cost()).model_copy(update={
                "wall_s": round(time.time() - started, 2)}))

    # --------------------------------------------------------- candidates

    @staticmethod
    def _candidates(plan: EditPlan, vision: VisionAnalysis | None,
                    starts: list[float], factor: float, xfade: float,
                    duration: float
                    ) -> tuple[list[ThumbnailCandidate], int]:
        """Every referenced frame the delivered video actually contains.

        Walks the KEPT segments, so a clip the cut dropped - or a moment its
        window trimmed away - can never be offered. Returns the candidates
        and how many upstream references fell outside the cut.
        """
        out: list[ThumbnailCandidate] = []
        placed: set[tuple] = set()

        def inside(seg, t: float) -> bool:
            pad = min(xfade + EDGE_PAD, seg.duration / 2)
            return seg.start + pad <= t <= seg.end - pad

        def add(seg, t, source, moment_ref=None, window_ref=None, what=""):
            if len(out) >= MAX_CANDIDATES:
                return
            if any(c.segment_index == seg.index
                   and abs(c.clip_at - t) < SAME_FRAME_S for c in out):
                return                      # the same frame, already offered
            at = (starts[seg.index - 1] + (t - seg.start)) / factor
            hi = max(duration - 0.1, 0.1) if duration else at
            out.append(ThumbnailCandidate(
                index=len(out), clip_id=seg.clip_id, source=source,
                moment_ref=moment_ref, window_ref=window_ref,
                segment_index=seg.index, clip_at=round(t, 3),
                frame_at=round(max(0.1, min(hi, at)), 3), what=what[:160]))

        for seg in plan.segments:
            vc = vision.for_clip(seg.clip_id) if vision else None
            if vc is not None:
                for k, mo in enumerate(vc.interesting_moments):
                    if inside(seg, mo.at):
                        add(seg, mo.at, "vision_moment", moment_ref=k,
                            what=mo.what)
                        placed.add(("m", seg.clip_id, k))
                w = vc.best_window()
                if w is not None and inside(seg, (w.start + w.end) / 2):
                    add(seg, (w.start + w.end) / 2, "measured_window",
                        window_ref=w.index)
                    placed.add(("w", seg.clip_id, w.index))
            add(seg, (seg.start + seg.end) / 2, "plan_segment",
                what=seg.reason if seg.reason else "")

        # references upstream produced that this cut does not contain
        unrealisable = 0
        for vc in (vision.analysed_clips if vision else []):
            for k, _mo in enumerate(vc.interesting_moments):
                if ("m", vc.clip_id, k) not in placed:
                    unrealisable += 1
            w = vc.best_window()
            if w is not None and ("w", vc.clip_id, w.index) not in placed:
                unrealisable += 1
        return out, unrealisable

    # ------------------------------------------------------------- choose

    def _ask(self, ctx: AgentContext, span: TraceSpan, story, strategy,
             vision, plan: EditPlan, basis: dict, started: float,
             ldesc: dict):
        """Returns ((chosen, rationale, raw), cost) or (None, cost)."""
        prompt = self._build_prompt(story, strategy, vision, plan,
                                    basis["candidates"])
        span.tool("llm.complete_json", purpose="thumbnail",
                  candidates=len(basis["candidates"]),
                  prompt_chars=len(prompt))
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=900, effort="low")
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            basis["llm_why"] = "the model reply was unusable"
            span.note(f"thumbnail reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return None, cost

        picks = raw.get("choices")
        if _is_index(picks):
            picks = [picks]
        n = len(basis["candidates"])
        valid = [p for p in (picks if isinstance(picks, list) else [])
                 [:MAX_CHOICES] if _is_index(p) and 0 <= p < n]
        dropped = (len(picks) if isinstance(picks, list) else 0) - len(valid)
        if dropped > 0:
            span.note(f"{dropped} choice(s) were not a listed frame and were "
                      f"dropped")
            basis.setdefault("notes", []).append(
                f"{dropped} choice(s) named a frame that is not on the list "
                f"and were dropped")
        if not valid:
            basis["llm_why"] = "the model chose no listed frame"
            span.note("thumbnail reply chose no listed frame")
            return None, cost
        return (valid[0], _text(raw.get("rationale"), 300), raw), cost

    @staticmethod
    def _deterministic_choice(candidates: list[ThumbnailCandidate],
                              plan: EditPlan, story: Story | None
                              ) -> tuple[int, str]:
        """The story's hook moment if it made the cut; otherwise the middle
        of the strongest planned segment - master_stage's own rule, applied
        to the window scores the plan actually carries."""
        hook = None
        if story is not None and story.beats:
            hook = story.beat_for("hook") or story.beats[0]
        if hook is not None and hook.moment_ref is not None:
            for c in candidates:
                if (c.source == "vision_moment" and c.clip_id == hook.clip_id
                        and c.moment_ref == hook.moment_ref):
                    return c.index, ("the story's hook moment, which made "
                                     "the cut")
        best = max(plan.segments, key=lambda s: (s.score, -s.index))
        mid = (best.start + best.end) / 2
        # the middle itself, or the offered frame standing in for it when a
        # moment or window already sat on the same spot
        own = [c for c in candidates if c.segment_index == best.index]
        if own:
            c = min(own, key=lambda c: abs(c.clip_at - mid))
            return c.index, (f"the middle of the strongest planned segment "
                             f"({best.index}, window score {best.score:.2f}) "
                             f"- master_stage's own rule, on the scores the "
                             f"plan carries")
        return 0, "the first frame offered"           # pragma: no cover

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, story, strategy, vision, plan: EditPlan,
                      candidates: list[ThumbnailCandidate]) -> str:
        brief = ""
        if strategy is not None and not strategy.insufficient_signal:
            rows = ["THE BRIEF"]
            for label, val in (("topic", strategy.central_topic),
                               ("angle", strategy.core_angle),
                               ("audience", strategy.audience)):
                if val:
                    rows.append(f"  {label:<9} {val}")
            if strategy.hook.clip_id or strategy.hook.description:
                rows.append("  hook      %s" % (strategy.hook.description
                                                or strategy.hook.clip_id))
            brief = "\n".join(rows) + "\n"
        stext = ""
        if story is not None and story.beats:
            rows = ["\nTHE STORY"]
            if story.logline:
                rows.append(f"  logline   {story.logline}")
            for b in story.beats:
                rows.append("    %-8s %s" % (b.role, b.narrative or b.clip_id))
            stext = "\n".join(rows) + "\n"

        n = len(plan.segments)
        rows = []
        seen_clips = set()
        for c in candidates:
            vc = vision.for_clip(c.clip_id) if vision else None
            if vc is not None and c.clip_id not in seen_clips and vc.summary:
                rows.append(f"  clip {c.clip_id}: {vc.summary} "
                            f"(hook {vc.hook_score:.0f}/100)")
            seen_clips.add(c.clip_id)
            where = ("opens the video" if c.segment_index == 1 else
                     "closes the video" if c.segment_index == n else
                     f"segment {c.segment_index} of {n}")
            kind = {"vision_moment": "a moment Vision picked out",
                    "measured_window": "the busiest measured stretch",
                    "plan_segment": "the middle of this segment"}[c.source]
            rows.append("    [%d] %s, %s%s" % (
                c.index, where, kind, f": {c.what}" if c.what else ""))
        return PROMPT.format(brief=brief, story=stext,
                             candidates="\n".join(rows))

    # ------------------------------------------------------------ realise

    def _grab(self, ctx: AgentContext, span: TraceSpan,
              cand: ThumbnailCandidate, render: RenderResult, engine: dict,
              revision: int) -> dict:
        """Grab the frame with master_stage's own operation, then measure it.

        The process belongs to the media pool, so a deadline or a cancel
        kills it like any render stage. Nothing is reported as made until the
        file is on disk and ffprobe has read its dimensions.
        """
        from .tools.verify_tools import probe_final

        if not _engine_ok(engine):
            return {"ok": False, "why": "the pipeline reports no frame-grab "
                                        "thumbnail operation to use"}
        video = Path(render.video_path)
        if not video.exists():
            return {"ok": False, "why": f"the delivered video {video.name} "
                                        f"is not on disk"}
        out = Path(ctx.job.build_dir) / f"thumbnail_agent_r{revision:02d}.jpg"
        out.unlink(missing_ok=True)       # never mistake an old file for new
        cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-y",
               "-ss", f"{cand.frame_at:.2f}", "-i", str(video),
               "-frames:v", "1", str(out)]
        span.tool("thumbnail.frame_grab", frame_at=cand.frame_at,
                  video=video.name, image=out.name)
        try:
            rc, log = ctx.pool.run_process(
                cmd, timeout_s=GRAB_TIMEOUT_S, job_id=ctx.job.job_id,
                label="thumbnail:frame_grab")
        except Exception as exc:                      # noqa: BLE001
            return {"ok": False, "why": f"the frame grab did not run "
                                        f"({type(exc).__name__}: {exc})"[:200]}
        if rc != 0 or not out.exists() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            return {"ok": False, "why": f"ffmpeg exited {rc} without an "
                                        f"image: {(log or '')[-160:]}"}
        info = probe_final(out)
        if not (info.get("width") and info.get("height")):
            out.unlink(missing_ok=True)
            return {"ok": False, "why": "the grabbed file is not a readable "
                                        "image"}
        span.tool("verify.probe_image", width=info["width"],
                  height=info["height"], size=out.stat().st_size)
        return {"ok": True, "path": str(out), "width": int(info["width"]),
                "height": int(info["height"]), "size": out.stat().st_size,
                "why": ""}

    # ------------------------------------------------------------ payload

    def _payload(self, b: dict, chosen: int, rationale: str, raw: dict,
                 made: dict, live: bool, model, fallback) -> Thumbnail:
        unsupported = _texts(raw.get("unsupported_requests"), each=160)
        timing = 0
        for k in TREATMENT_KEYS:
            v = _text(raw.get(k), 80)
            if v and v.lower() not in ("none", "no", "n/a"):
                unsupported.append(f"{k}: '{v}' - the pipeline grabs a frame "
                                   f"and does nothing else to it")
        for k in TIMING_KEYS:
            if raw.get(k) is not None:
                timing += 1

        limitations = _texts(raw.get("limitations"), each=200)
        if not b["v_ok"]:
            limitations.append("no semantic reading of the footage was "
                               "available, so no frame was offered for what "
                               "it shows - only for where it sits")
        if not b["offsets_measured"]:
            limitations.append("the renderer's recorded transition offsets "
                               "did not describe this plan, so each frame's "
                               "position was computed from the plan instead")
        if b["unrealisable"]:
            limitations.append(f"{b['unrealisable']} upstream reference(s) "
                               f"fall outside the final cut and could not be "
                               f"offered")
        limitations.append("a frame of the delivered video carries whatever "
                           "was burnt into it, including that segment's "
                           "caption")
        if not made["ok"]:
            limitations.append("no image was generated: " + made["why"])
        if unsupported:
            limitations.append("%d request(s) fall outside what this "
                               "pipeline can do to a frame and will not "
                               "happen" % len(unsupported))
        if timing:
            limitations.append(f"{timing} field(s) in the reply named a "
                               f"time, clip or path, which was ignored - the "
                               f"frame's position comes from the list")
        limitations += b.get("notes", [])

        confidence = _num(raw.get("confidence"), 0.0, 1.0) if live else 0.0
        if live and not b["v_ok"]:
            confidence = min(confidence, 0.5)

        return Thumbnail(
            domain=b["domain"],
            thumbnail_type=("reasoned" if live else
                            "degraded" if b["candidates"] else "minimal"),
            candidates=b["candidates"], chosen=chosen,
            rationale=rationale, unrealisable=b["unrealisable"],
            engine_operations=_engine_operations(b["engine"]),
            unsupported_requests=list(dict.fromkeys(unsupported))[:MAX_LIST],
            video_path=b["render"].video_path, verdict=b["verdict"],
            generated=made["ok"], image_path=made.get("path"),
            width=made.get("width"), height=made.get("height"),
            size=made.get("size", 0),
            vision_available=b["v_ok"], strategy_available=b["cs_ok"],
            story_available=b["st_ok"],
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=not live,
            confidence=confidence, live=live, model=model)

    # ---------------------------------------------------------- digests

    @staticmethod
    def _vision_digest(vision: VisionAnalysis | None):
        if vision is None:
            return None
        return sorted(
            (c.clip_id, c.summary, c.hook_score,
             tuple((m.at, m.what) for m in c.interesting_moments),
             (lambda w: (w.index, w.start, w.end) if w else None)(
                 c.best_window()))
            for c in vision.analysed_clips)

    @staticmethod
    def _strategy_digest(strategy: ContentStrategy | None):
        if strategy is None:
            return None
        return (strategy.central_topic, strategy.core_angle,
                strategy.audience, strategy.hook.clip_id,
                strategy.hook.description, strategy.insufficient_signal)

    @staticmethod
    def _story_digest(story: Story | None):
        if story is None:
            return None
        return (story.logline,
                [(b.index, b.role, b.clip_id, b.moment_ref, b.narrative)
                 for b in story.beats])

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Thumbnail, b: dict, made: dict,
                   fallback: str | None) -> list[Decision]:
        sources = ([s for s, ok in (("Vision", b["v_ok"]),
                                    ("ContentStrategy", b["cs_ok"]),
                                    ("Story", b["st_ok"])) if ok]
                   or ["neither"])
        out = [Decision(
            what="Evidence the thumbnail was chosen from",
            chose=" + ".join(sources),
            why=("recorded explicitly so a frame chosen without a reading of "
                 "the footage cannot later be read as though it had one"),
            decided_by="measurement", confidence=1.0)]

        e = b["engine"]
        out.append(Decision(
            what="What the pipeline can make",
            chose=("one frame of the delivered video, as a JPEG"
                   if p.engine_operations else
                   "nothing - no frame-grab operation found"),
            why=("read from extras.master_stage's own thumbnail call: a "
                 "single -frames:v 1 grab with no filter, crop or text, and "
                 "no image generation anywhere in the project"),
            decided_by="measurement", confidence=1.0))

        if e.get("rule_by_score") and e.get("scores_projected_zero"):
            out.append(Decision(
                what="What master_stage shows on its own",
                chose="the middle of segment 1, every time",
                why=("master_stage takes the middle of the highest-scoring "
                     "segment, but the agent path projects every segment "
                     "with score 0.0 - so its frame never depends on the "
                     "footage. That is the gap this agent fills."),
                decided_by="measurement", confidence=1.0))

        out.append(Decision(
            what="Frames that can be chosen",
            chose=(f"{len(p.candidates)} in the cut"
                   + (f", {p.unrealisable} reference(s) trimmed out"
                      if p.unrealisable else "")),
            why=("every Vision moment and measured window that falls inside "
                 "a kept segment clear of the crossfades, plus each "
                 "segment's middle; positions from the offsets the renderer "
                 "recorded" if b["offsets_measured"] else
                 "every Vision moment and measured window that falls inside "
                 "a kept segment, plus each segment's middle; positions "
                 "computed from the plan"),
            decided_by="measurement", confidence=1.0))

        c = p.choice
        out.append(Decision(
            what="Thumbnail frame",
            chose=(f"[{c.index}] {c.source} on {c.clip_id}, segment "
                   f"{c.segment_index}"
                   + (f" - {c.what}" if c.what else ""))[:250],
            why=((p.rationale or "no reason given")[:300] if p.live else
                 f"{fallback}; {p.rationale}"[:300]),
            decided_by="llm" if p.live else "fallback",
            confidence=round(p.confidence, 2) if p.live else 1.0))

        out.append(Decision(
            what="Thumbnail image",
            chose=(f"{Path(p.image_path).name} {p.width}x{p.height}, "
                   f"{p.size} bytes" if p.generated else
                   "not generated"),
            why=("grabbed from the delivered video through the media pool "
                 "and measured with ffprobe before being recorded"
                 if p.generated else
                 made["why"] + " - master_stage's own thumbnail remains the "
                               "one shown"),
            decided_by="measurement", confidence=1.0))

        if p.unsupported_requests:
            out.append(Decision(
                what="Asked for but not possible",
                chose=f"{len(p.unsupported_requests)} request(s)",
                why=("; ".join(p.unsupported_requests[:2])[:250]
                     + " - recorded as a gap rather than promised"),
                decided_by="policy", confidence=1.0))
        return out


__all__ = ["ThumbnailAgent"]
