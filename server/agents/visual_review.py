"""
Visual Reviewer - the Quality team's second pair of eyes.

WHY
---
The Critic measures the rendered file: frames, loudness, black frames and
how lively the first three seconds are. None of that says whether a shot
SHOWS anything. A video can pass every measurement and still be a run of
forearms, empty table and blur - which is what the finished videos looked
like before shot check (job 9e797275158b), and what can still happen when
every candidate moment of a clip is weak.

WHAT IT DOES
------------
After each render it takes three frames from every shot of the FINISHED
video - the pixels the viewer gets, captions and crop included - and asks
the vision model, in one request, to score each shot 0-10 for whether the
subject is clearly visible, and whether its caption matches what is on
screen.

A shot scoring below `WEAK_BELOW` is weak. The Critic turns weak shots into a
revision: the EditingDirector re-cuts them from a different moment of the
same footage, never the rejected one. That is the loop the manager's design
draws as "loop <= 3": a worker, a checker, and a bounded way back.

HONESTY
-------
Nothing is invented. With no vision model, a refusal or an unreadable answer,
the review says `reviewed=False`, the Critic judges as it always did, and the
run is never stopped by this agent. Off with `AT_VISUAL_REVIEW=0`.
"""

from __future__ import annotations

import base64
import os
import time

from .artifacts import (Artifact, Cost, Decision, EditPlan, RenderResult,
                        ShotReview, VisualReview)
from .base import Agent, AgentContext
from .trace import TraceSpan

WEAK_BELOW = 4.0
MAX_SHOTS = 12          # one request; beyond this the strips get too small

PROMPT = """You are reviewing a finished short video{about}, shot by shot. \
There are {n} images, one per shot, in order. Each image shows three moments \
of its shot side by side (start, middle, end), exactly as the viewer sees them.

For each shot:
- "score" 0-10: how clearly the shot shows the main subject (the thing being \
built, used or shown). High = subject clear, large enough, in focus. Low = \
mostly a hand or arm, an empty table, wall or floor, heavy blur, or the \
subject cut off.
- "caption_matches": true or false - does the caption describe what is on \
screen? Use null when the shot has no caption.
- "problem": a few words when the score is below 5, otherwise "".

The shots and their captions:
{shots}

Answer with JSON only: {{"shots": [{{"score": 0, "caption_matches": null, \
"problem": ""}}, ...]}} with exactly {n} entries, in order."""


def enabled() -> bool:
    return os.environ.get("AT_VISUAL_REVIEW", "1") != "0"


def timeline(plan: EditPlan, xfade: float) -> list[tuple[float, float]]:
    """(start, duration) of every segment in the FINISHED video.

    Crossfades overlap neighbours by `xfade`, the same arithmetic as the
    join stage (`select_tools.total_with_crossfades`)."""
    out, t = [], 0.0
    for i, s in enumerate(plan.segments):
        if i:
            t -= xfade
        out.append((max(0.0, t), s.duration))
        t += s.duration
    return out


def sample_times(at: float, dur: float, xfade: float) -> list[float]:
    """Three moments inside a shot, clear of the crossfades at both ends."""
    pad = min(xfade, dur / 4)
    a, b = at + pad, at + dur - pad
    return [round(a + (b - a) * k, 2) for k in (0.15, 0.5, 0.85)]


def parse(raw, n: int) -> list[dict] | None:
    """The model's answer as n clean rows, or None. Never raises."""
    rows = raw.get("shots") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) != n:
        return None
    out = []
    for r in rows:
        if isinstance(r, (int, float)):
            r = {"score": r}
        if not isinstance(r, dict):
            return None
        try:
            score = max(0.0, min(10.0, float(r.get("score"))))
        except (TypeError, ValueError):
            return None
        cm = r.get("caption_matches")
        out.append({"score": score,
                    "caption_ok": cm if isinstance(cm, bool) else None,
                    "problem": str(r.get("problem") or "")[:120]})
    return out


class VisualReviewAgent(Agent):
    id = "visual_review"
    produces = "VisualReview"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                render: RenderResult, plan: EditPlan, attempt: int,
                about: str = "", revision: int = 0
                ) -> Artifact[VisualReview]:
        from .tools import shot_check as SC

        started = time.time()
        inputs_hash = self.hash_inputs(render.video_path, attempt,
                                       [s.cache_key() for s in plan.segments])

        def finish(review: VisualReview, decisions, cost=None,
                   degraded=False, why=None):
            ctx.emit("visual review: " + review.summary)
            return self.wrap(
                ctx, review, revision=revision, inputs_hash=inputs_hash,
                span=span, decisions=decisions,
                confidence=0.7 if review.reviewed else 0.0,
                degraded=degraded, fallback_used=why,
                cost=cost or Cost(wall_s=round(time.time() - started, 2)))

        def not_reviewed(why: str):
            span.note("visual review skipped: " + why)
            return finish(
                VisualReview(attempt=attempt, reviewed=False,
                             weak_below=WEAK_BELOW,
                             summary=f"not reviewed: {why}"),
                [Decision(what="Visual review", chose="not reviewed",
                          why=why, decided_by="fallback", confidence=1.0)],
                degraded=True, why=why)

        if not enabled():
            return not_reviewed("switched off (AT_VISUAL_REVIEW=0)")
        if not ctx.llm.available():
            return not_reviewed("no vision model is configured")
        segs = list(plan.segments)
        if not segs:
            return not_reviewed("the plan has no shots")
        if len(segs) > MAX_SHOTS:
            span.note(f"{len(segs)} shots; the first {MAX_SHOTS} are reviewed")
            segs = segs[:MAX_SHOTS]

        xfade = float(ctx.job.settings.transition_duration or 0.0)
        places = timeline(plan, xfade)[:len(segs)]
        canvas = tuple(plan.canvas)
        frames, kept = [], []
        for s, (at, dur) in zip(segs, places):
            span.tool("review.strip", shot=s.index, at=round(at, 2))
            f = SC.strip_jpeg(render.video_path, sample_times(at, dur, xfade),
                              canvas, True)
            if f:
                frames.append(f)
                kept.append((s, at))
        if len(frames) < 1:
            return not_reviewed("no frames could be read from the video")

        listing = "\n".join(
            f"{k + 1}. caption: " + (f'"{s.caption}"' if s.caption
                                     else "(none)")
            + (f"; meant to show: {s.reason[:120]}" if s.reason else "")
            for k, (s, _) in enumerate(kept))
        prompt = PROMPT.format(
            n=len(frames), shots=listing,
            about=f" about: {about.strip()[:200]}" if about.strip() else "")
        try:
            raw, reply = ctx.llm.complete_json(
                prompt, max_tokens=900,
                images=[base64.b64encode(f).decode() for f in frames])
        except Exception as exc:                             # noqa: BLE001
            return not_reviewed(f"the vision model failed "
                                f"({type(exc).__name__})")
        cost = Cost(wall_s=round(time.time() - started, 2))
        if reply is not None:
            span.llm(reply.tokens_in, reply.tokens_out, reply.usd)
            cost = Cost(tokens_in=reply.tokens_in,
                        tokens_out=reply.tokens_out, usd=reply.usd,
                        wall_s=cost.wall_s)
        rows = parse(raw, len(frames))
        if rows is None:
            return not_reviewed("the vision model's answer could not be read")

        shots = [ShotReview(index=s.index, clip_id=s.clip_id, start=s.start,
                            end=s.end, at=round(at, 2), score=r["score"],
                            caption=s.caption,
                            caption_ok=r["caption_ok"] if s.caption else None,
                            problem=r["problem"])
                 for (s, at), r in zip(kept, rows)]
        weak = [x.index for x in shots if x.score < WEAK_BELOW]
        mean = round(sum(x.score for x in shots) / len(shots), 1)
        off_caption = [x.index for x in shots if x.caption_ok is False]
        summary = (f"{len(shots)} shot(s) looked at, average {mean}/10; "
                   + (f"weak: {', '.join(f'#{i}' for i in weak)}"
                      if weak else "no weak shots")
                   + (f"; caption off on {', '.join(f'#{i}' for i in off_caption)}"
                      if off_caption else ""))
        for x in shots:
            span.tool("review.shot", shot=x.index, score=x.score,
                      caption_ok=x.caption_ok, problem=x.problem)
        decisions = [Decision(
            what=f"Shot {x.index} ({x.clip_id})",
            chose=f"{x.score:.0f}/10" + (" - weak" if x.index in weak else ""),
            why=x.problem or "the subject is visible",
            decided_by="llm", confidence=0.7) for x in shots]
        return finish(
            VisualReview(attempt=attempt, reviewed=True, shots=shots,
                         weak=weak, weak_below=WEAK_BELOW, mean_score=mean,
                         summary=summary),
            decisions, cost=cost)


__all__ = ["VisualReviewAgent", "WEAK_BELOW", "enabled", "parse",
           "sample_times", "timeline"]
