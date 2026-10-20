"""Choosing a shot by looking at it.

WHY
---
The window search picks, for each shot, the stretch of a clip with the most
movement. In a hands-on build the most movement is usually a hand or an arm
passing in front of the lens, so finished videos were full of forearms,
empty table and wall (jobs 9e797275158b and b5e72b2c68a4, 17-18 Oct 2026).
The vision model saw four frames per clip at planning time, far too few to
say which seconds of a 35-minute clip are worth showing.

WHAT THIS DOES
--------------
For each shot the editing director asked for, the best few windows by the
same motion score become candidates (`select_tools.top_windows`). One frame
from the middle of each is taken - cropped the way the finished video will
show it, so a horizontal clip going into a vertical video is judged as the
slice the viewer will actually see - and the vision model scores them all in
one request: is the subject clearly visible, or is it a hand, an empty
table, a blur? The best-scoring candidate is used.

Nothing here invents a moment: every candidate is a window the pipeline's
own search produced, and the model only ranks them. When anything fails -
no vision model, a refusal, an unreadable reply, a frame that cannot be
read - the shot falls back to the busiest window, exactly as before.

COST
----
One vision request per shot with six small strips of three frames, so six shots
are six requests: tens of seconds and a few thousand tokens per video.
Off with `AT_SHOT_CHECK=0`.
"""

from __future__ import annotations

import base64
import os
import subprocess

from . import select_tools as ST

CANDIDATES = 6
LONG_SIDE = 384

PROMPT = """These {n} images are {n} candidate shots for one moment of a \
short video{about}. Each image shows three moments of its shot side by side - \
start, middle and end. This shot should show: {purpose}

Score every image from 0 to 10 for how good the WHOLE shot is; a shot whose \
subject is covered in any of its three moments scores low:
- high: the main subject (the object being built, used or shown) is clearly \
visible, large enough to read and in focus;
- low: the frame is mostly a hand, an arm or another body part, an empty \
table, wall or floor, heavily blurred, or the subject is cut off.

Answer with JSON only: {{"scores": [one number per image, in the order \
given]}}"""


def enabled() -> bool:
    return os.environ.get("AT_SHOT_CHECK", "1") != "0"


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def frame_jpeg(path, t: float, canvas: tuple[int, int] | None = None,
               fill: bool = False, long_side: int = LONG_SIDE,
               timeout: float = 30.0) -> bytes | None:
    """One frame at `t` seconds as a small JPEG, or None. Seeks before
    opening the file, so a frame 70 minutes in costs a fraction of a
    second rather than decoding everything before it."""
    if canvas and fill:
        cw, ch = canvas
        k = long_side / max(cw, ch)
        w, h = _even(cw * k), _even(ch * k)
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    else:
        vf = (f"scale='if(gt(iw,ih),{long_side},-2)':"
              f"'if(gt(iw,ih),-2,{long_side})'")
    cmd = [os.environ.get("AT_FFMPEG") or "ffmpeg", "-nostdin", "-v", "error",
           "-ss", f"{max(0.0, t):.2f}", "-i", str(path), "-frames:v", "1",
           "-vf", vf, "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    data = out.stdout
    return data if out.returncode == 0 and len(data) > 500 else None


def strip_jpeg(path, times: list[float],
               canvas: tuple[int, int] | None = None, fill: bool = False,
               tile: int = 256, timeout: float = 45.0) -> bytes | None:
    """Several moments of one shot side by side, as one small JPEG. One
    frame from the middle was judged at first; a hand that crossed the
    lens for the rest of the shot was then invisible to the judge
    (job 514798fec522, shot 2)."""
    if not times:
        return None
    if canvas and fill:
        cw, ch = canvas
        k = tile / max(cw, ch)
        w, h = _even(cw * k), _even(ch * k)
        each = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    else:
        each = f"scale=-2:{_even(tile * 9 / 16)}"
    cmd = [os.environ.get("AT_FFMPEG") or "ffmpeg", "-nostdin", "-v", "error"]
    for t in times:
        cmd += ["-ss", f"{max(0.0, t):.2f}", "-i", str(path)]
    chains = [f"[{i}:v]{each},setsar=1[f{i}]" for i in range(len(times))]
    joined = "".join(f"[f{i}]" for i in range(len(times)))
    graph = ";".join(chains) + (f";{joined}hstack=inputs={len(times)}[out]"
                                if len(times) > 1 else "")
    out_label = "[out]" if len(times) > 1 else "[f0]"
    cmd += ["-filter_complex", graph, "-map", out_label, "-frames:v", "1",
            "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    data = out.stdout
    return data if out.returncode == 0 and len(data) > 500 else None


def _scores(parsed, n: int) -> list[float] | None:
    vals = parsed.get("scores") if isinstance(parsed, dict) else parsed
    if not isinstance(vals, list) or len(vals) != n:
        return None
    out = []
    for v in vals:
        try:
            out.append(max(0.0, min(10.0, float(v))))
        except (TypeError, ValueError):
            return None
    return out


def rank(llm, frames: list[bytes], purpose: str, about: str = ""):
    """Scores for `frames`, in order, or (None, reply) when there is
    nothing usable. Never raises."""
    if len(frames) < 2:
        return None, None
    prompt = PROMPT.format(
        n=len(frames), purpose=(purpose or "the main subject").strip()[:300],
        about=f" about: {about.strip()[:200]}" if about.strip() else "")
    try:
        parsed, reply = llm.complete_json(
            prompt, max_tokens=300,
            images=[base64.b64encode(f).decode() for f in frames])
    except Exception:                                       # noqa: BLE001
        return None, None
    return _scores(parsed, len(frames)), reply


def choose(llm, span, clip, wanted: float, taken, purpose: str,
           canvas: tuple[int, int], framing: str, about: str = ""):
    """`(start, end, score, notes)` for the best-looking candidate, or None
    to fall back to the busiest window."""
    length, notes = ST.clamp_length(clip, wanted)
    cands = ST.top_windows(clip, length, taken or None, k=CANDIDATES,
                           spread=clip.technical.duration / (2 * CANDIDATES))
    if len(cands) < 2:
        return None
    vertical_canvas = canvas[1] > canvas[0]
    fill = (framing == "fill" and
            (clip.technical.orientation == "vertical") != vertical_canvas)
    frames, kept = [], []
    for i, (a, b, _s) in enumerate(cands):
        span_s = b - a
        f = strip_jpeg(clip.path, [a + 0.15 * span_s, a + 0.5 * span_s,
                                   a + 0.85 * span_s], canvas, fill)
        if f:
            frames.append(f)
            kept.append(i)
    scores, reply = rank(llm, frames, purpose, about)
    if reply is not None:
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)
    if not scores:
        span.note(f"shot check skipped for {clip.clip_id}: no usable answer; "
                  f"the busiest window was used")
        return None
    best = max(range(len(scores)), key=lambda j: (scores[j], -j))
    a, b, s = cands[kept[best]]
    span.tool("shot.check", clip=clip.clip_id, candidates=len(frames),
              chose=kept[best] + 1, score=scores[best],
              busiest_score=scores[0])
    notes.append(f"{clip.clip_id}: looked at {len(frames)} candidate shots and "
                 f"chose #{kept[best] + 1} ({scores[best]:.0f}/10 for showing "
                 f"the subject; the busiest one scored {scores[0]:.0f}/10)")
    if scores[best] < 3:
        notes.append(f"{clip.clip_id}: no candidate showed the subject clearly")
    return a, b, s, notes
