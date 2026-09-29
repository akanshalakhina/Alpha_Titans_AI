"""
The vision layer: a real vision-capable model looks at the footage.

WHAT THIS IS
------------
Unlike director.py (rule-based, measures motion), this module extracts real
frames from each clip and sends them to a real vision model over HTTP, then
feeds the structured result into the Director so the edit plan is informed by
what is actually in shot rather than only by how much the pixels move.

There are NO canned responses and no hardcoded scene descriptions anywhere in
this file. If no API key is configured the module reports itself unavailable
and the pipeline carries on with the rule-based Director exactly as before -
that fallback is the normal, supported path, not an error.

CONFIGURATION (all via environment / .env)
------------------------------------------
    AT_VISION_API_KEY    required to enable it at all; unset = disabled
    AT_VISION_PROVIDER   anthropic (default) | openai
    AT_VISION_MODEL      model id; provider-appropriate default if unset
    AT_VISION_BASE_URL   override the endpoint (self-hosted, gateway, proxy)
    AT_VISION_FRAMES     frames sampled per clip (default 4, max 8)
    AT_VISION_TIMEOUT    seconds per request (default 90)
    AT_VISION_MAX_CLIPS  safety cap on clips analysed per build (default 12)

HOW IT CALLS
------------
Two passes, which is both cheaper and more accurate than one:
  1. per clip - frames in, structured description out
  2. across clips - the text summaries only, no images, to spot duplicates
     and pick a hook and a story order

Everything the model returns is treated as untrusted data: it is parsed
defensively, range-checked, and clamped to clip names we actually uploaded. A
model that returns nonsense degrades to the rule-based path rather than
corrupting the edit.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import config
from adapter import build_dir, read_manifest, src_dir

DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o",
}
DEFAULT_BASE = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}

SHOTS = ("close-up", "medium", "wide", "unclear")
KINDS = ("assembly", "demonstration", "reveal", "detail", "other")
PHASES = ("beginning", "middle", "end", "unclear")


# ------------------------------------------------------------------ config

def _env(key: str, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def provider() -> str:
    p = (_env("AT_VISION_PROVIDER", "anthropic") or "").lower()
    return p if p in DEFAULT_MODEL else "anthropic"


def model() -> str:
    return _env("AT_VISION_MODEL", DEFAULT_MODEL[provider()])


def base_url() -> str:
    return (_env("AT_VISION_BASE_URL", DEFAULT_BASE[provider()]) or "").rstrip("/")


def available() -> bool:
    """True only when a key is configured. No key is a supported state."""
    return bool(_env("AT_VISION_API_KEY"))


def describe() -> dict:
    return {
        "enabled": available(),
        "provider": provider() if available() else None,
        "model": model() if available() else None,
        "base_url": base_url() if available() else None,
        "frames_per_clip": _frames(),
        "why_disabled": None if available() else
            "AT_VISION_API_KEY is not set, so the rule-based Director runs alone.",
    }


def _frames() -> int:
    try:
        return max(1, min(8, int(_env("AT_VISION_FRAMES", 4))))
    except (TypeError, ValueError):
        return 4


def _timeout() -> float:
    try:
        return max(10.0, float(_env("AT_VISION_TIMEOUT", 90)))
    except (TypeError, ValueError):
        return 90.0


# ------------------------------------------------------------------ frames

def sample_frames(path: Path, duration: float, n: int) -> list[str]:
    """n JPEG frames spread across the clip, base64'd. Real decode, no stubs."""
    out = []
    for i in range(n):
        at = duration * (i + 0.5) / n
        p = subprocess.run(
            [config.FFMPEG, "-nostdin", "-v", "error", "-ss", f"{at:.2f}",
             "-i", str(path), "-frames:v", "1",
             "-vf", "scale='min(768,iw)':-2", "-f", "image2", "-c:v", "mjpeg",
             "-q:v", "6", "pipe:1"],
            capture_output=True)
        if p.returncode == 0 and len(p.stdout) > 500:
            out.append(base64.b64encode(p.stdout).decode("ascii"))
    return out


# -------------------------------------------------------------- transport

def _post(path: str, payload: dict) -> dict:
    key = _env("AT_VISION_API_KEY")
    url = base_url() + path
    headers = {"Content-Type": "application/json"}
    if provider() == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = _env("AT_VISION_API_VERSION", "2023-06-01")
    else:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=_timeout()) as r:
        return json.loads(r.read())


def _ask(prompt: str, images: list[str], max_tokens: int = 1400) -> str:
    """One model call. Returns the raw text the model produced."""
    if provider() == "anthropic":
        content = [{"type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": b}} for b in images]
        content.append({"type": "text", "text": prompt})
        data = _post("/v1/messages", {
            "model": model(), "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        })
        parts = data.get("content") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    # openai-compatible
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
               for b in images]
    content.append({"type": "text", "text": prompt})
    data = _post("/v1/chat/completions", {
        "model": model(), "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    })
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


def _json_from(text: str):
    """Pull the first JSON object/array out of a model reply. Models wrap
    answers in prose and fences however they like; this does not trust the
    shape of the reply, only that a JSON value is in there somewhere."""
    if not text:
        return None
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M)
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        if i < 0:
            continue
        depth = 0
        for k in range(i, len(t)):
            if t[k] == opener:
                depth += 1
            elif t[k] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:k + 1])
                    except json.JSONDecodeError:
                        break
    return None


# --------------------------------------------------------------- prompting

CLIP_PROMPT = """These frames are sampled in order from ONE video clip of a LEGO build.

Describe only what you can actually see. If you cannot tell, say so - an
"unclear" answer is far more useful to me than a confident guess.

Reply with ONLY a JSON object, no prose around it:

{{
  "subject": "<what is being built/shown, in a few words, or 'unclear'>",
  "shot": "<close-up|medium|wide|unclear>",
  "kind": "<assembly|demonstration|reveal|detail|other>",
  "phase": "<beginning|middle|end|unclear - where in a build this sits>",
  "interesting_moments": [
    {{"at": <seconds from clip start, 0-{dur:.1f}>, "what": "<what happens>"}}
  ],
  "hook_score": <0-100, how well this would open a short video>,
  "caption": "<a short on-screen caption, or null if you are not confident>",
  "confidence": <0-100, how sure you are about `subject`>,
  "notes": "<anything else worth knowing, or empty>"
}}

The clip is {dur:.1f}s long and {n} frames are shown, evenly spaced."""

CROSS_PROMPT = """Here are descriptions of {n} clips from one LEGO shoot, each
already analysed. Decide how they fit together as ONE short video.

{summaries}

Reply with ONLY a JSON object:

{{
  "duplicates": [["<clip>", "<clip that covers the same thing>"]],
  "hook": "<the clip name that should open, or null>",
  "order": ["<clip>", "..."],
  "story": "<one sentence describing the sequence you chose>"
}}

Use ONLY these exact clip names: {names}
If two clips genuinely show different things, do NOT call them duplicates.
If no clip is a clear opener, set "hook" to null."""


# ---------------------------------------------------------------- analysis

def _clean_clip(raw, dur: float) -> dict:
    """Range-check and clamp one clip result. The model is untrusted input."""
    if not isinstance(raw, dict):
        return {}
    def s(key, allowed, fallback):
        v = str(raw.get(key) or "").strip().lower()
        return v if v in allowed else fallback
    def num(key, lo, hi, fallback=None):
        try:
            return max(lo, min(hi, float(raw.get(key))))
        except (TypeError, ValueError):
            return fallback
    moments = []
    for mm in (raw.get("interesting_moments") or [])[:6]:
        if not isinstance(mm, dict):
            continue
        try:
            at = max(0.0, min(float(mm.get("at")), dur))
        except (TypeError, ValueError):
            continue
        moments.append({"at": round(at, 2), "what": str(mm.get("what") or "")[:120]})
    cap = raw.get("caption")
    cap = str(cap).strip()[:58] if isinstance(cap, str) and cap.strip() else None
    return {
        "subject": str(raw.get("subject") or "unclear")[:80],
        "shot": s("shot", SHOTS, "unclear"),
        "kind": s("kind", KINDS, "other"),
        "phase": s("phase", PHASES, "unclear"),
        "interesting_moments": moments,
        "hook_score": num("hook_score", 0, 100, 0),
        "caption": cap,
        "confidence": num("confidence", 0, 100, 0),
        "notes": str(raw.get("notes") or "")[:200],
    }


def analyse(job_id: str, settings: dict, on_line) -> dict | None:
    """Look at every clip. Returns the analysis, or None if unavailable."""
    if not available():
        on_line("no AT_VISION_API_KEY configured - skipping the vision pass; "
                "the rule-based Director will plan this build on its own")
        return None

    m = read_manifest(job_id)
    if not m:
        raise RuntimeError("no manifest - probe has not run")
    clips = m.get("clips") or []
    try:
        cap = max(1, int(_env("AT_VISION_MAX_CLIPS", 12)))
    except (TypeError, ValueError):
        cap = 12
    clips = clips[:cap]

    on_line("vision: %s / %s, %d frames per clip, %d clip(s)"
            % (provider(), model(), _frames(), len(clips)))

    per_clip, failures = {}, []
    for c in clips:
        name, dur = c["name"], float(c.get("duration") or 0)
        path = src_dir(job_id) / name
        frames = sample_frames(path, dur, _frames())
        if not frames:
            failures.append(f"{name}: could not sample frames")
            continue
        try:
            txt = _ask(CLIP_PROMPT.format(dur=dur, n=len(frames)), frames)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, TimeoutError) as exc:
            failures.append(f"{name}: {type(exc).__name__} {exc}")
            on_line("  %s -> call failed (%s)" % (name, type(exc).__name__))
            continue
        got = _clean_clip(_json_from(txt), dur)
        if not got:
            failures.append(f"{name}: reply was not usable JSON")
            on_line("  %s -> reply was not usable JSON" % name)
            continue
        per_clip[name] = got
        on_line("  %-22s %s / %s / %s  hook %d  conf %d  %s"
                % (name, got["kind"], got["shot"], got["phase"],
                   got["hook_score"], got["confidence"],
                   (got["subject"] or "")[:34]))

    if not per_clip:
        on_line("vision produced nothing usable - falling back to the "
                "rule-based Director")
        return None

    # ---- pass 2: relationships, from the summaries only (no images) --------
    cross = {}
    names = list(per_clip)
    if len(names) > 1:
        summaries = "\n".join(
            "- %s: %s | %s | %s shot | phase %s | hook %d | conf %d | %s"
            % (n, per_clip[n]["subject"], per_clip[n]["kind"],
               per_clip[n]["shot"], per_clip[n]["phase"],
               per_clip[n]["hook_score"], per_clip[n]["confidence"],
               per_clip[n]["notes"])
            for n in names)
        try:
            txt = _ask(CROSS_PROMPT.format(n=len(names), summaries=summaries,
                                           names=", ".join(names)), [], 900)
            raw = _json_from(txt) or {}
            # clamp every name the model returns to one we actually have
            dups = []
            for pair in (raw.get("duplicates") or [])[:8]:
                if (isinstance(pair, (list, tuple)) and len(pair) >= 2
                        and pair[0] in per_clip and pair[1] in per_clip
                        and pair[0] != pair[1]):
                    dups.append([pair[0], pair[1]])
            hook = raw.get("hook")
            order = [n for n in (raw.get("order") or []) if n in per_clip]
            cross = {
                "duplicates": dups,
                "hook": hook if hook in per_clip else None,
                "order": order if len(order) == len(names) else [],
                "story": str(raw.get("story") or "")[:200],
            }
            if cross["duplicates"]:
                on_line("  duplicates: " + "; ".join(
                    f"{a} ~ {b}" for a, b in cross["duplicates"]))
            if cross["hook"]:
                on_line("  hook: %s" % cross["hook"])
            if cross["story"]:
                on_line("  story: %s" % cross["story"])
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, TimeoutError) as exc:
            failures.append(f"cross-clip pass: {type(exc).__name__} {exc}")
            on_line("  cross-clip pass failed (%s) - per-clip analysis kept"
                    % type(exc).__name__)

    result = {
        "provider": provider(), "model": model(),
        "frames_per_clip": _frames(),
        "clips": per_clip, "cross": cross, "failures": failures,
    }
    with open(build_dir(job_id) / "vision.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def read(job_id: str) -> dict | None:
    p = build_dir(job_id) / "vision.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
