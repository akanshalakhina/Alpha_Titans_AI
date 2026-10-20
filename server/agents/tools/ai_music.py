"""AI-composed music: Google's Lyria 3, directed by the Music agent.

WHO DECIDES WHAT
----------------
The Music agent (an LLM) decides what the score should be: a mood, a tempo
feeling, the instruments, a sonic direction and an energy for every story
beat. Until now that direction was turned into sound by `bgm_stdlib`, a
small synthesiser written in code. Here it is handed instead to Lyria 3, a
music-generation model on Google Cloud, which composes and performs a real
instrumental track from it.

Nothing about the timing is left to either model. The cut already exists
when this runs, so the prompt carries its real length, a tempo fitted to its
real transition marks by the existing beat engine, and one timestamped
section per part of the agent's arc, snapped to those marks. Lyria does not
always honour the length it is asked for (a 30-second request came back at
61 seconds when this was measured), so the track still goes through
`extras.prepare_track`, which loops, trims and fades it to the cut exactly
as it does an operator's own music.

WHEN IT RUNS
------------
Only with `AT_MUSIC_PROVIDER=lyria`, a project to bill (`AT_MUSIC_PROJECT`,
else `AT_GCP_PROJECT`, else the VM's own) and an identity: the VM's service
account from the metadata server, or `gcloud` on a workstation. Anything
missing, refused or failed returns `(None, plan)` with the reason in the
plan, and the caller falls back to the code-composed score. It can make the
music better; it cannot make a build silent.

PROMPTS THAT GET REFUSED
------------------------
Lyria refuses some requests with `content_blocked` and no reason, and the
refusals are NOT predictable from the words. Measured on 18 Oct 2026: 2 of
8 requests were refused; the exact prompt refused that morning was accepted
when resent, and two near-identical prompts went opposite ways. So a refusal
is simply asked again: first with a PLAIN prompt built only from closed
vocabularies (mood, tempo, energy, length - no free text from any model, in
case the words were the problem after all), then once more. At about one
refusal in four, three tries leave roughly one video in sixty on the
code-composed score.

COST
----
Lyria 3 Pro was listed at $0.08 per track on 18 Oct 2026. A track is cached
by the hash of its prompt, so a retry or resume of the same cut does not
pay twice; a revision that changes the cut is a new prompt and a new track.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..artifacts import Music

DEFAULT_MODEL = "lyria-3-pro-preview"
DEFAULT_LOCATION = "global"
URL = ("https://aiplatform.googleapis.com/v1beta1/projects/{project}"
       "/locations/{location}/interactions")
METADATA = "http://metadata.google.internal/computeMetadata/v1/"
MAX_SECONDS = 184.0                  # Lyria 3 Pro's longest track
# Asked for slightly MORE than the cut. Measured: 120 s asked came back at
# 118.1 s. `prepare_track` loops a track that is short, which would restart
# the song for the last two seconds; one that runs over is trimmed under a
# fade instead. (Short requests came back near 60 s regardless.)
PAD_SECONDS = 4.0

# How each of the agent's energies reads as Lyria's 1-10 intensity scale.
INTENSITY = {"low": 2, "falling": 3, "steady": 4, "building": 5,
             "high": 7, "peak": 8}

# The closed vocabulary the plain prompt is built from. Each mood is the
# engine's own (`artifacts.Mood`); none of these words came from a model.
GENRE = {
    "lofi": "warm lo-fi hip-hop with soft keys and a relaxed beat",
    "calm": "calm ambient music with gentle piano and soft pads",
    "bright": "bright acoustic pop with plucked guitar and light percussion",
    "upbeat": "upbeat feel-good pop with claps and a bouncy bass line",
    "beat": "steady beat-driven electronic music with a clean groove",
    "energetic": "energetic electronic rock with driving drums",
    "cinematic": "cinematic orchestral music with strings and soft brass",
}
_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MINOR = {"lofi", "calm", "cinematic"}

_TEXT_CAP = 220                     # one free-text field, at most


def enabled() -> bool:
    return (os.environ.get("AT_MUSIC_PROVIDER") or "").strip().lower() == "lyria"


# --------------------------------------------------------------- the prompt

def _clean(text: str, cap: int = _TEXT_CAP) -> str:
    """One line of the agent's words, safe to put inside a prompt."""
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'")
    return text[:cap].rstrip(" ,;:")


def _clock(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def section_spans(n: int, duration: float,
                  marks: list[float]) -> list[tuple[float, float]]:
    """`n` consecutive spans covering the cut, each boundary snapped to the
    nearest real transition mark when there is one to snap to, so a change
    in the music lands on a change of shot."""
    n = max(1, n)
    duration = max(0.1, float(duration))
    bounds = [0.0]
    pool = sorted(m for m in marks if 0.0 < m < duration)
    for k in range(1, n):
        want = duration * k / n
        if pool:
            near = min(pool, key=lambda m: abs(m - want))
            if near > bounds[-1] + 0.5:
                bounds.append(near)
                pool.remove(near)
                continue
        if want > bounds[-1] + 0.5:
            bounds.append(want)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def key_name(root_midi: int, mood: str) -> str:
    return f"{_NOTE[int(root_midi) % 12]} {'minor' if mood in _MINOR else 'major'}"


def build_prompt(music: Music, duration: float, marks: list[float],
                 bpm: float, root_midi: int, plain: bool = False) -> str:
    """The request, from the agent's direction and the measured cut.

    `plain=True` uses only closed vocabularies - the retry after a refusal.
    """
    secs = int(round(min(duration + PAD_SECONDS, MAX_SECONDS)))
    mood = music.mood or "lofi"
    lines = [f"Create a {secs}-second instrumental track at {int(round(bpm))} "
             f"BPM in {key_name(root_midi, mood)}.",
             f"Style: {GENRE.get(mood, GENRE['lofi'])}."]
    if not plain:
        if music.sonic_direction:
            lines.append(f"Sound: {_clean(music.sonic_direction)}.")
        if music.instrumentation:
            inst = ", ".join(_clean(i, 40) for i in music.instrumentation[:6]
                             if _clean(i, 40))
            if inst:
                lines.append(f"Instruments: {inst}.")
        if music.energy_arc:
            lines.append(f"Shape: {_clean(music.energy_arc)}.")
    sections = list(music.sections or [])
    spans = section_spans(len(sections) or 1, duration, marks)
    spans[-1] = (spans[-1][0], float(secs))   # the pad belongs to the ending
    for i, (a, b) in enumerate(spans):
        s = sections[i] if i < len(sections) else None
        energy = s.energy if s else "steady"
        level = INTENSITY.get(energy, 4)
        what = (_clean(s.intent, 120) if (s and s.intent and not plain)
                else f"{energy} section")
        lines.append(f"[{_clock(a)} - {_clock(b)}] {what}. "
                     f"Intensity: {level}/10.")
    lines.append("Background music for a video: no vocals, no lyrics, no "
                 "spoken words. Instrumental.")
    return "\n".join(lines)


def fitted_bpm(music: Music, job_id: str, marks: list[float]) -> tuple[float, int]:
    """A tempo whose beats land on the cut's transitions, and the key - from
    the same engine and the same mood tables the code-composed score uses,
    so the two paths agree about what this direction means."""
    from . import score_tools as ST
    plan = ST.settings_for(music, job_id, marks)
    bpm = (plan["bpm_lo"] + plan["bpm_hi"]) / 2
    eng = ST._engine()
    if eng is not None and marks:
        try:
            bpm, _err = eng.best_bpm(marks, plan["bpm_lo"], plan["bpm_hi"])
        except Exception:                                  # noqa: BLE001
            pass
    return round(float(bpm), 1), int(plan["root_midi"])


# ---------------------------------------------------------------- identity

def _metadata(path: str, http) -> str | None:
    try:
        req = urllib.request.Request(METADATA + path,
                                     headers={"Metadata-Flavor": "Google"})
        with http(req, timeout=2) as r:
            return r.read().decode("utf-8")
    except Exception:                                      # noqa: BLE001
        return None


def access_token(http=urllib.request.urlopen) -> str | None:
    """The VM's own service account, else `gcloud` on a workstation. Never
    a key file, never logged."""
    raw = _metadata("instance/service-accounts/default/token", http)
    if raw:
        try:
            return json.loads(raw)["access_token"]
        except (ValueError, KeyError):
            pass
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if exe:
        try:
            out = subprocess.run([exe, "auth", "print-access-token"],
                                 capture_output=True, text=True, timeout=30)
            tok = out.stdout.strip()
            if out.returncode == 0 and tok:
                return tok
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def project_id(http=urllib.request.urlopen) -> str | None:
    p = (os.environ.get("AT_MUSIC_PROJECT") or os.environ.get("AT_GCP_PROJECT")
         or _metadata("project/project-id", http))
    p = (p or "").strip()
    return p if re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", p) else None


# ---------------------------------------------------------------- the call

class Refused(Exception):
    """Lyria answered, and the answer was no."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status, self.code = status, code


def request_track(prompt: str, token: str, project: str,
                  model: str = DEFAULT_MODEL, location: str = DEFAULT_LOCATION,
                  timeout: float = 240.0,
                  http=urllib.request.urlopen,
                  images: list[bytes] | None = None) -> tuple[bytes, str, str]:
    """One Lyria call -> (audio bytes, mime type, the model's description).
    `images` are JPEG frames of the cut, sent beside the words so the music
    is written to what the video shows (measured working, 19 Oct 2026)."""
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for img in images or []:
        parts.append({"type": "image", "mime_type": "image/jpeg",
                      "data": base64.b64encode(img).decode()})
    body = json.dumps({"model": model, "input": parts}).encode()
    req = urllib.request.Request(
        URL.format(project=project, location=location), data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json; charset=utf-8"})
    try:
        with http(req, timeout=timeout) as r:
            reply = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8")).get("error") or {}
        except (ValueError, AttributeError):
            err = {}
        raise Refused(e.code, str(err.get("code") or err.get("status") or ""),
                      str(err.get("message") or "")[:200]) from None
    audio, mime, said = None, "", []
    for out in reply.get("outputs") or []:
        if not isinstance(out, dict):
            continue
        if out.get("type") == "audio" and out.get("data"):
            audio = base64.b64decode(out["data"])
            mime = out.get("mime_type") or out.get("mimeType") or ""
        elif out.get("type") == "text" and out.get("text"):
            said.append(str(out["text"]))
    if not audio:
        raise Refused(200, "no_audio", f"status {reply.get('status')!r}")
    return audio, mime, " ".join(said)[:400]


def compose(music: Music, job_id: str, out_dir: str | Path, duration: float,
            marks: list[float], on_line=None,
            http=urllib.request.urlopen,
            token: str | None = None,
            images: list[bytes] | None = None) -> tuple[Path | None, dict]:
    """Compose the score with Lyria. `(None, plan)` means use the fallback;
    `plan["error"]` says why."""
    emit = on_line or (lambda _l: None)
    model = os.environ.get("AT_MUSIC_MODEL") or DEFAULT_MODEL
    location = os.environ.get("AT_MUSIC_LOCATION") or DEFAULT_LOCATION
    plan: dict = {"provider": "lyria", "model": model,
                  "requested_seconds": round(float(duration), 2)}
    project = project_id(http)
    if not project:
        plan["error"] = "no Google Cloud project to bill"
        return None, plan
    bpm, root = fitted_bpm(music, job_id, marks)
    plan.update(bpm=bpm, key=key_name(root, music.mood or "lofi"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = token
    tries = (False, True, True)            # as directed, then plain twice
    for n_try, plain in enumerate(tries):
        last = n_try == len(tries) - 1
        prompt = build_prompt(music, duration, marks, bpm, root, plain=plain)
        # frames go with the agent's direction only; a refused request is
        # retried without them, in case they were what the filter disliked
        frames = images if (images and not plain) else []
        if frames:
            prompt += ("\nThe images are frames from this video, in order: "
                       "write the music to suit what they show.")
        h = hashlib.sha256(f"{model}\0{prompt}".encode())
        for img in frames:
            h.update(hashlib.sha256(img).digest())
        digest = h.hexdigest()[:16]
        cached = out_dir / f"_lyria_{digest}.mp3"
        plan.update(prompt=prompt, plain_prompt=plain, frames=len(frames))
        if cached.exists() and cached.stat().st_size > 1000:
            emit(f"music: reusing the track Lyria composed for this cut "
                 f"({cached.name})")
            plan["cached"] = True
            return cached, plan
        tok = tok or access_token(http)
        if not tok:
            plan["error"] = "no Google identity to call Lyria with"
            return None, plan
        emit(f"music: asking {model} for {duration:.1f}s at {bpm:.0f} BPM"
             + (f" with {len(frames)} frame(s) of the cut" if frames else "")
             + (f" (try {n_try + 1}, plain prompt, after a refusal)"
                if plain else ""))
        for attempt in (1, 2):
            t0 = time.monotonic()
            try:
                audio, mime, said = request_track(
                    prompt, tok, project, model, location,
                    float(os.environ.get("AT_MUSIC_TIMEOUT", 240)), http,
                    images=frames)
            except Refused as e:
                plan["error"] = str(e)
                if e.code == "content_blocked" and not last:
                    emit("music: Lyria refused the request; asking again")
                    break                       # -> the next try
                if e.status in (429, 500, 502, 503, 504) and attempt == 1:
                    time.sleep(5)
                    continue
                emit(f"music: Lyria failed ({e}); using the code-composed "
                     f"score")
                return None, plan
            except Exception as e:                         # noqa: BLE001
                plan["error"] = f"{type(e).__name__}: {e}"[:200]
                if attempt == 1:
                    time.sleep(5)
                    continue
                emit(f"music: Lyria unreachable ({plan['error']}); using "
                     f"the code-composed score")
                return None, plan
            cached.write_bytes(audio)
            plan.pop("error", None)
            plan.update(seconds_to_compose=round(time.monotonic() - t0, 1),
                        mime=mime, bytes=len(audio), description=said)
            emit(f"music: composed by {model} in "
                 f"{plan['seconds_to_compose']:.0f}s ({len(audio)} bytes)")
            return cached, plan
        else:
            return None, plan
    return None, plan


__all__ = ["enabled", "compose", "build_prompt", "section_spans",
           "request_track", "access_token", "project_id", "Refused"]
