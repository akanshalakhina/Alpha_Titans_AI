"""
The pieces the UI needs that the pipeline CLI did not already expose.

Deliberately thin. Nothing here re-implements measurement, planning or
cutting - those stay in build_video.py. This module only:

  * turns a pasted/uploaded script into the captions.json the pipeline
    already reads before `cut`
  * stores a voice note and probes it (no transcription engine is available
    in this environment, and we do not pretend otherwise)
  * runs the finishing pass: optional retime, score to the real transition
    marks using the existing music engines, loudness-normalise, thumbnail,
    and verify the result by counting decoded frames

Resolution and quality are NOT handled here - they are driven into the
pipeline's own render via the AT_CANVAS / AT_CRF environment overrides, so
segments are rendered at the target size in the first place rather than
being scaled afterwards.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import config
from adapter import (BEATBED, BGM_STDLIB, build_dir, ffprobe_json,
                     read_manifest, speed_factor, voice_dir)

FFMPEG, FFPROBE = config.FFMPEG, config.FFPROBE

# mood -> (engine, extra args). Every option maps to a real engine setting.
MOODS = {
    "lofi":       ("stdlib", ["--style", "lofi"]),
    "calm":       ("stdlib", ["--style", "lofi"]),
    "bright":     ("stdlib", ["--style", "bright"]),
    "upbeat":     ("stdlib", ["--style", "bright"]),
    "beat":       ("beatbed", []),
    "energetic":  ("beatbed", []),
    "cinematic":  ("beatbed", []),
}


def run(cmd, on_line=None, env=None) -> tuple[int, str]:
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env or dict(os.environ))
    buf = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        buf.append(line)
        if on_line and line:
            on_line(line)
    return proc.wait(), "\n".join(buf)


# ------------------------------------------------------------------ script

def split_script(script: str, n: int) -> list[str]:
    """Spread a script across n segment captions.

    Respects explicit line breaks first: one line per segment is the most
    predictable thing for an operator who wrote the script with the cut in
    mind. Falls back to sentence splitting, then to even chunking.
    """
    if n <= 0:
        return []
    script = (script or "").strip()
    if not script:
        return [""] * n

    lines = [l.strip(" -•\t") for l in script.splitlines()]
    lines = [l for l in lines if l]
    if len(lines) >= n:
        # more lines than segments: merge the tail into the last caption
        out = lines[:n - 1] + [" ".join(lines[n - 1:])]
        return [caption_trim(x) for x in out]
    if len(lines) > 1:
        out = lines + [""] * (n - len(lines))
        return [caption_trim(x) for x in out]

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", script) if p.strip()]
    if len(parts) >= n:
        out = parts[:n - 1] + [" ".join(parts[n - 1:])]
    else:
        out = parts + [""] * (n - len(parts))
    return [caption_trim(x) for x in out]


def caption_trim(text: str, limit: int = 58) -> str:
    """Captions are burnt in at a fixed size; over-long lines get shortened
    on a word boundary rather than being allowed to overflow the frame."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") or text[:limit]


def write_captions(job_id: str, script: str,
                   suggested: dict | None = None) -> list[str]:
    """Your script first; the vision model only fills genuine blanks.

    `suggested` maps a 1-based segment index to a caption the vision pass
    offered. A line you wrote always wins - the model never overwrites your
    words, it only speaks where you said nothing.
    """
    m = read_manifest(job_id) or {}
    segs = m.get("segments") or []
    caps = split_script(script, len(segs))
    for i in range(len(caps)):
        if not caps[i].strip() and suggested:
            got = suggested.get(str(i + 1)) or suggested.get(i + 1)
            if got:
                caps[i] = caption_trim(str(got))
    path = build_dir(job_id) / "captions.json"
    with open(path, "w") as f:
        json.dump(caps, f, indent=2)
    return caps


# ------------------------------------------------------------------- voice

_WHISPER_MODEL = None


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception:
            return None
    return _WHISPER_MODEL


def transcribe_voice_file(path: Path) -> tuple[str | None, str]:
    model = _get_whisper_model()
    if model is None:
        return None, "no_engine"
    try:
        segments, _ = model.transcribe(str(path))
        text = " ".join(s.text.strip() for s in segments).strip()
        return (text, "ok") if text else (None, "empty")
    except Exception as exc:
        return None, f"error: {exc}"


def store_voice_note(job_id: str, filename: str, blob: bytes) -> dict:
    d = voice_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "voice_note.webm")
    path = d / safe
    with open(path, "wb") as f:
        f.write(blob)
    info = ffprobe_json(path)
    dur = 0.0
    try:
        dur = float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        pass

    transcript, transcript_status = transcribe_voice_file(path)

    return {
        "filename": safe,
        "bytes": len(blob),
        "duration": round(dur, 2),
        "playable": dur > 0,
        "transcript": transcript,
        "transcript_status": transcript_status,
    }


# ------------------------------------------------------------------ finish

def retime(src: Path, dst: Path, factor: float, crf: str, on_line) -> None:
    """Change playback speed. Video only - the pipeline's joined file has no
    audio yet, so there is no atempo to keep in step."""
    on_line(f"retiming video x{factor:.2f}")
    rc, out = run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", src,
                   "-filter:v", f"setpts={1.0 / factor:.6f}*PTS",
                   "-an", "-c:v", "libx264", "-preset", "fast", "-crf", crf,
                   "-profile:v", "high", "-pix_fmt", "yuv420p", dst],
                  on_line)
    if rc != 0:
        raise RuntimeError(f"retime failed: {out[-800:]}")


def generate_music(job_id: str, duration: float, marks: list[float],
                   mood: str, on_line) -> Path | None:
    engine, extra = MOODS.get(mood, MOODS["beat"])
    wav = build_dir(job_id) / "score.wav"
    if engine == "beatbed" and BEATBED.exists():
        cmd = [sys.executable, str(BEATBED), str(wav), f"{duration:.2f}"]
        cmd += [f"{t:.3f}" for t in marks] or ["1.0"]
        # the transition marks double as action accents, which is what the
        # engine was written to key off
        if marks:
            cmd += ["--hits"] + [f"{t:.3f}" for t in marks]
        on_line(f"scoring with beatbed.py against {len(marks)} transition marks")
    elif BGM_STDLIB.exists():
        cmd = [sys.executable, str(BGM_STDLIB), str(wav), f"{duration:.2f}"]
        cmd += [f"{t:.3f}" for t in marks] or ["1.0"]
        cmd += extra
        on_line(f"scoring with bgm_stdlib.py ({' '.join(extra) or 'default'})")
    else:
        on_line("no music engine available - leaving the cut silent")
        return None
    rc, out = run(cmd, on_line)
    if (rc != 0 or not wav.exists()) and engine == "beatbed" and BGM_STDLIB.exists():
        # beatbed needs numpy, scipy and soundfile. If any is missing it fails,
        # and the cut used to go out silent although a score was asked for.
        # The standard-library engine always works, so fall back to it.
        on_line("beatbed.py failed (is `soundfile` installed? pip install soundfile) "
                "- scoring with bgm_stdlib.py instead")
        cmd = [sys.executable, str(BGM_STDLIB), str(wav), f"{duration:.2f}"]
        cmd += [f"{t:.3f}" for t in marks] or ["1.0"]
        cmd += ["--style", "bright"]
        rc, out = run(cmd, on_line)
    if rc != 0 or not wav.exists():
        on_line("music engine failed; continuing without a score")
        return None
    return wav


def prepare_track(src: Path, dst: Path, duration: float, on_line) -> Path | None:
    """Fit an uploaded track to the cut: loop if short, trim if long, fade out."""
    on_line(f"fitting uploaded track to {duration:.2f}s")
    rc, out = run([FFMPEG, "-nostdin", "-v", "error", "-y",
                   "-stream_loop", "-1", "-i", src, "-t", f"{duration:.3f}",
                   "-af", f"afade=t=out:st={max(duration - 2.5, 0):.3f}:d=2.5",
                   "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", dst],
                  on_line)
    if rc != 0:
        on_line("could not fit the uploaded track; continuing without it")
        return None
    return dst


def loudness(path: Path) -> float | None:
    rc, out = run([FFMPEG, "-nostdin", "-v", "info", "-i", path,
                   "-af", "ebur128", "-f", "null", "-"])
    tail = out.split("Summary:")[-1] if "Summary:" in out else ""
    m = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    return float(m.group(1)) if m else None


def frame_count(path: Path) -> int | None:
    rc, out = run([FFPROBE, "-v", "error", "-count_frames",
                   "-select_streams", "v:0", "-show_entries",
                   "stream=nb_read_frames", "-of", "csv=p=0", path])
    out = out.strip().splitlines()[-1] if out.strip() else ""
    return int(out) if out.isdigit() else None


def _plan_audio(job_id: str, settings: dict) -> dict:
    """Numbers both finishing stages agree on: the post-speed duration and the
    transition marks rescaled to match it."""
    m = read_manifest(job_id)
    if not m:
        raise RuntimeError("no manifest - the pipeline has not run")
    joined = build_dir(job_id) / "_joined.mp4"
    if not joined.exists():
        raise RuntimeError("_joined.mp4 missing - the join stage did not finish")
    factor = speed_factor(settings)
    probed = ffprobe_json(joined).get("format", {})
    try:
        base = float(probed.get("duration") or m.get("total_duration") or 0.0)
    except (TypeError, ValueError):
        base = float(m.get("total_duration") or 0.0)
    marks = [float(t) for t in (m.get("transitions") or [])]
    return {
        "joined": str(joined),
        "factor": factor,
        "duration": base / factor,
        "marks": [t / factor for t in marks],
    }


def _atempo(factor: float) -> str:
    """atempo only accepts 0.5-100 per instance, so slow speeds need two."""
    if abs(factor - 1.0) < 0.01:
        return ""
    if factor >= 0.5:
        return "atempo=%.6f," % factor
    half = factor ** 0.5
    return "atempo=%.6f,atempo=%.6f," % (half, half)


def source_audio(job_id: str, settings: dict, plan: dict, on_line) -> Path | None:
    """Rebuild the planned cut's ORIGINAL audio.

    The pipeline renders its segments video-only (`-an`), which is right for a
    silent montage but throws away the content itself when the footage is a
    tutorial, an interview or anything else where someone is talking. Rather
    than change the render, the same cut is reassembled here from the source
    files using the planner's own in/out points, crossfaded with `acrossfade`
    at exactly the `xfade` offsets the video used, so the two stay in step.

    Clips with no audio track contribute silence of the right length, so a
    mixed batch lines up instead of drifting.
    """
    m = read_manifest(job_id) or {}
    segs = m.get("segments") or []
    if not segs:
        return None
    src_root = Path(m.get("src") or "")
    bd = build_dir(job_id)

    have = 0
    args: list[str] = [FFMPEG, "-nostdin", "-v", "error", "-y"]
    chains: list[str] = []
    labels: list[str] = []
    for i, s in enumerate(segs):
        clip = src_root / s["clip"]
        dur = float(s["duration"])
        info = ffprobe_json(clip)
        has_audio = any(st.get("codec_type") == "audio"
                        for st in info.get("streams", []))
        if has_audio and clip.exists():
            have += 1
            args += ["-ss", f"{float(s['start']):.3f}", "-t", f"{dur:.3f}",
                     "-i", str(clip)]
            chains.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:"
                          f"channel_layouts=stereo,apad,atrim=0:{dur:.3f},"
                          f"asetpts=N/SR/TB[a{i}]")
        else:
            args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i",
                     "anullsrc=channel_layout=stereo:sample_rate=48000"]
            chains.append(f"[{i}:a]atrim=0:{dur:.3f},asetpts=N/SR/TB[a{i}]")
        labels.append(f"[a{i}]")

    if not have:
        on_line("none of the planned clips carry an audio track; "
                "nothing to keep")
        return None

    # chain acrossfade the same way join chains xfade, so the audio lands on the
    # video's own transition points and ends at the same total length
    xf = float(os.environ.get("AT_XFADE_T") or settings.get("transition_duration") or 0.4)
    prev = labels[0]
    for i in range(1, len(labels)):
        out = "[axf]" if i == len(labels) - 1 else f"[ax{i}]"
        chains.append(f"{prev}{labels[i]}acrossfade=d={xf:.3f}:c1=tri:c2=tri{out}")
        prev = out
    if len(labels) == 1:
        prev = labels[0]

    tempo = _atempo(float(plan.get("factor") or 1.0))
    chains.append(f"{prev}{tempo}aresample=48000[aout]")

    dst = bd / "source_audio.wav"
    args += ["-filter_complex", ";".join(chains), "-map", "[aout]",
             "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(dst)]
    on_line("rebuilding original audio from %d of %d planned segments"
            % (have, len(segs)))
    rc, out = run(args, on_line)
    if rc != 0 or not dst.exists():
        on_line("could not rebuild the original audio; continuing without it")
        return None
    return dst


def mix_beds(voice: Path, music: Path, dst: Path, duration: float,
             on_line) -> Path | None:
    """Music under the original audio, ducked out of its way.

    sidechaincompress keyed off the source audio, so the score drops whenever
    there is something to hear and comes back up when there is not.
    """
    on_line("mixing: original audio over the score, score ducked beneath it")
    rc, out = run([FFMPEG, "-nostdin", "-v", "error", "-y",
                   "-i", str(music), "-i", str(voice), "-filter_complex",
                   "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:"
                   "channel_layouts=stereo,volume=0.55[m];"
                   "[1:a]aformat=sample_fmts=fltp:sample_rates=48000:"
                   "channel_layouts=stereo[v];"
                   "[v]asplit=2[vm][vk];"
                   "[m][vk]sidechaincompress=threshold=0.05:ratio=8:"
                   "attack=12:release=420[md];"
                   "[md][vm]amix=inputs=2:duration=first:dropout_transition=0"
                   ",alimiter=limit=0.95[aout]",
                   "-map", "[aout]", "-t", f"{duration:.3f}",
                   "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(dst)],
                  on_line)
    if rc != 0 or not dst.exists():
        on_line("mix failed; falling back to the original audio alone")
        return None
    return dst


def _describe_audio(settings: dict, plan: dict, track) -> str:
    """What the finished file's audio actually is, in the UI's words."""
    if not track:
        return "silent"
    keep = plan.get("keep_audio", "none")
    got = bool(plan.get("has_source_audio"))
    if got and keep == "only":
        return "original audio"
    bed = ("library" if settings.get("music_source") == "upload"
           else "silent" if settings.get("music_source") == "none"
           else settings.get("mood", "beat"))
    if got and keep == "mix":
        return f"original audio + {bed}" if bed != "silent" else "original audio"
    return bed


def music_stage(job_id: str, settings: dict, on_line) -> dict:
    """Stage 7. Produce the audio bed only - no video work happens here."""
    plan = _plan_audio(job_id, settings)
    bd = build_dir(job_id)
    if abs(plan["factor"] - 1.0) > 0.01:
        on_line("speed x%.2f -> score length %.2fs, marks rescaled"
                % (plan["factor"], plan["duration"]))

    track = None
    source = settings.get("music_source", "generate")
    if source == "upload":
        up = settings.get("music_file")
        cand = Path(up) if up else None
        if cand and cand.exists():
            track = prepare_track(cand, bd / "track.wav", plan["duration"], on_line)
        else:
            on_line("music set to 'library' but no track was attached; "
                    "this build will be silent")
    elif source == "none":
        on_line("music disabled for this build")
    else:
        track = generate_music(job_id, plan["duration"], plan["marks"],
                               settings.get("mood", "beat"), on_line)

    # The footage's own audio. Off by default, because a silent montage is the
    # common case and wind noise is not content - but for a tutorial, interview
    # or talking-head clip the speech IS the content, so dropping it silently
    # would be the wrong default to force on everyone.
    keep = settings.get("keep_audio", "none")
    voice = None
    if keep in ("mix", "only"):
        voice = source_audio(job_id, settings, plan, on_line)

    if voice and keep == "only":
        track = voice
    elif voice and track:
        mixed = mix_beds(voice, Path(track), bd / "mixed.wav",
                         plan["duration"], on_line)
        track = mixed or voice
    elif voice:
        track = voice

    plan["track"] = str(track) if track else None
    plan["keep_audio"] = keep
    plan["has_source_audio"] = bool(voice)
    with open(bd / "_audio.json", "w") as f:
        json.dump(plan, f, indent=2)
    on_line("audio ready: %s" % (Path(plan["track"]).name if track else "silent"))
    return plan


def master_stage(job_id: str, settings: dict, on_line) -> dict:
    """Stage 8. Retime if asked, mux, normalise, thumbnail, then verify by
    counting decoded frames rather than trusting the container duration."""
    bd = build_dir(job_id)
    apath = bd / "_audio.json"
    plan = json.loads(apath.read_text()) if apath.exists() \
        else _plan_audio(job_id, settings)
    m = read_manifest(job_id) or {}
    crf = str(__import__("adapter").QUALITY.get(settings.get("quality"), 18))

    video = Path(plan["joined"])
    factor = float(plan.get("factor") or 1.0)
    if abs(factor - 1.0) > 0.01:
        video = bd / "_retimed.mp4"
        retime(Path(plan["joined"]), video, factor, crf, on_line)

    track = Path(plan["track"]) if plan.get("track") else None
    tag = settings.get("title") or "ALPHA_TITANS"
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "ALPHA_TITANS"
    canvas = m.get("canvas") or [1920, 1080]
    kind = "vertical" if canvas[1] > canvas[0] else "horizontal"
    final = bd / f"{tag}_{kind}.mp4"

    on_line("muxing and loudness-normalising")
    if track and track.exists():
        rc, out = run([FFMPEG, "-nostdin", "-v", "error", "-y",
                       "-i", video, "-i", track, "-filter_complex",
                       "[1:a]loudnorm=I=-15:TP=-1.5:LRA=11,"
                       "aformat=sample_fmts=fltp:sample_rates=48000:"
                       "channel_layouts=stereo[a]",
                       "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                       "-c:a", "aac", "-b:a", "192k",
                       "-movflags", "+faststart", final], on_line)
    else:
        rc, out = run([FFMPEG, "-nostdin", "-v", "error", "-y",
                       "-i", video, "-f", "lavfi", "-i",
                       "anullsrc=channel_layout=stereo:sample_rate=48000",
                       "-shortest", "-map", "0:v", "-map", "1:a",
                       "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                       "-movflags", "+faststart", final], on_line)
    if rc != 0:
        raise RuntimeError(f"master failed: {out[-800:]}")

    # thumbnail from the middle of the strongest planned segment
    segs = m.get("segments") or []
    dur_guess = float(plan.get("duration") or 0) or 1.0
    at = dur_guess * 0.35
    if segs:
        best = max(segs, key=lambda s: s.get("score", 0))
        acc = 0.0
        for s in segs:
            if s is best:
                acc += s.get("duration", 0) / 2
                break
            acc += s.get("duration", 0) - 0.4
        at = min(max(acc / factor, 0.2), max(dur_guess - 0.2, 0.2))
    thumb = bd / "thumbnail.jpg"
    run([FFMPEG, "-nostdin", "-v", "error", "-y", "-ss", f"{at:.2f}",
         "-i", final, "-frames:v", "1", thumb], on_line)

    on_line("verifying the finished file")
    frames = frame_count(final)
    lufs = loudness(final)
    rc, black = run([FFMPEG, "-nostdin", "-v", "error", "-i", final,
                     "-vf", "blackdetect=d=0.3", "-f", "null", "-"])
    info = ffprobe_json(final)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = info.get("format", {})
    dur = float(fmt.get("duration") or 0.0)
    expected = int(round(dur * 30))
    meta = {
        "file": final.name,
        "path": str(final),
        "thumbnail": thumb.name if thumb.exists() else None,
        "duration": round(dur, 2),
        "frames": frames,
        "frames_expected": expected,
        "frames_ok": (frames is not None and abs(frames - expected) <= 2),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": v.get("r_frame_rate"),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name"),
        "sample_rate": a.get("sample_rate"),
        "size": int(fmt.get("size") or 0),
        "lufs": lufs,
        "black_frames": bool(black.strip()),
        "speed": factor,
        "transitions": [round(t, 3) for t in (plan.get("marks") or [])],
        "segments": len(segs),
        "music": _describe_audio(settings, plan, track),
        "keep_audio": plan.get("keep_audio", "none"),
        "has_source_audio": bool(plan.get("has_source_audio")),
        "orientation": kind,
        "resolution": f"{v.get('width')}x{v.get('height')}",
    }
    with open(bd / "output.json", "w") as f:
        json.dump(meta, f, indent=2)
    on_line("done: %s  %.2fs  %sx%s  frames=%s  %s LUFS"
            % (final.name, meta["duration"], meta["width"], meta["height"],
               frames, lufs))
    return meta
