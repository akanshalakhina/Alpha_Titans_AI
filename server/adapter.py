"""
Adapter over the EXISTING Alpha Titans pipeline.

The pipeline itself is not reimplemented here. `build_video.py` is invoked
exactly as it is invoked by hand, one stage per subprocess, and its
`manifest.json` is the single source of truth for everything the UI shows:
clip probes, motion measurements, the cut sheet, transition timings.

Stage list and semantics come straight from build_video.py's own docstring:
    probe -> measure -> plan -> sheets -> cut -> join   (unchanged)
    finish                                              (see extras.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import config

# ---------------------------------------------------------------- locations
# All of these resolve inside the application folder via config.py, so the app
# is portable: copy the folder, install the requirements, run it.

REPO = config.APP_ROOT
PIPELINE = config.PIPELINE
BEATBED = config.BEATBED
BGM_STDLIB = config.BGM_STDLIB

# mirrored from build_video.py so the API accepts exactly what the pipeline
# is willing to glob
VIDEO_EXT = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

UPLOADS = config.UPLOADS_DIR
OUTPUTS = config.OUTPUTS_DIR
MUSIC_LIB = config.MUSIC_DIR


def preflight() -> dict:
    """What the server can actually do right now. Surfaced in the UI so the
    operator is never guessing which capabilities are live."""
    def have(cmd: str) -> bool:
        return subprocess.run(["which", cmd], capture_output=True).returncode == 0

    def importable(mod: str) -> bool:
        return subprocess.run([sys.executable, "-c", f"import {mod}"],
                              capture_output=True).returncode == 0

    return {
        "pipeline": PIPELINE.exists(),
        "pipeline_path": str(PIPELINE),
        "ffmpeg": have(config.FFMPEG),
        "ffprobe": have(config.FFPROBE),
        "beat_engine": (BEATBED.exists() and importable("numpy") and importable("scipy")
                        and importable("soundfile")),
        "stdlib_engine": BGM_STDLIB.exists(),
        # No offline STT engine could be installed in this environment, so
        # voice notes are captured and stored but not machine-transcribed.
        "transcription": importable("vosk") or importable("faster_whisper"),
    }


# ------------------------------------------------------------------ agents
#
# The work is split into five agents, each owning one job end to end. "Agent"
# here means an autonomous component with its own inputs, its own decisions and
# its own reported status - NOT that each one is a language model. Exactly one
# of them (Vision) calls a model; the other four are deterministic. That
# distinction is kept visible in the UI rather than blurred.
#
#   id        owns                                         model?
#   vision    looking at the footage                       YES (optional)
#   editor    which clips, what order, where to trim       no
#   animation transitions, crossfade length, motion feel    no
#   merge     rendering and assembling into one file       no
#   music     the score and the footage's own audio        no

AGENTS = [
    ("vision", "Vision Agent",
     "Looks at real frames and reports what each clip actually shows",
     True),
    ("editor", "Editor Agent",
     "Chooses which clips make the cut, their order, and where each is trimmed",
     False),
    ("animation", "Animation Agent",
     "Picks a transition for every individual cut, its length, and the motion feel",
     False),
    ("merge", "Merge Agent",
     "Renders each segment onto one canvas and assembles them into a single file",
     False),
    ("music", "Music Agent",
     "Scores the cut against its real transition marks and handles the source audio",
     False),
]
AGENT_IDS = [a[0] for a in AGENTS]
AGENT_BY_ID = {a[0]: {"id": a[0], "name": a[1], "does": a[2], "uses_model": a[3]}
               for a in AGENTS}


# -------------------------------------------------------------- stage calls

# (id, label, what it really does, rough share of total work)
STAGES = [
    ("probe", "Clip Probe",
     "ffprobe every upload; rotation, fps, codec, audio; drops iOS raw/edited duplicates", 4),
    ("measure", "Motion Analysis",
     "freezedetect + signalstats YDIF per clip to find dead air and real movement", 20),
    ("vision", "Vision Analysis",
     "a vision model looks at real frames from every clip; skipped when no API key is set", 14),
    ("plan", "Cut Planner",
     "scores windows and allocates the length budget by motion", 3),
    ("direct", "Edit Plan",
     "turns the measurements into per-cut transitions, length and mood, each with its reason", 2),
    ("sheets", "Contact Sheets",
     "a tiled preview per planned segment", 7),
    ("cut", "Segment Render",
     "each segment onto the target canvas with its caption burnt in", 31),
    ("join", "Transitions",
     "crossfades the segments and records the transition timings", 15),
    ("music", "Score",
     "writes a track against the real transition marks, or fits your uploaded one", 8),
    ("master", "Master & Verify",
     "retime, mux, loudness-normalise, thumbnail, then count decoded frames", 12),
]
STAGE_IDS = [s[0] for s in STAGES]
STAGE_WEIGHT = {s[0]: s[3] for s in STAGES}
TOTAL_WEIGHT = sum(s[3] for s in STAGES)

# Which agent runs which stage. `direct` is shared: the Editor decides the cut
# and the Animation agent decides the motion, both inside that one stage, so it
# is attributed to the editor here and each individual decision carries its own
# agent tag in edit_plan.json.
STAGE_AGENT = {
    "probe":   "editor",     # read every file
    "measure": "editor",     # find where the movement is
    "vision":  "vision",     # look at what is actually in shot
    "plan":    "editor",     # choose clips, order, trims, pacing
    "direct":  "editor",     # write the edit plan (animation decisions too)
    "sheets":  "merge",      # preview each planned segment
    "cut":     "merge",      # render segments onto one canvas
    "join":    "animation",  # apply the per-cut transitions it chose
    "music":   "music",      # score it, or carry the source audio
    "master":  "merge",      # assemble, normalise and verify the one file
}


def agents_manifest() -> list[dict]:
    """Every agent with the stages it owns - what the UI groups by."""
    out = []
    for aid, name, does, uses_model in AGENTS:
        mine = [s for s in STAGE_IDS if STAGE_AGENT.get(s) == aid]
        out.append({"id": aid, "name": name, "does": does,
                    "uses_model": uses_model, "stages": mine,
                    "weight": sum(STAGE_WEIGHT[s] for s in mine)})
    return out


def job_dir(job_id: str) -> Path:
    """Where this job's build lives (manifest, parts, master, state)."""
    return OUTPUTS / job_id


def src_dir(job_id: str) -> Path:
    """The operator's raw clips, exactly what the pipeline globs."""
    return UPLOADS / job_id


def build_dir(job_id: str) -> Path:
    return OUTPUTS / job_id


def voice_dir(job_id: str) -> Path:
    return UPLOADS / job_id / "_voice"


def poster_dir(job_id: str) -> Path:
    return OUTPUTS / job_id / "_posters"


def read_manifest(job_id: str) -> dict | None:
    p = build_dir(job_id) / "manifest.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def stage_command(job_id: str, stage: str, settings: dict) -> list[str]:
    """Build the exact CLI call, same as running the pipeline by hand."""
    out = str(build_dir(job_id))
    cmd = [sys.executable, str(PIPELINE), stage, "--out", out]
    if stage == "probe":
        cmd += ["--src", str(src_dir(job_id))]
        cmd += ["--orient", settings.get("orientation", "auto")]
    if stage == "plan":
        target = settings.get("target_seconds")
        if target:
            cmd += ["--target", str(float(target))]
        seq = settings.get("sequence")
        if seq in SEQUENCES:
            cmd += ["--sequence", seq]
        # Anything the vision pass concluded reaches the planner here, as
        # ordinary CLI flags. No vision run means no flags and identical
        # behaviour to before.
        import vision as _vision
        v = _vision.read(job_id) or {}
        cross = v.get("cross") or {}
        for a, b in (cross.get("duplicates") or []):
            cmd += ["--drop", "%s::the vision pass judged this the same "
                              "material as %s" % (a, b)]
        if cross.get("hook") and settings.get("sequence", "hook") == "hook":
            cmd += ["--hook", cross["hook"]]
    return cmd


def env_for(settings: dict) -> dict:
    """Render settings the pipeline reads as overrides.

    build_video.py was given env-var defaults for these four constants so the
    UI can drive resolution, quality and transition feel without changing any
    of its logic. Unset, every value stays exactly what it was.
    """
    env = dict(os.environ)
    res = settings.get("resolution")
    if res in RESOLUTIONS:
        w, h = RESOLUTIONS[res]
        orient = settings.get("orientation", "auto")
        # portrait output wants the long edge vertical
        if orient == "vertical":
            w, h = h, w
        env["AT_CANVAS"] = f"{w}x{h}"
    if settings.get("quality") in QUALITY:
        env["AT_CRF"] = str(QUALITY[settings["quality"]])
    # An explicit per-cut list from the director wins over a style preset;
    # this is how its choices actually reach the join stage.
    chosen = settings.get("transition_list")
    style = settings.get("transition_style")
    if chosen:
        env["AT_XFADES"] = ",".join(chosen)
    elif style in TRANSITIONS:
        env["AT_XFADES"] = ",".join(TRANSITIONS[style])
    dur = settings.get("transition_duration")
    if dur:
        env["AT_XFADE_T"] = str(float(dur))
    return env


RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}

QUALITY = {"draft": 24, "standard": 20, "high": 18, "master": 15}

TRANSITIONS = {
    "cut": ["fade"],
    "smooth": ["fade", "smoothleft", "fade", "smoothup"],
    "dynamic": ["fadewhite", "wipeup", "smoothleft", "fade", "wipeleft"],
    "flash": ["fadewhite", "fadewhite", "fade", "fadewhite"],
}

SPEEDS = {"slow": 0.75, "normal": 1.0, "fast": 1.35}

# Running order for the cut. All three are driven by real data - the measured
# motion, the recording timestamp in the container, or your own drag order.
SEQUENCES = {
    "hook": "liveliest clip opens, then your order - best for a reel",
    "recorded": "chronological by recording time - follows the build",
    "uploaded": "exactly the order you arranged them in",
}

# What to do with the footage's OWN audio. The pipeline renders segments
# video-only, so this is reassembled in extras.source_audio from the planner's
# in/out points rather than by changing the render.
KEEP_AUDIO = {
    "none": "music bed only - the footage's own audio is dropped",
    "mix": "original audio over the score, score ducked beneath it",
    "only": "original audio only, no score",
}


def speed_factor(settings: dict) -> float:
    mode = settings.get("speed", "normal")
    if mode == "custom":
        try:
            return max(0.4, min(2.5, float(settings.get("speed_custom", 1.0))))
        except (TypeError, ValueError):
            return 1.0
    return SPEEDS.get(mode, 1.0)


def run_stage(job_id: str, stage: str, settings: dict, on_line) -> int:
    """Run one pipeline stage, streaming its stdout back line by line."""
    cmd = stage_command(job_id, stage, settings)
    on_line("$ " + " ".join(cmd[1:]))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env_for(settings))
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            on_line(line)
    return proc.wait()


def ffprobe_json(path: str | Path) -> dict:
    out = subprocess.run(
        [config.FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {}


def media_summary(path: str | Path) -> dict:
    """Duration / size / dimensions for an upload, from ffprobe - never guessed."""
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    w, h = v.get("width"), v.get("height")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rot = int(sd["rotation"])
            except (TypeError, ValueError):
                rot = 0
    if w and h and abs(rot) % 180 == 90:
        w, h = h, w
    try:
        dur = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        dur = 0.0
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    recorded = tags.get("creation_time")
    if not recorded:
        vt = {k.lower(): val for k, val in (v.get("tags") or {}).items()}
        recorded = vt.get("creation_time")
    return {
        "duration": round(dur, 2),
        "recorded": recorded,
        "size": int(fmt.get("size", 0) or 0),
        "width": w,
        "height": h,
        "rotation": rot,
        "codec": v.get("codec_name"),
        "has_audio": a is not None,
        "orientation": ("vertical" if (w and h and h > w) else
                        "horizontal" if w else None),
    }
