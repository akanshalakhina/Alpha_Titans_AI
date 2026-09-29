"""
Configuration. Every path is relative to the application folder, so the whole
app can be copied to another machine and run without editing anything.

Values may be overridden in a .env file next to this folder (see .env.example)
or by real environment variables. No third-party dependency is used to read it.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value, # comments, optional quotes.
    Real environment variables always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)


_load_dotenv(APP_ROOT / ".env")


def _path(env_key: str, default: Path) -> Path:
    raw = os.environ.get(env_key)
    p = Path(raw).expanduser() if raw else default
    return p if p.is_absolute() else (APP_ROOT / p).resolve()


def _int(env_key: str, default: int) -> int:
    try:
        return int(os.environ.get(env_key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- locations

FRONTEND_DIR = _path("AT_FRONTEND_DIR", APP_ROOT / "frontend")
PIPELINE_DIR = _path("AT_PIPELINE_DIR", APP_ROOT / "pipeline")
UPLOADS_DIR = _path("AT_UPLOADS_DIR", APP_ROOT / "uploads")
OUTPUTS_DIR = _path("AT_OUTPUTS_DIR", APP_ROOT / "outputs")
MUSIC_DIR = _path("AT_MUSIC_DIR", APP_ROOT / "data" / "music")

PIPELINE = PIPELINE_DIR / "build_video.py"
BEATBED = PIPELINE_DIR / "beatbed.py"
BGM_STDLIB = PIPELINE_DIR / "bgm_stdlib.py"

# ------------------------------------------------------------------ runtime

HOST = os.environ.get("AT_HOST", "127.0.0.1")
PORT = _int("AT_PORT", 8000)
FFMPEG = os.environ.get("AT_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("AT_FFPROBE", "ffprobe")

MAX_CLIP_MB = _int("AT_MAX_CLIP_MB", 2048)
MAX_TRACK_MB = _int("AT_MAX_TRACK_MB", 120)
MAX_VOICE_MB = _int("AT_MAX_VOICE_MB", 40)
MAX_CLIP_BYTES = MAX_CLIP_MB * 1024 * 1024

# Comma-separated origins for CORS, only needed if the frontend is served from
# somewhere other than this server. Empty means same-origin only.
CORS_ORIGINS = [o.strip() for o in os.environ.get("AT_CORS_ORIGINS", "").split(",")
                if o.strip()]


def ensure_dirs() -> None:
    for d in (UPLOADS_DIR, OUTPUTS_DIR, MUSIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
