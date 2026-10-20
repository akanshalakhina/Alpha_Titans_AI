"""Turning names that came from a person into paths that cannot escape.

A file's name arrives from a browser and passes through Drive. Neither
makes it safe: `../../etc/cron.d/x`, `C:\\Windows\\...`, `con.mp4`, a NUL
byte and a name that is only dots have all been valid Drive file names at
one point or another. So nothing here trusts a name. It is reduced to a
bare, conservative filename, given a numbered prefix the code controls, and
then the resolved path is checked to still be inside its workspace -
belt and braces, because the second check is what catches the case the
first one did not think of.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .sanitize import CloudError

# The engine's own list - pipeline/build_video.py VIDEO_EXT. A test holds
# the two equal, so a format the renderer cannot read is refused at upload
# rather than after a worker has been started for it.
VIDEO_EXT = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

_RESERVED = {"con", "prn", "aux", "nul",
             *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}
_ID = re.compile(r"^[a-f0-9]{12}$")


def safe_name(name: str, max_len: int = 120) -> str:
    """A filename that is just a filename on every OS."""
    base = re.split(r"[\\/]", str(name or ""))[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    if stem.lower() in _RESERVED:
        stem = "_" + stem
    stem = stem[:max_len] or "file"
    return f"{stem}.{ext.lower()}" if ext else stem


def video_ext(name: str) -> str | None:
    ext = os.path.splitext(safe_name(name))[1].lower()
    return ext if ext in VIDEO_EXT else None


def check_job_id(job_id: str) -> str:
    """Job ids are ours - 12 hex characters - and are used as directory
    names, so anything else is refused outright."""
    if not _ID.match(job_id or ""):
        raise CloudError("invalid", "malformed job id")
    return job_id


def within(root: Path, *parts: str) -> Path:
    """root/parts, refusing anything that resolves outside `root`."""
    root = Path(root).resolve()
    target = root.joinpath(*parts).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise CloudError("invalid", "path escapes the job workspace") from None
    if target == root:
        raise CloudError("invalid", "path is the workspace itself")
    return target


def source_filename(idx: int, name: str) -> str:
    """`01_clip.mp4` - numbered so the engine's 'uploaded' sequence order
    matches the order the person gave, and so two files with the same name
    cannot collide."""
    return f"{int(idx) + 1:02d}_{safe_name(name)}"
