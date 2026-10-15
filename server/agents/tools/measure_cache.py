"""Measurement reuse across jobs, keyed on the footage itself.

WHY THIS EXISTS
---------------
`stage_measure` already skips a clip that has been measured: it keeps a
`<clip>.ydif.txt` beside the build and reuses it. But every job gets its own
build directory, so that cache only ever helps a RESUMED build - upload the
same footage again and the whole pass runs from scratch.

The cost is not small. Measuring one eight-minute clip decodes it twice,
once for `freezedetect` and once for `signalstats`, and took 586 seconds -
half of a twenty-minute production. Four separate runs over the same file
produced four byte-identical `.ydif.txt` files (md5 9c5be1a5...), so forty
minutes went into work that was already done.

WHAT IS CACHED
--------------
The measurement is a pure function of the file's bytes: the same video
always yields the same YDIF series and the same freeze runs. So the key is
a sha256 of the file - 1.5 seconds for 451 MB, against the ten minutes it
saves - and the value is the measured clip row plus its `.ydif.txt`.

Nothing here decides anything. It restores what `stage_measure` would have
computed, into the place `stage_measure` looks, and then that stage's own
"already measured" check skips the work. If anything at all is missing or
unreadable, the cache stays silent and the real pass runs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

# Everything the measure pass adds to a clip row. A cached row is only used
# when it carries all of them, so a row written by an older, narrower
# version of the stage can never silently stand in for a full measurement.
MEASURED_FIELDS = ("mean_ydif", "max_ydif", "live_share", "freeze_share",
                   "freeze_runs", "sample_fps", "ydif")

_CHUNK = 1 << 20


def file_key(path: Path) -> str | None:
    """sha256 of the file's bytes, or None when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(_CHUNK), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


def _entry(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.json", cache_dir / f"{key}.ydif.txt"


def _is_complete(row: dict) -> bool:
    return isinstance(row, dict) and all(f in row for f in MEASURED_FIELDS)


def load(cache_dir: Path, key: str) -> tuple[dict, bytes] | None:
    """The stored row and its stats file, or None for any kind of miss."""
    meta, stats = _entry(Path(cache_dir), key)
    if not (meta.exists() and stats.exists()):
        return None
    try:
        row = json.loads(meta.read_text(encoding="utf-8"))
        blob = stats.read_bytes()
    except (OSError, json.JSONDecodeError):
        return None
    return (row, blob) if _is_complete(row) else None


def save(cache_dir: Path, key: str, row: dict, stats_file: Path) -> bool:
    """Store one measured clip. False when there is nothing worth storing."""
    if not _is_complete(row) or not stats_file.exists():
        return False
    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta, stats = _entry(cache_dir, key)
        # the stats file first: a reader checks for BOTH, so the json
        # landing last means a half-written pair is never treated as a hit
        shutil.copy2(stats_file, stats)
        keep = {k: row[k] for k in MEASURED_FIELDS}
        meta.write_text(json.dumps(keep), encoding="utf-8")
        return True
    except OSError:
        return False


def restore(cache_dir: Path, src_dir: Path, build_dir: Path,
            manifest: dict, on_line=None) -> list[str]:
    """Fill in every clip already measured elsewhere. Returns their names.

    Writes into the manifest and the build's `measure/` folder, which is
    exactly where `stage_measure` looks before deciding what to do. It is
    then that stage - unmodified - which skips them.
    """
    cache_dir, src_dir, build_dir = (Path(cache_dir), Path(src_dir),
                                     Path(build_dir))
    if not cache_dir.exists():
        return []
    vdir = build_dir / "measure"
    hit: list[str] = []
    for clip in manifest.get("clips") or []:
        name = clip.get("name")
        if not name or "mean_ydif" in clip:
            continue                       # already measured in this build
        key = file_key(src_dir / name)
        if not key:
            continue
        got = load(cache_dir, key)
        if got is None:
            continue
        row, blob = got
        try:
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"{name}.ydif.txt").write_bytes(blob)
        except OSError:
            continue
        clip.update(row)
        hit.append(name)
        if on_line:
            on_line(f"reusing the measurement of {name} - the same file was "
                    f"measured before, and the result is the same")
    return hit


def store(cache_dir: Path, src_dir: Path, build_dir: Path,
          manifest: dict) -> list[str]:
    """Save every clip this build measured. Returns their names."""
    cache_dir, src_dir, build_dir = (Path(cache_dir), Path(src_dir),
                                     Path(build_dir))
    vdir = build_dir / "measure"
    saved: list[str] = []
    for clip in manifest.get("clips") or []:
        name = clip.get("name")
        if not name or not _is_complete(clip):
            continue
        key = file_key(src_dir / name)
        if key and save(cache_dir, key, clip, vdir / f"{name}.ydif.txt"):
            saved.append(name)
    return saved
