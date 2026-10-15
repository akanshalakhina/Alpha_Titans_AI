"""Published-video outcomes: the YouTube reference, and the exported rows.

WHY THIS EXISTS
---------------
The Learning agent can only learn from outcomes, and until now there were
none to read: `publishing` writes a PLAN and uploads nothing, so
`PerformanceAnalysis` reported "unavailable" forever and Learning sat at
0 usable samples of 30. That was honest, and it was also a dead end.

The outcomes exist - in the operator's warehouse, behind a query like

    SELECT title, video_id, SUM(views), SUM(watch_time_minutes),
           SUM(likes), SUM(subscribers_gained), MIN(date), MAX(date)
    FROM <catalog>.<schema>.youtube_video_daily
    GROUP BY title, video_id

but this project cannot run it: there is no Databricks or BigQuery driver
installed and adding one would mean a new dependency and the operator's
warehouse credentials. So the query's RESULT is what crosses the gap - a
CSV exported from the warehouse's own UI and dropped in a folder. Nothing
here reaches the network, and the whole reader is the standard library.

THE JOIN
--------
A view count is worthless unless it can be attributed to the edit that
earned it, and nothing in a build knows its own YouTube id - the upload
happens outside this system entirely. So the operator records the link
once, after uploading, and that reference is what a row is matched on.
Matching on the title instead was considered and rejected: the Title
agent's text is editable in YouTube Studio, and a silent mismatch would
attach one video's numbers to another video's decisions - the one error
that would poison everything Learning ever concluded.

An unmatched row is reported as unmatched. It is never guessed at.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# The folder a warehouse export is dropped into. Every *.csv in it is
# read; nothing is written back, so an export can be replaced at any time.
# It sits beside the music the renderer already reads from `data/`, so a
# single folder holds everything the operator supplies by hand.
EXPORT_DIRNAME = "analytics"


def export_dir() -> "Path":
    """`data/analytics`, created on demand so the first drop just works."""
    import config as _config                    # late: keeps import cost off
    d = pathlib.Path(_config.APP_ROOT) / "data" / EXPORT_DIRNAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:                                     # pragma: no cover
        pass
    return d

# Where a job records the video it became. Beside the build, so deleting a
# production takes its reference with it.
REFERENCE_FILE = "published.json"

# A YouTube id is exactly 11 characters of this alphabet. Anchoring on that
# is what lets a bare id, a watch URL, a share link and a Studio URL all be
# accepted without accepting, say, a channel id by mistake.
_ID = r"[A-Za-z0-9_-]{11}"
_PATTERNS = (
    re.compile(rf"youtu\.be/({_ID})"),
    re.compile(rf"[?&]v=({_ID})"),
    # `video` before `v`, so a Studio URL - the one most likely to be
    # copied, straight from the tab the upload finished in - is matched
    # by its own path rather than falling through to the bare-id pattern.
    re.compile(rf"/(?:embed|shorts|live|video|v)/({_ID})"),
    re.compile(rf"^({_ID})$"),
)

# Column names as the warehouse export writes them, and the aliases seen in
# practice. The left-hand name is what the rest of the code uses.
_COLUMNS: dict[str, tuple[str, ...]] = {
    "video_id": ("video_id", "videoid", "video id", "id"),
    "title": ("title", "video_title"),
    "views": ("views", "sum(views)", "total_views"),
    "watch_minutes": ("watch_minutes", "watch_time_minutes",
                      "sum(watch_time_minutes)", "watch_time"),
    "likes": ("likes", "sum(likes)", "total_likes"),
    "subs_gained": ("subs_gained", "subscribers_gained",
                    "sum(subscribers_gained)", "subscribers"),
    "first_day": ("first_day", "min(date)", "first_date", "start_date"),
    "last_day": ("last_day", "max(date)", "last_date", "end_date"),
    "days_tracked": ("days_tracked", "count(distinct date)", "days"),
}
_NUMERIC = ("views", "watch_minutes", "likes", "subs_gained", "days_tracked")


@dataclass(frozen=True)
class VideoRow:
    """One video's measured totals, as exported. Never estimated."""

    video_id: str
    title: str = ""
    views: float = 0.0
    watch_minutes: float = 0.0
    likes: float = 0.0
    subs_gained: float = 0.0
    first_day: str = ""
    last_day: str = ""
    days_tracked: float = 0.0
    source_file: str = ""

    @property
    def window_days(self) -> int:
        """Days the export actually covers, or 0 when it does not say.

        Learning's `time_window` gate needs a span, and `SUM(views)` over
        all time does not carry one - which is why the export is asked for
        MIN(date) and MAX(date) as well.
        """
        a, b = _date(self.first_day), _date(self.last_day)
        if a is None or b is None:
            return int(self.days_tracked or 0)
        return max(0, int(round((b - a) / 86_400.0)) + 1)


@dataclass(frozen=True)
class Export:
    """Everything the export folder holds, and what could not be read."""

    rows: list[VideoRow] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def by_id(self, video_id: str) -> VideoRow | None:
        for r in self.rows:
            if r.video_id == video_id:
                return r
        return None


def _date(text: str) -> float | None:
    text = (text or "").strip()[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _number(text) -> float:
    """A spreadsheet's idea of a number: commas, currency, blanks, '1.2K'."""
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text or "").strip().replace(",", "").replace("%", "")
    if not s:
        return 0.0
    mult = {"k": 1_000, "m": 1_000_000}.get(s[-1:].lower())
    if mult and s[:-1].replace(".", "", 1).isdigit():
        s, = (s[:-1],)
    else:
        mult = 1
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def video_id_from(text: str) -> str | None:
    """The video id in a link, or a bare id, or None.

    None is returned rather than a best guess: a wrong id would attach one
    video's numbers to another video's decisions.
    """
    text = (text or "").strip()
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


# ------------------------------------------------------- the job's record

def reference_path(build_dir: str | Path) -> Path:
    return Path(build_dir) / REFERENCE_FILE


def save_reference(build_dir: str | Path, video_id: str,
                   url: str = "") -> dict:
    """Record which YouTube video this build became."""
    rec = {"video_id": video_id, "url": url, "recorded_at": time.time()}
    p = reference_path(build_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def load_reference(build_dir: str | Path) -> dict | None:
    """The recorded video, or None. A malformed file reads as absent."""
    p = reference_path(build_dir)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return rec if isinstance(rec, dict) and rec.get("video_id") else None


def clear_reference(build_dir: str | Path) -> bool:
    p = reference_path(build_dir)
    if not p.exists():
        return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


# ------------------------------------------------------------ the export

def _header_map(fieldnames) -> dict[str, str]:
    """Which CSV column feeds which field. Case and spacing are ignored."""
    seen = {}
    for raw in (fieldnames or []):
        key = (raw or "").strip().lower().replace("_", " ")
        for field_name, aliases in _COLUMNS.items():
            if field_name in seen:
                continue
            if key in {a.replace("_", " ") for a in aliases}:
                seen[field_name] = raw
    return seen


def read_export(folder: str | Path) -> Export:
    """Every row in every CSV in `folder`.

    A file without a `video_id` column is skipped WITH its name, not
    silently dropped: an export of the wrong query is the likeliest
    mistake here and the person has to be told which file it was.
    """
    root = Path(folder)
    if not root.is_dir():
        return Export()

    rows: list[VideoRow] = []
    files: list[str] = []
    skipped: list[str] = []
    for path in sorted(root.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                cols = _header_map(reader.fieldnames)
                if "video_id" not in cols:
                    have = ", ".join(reader.fieldnames or []) or "(no header)"
                    skipped.append(f"{path.name}: no video_id column ({have})")
                    continue
                files.append(path.name)
                for raw in reader:
                    vid = (raw.get(cols["video_id"]) or "").strip()
                    if not vid:
                        continue
                    vals = {}
                    for fname, col in cols.items():
                        if fname == "video_id":
                            continue
                        v = raw.get(col)
                        vals[fname] = (_number(v) if fname in _NUMERIC
                                       else (v or "").strip())
                    rows.append(VideoRow(video_id=vid,
                                         source_file=path.name, **vals))
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            skipped.append(f"{path.name}: {type(exc).__name__}: {exc}")

    # A video appearing in two exports is the same video; the later file
    # wins, because an export is a snapshot and the newer one is riper.
    merged: dict[str, VideoRow] = {}
    for r in rows:
        merged[r.video_id] = r
    return Export(rows=list(merged.values()), files=files, skipped=skipped)
