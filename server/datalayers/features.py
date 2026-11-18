"""Feature Analyst and Opening Reviewer: what each published video IS.

WHY
---
The warehouse says how each video did - views, watch time - and its title.
It says nothing about the video itself, so a lesson could only ever be
"Shorts do better" or "this kind of title does better". To learn WHY, the
videos have to be measured: how long, how lively the first three seconds,
how many shots, and what a viewer actually sees at the start.

WHERE THE FILES COME FROM
-------------------------
The channel's published MP4s, downloaded from YouTube Studio by the channel
owner, dropped into `data/channel_videos/`. Studio names a download after the
video's title, so a file is matched to its warehouse row by its title (or by
the 11-character video id, if the name carries one). A file that matches no
row, or more than one, is reported and left out - never guessed at.

WHAT IS MEASURED
----------------
Feature Analyst (code, ffmpeg - the same measurements the Critic makes on
our own renders):
  duration_s, vertical, opening_energy (motion in the first 3 s against the
  video's own median), shots and average shot length (scene cuts),
  cuts_first5 (cuts in the first five seconds), loudness.
Opening Reviewer (the vision model, one request per video, two strips of
frames - the opening, and five moments across the whole video):
  result_first (is the finished build on screen in the first 2 s),
  hands_cover (hands or arms hide the subject in the opening),
  text_on_screen, framing (close / medium / wide), and a short description.

A video is measured once; a file that has not changed is not measured again.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import time
from pathlib import Path

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
_ID = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")

SCHEMA = """
CREATE TABLE IF NOT EXISTS video_features (
  video_id TEXT PRIMARY KEY, file TEXT, file_sig TEXT, duration_s REAL,
  vertical INTEGER, opening_energy REAL, shots INTEGER, avg_shot_s REAL,
  cuts_first5 INTEGER, lufs REAL, measured_at REAL);
CREATE TABLE IF NOT EXISTS ai_video_looks (
  video_id TEXT PRIMARY KEY, file_sig TEXT, result_first INTEGER,
  hands_cover INTEGER, text_on_screen INTEGER, framing TEXT, opening TEXT,
  looked_by TEXT, looked_at REAL);
"""

LOOK_PROMPT = """You are looking at a published YouTube video about building \
with LEGO, titled: "{title}".

Image 1 shows its first two seconds (three moments, left to right).
Image 2 shows five moments spread across the whole video.

Answer with JSON only:
{{"result_first": true or false - is the finished or working build clearly \
shown within the first two seconds (image 1)?,
 "hands_cover": true or false - in image 1, do hands or arms hide most of \
the build?,
 "text_on_screen": true or false - is there text or a caption on screen in \
image 1?,
 "framing": "close" or "medium" or "wide" - how the build is framed in most \
of image 2,
 "opening": "what the first two seconds show, in at most 12 words"}}"""


def channel_dir() -> Path:
    """`data/channel_videos`, or AT_CHANNEL_VIDEOS when set (tests)."""
    import config
    d = Path(os.environ.get("AT_CHANNEL_VIDEOS")
             or Path(config.APP_ROOT) / "data" / "channel_videos")
    d.mkdir(parents=True, exist_ok=True)
    return d


def norm(text: str) -> str:
    """A title or file name reduced to lowercase words, so the full-width
    bar YouTube Studio puts in file names matches the title's `|`."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).split())


def match_files(folder: Path, videos: list[tuple[str, str]]
                ) -> tuple[dict[str, Path], list[str]]:
    """{video_id: file} for every file that names exactly one video, and a
    note for every file that does not."""
    ids = {v for v, _ in videos}
    by_title: dict[str, list[str]] = {}
    for vid, title in videos:
        by_title.setdefault(norm(title), []).append(vid)
    found: dict[str, Path] = {}
    notes: list[str] = []
    for f in sorted(Path(folder).iterdir()) if Path(folder).is_dir() else []:
        if f.suffix.lower() not in VIDEO_EXT:
            continue
        hit = [m for m in _ID.findall(f.stem) if m in ids]
        if len(hit) == 1:
            found[hit[0]] = f
            continue
        cands = by_title.get(norm(f.stem), [])
        if len(cands) == 1:
            found[cands[0]] = f
        elif len(cands) > 1:
            notes.append(f"{f.name}: matches {len(cands)} videos with that "
                         f"title; add the video id to the file name")
        else:
            notes.append(f"{f.name}: no video in the warehouse has this title")
    return found, notes


def _sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def scene_cuts(path: Path, duration: float | None = None,
               timeout: float = 600) -> list[float] | None:
    """Times (s) where the picture changes to a new shot.

    Frames a quarter-second apart are compared, not neighbouring frames:
    a crossfade spreads its change over several frames, so a per-frame
    detector missed five of the six cuts in a 26-second render (check1117,
    17 Nov 2026) while this one counted all seven shots. A cut is a jump at
    least four times the video's typical change, a local peak, and a second
    or more from the previous one; the last second (fade-out) is ignored."""
    import statistics
    import tempfile
    ff = os.environ.get("AT_FFMPEG") or "ffmpeg"
    with tempfile.TemporaryDirectory(prefix="at_cuts_") as td:
        stats = Path(td) / "diff.txt"
        esc = str(stats).replace("\\", "/").replace(":", r"\:")
        try:
            out = subprocess.run(
                [ff, "-nostdin", "-v", "error", "-i", str(path), "-vf",
                 "scale=96:-2,fps=4,format=gray,tblend=all_mode=difference,"
                 f"signalstats,metadata=print:file='{esc}'",
                 "-an", "-f", "null", "-"],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not stats.exists():
            return None
        txt = stats.read_text(errors="replace")
    times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", txt)]
    vals = [float(v) for v in re.findall(r"signalstats\.YAVG=([0-9.]+)", txt)]
    if not vals:
        return []
    med = statistics.median(vals)
    end = (duration or (times[-1] if times else 0)) - 1.0
    cuts, last = [], -9.0
    for i, (t, v) in enumerate(zip(times, vals)):
        if t >= end:
            break
        if (v >= max(4 * med, 5.0) and v == max(vals[max(0, i - 1):i + 2])
                and t - last >= 1.0):
            cuts.append(round(t, 2))
            last = t
    return cuts


def measure(path: Path) -> dict | None:
    """Feature Analyst: the measurable facts of one video, or None."""
    from agents.tools import verify_tools as VT
    try:
        probe = VT.probe_final(path)
    except Exception:                                       # noqa: BLE001
        return None
    dur = float(probe.get("duration") or 0)
    if dur <= 0:
        return None
    prof = VT.output_motion_profile(path, 10.0, 3.0)
    energy = None
    if prof.get("measured") and prof.get("overall_median"):
        energy = round(prof["hook_mean"] / prof["overall_median"], 2)
    cuts = scene_cuts(path, dur) or []
    shots = len(cuts) + 1
    try:
        lufs = VT.loudness(path)
    except Exception:                                       # noqa: BLE001
        lufs = None
    w, h = probe.get("width") or 0, probe.get("height") or 0
    return {"duration_s": round(dur, 2), "vertical": int(h > w),
            "opening_energy": energy, "shots": shots,
            "avg_shot_s": round(dur / shots, 2),
            "cuts_first5": sum(1 for c in cuts if c < 5.0), "lufs": lufs}


def _strip(path: Path, times: list[float]) -> bytes | None:
    from agents.tools import shot_check as SC
    return SC.strip_jpeg(path, times)


def look(llm, path: Path, title: str, duration: float) -> dict | None:
    """Opening Reviewer: what a viewer sees, from the vision model, or None
    (no model, a refusal, an unreadable answer). Never raises."""
    if llm is None or not llm.available():
        return None
    opening = _strip(path, [0.3, 1.0, 1.8])
    overall = _strip(path, [duration * k for k in (0.1, 0.3, 0.5, 0.7, 0.9)])
    if not opening or not overall:
        return None
    try:
        raw, reply = llm.complete_json(
            LOOK_PROMPT.format(title=(title or "")[:150]), max_tokens=400,
            images=[base64.b64encode(opening).decode(),
                    base64.b64encode(overall).decode()])
    except Exception:                                       # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None

    def flag(k):
        v = raw.get(k)
        return int(v) if isinstance(v, bool) else None

    framing = str(raw.get("framing") or "").strip().lower()
    return {"result_first": flag("result_first"),
            "hands_cover": flag("hands_cover"),
            "text_on_screen": flag("text_on_screen"),
            "framing": framing if framing in ("close", "medium", "wide") else None,
            "opening": str(raw.get("opening") or "")[:120],
            "looked_by": ((reply.meta or {}).get("model") if reply else "") or "vision"}


def analyse_features(db, folder: Path | None = None) -> list[dict]:
    """Feature Analyst's step. Measures every matched video once."""
    db.executescript(SCHEMA)
    folder = Path(folder) if folder else channel_dir()
    videos = [(r["video_id"], r["title"]) for r in db.execute(
        "SELECT video_id, title FROM staging_video_day7")]
    found, notes = match_files(folder, videos)
    done = {r["video_id"]: r["file_sig"] for r in db.execute(
        "SELECT video_id, file_sig FROM video_features")}
    measured, failed = 0, []
    for vid, path in found.items():
        sig = _sig(path)
        if done.get(vid) == sig:
            continue
        m = measure(path)
        if m is None:
            failed.append(path.name)
            continue
        db.execute("INSERT OR REPLACE INTO video_features VALUES "
                   "(?,?,?,?,?,?,?,?,?,?,?)",
                   (vid, path.name, sig, m["duration_s"], m["vertical"],
                    m["opening_energy"], m["shots"], m["avg_shot_s"],
                    m["cuts_first5"], m["lufs"], time.time()))
        measured += 1
    db.commit()
    have = db.execute("SELECT COUNT(*) FROM video_features").fetchone()[0]
    return [
        {"name": "files_matched", "ok": True,
         "detail": (f"{len(found)} of {len(videos)} video(s) have a file in "
                    f"{folder.name}/" if found else
                    f"no video files yet: download them from YouTube Studio "
                    f"into data/{folder.name}/")
                   + (f"; {len(notes)} file(s) not matched" if notes else "")},
        {"name": "measured", "ok": not failed,
         "detail": f"{measured} newly measured, {have} in total"
                   + (f"; could not read {', '.join(failed)}" if failed else "")},
    ] + [{"name": "unmatched_file", "ok": True, "detail": n} for n in notes[:5]]


def review_openings(db, llm, folder: Path | None = None) -> list[dict]:
    """Opening Reviewer's step: the vision model looks once at each video."""
    db.executescript(SCHEMA)
    folder = Path(folder) if folder else channel_dir()
    rows = db.execute(
        "SELECT f.video_id, f.file, f.file_sig, f.duration_s, d.title "
        "FROM video_features f JOIN staging_video_day7 d USING (video_id)"
    ).fetchall()
    seen = {r["video_id"]: r["file_sig"] for r in db.execute(
        "SELECT video_id, file_sig FROM ai_video_looks")}
    looked, missed = 0, 0
    for r in rows:
        if seen.get(r["video_id"]) == r["file_sig"]:
            continue
        got = look(llm, folder / r["file"], r["title"], r["duration_s"])
        if got is None:
            missed += 1
            continue
        db.execute("INSERT OR REPLACE INTO ai_video_looks VALUES "
                   "(?,?,?,?,?,?,?,?,?)",
                   (r["video_id"], r["file_sig"], got["result_first"],
                    got["hands_cover"], got["text_on_screen"], got["framing"],
                    got["opening"], got["looked_by"], time.time()))
        looked += 1
    db.commit()
    have = db.execute("SELECT COUNT(*) FROM ai_video_looks").fetchone()[0]
    if not rows:
        detail = "nothing to look at until videos are measured"
    elif llm is None or not llm.available():
        detail = "no vision model configured; openings not reviewed"
    else:
        detail = f"{looked} newly reviewed, {have} in total" + (
            f"; {missed} could not be reviewed (refusal or unreadable answer)"
            if missed else "")
    return [{"name": "openings_reviewed", "ok": True, "detail": detail}]


__all__ = ["SCHEMA", "analyse_features", "channel_dir", "look", "match_files",
           "measure", "norm", "review_openings", "scene_cuts"]
