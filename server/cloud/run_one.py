"""Run the EXISTING pipeline for one cloud job, in its own process.

    python -m cloud.run_one --job-id <id> --spec spec.json --result out.json

The worker launches this as a subprocess so that a crash, a hang or a
deadline kills the pipeline and leaves the worker standing to report it.
Nothing here decides anything about the edit. It calls the same three
functions `/api/v2/jobs/{id}/produce` calls - `make_job`, `save_settings`,
`run_job` - and `resume_job` when a checkpoint says the graph got part of the
way before. Then it reports which files that run delivered, using the
existing app's own rule for which video that is.

RESUMING
--------
`checkpoints.db` exists for this job when an earlier attempt got some way.
LangGraph then knows which nodes finished. `resume_job` continues from the
first unfinished one, so a retry after the worker died at `render` does not
re-run vision, story or the director; and anything it does re-run hits the
`inputs_hash` artifact cache and the sha256 measurement cache. A checkpoint
with nothing left to run means the graph finished and only the upload
failed, so nothing is run at all.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

from .paths import safe_name


def _deliverables(job_id: str, build: Path) -> dict:
    from agents.artifacts import CriticResult, RenderResult, Thumbnail
    from agents.compat import _project_output
    from agents.store import shared_store

    store = shared_store(outputs=build.parent)
    render = store.get(job_id, "RenderResult", RenderResult)
    critic = store.get(job_id, "CriticResult", CriticResult)
    thumb = store.get(job_id, "Thumbnail", Thumbnail)
    rp = render.payload if render else None
    attempts = len(store.all_revisions(job_id, "RenderResult", RenderResult))
    meta = _project_output(build, rp, attempts,
                           thumb.payload if thumb else None) or {}

    video = None
    if meta.get("path"):
        p = Path(meta["path"])
        video = p if p.exists() else build / Path(meta["path"]).name
    if (video is None or not video.exists()) and rp is not None:
        video = Path(rp.video_path)
    thumb_file = build / meta["thumbnail"] if meta.get("thumbnail") else None
    return {
        "video_path": str(video) if video and video.exists() else None,
        "thumbnail_path": (str(thumb_file) if thumb_file and
                           thumb_file.exists() else None),
        "verdict": critic.payload.verdict if critic else None,
        "score": critic.payload.overall_score if critic else None,
        "attempts": attempts,
    }


# Next steps the Publishing agent writes for the LOCAL app, which are not
# true of a cloud job: the video has left the machine, and `/api/jobs` is
# not served by the control plane.
_LOCAL_ONLY = ("Review the video in the app", "Download it from /api/")
_THUMB_STEP = re.compile(r"^Upload (\S+) as its custom thumbnail$")


def _cloud_steps(job_id: str, steps: list[str],
                 thumb: str | None) -> list[str]:
    out = ["Watch it on the cloud studio page before posting it"]
    for s in steps:
        if s.startswith(_LOCAL_ONLY):
            continue
        if thumb and _THUMB_STEP.match(s):
            # the name it has in the browser's downloads, not on the worker
            s = (f"Upload {job_id}_{safe_name(Path(thumb).name)} (it "
                 "downloads with the video) as its custom thumbnail")
        out.append(s)
    return out


def _youtube_sheet(job_id: str, build: Path,
                   thumb: str | None = None) -> Path | None:
    """`youtube.txt`: what a person needs beside the video to post it -
    the title, the alternatives that also passed, YouTube's own limits
    checked against the actual files, and what is left to do by hand.

    Read from the Title and Publishing agents' artifacts; nothing is
    written here that those agents did not decide or measure, except that
    steps only true of the local app are left out (`_LOCAL_ONLY`). Uploading
    stays the operator's step: this project never touches a channel.
    """
    from agents.artifacts import Publishing, Title
    from agents.store import shared_store

    store = shared_store(outputs=build.parent)
    title = store.get(job_id, "Title", Title)
    pub = store.get(job_id, "Publishing", Publishing)
    if title is None and pub is None:
        return None
    lines = ["YOUTUBE UPLOAD SHEET", ""]
    t = (pub.payload.title if pub else "") or (
        title.payload.text if title else "")
    lines += ["Title", f"  {t or '(none - write your own)'}", ""]
    if title and title.payload.alternatives:
        lines += ["Other titles that also passed"]
        lines += [f"  - {a}" for a in title.payload.alternatives[:5]]
        lines += [""]
    if pub:
        p = pub.payload
        lines += [f"Status: {p.status}   (recommended visibility: "
                  f"{p.recommended_visibility})", ""]
        if p.checks:
            lines += ["Checked against YouTube's limits"]
            for c in p.checks:
                bits = [c.value, f"limit {c.limit}" if c.limit else ""]
                extra = ", ".join(b for b in bits if b)
                lines.append(f"  [{'ok' if c.passed else 'FAIL'}] {c.check}"
                             + (f"  ({extra})" if extra else ""))
            lines += [""]
        steps = _cloud_steps(job_id, p.next_steps, thumb)
        lines += ["Next steps"] + [f"  - {s}" for s in steps]
    out = build / "youtube.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run(job_id: str, spec: dict) -> dict:
    from agents import runner
    from agents.artifacts import JobSettings

    settings = JobSettings(**(spec.get("settings") or {}))
    _src, build = runner.job_paths(job_id)
    job = runner.make_job(src=None, job_id=job_id, settings=settings,
                          max_revisions=int(spec.get("max_revisions", 2)))
    runner.save_settings(job)

    mode = "run"
    if (build / runner.CHECKPOINT_DB).exists():
        state = runner.peek_state(job_id)
        if state.get("next"):
            mode = "resume"
        elif state.get("values"):
            mode = "already_done"
    if mode == "run":
        runner.run_job(job)
    elif mode == "resume":
        runner.resume_job(job_id, settings=settings)

    out = _deliverables(job_id, build)
    try:
        _youtube_sheet(job_id, build, out.get("thumbnail_path"))
    except Exception:                                 # noqa: BLE001
        pass                         # a nicety: never fail a finished render
    out["mode"] = mode
    out["build_dir"] = str(build)
    out["ok"] = bool(out["video_path"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cloud.run_one")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--result", required=True)
    a = ap.parse_args(argv)

    from .paths import check_job_id
    check_job_id(a.job_id)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    try:
        out = run(a.job_id, spec)
    except Exception as exc:                          # noqa: BLE001
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-3000:]}
    Path(a.result).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if out.get("ok") else 2


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(main())
