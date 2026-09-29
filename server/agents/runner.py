"""
Runner - wires the graph up and drives it.

This is the Phase 1 entry point. It does NOT touch server/main.py or
server/jobs.py: the legacy build path keeps working exactly as before, and
this is a second, parallel way to build the same job folders.

    python -m agents.runner run    --src uploads/<id> [--job-id <id>]
    python -m agents.runner resume --job-id <id>
    python -m agents.runner trace  --job-id <id>
    python -m agents.runner show   --job-id <id>

Paths follow the existing layout (uploads/<job_id>, outputs/<job_id>) so a
video built this way is visible to the current UI and downloadable through the
current API without any change to either.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

from .artifacts import (ClipAnalysisSet, CriticResult, EditPlan, JobSettings,
                        JobState, RenderResult)
from .critic import DEFAULT_PASS_THRESHOLD
from .graph import (Runtime, build_graph, build_graph_async,
                    initial_state, register, unregister)
from .render_pool import MediaPool
from .store import ArtifactStore, resolve_job_db, shared_store
from .tools.llm_tools import LLMClient
from .tools.pipeline_bridge import config_mod
from .trace import TraceRecorder, shared_tracer

STATE_DB = "agent_state.db"
CHECKPOINT_DB = "checkpoints.db"


# ------------------------------------------------------------------ paths

def job_paths(job_id: str, root: Path | None = None) -> tuple[Path, Path]:
    """(src_dir, build_dir) for a job, in the existing layout."""
    if root is not None:
        return root / "uploads" / job_id, root / "outputs" / job_id
    return (Path(config_mod.UPLOADS_DIR) / job_id,
            Path(config_mod.OUTPUTS_DIR) / job_id)


def make_job(src: Path | None = None, job_id: str | None = None,
             root: Path | None = None, settings: JobSettings | None = None,
             max_revisions: int = 2) -> JobState:
    """Create (or adopt) a job folder pair and return its typed state."""
    job_id = job_id or uuid.uuid4().hex[:12]
    src_dir, build_dir = job_paths(job_id, root)
    src_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    if src is not None and Path(src).resolve() != src_dir.resolve():
        # Hard-link the clips where possible so a demo run costs no extra disk
        # and never mutates the operator's originals.
        from .tools.pipeline_bridge import build_video
        for p in sorted(Path(src).iterdir()):
            if p.suffix.lower() in build_video.VIDEO_EXT:
                target = src_dir / p.name
                if target.exists():
                    continue
                try:
                    import os
                    os.link(p, target)
                except OSError:
                    shutil.copy2(p, target)

    return JobState(job_id=job_id, src_dir=str(src_dir),
                    build_dir=str(build_dir),
                    settings=settings or JobSettings(),
                    max_revisions=max_revisions)


def build_runtime(job: JobState, llm: LLMClient | None = None,
                  pool: MediaPool | None = None,
                  root: Path | None = None) -> Runtime:
    # Writes always go to the one shared store, resolved from the job's own
    # build folder so it always sits beside the job folders - not wherever
    # process config happens to point. Reads of OLD jobs may still come off
    # a per-job database; see store.resolve_job_db.
    outputs = Path(job.build_dir).resolve().parent
    store = shared_store(outputs=outputs)
    tracer = shared_tracer(outputs=outputs)
    rt = Runtime(job=job, store=store, tracer=tracer,
                 pool=pool or MediaPool(name=job.job_id[:8]),
                 llm=llm or LLMClient())
    register(rt)
    return rt


# -------------------------------------------------------------- running

def run_job(job: JobState, llm: LLMClient | None = None,
            pass_threshold: float = DEFAULT_PASS_THRESHOLD,
            pool: MediaPool | None = None,
            on_line=None, recursion_limit: int = 50,
            root: Path | None = None) -> dict:
    """Start a job from scratch. Returns the final graph state."""
    rt = build_runtime(job, llm=llm, pool=pool, root=root)
    if on_line:
        rt.log = _TeeLog(on_line)
    cfg = {"configurable": {"thread_id": job.job_id},
           "recursion_limit": recursion_limit}
    try:
        return _drive(Path(job.build_dir) / CHECKPOINT_DB,
                      initial_state(job, pass_threshold), cfg)
    finally:
        # success, escalation or crash - all three should be visible in the
        # existing app rather than only the happy path
        _auto_project(job, root)


def resume_job(job_id: str, root: Path | None = None,
               llm: LLMClient | None = None, pool: MediaPool | None = None,
               settings: JobSettings | None = None,
               max_revisions: int = 2, on_line=None,
               recursion_limit: int = 50) -> dict:
    """Continue a job from its last checkpoint.

    Passing None as the input is what tells LangGraph to resume the thread
    rather than start it: the checkpoint holds the state, and the nodes that
    already completed are not re-run.
    """
    src_dir, build_dir = job_paths(job_id, root)
    if not (build_dir / CHECKPOINT_DB).exists():
        raise FileNotFoundError(
            f"no checkpoint for job {job_id} at {build_dir / CHECKPOINT_DB}")

    job = JobState(job_id=job_id, src_dir=str(src_dir),
                   build_dir=str(build_dir),
                   settings=settings or _restore_settings(build_dir),
                   max_revisions=max_revisions)
    rt = build_runtime(job, llm=llm, pool=pool, root=root)
    if on_line:
        rt.log = _TeeLog(on_line)
    cfg = {"configurable": {"thread_id": job_id},
           "recursion_limit": recursion_limit}
    try:
        # None as the input is what tells LangGraph to resume the thread
        return _drive(build_dir / CHECKPOINT_DB, None, cfg)
    finally:
        _auto_project(job, root)


def _auto_project(job: JobState, root: Path | None) -> None:
    """Publish the job into the legacy compatibility layer.

    Deliberately swallows everything. A build that produced a verified video
    must never be reported as failed because a derived projection could not be
    written - job.json is a convenience, the artifacts are the record.
    """
    from . import compat
    try:
        cancelled = compat.cancel_requested(job.job_id, root=root)
    except Exception:                                  # noqa: BLE001
        cancelled = False
    try:
        compat.project(job.job_id, root=root)
    except compat.NotAnAgentJob:
        # stopped before any render finished - cancelled during analysis, or
        # crashed on the first attempt. There is nothing to publish, but the
        # last heartbeat left the job reading "running", and a build that
        # never ends is worse than one reported as stopped.
        try:
            compat.finalize(
                job.job_id, root=root,
                status="cancelled" if cancelled else "error",
                error=("the build was cancelled before a video was produced"
                       if cancelled else
                       "the agent build stopped before producing a video"))
        except Exception:                              # noqa: BLE001
            pass
    except Exception:                                  # noqa: BLE001
        pass
    try:
        compat.clear_cancel_request(job.job_id, root=root)
    except Exception:                                  # noqa: BLE001
        pass


def _drive(checkpoint: Path, state, cfg: dict) -> dict:
    """Run the graph to completion under an event loop.

    `node_render` is a coroutine that awaits the scheduler, so the graph must
    be driven with `ainvoke`. Callers stay synchronous - this owns the loop -
    so every existing entry point and test keeps working unchanged.
    """
    async def go():
        async with build_graph_async(checkpoint) as graph:
            return await graph.ainvoke(state, cfg)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(go())
    # already inside a loop (a notebook, an async server): run in its own
    # thread so we never try to nest one event loop inside another
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(go())).result()


def peek_state(job_id: str, root: Path | None = None) -> dict:
    """The checkpointed state without running anything. Used by the resume
    test to prove work survived a kill."""
    _, build_dir = job_paths(job_id, root)
    graph, conn = build_graph(build_dir / CHECKPOINT_DB)
    try:
        snap = graph.get_state({"configurable": {"thread_id": job_id}})
        return {"values": dict(snap.values or {}),
                "next": list(snap.next or []),
                "created_at": getattr(snap, "created_at", None)}
    finally:
        conn.close()


def _restore_settings(build_dir: Path) -> JobSettings:
    p = build_dir / "agent_settings.json"
    if p.exists():
        try:
            return JobSettings(**json.loads(p.read_text()))
        except Exception:
            pass
    return JobSettings()


def save_settings(job: JobState) -> None:
    (Path(job.build_dir) / "agent_settings.json").write_text(
        job.settings.model_dump_json(indent=2))


class _TeeLog(list):
    """A log list that also forwards every line to a callback."""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def append(self, item):
        super().append(item)
        try:
            self._sink(item)
        except Exception:
            pass


# ------------------------------------------------------------------- cli

def _cli() -> int:
    ap = argparse.ArgumentParser(prog="agents.runner",
                                 description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build a video through the agent graph")
    r.add_argument("--src", required=True, help="folder of source clips")
    r.add_argument("--job-id")
    r.add_argument("--root", help="override the app root for uploads/outputs")
    r.add_argument("--threshold", type=float, default=DEFAULT_PASS_THRESHOLD)
    r.add_argument("--max-revisions", type=int, default=2)
    r.add_argument("--orientation", default="auto",
                   choices=["auto", "horizontal", "vertical"])
    r.add_argument("--resolution", default="1080p",
                   choices=["720p", "1080p", "1440p", "4k"])
    r.add_argument("--quality", default="high",
                   choices=["draft", "standard", "high", "master"])
    r.add_argument("--target", type=float, default=None)
    r.add_argument("--title", default="ALPHA_TITANS")
    r.add_argument("--force-open", default=None,
                   help="pin a clip to open the cut (used to demonstrate a "
                        "genuinely weak hook)")

    s = sub.add_parser("resume", help="continue from the last checkpoint")
    s.add_argument("--job-id", required=True)
    s.add_argument("--root")

    t = sub.add_parser("trace", help="print the execution trace")
    t.add_argument("--job-id", required=True)
    t.add_argument("--root")
    t.add_argument("--json", action="store_true")

    w = sub.add_parser("show", help="print artifacts for a job")
    w.add_argument("--job-id", required=True)
    w.add_argument("--root")
    w.add_argument("--kind", default=None)

    p = sub.add_parser("state", help="print the checkpointed graph state")
    p.add_argument("--job-id", required=True)
    p.add_argument("--root")

    args = ap.parse_args()
    root = Path(args.root).resolve() if getattr(args, "root", None) else None

    if args.cmd == "run":
        settings = JobSettings(
            orientation=args.orientation, resolution=args.resolution,
            quality=args.quality, target_seconds=args.target,
            title=args.title, force_opening_clip=args.force_open)
        job = make_job(Path(args.src), args.job_id, root, settings,
                       max_revisions=args.max_revisions)
        save_settings(job)
        print(f"job {job.job_id}")
        print(f"  src   {job.src_dir}")
        print(f"  build {job.build_dir}")
        llm = LLMClient()
        print(f"  llm   {llm.describe()}")
        print()
        final = run_job(job, llm=llm, pass_threshold=args.threshold,
                        on_line=lambda ln: print("   " + ln))
        _print_final(final, job.job_id, root)
        return 0 if final.get("status") == "done" else 1

    if args.cmd == "resume":
        print(f"resuming job {args.job_id}")
        final = resume_job(args.job_id, root,
                           on_line=lambda ln: print("   " + ln))
        _print_final(final, args.job_id, root)
        return 0 if final.get("status") == "done" else 1

    if args.cmd == "trace":
        db, _source = resolve_job_db(args.job_id, root)
        tracer = TraceRecorder(db)
        if args.json:
            print(json.dumps(tracer.for_job(args.job_id), indent=2))
        else:
            print(tracer.render_table(args.job_id))
            print()
            print("totals:", json.dumps(tracer.totals(args.job_id)))
        return 0

    if args.cmd == "show":
        db, _source = resolve_job_db(args.job_id, root)
        store = ArtifactStore(db)
        kinds = {"ClipAnalysis": ClipAnalysisSet, "EditPlan": EditPlan,
                 "RenderResult": RenderResult, "CriticResult": CriticResult}
        for row in store.manifest(args.job_id):
            if args.kind and row["kind"] != args.kind:
                continue
            print("-" * 78)
            print(f"{row['kind']} rev{row['revision']}  by {row['produced_by']}"
                  f"  conf={row['confidence']:.2f}"
                  f"  degraded={bool(row['degraded'])}"
                  f"  fallback={row['fallback_used'] or '-'}")
            model = kinds.get(row["kind"])
            if model:
                art = store.get(args.job_id, row["kind"], model,
                                revision=row["revision"])
                if art:
                    print(json.dumps(art.payload.model_dump(mode="json"),
                                     indent=2, default=str)[:4000])
                    if art.decisions:
                        print("  decisions:")
                        for d in art.decisions:
                            print(f"    [{d.decided_by}] {d.what}: {d.chose}")
                            print(f"        why: {d.why[:200]}")
        return 0

    if args.cmd == "state":
        print(json.dumps(peek_state(args.job_id, root), indent=2, default=str))
        return 0

    return 2                                           # pragma: no cover


def _print_final(final: dict, job_id: str, root: Path | None) -> None:
    print()
    print("=" * 70)
    print(f"status          {final.get('status')}")
    print(f"attempts        {final.get('attempt')}")
    print(f"revisions       {final.get('revision_count')}")
    print(f"final verdict   {final.get('verdict')}")
    if final.get("error"):
        print(f"error           {final['error']}")
    for row in final.get("history", []):
        print(f"  attempt {row['attempt']}: {row['verdict']} "
              f"score={row['score']} cache_hits={row['cache_hits']}")
        print(f"    {row['video']}")
    print("=" * 70)


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(_cli())
