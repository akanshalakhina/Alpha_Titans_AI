"""
Mission Control: which agent is working now, and what it is doing.

WHAT THIS DOES
--------------
Turns the trace the graph already writes (one row per node attempt, with live
notes and tool calls) into a picture of the hierarchy in `teams.py`:

    Chief > teams > lead, workers, checker

for ONE job: each worker's state, the line it is working on right now, a live
activity feed, and each team checker's verdict.

WHAT IT DOES NOT DO
-------------------
* It writes nothing and runs nothing. It is a read of the trace and the
  artifact store, so it works while a job runs in another process (the VM
  worker, the CLI) exactly as it does in the web server's own.
* It invents nothing. A planned worker is "planned", never "working". A state
  is only ever one the trace supports.
* The checkers are DERIVED at read time. They report whether a team's result
  looks sound; they do not stop the run. Making a checker a gate that routes a
  retry is a later step, and the screen says so.

THE CHECKS
----------
Generic, for every team once all its built workers have finished:
  workers_ran      every built worker finished (an essential one failing is a
                   fail; an optional one failing or skipping is a warning)
  outputs_recorded each finished worker recorded an artifact
  not_degraded     no worker's artifact is marked degraded or fell back
Specific, where the artifact makes a real check possible:
  footage   clips were found and measured
  edit      the plan has segments
  render    the video file exists on this machine
  quality   the critic recorded a verdict

STALE ROWS
----------
A process killed mid-node leaves its trace row saying "running" for ever. A
node cannot legitimately run longer than its deadline, so a "running" row that
has been quiet longer than that is reported as interrupted, not working.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import teams as T

FEED_LIMIT = 300
_DEFAULT_DEADLINE_S = 3600.0
_ENDED = ("done", "failed", "stopped", "cancelled")


# ----------------------------------------------------------------- helpers

@functools.lru_cache(maxsize=1)
def _produces() -> dict[str, str]:
    """agent id -> artifact kind, from the registry (derived, not typed)."""
    try:
        from .base import agent_registry
        return {a["id"]: a["produces"] for a in agent_registry().values()}
    except Exception:                                  # noqa: BLE001
        return {}


def _deadline(node: str) -> float:
    try:
        from .graph import POLICIES
        t = POLICIES[node].timeout_s
        return float(t) if t else _DEFAULT_DEADLINE_S
    except Exception:                                  # noqa: BLE001
        return _DEFAULT_DEADLINE_S


def _last_activity(row: dict) -> float:
    """When this row last did anything: a tool call, a note, or its end.

    Notes carry no timestamp, so they cannot move this; tool calls and the
    end of the row can, which is enough to tell a quiet node from a dead one.
    """
    ts = float(row.get("started_at") or 0.0)
    for c in row.get("tool_calls") or []:
        try:
            ts = max(ts, float(row["started_at"]) + float(c.get("at") or 0))
        except (TypeError, ValueError, KeyError):
            continue
    if row.get("ended_at"):
        ts = max(ts, float(row["ended_at"]))
    return ts


def _end_of(row: dict, now: float) -> float:
    """When a row stopped counting: its end, or now if it is genuinely still
    running, or its last sign of life if it was killed and never closed."""
    if row.get("ended_at"):
        return float(row["ended_at"])
    return _last_activity(row) if _is_stale(row, now) else now


def _short(value: Any, n: int) -> str:
    s = str(value)
    return s if len(s) <= n else s[: n - 3] + "..."


def _tool_text(call: dict) -> str:
    name = str(call.get("tool", "tool"))
    bits = [f"{k}={_short(v, 40)}" for k, v in call.items()
            if k not in ("tool", "at")][:3]
    return name + (f" ({', '.join(bits)})" if bits else "")


def _doing(row: dict) -> str:
    """The line that says what a working agent is doing right now."""
    calls = row.get("tool_calls") or []
    if calls:
        return "tool: " + _tool_text(calls[-1])
    notes = row.get("notes") or []
    if notes:
        return _short(notes[-1], 140)
    return "starting"


def _art(store, job_id: str, kind: str) -> dict | None:
    if store is None:
        return None
    try:
        return store.get_raw(job_id, kind)
    except Exception:                                  # noqa: BLE001
        return None


# ------------------------------------------------------------- job status

def _is_stale(row: dict, now: float) -> bool:
    return (row.get("status") == "running"
            and now - _last_activity(row) > _deadline(row["node"]) + 60)


def job_status(by_node: dict[str, list[dict]], rows: list[dict],
               now: float, producing: bool | None) -> str:
    """queued | running | done | failed | stopped | cancelled."""
    if not rows:
        return "queued"
    if any("WorkCancelled" in " ".join(r.get("notes") or []) for r in rows):
        return "cancelled"
    running = [r for r in rows if r.get("status") == "running"]
    if running and (producing or any(not _is_stale(r, now) for r in running)):
        return "running"
    for w in (w for t in T.TEAMS for w in t.built_workers if w.essential):
        last = (by_node.get(w.node) or [None])[-1]
        if last and last.get("status") == "error":
            return "failed"
    final = (by_node.get("learning") or [None])[-1]
    if final and final.get("status") in ("ok", "error"):
        return "done"
    return "stopped" if not producing else "running"


# ------------------------------------------------------------ worker view

def _worker_view(w: T.Worker, rows: list[dict], now: float,
                 status: str) -> dict:
    base = {"key": w.key, "label": w.label, "does": w.does, "node": w.node,
            "built": w.built, "essential": w.essential}
    if not w.built:
        return {**base, "state": "planned"}
    if not rows:
        return {**base, "state": "waiting" if status in ("queued", "running")
                else "not_run"}

    last = rows[-1]
    st = last.get("status")
    if st == "running":
        state = "interrupted" if _is_stale(last, now) else "working"
    else:
        state = {"ok": "done", "error": "failed",
                 "skipped": "skipped"}.get(st, "done")

    started = min(float(r["started_at"]) for r in rows)
    wall = sum(_end_of(r, now) - float(r["started_at"]) for r in rows)
    fallbacks = [r["fallback_used"] for r in rows if r.get("fallback_used")]
    view = {
        **base, "state": state,
        "rounds": len({r.get("attempt") for r in rows}),
        "tries": len(rows),
        "started_at": started,
        "duration_s": round(wall, 2),
        "tools": sum(len(r.get("tool_calls") or []) for r in rows),
        "tokens": sum(int(r.get("tokens_in") or 0) + int(r.get("tokens_out")
                      or 0) for r in rows),
        "usd": round(sum(float(r.get("usd") or 0) for r in rows), 4),
        "fallback": fallbacks[-1] if fallbacks else None,
        "error": last.get("error") if state == "failed" else None,
        "outputs": len(last.get("output_artifacts") or []),
    }
    if state == "working":
        view["doing"] = _doing(last)
        view["quiet_s"] = round(max(0.0, now - _last_activity(last)), 1)
    return view


# ---------------------------------------------------------------- checker

def _check(name: str, state: str, detail: str) -> dict:
    return {"name": name, "state": state, "detail": detail}


def _generic_checks(workers: list[dict], store, job_id: str) -> list[dict]:
    built = [w for w in workers if w["built"]]
    out: list[dict] = []

    bad = [w for w in built if w["state"] in ("failed", "not_run",
                                              "interrupted")]
    soft = [w for w in built if w["state"] == "skipped"]
    if any(w["essential"] for w in bad):
        out.append(_check("workers_ran", "fail", "; ".join(
            f"{w['label']} {w['state'].replace('_', ' ')}"
            for w in bad if w["essential"])))
    elif bad or soft:
        out.append(_check("workers_ran", "warn", "; ".join(
            f"{w['label']} {w['state'].replace('_', ' ')}"
            for w in bad + soft)))
    else:
        out.append(_check("workers_ran", "pass",
                          f"all {len(built)} built workers finished"))

    empty = [w["label"] for w in built
             if w["state"] == "done" and not w.get("outputs")]
    out.append(_check("outputs_recorded", "fail" if empty else "pass",
                      f"no artifact from {', '.join(empty)}" if empty
                      else "every finished worker recorded an artifact"))

    produced = _produces()
    soft_notes = []
    for w in built:
        if w["state"] != "done":
            continue
        if w.get("fallback"):
            soft_notes.append(f"{w['label']} used a fallback "
                              f"({_short(w['fallback'], 50)})")
            continue
        kind = produced.get(_agent_id(w["key"]))
        art = _art(store, job_id, kind) if kind else None
        if art and art.get("degraded"):
            soft_notes.append(f"{w['label']} is marked degraded")
    out.append(_check("not_degraded", "warn" if soft_notes else "pass",
                      "; ".join(soft_notes) if soft_notes
                      else "no worker fell back or is degraded"))
    return out


def _agent_id(key: str) -> str:
    w = T.worker_for_node(key)
    return (w.agent_id if w and w.agent_id else key)


def _specific_checks(team_id: str, store, job_id: str) -> list[dict]:
    if store is None:
        return []
    if team_id == "footage":
        a = _art(store, job_id, "ClipAnalysis")
        clips = (a or {}).get("payload", {}).get("clips") \
            if isinstance((a or {}).get("payload"), dict) else None
        if not isinstance(clips, list):
            return [_check("clips_found", "fail", "no clip analysis stored")]
        usable = sum(1 for c in clips
                     if (c.get("usability") or {}).get("usable", True))
        return [_check("clips_found", "pass" if usable else "fail",
                       f"{len(clips)} clips measured, {usable} usable")]
    if team_id == "edit":
        a = _art(store, job_id, "EditPlan")
        p = (a or {}).get("payload")
        segs = p.get("segments") if isinstance(p, dict) else None
        if not segs:
            return [_check("plan_has_segments", "fail", "the edit plan "
                           "has no segments")]
        return [_check("plan_has_segments", "pass",
                       f"{len(segs)} segments, "
                       f"{float(p.get('total_duration') or 0):.1f}s of "
                       f"{float(p.get('target_seconds') or 0):.0f}s target")]
    if team_id == "render":
        a = _art(store, job_id, "RenderResult")
        p = (a or {}).get("payload")
        path = p.get("video_path") if isinstance(p, dict) else None
        if not path:
            return [_check("video_file_exists", "fail",
                           "no render result stored")]
        ok = Path(path).exists()
        return [_check("video_file_exists", "pass" if ok else "warn",
                       Path(path).name if ok else
                       f"{Path(path).name} not found on this machine")]
    if team_id == "quality":
        a = _art(store, job_id, "CriticResult")
        p = (a or {}).get("payload")
        if not isinstance(p, dict) or not p.get("verdict"):
            return [_check("verdict_recorded", "fail",
                           "the critic recorded no verdict")]
        v = str(p["verdict"])
        return [_check("verdict_recorded", "pass" if v == "PASS" else "warn",
                       f"{v}, score {float(p.get('overall_score') or 0):.0f}"
                       f"/100")]
    return []


def _checker_view(team: T.Team, workers: list[dict], store,
                  job_id: str) -> dict:
    built = [w for w in workers if w["built"]]
    base = {"label": team.checker_label}
    if not built:
        return {**base, "state": "planned", "checks": []}
    unfinished = [w for w in built if w["state"] in ("waiting", "working")]
    if unfinished:
        return {**base, "state": "pending", "checks": [],
                "detail": f"waiting for {len(unfinished)} worker(s)"}
    checks = _generic_checks(workers, store, job_id) + \
        _specific_checks(team.id, store, job_id)
    states = {c["state"] for c in checks}
    state = ("fail" if "fail" in states else
             "warn" if "warn" in states else "pass")
    return {**base, "state": state, "checks": checks}


# ------------------------------------------------------------- team view

def _team_state(workers: list[dict], status: str) -> str:
    built = [w for w in workers if w["built"]]
    states = [w["state"] for w in built]
    if not states:
        return "planned"
    if "working" in states:
        return "working"
    if all(s == "waiting" for s in states):
        return "waiting"
    if any(w["state"] == "failed" and w["essential"] for w in built):
        return "failed"
    if "interrupted" in states:
        return "interrupted"
    if all(s in ("done", "skipped") for s in states):
        return "done"
    if status in _ENDED:
        return "incomplete"
    return "working"              # between workers, the job is still going


def _lead_summary(team: T.Team, workers: list[dict], state: str,
                  checker: dict) -> str:
    built = [w for w in workers if w["built"]]
    planned = len(workers) - len(built)
    done = sum(1 for w in built if w["state"] == "done")
    if state == "planned":
        return f"{planned} planned worker(s), none built yet"
    if state == "waiting":
        return "waiting for its turn"
    if state == "working":
        active = [w["label"] for w in built if w["state"] == "working"]
        who = f"{', '.join(active)} working. " if active else ""
        return f"{who}{done} of {len(built)} workers done"
    parts = [f"{done} of {len(built)} workers done"]
    verdict = {"pass": "checker passed",
               "warn": "checker passed with warnings",
               "fail": "checker failed"}.get(checker["state"])
    if verdict:
        parts.append(verdict)
    if planned:
        parts.append(f"{planned} planned")
    return " · ".join(parts)


# ------------------------------------------------------------------ feed

def _feed(rows: list[dict]) -> list[dict]:
    ev: list[dict] = []
    for r in rows:
        node = r["node"]
        t = T.team_of_node(node)
        w = T.worker_for_node(node)
        base = {"team": t.id if t else "other",
                "worker": w.label if w else r.get("agent", node),
                "node": node}
        t0 = float(r["started_at"])
        rnd = int(r.get("attempt") or 1)
        ev.append({**base, "t": t0, "kind": "start",
                   "text": "started" + (f" (round {rnd})" if rnd > 1 else "")})
        for n in r.get("notes") or []:
            ev.append({**base, "t": t0, "kind": "note",
                       "text": _short(n, 200)})
        for c in r.get("tool_calls") or []:
            try:
                at = float(c.get("at") or 0)
            except (TypeError, ValueError):
                at = 0.0
            ev.append({**base, "t": t0 + at, "kind": "tool",
                       "text": _tool_text(c)})
        end = float(r.get("ended_at") or t0)
        if r.get("status") == "ok":
            ev.append({**base, "t": end, "kind": "ok", "text": "finished"})
        elif r.get("status") == "error":
            ev.append({**base, "t": end, "kind": "error",
                       "text": _short(r.get("error") or "failed", 200)})
        elif r.get("status") == "skipped":
            ev.append({**base, "t": end, "kind": "skip", "text": "skipped"})
    ev.sort(key=lambda e: e["t"])
    return ev


# -------------------------------------------------------------- snapshot

def build_snapshot(job_id: str, rows: list[dict], store=None, *,
                   now: float | None = None,
                   producing: bool | None = None) -> dict:
    """Everything the Mission Control screen draws, for one job."""
    now = time.time() if now is None else float(now)
    by_node: dict[str, list[dict]] = {}
    for r in rows:
        by_node.setdefault(r["node"], []).append(r)
    status = job_status(by_node, rows, now, producing)

    team_views: list[dict] = []
    lead_events: list[dict] = []
    now_line: dict | None = None
    for t in T.TEAMS:
        workers = [_worker_view(w, by_node.get(w.node, []) if w.node else [],
                                now, status) for w in t.workers]
        state = _team_state(workers, status)
        checker = _checker_view(t, workers, store, job_id)
        summary = _lead_summary(t, workers, state, checker)
        team_views.append({
            "id": t.id, "label": t.label, "phase": t.phase,
            "purpose": t.purpose, "state": state,
            "degraded": checker["state"] == "warn",
            "lead": {"label": t.lead_label, "state": state,
                     "summary": summary},
            "workers": workers, "checker": checker,
        })
        if state == "done":
            ends = [float(r["ended_at"]) for w in t.built_workers
                    for r in by_node.get(w.node, []) if r.get("ended_at")]
            if ends:
                lead_events.append({
                    "team": t.id, "worker": t.lead_label, "node": None,
                    "t": max(ends), "kind": "lead",
                    "text": f"{t.label} complete. Checker: "
                            f"{checker['state']}."})
        for w in workers:
            if w["state"] == "working":
                now_line = {"team": t.id, "team_label": t.label,
                            "worker": w["label"], "text": w.get("doing", "")}

    feed = sorted(_feed(rows) + lead_events, key=lambda e: e["t"])[-FEED_LIMIT:]
    built = [w for tv in team_views for w in tv["workers"] if w["built"]]
    starts = [float(r["started_at"]) for r in rows]
    ends = [_end_of(r, now) for r in rows]
    return {
        "job_id": job_id,
        "status": status,
        "chief": {**T.CHIEF, "state": status},
        "now": now_line,
        "teams": team_views,
        "feed": feed,
        "totals": {
            "built_workers": len(built),
            "done_workers": sum(1 for w in built if w["state"] == "done"),
            "planned_workers": sum(1 for tv in team_views
                                   for w in tv["workers"] if not w["built"]),
            "tool_calls": sum(w.get("tools", 0) for w in built),
            "tokens": sum(w.get("tokens", 0) for w in built),
            "usd": round(sum(w.get("usd", 0.0) for w in built), 4),
            "rounds": max([int(r.get("attempt") or 1) for r in rows] or [0]),
            "wall_s": round(max(ends) - min(starts), 1) if rows else 0.0,
        },
        "generated_at": now,
    }


def signature(rows: list[dict]) -> str:
    """A cheap fingerprint of the trace: changes whenever anything a viewer
    could see changes (a row opens or closes, a note or tool call lands)."""
    h = hashlib.sha1()
    for r in rows:
        h.update(json.dumps([
            r.get("trace_id"), r.get("status"), r.get("retry_count"),
            len(r.get("notes") or []), len(r.get("tool_calls") or []),
            len(r.get("output_artifacts") or []), r.get("fallback_used"),
        ], default=str).encode())
    return h.hexdigest()


__all__ = ["build_snapshot", "job_status", "signature"]
