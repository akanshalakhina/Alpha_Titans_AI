"""
Execution tracing.

The point of this module is evidential: after a run it must be possible to
prove which agents executed, in what order, how long each took, which tools
they called, whether a fallback fired, whether the cache was hit, and what the
LLM cost. "The system says it ran the agent" is not evidence; a persisted
trace with input and output artifact ids is.

Traces are written to SQLite next to the LangGraph checkpoints, so a trace
survives the process being killed - which is exactly the scenario the
checkpoint/resume test exercises.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .artifacts import new_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    node            TEXT NOT NULL,
    agent           TEXT NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 1,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    status          TEXT NOT NULL,
    input_artifacts TEXT NOT NULL DEFAULT '[]',
    output_artifacts TEXT NOT NULL DEFAULT '[]',
    tool_calls      TEXT NOT NULL DEFAULT '[]',
    error           TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    usd             REAL NOT NULL DEFAULT 0.0,
    fallback_used   TEXT,
    cache_hits      INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS traces_job ON traces(job_id, seq);
"""


def shared_tracer(root: Path | None = None,
                  outputs: Path | None = None) -> "TraceRecorder":
    """The trace half of the one shared store: outputs/_agents/state.db.

    Traces live in the same file as artifacts, as they always have - the only
    change is that the file is now shared rather than per job. `traces` was
    already keyed on job_id with a (job_id, seq) index, so nothing about
    isolation changes.
    """
    from .store import shared_db_path
    return TraceRecorder(shared_db_path(root, outputs))


class TraceRecorder:
    """Append-only trace store. One row per node execution attempt."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        # see the note in store.ArtifactStore._connect - one shared file means
        # concurrent jobs contend, which per-job files never had to
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
        except sqlite3.DatabaseError:            # pragma: no cover
            pass
        return con

    def jobs(self) -> list[str]:
        """Every job with trace rows here."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT job_id, MAX(started_at) AS last FROM traces"
                " GROUP BY job_id ORDER BY last DESC").fetchall()
        return [r["job_id"] for r in rows]

    def purge_job(self, job_id: str) -> int:
        with self._lock, self._connect() as con:
            cur = con.execute("DELETE FROM traces WHERE job_id = ?", (job_id,))
            con.commit()
            return cur.rowcount

    def _next_seq(self, job_id: str) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM traces WHERE job_id = ?",
                (job_id,)).fetchone()
        return int(row["m"]) + 1

    @contextmanager
    def node(self, job_id: str, node: str, agent: str, attempt: int = 1,
             input_artifacts: list[str] | None = None):
        """Wrap one node execution. Records start, end and outcome.

        Records a terminal row whether the node succeeds or raises, which is
        the case that matters: an untraced failure is indistinguishable from a
        node that never ran.
        """
        span = TraceSpan(
            recorder=self, trace_id=new_id("tr"), job_id=job_id, node=node,
            agent=agent, attempt=attempt,
            input_artifacts=list(input_artifacts or []),
            seq=self._next_seq(job_id),
        )
        span._open()
        try:
            yield span
        except Exception as exc:
            span.fail(f"{type(exc).__name__}: {exc}")
            raise
        else:
            if span.status == "running":
                span.ok()

    # ------------------------------------------------------------- reads

    def for_job(self, job_id: str) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM traces WHERE job_id = ? ORDER BY seq", (job_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("input_artifacts", "output_artifacts", "tool_calls", "notes"):
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, json.JSONDecodeError):
                    d[k] = []
            out.append(d)
        return out

    def render_table(self, job_id: str) -> str:
        rows = self.for_job(job_id)
        if not rows:
            return "(no trace rows)"
        lines = [
            "%-4s %-18s %-18s %-4s %-9s %8s %6s %7s %6s %s"
            % ("seq", "node", "agent", "att", "status", "wall_s",
               "tools", "tokens", "cache", "fallback"),
            "-" * 108,
        ]
        for r in rows:
            wall = (r["ended_at"] - r["started_at"]) if r["ended_at"] else 0.0
            lines.append(
                "%-4d %-18s %-18s %-4d %-9s %8.2f %6d %7d %6d %s"
                % (r["seq"], r["node"][:18], r["agent"][:18], r["attempt"],
                   r["status"], wall, len(r["tool_calls"]),
                   r["tokens_in"] + r["tokens_out"], r["cache_hits"],
                   r["fallback_used"] or "-"))
            if r["error"]:
                lines.append("     error: %s" % r["error"][:100])
        return "\n".join(lines)

    def totals(self, job_id: str) -> dict:
        rows = self.for_job(job_id)
        return {
            "nodes": len(rows),
            "wall_s": round(sum((r["ended_at"] or r["started_at"]) - r["started_at"]
                                for r in rows), 2),
            "tool_calls": sum(len(r["tool_calls"]) for r in rows),
            "tokens": sum(r["tokens_in"] + r["tokens_out"] for r in rows),
            "usd": round(sum(r["usd"] for r in rows), 4),
            "cache_hits": sum(r["cache_hits"] for r in rows),
            "fallbacks": [r["fallback_used"] for r in rows if r["fallback_used"]],
            "errors": [r["error"] for r in rows if r["error"]],
        }


class TraceSpan:
    """One node execution. Mutable while open, then terminal."""

    def __init__(self, recorder: TraceRecorder, trace_id: str, job_id: str,
                 node: str, agent: str, attempt: int,
                 input_artifacts: list[str], seq: int):
        self.recorder = recorder
        self.trace_id = trace_id
        self.job_id = job_id
        self.node = node
        self.agent = agent
        self.attempt = attempt
        self.seq = seq
        self.input_artifacts = input_artifacts
        self.output_artifacts: list[str] = []
        self.tool_calls: list[dict] = []
        self.notes: list[str] = []
        self.status = "running"
        self.error: str | None = None
        self.retry_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.usd = 0.0
        self.fallback_used: str | None = None
        self.cache_hits = 0
        self.started_at = time.time()
        self.ended_at: float | None = None

    # ------------------------------------------------------------ writes

    def tool(self, name: str, **detail) -> None:
        """Record a tool invocation. This is what proves an agent used the
        deterministic layer rather than inventing an answer."""
        self.tool_calls.append(
            {"tool": name, "at": round(time.time() - self.started_at, 3), **detail})
        self._flush()

    def note(self, text: str) -> None:
        self.notes.append(text)
        self._flush()

    def llm(self, tokens_in: int, tokens_out: int, usd: float = 0.0) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.usd += usd
        self._flush()

    def output(self, artifact_id: str) -> None:
        self.output_artifacts.append(artifact_id)
        self._flush()

    def fallback(self, name: str) -> None:
        self.fallback_used = name
        self._flush()

    def cache(self, hits: int) -> None:
        self.cache_hits = hits
        self._flush()

    def retry(self) -> None:
        self.retry_count += 1
        self._flush()

    def ok(self) -> None:
        self.status = "ok"
        self.ended_at = time.time()
        self._flush()

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error
        self.ended_at = time.time()
        self._flush()

    def skipped(self, why: str) -> None:
        self.status = "skipped"
        self.notes.append(why)
        self.ended_at = time.time()
        self._flush()

    # ----------------------------------------------------------- persist

    def _open(self) -> None:
        self._flush(insert=True)

    def _flush(self, insert: bool = False) -> None:
        payload = (
            self.trace_id, self.job_id, self.seq, self.node, self.agent,
            self.attempt, self.started_at, self.ended_at, self.status,
            json.dumps(self.input_artifacts), json.dumps(self.output_artifacts),
            json.dumps(self.tool_calls), self.error, self.retry_count,
            self.tokens_in, self.tokens_out, self.usd, self.fallback_used,
            self.cache_hits, json.dumps(self.notes),
        )
        with self.recorder._lock, self.recorder._connect() as con:
            con.execute(
                "INSERT INTO traces (trace_id, job_id, seq, node, agent, attempt,"
                " started_at, ended_at, status, input_artifacts, output_artifacts,"
                " tool_calls, error, retry_count, tokens_in, tokens_out, usd,"
                " fallback_used, cache_hits, notes)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(trace_id) DO UPDATE SET"
                "   ended_at=excluded.ended_at, status=excluded.status,"
                "   output_artifacts=excluded.output_artifacts,"
                "   tool_calls=excluded.tool_calls, error=excluded.error,"
                "   retry_count=excluded.retry_count,"
                "   tokens_in=excluded.tokens_in, tokens_out=excluded.tokens_out,"
                "   usd=excluded.usd, fallback_used=excluded.fallback_used,"
                "   cache_hits=excluded.cache_hits, notes=excluded.notes",
                payload)
            con.commit()


__all__ = ["TraceRecorder", "TraceSpan", "shared_tracer"]
