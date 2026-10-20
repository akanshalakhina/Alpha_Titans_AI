"""The durable job queue. The control plane is its ONLY writer.

WHY SQLITE, AND WHY ONLY ONE WRITER
-----------------------------------
The queue lives on the control plane's persistent disk. The worker VM never
opens this file: it claims, heartbeats and completes over authenticated HTTP,
and the control plane performs every write. That makes SQLite's locking
sufficient. Writers are serialised by `BEGIN IMMEDIATE`, readers proceed
under WAL. It also avoids the trap of two machines each holding a file they
believe is the queue.

AT-LEAST-ONCE, FENCED
---------------------
A claim moves a job to `downloading`, sets a lease and increments
`claim_token`. Every later call from the worker must present that token. If
the lease lapses, the reconciler re-queues the job and the next claim issues a
new token, so a worker that was merely slow - not dead - is refused with
`LeaseLost` the moment it reports back. It cannot complete, or overwrite, a
job someone else now owns.

Exactly-once is not promised, because it cannot be: a worker can finish and
crash before its `complete` lands. The retry is cheap instead. The pipeline
resumes from its checkpoint, artifacts and measurements are cached, and
outputs already in Drive are not uploaded twice.

THE STATE MACHINE IS A TABLE
----------------------------
Every transition is checked against `TRANSITIONS`, and recorded in
`job_events`. A transition not in the table raises; nothing slides into a
state by accident.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .sanitize import CloudError, is_retryable, redact

# ------------------------------------------------------------------ states

UPLOADING = "uploading"
QUEUED = "queued"
STARTING_VM = "starting_vm"
DOWNLOADING = "downloading"
PROCESSING = "processing"
UPLOADING_RESULTS = "uploading_results"
COMPLETED = "completed"
RETRYING = "retrying"
FAILED = "failed"
CANCELLED = "cancelled"

STATES = (UPLOADING, QUEUED, STARTING_VM, DOWNLOADING, PROCESSING,
          UPLOADING_RESULTS, COMPLETED, RETRYING, FAILED, CANCELLED)
TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})
# a worker holds a lease on these
ACTIVE = frozenset({DOWNLOADING, PROCESSING, UPLOADING_RESULTS})
# a worker may claim these (retrying only once `not_before` has passed)
CLAIMABLE = (QUEUED, STARTING_VM, RETRYING)

TRANSITIONS: dict[str, frozenset] = {
    UPLOADING: frozenset({QUEUED, CANCELLED, FAILED}),
    QUEUED: frozenset({STARTING_VM, DOWNLOADING, CANCELLED, FAILED}),
    STARTING_VM: frozenset({QUEUED, DOWNLOADING, CANCELLED, FAILED}),
    DOWNLOADING: frozenset({PROCESSING, RETRYING, FAILED, CANCELLED}),
    PROCESSING: frozenset({UPLOADING_RESULTS, RETRYING, FAILED, CANCELLED}),
    UPLOADING_RESULTS: frozenset({COMPLETED, RETRYING, FAILED, CANCELLED}),
    RETRYING: frozenset({STARTING_VM, DOWNLOADING, QUEUED, CANCELLED,
                         FAILED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

# VM row states, as the control plane understands them
VM_STOPPED = "stopped"
VM_STARTING = "starting"
VM_READY = "ready"
VM_STOPPING = "stopping"
VM_START_FAILED = "start_failed"


class InvalidTransition(CloudError):
    def __init__(self, job_id: str, frm: str, to: str):
        super().__init__("internal", f"{job_id}: {frm} -> {to} is not "
                                     f"allowed")


class LeaseLost(CloudError):
    """The caller no longer owns this job: its lease expired and the job was
    re-queued or claimed again, or it was cancelled."""

    def __init__(self, job_id: str):
        super().__init__("lease_expired", f"{job_id}: lease lost")


class NotFound(CloudError):
    def __init__(self, what: str):
        super().__init__("not_found", what)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    idempotency_key   TEXT UNIQUE,
    state             TEXT NOT NULL,
    settings_json     TEXT NOT NULL DEFAULT '{}',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    queued_at         REAL,
    started_at        REAL,
    finished_at       REAL,
    attempt           INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,
    not_before        REAL NOT NULL DEFAULT 0,
    lease_owner       TEXT,
    lease_expires_at  REAL,
    claim_token       INTEGER NOT NULL DEFAULT 0,
    worker_id         TEXT,
    stage             TEXT,
    progress_json     TEXT NOT NULL DEFAULT '{}',
    error_category    TEXT,
    error_message     TEXT,
    outputs_json      TEXT NOT NULL DEFAULT '[]',
    trace_json        TEXT,
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    output_folder_id  TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, not_before, queued_at);

CREATE TABLE IF NOT EXISTS job_sources (
    job_id         TEXT NOT NULL,
    idx            INTEGER NOT NULL,
    name           TEXT NOT NULL,
    size           INTEGER NOT NULL,
    mime           TEXT NOT NULL,
    session_uri    TEXT,              -- bearer capability: never returned
    received       INTEGER NOT NULL DEFAULT 0,
    drive_file_id  TEXT,
    md5            TEXT,
    verified       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS job_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL,
    at       REAL NOT NULL,
    frm      TEXT,
    to_state TEXT NOT NULL,
    detail   TEXT
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, id);

CREATE TABLE IF NOT EXISTS vm (
    id              TEXT PRIMARY KEY CHECK (id = 'worker'),
    state           TEXT NOT NULL,
    op_started_at   REAL,
    last_start_at   REAL,
    starts_json     TEXT NOT NULL DEFAULT '[]',
    start_failures  INTEGER NOT NULL DEFAULT 0,
    worker_id       TEXT,
    ready_at        REAL,
    last_seen_at    REAL,
    last_activity_at REAL,
    activation_at   REAL,
    stop_requested_at REAL,
    last_error      TEXT,
    updated_at      REAL NOT NULL
);
"""


@dataclass(frozen=True)
class Claim:
    job_id: str
    claim_token: int
    attempt: int
    settings: dict
    sources: list
    output_folder_id: str | None


class JobQueue:
    """All reads and writes of the queue. Thread-safe: every call opens its
    own connection, and writes go through `BEGIN IMMEDIATE`."""

    def __init__(self, db_path: str | Path, clock=time.time,
                 lease_s: float = 300.0, max_attempts: int = 3,
                 retry_base_s: float = 60.0,
                 rate_limit_backoff_s: float = 900.0):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.lease_s = lease_s
        self.max_attempts = max_attempts
        self.retry_base_s = retry_base_s
        self.rate_limit_backoff_s = rate_limit_backoff_s
        self._init_lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)
            now = self.clock()
            con.execute("INSERT OR IGNORE INTO vm (id, state, updated_at) "
                        "VALUES ('worker', ?, ?)", (VM_STOPPED, now))
        try:
            # the session URIs in job_sources are bearer capabilities
            Path(self.db_path).chmod(0o600)
        except OSError:                               # pragma: no cover
            pass

    # ------------------------------------------------------------- plumbing

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30.0,
                              isolation_level=None, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextmanager
    def _tx(self):
        """One write transaction. `BEGIN IMMEDIATE` takes the write lock up
        front, so read-then-write sequences (claim, start) cannot interleave
        with another writer between the read and the write."""
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.execute("COMMIT")
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except sqlite3.Error:                     # pragma: no cover
                pass
            raise
        finally:
            con.close()

    def _move(self, con, job_id: str, to: str, detail: str = "",
              **cols) -> None:
        row = con.execute("SELECT state FROM jobs WHERE job_id = ?",
                          (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job {job_id}")
        frm = row["state"]
        if to != frm and to not in TRANSITIONS[frm]:
            raise InvalidTransition(job_id, frm, to)
        now = self.clock()
        cols = {**cols, "state": to, "updated_at": now}
        sets = ", ".join(f"{k} = ?" for k in cols)
        con.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?",
                    (*cols.values(), job_id))
        if to != frm or detail:
            con.execute("INSERT INTO job_events (job_id, at, frm, to_state, "
                        "detail) VALUES (?, ?, ?, ?, ?)",
                        (job_id, now, frm, to, redact(detail, 300)))

    # ------------------------------------------------------------- creation

    def create(self, idempotency_key: str, files: list[dict],
               settings: dict, max_attempts: int | None = None,
               cap_per_day: int | None = None) -> tuple[dict, bool]:
        """Create a job in `uploading`, or return the existing one for this
        key. Returns (job, created). One upload intent, one job, however many
        times the browser retries the request."""
        if not idempotency_key or len(idempotency_key) > 200:
            raise CloudError("invalid", "idempotency_key is required")
        with self._tx() as con:
            row = con.execute("SELECT job_id FROM jobs WHERE "
                              "idempotency_key = ?",
                              (idempotency_key,)).fetchone()
            if row is not None:
                return self._job(con, row["job_id"]), False
            now = self.clock()
            if cap_per_day is not None:
                # inside the same transaction as the insert, so two
                # browsers at the limit cannot both slip under it
                made = con.execute("SELECT COUNT(*) FROM jobs WHERE "
                                   "created_at > ?",
                                   (now - 86400,)).fetchone()[0]
                if made >= cap_per_day:
                    raise CloudError("daily_cap", f"{made} jobs created in "
                                                  f"the last 24 h")
            job_id = uuid.uuid4().hex[:12]
            con.execute(
                "INSERT INTO jobs (job_id, idempotency_key, state, "
                "settings_json, created_at, updated_at, max_attempts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, idempotency_key, UPLOADING, json.dumps(settings),
                 now, now, max_attempts or self.max_attempts))
            for i, f in enumerate(files):
                con.execute(
                    "INSERT INTO job_sources (job_id, idx, name, size, mime) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (job_id, i, f["name"], int(f["size"]), f["mime"]))
            con.execute("INSERT INTO job_events (job_id, at, frm, to_state, "
                        "detail) VALUES (?, ?, NULL, ?, 'created')",
                        (job_id, now, UPLOADING))
            return self._job(con, job_id), True

    def update_settings(self, job_id: str, settings: dict) -> None:
        """The person's final choices, sent with "Create my video". Only
        while the job is still `uploading`: once queued, a worker may
        already be reading them."""
        with self._tx() as con:
            row = con.execute("SELECT state FROM jobs WHERE job_id = ?",
                              (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"job {job_id}")
            if row["state"] != UPLOADING:
                raise CloudError("invalid", "settings can only change "
                                            "before the job is queued")
            con.execute("UPDATE jobs SET settings_json = ?, updated_at = ? "
                        "WHERE job_id = ?",
                        (json.dumps(settings), self.clock(), job_id))

    def set_session(self, job_id: str, idx: int, session_uri: str) -> None:
        with self._tx() as con:
            con.execute("UPDATE job_sources SET session_uri = ? "
                        "WHERE job_id = ? AND idx = ?",
                        (session_uri, job_id, idx))

    def session(self, job_id: str, idx: int) -> dict:
        """The source row INCLUDING its session URI. Server-side use only."""
        con = self._connect()
        try:
            row = con.execute("SELECT * FROM job_sources WHERE job_id = ? "
                              "AND idx = ?", (job_id, idx)).fetchone()
        finally:
            con.close()
        if row is None:
            raise NotFound(f"source {job_id}/{idx}")
        return dict(row)

    def record_received(self, job_id: str, idx: int, received: int) -> None:
        with self._tx() as con:
            con.execute("UPDATE job_sources SET received = MAX(received, ?) "
                        "WHERE job_id = ? AND idx = ?",
                        (int(received), job_id, idx))
            con.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                        (self.clock(), job_id))

    def mark_source_uploaded(self, job_id: str, idx: int, drive_file_id: str,
                             md5: str | None, size: int) -> None:
        """A source the provider confirms it holds, verified by the caller
        against the declared size. The session is forgotten: it has done
        its job and is a capability nobody needs any more."""
        with self._tx() as con:
            con.execute(
                "UPDATE job_sources SET drive_file_id = ?, md5 = ?, "
                "received = ?, verified = 1, session_uri = NULL "
                "WHERE job_id = ? AND idx = ?",
                (drive_file_id, md5, int(size), job_id, idx))

    def queue_if_uploaded(self, job_id: str) -> tuple[dict, bool]:
        """`uploading` -> `queued`, ONLY when every declared source is
        verified. Idempotent: an already-queued job is returned unchanged.
        Returns (job, newly_queued)."""
        with self._tx() as con:
            job = self._job(con, job_id)
            if job["state"] != UPLOADING:
                return job, False
            pending = [s for s in job["sources"] if not s["verified"]]
            if pending:
                raise CloudError(
                    "invalid", f"{len(pending)} file(s) not uploaded yet")
            now = self.clock()
            self._move(con, job_id, QUEUED, "all sources verified",
                       queued_at=now, not_before=0)
            return self._job(con, job_id), True

    def set_output_folder(self, job_id: str, folder_id: str) -> None:
        with self._tx() as con:
            con.execute("UPDATE jobs SET output_folder_id = ? "
                        "WHERE job_id = ? AND output_folder_id IS NULL",
                        (folder_id, job_id))

    # ------------------------------------------------------------- claiming

    def claim(self, worker_id: str) -> Claim | None:
        """Atomically take the oldest claimable job. Two workers calling at
        the same instant get two different jobs or one job and None - never
        the same job."""
        now = self.clock()
        with self._tx() as con:
            row = con.execute(
                f"SELECT job_id FROM jobs WHERE state IN "
                f"({','.join('?' * len(CLAIMABLE))}) AND not_before <= ? "
                f"AND cancel_requested = 0 "
                f"ORDER BY COALESCE(queued_at, created_at) LIMIT 1",
                (*CLAIMABLE, now)).fetchone()
            if row is None:
                return None
            jid = row["job_id"]
            cur = con.execute("SELECT attempt, claim_token FROM jobs "
                              "WHERE job_id = ?", (jid,)).fetchone()
            self._move(con, jid, DOWNLOADING, f"claimed by {worker_id}",
                       lease_owner=worker_id, worker_id=worker_id,
                       lease_expires_at=now + self.lease_s,
                       claim_token=cur["claim_token"] + 1,
                       attempt=cur["attempt"] + 1,
                       started_at=now, error_category=None,
                       error_message=None, stage="downloading")
            job = self._job(con, jid)
            return Claim(job_id=jid, claim_token=job["claim_token"],
                         attempt=job["attempt"], settings=job["settings"],
                         sources=[{k: s[k] for k in
                                   ("idx", "name", "size", "mime",
                                    "drive_file_id", "md5")}
                                  for s in job["sources"]],
                         output_folder_id=job["output_folder_id"])

    def _owned(self, con, job_id: str, worker_id: str, token: int) -> dict:
        row = con.execute("SELECT * FROM jobs WHERE job_id = ?",
                          (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job {job_id}")
        if (row["state"] not in ACTIVE or row["lease_owner"] != worker_id
                or row["claim_token"] != int(token)):
            raise LeaseLost(job_id)
        return dict(row)

    def heartbeat(self, job_id: str, worker_id: str, token: int,
                  state: str | None = None, stage: str | None = None,
                  progress: dict | None = None) -> dict:
        """Extend the lease; optionally advance within the active states.
        Returns {"cancel": bool}. Raises LeaseLost to a worker that no
        longer owns the job."""
        with self._tx() as con:
            row = self._owned(con, job_id, worker_id, token)
            cols = {"lease_expires_at": self.clock() + self.lease_s}
            if stage is not None:
                cols["stage"] = stage[:80]
            if progress is not None:
                cols["progress_json"] = json.dumps(progress)[:20000]
            to = state or row["state"]
            if to not in ACTIVE:
                raise InvalidTransition(job_id, row["state"], to)
            self._move(con, job_id, to, "", **cols)
            return {"cancel": bool(row["cancel_requested"])}

    def complete(self, job_id: str, worker_id: str, token: int,
                 outputs: list[dict], trace: dict | None = None) -> dict:
        """Only from `uploading_results`, only by the lease holder, and only
        with every output verified in Drive. A worker that could not upload
        its results cannot mark the job complete, whatever it believes."""
        problems = _check_outputs(outputs)
        if problems:
            raise CloudError("verification", "; ".join(problems))
        with self._tx() as con:
            row = self._owned(con, job_id, worker_id, token)
            if row["state"] != UPLOADING_RESULTS:
                raise InvalidTransition(job_id, row["state"], COMPLETED)
            self._move(con, job_id, COMPLETED, "outputs verified",
                       outputs_json=json.dumps(outputs),
                       trace_json=json.dumps(trace)[:500_000] if trace
                       else None,
                       finished_at=self.clock(), lease_owner=None,
                       lease_expires_at=None, stage="completed")
            return self._job(con, job_id)

    def fail(self, job_id: str, worker_id: str, token: int, category: str,
             message: str, retryable: bool | None = None) -> dict:
        """A worker reporting its own failure. Retried if the category is
        retryable and attempts remain; otherwise terminal."""
        with self._tx() as con:
            row = self._owned(con, job_id, worker_id, token)
            return self._settle(con, row, category, message, retryable)

    def _settle(self, con, row: dict, category: str, message: str,
                retryable: bool | None) -> dict:
        jid = row["job_id"]
        clear = {"lease_owner": None, "lease_expires_at": None}
        if row["cancel_requested"] or category == "cancelled":
            self._move(con, jid, CANCELLED, "cancelled", **clear,
                       error_category="cancelled",
                       error_message="cancelled",
                       finished_at=self.clock())
            return self._job(con, jid)
        retry = is_retryable(category) if retryable is None else retryable
        if retry and row["attempt"] < row["max_attempts"]:
            self._move(con, jid, RETRYING, f"{category}: {message}", **clear,
                       error_category=category,
                       error_message=redact(message, 400),
                       not_before=self.clock() + self.backoff(
                           row["attempt"], category))
        else:
            self._move(con, jid, FAILED, f"{category}: {message}", **clear,
                       error_category=category,
                       error_message=redact(message, 400),
                       finished_at=self.clock())
        return self._job(con, jid)

    def backoff(self, attempt: int, category: str) -> float:
        """Exponential from `retry_base_s`, capped at an hour. A rate limit
        waits at least `rate_limit_backoff_s`: retrying sooner would only
        hold a worker VM up while a provider's window is still shut."""
        base = self.retry_base_s * (2 ** max(0, attempt - 1))
        if category in ("rate_limited", "quota"):
            base = max(base, self.rate_limit_backoff_s)
        return min(base, 3600.0)

    # ------------------------------------------------------------ user side

    def cancel(self, job_id: str) -> dict:
        """Not-yet-running jobs are cancelled at once. A running one is
        flagged; its worker sees the flag on the next heartbeat and stops."""
        with self._tx() as con:
            job = self._job(con, job_id)
            if job["state"] in TERMINAL:
                return job
            if job["state"] in ACTIVE:
                con.execute("UPDATE jobs SET cancel_requested = 1, "
                            "updated_at = ? WHERE job_id = ?",
                            (self.clock(), job_id))
                con.execute("INSERT INTO job_events (job_id, at, frm, "
                            "to_state, detail) VALUES (?, ?, ?, ?, "
                            "'cancel requested')",
                            (job_id, self.clock(), job["state"],
                             job["state"]))
            else:
                self._move(con, job_id, CANCELLED, "cancelled by user",
                           cancel_requested=1, finished_at=self.clock(),
                           error_category="cancelled",
                           error_message="cancelled")
            return self._job(con, job_id)

    # -------------------------------------------------------- reconciliation

    def expire_leases(self) -> list[str]:
        """Active jobs whose worker went quiet. Re-queued for retry, or
        failed when out of attempts, or cancelled if that was asked for."""
        now = self.clock()
        moved = []
        with self._tx() as con:
            rows = con.execute(
                f"SELECT * FROM jobs WHERE state IN "
                f"({','.join('?' * len(ACTIVE))}) "
                f"AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
                (*ACTIVE, now)).fetchall()
            for r in rows:
                self._settle(con, dict(r), "lease_expired",
                             f"no heartbeat from {r['lease_owner']} since "
                             f"{r['lease_expires_at'] - self.lease_s:.0f}",
                             None)
                moved.append(r["job_id"])
        return moved

    def expire_stale(self, upload_timeout_s: float, queue_timeout_s: float
                     ) -> list[str]:
        """Uploads abandoned half-way, and work that waited too long for a
        worker that never came. Both end visibly rather than lingering."""
        now = self.clock()
        out = []
        with self._tx() as con:
            # fetched whole before updating: never mutate a table while a
            # cursor over it is still being read
            for r in con.execute("SELECT job_id FROM jobs WHERE state = ? "
                                 "AND updated_at < ?",
                                 (UPLOADING, now - upload_timeout_s)
                                 ).fetchall():
                self._move(con, r["job_id"], FAILED, "upload abandoned",
                           error_category="invalid",
                           error_message="the upload was never finished",
                           finished_at=now)
                out.append(r["job_id"])
            for r in con.execute(
                    f"SELECT job_id FROM jobs WHERE state IN "
                    f"({','.join('?' * len(CLAIMABLE))}) "
                    f"AND COALESCE(queued_at, created_at) < ?",
                    (*CLAIMABLE, now - queue_timeout_s)).fetchall():
                self._move(con, r["job_id"], FAILED, "queue timeout",
                           error_category="worker_unavailable",
                           error_message="no worker picked this job up in "
                                         "time", finished_at=now)
                out.append(r["job_id"])
        return out

    def counts(self, con=None) -> dict:
        """What the VM manager needs to decide anything. Pass `con` to read
        inside an open transaction, so a decision and the facts it rests
        on cannot be separated by another writer."""
        now = self.clock()
        own = con is None
        con = con or self._connect()
        try:
            claimable = con.execute(
                f"SELECT COUNT(*) FROM jobs WHERE state IN "
                f"({','.join('?' * len(CLAIMABLE))}) AND not_before <= ? "
                f"AND cancel_requested = 0", (*CLAIMABLE, now)).fetchone()[0]
            waiting = con.execute(
                f"SELECT COUNT(*) FROM jobs WHERE state IN "
                f"({','.join('?' * len(CLAIMABLE))}) "
                f"AND cancel_requested = 0", CLAIMABLE).fetchone()[0]
            active = con.execute(
                f"SELECT COUNT(*) FROM jobs WHERE state IN "
                f"({','.join('?' * len(ACTIVE))})", tuple(ACTIVE)
            ).fetchone()[0]
            next_due = con.execute(
                "SELECT MIN(not_before) FROM jobs WHERE state = ? "
                "AND cancel_requested = 0", (RETRYING,)).fetchone()[0]
        finally:
            if own:
                con.close()
        return {"claimable": claimable, "waiting": waiting, "active": active,
                "next_retry_at": next_due}

    def mark_starting(self) -> int:
        """Claimable `queued` jobs -> `starting_vm`, so the UI can say a
        worker is being started for them."""
        now = self.clock()
        n = 0
        with self._tx() as con:
            for r in con.execute(
                    "SELECT job_id FROM jobs WHERE state IN (?, ?) "
                    "AND not_before <= ? AND cancel_requested = 0",
                    (QUEUED, RETRYING, now)).fetchall():
                self._move(con, r["job_id"], STARTING_VM,
                           "worker start requested")
                n += 1
        return n

    def unmark_starting(self, why: str) -> int:
        """The start did not happen: `starting_vm` -> `queued` with a note,
        so nothing claims to be starting that is not."""
        n = 0
        with self._tx() as con:
            for r in con.execute("SELECT job_id FROM jobs WHERE state = ?",
                                 (STARTING_VM,)).fetchall():
                self._move(con, r["job_id"], QUEUED, why)
                n += 1
        return n

    # ------------------------------------------------------------- the VM row

    def vm(self) -> dict:
        con = self._connect()
        try:
            row = dict(con.execute("SELECT * FROM vm WHERE id = 'worker'")
                       .fetchone())
        finally:
            con.close()
        row["starts"] = json.loads(row.pop("starts_json") or "[]")
        return row

    @contextmanager
    def vm_tx(self):
        """The VM row inside a write transaction, for single-flight
        decisions. Yields (connection, row-dict); write with `vm_update`."""
        with self._tx() as con:
            row = dict(con.execute("SELECT * FROM vm WHERE id = 'worker'")
                       .fetchone())
            row["starts"] = json.loads(row.pop("starts_json") or "[]")
            yield con, row

    def vm_update(self, con=None, **cols) -> None:
        if "starts" in cols:
            cols["starts_json"] = json.dumps(cols.pop("starts"))
        if "last_error" in cols and cols["last_error"]:
            cols["last_error"] = redact(cols["last_error"], 400)
        cols["updated_at"] = self.clock()
        sets = ", ".join(f"{k} = ?" for k in cols)
        if con is not None:
            con.execute(f"UPDATE vm SET {sets} WHERE id = 'worker'",
                        tuple(cols.values()))
            return
        with self._tx() as c:
            c.execute(f"UPDATE vm SET {sets} WHERE id = 'worker'",
                      tuple(cols.values()))

    # ----------------------------------------------------------------- reads

    def _job(self, con, job_id: str) -> dict:
        row = con.execute("SELECT * FROM jobs WHERE job_id = ?",
                          (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job {job_id}")
        job = dict(row)
        job["settings"] = json.loads(job.pop("settings_json") or "{}")
        job["progress"] = json.loads(job.pop("progress_json") or "{}")
        job["outputs"] = json.loads(job.pop("outputs_json") or "[]")
        trace = job.pop("trace_json")
        job["has_trace"] = bool(trace)
        job["sources"] = [
            {k: v for k, v in dict(s).items() if k != "session_uri"}
            for s in con.execute("SELECT * FROM job_sources WHERE job_id = ? "
                                 "ORDER BY idx", (job_id,))]
        return job

    def get(self, job_id: str) -> dict:
        con = self._connect()
        try:
            return self._job(con, job_id)
        finally:
            con.close()

    def trace(self, job_id: str) -> dict | None:
        con = self._connect()
        try:
            row = con.execute("SELECT trace_json FROM jobs WHERE job_id = ?",
                              (job_id,)).fetchone()
        finally:
            con.close()
        if row is None:
            raise NotFound(f"job {job_id}")
        return json.loads(row["trace_json"]) if row["trace_json"] else None

    def events(self, job_id: str) -> list[dict]:
        con = self._connect()
        try:
            return [dict(r) for r in con.execute(
                "SELECT at, frm, to_state, detail FROM job_events "
                "WHERE job_id = ? ORDER BY id", (job_id,))]
        finally:
            con.close()

    def recent(self, limit: int = 30) -> list[dict]:
        con = self._connect()
        try:
            ids = [r["job_id"] for r in con.execute(
                "SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT ?",
                (int(limit),))]
            return [self._job(con, i) for i in ids]
        finally:
            con.close()


def _check_outputs(outputs) -> list[str]:
    """What `complete` refuses. The primary video must be there, and every
    declared output must carry a Drive ID and the size and md5 the worker
    verified against Drive's own record of the file."""
    if not isinstance(outputs, list) or not outputs:
        return ["no outputs"]
    problems = []
    kinds = set()
    for i, o in enumerate(outputs):
        if not isinstance(o, dict):
            problems.append(f"output {i} is not an object")
            continue
        kinds.add(o.get("kind"))
        if not o.get("drive_file_id"):
            problems.append(f"output {i} has no Drive file id")
        if not o.get("verified"):
            problems.append(f"output {i} was not verified in Drive")
        if not isinstance(o.get("size"), int) or o["size"] <= 0:
            problems.append(f"output {i} has no size")
    if "video" not in kinds:
        problems.append("the finished video is not among the outputs")
    return problems
