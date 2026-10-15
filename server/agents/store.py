"""
Artifact store.

Immutable, append-only, revision-aware. This is the Phase 1 stand-in for the
`artifacts` table in the target architecture - same contract, SQLite instead of
Postgres, which is all Phase 1 needs.

ONE STORE, NOT ONE PER JOB
--------------------------
Artifacts live in a single `outputs/_agents/state.db`. Every query in this
module was ALREADY `WHERE job_id = ?` and the table was already keyed
`UNIQUE (job_id, kind, revision)`, so isolation between jobs is unchanged -
sharing the file needed no schema change at all. What it buys is the
cross-job reads (`jobs()`, and later: which edit decisions preceded which
published video) that a per-job file makes impossible.

Two properties are doing real work here:

  * (job_id, kind, revision) is UNIQUE. An agent cannot overwrite its previous
    output; a revision writes a new row. That is what lets the critic compare
    attempt 1 against attempt 2 and ship the better one.

  * inputs_hash is indexed. An agent whose inputs have not changed can be
    skipped entirely, which is what makes a revision cheap: re-planning the
    edit does not re-measure the clips.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from .artifacts import Artifact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id     TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL,
    kind            TEXT NOT NULL,
    schema_version  INTEGER NOT NULL,
    revision        INTEGER NOT NULL,
    produced_by     TEXT NOT NULL,
    produced_at     REAL NOT NULL,
    inputs_hash     TEXT NOT NULL,
    trace_id        TEXT NOT NULL,
    payload         TEXT NOT NULL,
    decisions       TEXT NOT NULL,
    confidence      REAL NOT NULL,
    degraded        INTEGER NOT NULL,
    fallback_used   TEXT,
    cost            TEXT NOT NULL,
    UNIQUE (job_id, kind, revision)
);
CREATE INDEX IF NOT EXISTS artifacts_job_kind ON artifacts(job_id, kind, revision);
CREATE INDEX IF NOT EXISTS artifacts_inputs   ON artifacts(job_id, kind, inputs_hash);
"""

T = TypeVar("T", bound=BaseModel)


SHARED_DIR = "_agents"
SHARED_DB = "state.db"


def outputs_dir(root: Path | None = None) -> Path:
    """The outputs folder a job lives under.

    `config` is imported lazily so this module stays importable on its own,
    without dragging in the pipeline bridge.
    """
    if root is not None:
        return Path(root) / "outputs"
    from .tools.pipeline_bridge import config_mod
    return Path(config_mod.OUTPUTS_DIR)


def shared_db_path(root: Path | None = None,
                   outputs: Path | None = None) -> Path:
    """The one artifact store: <outputs>/_agents/state.db.

    A folder named `_agents` sits safely alongside the job folders: the legacy
    `jobs.listing()` walks outputs/, calls `jobs._restore("_agents")`, finds no
    job.json there and skips it. Nothing in the existing app has to know.

    `outputs` wins over `root` and over config. That matters because the store
    must sit beside the job folders WHEREVER they are - deriving it from config
    alone would send a job built under a test root's outputs/ to the process
    default instead, and silently split one run's state across two files.
    """
    base = Path(outputs) if outputs is not None else outputs_dir(root)
    return base / SHARED_DIR / SHARED_DB


def shared_store(root: Path | None = None,
                 outputs: Path | None = None) -> "ArtifactStore":
    return ArtifactStore(shared_db_path(root, outputs))


def store_for_build_dir(build_dir: str | Path) -> "ArtifactStore":
    """The shared store that belongs beside this job's build folder."""
    return ArtifactStore(
        shared_db_path(outputs=Path(build_dir).resolve().parent))


LEGACY_DB = "agent_state.db"


def legacy_db_path(job_id: str, root: Path | None = None,
                   outputs: Path | None = None) -> Path:
    """Where a pre-shared-store job kept its own database."""
    base = Path(outputs) if outputs is not None else outputs_dir(root)
    return base / job_id / LEGACY_DB


def resolve_job_db(job_id: str, root: Path | None = None,
                   outputs: Path | None = None) -> tuple[Path, str]:
    """Which database actually holds this job. Returns (path, "shared"|"legacy").

    Reads prefer the shared store, and fall back to the job's own database when
    the shared store has nothing for it. That fallback is what lets a Phase 1
    job built before this change keep working WITHOUT being migrated - running
    `python -m agents.migrate` is then a tidy-up rather than a prerequisite.

    Writes always go to the shared store; only reads consult the fallback.
    """
    shared = shared_db_path(root, outputs)
    if shared.exists():
        try:
            if ArtifactStore(shared).has_job(job_id):
                return shared, "shared"
        except Exception:                        # pragma: no cover
            pass
    legacy = legacy_db_path(job_id, root, outputs)
    if legacy.exists():
        return legacy, "legacy"
    return shared, "shared"


class ArtifactStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        # One shared file means concurrent jobs now contend for it, which the
        # previous per-job files never had to. WAL lets readers work while a
        # writer holds the file, and busy_timeout turns a momentary lock into
        # a short wait instead of an immediate "database is locked".
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
        except sqlite3.DatabaseError:            # pragma: no cover
            pass                                 # read-only media, etc.
        return con

    # ------------------------------------------------------------- write

    def put(self, artifact: Artifact) -> Artifact:
        """Persist an artifact. Rejects an overwrite rather than allowing it.

        A duplicate (job, kind, revision) means an agent tried to mutate its
        own past output - the exact failure mode the immutability rule exists
        to prevent - so it is surfaced instead of silently swallowed.
        """
        row = (
            artifact.artifact_id, artifact.job_id, artifact.kind,
            artifact.schema_version, artifact.revision, artifact.produced_by,
            artifact.produced_at, artifact.inputs_hash, artifact.trace_id,
            artifact.payload.model_dump_json(),
            json.dumps([d.model_dump(mode="json") for d in artifact.decisions]),
            artifact.confidence, int(artifact.degraded), artifact.fallback_used,
            artifact.cost.model_dump_json(),
        )
        with self._lock, self._connect() as con:
            try:
                con.execute(
                    "INSERT INTO artifacts (artifact_id, job_id, kind,"
                    " schema_version, revision, produced_by, produced_at,"
                    " inputs_hash, trace_id, payload, decisions, confidence,"
                    " degraded, fallback_used, cost)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                con.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"artifact {artifact.kind} rev{artifact.revision} already "
                    f"exists for job {artifact.job_id}; artifacts are immutable "
                    f"- write a new revision instead") from exc
        return artifact

    # -------------------------------------------------------------- read

    def get(self, job_id: str, kind: str, payload_type: Type[T],
            revision: int | None = None) -> Artifact[T] | None:
        sql = ("SELECT * FROM artifacts WHERE job_id = ? AND kind = ?"
               + (" AND revision = ?" if revision is not None
                  else " ORDER BY revision DESC") + " LIMIT 1")
        args = (job_id, kind) + ((revision,) if revision is not None else ())
        with self._connect() as con:
            row = con.execute(sql, args).fetchone()
        return self._hydrate(row, payload_type) if row else None

    def get_raw(self, job_id: str, kind: str,
                revision: int | None = None) -> dict | None:
        """The stored artifact as plain JSON, without a payload type.

        `get` needs the pydantic class for the kind, which a caller that
        only wants to SHOW the thing - the UI, a trace viewer - has no
        business knowing. The envelope fields travel with it so nobody can
        present a degraded answer as a reasoned one.
        """
        sql = ("SELECT * FROM artifacts WHERE job_id = ? AND kind = ?"
               + (" AND revision = ?" if revision is not None
                  else " ORDER BY revision DESC") + " LIMIT 1")
        args = (job_id, kind) + ((revision,) if revision is not None else ())
        with self._connect() as con:
            row = con.execute(sql, args).fetchone()
        if row is None:
            return None
        out = dict(row)
        for f in ("payload", "decisions", "input_artifacts"):
            if isinstance(out.get(f), str):
                try:
                    out[f] = json.loads(out[f])
                except json.JSONDecodeError:
                    pass
        return out

    def get_by_inputs(self, job_id: str, kind: str, inputs_hash: str,
                      payload_type: Type[T]) -> Artifact[T] | None:
        """Cache lookup: has this agent already run on exactly these inputs?"""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM artifacts WHERE job_id = ? AND kind = ?"
                " AND inputs_hash = ? ORDER BY revision DESC LIMIT 1",
                (job_id, kind, inputs_hash)).fetchone()
        return self._hydrate(row, payload_type) if row else None

    def all_revisions(self, job_id: str, kind: str,
                      payload_type: Type[T]) -> list[Artifact[T]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM artifacts WHERE job_id = ? AND kind = ?"
                " ORDER BY revision", (job_id, kind)).fetchall()
        return [self._hydrate(r, payload_type) for r in rows]

    def next_revision(self, job_id: str, kind: str) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(revision), -1) AS m FROM artifacts"
                " WHERE job_id = ? AND kind = ?", (job_id, kind)).fetchone()
        return int(row["m"]) + 1

    # ------------------------------------------------ cross-job (shared db)

    def jobs(self) -> list[str]:
        """Every job that has artifacts here. Meaningless on a per-job file;
        the reason the shared store exists."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT job_id, MAX(produced_at) AS last FROM artifacts"
                " GROUP BY job_id ORDER BY last DESC").fetchall()
        return [r["job_id"] for r in rows]

    def has_job(self, job_id: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM artifacts WHERE job_id = ? LIMIT 1",
                (job_id,)).fetchone()
        return row is not None

    def purge_job(self, job_id: str) -> int:
        """Delete one job's artifacts. Returns the row count removed.

        This exists because of a hazard the shared store introduces: deleting
        outputs/<job_id>/ used to delete that job's artifacts with it. Now they
        survive, so REUSING a job id later appends revisions to the old job's
        history instead of starting clean. `next_revision` returns max+1 so
        nothing raises - the two builds just silently merge. Purge first when
        deliberately reusing an id.
        """
        with self._lock, self._connect() as con:
            cur = con.execute("DELETE FROM artifacts WHERE job_id = ?", (job_id,))
            con.commit()
            return cur.rowcount

    def manifest(self, job_id: str) -> list[dict]:
        """Every artifact for a job, newest first - for the report."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT artifact_id, kind, revision, produced_by, produced_at,"
                " inputs_hash, confidence, degraded, fallback_used"
                " FROM artifacts WHERE job_id = ?"
                " ORDER BY produced_at", (job_id,)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _hydrate(row: sqlite3.Row, payload_type: Type[T]) -> Artifact[T]:
        from .artifacts import Cost, Decision
        return Artifact[payload_type](
            artifact_id=row["artifact_id"], job_id=row["job_id"],
            kind=row["kind"], schema_version=row["schema_version"],
            revision=row["revision"], produced_by=row["produced_by"],
            produced_at=row["produced_at"], inputs_hash=row["inputs_hash"],
            trace_id=row["trace_id"],
            payload=payload_type.model_validate_json(row["payload"]),
            decisions=[Decision(**d) for d in json.loads(row["decisions"])],
            confidence=row["confidence"], degraded=bool(row["degraded"]),
            fallback_used=row["fallback_used"],
            cost=Cost.model_validate_json(row["cost"]),
        )


__all__ = ["ArtifactStore", "shared_store", "shared_db_path",
           "legacy_db_path", "resolve_job_db", "store_for_build_dir",
           "outputs_dir",
           "SHARED_DIR", "SHARED_DB", "LEGACY_DB"]
