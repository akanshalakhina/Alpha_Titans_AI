"""
Migrate per-job agent stores into the one shared store.

    outputs/<job_id>/agent_state.db   ->   outputs/_agents/state.db

WHY THIS IS SAFE TO RUN, AND TO RUN AGAIN
-----------------------------------------
Both tables are copied with INSERT OR IGNORE keyed on their PRIMARY KEY -
`artifacts.artifact_id` and `traces.trace_id`. Those ids are minted once, when
the row is created, and never change. So:

  deterministic   the same source produces the same shared rows
  idempotent      a second run inserts nothing and reports 0 copied
  partial-safe    a run interrupted half way is simply completed by the next
  additive        the source database is NEVER deleted or modified

CONFLICTS ARE REPORTED, NOT SWALLOWED
-------------------------------------
`artifacts` also carries UNIQUE (job_id, kind, revision). If the shared store
already holds a DIFFERENT artifact_id at the same (job, kind, revision) - which
means two genuinely different builds reused one job id - INSERT OR IGNORE would
quietly drop the incoming row. That case is detected up front and counted as a
conflict, so it surfaces instead of turning into a silent data loss.

MIGRATION IS OPTIONAL
---------------------
Reads fall back to the per-job database when the shared store has nothing for a
job (see runner.resolve_store), so an un-migrated Phase 1 job keeps working.
Migrating just moves it onto the path everything else now uses.

    python -m agents.migrate --all
    python -m agents.migrate --all --dry-run
    python -m agents.migrate --job-id phase1demo
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .store import ArtifactStore, shared_db_path
from .trace import TraceRecorder

LEGACY_DB = "agent_state.db"


@dataclass
class JobMigration:
    job_id: str
    source: str
    artifacts_copied: int = 0
    artifacts_present: int = 0
    traces_copied: int = 0
    traces_present: int = 0
    conflicts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.conflicts

    def summary(self) -> str:
        if self.error:
            return f"error: {self.error}"
        bits = [f"artifacts +{self.artifacts_copied}/{self.artifacts_present}",
                f"traces +{self.traces_copied}/{self.traces_present}"]
        if self.conflicts:
            bits.append(f"CONFLICTS {len(self.conflicts)}")
        return "  ".join(bits)


# ------------------------------------------------------------- discovery

def legacy_stores(root: Path | None = None) -> list[tuple[str, Path]]:
    """Every per-job database still on disk, as (job_id, path)."""
    if root is not None:
        outputs = Path(root) / "outputs"
    else:
        from .tools.pipeline_bridge import config_mod
        outputs = Path(config_mod.OUTPUTS_DIR)
    if not outputs.exists():
        return []
    found = []
    for d in sorted(outputs.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue                       # never treat _agents/ as a job
        db = d / LEGACY_DB
        if db.exists():
            found.append((d.name, db))
    return found


# ------------------------------------------------------------- migration

def _rows(con: sqlite3.Connection, table: str, job_id: str) -> list[sqlite3.Row]:
    try:
        return con.execute(f"SELECT * FROM {table} WHERE job_id = ?",
                           (job_id,)).fetchall()
    except sqlite3.DatabaseError:
        return []


def _copy(src_rows: list[sqlite3.Row], dst: sqlite3.Connection,
          table: str) -> int:
    if not src_rows:
        return 0
    cols = src_rows[0].keys()
    sql = (f"INSERT OR IGNORE INTO {table} ({','.join(cols)})"
           f" VALUES ({','.join('?' * len(cols))})")
    before = dst.total_changes
    dst.executemany(sql, [tuple(r[c] for c in cols) for r in src_rows])
    return dst.total_changes - before


def migrate_job(job_id: str, src_db: Path, root: Path | None = None,
                dry_run: bool = False) -> JobMigration:
    """Copy one job's rows into the shared store."""
    result = JobMigration(job_id=job_id, source=str(src_db))
    shared_path = shared_db_path(root)
    # BOTH schemas, not just artifacts: the two tables live in one file but are
    # created by two different classes, and migrating into a shared store that
    # no tracer had touched yet failed on "no such table: traces".
    ArtifactStore(shared_path)
    TraceRecorder(shared_path)

    try:
        src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True, timeout=30)
        src.row_factory = sqlite3.Row
    except sqlite3.DatabaseError as exc:
        result.error = f"cannot open source: {exc}"
        return result

    try:
        arts = _rows(src, "artifacts", job_id)
        trcs = _rows(src, "traces", job_id)
        result.artifacts_present = len(arts)
        result.traces_present = len(trcs)

        dst = sqlite3.connect(shared_path, timeout=30)
        dst.row_factory = sqlite3.Row
        try:
            dst.execute("PRAGMA busy_timeout=30000")

            # Detect a real clash BEFORE writing: the same (job, kind,
            # revision) already present under a different artifact_id means
            # two different builds share a job id. INSERT OR IGNORE would drop
            # the incoming row without a word, so surface it instead.
            for a in arts:
                row = dst.execute(
                    "SELECT artifact_id FROM artifacts"
                    " WHERE job_id = ? AND kind = ? AND revision = ?",
                    (a["job_id"], a["kind"], a["revision"])).fetchone()
                if row and row["artifact_id"] != a["artifact_id"]:
                    result.conflicts.append(
                        f"{a['kind']} rev{a['revision']}: shared holds "
                        f"{row['artifact_id']}, source has {a['artifact_id']}")

            if result.conflicts:
                result.error = ("refusing to merge - the shared store already "
                                "holds different artifacts for this job id")
                return result

            if not dry_run:
                result.artifacts_copied = _copy(arts, dst, "artifacts")
                result.traces_copied = _copy(trcs, dst, "traces")
                dst.commit()
            else:
                have_a = {r["artifact_id"] for r in dst.execute(
                    "SELECT artifact_id FROM artifacts WHERE job_id = ?",
                    (job_id,))}
                have_t = {r["trace_id"] for r in dst.execute(
                    "SELECT trace_id FROM traces WHERE job_id = ?", (job_id,))}
                result.artifacts_copied = sum(
                    1 for a in arts if a["artifact_id"] not in have_a)
                result.traces_copied = sum(
                    1 for t in trcs if t["trace_id"] not in have_t)
        finally:
            dst.close()
    except sqlite3.DatabaseError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        src.close()
    return result


def migrate_all(root: Path | None = None,
                dry_run: bool = False) -> list[JobMigration]:
    return [migrate_job(jid, db, root, dry_run)
            for jid, db in legacy_stores(root)]


# ------------------------------------------------------------------- cli

def _cli() -> int:
    ap = argparse.ArgumentParser(
        prog="agents.migrate",
        description="Copy per-job agent stores into outputs/_agents/state.db. "
                    "Never deletes the source; safe to run twice.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-id")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve() if args.root else None

    print(f"shared store: {shared_db_path(root)}")
    if args.dry_run:
        print("(dry run - nothing will be written)")

    if args.all:
        results = migrate_all(root, args.dry_run)
    else:
        found = dict(legacy_stores(root))
        if args.job_id not in found:
            print(f"no per-job store for {args.job_id}", file=sys.stderr)
            return 1
        results = [migrate_job(args.job_id, found[args.job_id], root,
                               args.dry_run)]

    if not results:
        print("nothing to migrate")
        return 0

    bad = 0
    for r in results:
        print(f"  {r.job_id:16} {r.summary()}")
        for c in r.conflicts:
            print(f"      conflict: {c}")
        bad += 0 if r.ok else 1
    print(f"{len(results) - bad} of {len(results)} job(s) "
          f"{'would migrate' if args.dry_run else 'migrated'} cleanly")
    print("the per-job databases were left exactly where they are")
    return 1 if bad else 0


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(_cli())
