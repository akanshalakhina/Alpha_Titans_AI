"""
The shared artifact store: outputs/_agents/state.db.

The schema did not change - every query was already `WHERE job_id = ?` and the
table was already `UNIQUE (job_id, kind, revision)`. So these tests are about
the things sharing one file actually puts at risk: isolation between jobs,
cache correctness across jobs, the migration, and the fallback that keeps
un-migrated Phase 1 jobs readable.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from agents import migrate
from agents.artifacts import Artifact, Cost, Decision, EditPlan, Segment
from agents.store import (ArtifactStore, legacy_db_path, resolve_job_db,
                          shared_db_path, shared_store, store_for_build_dir)
from agents.trace import TraceRecorder, shared_tracer


def seg(clip_id="a.mp4", start=0.0, end=4.0):
    return Segment(index=1, clip_id=clip_id, start=start, end=end,
                   duration=round(end - start, 2), orientation="horizontal")


def plan(total=4.0, clip_id="a.mp4"):
    return EditPlan(segments=[seg(clip_id)], total_duration=total,
                    target_seconds=20.0, canvas=(1920, 1080))


def art(job, revision=0, inputs_hash="h1", payload=None, **kw):
    return Artifact[EditPlan](
        job_id=job, kind="EditPlan", revision=revision,
        produced_by="editing_director", inputs_hash=inputs_hash,
        payload=payload or plan(), **kw)


# ----------------------------------------------------------------- paths

def test_shared_path_sits_beside_the_job_folders(tmp_path):
    assert shared_db_path(root=tmp_path) == \
           tmp_path / "outputs" / "_agents" / "state.db"
    assert shared_db_path(outputs=tmp_path / "out") == \
           tmp_path / "out" / "_agents" / "state.db"


def test_outputs_wins_over_root(tmp_path):
    """The store must follow the job folders, not process config - otherwise
    one run's state silently splits across two files."""
    got = shared_db_path(root=tmp_path, outputs=tmp_path / "elsewhere")
    assert got == tmp_path / "elsewhere" / "_agents" / "state.db"


def test_store_for_build_dir_resolves_from_the_job(tmp_path):
    build = tmp_path / "outputs" / "job1"
    build.mkdir(parents=True)
    s = store_for_build_dir(build)
    assert Path(s.db_path) == tmp_path / "outputs" / "_agents" / "state.db"


def test_the_shared_folder_is_not_mistaken_for_a_job(tmp_path):
    """jobs.listing() walks outputs/ and calls _restore on every directory.
    `_agents` has no job.json, so the legacy app must simply skip it."""
    import jobs as legacy_jobs

    outputs = tmp_path / "outputs"
    (outputs / "_agents").mkdir(parents=True)
    shared_store(outputs=outputs)                 # creates state.db
    assert (outputs / "_agents" / "state.db").exists()

    import adapter
    previous = adapter.OUTPUTS
    adapter.OUTPUTS = outputs
    try:
        legacy_jobs._JOBS.clear()
        assert legacy_jobs._restore("_agents") is None
        assert legacy_jobs.listing() == []
    finally:
        adapter.OUTPUTS = previous
        legacy_jobs._JOBS.clear()


# ------------------------------------------------------- 1. no collisions

def test_two_jobs_share_one_file_without_colliding(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0, payload=plan(4.0, "a.mp4")))
    s.put(art("jobB", 0, payload=plan(9.0, "b.mp4")))

    a = s.get("jobA", "EditPlan", EditPlan)
    b = s.get("jobB", "EditPlan", EditPlan)
    assert a.payload.total_duration == 4.0
    assert b.payload.total_duration == 9.0
    assert a.artifact_id != b.artifact_id
    assert set(s.jobs()) == {"jobA", "jobB"}


def test_same_revision_number_in_two_jobs_is_not_a_conflict(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0))
    s.put(art("jobB", 0))          # same kind, same revision, different job
    assert s.next_revision("jobA", "EditPlan") == 1
    assert s.next_revision("jobB", "EditPlan") == 1


def test_one_job_cannot_see_anothers_artifacts(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0))
    assert s.get("jobB", "EditPlan", EditPlan) is None
    assert s.all_revisions("jobB", "EditPlan", EditPlan) == []
    assert s.manifest("jobB") == []
    assert s.has_job("jobA") and not s.has_job("jobB")


def test_traces_are_isolated_and_sequence_per_job(tmp_path):
    t = shared_tracer(root=tmp_path)
    for job in ("jobA", "jobB"):
        for node in ("clip_analysis", "render"):
            with t.node(job, node, node) as span:
                span.tool("x")
    assert [r["seq"] for r in t.for_job("jobA")] == [1, 2]
    assert [r["seq"] for r in t.for_job("jobB")] == [1, 2]
    assert set(t.jobs()) == {"jobA", "jobB"}


# ------------------------------------------------- 2/3. caching and hashes

def test_cache_reuse_still_works_within_a_job(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0, inputs_hash="abc"))
    assert s.get_by_inputs("jobA", "EditPlan", "abc", EditPlan) is not None


def test_an_identical_hash_in_another_job_is_not_a_cache_hit(tmp_path):
    """The hazard sharing a file introduces: two jobs can legitimately produce
    the same inputs_hash, and must NOT read each other's artifacts."""
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0, inputs_hash="same", payload=plan(4.0, "a.mp4")))
    assert s.get_by_inputs("jobB", "EditPlan", "same", EditPlan) is None

    s.put(art("jobB", 0, inputs_hash="same", payload=plan(9.0, "b.mp4")))
    got = s.get_by_inputs("jobB", "EditPlan", "same", EditPlan)
    assert got.payload.total_duration == 9.0, "served the wrong job's artifact"


def test_hashes_survive_the_round_trip(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0, inputs_hash="deadbeef1234"))
    assert s.get("jobA", "EditPlan", EditPlan).inputs_hash == "deadbeef1234"


# ------------------------------------------------------ 4. immutability

def test_immutability_is_enforced_per_job(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0))
    with pytest.raises(ValueError, match="immutable"):
        s.put(art("jobA", 0, payload=plan(99.0)))
    # the same revision in a DIFFERENT job is fine
    s.put(art("jobB", 0))


def test_revisions_still_accumulate(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0, payload=plan(4.0)))
    s.put(art("jobA", 1, payload=plan(8.0)))
    assert s.get("jobA", "EditPlan", EditPlan).payload.total_duration == 8.0
    assert len(s.all_revisions("jobA", "EditPlan", EditPlan)) == 2


def test_purge_removes_only_the_named_job(tmp_path):
    """The mitigation for the one real hazard sharing introduces: deleting a
    job folder no longer deletes its artifacts, so reusing that id would merge
    two unrelated builds."""
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0))
    s.put(art("jobB", 0))
    assert s.purge_job("jobA") == 1
    assert not s.has_job("jobA")
    assert s.has_job("jobB")
    assert s.next_revision("jobA", "EditPlan") == 0    # clean slate


# ------------------------------------------------------------ 5. wal mode

def test_the_shared_store_uses_wal(tmp_path):
    """One file means concurrent jobs now contend for it; the per-job files
    never had to."""
    s = shared_store(root=tmp_path)
    con = sqlite3.connect(s.db_path)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        con.close()


def test_a_second_connection_can_read_while_another_is_open(tmp_path):
    s = shared_store(root=tmp_path)
    s.put(art("jobA", 0))
    other = ArtifactStore(shared_db_path(root=tmp_path))
    assert other.get("jobA", "EditPlan", EditPlan) is not None


# --------------------------------------------------------- 6. migration

def _make_legacy_job(tmp_path, job_id, revisions=1):
    build = tmp_path / "outputs" / job_id
    build.mkdir(parents=True, exist_ok=True)
    db = build / "agent_state.db"
    s = ArtifactStore(db)
    t = TraceRecorder(db)
    for r in range(revisions):
        s.put(art(job_id, r, payload=plan(4.0 + r)))
    with t.node(job_id, "clip_analysis", "clip_analysis") as span:
        span.tool("pipeline.probe")
    return db


def test_migration_copies_artifacts_and_traces(tmp_path):
    _make_legacy_job(tmp_path, "oldjob", revisions=2)
    results = migrate.migrate_all(root=tmp_path)
    assert len(results) == 1
    r = results[0]
    assert r.ok and r.artifacts_copied == 2 and r.traces_copied == 1

    s = shared_store(root=tmp_path)
    assert s.has_job("oldjob")
    assert len(s.all_revisions("oldjob", "EditPlan", EditPlan)) == 2
    assert len(shared_tracer(root=tmp_path).for_job("oldjob")) == 1


def test_migration_is_idempotent(tmp_path):
    _make_legacy_job(tmp_path, "oldjob", revisions=2)
    first = migrate.migrate_all(root=tmp_path)[0]
    second = migrate.migrate_all(root=tmp_path)[0]
    assert first.artifacts_copied == 2
    assert second.artifacts_copied == 0 and second.traces_copied == 0
    assert second.ok
    assert len(shared_store(root=tmp_path).all_revisions(
        "oldjob", "EditPlan", EditPlan)) == 2


def test_migration_never_deletes_the_source(tmp_path):
    db = _make_legacy_job(tmp_path, "oldjob")
    before = db.read_bytes()
    migrate.migrate_all(root=tmp_path)
    assert db.exists() and db.read_bytes() == before


def test_dry_run_writes_nothing_but_reports_what_would_move(tmp_path):
    _make_legacy_job(tmp_path, "oldjob", revisions=2)
    r = migrate.migrate_all(root=tmp_path, dry_run=True)[0]
    assert r.artifacts_copied == 2 and r.traces_copied == 1
    assert not shared_store(root=tmp_path).has_job("oldjob")


def test_partially_migrated_jobs_are_completed_not_duplicated(tmp_path):
    db = _make_legacy_job(tmp_path, "oldjob", revisions=3)
    # simulate an interrupted migration: copy only revision 0 across
    src = ArtifactStore(db)
    shared = shared_store(root=tmp_path)
    shared.put(src.get("oldjob", "EditPlan", EditPlan, revision=0))

    r = migrate.migrate_all(root=tmp_path)[0]
    assert r.ok and r.artifacts_copied == 2      # the two that were missing
    assert len(shared.all_revisions("oldjob", "EditPlan", EditPlan)) == 3


def test_a_genuine_id_clash_is_reported_not_silently_dropped(tmp_path):
    """Two different builds under one job id must surface, because
    INSERT OR IGNORE would otherwise drop the incoming rows without a word."""
    _make_legacy_job(tmp_path, "oldjob", revisions=1)
    shared = shared_store(root=tmp_path)
    # a DIFFERENT artifact already occupying (oldjob, EditPlan, 0)
    shared.put(art("oldjob", 0, payload=plan(77.0)))

    r = migrate.migrate_all(root=tmp_path)[0]
    assert not r.ok
    assert r.conflicts and "EditPlan rev0" in r.conflicts[0]
    assert r.artifacts_copied == 0
    # the incumbent was left untouched
    assert shared.get("oldjob", "EditPlan",
                      EditPlan).payload.total_duration == 77.0


def test_the_shared_folder_is_never_treated_as_a_job_to_migrate(tmp_path):
    shared_store(root=tmp_path)
    _make_legacy_job(tmp_path, "oldjob")
    assert [j for j, _ in migrate.legacy_stores(root=tmp_path)] == ["oldjob"]


# --------------------------------------------- 7. fallback for old jobs

def test_an_unmigrated_job_still_resolves_to_its_own_database(tmp_path):
    """Migration must be a tidy-up, not a prerequisite."""
    _make_legacy_job(tmp_path, "oldjob")
    db, source = resolve_job_db("oldjob", root=tmp_path)
    assert source == "legacy"
    assert db == legacy_db_path("oldjob", root=tmp_path)
    assert ArtifactStore(db).get("oldjob", "EditPlan", EditPlan) is not None


def test_after_migration_the_shared_store_takes_over(tmp_path):
    _make_legacy_job(tmp_path, "oldjob")
    migrate.migrate_all(root=tmp_path)
    db, source = resolve_job_db("oldjob", root=tmp_path)
    assert source == "shared"
    assert db == shared_db_path(root=tmp_path)


def test_an_unknown_job_resolves_to_the_shared_store(tmp_path):
    _, source = resolve_job_db("nosuchjob", root=tmp_path)
    assert source == "shared"


# ------------------------------- 8. the real Phase 1 jobs are still readable

@pytest.mark.parametrize("job_id", ["phase1demo", "phase1revise"])
def test_the_real_phase1_jobs_remain_readable(job_id):
    """Against the actual repository, not a fixture."""
    from agents.artifacts import ClipAnalysisSet, CriticResult, RenderResult

    repo = Path(__file__).resolve().parents[2]
    outputs = repo / "outputs"
    if not (outputs / job_id).exists():
        pytest.skip(f"{job_id} is not present in this checkout")

    db, source = resolve_job_db(job_id, outputs=outputs)
    store = ArtifactStore(db)
    assert store.has_job(job_id), f"{job_id} unreadable from the {source} store"
    assert store.get(job_id, "ClipAnalysis", ClipAnalysisSet) is not None
    assert store.get(job_id, "RenderResult", RenderResult) is not None
    assert store.get(job_id, "CriticResult", CriticResult) is not None
    assert TraceRecorder(db).for_job(job_id), "no trace rows"
