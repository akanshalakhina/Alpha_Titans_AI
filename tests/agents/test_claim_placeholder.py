"""A video made through the UI must be downloadable.

THE BUG THIS PINS
-----------------
The browser's create flow calls `POST /api/jobs`, then
`POST /api/v2/jobs/{id}/produce`. The first writes a legacy job.json.
`compat.project()` refuses to overwrite a legacy job.json - correctly, in
general, because clobbering a real build's state file would corrupt it -
and `_auto_project` swallows that refusal.

So every run through the UI finished, rendered a verified video, passed the
critic, and left `output: null` with `status: "draft"`. The legacy
`/api/jobs/{id}/download` route reads `job.output`, so EVERY video made in
the browser answered 409 while the finished mp4 sat on disk beside it. Only
CLI-built jobs, which never create a legacy record, could be downloaded.

The fix claims the placeholder - and ONLY while it is still a placeholder,
so a real legacy build is never touched.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import compat                              # noqa: E402

PENDING = {"probe": {"id": "probe", "status": "pending"},
           "measure": {"id": "measure", "status": "pending"}}


def _placeholder(**over) -> dict:
    """Exactly what `POST /api/jobs` stamps out."""
    d = {"id": "j1", "settings": {}, "created": 1.0, "status": "draft",
         "error": None, "started": None, "finished": None,
         "output": None, "stages": json.loads(json.dumps(PENDING))}
    d.update(over)
    return d


@pytest.fixture
def job(tmp_path, monkeypatch):
    build = tmp_path / "outputs" / "j1"
    build.mkdir(parents=True)
    monkeypatch.setattr(compat, "job_paths",
                        lambda jid, root=None: (tmp_path / "uploads" / jid,
                                                tmp_path / "outputs" / jid))
    return build / compat.JOB_FILE


def test_an_empty_placeholder_is_claimed(job):
    job.write_text(json.dumps(_placeholder()))
    assert compat.claim_placeholder("j1") is True
    after = json.loads(job.read_text())
    assert after["settings"]["_produced_by"] == compat.MARKER
    assert compat._is_our_projection(job)


def test_no_file_at_all_is_nothing_in_the_way(job):
    assert compat.claim_placeholder("j1") is True


def test_a_record_we_already_own_is_left_as_is(job):
    d = _placeholder(settings={"_produced_by": compat.MARKER, "keep": 1})
    job.write_text(json.dumps(d))
    assert compat.claim_placeholder("j1") is True
    assert json.loads(job.read_text())["settings"]["keep"] == 1


@pytest.mark.parametrize("over", [
    {"status": "done"},
    {"status": "running"},
    {"output": {"file": "real.mp4"}},
    {"started": 123.0},
    {"finished": 456.0},
    {"error": "it broke"},
    {"stages": {"probe": {"id": "probe", "status": "done"}}},
])
def test_a_real_legacy_build_is_never_claimed(job, over):
    """Anything that has actually been built keeps its own state file."""
    job.write_text(json.dumps(_placeholder(**over)))
    assert compat.claim_placeholder("j1") is False
    assert "_produced_by" not in json.loads(job.read_text())["settings"]


def test_an_unreadable_record_is_left_alone(job):
    job.write_text("{not json")
    assert compat.claim_placeholder("j1") is False


def test_project_succeeds_once_the_placeholder_is_claimed(job, monkeypatch):
    """The end-to-end point: refusing here is what broke /download."""
    job.write_text(json.dumps(_placeholder()))
    monkeypatch.setattr(compat, "build_job_json",
                        lambda jid, root=None: {"id": jid, "status": "done",
                                                "output": {"file": "v.mp4"},
                                                "settings": {
                                                    "_produced_by":
                                                        compat.MARKER}})
    monkeypatch.setattr(compat, "project_edit_plan",
                        lambda jid, root=None: None)

    with pytest.raises(FileExistsError):
        compat.project("j1")             # the behaviour before the fix

    assert compat.claim_placeholder("j1") is True
    out = compat.project("j1")           # now it lands
    assert out["output"]["file"] == "v.mp4"
    assert json.loads(job.read_text())["status"] == "done"


# ------------------------- the second half of the same bug: a stale cache

def test_the_legacy_record_is_evicted_after_a_run():
    """Claiming the file was not enough on its own.

    `POST /api/jobs` puts a Job in `jobs._JOBS`; the agent path then rewrites
    job.json on disk behind that object's back. The legacy routes read the
    in-memory copy, so `/api/jobs/{id}/download` kept answering 409 - and the
    result page's <video>, which streams from that same route, kept playing
    nothing - until the server was restarted. Observed end to end: job.json
    on disk said `status: done, output: set` while the live server reported
    `status: draft, output: NULL` for the same job.
    """
    import jobs as jobs_mod
    from agents.api import _refresh_legacy_record

    job = jobs_mod.create({})
    assert job.id in jobs_mod._JOBS
    _refresh_legacy_record(job.id)
    assert job.id not in jobs_mod._JOBS, "the stale object must be dropped"
    # jobs.get() restores from job.json whenever the id is absent, so the
    # next read sees whatever the projection wrote
    assert jobs_mod.get(job.id) is not None


def test_evicting_an_unknown_job_is_harmless():
    from agents.api import _refresh_legacy_record
    _refresh_legacy_record("never-existed")          # must not raise
