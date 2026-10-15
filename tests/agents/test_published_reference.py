"""Recording which video a build became, through the API.

Nothing in a build can know its own YouTube id - the upload happens
outside this system - so the one link between an edit and its reception
is given by hand, once. Without it the Learning agent has nothing to
learn FROM: a view count that cannot be attributed to a decision teaches
nothing, and a view count attributed to the WRONG decision teaches the
opposite of the truth.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import api as AGENT_API                        # noqa: E402
from agents.tools import analytics_store as AS             # noqa: E402

JOB = "testjob01"
VID = "dQw4w9WgXcQ"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A job that exists, an empty export folder, nothing else."""
    build = tmp_path / "outputs" / JOB
    build.mkdir(parents=True)
    exports = tmp_path / "analytics"
    exports.mkdir()

    monkeypatch.setattr(AGENT_API, "_require_agent_job", lambda j: None)
    monkeypatch.setattr(AS, "export_dir", lambda: exports)

    from agents import runner
    monkeypatch.setattr(runner, "job_paths",
                        lambda j, *a, **k: (tmp_path / "uploads" / j, build))

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(AGENT_API.router)
    c = TestClient(app)
    c.build, c.exports = build, exports
    return c


def _post(c, url):
    return c.post(f"/api/v2/jobs/{JOB}/published", json={"url": url})


# ------------------------------------------------------------ recording

def test_a_link_is_recorded(client):
    r = _post(client, f"https://youtu.be/{VID}")
    assert r.status_code == 200 and r.json()["video_id"] == VID
    assert AS.load_reference(client.build)["video_id"] == VID


def test_a_studio_url_works_too(client):
    assert _post(client,
                 f"https://studio.youtube.com/video/{VID}/edit"
                 ).json()["video_id"] == VID


def test_something_that_is_not_a_video_is_refused(client):
    r = _post(client, "https://www.youtube.com/@mychannel")
    assert r.status_code == 422
    assert AS.load_reference(client.build) is None, \
        "a wrong id would credit one video's numbers to another's decisions"


def test_an_empty_body_is_refused(client):
    assert client.post(f"/api/v2/jobs/{JOB}/published", json={}).status_code \
        == 422


def test_a_reference_can_be_withdrawn(client):
    _post(client, VID)
    assert client.delete(f"/api/v2/jobs/{JOB}/published").json()["cleared"]
    assert AS.load_reference(client.build) is None


# ------------------------------------------------------------- readback

def test_before_anything_is_recorded_it_says_so(client):
    body = client.get(f"/api/v2/jobs/{JOB}/published").json()
    assert body["video_id"] is None and body["matched"] is False
    assert "no video" in body["why"]


def test_recorded_but_no_export_yet_is_not_zeroes(client):
    _post(client, VID)
    body = client.get(f"/api/v2/jobs/{JOB}/published").json()
    assert body["matched"] is False
    assert "metrics" not in body, "absent data must not read as zero views"
    assert "data/analytics" in body["why"]


def test_the_numbers_come_back_once_the_export_is_there(client):
    _post(client, VID)
    (client.exports / "oct.csv").write_text(
        "title,video_id,views,watch_time_minutes,likes,subscribers_gained,"
        "first_day,last_day\n"
        f"LEGO build,{VID},10432,8120.5,310,47,2026-09-01,2026-09-30\n",
        encoding="utf-8")

    body = client.get(f"/api/v2/jobs/{JOB}/published").json()
    assert body["matched"] is True
    m = body["metrics"]
    assert m["views"] == 10432 and m["subscribers_gained"] == 47
    assert m["window_days"] == 30
    assert m["source_file"] == "oct.csv"


def test_an_export_without_this_video_says_which_files_were_read(client):
    _post(client, VID)
    (client.exports / "oct.csv").write_text(
        "title,video_id,views\nother,aaaaaaaaaaa,5\n", encoding="utf-8")

    body = client.get(f"/api/v2/jobs/{JOB}/published").json()
    assert body["matched"] is False and "oct.csv" in body["why"]


def test_the_wrong_query_is_reported_back_to_the_operator(client):
    _post(client, VID)
    (client.exports / "wrong.csv").write_text("channel,subs\nmine,42\n",
                                              encoding="utf-8")
    body = client.get(f"/api/v2/jobs/{JOB}/published").json()
    assert any("wrong.csv" in s for s in body["export_skipped"])
