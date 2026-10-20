"""No sign-in: what anyone with the link can and cannot do, and the
progress estimate the page shows.

With `AT_CLOUD_OPEN_ACCESS=1` the page asks for no token. What keeps that
safe enough for a link shared with a few people:
* a job is reached only by its id (48 random bits), so each browser sees
  the jobs it made and nobody can list anyone else's;
* new jobs are capped per day, so a leaked link cannot spend the credit;
* the admin routes (the full list, reconcile) still need the token.
"""

from __future__ import annotations

import os

import pytest

from cloud import estimate as E
from cloud import queue as Q
from cloud.config import CloudConfig
from fakes import API_TOKEN, cloud_env
from test_api import create, send, spec

ADMIN = {"Authorization": f"Bearer {API_TOKEN}"}
NOBODY: dict = {}


@pytest.fixture
def open_plane(tmp_path, drive, compute, clock):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cloud.app import mount

    env = cloud_env(tmp_path, AT_CLOUD_OPEN_ACCESS="1",
                    AT_CLOUD_MAX_JOBS_PER_DAY="2")
    app = FastAPI()
    ctx = mount(app, host="0.0.0.0", env=env, drive=drive, compute=compute,
                clock=clock)
    assert ctx is not None
    with TestClient(app) as client:
        yield client, ctx


@pytest.fixture
def clip():
    return os.urandom(300_000)


# ------------------------------------------------------------ configuration

def test_open_access_is_an_explicit_choice(tmp_path):
    shut = cloud_env(tmp_path, AT_CLOUD_API_TOKEN="")
    assert any("AT_CLOUD_API_TOKEN" in p for p in
               CloudConfig.from_env(shut).problems("control", "0.0.0.0"))
    open_ = cloud_env(tmp_path, AT_CLOUD_API_TOKEN="",
                      AT_CLOUD_OPEN_ACCESS="1")
    assert not any("AT_CLOUD_API_TOKEN" in p for p in
                   CloudConfig.from_env(open_).problems("control", "0.0.0.0"))


def test_the_page_is_told_no_sign_in_is_needed(open_plane):
    client, _ = open_plane
    cfg = client.get("/api/cloud/config").json()
    assert cfg["auth_required"] is False and cfg["open_access"] is True


# ---------------------------------------------------- what a visitor can do

def test_a_visitor_makes_a_video_without_a_token(open_plane, clip):
    client, ctx = open_plane
    made = create(client, NOBODY, [spec(clip)])
    send(client, NOBODY, made["files"][0]["url"], clip)
    jid = made["job"]["job_id"]
    r = client.post(f"/api/cloud/uploads/{jid}/complete", json={})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/cloud/jobs/{jid}").status_code == 200


def test_a_visitor_sees_only_the_jobs_it_names(open_plane, clip):
    client, _ = open_plane
    mine = create(client, NOBODY, [spec(clip)], key="mine-00000001")
    theirs = create(client, NOBODY, [spec(clip)], key="theirs-0000001")
    assert client.get("/api/cloud/jobs").json()["jobs"] == [], \
        "no ids, no list"
    got = client.get("/api/cloud/jobs",
                     params={"ids": mine["job"]["job_id"]}).json()["jobs"]
    assert [j["job_id"] for j in got] == [mine["job"]["job_id"]]
    assert theirs["job"]["job_id"] not in str(got)


def test_made_up_or_malformed_ids_are_ignored(open_plane):
    client, _ = open_plane
    r = client.get("/api/cloud/jobs", params={"ids": "abcdef123456,../x,zz"})
    assert r.status_code == 200 and r.json()["jobs"] == []


def test_the_admin_token_still_lists_everything(open_plane, clip):
    client, _ = open_plane
    create(client, NOBODY, [spec(clip)], key="anyone-0000001")
    assert len(client.get("/api/cloud/jobs", headers=ADMIN).json()["jobs"]) == 1


def test_reconcile_stays_admin_only(open_plane):
    client, _ = open_plane
    assert client.post("/api/cloud/reconcile").status_code == 401
    assert client.post("/api/cloud/reconcile", headers=ADMIN).status_code == 200


# ---------------------------------------------------------- the daily cap

def test_new_videos_are_capped_per_day(open_plane, clip):
    client, _ = open_plane
    create(client, NOBODY, [spec(clip)], key="day-000000001")
    create(client, NOBODY, [spec(clip)], key="day-000000002")
    r = client.post("/api/cloud/uploads", json={
        "idempotency_key": "day-000000003", "files": [spec(clip)],
        "settings": {}})
    assert r.status_code == 429
    assert "limit" in r.json()["message"].lower()


def test_resuming_an_upload_is_not_a_new_video(open_plane, clip):
    client, _ = open_plane
    a = create(client, NOBODY, [spec(clip)], key="day-000000001")
    create(client, NOBODY, [spec(clip)], key="day-000000002")
    again = create(client, NOBODY, [spec(clip)], key="day-000000001")
    assert again["job"]["job_id"] == a["job"]["job_id"]


def test_the_cap_resets_after_a_day(open_plane, clip, clock):
    client, _ = open_plane
    create(client, NOBODY, [spec(clip)], key="day-000000001")
    create(client, NOBODY, [spec(clip)], key="day-000000002")
    clock.advance(86_401)
    create(client, NOBODY, [spec(clip)], key="day-000000003")


def test_the_admin_is_not_capped(open_plane, clip):
    client, _ = open_plane
    for i in range(3):
        create(client, ADMIN, [spec(clip)], key=f"admin-00000{i}")


# ------------------------------------------- choices made after the upload

def test_the_settings_sent_with_create_are_the_ones_used(open_plane, clip):
    client, ctx = open_plane
    made = create(client, NOBODY, [spec(clip)], settings={})
    send(client, NOBODY, made["files"][0]["url"], clip)
    jid = made["job"]["job_id"]
    r = client.post(f"/api/cloud/uploads/{jid}/complete", json={
        "settings": {"title": "Chosen later", "style": "cinematic",
                     "target_seconds": 30}})
    assert r.status_code == 200, r.text
    s = ctx.queue.get(jid)["settings"]
    assert s["title"] == "Chosen later" and s["style"] == "cinematic"


def test_bad_settings_at_create_are_refused(open_plane, clip):
    client, _ = open_plane
    made = create(client, NOBODY, [spec(clip)], settings={})
    send(client, NOBODY, made["files"][0]["url"], clip)
    jid = made["job"]["job_id"]
    r = client.post(f"/api/cloud/uploads/{jid}/complete",
                    json={"settings": {"style": "not-a-style"}})
    assert r.status_code == 400


def test_settings_cannot_change_once_queued(queue):
    from test_queue import queued
    jid = queued(queue)
    with pytest.raises(Exception):
        queue.update_settings(jid, {"title": "too late"})


# --------------------------------------------------------- the estimate

def _job(state, mb=50, progress=None):
    return {"state": state, "progress": progress or {},
            "sources": [{"size": int(mb * 1e6)}]}


def _events(**at):
    return [{"to": k, "at": v} for k, v in at.items()]


HIST = E.learn([])                       # the defaults: no history yet


def test_nothing_is_promised_while_uploading_or_waiting_to_retry():
    for state in ("uploading", "retrying", "failed", "cancelled"):
        assert E.estimate(_job(state), [], "stopped", HIST, 0) is None


def test_a_finished_job_is_a_hundred_percent():
    got = E.estimate(_job("completed"), [], "ready", HIST, 0)
    assert got["percent"] == 100 and got["remaining_s"] == 0


def test_more_agents_done_is_further_along_and_closer_to_done():
    ev = _events(queued=0, downloading=50, processing=60)
    early = E.estimate(_job("processing", progress={
        "nodes_done": ["clip_analysis"], "current_node": "vision",
        "current_started_at": 100}), ev, "ready", HIST, 100)
    late = E.estimate(_job("processing", progress={
        "nodes_done": E.NODES[:11], "current_node": "render",
        "current_started_at": 300}), ev, "ready", HIST, 300)
    assert 0 < early["percent"] < late["percent"] < 100
    assert early["remaining_s"] > late["remaining_s"] > 0


def test_a_step_that_overruns_never_counts_below_zero():
    ev = _events(queued=0, downloading=50, processing=60)
    got = E.estimate(_job("processing", progress={
        "nodes_done": E.NODES[:11], "current_node": "render",
        "current_started_at": 0}), ev, "ready", HIST, 100_000)
    assert got["remaining_s"] >= 0 and got["percent"] <= 99


def test_a_sleeping_worker_adds_its_start_up_time():
    ev = _events(queued=0)
    cold = E.estimate(_job("queued"), ev, "stopped", HIST, 0)
    warm = E.estimate(_job("queued"), ev, "ready", HIST, 0)
    assert cold["remaining_s"] > warm["remaining_s"]


def test_more_footage_means_longer_to_measure():
    ev = _events(queued=0)
    small = E.estimate(_job("queued", mb=50), ev, "ready", HIST, 0)
    big = E.estimate(_job("queued", mb=2000), ev, "ready", HIST, 0)
    assert big["remaining_s"] > small["remaining_s"]


def test_history_replaces_the_defaults_with_what_was_measured():
    job = _job("completed", mb=100)
    ev = _events(queued=0, starting_vm=0, downloading=40, processing=48,
                 uploading_results=400, completed=410)
    trace = [{"node": "render", "status": "ok", "started_at": 100,
              "ended_at": 400}]
    hist = E.learn([(job, ev, trace)])
    assert hist["jobs"] == 1 and hist["node_s"]["render"] == 300
    assert hist["cold_start_s"] == 40 and hist["save_s"] == 10
    assert hist["node_s"]["critic"] == E.DEFAULT_NODE_S["critic"]


def test_the_job_view_carries_the_estimate(open_plane, clip):
    client, _ = open_plane
    made = create(client, NOBODY, [spec(clip)])
    send(client, NOBODY, made["files"][0]["url"], clip)
    jid = made["job"]["job_id"]
    client.post(f"/api/cloud/uploads/{jid}/complete", json={})
    got = client.get(f"/api/cloud/jobs/{jid}").json()["estimate"]
    assert set(got) == {"percent", "remaining_s", "based_on"}
    assert 0 < got["percent"] < 100 and got["remaining_s"] > 0
