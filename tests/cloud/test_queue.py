"""The durable queue: states, idempotency, atomic claims, leases, fencing.

Brief, section 13: tests 1, 2, 4 and 6 live here.
"""

from __future__ import annotations

import threading

import pytest

from cloud import queue as Q
from cloud.sanitize import CloudError

FILES = [{"name": "01_a.mp4", "size": 1000, "mime": "video/mp4"}]


def uploaded(queue, key="key-00000001", files=FILES):
    """A job whose sources are all verified, ready to be queued."""
    job, _ = queue.create(key, files, {})
    for s in job["sources"]:
        queue.mark_source_uploaded(job["job_id"], s["idx"],
                                   f"drive_file_{s['idx']:08d}", "ab" * 16,
                                   s["size"])
    return job["job_id"]


def queued(queue, key="key-00000001"):
    jid = uploaded(queue, key)
    queue.queue_if_uploaded(jid)
    return jid


# ---------------------------------------------------- 1. exactly one job

def test_a_completed_upload_creates_exactly_one_job(queue):
    """The browser retries the create call; the upload intent is one."""
    a, created_a = queue.create("intent-0001", FILES, {"title": "x"})
    b, created_b = queue.create("intent-0001", FILES, {"title": "x"})
    assert created_a and not created_b
    assert a["job_id"] == b["job_id"]
    assert len(queue.recent()) == 1


def test_completing_twice_queues_once(queue):
    jid = uploaded(queue)
    _job, first = queue.queue_if_uploaded(jid)
    _job, second = queue.queue_if_uploaded(jid)
    assert first is True and second is False
    states = [e["to_state"] for e in queue.events(jid)]
    assert states.count(Q.QUEUED) == 1


def test_concurrent_creates_with_one_key_make_one_job(queue):
    ids, errors = [], []

    def go():
        try:
            ids.append(queue.create("same-intent-01", FILES, {})[0]["job_id"])
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(ids)) == 1


# ------------------------------------- 2. an incomplete upload never runs

def test_an_incomplete_upload_is_never_queued(queue):
    job, _ = queue.create("partial-0001", FILES, {})
    queue.record_received(job["job_id"], 0, 500)          # half of it
    with pytest.raises(CloudError) as e:
        queue.queue_if_uploaded(job["job_id"])
    assert e.value.category == "invalid"
    assert queue.get(job["job_id"])["state"] == Q.UPLOADING


def test_an_uploading_job_is_never_claimable(queue):
    queue.create("partial-0002", FILES, {})
    assert queue.claim("w1") is None
    assert queue.counts()["claimable"] == 0


def test_one_missing_file_of_two_keeps_the_job_back(queue):
    files = FILES + [{"name": "02_b.mp4", "size": 2000, "mime": "video/mp4"}]
    job, _ = queue.create("two-files-01", files, {})
    queue.mark_source_uploaded(job["job_id"], 0, "drive_file_00000001",
                               "ab" * 16, 1000)
    with pytest.raises(CloudError):
        queue.queue_if_uploaded(job["job_id"])


# ---------------------------------------- 4. no duplicate claims, ever

def test_simultaneous_claims_never_share_a_job(queue):
    """Twenty claimers, five jobs: each job claimed exactly once."""
    for i in range(5):
        queued(queue, f"key-{i:08d}")
    got, errors = [], []
    barrier = threading.Barrier(20)

    def claimer(n):
        barrier.wait()
        try:
            c = queue.claim(f"worker-{n}")
            if c:
                got.append(c.job_id)
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=claimer, args=(n,))
               for n in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert len(got) == 5 and len(set(got)) == 5


def test_claims_come_out_oldest_first(queue, clock):
    a = queued(queue, "key-aaaaaaa1")
    clock.advance(5)
    b = queued(queue, "key-bbbbbbb2")
    assert queue.claim("w").job_id == a
    assert queue.claim("w").job_id == b


# ------------------------------------------- 6. a dead worker's job recovers

def test_a_crashed_worker_lease_expires_and_the_job_retries(queue, clock):
    jid = queued(queue)
    c = queue.claim("dead-worker")
    clock.advance(queue.lease_s + 1)
    assert queue.expire_leases() == [jid]
    job = queue.get(jid)
    assert job["state"] == Q.RETRYING
    assert job["lease_owner"] is None
    assert job["error_category"] == "lease_expired"
    clock.advance(job["not_before"] - clock() + 1)
    again = queue.claim("new-worker")
    assert again.job_id == jid and again.claim_token == c.claim_token + 1
    assert again.attempt == 2


def test_a_live_heartbeat_keeps_the_lease(queue, clock):
    jid = queued(queue)
    c = queue.claim("w")
    for _ in range(5):
        clock.advance(queue.lease_s * 0.8)
        queue.heartbeat(jid, "w", c.claim_token)
    assert queue.expire_leases() == []


def test_a_zombie_worker_is_fenced_out(queue, clock):
    """Slow, not dead: it comes back after its lease was given away, and
    must not be able to complete, fail or heartbeat the job."""
    jid = queued(queue)
    old = queue.claim("slow")
    clock.advance(queue.lease_s + 1)
    queue.expire_leases()
    clock.advance(10_000)
    new = queue.claim("fresh")
    for call in (lambda: queue.heartbeat(jid, "slow", old.claim_token),
                 lambda: queue.fail(jid, "slow", old.claim_token,
                                    "transient", "x"),
                 lambda: queue.complete(jid, "slow", old.claim_token,
                                        [_video()])):
        with pytest.raises(Q.LeaseLost):
            call()
    assert queue.get(jid)["lease_owner"] == "fresh"
    queue.heartbeat(jid, "fresh", new.claim_token)


def test_attempts_are_bounded(queue, clock):
    jid = queued(queue)
    for _ in range(queue.max_attempts):
        c = queue.claim("w")
        assert c is not None
        clock.advance(queue.lease_s + 1)
        queue.expire_leases()
        clock.advance(4000)
    job = queue.get(jid)
    assert job["state"] == Q.FAILED
    assert job["attempt"] == queue.max_attempts
    assert queue.claim("w") is None


# ------------------------------------------------- 8. no completion without

def _video(**kw):
    return {"kind": "video", "name": "v.mp4", "size": 10, "md5": "a" * 32,
            "drive_file_id": "drive_out_0000001", "verified": True, **kw}


def _to_uploading_results(queue, jid, worker="w"):
    c = queue.claim(worker)
    queue.heartbeat(jid, worker, c.claim_token, Q.PROCESSING)
    queue.heartbeat(jid, worker, c.claim_token, Q.UPLOADING_RESULTS)
    return c


@pytest.mark.parametrize("outputs,why", [
    ([], "no outputs"),
    ([_video(verified=False)], "not verified"),
    ([_video(drive_file_id=None)], "no Drive file id"),
    ([_video(size=0)], "no size"),
    ([{"kind": "thumbnail", "name": "t.jpg", "size": 5,
       "drive_file_id": "drive_out_0000002", "verified": True}],
     "finished video"),
])
def test_unverified_outputs_never_complete_a_job(queue, outputs, why):
    jid = queued(queue)
    c = _to_uploading_results(queue, jid)
    with pytest.raises(CloudError) as e:
        queue.complete(jid, "w", c.claim_token, outputs)
    assert why in e.value.detail
    assert queue.get(jid)["state"] == Q.UPLOADING_RESULTS


def test_completion_is_refused_before_results_are_uploading(queue):
    jid = queued(queue)
    c = queue.claim("w")
    with pytest.raises(Q.InvalidTransition):
        queue.complete(jid, "w", c.claim_token, [_video()])


def test_a_verified_completion_lands(queue):
    jid = queued(queue)
    c = _to_uploading_results(queue, jid)
    job = queue.complete(jid, "w", c.claim_token, [_video()],
                         {"trace": [{"node": "render"}]})
    assert job["state"] == Q.COMPLETED and job["lease_owner"] is None
    assert queue.trace(jid)["trace"][0]["node"] == "render"


# ------------------------------------------------------------ the machine

def test_the_state_machine_refuses_skipping_steps(queue):
    jid = queued(queue)
    c = queue.claim("w")
    with pytest.raises(Q.InvalidTransition):
        queue.heartbeat(jid, "w", c.claim_token, Q.COMPLETED)
    with pytest.raises(Q.InvalidTransition):
        queue.heartbeat(jid, "w", c.claim_token, Q.UPLOADING_RESULTS)


def test_every_transition_is_in_the_audit_trail(queue):
    jid = queued(queue)
    c = _to_uploading_results(queue, jid)
    queue.complete(jid, "w", c.claim_token, [_video()])
    path = [e["to_state"] for e in queue.events(jid)]
    assert path == [Q.UPLOADING, Q.QUEUED, Q.DOWNLOADING, Q.PROCESSING,
                    Q.UPLOADING_RESULTS, Q.COMPLETED]


def test_terminal_states_have_no_way_out():
    for s in Q.TERMINAL:
        assert Q.TRANSITIONS[s] == frozenset()


# ------------------------------------------------------------ retry policy

def test_a_retryable_failure_backs_off(queue, clock):
    jid = queued(queue)
    c = queue.claim("w")
    job = queue.fail(jid, "w", c.claim_token, "transient", "503")
    assert job["state"] == Q.RETRYING
    assert job["not_before"] >= clock() + queue.retry_base_s - 1
    assert queue.claim("w") is None, "not before its backoff has passed"


def test_a_rate_limit_waits_long_enough_to_matter(queue, clock):
    """Retrying into a closed rate-limit window only keeps a worker VM
    billing. The backoff is at least the configured rate-limit pause."""
    jid = queued(queue)
    c = queue.claim("w")
    job = queue.fail(jid, "w", c.claim_token, "rate_limited", "429")
    assert job["not_before"] - clock() >= queue.rate_limit_backoff_s


def test_a_non_retryable_failure_is_terminal(queue):
    jid = queued(queue)
    c = queue.claim("w")
    job = queue.fail(jid, "w", c.claim_token, "pipeline",
                     "no readable clips")
    assert job["state"] == Q.FAILED


def test_error_messages_are_redacted_before_storage(queue):
    jid = queued(queue)
    c = queue.claim("w")
    queue.fail(jid, "w", c.claim_token, "transient",
               "auth Bearer ya29.abcdefSECRET failed at "
               "https://x/?upload_id=SECRETSESSION")
    job = queue.get(jid)
    assert "SECRET" not in job["error_message"]
    assert all("SECRET" not in (e["detail"] or "")
               for e in queue.events(jid))


# ------------------------------------------------------------- cancelling

def test_cancelling_a_queued_job_is_immediate(queue):
    jid = queued(queue)
    assert queue.cancel(jid)["state"] == Q.CANCELLED
    assert queue.claim("w") is None


def test_cancelling_a_running_job_flags_it_for_its_worker(queue):
    jid = queued(queue)
    c = queue.claim("w")
    queue.cancel(jid)
    assert queue.heartbeat(jid, "w", c.claim_token) == {"cancel": True}
    job = queue.fail(jid, "w", c.claim_token, "cancelled", "stopped")
    assert job["state"] == Q.CANCELLED


def test_a_cancelled_job_whose_worker_died_ends_cancelled(queue, clock):
    jid = queued(queue)
    queue.claim("w")
    queue.cancel(jid)
    clock.advance(queue.lease_s + 1)
    queue.expire_leases()
    assert queue.get(jid)["state"] == Q.CANCELLED


# -------------------------------------------------------- stale and lost

def test_an_abandoned_upload_fails_visibly(queue, clock):
    job, _ = queue.create("abandoned-01", FILES, {})
    clock.advance(25 * 3600)
    assert queue.expire_stale(24 * 3600, 48 * 3600) == [job["job_id"]]
    assert queue.get(job["job_id"])["state"] == Q.FAILED


def test_the_session_uri_is_never_in_a_job_view(queue):
    job, _ = queue.create("session-0001", FILES, {})
    queue.set_session(job["job_id"], 0, "https://up/?upload_id=CAPABILITY")
    assert "CAPABILITY" not in repr(queue.get(job["job_id"]))
    assert "CAPABILITY" not in repr(queue.recent())
    assert queue.session(job["job_id"], 0)["session_uri"].endswith(
        "CAPABILITY")


def test_a_verified_source_forgets_its_session(queue):
    job, _ = queue.create("session-0002", FILES, {})
    queue.set_session(job["job_id"], 0, "https://up/?upload_id=CAP")
    queue.mark_source_uploaded(job["job_id"], 0, "drive_file_00000009",
                               "cd" * 16, 1000)
    assert queue.session(job["job_id"], 0)["session_uri"] is None
