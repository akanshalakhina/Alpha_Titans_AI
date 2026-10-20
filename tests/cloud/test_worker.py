"""The worker, end to end, against the REAL control plane.

The production worker client talks HTTP to the production routes (through
TestClient), which drive the production queue and VM manager. Only Compute,
Drive and the pipeline itself are stand-ins - the pipeline because a real
render takes minutes and is proven elsewhere; here what matters is what the
worker does AROUND it.

Brief, section 13: tests 7, 8, 9, 10, 12 and 13, worker side.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cloud import queue as Q
from cloud.worker import (EXIT_NOT_READY, EXIT_POWEROFF, SubprocessPipeline,
                          Worker)

from fakes import API_TOKEN

AUTH = {"Authorization": f"Bearer {API_TOKEN}"}
CHUNK = 256 * 1024


class FakePipeline:
    """Writes what a render writes, or fails the way a render fails."""

    def __init__(self, mode="ok", hook=None, clock=None):
        self.mode, self.hook, self.clock = mode, hook, clock
        self.calls: list[str] = []

    def run(self, job_id, settings, timeout_s, should_stop, workdir):
        self.calls.append(job_id)
        if self.hook:
            self.hook(job_id)
        mode = self.mode(job_id) if callable(self.mode) else self.mode
        workdir.mkdir(parents=True, exist_ok=True)
        if mode == "ok":
            # as run_one does: a graph that already finished is not run
            # again, so a retry finds the very same files on disk
            video = workdir / "Test_horizontal.mp4"
            if not video.exists():
                video.write_bytes(os.urandom(8192))
            thumb = workdir / "thumbnail_agent_r00.jpg"
            if not thumb.exists():
                thumb.write_bytes(b"\xff\xd8jpeg" + os.urandom(64))
            (workdir / "run.log").write_text("rendered\n")
            (workdir / "output.json").write_text('{"ok": true}')
            (workdir / "parts").mkdir(exist_ok=True)
            (workdir / "parts" / "p01.mp4").write_bytes(b"part")
            return {"ok": True, "video_path": str(video),
                    "thumbnail_path": str(thumb)}
        if mode == "fail":
            return {"ok": False, "error": "probe found no readable clips"}
        if mode == "timeout":
            return {"ok": False, "timeout": True}
        if mode == "escape":
            return {"ok": True, "video_path": str(Path(sys.executable))}
        raise AssertionError(mode)


@pytest.fixture
def h(plane, worker_cp, drive, clock, tmp_path):
    """Seed jobs through the real upload routes; build workers."""
    client, ctx = plane

    class Harness:
        def seed(self, n_bytes=CHUNK + 99, key=None, name="clip.mp4"):
            data = os.urandom(n_bytes)
            r = client.post("/api/cloud/uploads", headers=AUTH, json={
                "idempotency_key": key or f"seed-{os.urandom(6).hex()}",
                "files": [{"name": name, "size": len(data),
                           "mime": "video/mp4"}],
                "settings": {"title": "Test"}}).json()
            jid = r["job"]["job_id"]
            off = 0
            while off < len(data):
                end = min(off + CHUNK, len(data)) - 1
                off = client.put(r["files"][0]["url"], headers={
                    **AUTH, "Content-Range": f"bytes {off}-{end}/"
                                            f"{len(data)}"},
                    content=data[off:end + 1]).json()["received"]
            client.post(f"/api/cloud/uploads/{jid}/complete", headers=AUTH)
            return jid

        def worker(self, pipeline=None, checks=None, **kw):
            return Worker(ctx.cfg, worker_cp, drive,
                          pipeline or FakePipeline(),
                          uploads_dir=tmp_path / "uploads",
                          outputs_dir=tmp_path / "outputs",
                          worker_id="test-host:boot0001", clock=clock,
                          sleep=clock.sleep,
                          checks=checks or (lambda: ([], {"gpu": "L4"})),
                          **kw)

        def state(self, jid):
            return ctx.queue.get(jid)["state"]

    hh = Harness()
    hh.client, hh.ctx = client, ctx
    hh.uploads, hh.outputs = tmp_path / "uploads", tmp_path / "outputs"
    return hh


# ------------------------------------------- 10. drain, then power off

def test_one_activation_drains_the_queue_then_powers_off(h, compute):
    jobs = [h.seed() for _ in range(3)]
    assert compute.starts() == 1, "three uploads, one start"
    w = h.worker()
    assert w.run() == EXIT_POWEROFF
    assert [o for _, o in w.outcomes] == ["completed"] * 3
    assert all(h.state(j) == Q.COMPLETED for j in jobs)
    assert h.ctx.queue.vm()["state"] == Q.VM_STOPPING


def test_it_powers_off_only_after_the_idle_grace(h, clock):
    h.seed()
    w = h.worker()
    t0 = clock()
    w.run()
    assert clock() - t0 >= h.ctx.cfg.idle_grace_s


def test_a_job_arriving_during_the_grace_is_taken(h, clock):
    h.seed()
    w = h.worker()
    late = []
    real_sleep = w.sleep

    def sleep(s):
        real_sleep(s)
        if not late:
            late.append(h.seed())                 # arrives while idling
    w.sleep = sleep
    assert w.run() == EXIT_POWEROFF
    assert h.state(late[0]) == Q.COMPLETED


# ---------------------------------------- 9. never off while work is held

def test_the_control_plane_says_wait_while_a_job_is_running(h):
    seen = []

    def peek(job_id):
        seen.append(h.ctx.vm.idle("some-other-worker"))
    h.seed()
    h.worker(FakePipeline(hook=peek)).run()
    assert seen == ["wait"]


# ------------------------------------ 8. a failed upload does not complete

def test_an_output_upload_that_fails_does_not_complete_the_job(h, drive):
    jid = h.seed()
    drive.fail_upload = 99
    w = h.worker()
    w.process(h.ctx.queue.claim("test-host:boot0001").__dict__ | {
        "output_root": h.ctx.cfg.drive_output_folder})
    job = h.ctx.queue.get(jid)
    assert job["state"] == Q.RETRYING, "transient: tried again later"
    assert job["outputs"] == []


def test_an_output_drive_cannot_verify_does_not_complete_the_job(h, drive):
    jid = h.seed()
    drive.corrupt_upload = True
    w = h.worker()
    w.run()
    job = h.ctx.queue.get(jid)
    assert job["state"] != Q.COMPLETED
    assert job["error_category"] == "verification"


def test_a_render_that_delivered_nothing_fails_without_retry(h):
    jid = h.seed()
    h.worker(FakePipeline("fail")).run()
    job = h.ctx.queue.get(jid)
    assert job["state"] == Q.FAILED and job["error_category"] == "pipeline"


# ----------------------------------------- 12. bounded, never prolonged

def test_a_render_past_its_deadline_gets_one_more_try_not_three(h, clock):
    """The retry backoff (60s) is shorter than the idle grace (180s), so
    the same activation takes the retry - and stops there."""
    jid = h.seed()
    pipe = FakePipeline("timeout")
    h.worker(pipe).run()
    job = h.ctx.queue.get(jid)
    assert job["state"] == Q.FAILED and job["error_category"] == "timeout"
    assert job["attempt"] == 2 < job["max_attempts"]
    assert pipe.calls == [jid, jid], "two renders, not three"


def test_a_rate_limited_job_does_not_hold_the_vm_up(h, clock):
    """The worker reports the rate limit and goes idle. It powers off well
    before the backoff ends instead of sitting on a billing VM."""
    jid = h.seed()
    w = h.worker()

    class RateLimited(FakePipeline):
        def run(self, *a, **k):
            from cloud.sanitize import CloudError
            raise CloudError("rate_limited", "429 from the provider")
    w.pipeline = RateLimited()
    t0 = clock()
    assert w.run() == EXIT_POWEROFF
    job = h.ctx.queue.get(jid)
    assert job["state"] == Q.RETRYING
    assert job["not_before"] > clock(), "powered off before the retry"
    assert clock() - t0 < h.ctx.cfg.rate_limit_backoff_s


def test_the_real_subprocess_is_killed_at_its_deadline(tmp_path):
    """Not a fake: a real child process that would sleep for a minute is
    killed - with everything it started - when its deadline passes."""
    pkg = tmp_path / "srv" / "cloud"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "run_one.py").write_text(textwrap.dedent("""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        time.sleep(60)
    """))
    pipe = SubprocessPipeline(server_dir=tmp_path / "srv",
                              sleep=lambda s: time.sleep(0.2))
    t0 = time.time()
    out = pipe.run("abcdef123456", {}, timeout_s=2.0,
                   should_stop=lambda: False, workdir=tmp_path / "w")
    assert out.get("timeout") is True
    assert time.time() - t0 < 30


def test_the_real_subprocess_result_is_read(tmp_path):
    pkg = tmp_path / "srv" / "cloud"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "run_one.py").write_text(textwrap.dedent("""
        import json, sys
        a = sys.argv
        out = a[a.index("--result") + 1]
        json.dump({"ok": True, "video_path": "v.mp4"}, open(out, "w"))
    """))
    out = SubprocessPipeline(server_dir=tmp_path / "srv").run(
        "abcdef123456", {"title": "x"}, 30, lambda: False, tmp_path / "w")
    assert out["ok"] is True and out["returncode"] == 0


# ------------------------------------------------------- cancel, fencing

def test_a_cancel_during_the_render_uploads_nothing(h, drive):
    def cancel(job_id):
        h.client.post(f"/api/cloud/jobs/{job_id}/cancel", headers=AUTH)
    jid = h.seed()
    h.worker(FakePipeline(hook=cancel)).run()
    assert h.state(jid) == Q.CANCELLED
    assert drive.uploads == []


def test_a_worker_whose_lease_was_taken_steps_aside_quietly(h, clock):
    """It was slow, not dead. By the time it reports, the job is someone
    else's - so it reports nothing, and cannot complete it."""
    def lose(job_id):
        clock.advance(h.ctx.cfg.lease_s + 1)
        h.ctx.reconciler.tick()
        clock.advance(10_000)
        assert h.ctx.queue.claim("another-worker") is not None
    jid = h.seed()
    w = h.worker(FakePipeline(hook=lose))
    claim = h.ctx.queue.claim("test-host:boot0001")
    assert w.process({**claim.__dict__,
                      "output_root": h.ctx.cfg.drive_output_folder}) \
        == "abandoned"
    job = h.ctx.queue.get(jid)
    assert job["lease_owner"] == "another-worker"
    assert job["state"] == Q.DOWNLOADING


# --------------------------------- 7. a retry redoes nothing already done

def test_a_retry_does_not_download_again_or_upload_twice(h, drive):
    jid = h.seed()
    w = h.worker()
    # first attempt: everything uploads, then the completion is lost
    real = w.cp.complete

    def lost(*a, **k):
        from cloud.sanitize import CloudError
        raise CloudError("transient", "connection reset")
    w.cp.complete = lost
    claim = h.ctx.queue.claim("test-host:boot0001")
    w.process({**claim.__dict__,
               "output_root": h.ctx.cfg.drive_output_folder})
    first_downloads, first_uploads = len(drive.downloads), \
        len(drive.uploads)
    assert h.state(jid) == Q.RETRYING
    # second attempt
    w.cp.complete = real
    h.ctx.queue.vm_update(state=Q.VM_READY)
    job = h.ctx.queue.get(jid)
    h.ctx.queue.clock = lambda: job["not_before"] + 1
    claim = h.ctx.queue.claim("test-host:boot0001")
    w.process({**claim.__dict__,
               "output_root": h.ctx.cfg.drive_output_folder})
    assert h.state(jid) == Q.COMPLETED
    assert len(drive.downloads) == first_downloads, "source not refetched"
    assert len(drive.uploads) == first_uploads, "outputs not re-uploaded"


def test_run_one_resumes_from_a_checkpoint(monkeypatch, tmp_path):
    """The decision that makes a retry cheap: no checkpoint -> run; a
    checkpoint with work left -> resume; a finished graph -> run nothing."""
    from agents import runner
    from cloud import run_one

    calls = []
    build = tmp_path / "outputs" / "abcdef123456"
    build.mkdir(parents=True)
    monkeypatch.setattr(runner, "job_paths",
                        lambda j, root=None: (tmp_path / "u", build))
    monkeypatch.setattr(runner, "make_job", lambda **k: object())
    monkeypatch.setattr(runner, "save_settings", lambda j: None)
    monkeypatch.setattr(runner, "run_job", lambda j: calls.append("run"))
    monkeypatch.setattr(runner, "resume_job",
                        lambda j, **k: calls.append("resume"))
    monkeypatch.setattr(run_one, "_deliverables",
                        lambda j, b: {"video_path": "v.mp4"})

    assert run_one.run("abcdef123456", {})["mode"] == "run"
    (build / runner.CHECKPOINT_DB).write_bytes(b"x")
    monkeypatch.setattr(runner, "peek_state",
                        lambda j: {"next": ["render"], "values": {"a": 1}})
    assert run_one.run("abcdef123456", {})["mode"] == "resume"
    monkeypatch.setattr(runner, "peek_state",
                        lambda j: {"next": [], "values": {"a": 1}})
    assert run_one.run("abcdef123456", {})["mode"] == "already_done"
    assert calls == ["run", "resume"]


# ----------------------------------------- 13. names cannot escape

@pytest.mark.parametrize("evil", [
    "../../../../etc/cron.d/x.mp4", "..\\..\\..\\Windows\\x.mp4",
    "C:\\Windows\\System32\\x.mp4", "/etc/passwd.mp4", "con.mp4",
    "....//....//x.mp4", "a\x00b.mp4"])
def test_a_hostile_source_name_stays_in_the_workspace(h, drive, evil):
    meta = drive.add(b"x" * 100, "src")
    w = h.worker()
    from cloud.worker import Heartbeat

    class NoCP:
        def heartbeat(self, *a, **k):
            return {}
    beat = Heartbeat(NoCP(), "abcdef123456", 1, 30, 300)
    paths = w.download({"job_id": "abcdef123456", "sources": [
        {"idx": 0, "name": evil, "size": 100, "mime": "video/mp4",
         "drive_file_id": meta["id"], "md5": meta["md5Checksum"]}]}, beat)
    ws = (h.uploads / "abcdef123456").resolve()
    assert paths[0].resolve().parent == ws
    assert paths[0].name.startswith("01_")


def test_a_name_that_is_only_dots_is_refused_not_written(h, drive):
    """`. .mp4` reduces to no extension at all. Refusing it is the safe
    answer; the upload route would already have refused it too."""
    from cloud.sanitize import CloudError
    from cloud.worker import Heartbeat

    class NoCP:
        def heartbeat(self, *a, **k):
            return {}
    meta = drive.add(b"x" * 100, "src")
    with pytest.raises(CloudError):
        h.worker().download({"job_id": "abcdef123456", "sources": [
            {"idx": 0, "name": ". .mp4", "size": 100, "mime": "video/mp4",
             "drive_file_id": meta["id"], "md5": meta["md5Checksum"]}]},
            Heartbeat(NoCP(), "abcdef123456", 1, 30, 300))
    assert not list((h.uploads / "abcdef123456").glob("*"))


def test_a_deliverable_outside_the_job_folder_is_never_uploaded(h, drive):
    jid = h.seed()
    h.worker(FakePipeline("escape")).run()
    assert h.state(jid) == Q.FAILED
    assert drive.uploads == []


def test_not_enough_disk_is_refused_before_downloading(h, drive):
    jid = h.seed(n_bytes=CHUNK + 1)
    w = h.worker()
    w.cfg = type(w.cfg)(**{**w.cfg.__dict__, "min_free_bytes": 10 ** 18})
    w.run()
    assert drive.downloads == []
    assert h.ctx.queue.get(jid)["error_category"] == "quota"


# ------------------------------------------------ host and plane problems

def test_a_host_that_cannot_work_says_so_and_exits(h, compute):
    h.seed()
    w = h.worker(checks=lambda: (["ffmpeg is not runnable"], {}))
    assert w.run() == EXIT_NOT_READY
    assert "ffmpeg" in h.ctx.queue.vm()["last_error"]
    h.ctx.reconciler.tick()
    assert compute.calls[-1] == "stop", "an unready VM is not left billing"


def test_a_vanished_control_plane_powers_the_vm_off(h, clock):
    """Better off than idle: the control plane starts it again when it
    has work."""
    w = h.worker()

    class Gone:
        def hello(self, *a):
            return {}

        def claim(self):
            from cloud.sanitize import CloudError
            raise CloudError("transient", "connection refused")
        idle = claim
    w.cp = Gone()
    t0 = clock()
    assert w.run() == EXIT_POWEROFF
    assert clock() - t0 <= 700


# --------------------------------------------------------------- cleanup

def test_after_completion_media_goes_and_records_stay(h):
    jid = h.seed()
    h.worker().run()
    build = h.outputs / jid
    assert not (h.uploads / jid).exists(), "source copies removed"
    assert not (build / "parts").exists()
    assert not list(build.glob("*.mp4"))
    assert (build / "run.log").exists() and (build / "output.json").exists()


def test_completion_carries_every_deliverable_verified(h):
    jid = h.seed()
    h.worker().run()
    outs = h.ctx.queue.get(jid)["outputs"]
    kinds = {o["kind"] for o in outs}
    assert {"video", "thumbnail", "report", "log"} <= kinds
    assert all(o["verified"] and o["drive_file_id"] for o in outs)


def test_the_youtube_sheet_is_written_from_the_agents_artifacts(tmp_path):
    """Title, YouTube's limits and next steps, read from the Title and
    Publishing artifacts - nothing invented here."""
    from agents.artifacts import (Artifact, PublishCheck, Publishing,
                                  Title)
    from agents.store import shared_store
    from cloud.run_one import _youtube_sheet

    build = tmp_path / "outputs" / "abcdef123456"
    build.mkdir(parents=True)
    store = shared_store(outputs=build.parent)
    store.put(Artifact[Title](
        job_id="abcdef123456", kind="Title", produced_by="title",
        inputs_hash="t", payload=Title(text="Motor test", source="operator", operator_title="Motor test",
                                       alternatives=["LEGO motor test"])))
    store.put(Artifact[Publishing](
        job_id="abcdef123456", kind="Publishing", produced_by="publishing",
        inputs_hash="p", payload=Publishing(
            status="needs_review", title="Motor test", title_from="operator",
            checks=[PublishCheck(check="the title fits YouTube's limit",
                                 passed=True, value="10 chars",
                                 limit="100"),
                    PublishCheck(check="the thumbnail is wide enough",
                                 passed=False, value="480 px",
                                 limit="640 px")],
            next_steps=[
                "Review the video in the app - nothing has left this machine",
                "Download it from /api/jobs/abcdef123456/download (v.mp4)",
                "Upload thumbnail_agent_r00.jpg as its custom thumbnail",
                "Upload it in YouTube Studio"])))
    sheet = _youtube_sheet("abcdef123456", build,
                           str(build / "thumbnail.jpg")).read_text(
                               encoding="utf-8")
    assert "Motor test" in sheet and "LEGO motor test" in sheet
    assert "[ok] the title fits YouTube's limit" in sheet
    assert "[FAIL] the thumbnail is wide enough" in sheet, "failures show"
    assert "Upload it in YouTube Studio" in sheet
    # steps only true of the local app are not repeated for a cloud job,
    # and the thumbnail is named as it lands in the browser's downloads
    assert "nothing has left this machine" not in sheet
    assert "/api/jobs/" not in sheet
    assert "Upload abcdef123456_thumbnail.jpg (it downloads" in sheet
