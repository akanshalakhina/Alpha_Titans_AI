"""
Publishing Agent.

Deterministic - there is no model to inject - and the checks are REAL: real
ffprobe on real files, a real full decode through the media pool. The
properties most of these defend:

  * nothing in the project can upload, the agent reads that from the source,
    and a parity test fails the moment an upload path appears
  * nothing it produces can say a video was published: no field for an
    upload id, a URL or a metric, and `published` is the literal False
  * the status follows from the checks and cannot contradict them
  * publishing stays the operator's act - approval required, private first
  * it never touches the network
"""

import shutil
import socket
import subprocess
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.artifacts import (Publishing, PublishCheck, PublishProvider,
                              RenderResult, RenderTechnical, Thumbnail,
                              ThumbnailCandidate, Title)
from agents.publishing import (CRITIC_PASSED, THUMB_FORMAT, THUMB_PRESENT,
                               THUMB_SIZE, THUMB_WIDTH, TITLE_CHARS,
                               TITLE_LENGTH, TITLE_PRESENT, VIDEO_DECODES,
                               VIDEO_ON_DISK, VIDEO_READABLE, PLACEHOLDER,
                               PublishingAgent, _code, _platform, _scan)
from agents.tools.pipeline_bridge import config_mod

FFMPEG = config_mod.FFMPEG


# ------------------------------------------------------------- helpers

def make_image(path: Path, size="1280x720", fmt=None) -> Path:
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi",
           "-i", f"testsrc2=s={size}:d=1", "-frames:v", "1"]
    if fmt:
        cmd += ["-c:v", fmt]
    subprocess.run(cmd + [str(path)], check=True, capture_output=True)
    return path


@pytest.fixture
def pub_ctx(director_ctx, tmp_path):
    """A real pool and store with a PRIVATE build folder, and the operator
    title left at the placeholder."""
    ctx, _cs = director_ctx
    build = tmp_path / "build"
    build.mkdir()
    ctx.job = ctx.job.model_copy(update={
        "build_dir": str(build),
        "settings": ctx.job.settings.model_copy(
            update={"title": PLACEHOLDER})})
    return ctx


@pytest.fixture
def build(pub_ctx):
    return Path(pub_ctx.job.build_dir)


@pytest.fixture
def video(lively_clip, build):
    """The delivered file: a real 6s 320x240 h264 mp4, under the canonical
    name master_stage gives it, plus the per-attempt archive copy."""
    canonical = build / "ALPHA_TITANS_horizontal.mp4"
    shutil.copy2(lively_clip, canonical)
    archive = build / "attempts" / "attempt_01.mp4"
    archive.parent.mkdir()
    shutil.copy2(lively_clip, archive)
    return canonical, archive


def a_render(video, master_thumb=None):
    canonical, archive = video
    return RenderResult(attempt=1, video_path=str(archive),
                        canonical_path=str(canonical),
                        thumbnail_path=str(master_thumb) if master_thumb
                        else None,
                        technical=RenderTechnical(duration=6.0, width=320,
                                                  height=240))


def a_thumbnail(image, video_path):
    cand = ThumbnailCandidate(index=0, clip_id="02_lively.mp4",
                              source="plan_segment", segment_index=1,
                              clip_at=2.0, frame_at=2.0)
    return Thumbnail(candidates=[cand], chosen=0, video_path=str(video_path),
                     generated=True, image_path=str(image), width=1280,
                     height=720, size=Path(image).stat().st_size)


def a_title(text="Building LEGO Technic Mechanisms"):
    return Title(text=text, source="model", published=True)


def named(ctx, title):
    ctx.job = ctx.job.model_copy(update={
        "settings": ctx.job.settings.model_copy(update={"title": title})})
    return ctx


def run(ctx, tracer_span, render, thumbnail="none", title="default",
        verdict="PASS", revision=0):
    with tracer_span(ctx, "publishing") as span:
        return PublishingAgent().execute(
            ctx, span, render=render,
            thumbnail=None if thumbnail == "none" else thumbnail,
            title=a_title() if title == "default" else title,
            verdict=verdict, revision=revision)


@pytest.fixture
def sound(pub_ctx, build, video):
    """Everything a ready package needs: the agent's 1280x720 thumbnail of
    this very render, master_stage's own, and a published title."""
    agent_img = make_image(build / "thumbnail_agent_r00.jpg")
    master_img = make_image(build / "thumbnail.jpg", "320x240")
    render = a_render(video, master_img)
    return render, a_thumbnail(agent_img, render.video_path)


def check(p: Publishing, name: str) -> PublishCheck:
    return next(c for c in p.checks if c.check == name)


# -------------------- 1. what the project can actually do

def test_nothing_in_the_project_can_upload():
    assert _platform()["upload"] == [], \
        "an upload path exists - the provider literal must be revisited"
    assert typing.get_args(PublishProvider) == ("none",)
    assert typing.get_args(Publishing.model_fields["published"].annotation) \
        == (False,)


def test_the_only_youtube_code_is_read_only_search():
    assert _platform()["search"] == ["server/agents/tools/search_tools.py"]
    root = Path(__file__).resolve().parents[2]
    code = _code((root / "server/agents/tools/search_tools.py").read_text())
    assert "youtube/v3/search" in code
    assert "youtube/v3/videos" not in code and "oauth" not in code.lower()


def test_the_scan_reads_code_not_prose():
    prose = ('"""an upload would need videos().insert and youtube/v3/videos'
             '"""\nx = 1  # googleapiclient, InstalledAppFlow\n')
    code = ("from googleapiclient.discovery import build\n"
            "yt.videos().insert(part='snippet', body={})\n")
    from agents.publishing import _UPLOAD_MARKERS
    assert not [m for m in _UPLOAD_MARKERS if m in _code(prose)]
    assert {"videos().insert", "googleapiclient"} <= \
        {m for m in _UPLOAD_MARKERS if m in _code(code)}


def test_a_drive_upload_is_not_mistaken_for_a_youtube_one(tmp_path):
    """The cloud worker stores clips and renders in Google Drive through
    `googleapis.com/upload/drive/v3/files`. That is not a way to publish
    to a channel, which is the only thing this guard is about."""
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "storage.py").write_text(
        "URL = 'https://www.googleapis.com/upload/drive/v3/files'\n")
    found = _scan(str(tmp_path), (("server/storage.py", 1),))
    assert found[1] == ()


def test_narrowing_the_marker_still_catches_youtube(tmp_path):
    """Every spelling of a YouTube upload path the guard was written for
    is still found after the Drive endpoint was let through."""
    (tmp_path / "server").mkdir()
    for i, line in enumerate((
            "U = 'https://www.googleapis.com/upload/youtube/v3/videos'",
            "r = youtube.videos().insert(part='snippet', body=b)",
            "from google_auth_oauthlib.flow import InstalledAppFlow")):
        (tmp_path / "server" / f"u{i}.py").write_text(line + "\n")
    found = _scan(str(tmp_path), tuple((f"server/u{i}.py", 1)
                                       for i in range(3)))
    assert len(found[1]) == 3


def test_the_scan_would_catch_a_real_upload_path(tmp_path):
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "uploader.py").write_text(
        "URL = 'https://www.googleapis.com/upload/youtube/v3/videos'\n")
    found = _scan(str(tmp_path), (("server/uploader.py", 1),))
    assert found[1] == ("server/uploader.py",)


# ------------------------------------- 2. a sound package, and no claim

def test_a_sound_package_is_ready_for_the_operator(pub_ctx, tracer_span,
                                                   sound):
    render, thumb = sound
    art = run(pub_ctx, tracer_span, render, thumb)
    p = art.payload
    assert p.status == "ready", [c for c in p.checks if not c.passed]
    assert p.published is False and p.provider == "none"
    assert p.requires_operator_approval is True
    assert p.recommended_visibility == "private"
    assert p.video.file_name == "ALPHA_TITANS_horizontal.mp4"
    assert (p.video.codec, p.video.width, p.video.height) == \
        ("h264", 320, 240)
    assert p.thumbnail.file_name == "thumbnail_agent_r00.jpg"
    assert (p.thumbnail_from, p.title_from) == ("thumbnail_agent",
                                                "title_agent")
    assert p.title == "Building LEGO Technic Mechanisms"
    assert all(c.passed for c in p.checks)
    steps = " ".join(p.next_steps)
    assert f"/api/jobs/{pub_ctx.job.job_id}/download" in steps
    assert "as Private" in steps and "nothing has been uploaded" in steps
    assert not art.degraded and art.confidence == 1.0


@pytest.mark.parametrize("kw", [
    {"published": True}, {"provider": "youtube"},
    {"requires_operator_approval": False},
    {"recommended_visibility": "public"},
    {"upload_id": "abc123"}, {"video_id": "dQw4w9WgXcQ"},
    {"url": "https://youtu.be/x"}, {"views": 100},
    {"published_at": "2026-10-01"}, {"scheduled_for": "tomorrow"},
])
def test_the_schema_cannot_claim_a_publication(kw):
    with pytest.raises(ValidationError):
        Publishing(**kw)


def test_the_status_cannot_contradict_the_checks():
    ok = PublishCheck(check="a", passed=True)
    soft = PublishCheck(check="b", passed=False)
    hard = PublishCheck(check="c", passed=False, blocking=True)
    for status, checks in (("ready", [ok, soft]), ("needs_review", [ok]),
                           ("needs_review", [hard]), ("ready", []),
                           ("blocked", [ok])):
        with pytest.raises(ValidationError):
            Publishing(status=status, checks=checks)


def test_ready_needs_a_video_a_thumbnail_and_a_title():
    with pytest.raises(ValidationError):
        Publishing(status="ready",
                   checks=[PublishCheck(check="a", passed=True)])


def test_a_source_exists_exactly_when_its_deliverable_does():
    with pytest.raises(ValidationError):
        Publishing(title="x", checks=[])
    with pytest.raises(ValidationError):
        Publishing(title_from="operator", checks=[])


def test_the_agent_is_deterministic_by_design():
    assert PublishingAgent.uses_llm is False
    assert PublishingAgent.agency == "deterministic"


# ------------------------------ 3. what needs a person, and what blocks

def test_an_escalated_video_needs_review(pub_ctx, tracer_span, sound):
    render, thumb = sound
    p = run(pub_ctx, tracer_span, render, thumb, verdict="ESCALATE").payload
    assert p.status == "needs_review"
    assert not check(p, CRITIC_PASSED).passed
    assert any(s.startswith("Resolve first: the critic passed it")
               for s in p.next_steps)


def test_no_title_needs_review_and_says_the_placeholder_is_not_one(
        pub_ctx, tracer_span, sound):
    render, thumb = sound
    p = run(pub_ctx, tracer_span, render, thumb, title=None).payload
    assert p.status == "needs_review" and p.title == ""
    c = check(p, TITLE_PRESENT)
    assert not c.passed and PLACEHOLDER in c.value
    assert "Name the video first - it has no title yet" in p.next_steps


def test_an_unpublished_agent_title_is_not_used(pub_ctx, tracer_span,
                                                sound):
    render, thumb = sound
    p = run(pub_ctx, tracer_span, render, thumb,
            title=Title(text="Draft", source="model",
                        published=False)).payload
    assert p.title == "" and not check(p, TITLE_PRESENT).passed


def test_an_operator_title_wins_and_is_the_humans_decision(pub_ctx,
                                                           tracer_span,
                                                           sound):
    render, thumb = sound
    art = run(named(pub_ctx, "Sunday Gearbox"), tracer_span, render, thumb)
    assert (art.payload.title, art.payload.title_from) == ("Sunday Gearbox",
                                                           "operator")
    d = next(d for d in art.decisions if d.what == "Title in the plan")
    assert d.decided_by == "human"


@pytest.mark.parametrize("title,failing", [
    ("Gears <b>bold</b>", TITLE_CHARS),
    ("x" * 120, TITLE_LENGTH),
])
def test_youtubes_title_limits_are_checked_locally(pub_ctx, tracer_span,
                                                   sound, title, failing):
    render, thumb = sound
    p = run(named(pub_ctx, title), tracer_span, render, thumb).payload
    c = check(p, failing)
    assert not c.passed and c.source == "youtube_limit"
    assert p.status == "needs_review"


def test_a_narrow_thumbnail_is_flagged(pub_ctx, tracer_span, build, video):
    img = make_image(build / "thumbnail_agent_r00.jpg", "320x240")
    render = a_render(video)
    p = run(pub_ctx, tracer_span, render,
            a_thumbnail(img, render.video_path)).payload
    assert not check(p, THUMB_WIDTH).passed and p.status == "needs_review"


def test_an_oversized_thumbnail_is_flagged(pub_ctx, tracer_span, build,
                                           video):
    img = make_image(build / "thumbnail_agent_r00.jpg")
    with open(img, "ab") as f:                       # > 2 MB, still a JPEG
        f.write(b"\0" * (2 * 1024 * 1024 + 1))
    render = a_render(video)
    p = run(pub_ctx, tracer_span, render,
            a_thumbnail(img, render.video_path)).payload
    c = check(p, THUMB_SIZE)
    assert not c.passed and c.source == "youtube_limit"


def test_a_thumbnail_in_a_format_youtube_refuses_is_flagged(
        pub_ctx, tracer_span, build, video):
    img = make_image(build / "thumbnail.bmp", fmt="bmp")
    render = a_render(video, img)
    p = run(pub_ctx, tracer_span, render).payload
    assert p.thumbnail_from == "master_stage"
    assert not check(p, THUMB_FORMAT).passed


def test_the_agent_thumbnail_is_used_only_for_this_render(pub_ctx,
                                                          tracer_span,
                                                          build, video):
    agent_img = make_image(build / "thumbnail_agent_r00.jpg")
    master_img = make_image(build / "thumbnail.jpg")
    render = a_render(video, master_img)
    p = run(pub_ctx, tracer_span, render,
            a_thumbnail(agent_img, "some/other/attempt.mp4")).payload
    assert (p.thumbnail.file_name, p.thumbnail_from) == ("thumbnail.jpg",
                                                         "master_stage")
    assert any("master_stage's frame is in the plan" in x
               for x in p.limitations)


def test_no_thumbnail_at_all_needs_review(pub_ctx, tracer_span, video):
    p = run(pub_ctx, tracer_span, a_render(video)).payload
    assert p.thumbnail is None and p.thumbnail_from is None
    assert not check(p, THUMB_PRESENT).passed
    assert p.status == "needs_review"


def test_a_missing_video_blocks_everything(pub_ctx, tracer_span, video,
                                           sound):
    render, thumb = sound
    for f in video:
        f.unlink()
    p = run(pub_ctx, tracer_span, render, thumb).payload
    assert p.status == "blocked" and p.video is None
    c = check(p, VIDEO_ON_DISK)
    assert not c.passed and c.blocking
    assert p.next_steps[0].startswith("Nothing can be published")
    assert not any("/download" in s for s in p.next_steps)


def test_an_unreadable_video_is_blocked(pub_ctx, tracer_span, video, sound):
    render, thumb = sound
    for f in video:
        f.write_bytes(b"not a video at all " * 100)
    p = run(pub_ctx, tracer_span, render, thumb).payload
    assert p.status == "blocked"
    assert not check(p, VIDEO_READABLE).passed


def test_a_video_that_fails_to_decode_is_blocked(pub_ctx, tracer_span,
                                                 sound, monkeypatch):
    render, thumb = sound
    monkeypatch.setattr(pub_ctx.pool, "run_process",
                        lambda cmd, **kw: (0, "[h264] error while decoding"))
    p = run(pub_ctx, tracer_span, render, thumb).payload
    assert p.status == "blocked"
    c = check(p, VIDEO_DECODES)
    assert not c.passed and "error while decoding" in c.value


def test_the_file_the_app_downloads_is_the_one_checked(pub_ctx, tracer_span,
                                                       video, sound):
    render, thumb = sound
    p = run(pub_ctx, tracer_span, render, thumb).payload
    assert p.video.path == render.canonical_path
    video[0].unlink()                                # canonical gone
    p = run(pub_ctx, tracer_span, render, thumb, revision=1).payload
    assert p.video.path == render.video_path         # the archive instead


# ------------------------------------------ 4. processes and network

def test_the_decode_is_owned_by_the_pool(pub_ctx, tracer_span, sound,
                                         monkeypatch):
    render, thumb = sound
    seen = {}
    real = pub_ctx.pool.run_process

    def spy(cmd, **kw):
        seen.update(kw, cmd=cmd)
        return real(cmd, **kw)
    monkeypatch.setattr(pub_ctx.pool, "run_process", spy)
    run(pub_ctx, tracer_span, render, thumb)
    assert seen["label"] == "publishing:decode"
    assert seen["job_id"] == pub_ctx.job.job_id and seen["timeout_s"]
    assert seen["cmd"][-3:] == ["-f", "null", "-"]
    assert render.canonical_path in seen["cmd"]
    assert pub_ctx.pool.active_processes(pub_ctx.job.job_id) == []


def test_it_never_touches_the_network(pub_ctx, tracer_span, sound,
                                      monkeypatch):
    render, thumb = sound
    attempts = []

    def refuse(self, *a, **kw):
        attempts.append(a)
        raise AssertionError("the publishing agent opened a connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    p = run(pub_ctx, tracer_span, render, thumb).payload
    assert attempts == [] and p.status == "ready"


# ---------------------------------------------------------- 5. caching

def test_identical_inputs_reuse_the_plan_without_decoding_again(
        pub_ctx, tracer_span, sound, monkeypatch):
    render, thumb = sound
    a = run(pub_ctx, tracer_span, render, thumb)
    calls = []
    monkeypatch.setattr(pub_ctx.pool, "run_process",
                        lambda cmd, **kw: calls.append(cmd) or (1, "x"))
    b = run(pub_ctx, tracer_span, render, thumb, revision=1)
    assert a.artifact_id == b.artifact_id and calls == []


def test_changed_bytes_on_disk_are_checked_again(pub_ctx, tracer_span,
                                                 video, sound):
    render, thumb = sound
    a = run(pub_ctx, tracer_span, render, thumb)
    with open(video[0], "ab") as f:
        f.write(b"\0")
    b = run(pub_ctx, tracer_span, render, thumb, revision=1)
    assert a.inputs_hash != b.inputs_hash


@pytest.mark.parametrize("change", ["title", "verdict", "thumbnail"])
def test_the_plan_is_remade_when_an_input_changes(pub_ctx, tracer_span,
                                                  sound, change):
    render, thumb = sound
    a = run(pub_ctx, tracer_span, render, thumb)
    kw = {"title": {"title": a_title("A Different Title")},
          "verdict": {"verdict": "ESCALATE"},
          "thumbnail": {"thumbnail": "none"}}[change]
    b = run(pub_ctx, tracer_span, render, **{"thumbnail": thumb, **kw},
            revision=1)
    assert a.inputs_hash != b.inputs_hash


# --------------------------------------------- 6. trace and decisions

def test_the_trace_records_the_scan_and_the_decode(pub_ctx, tracer_span,
                                                   sound):
    render, thumb = sound
    with tracer_span(pub_ctx, "publishing") as span:
        art = PublishingAgent().execute(pub_ctx, span, render=render,
                                        thumbnail=thumb, title=a_title(),
                                        verdict="PASS")
        assert art.trace_id == span.trace_id
    rows = [r for r in pub_ctx.tracer.for_job(pub_ctx.job.job_id)
            if r["node"] == "publishing"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert "publishing.scan_platform" in tools
    assert "publishing.decode_check" in tools
    assert "llm.complete_json" not in tools
    assert art.artifact_id in rows[-1]["output_artifacts"]
    assert rows[-1]["tokens_in"] == 0


def test_decisions_are_measured_or_policy_never_a_model(pub_ctx,
                                                        tracer_span, sound):
    from agents.artifacts import DecidedBy
    render, thumb = sound
    art = run(pub_ctx, tracer_span, render, thumb)
    assert {d.decided_by for d in art.decisions} <= set(
        typing.get_args(DecidedBy)) - {"llm", "fallback"}
    by = {d.what: d for d in art.decisions}
    assert by["Publishing provider"].chose == \
        "none - nothing in this project can upload"
    assert by["Publishing provider"].decided_by == "measurement"
    assert by["Operator approval"].decided_by == "policy"
    assert "private" in by["Operator approval"].chose
    assert by["Readiness"].chose.startswith("ready:")
    assert by["Delivered video"].decided_by == "measurement"


def test_the_platform_capabilities_are_stated(pub_ctx, tracer_span, sound):
    render, thumb = sound
    p = run(pub_ctx, tracer_span, render, thumb).payload
    caps = " ".join(p.platform_capabilities)
    assert "read-only" in caps and "search_tools.py" in caps
    assert "no YouTube upload, OAuth or channel access" in caps


# ------------------------------------------ 7. reported through the app

def test_the_projection_says_nothing_was_published(pub_ctx, tracer_span,
                                                   sound):
    from agents.compat import _project_publishing
    render, thumb = sound
    p = run(pub_ctx, tracer_span, render, thumb, verdict="ESCALATE").payload
    out = _project_publishing(p)
    assert out["published"] is False and out["provider"] == "none"
    assert out["status"] == "needs_review"
    assert out["warnings"] == [CRITIC_PASSED] and out["blockers"] == []
    assert out["thumbnail"] == "thumbnail_agent_r00.jpg"
    assert out["next_steps"] == p.next_steps
