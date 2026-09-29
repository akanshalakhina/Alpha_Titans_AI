"""End-to-end: every user-facing feature, in a real browser, on the real server."""
import json
import shutil
import time
import urllib.request
from pathlib import Path

import pytest
from conftest import FIX, playable, shot, toasts, upload

ALL3 = ("Gear box build.mp4", "robot_arm.mp4", "vertical tower.mp4")


def api(server, path):
    return json.load(urllib.request.urlopen(server.url + path, timeout=10))


def job_id(pg):
    # the id the page is working on, read from a clip's poster URL
    src = pg.get_attribute("#clipList .clip video", "poster")
    return src.split("/api/jobs/")[1].split("/")[0]


def server_names(server, jid):
    return [c["name"] for c in api(server, f"/api/jobs/{jid}/clips")["clips"]]


def to_step2(pg):
    pg.click("#toStep2"); pg.wait_for_selector("#step-2.on")


def run_build(pg, timeout=240):
    pg.click("#startBuild")
    pg.wait_for_selector("#step-3.on", timeout=30000)
    pg.wait_for_selector("#step-4.on", timeout=timeout * 1000)


# ───────────────────────────── boot / shell ─────────────────────────────

def test_boot_splash_then_media_step(page):
    assert page.is_visible("#step-1.on")
    assert page.is_hidden("#loader")
    assert "Studio online" in page.inner_text("#health")
    assert page.get_attribute("#stepper li[data-step='1']", "class").strip() == "on"
    assert page.is_disabled("#toStep2")
    assert page.errors == []


def test_splash_shows_elapsed_timer_on_slow_api(browser, server):
    ctx = browser.new_context()
    pg = ctx.new_page()
    held = []
    pg.route("**/api/health", lambda route: held.append(route))   # hold the answer
    pg.goto(server.url + "/", wait_until="domcontentloaded")
    pg.wait_for_timeout(1200)
    assert pg.is_visible("#loader") and "Starting the studio" in pg.inner_text("#lrTitle")
    assert "spin" in pg.get_attribute("#lrRing", "class")
    assert float(pg.inner_text("#lrElapsed").rstrip("s")) >= 0.8     # the timer is counting
    held[0].continue_()
    pg.wait_for_selector("#loader", state="hidden", timeout=10000)
    ctx.close()


def test_server_down_shows_retry(browser, server):
    ctx = browser.new_context()
    pg = ctx.new_page()
    pg.route("**/api/health", lambda r: r.abort())
    pg.goto(server.url + "/")
    pg.wait_for_selector("#lrRetry", state="visible", timeout=5000)
    assert "Can't reach" in pg.inner_text("#lrTitle")
    ctx.close()


# ───────────────────────────── step 1: media ─────────────────────────────

def test_upload_shows_blurred_modal_with_progress(page):
    page.set_input_files("#clipInput", [str(FIX / n) for n in ALL3])
    page.wait_for_selector("#loader:not([hidden])", timeout=5000)
    bf = page.eval_on_selector("#loader", "e => getComputedStyle(e).backdropFilter || getComputedStyle(e).webkitBackdropFilter")
    assert "blur" in bf
    title = page.inner_text("#lrTitle")
    assert "Uploading 3 clips" in title or "Preparing previews" in title
    shot(page, "e2e_upload_modal")
    page.wait_for_selector("#loader", state="hidden", timeout=60000)
    assert page.locator("#clipList .clip").count() == 3
    assert "3 clips ready" in page.inner_text("#clipTally")
    assert page.is_enabled("#toStep2")
    assert any("Added 3 clips" in t for t in toasts(page))
    assert page.errors == []


def test_clip_cards_hide_technical_metadata_until_asked(page):
    upload(page, *ALL3, wait_count=3)
    card = page.locator("#clipList .clip").nth(0)
    assert card.locator(".nm").inner_text() == "Gear box build"
    assert not card.locator(".details").is_visible()
    body = page.inner_text("#clipList")
    for tech in ("h264", "1280×720", "MB"):
        assert tech not in body
    card.locator(".infobtn").click()
    assert card.locator(".details").is_visible()
    d = card.locator(".details").inner_text()
    assert "h264" in d and "1280×720" in d and "Landscape" in d
    assert page.locator("#clipList .clip").nth(2).locator(".details").is_hidden()
    card.locator(".infobtn").click()
    assert not card.locator(".details").is_visible()


def test_play_button_plays_and_pauses_clip(page):
    """Bug 4: the play icon vanished on hover and clicking did nothing."""
    playable(page)
    upload(page, *ALL3, wait_count=3)
    card = page.locator("#clipList .clip").nth(1)
    card.locator(".strip").hover()
    page.wait_for_timeout(500)
    assert card.locator(".playhint").evaluate("e => getComputedStyle(e).opacity") != "0"
    assert card.locator("video").evaluate("v => v.paused"), "hover must not auto-play"
    card.locator(".strip").click()
    page.wait_for_function("() => { const v = document.querySelectorAll('#clipList video')[1]; return v && !v.paused && v.currentTime > 0.2; }", timeout=10000)
    assert "playing" in card.get_attribute("class")
    card.locator(".strip").click()
    page.wait_for_timeout(200)
    assert card.locator("video").evaluate("v => v.paused")
    # only one preview plays at a time
    page.locator("#clipList .clip").nth(0).locator(".strip").click()
    page.wait_for_timeout(600)
    card.locator(".strip").click()
    page.wait_for_timeout(600)
    playing = page.eval_on_selector_all("#clipList video", "vs => vs.filter(v => !v.paused).length")
    assert playing == 1


def test_unplayable_clip_says_so(page):
    """A codec the browser cannot decode (here: H.264 in open-source Chromium,
    for users e.g. HEVC iPhone clips) must explain itself, not fail silently."""
    upload(page, "robot_arm.mp4", wait_count=1)
    page.locator("#clipList .clip .strip").click()
    page.wait_for_timeout(1500)
    assert any("cannot be previewed" in t for t in toasts(page))
    assert "playing" not in page.get_attribute("#clipList .clip", "class")


def test_drag_reorder_updates_server_order(page, server):
    """Bug 5: reorder failed with a 500."""
    playable(page)
    upload(page, *ALL3, wait_count=3)
    jid = job_id(page)
    # start a preview first: the original failure happened with a clip open
    page.locator("#clipList .clip").nth(2).locator(".strip").click()
    page.wait_for_timeout(800)
    page.drag_and_drop("#clipList .clip:nth-child(3)", "#clipList .clip:nth-child(1)")
    page.wait_for_function("document.querySelector('#clipList .clip .nm').textContent === 'vertical tower'", timeout=10000)
    assert server_names(server, jid) == ["01_vertical tower.mp4", "02_Gear box build.mp4", "03_robot_arm.mp4"]
    assert [page.locator("#clipList .clip .ord").nth(i).inner_text() for i in range(3)] == ["1", "2", "3"]
    assert not any("bad" in t for t in toasts(page))
    assert page.errors == []


def test_remove_clip(page, server):
    upload(page, *ALL3, wait_count=3)
    jid = job_id(page)
    card = page.locator("#clipList .clip").nth(1)
    card.hover()
    card.locator(".rm").click()
    page.wait_for_function("document.querySelectorAll('#clipList .clip').length === 2")
    assert server_names(server, jid) == ["01_Gear box build.mp4", "03_vertical tower.mp4"]


def test_non_video_is_rejected_with_message(page):
    page.set_input_files("#clipInput", [str(FIX / "bad.txt")])
    page.wait_for_selector("#loader", state="hidden", timeout=20000)
    page.wait_for_timeout(300)
    assert any("not a video" in t for t in toasts(page))
    assert page.locator("#clipList .clip").count() == 0


# ───────────────────────────── step 2: customize ─────────────────────────────

def test_customize_controls_update_summary(page):
    upload(page, *ALL3, wait_count=3)
    to_step2(page)
    summary = lambda: page.inner_text("#customSummary")
    assert "3 clips" in summary() and "1080p" in summary() and "AI picks" in summary()
    page.click("#orientation button[data-v=vertical]")
    page.click("#resolution button[data-v='720p']")
    page.click("#speed button[data-v=fast]")
    page.click("#transition button[data-v=cut]")
    page.click("#mood button[data-v=calm]")
    s = summary()
    assert "Vertical" in s and "720p" in s and "Punchy" in s and "Clean" in s and "Calm" in s
    page.click("#speed button[data-v=custom]")
    assert page.is_visible("#speedCustom")
    page.fill("#speedRange", "1.5")
    page.dispatch_event("#speedRange", "input")
    assert "1.50×" in summary()
    page.click("#musicSource button[data-v=upload]")
    assert page.is_visible("#libraryWrap") and page.is_hidden("#moodWrap")
    page.click("#musicSource button[data-v=none]")
    assert "None" in summary()


def test_quality_tiles_explain_use_cases(page):
    """Question 3: what is Export > Advanced > Quality for?"""
    upload(page, *ALL3, wait_count=3)
    to_step2(page)
    page.click(".advanced summary")
    tiles = page.inner_text("#quality")
    for word in ("Draft", "Standard", "High", "Master", "WhatsApp", "YouTube", "archive"):
        assert word in tiles
    assert "does not change the resolution" in page.inner_text(".advanced-body")
    page.click("#quality button[data-v=draft]")
    assert "quick look" in page.inner_text("#qualityHint").lower() or "fastest" in page.inner_text("#qualityHint").lower()
    assert "CRF" not in page.inner_text("#step-2")


def test_audio_notice_when_clips_have_sound(page):
    upload(page, *ALL3, wait_count=3)
    to_step2(page)
    assert page.is_visible("#audioDetected")
    assert "1 of 3 clips" in page.inner_text("#audioDetected")
    page.click("#keepAudio button[data-v=mix]")
    assert page.is_hidden("#audioDetected")


def test_voice_note_record_play_and_rerecord(page, server):
    """Bug 6: recording was unreliable and the play button hid the label."""
    upload(page, *ALL3, wait_count=3)
    to_step2(page)
    page.click("#recordBtn")
    page.wait_for_function("document.querySelector('#recordBtn').classList.contains('recording')", timeout=5000)
    page.wait_for_timeout(2000)
    page.click("#recordBtn")
    page.wait_for_selector("#voiceBox .vplayer", timeout=15000)
    assert any("Voice note saved" in t for t in toasts(page))
    first_src = page.get_attribute("#voiceBox audio", "src")
    label = page.locator("#voiceBox .vhead strong")
    page.hover("#voiceBox .vplay")
    assert label.is_visible() and label.inner_text() == "Voice note"
    page.click("#voiceBox .vplay")
    # playing and the button showing "pause" must be true at the same moment
    page.wait_for_function("""() => { const a = document.querySelector('#voiceBox audio');
        return a && !a.paused && document.querySelector('#voiceBox .vplay').classList.contains('on'); }""", timeout=5000)
    assert label.is_visible()
    page.wait_for_function("document.querySelector('#voiceBox audio').currentTime > 0.2", timeout=5000)
    assert page.inner_text("#voiceBox .vtime").startswith("0:0")
    page.click("#voiceBox .vplay")
    # second recording must replace the first, not replay a cached copy
    page.wait_for_timeout(1100)
    page.click("#recordBtn"); page.wait_for_timeout(1600); page.click("#recordBtn")
    page.wait_for_function(f"document.querySelector('#voiceBox audio').getAttribute('src') !== {json.dumps(first_src)}", timeout=15000)
    assert page.is_enabled("#recordBtn")
    assert page.errors == []


def test_voice_note_too_short_is_refused_kindly(page):
    upload(page, "robot_arm.mp4", wait_count=1)
    to_step2(page)
    page.click("#recordBtn")
    page.wait_for_function("document.querySelector('#recordBtn').classList.contains('recording')")
    page.evaluate("document.querySelector('#recordBtn').click()")   # stop at once
    page.wait_for_function("[...document.querySelectorAll('.toast')].some(t => /too short/.test(t.textContent))", timeout=5000)
    assert page.is_enabled("#recordBtn")


def test_voice_double_click_does_not_break_recorder(page):
    upload(page, "robot_arm.mp4", wait_count=1)
    to_step2(page)
    page.dblclick("#recordBtn")
    page.wait_for_timeout(1500)
    state = page.eval_on_selector("#recordBtn", "b => b.classList.contains('recording')")
    if state:
        page.click("#recordBtn")
    page.wait_for_timeout(2000)
    assert page.is_enabled("#recordBtn")
    assert page.errors == []


# ───────────────────────────── steps 3 + 4: build ─────────────────────────────

def test_full_flow_build_progress_and_final_video(page, server):
    """Bug 1 (ring stuck), bug 7 (log paths) and the whole happy path."""
    playable(page)
    upload(page, *ALL3, wait_count=3)
    to_step2(page)
    page.fill("#script", "Frame goes together\nMotor spins up")
    page.click("#resolution button[data-v='720p']")
    page.click(".advanced summary")
    page.click("#quality button[data-v=draft]")
    page.fill("#title", "E2E Gearbox")
    page.click("#startBuild")
    page.wait_for_selector("#step-3.on", timeout=30000)
    assert "live" in page.get_attribute("#ring", "class")
    samples = []
    for _ in range(12):
        if page.is_visible("#step-4.on"):
            break
        samples.append(float(page.inner_text("#ringPct").rstrip("%")))
        page.wait_for_timeout(1000)
    shot(page, "e2e_processing")
    assert samples == sorted(samples), f"ring went backwards: {samples}"
    assert len(samples) < 3 or samples[-1] > samples[0], f"ring did not move: {samples}"
    if page.is_visible("#step-3.on"):
        assert "Elapsed" in page.inner_text("#procTime") and "left" in page.inner_text("#procTime")
    page.wait_for_selector("#step-4.on", timeout=240000)
    # final video
    assert page.inner_text("#outName") == "E2E Gearbox"
    assert "All quality checks passed" in page.inner_text("#verifySummary")
    page.wait_for_function("document.querySelector('#player').readyState >= 1", timeout=20000)
    assert page.eval_on_selector("#player", "v => v.duration") > 5
    page.eval_on_selector("#player", "v => v.play()")
    page.wait_for_function("document.querySelector('#player').currentTime > 0.3", timeout=10000)
    assert page.locator("#timeline .tl-seg").count() >= 1
    assert "Made in" in page.inner_text("#outChips")
    assert page.is_hidden("#verify")                     # technical details folded away
    page.click("#step-4 .fold summary")
    assert page.locator("#verify .vrow.ok").count() == 4
    shot(page, "e2e_final", full=True)
    with page.expect_download() as dl:
        page.click("#downloadBtn")
    path = dl.value.path()
    assert Path(path).stat().st_size > 50000 and dl.value.suggested_filename.endswith(".mp4")
    # log paths are short (bug 7)
    page.click("#backTo3")
    page.click("#logDetails summary")
    log = page.inner_text("#log")
    assert "pipeline/build_video.py" in log
    assert str(server.tmp) not in log and "/home/" not in log and ":\\" not in log
    assert "Finished in" in page.inner_text("#procTime")
    assert page.errors == []


def test_my_videos_drawer_reopens_finished_video(page, server):
    page.click("#libraryBtn")
    page.wait_for_selector("#drawer:not([hidden]) .buildcard")
    assert "finished" in page.inner_text("#buildsSub")
    shot(page, "e2e_drawer")
    card = page.locator(".buildcard", has=page.locator(".badge.done")).first
    card.click()
    page.wait_for_selector("#step-4.on", timeout=15000)
    assert page.is_hidden("#drawer")
    assert page.get_attribute("#downloadBtn", "href").endswith("/download")
    # every card shows its name, even with many videos (cards used to be squeezed)
    page.click("#libraryBtn")
    page.wait_for_selector("#drawer:not([hidden]) .buildcard")
    heights = page.eval_on_selector_all(".buildcard .body", "bs => bs.map(b => b.getBoundingClientRect().height)")
    assert len(heights) >= 2 and min(heights) > 30, heights
    assert "Untitled build" not in page.inner_text("#builds")
    page.keyboard.press("Escape")
    assert page.is_hidden("#drawer")


def test_regenerate_starts_new_build(page, server):
    page.click("#libraryBtn")
    page.locator(".buildcard", has=page.locator(".badge.done")).first.click()
    page.wait_for_selector("#step-4.on")
    before = page.get_attribute("#downloadBtn", "href")
    page.click("#regenBtn")
    page.wait_for_selector("#step-3.on", timeout=15000)
    page.wait_for_selector("#step-4.on", timeout=240000)
    assert page.get_attribute("#downloadBtn", "href") != before


def test_cancel_build(page):
    upload(page, *ALL3, wait_count=3)
    to_step2(page)
    page.click("#startBuild")
    page.wait_for_selector("#step-3.on")
    page.wait_for_selector("#cancelBuild:not([hidden])")
    page.click("#cancelBuild")
    page.wait_for_function("/cancelled|failed/i.test(document.querySelector('#procTitle').textContent)", timeout=60000)


# ───────────────────────────── session loss (bug 2) ─────────────────────────────

def test_lost_session_recovers_without_no_job_errors(page, server):
    """Bug 2: 'Could not save the script: No job …' / 'Could not start: No job …'."""
    upload(page, *ALL3, wait_count=3)
    jid = job_id(page)
    to_step2(page)
    page.fill("#script", "caption")
    server.stop()
    shutil.rmtree(server.tmp / "uploads" / jid, ignore_errors=True)
    shutil.rmtree(server.tmp / "outputs" / jid, ignore_errors=True)
    server.start()
    page.click("#startBuild")
    page.wait_for_selector("#step-1.on", timeout=15000)
    assert page.is_visible("#sessionNotice")
    t = " | ".join(toasts(page))
    assert "No job" not in t and "Could not save the script" not in t and "Could not start" not in t
    assert "session expired" in t
    assert page.locator("#clipList .clip").count() == 0
    page.wait_for_selector("#loader", state="hidden", timeout=3000)   # fades out
    # and the user can carry on
    upload(page, "robot_arm.mp4", wait_count=1)
    assert page.is_hidden("#sessionNotice")
    assert page.errors == []
