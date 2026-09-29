"""Regression suite: one test per bug reported against the new UI, plus
page-integrity and layout checks. Each must stay green on every change."""
import json
import re
import time
from pathlib import Path

import pytest
from conftest import FIX, ROOT, playable, shot, toasts, upload

INDEX = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")


# ─────────────── static integrity ───────────────

def test_every_id_the_script_uses_exists_in_the_page():
    used = set(re.findall(r"\$\('#([\w-]+)'", APP)) | set(re.findall(r"getElementById\('([\w-]+)'\)", APP))
    have = set(re.findall(r'id="([\w-]+)"', INDEX))
    dynamic = {"useTranscriptBtn"}                     # created at runtime
    assert used - have - dynamic == set()


def test_page_and_script_versions_match():
    page_v = re.search(r'name="at-ui" content="(\w+)"', INDEX).group(1)
    js_v = re.search(r"UI_VERSION = '(\w+)'", APP).group(1)
    assert page_v == js_v
    assert f"app.js?v={js_v}" in INDEX and f"styles.css?v={js_v}" in INDEX


# ─────────────── R1: ring stuck at 2% during AI processing ───────────────

def test_R1_ring_moves_while_a_long_stage_runs(page, server):
    """Server progress is flat (11.2%) for the whole of a 4-minute motion
    analysis; the ring must still climb and show time left."""
    now = time.time()
    stages = [{"id": i, "label": i, "detail": "", "weight": w, "lines": 3,
               "status": st, "started": s0, "finished": f0}
              for i, w, st, s0, f0 in [
                  ("probe", 4, "done", now - 200, now - 188), ("measure", 20, "running", now - 188, None),
                  ("vision", 14, "pending", None, None), ("plan", 3, "pending", None, None),
                  ("direct", 2, "pending", None, None), ("sheets", 7, "pending", None, None),
                  ("cut", 31, "pending", None, None), ("join", 15, "pending", None, None),
                  ("music", 8, "pending", None, None), ("master", 12, "pending", None, None)]]
    job = {"id": "feedfacecafe", "status": "running", "progress": 11.2, "started": now - 200,
           "finished": None, "current": "Motion Analysis", "settings": {}, "stages": stages,
           "script": "", "captions": [], "voice": None, "output": None, "edit_plan": None, "vision": None,
           "analysis": {"total_duration": 439, "target": 120, "clips": [], "segments": [], "rejected": []}}
    page.route("**/api/jobs", lambda r: r.fulfill(json={"jobs": [{"id": job["id"], "status": "running",
               "created": now, "title": "long build", "progress": 11.2, "clips": 2}]}) if r.request.method == "GET" else r.continue_())
    page.route("**/api/jobs/feedfacecafe", lambda r: r.fulfill(json=job))
    page.route("**/api/jobs/feedfacecafe/clips", lambda r: r.fulfill(json={"clips": []}))
    page.route("**/api/jobs/feedfacecafe/events", lambda r: r.fulfill(
        body="data: " + json.dumps({"kind": "snapshot", "job": job}) + "\n\n", content_type="text/event-stream"))
    page.click("#libraryBtn")
    page.click(".buildcard")
    page.wait_for_selector("#step-3.on")
    a = float(page.inner_text("#ringPct").rstrip("%"))
    page.wait_for_timeout(3000)
    b = float(page.inner_text("#ringPct").rstrip("%"))
    assert a > 20, f"ring shows {a}% although 200 s of ~480 s have passed"
    assert b > a, f"ring did not move: {a} -> {b}"
    assert "left" in page.inner_text("#procTime")
    assert "live" in page.get_attribute("#ring", "class")
    shot(page, "reg_R1_ring_moving")


# ─────────────── R2: "No job …" errors ───────────────
# covered end-to-end by test_e2e::test_lost_session_recovers_without_no_job_errors

def test_R2_stale_page_markup_does_not_crash(page, server):
    """The real cause of R2: new app.js served with an older index.html."""
    stale = INDEX
    for block in (r'\s*<div id="loader".*?</button>\s*</div>\s*</div>', r'\s*<div id="sessionNotice".*?</div>\s*</div>',
                  r'\s*<div class="proc-time" id="procTime" hidden></div>', r'\s*<meta name="at-ui"[^>]*>'):
        stale = re.sub(block, "", stale, flags=re.S)
    assert 'id="loader"' not in stale and 'id="sessionNotice"' not in stale
    page.route("**/", lambda r: r.fulfill(body=stale, content_type="text/html") if r.request.resource_type == "document" else r.continue_())
    page.errors.clear()
    page.goto(server.url + "/")
    page.wait_for_selector("#step-1.on", timeout=10000)
    assert page.errors == [], page.errors
    assert any("out of date" in t for t in toasts(page))
    upload(page, "robot_arm.mp4", wait_count=1)        # still usable
    assert page.errors == []


# ─────────────── R4: play icon ───────────────

def test_R4_play_icon_visible_after_upload_and_on_hover(page):
    upload(page, "robot_arm.mp4", wait_count=1)
    hint = page.locator("#clipList .clip .playhint")
    assert hint.evaluate("e => getComputedStyle(e).opacity") != "0"
    page.hover("#clipList .clip .strip")
    page.wait_for_timeout(300)
    assert hint.evaluate("e => getComputedStyle(e).opacity") != "0"
    assert page.eval_on_selector("#clipList video", "v => v.paused")


# ─────────────── R5: reorder 500 ───────────────

def test_R5_reorder_while_preview_playing_and_repeatedly(page, server):
    playable(page)
    upload(page, "Gear box build.mp4", "robot_arm.mp4", "vertical tower.mp4", wait_count=3)
    for i in range(3):
        page.locator("#clipList .clip").nth(0).locator(".strip").click()
        page.wait_for_timeout(400)
        page.drag_and_drop("#clipList .clip:nth-child(1)", "#clipList .clip:nth-child(3)")
        page.wait_for_timeout(1200)
    assert not any("bad" in t for t in toasts(page)), toasts(page)
    assert page.locator("#clipList .clip").count() == 3


# ─────────────── R6: voice note label hidden on hover ───────────────

def test_R6_voice_player_label_stays_visible(page):
    upload(page, "robot_arm.mp4", wait_count=1)
    page.click("#toStep2")
    page.click("#recordBtn"); page.wait_for_timeout(1800); page.click("#recordBtn")
    page.wait_for_selector("#voiceBox .vplayer", timeout=15000)
    assert page.locator("#voiceBox audio[controls]").count() == 0   # no native player
    for action in ("hover", "click"):
        getattr(page, action)("#voiceBox .vplay")
        page.wait_for_timeout(300)
        box = page.locator("#voiceBox .vhead strong").bounding_box()
        assert box and box["width"] > 20 and page.is_visible("#voiceBox .vhead strong")
        assert page.is_visible("#voiceBox .vtime")


# ─────────────── layout at three widths ───────────────

@pytest.mark.parametrize("width", [390, 768, 1440])
def test_layout_no_horizontal_scroll_on_any_step(browser, server, width):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(server.url + "/"); pg.wait_for_selector("#loader", state="hidden")
    pg.set_input_files("#clipInput", [str(FIX / n) for n in ("Gear box build.mp4", "vertical tower.mp4")])
    pg.wait_for_selector("#loader", state="hidden", timeout=60000)
    for step in (1, 2, 3, 4):
        pg.evaluate(f"document.querySelectorAll('.step').forEach(s => s.classList.toggle('on', s.id === 'step-{step}'))")
        pg.wait_for_timeout(250)
        over = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert over <= 1, f"step {step} at {width}px overflows by {over}px"
        shot(pg, f"layout_{width}_step{step}", full=True)
    assert errs == []
    ctx.close()
