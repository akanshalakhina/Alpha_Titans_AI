"""E2E fixtures: a real Alpha Titans server (real pipeline, real ffmpeg) in
temporary folders, and a Chromium page with a fake microphone."""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
SHOTS = ROOT / "tests" / "screenshots"


class Server:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="at_e2e_"))
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]; s.close()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = None

    def start(self):
        env = dict(os.environ, AT_UPLOADS_DIR=str(self.tmp / "uploads"),
                   AT_OUTPUTS_DIR=str(self.tmp / "outputs"), AT_MUSIC_DIR=str(self.tmp / "music"))
        env.pop("AT_VISION_API_KEY", None)
        for d in ("uploads", "outputs", "music"):
            (self.tmp / d).mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "server"),
             "--host", "127.0.0.1", "--port", str(self.port)],
            env=env, stdout=open(self.tmp / "server.log", "ab"), stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                urllib.request.urlopen(self.url + "/api/health", timeout=1)
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("server did not start")

    def stop(self):
        if self.proc:
            self.proc.terminate(); self.proc.wait(10); self.proc = None

    def restart(self):
        self.stop(); self.start()


@pytest.fixture(scope="session")
def server():
    s = Server(); s.start()
    yield s
    s.stop()
    shutil.rmtree(s.tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        # AT_TEST_BROWSER=msedge or chrome runs the suite in the real browser
        # (which, unlike the bundled Chromium, plays H.264)
        channel = os.environ.get("AT_TEST_BROWSER") or None
        b = p.chromium.launch(channel=channel, args=["--use-fake-ui-for-media-stream",
                                    "--use-fake-device-for-media-stream",
                                    "--autoplay-policy=no-user-gesture-required"])
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                              permissions=["microphone"], accept_downloads=True)
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text
          and pg.errors.append(m.text))
    pg.goto(server.url + "/")
    pg.wait_for_selector("#loader", state="hidden", timeout=15000)
    yield pg
    ctx.close()


def playable(pg):
    """The open-source Chromium used for testing has no H.264 decoder (Chrome
    and Edge do). Hand it a VP9 copy for clip previews and the final player so
    the app's playback logic can be exercised; the real H.264 render is checked
    by ffprobe in the backend suite."""
    body = (FIX / "preview.webm").read_bytes()
    handler = lambda r: r.fulfill(status=200, body=body, content_type="video/webm",
                                  headers={"Accept-Ranges": "none"})
    pg.route("**/api/jobs/*/clip/**", handler)
    pg.route("**/api/jobs/*/file/*.mp4", handler)


def upload(pg, *names, wait_count=None):
    pg.set_input_files("#clipInput", [str(FIX / n) for n in names])
    pg.wait_for_selector("#loader", state="hidden", timeout=60000)
    if wait_count:
        pg.wait_for_function(f"document.querySelectorAll('#clipList .clip').length === {wait_count}")


def toasts(pg):
    return pg.eval_on_selector_all(".toast", "els => els.map(e => e.className + ' ' + e.textContent)")


def shot(pg, name, full=False):
    SHOTS.mkdir(parents=True, exist_ok=True)
    pg.screenshot(path=str(SHOTS / f"{name}.png"), full_page=full)
