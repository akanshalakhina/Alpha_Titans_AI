"""
Shared setup for the Phase 1 agent tests.

Every session gets its own uploads/outputs root so nothing here can touch a
real build. Small real clips are synthesised with ffmpeg rather than shipped,
so the suite stays fast and the fixtures directory does not grow.

No test requires an API key. The EditingDirector's LLM is exercised through an
injected transport, which is the only place a mock appears: the ffmpeg work,
the measurements, the renders and the critic's inspection are all real.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_TMP = Path(tempfile.mkdtemp(prefix="at_agents_"))

os.environ["AT_UPLOADS_DIR"] = str(_TMP / "uploads")
os.environ["AT_OUTPUTS_DIR"] = str(_TMP / "outputs")
os.environ["AT_MUSIC_DIR"] = str(_TMP / "music")
os.environ.pop("AT_VISION_API_KEY", None)
os.environ.pop("AT_LLM_API_KEY", None)

sys.path.insert(0, str(ROOT / "server"))

import config  # noqa: E402

config.ensure_dirs()

from agents.artifacts import (ClipAnalysis, ClipAnalysisSet, ClipMotion,  # noqa: E402
                              ClipTechnical, ClipUsability)
from agents.render_pool import MediaPool  # noqa: E402
from agents.tools.llm_tools import LLMClient, LLMReply  # noqa: E402

FFMPEG = os.environ.get("AT_FFMPEG", "ffmpeg")


# ------------------------------------------------------------ real media

def _make_clip(path: Path, source: str, seconds: float = 6.0,
               size: str = "320x240") -> Path:
    """Render a real, decodable clip with ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-y",
           "-f", "lavfi", "-i", f"{source}=s={size}:r=30:d={seconds}",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
           "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def media_root() -> Path:
    d = _TMP / "media"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def lively_clip(media_root) -> Path:
    """A busy clip: a moving test pattern reads high on YDIF."""
    return _make_clip(media_root / "lively.mp4", "testsrc2")


@pytest.fixture(scope="session")
def calm_clip(media_root) -> Path:
    """A calm but genuinely ALIVE clip - the shape a weak hook has.

    A slowly drifting gradient: nearly every frame differs (so the pipeline's
    live-share floor keeps it, exactly as it keeps a locked-off interview)
    while the per-frame change is small. Measured on this machine it reads
    ~0.99 mean YDIF against the lively clip's ~5.9, which is the contrast the
    hook tests depend on.

    An earlier attempt used temporal noise on flat colour; x264 removed the
    noise entirely and the clip measured as genuinely static, so the pipeline
    correctly rejected it as dead. The encoder has to be survivable for the
    fixture to mean anything.
    """
    path = media_root / "calm.mp4"
    if path.exists():
        return path
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", "gradients=s=320x240:r=30:d=6:speed=0.02",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True)
    return path


@pytest.fixture
def clip_dir(tmp_path, lively_clip, calm_clip) -> Path:
    """A source folder in the pipeline's numeric-prefix convention."""
    d = tmp_path / "clips"
    d.mkdir()
    import shutil
    shutil.copy2(calm_clip, d / "01_calm.mp4")       # calm one FIRST
    shutil.copy2(lively_clip, d / "02_lively.mp4")
    return d


@pytest.fixture
def agent_root(tmp_path) -> Path:
    """An isolated uploads/outputs root for one test."""
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def pool():
    p = MediaPool(max_workers=2, name="test")
    yield p
    p.close()


# ------------------------------------------------------- synthetic input

def make_clip_analysis(clip_id: str, duration: float = 10.0,
                       mean_ydif: float = 5.0, live_share: float = 0.99,
                       orientation: str = "horizontal",
                       usable: bool = True,
                       series: list[float] | None = None) -> ClipAnalysis:
    """A ClipAnalysis with a controllable motion series.

    The series matters: best_window scans it, so a test that wants the
    liveliest window at a known offset can put it there.
    """
    rate = 10.0
    n = int(duration * rate)
    series = series if series is not None else [mean_ydif] * n
    return ClipAnalysis(
        clip_id=clip_id, path=f"/tmp/{clip_id}",
        technical=ClipTechnical(
            duration=duration, coded=(1920, 1080), display=(1920, 1080),
            fps=30.0, codec="h264", orientation=orientation),
        motion=ClipMotion(
            sample_fps=rate, mean_ydif=mean_ydif,
            max_ydif=max(series) if series else mean_ydif,
            live_share=live_share, freeze_runs=[], freeze_share=0.0,
            ydif_series=series, measured=True),
        usability=ClipUsability(usable=usable,
                                reason="" if usable else "nothing changes in it"),
    )


@pytest.fixture
def clip_set() -> ClipAnalysisSet:
    """Two usable clips: a calm one first, a lively one second."""
    calm = make_clip_analysis("01_calm.mp4", duration=10.0, mean_ydif=1.0,
                              series=[1.0] * 100)
    lively = make_clip_analysis("02_lively.mp4", duration=10.0, mean_ydif=9.0,
                                series=[2.0] * 40 + [16.0] * 40 + [2.0] * 20)
    return ClipAnalysisSet(clips=[calm, lively], orientation="horizontal",
                           canvas=(1280, 720), target_seconds=20.0)


# --------------------------------------------------------------- fake LLM

def llm_returning(text: str, tokens=(120, 80)) -> LLMClient:
    """An LLMClient whose transport returns fixed text. No network, no key."""
    def transport(prompt: str, max_tokens: int) -> LLMReply:
        return LLMReply(text=text, tokens_in=tokens[0], tokens_out=tokens[1],
                        usd=0.001, meta={"provider": "injected"})
    return LLMClient(transport=transport)


def llm_returning_plan(segments, strategy="test strategy",
                       rationale="test rationale") -> LLMClient:
    """A well-formed plan reply, for the happy path."""
    return llm_returning(json.dumps({
        "strategy": strategy, "opening_rationale": rationale,
        "segments": segments, "rejected": [],
    }))


def llm_failing(error="ConnectionError: refused") -> LLMClient:
    def transport(prompt: str, max_tokens: int) -> LLMReply:
        return LLMReply(text="", error=error)
    return LLMClient(transport=transport)


@pytest.fixture
def no_llm() -> LLMClient:
    """The default state of this machine: no key configured."""
    return LLMClient()


# ------------------------------------------------- a real analysed job

import contextlib  # noqa: E402

from agents.base import AgentContext  # noqa: E402
from agents.clip_analysis import ClipAnalysisAgent  # noqa: E402
import shutil  # noqa: E402

# the two fixture clip names, used by the shared fixtures below
CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"

from agents import compat  # noqa: E402
from agents.artifacts import JobSettings  # noqa: E402
from agents.runner import make_job, run_job  # noqa: E402
from agents.store import ArtifactStore, shared_store  # noqa: E402
from agents.trace import TraceRecorder, shared_tracer  # noqa: E402


@pytest.fixture
def tracer_span():
    """Open a trace span the way the graph does, for direct agent calls."""
    @contextlib.contextmanager
    def _span(ctx, node, agent=None, attempt=1, inputs=None):
        with ctx.tracer.node(ctx.job.job_id, node, agent or node,
                             attempt=attempt,
                             input_artifacts=inputs or []) as span:
            yield span
    return _span


@pytest.fixture(scope="session")
def _analyzed_session(lively_clip, calm_clip):
    """Probe and measure two real clips once, and reuse across tests.

    This is real ffmpeg work on real files - it is the fixture that makes the
    EditingDirector's deterministic fallback genuine, because that fallback
    runs the production `plan` stage against this manifest.
    """
    import shutil

    root = _TMP / "analyzed"
    src = root / "clips"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / "01_calm.mp4")
    shutil.copy2(lively_clip, src / "02_lively.mp4")

    from agents.artifacts import JobSettings
    job = make_job(src=src, job_id="testanalyzed", root=root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=12.0))
    outputs = Path(job.build_dir).resolve().parent
    store = shared_store(outputs=outputs)
    tracer = shared_tracer(outputs=outputs)
    pool = MediaPool(max_workers=2, name="analyzed")
    ctx = AgentContext(job=job, store=store, tracer=tracer, pool=pool,
                       llm=LLMClient())

    agent = ClipAnalysisAgent()
    with tracer.node(job.job_id, "clip_analysis", "clip_analysis") as span:
        art = agent.execute(ctx, span, revision=0)
    yield art, ctx
    pool.close()


@pytest.fixture
def analyzed(_analyzed_session):
    return _analyzed_session


@pytest.fixture
def director_ctx(_analyzed_session, tmp_path):
    """A fresh context over the real analysed job.

    The build folder (and therefore the real manifest the production planner
    needs) is shared; the artifact store and trace are per-test so revision
    counters and assertions cannot bleed between tests.
    """
    art, base = _analyzed_session
    store = ArtifactStore(tmp_path / "state.db")
    tracer = TraceRecorder(tmp_path / "state.db")
    ctx = AgentContext(job=base.job, store=store, tracer=tracer,
                       pool=base.pool, llm=LLMClient())
    return ctx, art.payload


# ------------------------------- a published agent job + the real app
# Shared by test_compat.py and test_compat_live.py: both drive the REAL,
# unmodified FastAPI app against a real agent build, so the fixtures belong
# here rather than in whichever test file happened to need them first.

@pytest.fixture(scope="session")
def published_job(lively_clip, calm_clip):
    """A REAL agent build inside the session's config dirs, then published.

    `root=None` means make_job uses config.UPLOADS_DIR / OUTPUTS_DIR, which
    conftest pointed at this session's temp folder - the same folders
    adapter.OUTPUTS walks, so jobs.listing() can genuinely discover it.
    """
    import config

    src = Path(config.UPLOADS_DIR).parent / "compat_src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lively_clip, src / LIVELY)
    shutil.copy2(calm_clip, src / CALM)

    job = make_job(src=src, job_id="compatjob", root=None,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=10.0,
                                        title="COMPAT_TEST"))
    (Path(job.build_dir) / "agent_settings.json").write_text(
        job.settings.model_dump_json(indent=2))

    llm = llm_returning_plan(
        [{"clip_id": LIVELY, "target_seconds": 5.0, "reason": "open strong"},
         {"clip_id": CALM, "target_seconds": 5.0, "reason": "settle"}])
    final = run_job(job, llm=llm)
    assert final["status"] == "done", final

    payload = compat.project(job.job_id)
    return job, payload


@pytest.fixture(scope="session")
def client(published_job):
    """The real FastAPI app, unmodified."""
    from fastapi.testclient import TestClient

    import jobs as legacy_jobs
    import main

    # a clean registry, so discovery has to come off disk exactly as it would
    # on a freshly started server
    legacy_jobs._JOBS.clear()
    with TestClient(main.app) as c:
        yield c
