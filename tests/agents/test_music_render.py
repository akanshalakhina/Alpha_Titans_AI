"""
Music -> Render wiring.

The Music agent chose the score; this is the path that makes it audible:

    Music.mood
        -> RenderSpec.music_source / mood / keep_audio   (compile_render_spec)
        -> extras.music_stage  -> generate_music -> score.wav + _audio.json
        -> extras.master_stage -> the muxed, loudness-normalised mp4
        -> RenderResult.music_source / mood              (what the critic reads)

WHY THIS FILE EXISTS
--------------------
`run_master` used to pass two literals:

    "music_source": "none",
    "keep_audio": "none",

with a comment reading "Phase 1 has no Music Agent". The Music agent arrived,
the literals did not change, `music_stage` was never called at all, and every
agent-built video shipped as digital silence - measured at -91 dB against the
classic pipeline's -17.9 dB. The critic scored those renders 100/100, because
its loudness check returned passed=True unconditionally for the same reason.

So the property under test is not "a field is set". It is: the Music artifact
changes the bytes of the delivered file, and a silent delivery can no longer
pass.
"""

import inspect
import pathlib
import typing

import pytest

from agents.artifacts import (EditPlan, JobSettings, Music, MusicSection,
                              RenderResult, RenderSpec, RenderTechnical,
                              Segment)
from agents.tools import render_tools as RT
from agents.tools.render_tools import compile_render_spec

import extras as extras_mod

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"
RENDER_TOOLS = pathlib.Path(RT.__file__)


# ------------------------------------------------------------- helpers

def a_music(mood="beat", n=2):
    return Music(
        domain="model engineering", music_type="reasoned", mood=mood,
        mood_rationale="measured motion sits in the beat band",
        engine="beatbed", tempo_feel="moderate",
        sections=[MusicSection(index=i + 1, beat_index=i + 1,
                               clip_id=LIVELY if i % 2 == 0 else CALM,
                               energy="steady")
                  for i in range(n)],
        insufficient_signal=False, confidence=0.8, live=True)


def a_plan(n=2):
    segs = [Segment(index=i + 1, clip_id=f"c{i}.mp4", start=0.0, end=4.0,
                    duration=4.0, orientation="horizontal")
            for i in range(n)]
    return EditPlan(segments=segs, total_duration=float(4 * n),
                    target_seconds=float(4 * n), canvas=(1920, 1080))


def a_result(**kw):
    base = dict(attempt=1, video_path="x.mp4", technical=RenderTechnical())
    base.update(kw)
    return RenderResult(**base)


# ------------------------- 1. the artifact reaches the spec

def test_a_music_artifact_turns_the_bed_on():
    spec = compile_render_spec("j", "b", "s", a_plan(), music=a_music())
    assert spec.music_source == "generate"
    assert spec.mood == "beat"


def test_the_mood_carried_is_the_one_the_agent_chose():
    for mood in typing.get_args(Music.model_fields["mood"].annotation):
        spec = compile_render_spec("j", "b", "s", a_plan(),
                                   music=a_music(mood=mood))
        assert spec.mood == mood


def test_no_music_artifact_keeps_the_previous_silent_path():
    """A job built or resumed before the Music agent existed must not change."""
    spec = compile_render_spec("j", "b", "s", a_plan())
    assert spec.music_source == "none"
    assert spec.mood is None
    assert spec.audio_settings() == {"music_source": "none",
                                     "keep_audio": "none"}


def test_keep_audio_defaults_to_the_apps_own_default():
    spec = compile_render_spec("j", "b", "s", a_plan(), music=a_music())
    assert spec.keep_audio == "none"


# ------------------------- 2. the vocabulary cannot drift

def test_the_mood_vocabulary_is_exactly_what_the_engine_implements():
    assert set(typing.get_args(Music.model_fields["mood"].annotation)) == \
        set(extras_mod.MOODS)


def _flatten_literal(annotation):
    """Mood | None -> the mood strings. get_args gives (Literal[...], None)."""
    out = set()
    for arg in typing.get_args(annotation) or (annotation,):
        if arg is type(None):
            continue
        inner = typing.get_args(arg)
        out |= set(inner) if inner else {arg}
    return out


def test_render_spec_mood_matches_the_engine_too():
    assert _flatten_literal(RenderSpec.model_fields["mood"].annotation) ==         set(extras_mod.MOODS)


def test_keep_audio_vocabulary_is_the_adapters():
    import adapter
    allowed = set(typing.get_args(
        RenderSpec.model_fields["keep_audio"].annotation))
    assert allowed == set(adapter.KEEP_AUDIO)


@pytest.mark.parametrize("bad", ["disco", "techno", "", "BEAT", "fade",
                                 "../../etc", "beat;rm -rf /"])
def test_an_unsupported_mood_cannot_reach_the_spec(bad):
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        RenderSpec(job_id="j", build_dir="b", src_dir="s",
                   canvas=(1920, 1080), crf=18, segments=[],
                   music_source="generate", mood=bad)


def test_generating_without_a_mood_is_refused():
    """generate_music() silently falls back to 'beat'; refuse instead."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        RenderSpec(job_id="j", build_dir="b", src_dir="s",
                   canvas=(1920, 1080), crf=18, segments=[],
                   music_source="generate")


# ------------------------- 3. the spec reaches the real stages

def test_audio_settings_are_the_keys_extras_actually_reads():
    """The dict must use the names music_stage/master_stage look up."""
    src = pathlib.Path(extras_mod.__file__).read_text(encoding="utf-8",
                                                      errors="replace")
    spec = compile_render_spec("j", "b", "s", a_plan(), music=a_music())
    for key in spec.audio_settings():
        assert f'"{key}"' in src or f"'{key}'" in src, \
            f"extras.py never reads settings[{key!r}]"


def test_run_music_hands_the_spec_to_the_existing_music_stage(monkeypatch):
    seen = {}
    monkeypatch.setattr(extras_mod, "music_stage",
                        lambda job_id, settings, on_line: seen.update(
                            job_id=job_id, **settings) or {"track": "s.wav"})
    spec = compile_render_spec("j", "b", "s", a_plan(), music=a_music("lofi"))
    RT.run_music("j", spec)
    assert seen["music_source"] == "generate"
    assert seen["mood"] == "lofi"
    assert seen["keep_audio"] == "none"


def test_run_master_no_longer_hardcodes_silence(monkeypatch):
    seen = {}
    monkeypatch.setattr(extras_mod, "master_stage",
                        lambda job_id, settings, on_line: seen.update(
                            settings) or {"path": "out.mp4"})
    spec = compile_render_spec("j", "b", "s", a_plan(), music=a_music("calm"))
    RT.run_master("j", spec)
    assert seen["music_source"] == "generate"
    assert seen["mood"] == "calm"


def test_the_two_stages_are_told_the_same_thing(monkeypatch):
    """One decision, read twice - not two independent audio opinions."""
    got = {}
    monkeypatch.setattr(extras_mod, "music_stage",
                        lambda j, s, o: got.__setitem__("music", dict(s))
                        or {"track": None})
    monkeypatch.setattr(extras_mod, "master_stage",
                        lambda j, s, o: got.__setitem__("master", dict(s))
                        or {"path": "o.mp4"})
    spec = compile_render_spec("j", "b", "s", a_plan(), music=a_music())
    RT.run_music("j", spec)
    RT.run_master("j", spec)
    for key in ("music_source", "mood", "keep_audio"):
        assert got["music"][key] == got["master"][key]


def test_the_literal_silence_defaults_are_gone():
    """Regression guard: the two literals that caused the bug."""
    src = RENDER_TOOLS.read_text(encoding="utf-8")
    body = src[src.index("def run_master"):]
    assert '"music_source": "none"' not in body
    assert '"keep_audio": "none"' not in body


def test_the_render_agent_runs_music_before_master():
    import agents.rendering as R
    src = inspect.getsource(R.RenderingAgent.execute)
    assert "run_music" in src, "the music stage is never invoked"
    assert src.index("run_music") < src.index("run_master"), \
        "music_stage must run first; master_stage reads its _audio.json"


# ------------------------- 4. provenance the critic can read

def test_the_render_result_records_what_was_asked_for():
    r = a_result(music_source="generate", mood="beat")
    assert r.wanted_audio is True
    assert r.mood == "beat"


def test_an_old_render_result_still_loads_and_reads_as_silent():
    r = a_result()
    assert r.wanted_audio is False
    assert r.music_source == "none"


def test_the_agent_puts_the_spec_on_the_artifact():
    import agents.rendering as R
    src = inspect.getsource(R.RenderingAgent.execute)
    assert "music_source=spec.music_source" in src
    assert "mood=spec.mood" in src


# ------------------------- 5. the critic can no longer bless silence

def _loudness_checks(render, lufs):
    """Run just the critic's loudness branch against a fake measurement."""
    from agents.critic import SILENCE_LUFS
    wanted = render.wanted_audio
    silent = lufs is None or lufs <= SILENCE_LUFS
    return wanted, silent


def test_silence_fails_when_a_bed_was_requested():
    wanted, silent = _loudness_checks(
        a_result(music_source="generate", mood="beat"), -70.0)
    assert wanted and silent, "this combination must be judged a failure"


def test_silence_passes_when_none_was_requested():
    wanted, _ = _loudness_checks(a_result(), -70.0)
    assert not wanted


def test_an_audible_bed_is_not_flagged():
    _, silent = _loudness_checks(
        a_result(music_source="generate", mood="beat"), -16.2)
    assert not silent


def test_the_critic_no_longer_passes_loudness_unconditionally():
    import agents.critic as C
    src = pathlib.Path(C.__file__).read_text(encoding="utf-8")
    body = src[src.index("# --- loudness"):src.index("def _creative")]
    assert "severity=\"blocking\"" in body, \
        "a silent delivery must be able to block"
    assert "render.wanted_audio" in body, \
        "the check must consider what was actually requested"


def test_the_silence_floor_sits_well_clear_of_a_normalised_bed():
    from agents.critic import SILENCE_LUFS, TARGET_LUFS, LUFS_TOLERANCE
    assert SILENCE_LUFS < TARGET_LUFS - LUFS_TOLERANCE - 20, \
        "the floor must not be able to catch a correctly normalised bed"


# ------------------------- 6. REAL MEDIA: the file is not silent

@pytest.fixture
def music_render_ctx(agent_root, calm_clip, lively_clip, pool):
    import shutil
    from agents.base import AgentContext
    from agents.clip_analysis import ClipAnalysisAgent
    from agents.runner import make_job
    from agents.store import shared_store
    from agents.trace import shared_tracer
    from agents.tools.llm_tools import LLMClient

    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    job = make_job(src=src, job_id="musicjob", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=8.0))
    outputs = pathlib.Path(job.build_dir).resolve().parent
    ctx = AgentContext(job=job, store=shared_store(outputs=outputs),
                       tracer=shared_tracer(outputs=outputs),
                       pool=pool, llm=LLMClient())
    with ctx.tracer.node(job.job_id, "clip_analysis", "clip_analysis") as span:
        ClipAnalysisAgent().execute(ctx, span, revision=0)
    return ctx


def _real_plan():
    return EditPlan(
        segments=[
            Segment(index=1, clip_id=LIVELY, start=0.0, end=4.0,
                    duration=4.0, orientation="horizontal"),
            Segment(index=2, clip_id=CALM, start=0.0, end=4.0,
                    duration=4.0, orientation="horizontal"),
        ],
        total_duration=7.6, target_seconds=8.0, canvas=(1280, 720))


def _render(ctx, music):
    from agents.rendering import RenderingAgent
    rev = ctx.store.next_revision(ctx.job.job_id, "RenderResult")
    with ctx.tracer.node(ctx.job.job_id, "render", "rendering",
                         attempt=1) as span:
        return RenderingAgent().execute(ctx, span, plan=_real_plan(),
                                        attempt=1, revision=rev, music=music)


@pytest.mark.slow
def test_real_render_with_music_is_not_digital_silence(music_render_ctx):
    """The whole point. A real file, measured, must be audible.

    The source clips are generated video-only, so every decibel in the
    output came from the bed the Music artifact asked for.
    """
    from agents.critic import SILENCE_LUFS
    from agents.tools import verify_tools as VT

    art = _render(music_render_ctx, a_music())
    out = art.payload.canonical_path or art.payload.video_path
    assert pathlib.Path(out).exists()

    lufs = VT.loudness(out)
    assert lufs is not None, "loudness could not be measured at all"
    assert lufs > SILENCE_LUFS, (
        f"delivered file measures {lufs} LUFS - that is digital silence, "
        f"which is the bug this wiring exists to fix")


@pytest.mark.slow
def test_real_render_writes_the_score_the_engine_was_asked_for(
        music_render_ctx):
    art = _render(music_render_ctx, a_music())
    build = pathlib.Path(music_render_ctx.job.build_dir)
    assert (build / "_audio.json").exists(), "music_stage never ran"
    assert art.payload.music_source == "generate"
    assert art.payload.mood == "beat"
    assert art.payload.audio_track, "no track recorded on the artifact"


@pytest.mark.slow
def test_real_render_without_music_stays_silent_and_says_so(music_render_ctx):
    """The old behaviour is still reachable and still honest."""
    art = _render(music_render_ctx, None)
    assert art.payload.music_source == "none"
    assert art.payload.wanted_audio is False
