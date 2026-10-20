"""AI-composed music: Lyria 3 composes, the Music agent directs.

What must hold:
* the prompt carries the cut's real length, a tempo fitted to its marks,
  and the agent's sections at timestamps that land on real transitions;
* a refused prompt is retried once with words from closed vocabularies
  only, so nothing a model wrote reaches the second request;
* a track is paid for once per cut, not once per call;
* anything missing or failing hands over to the code-composed score, so
  the cut is never left silent because an AI service said no.

No test here reaches the network: the HTTP opener is injected.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error

import pytest

from agents.artifacts import EditPlan, Music, MusicSection, Segment
from agents.tools import ai_music as AM
from agents.tools import render_tools as RT
from agents.tools.render_tools import compile_render_spec
import extras as extras_mod


def a_music(**kw) -> Music:
    base = dict(music_type="reasoned", mood="lofi", tempo_feel="moderate",
                engine="stdlib",
                sonic_direction="dusty keys under an ACME-brand sparkle",
                energy_arc="settles, lifts at the reveal, then resolves",
                instrumentation=["electric piano", "brushed drums"],
                sections=[MusicSection(index=i + 1, beat_index=i + 1,
                                       clip_id="a.mp4", energy=e,
                                       intent=f"part {i + 1} of the build")
                          for i, e in enumerate(["low", "building", "peak"])])
    base.update(kw)
    return Music(**base)


class Reply:
    def __init__(self, body: bytes):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def lyria_ok(audio=b"ID3" + b"\x00" * 4000):
    return json.dumps({"status": "completed", "outputs": [
        {"type": "text", "text": "bpm: 90 duration_secs: 30"},
        {"type": "audio", "mime_type": "audio/mpeg",
         "data": base64.b64encode(audio).decode()}]}).encode()


def refused(code="content_blocked", status=400):
    body = json.dumps({"error": {"code": code, "message": "blocked"}}).encode()
    return urllib.error.HTTPError("u", status, "x", {}, io.BytesIO(body))


class FakeHttp:
    """Answers the metadata server and Lyria from a script."""

    def __init__(self, *lyria):
        self.lyria = list(lyria)
        self.prompts: list[str] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        if "metadata.google.internal" in url:
            if url.endswith("/token"):
                return Reply(json.dumps({"access_token": "tok"}).encode())
            return Reply(b"test-project-123")
        self.prompts.append(json.loads(req.data)["input"][0]["text"])
        nxt = self.lyria.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return Reply(nxt)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(AM.time, "sleep", lambda s: None)
    monkeypatch.delenv("AT_MUSIC_PROJECT", raising=False)
    monkeypatch.delenv("AT_GCP_PROJECT", raising=False)


# ------------------------------------------------------------- the prompt

def test_sections_land_on_the_cuts_real_transitions():
    spans = AM.section_spans(3, 30.0, [9.6, 19.8, 25.0])
    assert spans[0][0] == 0.0 and spans[-1][1] == 30.0
    assert [round(a, 1) for a, _ in spans[1:]] == [9.6, 19.8]


def test_without_marks_the_sections_split_the_cut_evenly():
    assert AM.section_spans(2, 20.0, []) == [(0.0, 10.0), (10.0, 20.0)]


def test_the_prompt_carries_length_tempo_key_and_the_agents_arc():
    p = AM.build_prompt(a_music(), 28.0, [9.6, 19.8], 92.4, 55)
    assert p.startswith("Create a 32-second instrumental track at 92 BPM "
                        "in G minor."), "the cut plus a pad, never less"
    assert "[0:00 - 0:10] part 1 of the build. Intensity: 2/10." in p
    assert "Intensity: 5/10" in p and "Intensity: 8/10" in p
    assert "electric piano" in p and "no vocals" in p.lower()


def test_the_plain_prompt_carries_no_words_a_model_wrote():
    p = AM.build_prompt(a_music(), 28.0, [9.6, 19.8], 92.0, 55, plain=True)
    assert "ACME" not in p and "part 1 of the build" not in p
    assert "electric piano" not in p
    assert "Intensity: 8/10" in p, "the arc survives, as numbers"


def test_the_last_section_runs_to_the_end_of_what_is_asked_for():
    p = AM.build_prompt(a_music(), 28.0, [9.6, 19.8], 92.0, 55)
    assert "[0:20 - 0:32]" in p


def test_a_very_long_cut_asks_for_no_more_than_lyria_can_make():
    p = AM.build_prompt(a_music(), 400.0, [], 90.0, 55)
    assert p.startswith("Create a 184-second")


# --------------------------------------------------------------- the call

def test_a_completed_reply_gives_the_audio_and_what_the_model_said():
    audio, mime, said = AM.request_track("p", "tok", "proj",
                                         http=FakeHttp(lyria_ok(b"MP3")))
    assert audio == b"MP3" and mime == "audio/mpeg" and "bpm: 90" in said


def test_a_reply_without_audio_is_a_refusal_not_a_silent_success():
    body = json.dumps({"status": "completed", "outputs": []}).encode()
    with pytest.raises(AM.Refused) as e:
        AM.request_track("p", "tok", "proj", http=FakeHttp(body))
    assert e.value.code == "no_audio"


def test_composing_writes_the_track_and_records_how(tmp_path):
    http = FakeHttp(lyria_ok())
    path, plan = AM.compose(a_music(), "job1", tmp_path, 28.0, [9.6, 19.8],
                            http=http)
    assert path is not None and path.read_bytes().startswith(b"ID3")
    assert plan["provider"] == "lyria" and plan["model"] == AM.DEFAULT_MODEL
    assert plan["bpm"] and "error" not in plan
    assert plan["plain_prompt"] is False


def test_a_refused_prompt_is_retried_once_with_the_plain_one(tmp_path):
    http = FakeHttp(refused(), lyria_ok())
    path, plan = AM.compose(a_music(), "job1", tmp_path, 28.0, [9.6],
                            http=http)
    assert path is not None and plan["plain_prompt"] is True
    assert "ACME" in http.prompts[0] and "ACME" not in http.prompts[1]


def test_refusals_are_random_so_the_plain_prompt_is_asked_twice(tmp_path):
    """Measured: the same prompt refused, then accepted on resend."""
    http = FakeHttp(refused(), refused(), lyria_ok())
    path, plan = AM.compose(a_music(), "job1", tmp_path, 28.0, [], http=http)
    assert path is not None and plan["plain_prompt"] is True
    assert http.prompts[1] == http.prompts[2], "the same plain prompt again"


def test_three_refusals_give_up_for_the_fallback(tmp_path):
    http = FakeHttp(refused(), refused(), refused())
    path, plan = AM.compose(a_music(), "job1", tmp_path, 28.0, [], http=http)
    assert path is None and "content_blocked" in plan["error"]
    assert len(http.prompts) == 3, "three requests, never a fourth"


def test_a_busy_service_is_tried_once_more(tmp_path):
    http = FakeHttp(refused("RESOURCE_EXHAUSTED", 429), lyria_ok())
    path, _ = AM.compose(a_music(), "job1", tmp_path, 28.0, [], http=http)
    assert path is not None and len(http.prompts) == 2


def test_the_same_cut_is_paid_for_once(tmp_path):
    first = FakeHttp(lyria_ok())
    AM.compose(a_music(), "job1", tmp_path, 28.0, [9.6], http=first)
    again = FakeHttp()                     # any request would fail: none left
    path, plan = AM.compose(a_music(), "job1", tmp_path, 28.0, [9.6],
                            http=again)
    assert path is not None and plan.get("cached") is True
    assert again.prompts == []


def test_no_project_means_no_request(tmp_path, monkeypatch):
    def offline(req, timeout=None):
        raise OSError("no metadata server here")
    path, plan = AM.compose(a_music(), "job1", tmp_path, 28.0, [],
                            http=offline)
    assert path is None and "project" in plan["error"]


def test_it_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("AT_MUSIC_PROVIDER", raising=False)
    assert AM.enabled() is False
    monkeypatch.setenv("AT_MUSIC_PROVIDER", "lyria")
    assert AM.enabled() is True


# ------------------------------------------------- inside the music stage

def a_plan(n=2):
    segs = [Segment(index=i + 1, clip_id=f"c{i}.mp4", start=0.0, end=4.0,
                    duration=4.0, orientation="horizontal") for i in range(n)]
    return EditPlan(segments=segs, total_duration=float(4 * n),
                    target_seconds=float(4 * n), canvas=(1920, 1080))


def _stage(monkeypatch):
    seen = {}
    monkeypatch.setattr(extras_mod, "music_stage",
                        lambda job_id, settings, on_line: seen.update(settings)
                        or {"track": "t.wav"})
    return seen


def test_the_ai_track_is_what_the_music_stage_fits(monkeypatch, tmp_path):
    monkeypatch.setenv("AT_MUSIC_PROVIDER", "lyria")
    track = tmp_path / "_lyria_x.mp3"
    track.write_bytes(b"ID3")
    monkeypatch.setattr(AM, "compose", lambda *a, **k: (
        track, {"provider": "lyria", "model": AM.DEFAULT_MODEL, "bpm": 90}))
    seen = _stage(monkeypatch)
    spec = compile_render_spec("j", str(tmp_path), "s", a_plan(),
                               music=a_music())
    out = RT.run_music("j", spec, None, a_music())
    assert seen["music_source"] == "upload"
    assert seen["music_file"] == str(track)
    assert out["score_plan"]["provider"] == "lyria"


def test_a_failed_ai_track_falls_back_to_the_code_score(monkeypatch,
                                                        tmp_path):
    monkeypatch.setenv("AT_MUSIC_PROVIDER", "lyria")
    monkeypatch.setattr(AM, "compose", lambda *a, **k: (
        None, {"provider": "lyria", "error": "HTTP 403 PERMISSION_DENIED"}))
    from agents.tools import score_tools as SC
    wav = tmp_path / "_agent_score.wav"
    monkeypatch.setattr(SC, "compose", lambda *a, **k: (
        wav, {"style": "lofi", "bpm": 88.0, "root_midi": 55, "seed": 1}))
    seen = _stage(monkeypatch)
    spec = compile_render_spec("j", str(tmp_path), "s", a_plan(),
                               music=a_music())
    out = RT.run_music("j", spec, None, a_music())
    assert seen["music_file"] == str(wav), "still scored, by code"
    assert out["score_plan"]["ai_attempt"]["error"].startswith("HTTP 403")


def test_with_the_provider_off_lyria_is_never_called(monkeypatch, tmp_path):
    monkeypatch.delenv("AT_MUSIC_PROVIDER", raising=False)
    monkeypatch.setattr(AM, "compose", lambda *a, **k: pytest.fail(
        "Lyria was called with AT_MUSIC_PROVIDER unset"))
    from agents.tools import score_tools as SC
    wav = tmp_path / "_agent_score.wav"
    monkeypatch.setattr(SC, "compose", lambda *a, **k: (
        wav, {"style": "lofi", "bpm": 88.0, "root_midi": 55, "seed": 1}))
    seen = _stage(monkeypatch)
    spec = compile_render_spec("j", str(tmp_path), "s", a_plan(),
                               music=a_music())
    out = RT.run_music("j", spec, None, a_music())
    assert seen["music_file"] == str(wav)
    assert "ai_attempt" not in out["score_plan"]
