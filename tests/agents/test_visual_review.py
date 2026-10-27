"""The Quality team's Visual Reviewer, and the loop it closes.

What must hold:
* every shot of the FINISHED video is located correctly, crossfades
  included, and looked at clear of its fades;
* a model answer is used only when it is complete and well-formed - a short
  list, a word instead of a number, is "not reviewed", never a guess;
* a weak shot turns a passing video into a revision whose constraint names
  the rejected moment, and rejected moments from earlier attempts are
  carried forward so the loop cannot drift back to them;
* the EditingDirector, the shot check and the length top-up all treat a
  rejected moment as spent;
* end to end, a weak shot is re-cut from somewhere else and the second
  render passes.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from agents import visual_review as VR
from agents.artifacts import (ClipAnalysisSet, CriticResult, EditPlan,
                              JobSettings, RevisionRequest, Segment,
                              ShotReview, VisualReview)
from agents.critic import CriticAgent
from agents.editing_director import EditingDirectorAgent, rejected_windows
from agents.tools import select_tools as ST
from agents.tools.llm_tools import LLMClient, LLMReply
from conftest import make_clip_analysis


def seg(i, clip="a.mp4", start=0.0, dur=5.0, caption=""):
    return Segment(index=i, clip_id=clip, start=start, end=start + dur,
                   duration=dur, orientation="horizontal", caption=caption)


def plan(*segs):
    return EditPlan(segments=list(segs),
                    total_duration=sum(s.duration for s in segs),
                    target_seconds=15.0, canvas=(1280, 720), strategy="s",
                    opening_rationale="r")


# -------------------------------------------------------------- timeline

def test_shots_are_placed_on_the_finished_timeline_with_crossfades():
    p = plan(seg(1, dur=5.0), seg(2, dur=4.0), seg(3, dur=6.0))
    assert VR.timeline(p, 0.5) == [(0.0, 5.0), (4.5, 4.0), (8.0, 6.0)]
    assert VR.timeline(p, 0.0) == [(0.0, 5.0), (5.0, 4.0), (9.0, 6.0)]


def test_frames_are_taken_inside_the_shot_clear_of_its_fades():
    times = VR.sample_times(10.0, 4.0, 0.5)
    assert len(times) == 3 and times == sorted(times)
    assert all(10.5 <= t <= 13.5 for t in times)


# --------------------------------------------------------------- parsing

def test_a_complete_answer_is_parsed_and_clamped():
    rows = VR.parse({"shots": [{"score": 12, "caption_matches": True},
                               {"score": "3", "problem": "a hand"},
                               7]}, 3)
    assert [r["score"] for r in rows] == [10.0, 3.0, 7.0]
    assert rows[0]["caption_ok"] is True and rows[1]["caption_ok"] is None
    assert rows[1]["problem"] == "a hand"


@pytest.mark.parametrize("raw", [
    {"shots": [{"score": 5}]},                     # too few
    {"shots": [{"score": "high"}, {"score": 2}]},  # not a number
    {"nothing": []},
    None,
])
def test_an_incomplete_answer_is_not_a_review(raw):
    assert VR.parse(raw, 2) is None


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AT_VISUAL_REVIEW", "0")
    assert VR.enabled() is False
    monkeypatch.setenv("AT_VISUAL_REVIEW", "1")
    assert VR.enabled() is True


# ----------------------------------------------------------------- critic

def review(*scores, attempt=1):
    shots = [ShotReview(index=i + 1, clip_id="a.mp4", start=10.0 * i,
                        end=10.0 * i + 5, at=5.0 * i, score=s)
             for i, s in enumerate(scores)]
    return VisualReview(attempt=attempt, reviewed=True, shots=shots,
                        weak=[x.index for x in shots if x.score < 4],
                        mean_score=sum(scores) / len(scores), summary="x")


def test_scores_become_a_creative_dimension_only_when_someone_looked():
    c = CriticAgent._shots(review(8, 6))
    assert c.dimension == "shots_show_subject" and c.score == 70.0
    assert CriticAgent._shots(VisualReview(attempt=1, reviewed=False)) is None
    assert CriticAgent._shots(None) is None


def test_weak_shots_become_a_revision_naming_the_moments():
    weak = CriticAgent._weak_windows(review(9, 2, 1))
    assert [w["index"] for w in weak] == [2, 3]
    req = CriticAgent._with_replacements(None, 1, weak, review(9, 2, 1), [])
    assert req.target_agent == "editing_director"
    assert req.constraints["replace_windows"] == [
        {"clip_id": "a.mp4", "start": 10.0, "end": 15.0},
        {"clip_id": "a.mp4", "start": 20.0, "end": 25.0}]


def test_moments_rejected_earlier_are_carried_forward():
    earlier = [{"key": "k", "constraints": {"replace_windows": [
        {"clip_id": "b.mp4", "start": 1.0, "end": 4.0}]}}]
    weak = CriticAgent._weak_windows(review(9, 2))
    req = CriticAgent._with_replacements(None, 2, weak, review(9, 2), earlier)
    got = {(w["clip_id"], w["start"]) for w in req.constraints["replace_windows"]}
    assert got == {("b.mp4", 1.0), ("a.mp4", 10.0)}


def test_a_hook_revision_keeps_its_constraints_and_gains_the_shots():
    hook = RevisionRequest(from_attempt=1, target_agent="editing_director",
                           issue="flat opening", severity="high",
                           constraints={"opening_max_duration": 8.0})
    weak = CriticAgent._weak_windows(review(2, 9))
    req = CriticAgent._with_replacements(hook, 1, weak, review(2, 9), [])
    assert req.constraints["opening_max_duration"] == 8.0
    assert req.constraints["replace_windows"]
    assert "flat opening" in req.issue and "shot 1" in req.issue


def test_no_weak_shots_leaves_the_request_alone():
    assert CriticAgent._with_replacements(None, 1, [], review(9), []) is None


# --------------------------------------------------------------- director

def busy(duration=60.0):
    series = [1.0] * int(duration * 10)
    for start in (50, 200, 350, 500):
        for i in range(start, min(start + 60, len(series))):
            series[i] = 20.0
    return make_clip_analysis("a.mp4", duration=duration, series=series)


def test_malformed_rejections_are_skipped_not_fatal():
    got = rejected_windows({"replace_windows": [
        {"clip_id": "a.mp4", "start": 1, "end": 4}, {"clip_id": "x"}, "junk"]})
    assert got == [("a.mp4", 1.0, 4.0)]
    assert rejected_windows({}) == []


class Span:
    def __init__(self):
        self.tools, self.notes = [], []

    def tool(self, name, **kw):
        self.tools.append((name, kw))

    def note(self, text):
        self.notes.append(text)


def test_the_director_never_recuts_a_rejected_moment():
    clip = busy()
    cs = ClipAnalysisSet(clips=[clip], orientation="horizontal",
                         canvas=(1280, 720), target_seconds=6.0)
    proposal = {"segments": [{"clip_id": "a.mp4", "target_seconds": 6.0,
                              "reason": "the build"}]}
    first, _ = EditingDirectorAgent()._segments_from_proposal(
        Span(), cs, proposal, {})
    a, b = first[0].start, first[0].end
    again, _ = EditingDirectorAgent()._segments_from_proposal(
        Span(), cs, proposal,
        {"replace_windows": [{"clip_id": "a.mp4", "start": a, "end": b}]})
    s = again[0]
    assert s.end <= a or s.start >= b, "the rejected window came back"


def test_the_length_top_up_never_picks_a_rejected_moment():
    clip = busy()
    cs = ClipAnalysisSet(clips=[clip], orientation="horizontal",
                         canvas=(1280, 720), target_seconds=30.0)
    start = [seg(1, start=0.0, dur=6.0)]
    banned = [("a.mp4", 4.0, 12.0), ("a.mp4", 19.0, 27.0)]
    out, _ = ST.top_up_to_target(cs, start, 30.0, avoid=banned)
    for s in out[1:]:
        for _, a, b in banned:
            assert s.end <= a or s.start >= b, (s.start, s.end)


def test_the_director_reports_whether_the_rejections_were_honoured():
    clip = busy()
    cs = ClipAnalysisSet(clips=[clip], orientation="horizontal",
                         canvas=(1280, 720), target_seconds=6.0)
    c = {"replace_windows": [{"clip_id": "a.mp4", "start": 0.0, "end": 5.0}]}
    _, applied, unmet = EditingDirectorAgent()._apply_constraints(
        Span(), cs, [seg(1, start=20.0, dur=5.0)], c)
    assert any("avoided" in x["chose"] for x in applied) and not unmet
    _, applied, unmet = EditingDirectorAgent()._apply_constraints(
        Span(), cs, [seg(1, start=2.0, dur=5.0)], c)
    assert any("re-used" in x["chose"] for x in unmet)


# ------------------------------------------------------------- end to end

CALM, LIVELY = "01_calm.mp4", "02_lively.mp4"


def reviewing_llm(plan_segments):
    """Plans like `llm_returning_plan`; as the reviewer, calls the LAST shot
    of the first render weak and every shot of later renders fine."""
    reviews = []
    plan_text = json.dumps({"strategy": "s", "opening_rationale": "r",
                            "segments": plan_segments, "rejected": []})

    def transport(prompt, max_tokens, images=None):
        if "reviewing a finished short video" in prompt:
            n = len(re.findall(r"^\d+\. caption:", prompt, re.M))
            reviews.append(n)
            scores = [9] * n
            if len(reviews) == 1:
                scores[-1] = 1
            text = json.dumps({"shots": [{"score": s, "caption_matches": None,
                                          "problem": "" if s > 4 else "a hand"}
                                         for s in scores]})
        else:
            text = plan_text
        return LLMReply(text=text, tokens_in=100, tokens_out=50, usd=0.0,
                        meta={"provider": "injected"})

    client = LLMClient(transport=transport)
    client.reviews = reviews
    return client


def test_a_weak_shot_is_recut_from_another_moment_and_then_passes(
        agent_root, calm_clip, lively_clip, pool, monkeypatch):
    from agents.runner import make_job, run_job
    from agents.store import store_for_build_dir

    monkeypatch.setenv("AT_VISUAL_REVIEW", "1")
    src = agent_root / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calm_clip, src / CALM)
    shutil.copy2(lively_clip, src / LIVELY)
    job = make_job(src=src, job_id="gvisual", root=agent_root,
                   settings=JobSettings(orientation="horizontal",
                                        resolution="720p", quality="draft",
                                        target_seconds=5.0),
                   max_revisions=2)
    llm = reviewing_llm(
        [{"clip_id": LIVELY, "target_seconds": 2.5, "reason": "open strong"},
         {"clip_id": CALM, "target_seconds": 2.5, "reason": "the detail"}])

    final = run_job(job, llm=llm, pool=pool)

    assert len(llm.reviews) == 2, "every render is reviewed"
    assert [h["verdict"] for h in final["history"]] == ["REVISE", "PASS"]
    assert final["status"] == "done"

    store = store_for_build_dir(job.build_dir)
    reviews = store.all_revisions(job.job_id, "VisualReview", VisualReview)
    assert [r.payload.attempt for r in reviews] == [1, 2]
    weak = [x for x in reviews[0].payload.shots
            if x.index in reviews[0].payload.weak]
    assert len(weak) == 1

    critics = store.all_revisions(job.job_id, "CriticResult", CriticResult)
    assert any(c.dimension == "shots_show_subject"
               for c in critics[0].payload.creative)

    plans = store.all_revisions(job.job_id, "EditPlan", EditPlan)
    assert "replace_windows" in plans[1].payload.constraints_applied
    w = weak[0]
    again = [s for s in plans[1].payload.segments
             if s.clip_id == w.clip_id and s.start < w.end and w.start < s.end]
    honest = [d for d in plans[1].decisions
              if d.what == "Revision constraint UNMET" and "re-used" in d.chose]
    assert not again or honest, "a rejected moment came back unreported"
