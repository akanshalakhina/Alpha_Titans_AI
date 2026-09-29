"""Typed artifacts: validation, immutability, revision-awareness, hashing."""

import pytest
from pydantic import ValidationError

from agents.artifacts import (Artifact, ClipAnalysisSet, Cost, Decision,
                              EditPlan, JobState, RevisionRequest, Segment,
                              content_hash)


def seg(index=1, clip_id="a.mp4", start=0.0, end=4.0, **kw):
    return Segment(index=index, clip_id=clip_id, start=start, end=end,
                   duration=round(end - start, 2),
                   orientation=kw.pop("orientation", "horizontal"), **kw)


# ------------------------------------------------------------- validation

def test_segment_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Segment(index=1, clip_id="a.mp4", start=5.0, end=2.0, duration=3.0,
                orientation="horizontal")


def test_segment_rejects_negative_start():
    with pytest.raises(ValidationError):
        Segment(index=1, clip_id="a.mp4", start=-1.0, end=4.0, duration=5.0,
                orientation="horizontal")


def test_segment_rejects_zero_duration():
    with pytest.raises(ValidationError):
        Segment(index=1, clip_id="a.mp4", start=1.0, end=1.0, duration=0.0,
                orientation="horizontal")


def test_edit_plan_requires_contiguous_indices():
    with pytest.raises(ValidationError) as e:
        EditPlan(segments=[seg(1), seg(3, "b.mp4")], total_duration=8.0,
                 target_seconds=20.0, canvas=(1920, 1080))
    assert "1..n" in str(e.value)


def test_edit_plan_rejects_empty():
    with pytest.raises(ValidationError):
        EditPlan(segments=[], total_duration=0.0, target_seconds=20.0,
                 canvas=(1920, 1080))


def test_decision_confidence_is_bounded():
    with pytest.raises(ValidationError):
        Decision(what="x", chose="y", why="z", decided_by="llm", confidence=1.4)


def test_decided_by_is_a_closed_set():
    with pytest.raises(ValidationError):
        Decision(what="x", chose="y", why="z", decided_by="vibes")


# ----------------------------------------------------------- immutability

def test_segment_is_frozen():
    s = seg()
    with pytest.raises(ValidationError):
        s.start = 9.0


def test_edit_plan_is_frozen():
    p = EditPlan(segments=[seg()], total_duration=4.0, target_seconds=20.0,
                 canvas=(1920, 1080))
    with pytest.raises(ValidationError):
        p.total_duration = 99.0


def test_artifact_envelope_is_frozen():
    art = Artifact[EditPlan](
        job_id="j1", kind="EditPlan", produced_by="editing_director",
        payload=EditPlan(segments=[seg()], total_duration=4.0,
                         target_seconds=20.0, canvas=(1920, 1080)))
    with pytest.raises(ValidationError):
        art.revision = 5


def test_model_copy_produces_a_new_object_not_a_mutation():
    """The revision mechanism: copy, never edit."""
    a = seg(start=0.0, end=4.0)
    b = a.model_copy(update={"start": 1.0, "end": 5.0})
    assert a.start == 0.0 and b.start == 1.0
    assert a is not b


# ----------------------------------------------------------------- hashes

def test_content_hash_is_order_independent_for_dict_keys():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_changes_with_content():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_segment_cache_key_tracks_the_pixels_it_describes():
    base = seg(start=1.0, end=5.0)
    assert base.cache_key() == seg(start=1.0, end=5.0).cache_key()
    # a different in-point is different pixels
    assert base.cache_key() != seg(start=2.0, end=6.0).cache_key()
    # a different caption is different pixels (it is burnt in)
    assert base.cache_key() != seg(start=1.0, end=5.0, caption="hi").cache_key()
    # the index alone is NOT part of the identity - the same shot moved to a
    # different slot is still the same render
    assert base.cache_key() == seg(index=7, start=1.0, end=5.0).cache_key()


def test_revision_request_constraint_key_detects_a_repeat():
    a = RevisionRequest(from_attempt=1, target_agent="editing_director",
                        issue="x", severity="high",
                        constraints={"opening_max_duration": 8.0})
    b = RevisionRequest(from_attempt=2, target_agent="editing_director",
                        issue="y", severity="high",
                        constraints={"opening_max_duration": 8.0})
    c = RevisionRequest(from_attempt=2, target_agent="editing_director",
                        issue="y", severity="high",
                        constraints={"opening_max_duration": 4.0})
    assert a.constraint_key() == b.constraint_key()
    assert a.constraint_key() != c.constraint_key()


# ---------------------------------------------------------------- shapes

def test_clip_analysis_exposes_the_shape_the_pipeline_expects():
    """to_planner_dict must satisfy build_video.best_window unchanged."""
    from agents.tools.pipeline_bridge import build_video
    from conftest import make_clip_analysis

    c = make_clip_analysis("x.mp4", duration=10.0, mean_ydif=4.0)
    d = c.to_planner_dict()
    for key in ("name", "duration", "sample_fps", "ydif", "mean_ydif",
                "freeze_runs", "orientation"):
        assert key in d, key
    a, b, score = build_video.best_window(d, 4.0)
    assert 0.0 <= a < b <= 10.0 and score >= 0.0


def test_render_spec_compiles_the_env_bridge():
    from agents.artifacts import RenderSpec, SegmentRenderSpec
    spec = RenderSpec(
        job_id="j", build_dir="/b", src_dir="/s", canvas=(1280, 720), crf=20,
        xfade_duration=0.25, xfade_list=["fade", "wipeup"],
        segments=[SegmentRenderSpec(index=1, clip_id="a.mp4", start=0, end=4,
                                    duration=4, caption="", cache_key="k",
                                    orientation="horizontal")])
    env = spec.env_overrides()
    assert env["AT_CANVAS"] == "1280x720"
    assert env["AT_CRF"] == "20"
    assert env["AT_XFADES"] == "fade,wipeup"
    assert env["AT_XFADE_T"] == "0.25"


def test_job_state_defaults_are_conservative():
    j = JobState(job_id="j", src_dir="/s", build_dir="/b")
    assert j.max_revisions == 2
    assert j.settings.force_opening_clip is None
