"""One bad minute must not fix a job's answer for ever.

WHAT HAPPENED
-------------
The Story agent took an HTTP 400 on one call, fell back to its
deterministic path and produced two beats instead of six. That artifact
was then cached against its inputs hash. Re-running the job did not
re-ask the model: it served the degraded story from cache, the Caption
agent saw `beats_eligible: [1, 2]`, and the finished video carried two
captions instead of six.

Nothing was broken, and nothing could be fixed by trying again. A
transient failure had become a permanent property of that job.

THE DISTINCTION THAT MATTERS
----------------------------
Most degradations SHOULD be cached. Research has no domain to research;
PerformanceAnalysis has no published video to measure. Those answers are
about this job and will not change, and re-asking would spend tokens to
be told the same thing.

A 400, a 429 or a timeout says nothing about the job. It says something
about that minute. Only those are refused from cache.

"the model reply was unusable" is deliberately NOT transient: a model
that answers badly on these inputs will answer badly again, and
re-asking would spend tokens to be disappointed twice.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import Artifact, Story                 # noqa: E402
from agents.store import ArtifactStore, is_transient         # noqa: E402


# =========================================== telling the two kinds apart

@pytest.mark.parametrize("fallback", [
    "deterministic story (the model call failed (HTTPError: HTTP Error 400: "
    "Bad Request))",
    "deterministic_planner (the model call failed (HTTPError: HTTP Error "
    "429: Too Many Requests))",
    "deterministic music (URLError: <urlopen error timed out>)",
    "deterministic title (TimeoutError)",
    "deterministic caption (HTTPError: 502 Bad Gateway)",
    "deterministic story (HTTPError: 429 (gave up after 150s of waiting))",
])
def test_a_provider_failure_is_transient(fallback):
    assert is_transient(fallback), fallback


@pytest.mark.parametrize("fallback", [
    "insufficient signal (no domain or topic was given for this job)",
    "no performance data (the video has not been published)",
    "no learning (insufficient data: 0 usable sample(s) of 30 needed)",
    "deterministic story (the model reply was unusable)",
    "deterministic_planner (assembled plan failed validation)",
    "no model configured",
    None,
    "",
])
def test_a_fact_about_this_job_is_not_transient(fallback):
    """These answers will not change on a second ask, and re-asking
    would spend tokens to hear them again."""
    assert not is_transient(fallback), fallback


def test_an_unusable_reply_is_not_treated_as_the_providers_fault():
    """The sharpest edge of the rule. A model that returns prose instead
    of JSON on these inputs will do it again; only the transport being
    broken earns a retry."""
    assert not is_transient("deterministic story (the model reply was "
                            "unusable)")
    assert is_transient("deterministic story (the model call failed "
                        "(HTTPError: HTTP Error 400: Bad Request))")


# ================================================ what the store serves

def _story(**kw):
    return Story(story_type=kw.pop("story_type", "reasoned"), beats=[])


def _put(store, job, fallback=None, degraded=False):
    art = Artifact[Story](
        job_id=job, kind="Story", produced_by="story",
        inputs_hash="same-inputs", payload=_story(),
        degraded=degraded, fallback_used=fallback)
    store.put(art)
    return art


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "state.db")


def test_a_good_artifact_is_served_from_cache(store):
    _put(store, "j1")
    assert store.get_by_inputs("j1", "Story", "same-inputs", Story)


def test_a_structural_degradation_is_served_from_cache(store):
    """Re-asking would spend tokens to be told the same thing."""
    _put(store, "j1", "insufficient signal (no domain was given)", True)
    assert store.get_by_inputs("j1", "Story", "same-inputs", Story)


def test_a_provider_failure_is_not_served_from_cache(store):
    """This is the bug: a 400 once, degraded for ever."""
    _put(store, "j1",
         "deterministic story (the model call failed (HTTPError: HTTP "
         "Error 400: Bad Request))", True)
    assert store.get_by_inputs("j1", "Story", "same-inputs", Story) is None, \
        "running again must re-ask the model, not reuse the bad minute"


def test_the_row_is_still_there_for_anyone_who_wants_it(store):
    """Refusing it from CACHE is not deleting it. The trace, the UI and
    anything reading history must still see what happened."""
    _put(store, "j1", "deterministic story (HTTPError: 400)", True)
    assert store.get("j1", "Story", Story) is not None
    assert store.get_by_inputs("j1", "Story", "same-inputs", Story,
                               reuse_transient=True) is not None


def test_a_later_good_run_is_cached_normally(store):
    """Once the model answers, that answer caches like any other."""
    _put(store, "j1", "deterministic story (HTTPError: 400)", True)
    store.put(Artifact[Story](
        job_id="j1", kind="Story", produced_by="story", revision=1,
        inputs_hash="same-inputs", payload=_story()))
    got = store.get_by_inputs("j1", "Story", "same-inputs", Story)
    assert got is not None and not got.degraded


def test_nothing_cached_is_still_nothing(store):
    assert store.get_by_inputs("j1", "Story", "never-seen", Story) is None
