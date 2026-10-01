"""
Learning Agent.

Deterministic - there is no model to inject, and no process or connection to
make. The properties most of these defend:

  * nothing can be learned: no performance report can be a usable sample,
    the agent reads that from the whole shared store, and parity tests fail
    the moment it changes
  * the architecture's gate - history, valid non-zero data, cohort size,
    time window, holdout, practical effect, human approval - is reported
    gate by gate, with its thresholds and where they come from, and no code
    path in the agent can pass a gate
  * nothing it produces can carry an insight, a correlation, an effect, a
    causal claim, an experiment or a proposal, and `learned` and
    `configuration_changed` are the literal False
  * a pre-publication signal can never be taken for an outcome, and not one
    of its figures reaches the report
  * configuration only, never code: the configurable surface is read from
    the code that consumes it, and running the agent changes no file
"""

import json
import socket
import sqlite3
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.artifacts import (Artifact, ConfigTarget, ContentStrategy,
                              CreativeCheck, CriticResult, EditPlan,
                              GateState, Hook, JobSettings, Learning,
                              LearningGate, LearningGateName, LearningHistory,
                              LearningInput, LearningStatus,
                              PerformanceAnalysis, PerformanceGap,
                              PerformanceMetric, PerformanceStatus,
                              PublishAsset, PublishCheck, Publishing, Segment,
                              UnmeasuredMetric)
from agents.learning import (HOLDOUT_VIDEOS, MIN_COHORT, MIN_EFFECT,
                             MIN_HISTORY, MIN_METRIC_AGE_DAYS,
                             MIN_WINDOW_DAYS, LearningAgent, _configurable,
                             _gates, _history)
from agents.publishing import VIDEO_ON_DISK, _code

GATES = typing.get_args(LearningGateName)
ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------- helpers

@pytest.fixture
def learn_ctx(director_ctx):
    """A real store and tracer, private to the test."""
    ctx, _cs = director_ctx
    return ctx


def a_performance(plan="ready") -> PerformanceAnalysis:
    """A performance report as that agent leaves it - validated by the real
    schema, so it can only be a state that agent can produce."""
    gaps = ([PerformanceGap(reason="not_published", evidence="x")]
            if plan else
            [PerformanceGap(reason="no_publishing_plan", evidence="x")])
    gaps += [PerformanceGap(reason="no_video_reference", evidence="x"),
             PerformanceGap(reason="no_analytics_provider", evidence="x")]
    return PerformanceAnalysis(
        publishing_status=plan, gaps=gaps,
        unmeasured=[UnmeasuredMetric(metric=m, needs="x")
                    for m in typing.get_args(PerformanceMetric)])


def a_strategy() -> ContentStrategy:
    return ContentStrategy(domain="LEGO Technic", strategy_type="minimal",
                           pacing="moderate",
                           hook=Hook(clip_id="02_lively.mp4"))


def an_edit() -> EditPlan:
    return EditPlan(
        segments=[Segment(index=1, clip_id="02_lively.mp4", start=1.0,
                          end=6.0, duration=5.0, score=8.8,
                          orientation="horizontal"),
                  Segment(index=2, clip_id="01_calm.mp4", start=0.0, end=5.0,
                          duration=5.0, score=1.1,
                          orientation="horizontal")],
        total_duration=10.0, target_seconds=10.0, canvas=(1280, 720))


def a_critic(score=97.3) -> CriticResult:
    return CriticResult(
        attempt=1, verdict="PASS", overall_score=score, pass_threshold=65.0,
        creative=[CreativeCheck(dimension="hook", score=score,
                                evidence="first 3s measures 9.1 mean YDIF",
                                confidence=1.0)],
        summary=f"hook scores {score} against a 65 threshold")


def a_plan() -> Publishing:
    video = PublishAsset(kind="video", path="/b/v.mp4", file_name="v.mp4",
                         size=13_000, width=1920, height=1080, codec="h264",
                         duration=23.5)
    thumb = PublishAsset(kind="thumbnail", path="/b/t.jpg",
                         file_name="t.jpg", size=60_000, width=1920,
                         height=1080, codec="mjpeg")
    return Publishing(status="ready", video=video, thumbnail=thumb,
                      thumbnail_from="thumbnail_agent", title="LEGO Gears",
                      title_from="title_agent",
                      checks=[PublishCheck(check=VIDEO_ON_DISK, passed=True,
                                           blocking=True)])


def run(ctx, tracer_span, performance="default", strategy="default",
        plan="default", critic="default", publishing="default", revision=0):
    pick = {"performance": a_performance, "strategy": a_strategy,
            "plan": an_edit, "critic": a_critic, "publishing": a_plan}
    given = {"performance": performance, "strategy": strategy, "plan": plan,
             "critic": critic, "publishing": publishing}
    kw = {k: pick[k]() if v == "default" else v for k, v in given.items()}
    with tracer_span(ctx, "learning") as span:
        return LearningAgent().execute(ctx, span, revision=revision, **kw)


def other_job(ctx, job_id, payload=None):
    """Another job's performance report in the same shared store."""
    ctx.store.put(Artifact[PerformanceAnalysis](
        job_id=job_id, kind="PerformanceAnalysis",
        produced_by="performance_analysis", inputs_hash="h",
        payload=payload or a_performance()))


def gate(p: Learning, name: str) -> LearningGate:
    return next(g for g in p.gates if g.gate == name)


def every_gate(**states) -> list[LearningGate]:
    return [LearningGate(gate=g, state=states.get(g, "failed"),
                         required="x", observed="x") for g in GATES]


def history(**kw) -> LearningHistory:
    base = dict(jobs_in_store=1, performance_reports=1,
                unavailable_reports=1, unreadable_reports=0,
                usable_samples=0, samples_needed=MIN_HISTORY)
    return LearningHistory(**{**base, **kw})


# -------------------- 1. what there is to learn from, in the real project

def test_no_performance_report_can_be_a_usable_sample():
    assert typing.get_args(PerformanceStatus) == ("unavailable",), \
        "performance can now be measured - write the sample rule against it"


def test_the_schema_has_one_honest_state():
    assert typing.get_args(LearningStatus) == ("insufficient_data",)
    for field in ("learned", "configuration_changed"):
        assert typing.get_args(
            Learning.model_fields[field].annotation) == (False,)


def test_the_gate_is_the_architectures():
    assert GATES == ("sufficient_history", "valid_nonzero_data",
                     "cohort_size", "time_window", "holdout_validation",
                     "practical_effect", "human_approval")
    assert typing.get_args(GateState) == ("passed", "failed",
                                          "not_evaluated")
    # the architecture's own numbers
    assert MIN_COHORT == 12 and MIN_WINDOW_DAYS == 30
    assert MIN_METRIC_AGE_DAYS == 7
    # the policy ones, consistent with each other
    assert MIN_HISTORY == 2 * MIN_COHORT + HOLDOUT_VIDEOS
    assert 0 < MIN_EFFECT < 1


def test_the_configurable_surface_is_read_from_the_code():
    import inspect

    from agents import runner
    from agents.critic import DEFAULT_PASS_THRESHOLD
    got = {t.key: t.current for t in _configurable()}
    assert got == {
        "JobSettings.target_seconds": "unset",
        "JobSettings.transition_duration":
            f"{JobSettings.model_fields['transition_duration'].default:g}",
        "run_job.pass_threshold": f"{DEFAULT_PASS_THRESHOLD:g}",
        "make_job.max_revisions": str(inspect.signature(
            runner.make_job).parameters["max_revisions"].default)}


def test_code_is_never_a_configuration_target():
    keys = [t.key for t in _configurable()]
    assert not [k for k in keys if k.endswith(".py") or "/" in k]
    for constant in ("HOOK_SLOPE", "DEFAULT_PASS_THRESHOLD",
                     "HOOK_OPENING_CAP_S", "MIN_COHORT"):
        assert not [k for k in keys if constant in k]


def test_the_agent_has_no_path_to_write_or_run_anything():
    code = _code((ROOT / "server/agents/learning.py").read_text())
    for call in ("open(", "write_text", "write_bytes", "unlink", "rmtree",
                 "os.replace", "os.rename", "os.remove", "subprocess",
                 "exec(", "eval(", "importlib", "setattr(", "run_process",
                 "complete_json", "urlopen", "requests"):
        assert call not in code, call


def test_running_it_changes_no_file(learn_ctx, tracer_span):
    def snapshot():
        out = {}
        for base in ("server", "pipeline", "frontend"):
            for f in (ROOT / base).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    st = f.stat()
                    out[str(f)] = (st.st_size, st.st_mtime_ns)
        build = Path(learn_ctx.job.build_dir)
        for f in build.rglob("*"):
            if f.is_file():
                out[str(f)] = (f.stat().st_size, f.stat().st_mtime_ns)
        return out
    before = snapshot()
    run(learn_ctx, tracer_span)
    assert snapshot() == before


# --------------------------- 2. unavailable performance: nothing learned

def test_unavailable_performance_learns_nothing(learn_ctx, tracer_span):
    art = run(learn_ctx, tracer_span)
    p = art.payload
    assert p.status == "insufficient_data"
    assert p.learned is False and p.configuration_changed is False
    assert p.performance_status == "unavailable"
    assert p.history.usable_samples == 0
    assert p.history.samples_needed == MIN_HISTORY
    assert p.failed_gates == ["sufficient_history", "valid_nonzero_data",
                              "cohort_size", "time_window"]
    assert p.unevaluated_gates == ["holdout_validation", "practical_effect",
                                   "human_approval"]
    steps = " ".join(p.next_steps)
    assert "Nothing to approve: no proposal was made" in steps
    assert "never applied" in steps


def test_insufficient_data_is_a_declared_fallback(learn_ctx, tracer_span):
    art = run(learn_ctx, tracer_span)
    assert art.degraded is True and art.confidence == 0.0
    assert art.fallback_used.startswith("no learning (insufficient data: "
                                        "0 usable sample(s) of 30")
    assert art.produced_by == "learning" and art.kind == "Learning"
    assert art.inputs_hash


def test_no_performance_report_still_learns_nothing(learn_ctx, tracer_span):
    p = run(learn_ctx, tracer_span, performance=None).payload
    assert p.status == "insufficient_data" and p.performance_status is None
    assert "PerformanceAnalysis" not in [i.kind for i in p.inputs]
    assert p.next_steps[0].startswith("Make the performance report first")


def test_nothing_at_all_still_learns_nothing(learn_ctx, tracer_span):
    p = run(learn_ctx, tracer_span, performance=None, strategy=None,
            plan=None, critic=None, publishing=None).payload
    assert p.status == "insufficient_data" and p.inputs == []


# ------------------------------------- 3. the history is the whole store

def test_the_history_counts_every_job_in_the_store(learn_ctx, tracer_span):
    for i in range(3):
        other_job(learn_ctx, f"other{i}")
    learn_ctx.store.put(Artifact[ContentStrategy](
        job_id="noreport", kind="ContentStrategy", produced_by="x",
        inputs_hash="h", payload=a_strategy()))
    h = run(learn_ctx, tracer_span).payload.history
    # three others with reports, one without, and this job itself
    assert (h.jobs_in_store, h.performance_reports, h.unavailable_reports,
            h.usable_samples) == (5, 3, 3, 0)


def test_this_job_is_always_part_of_its_own_history(learn_ctx, tracer_span):
    """Counted before it has written anything, and the same after - so the
    agent's own output never shifts the count or the cache key."""
    a = run(learn_ctx, tracer_span).payload.history
    assert a.jobs_in_store == 1
    assert learn_ctx.store.has_job(learn_ctx.job.job_id)
    assert _history(learn_ctx.store, learn_ctx.job.job_id) == a


def test_a_mountain_of_unavailable_reports_is_still_no_data(learn_ctx,
                                                            tracer_span):
    """Volume is not evidence: 45 reports of nothing are nothing."""
    for i in range(45):
        other_job(learn_ctx, f"bulk{i:02d}")
    p = run(learn_ctx, tracer_span).payload
    assert p.history.performance_reports == 45
    assert p.history.usable_samples == 0
    assert p.status == "insufficient_data"
    assert gate(p, "sufficient_history").state == "failed"
    assert not [g for g in p.gates if g.state == "passed"]


def test_an_unreadable_report_is_counted_and_never_used(learn_ctx,
                                                       tracer_span):
    with sqlite3.connect(learn_ctx.store.db_path) as con:
        con.execute(
            "INSERT INTO artifacts (artifact_id, job_id, kind,"
            " schema_version, revision, produced_by, produced_at,"
            " inputs_hash, trace_id, payload, decisions, confidence,"
            " degraded, fallback_used, cost) VALUES"
            " ('art_bad', 'corrupt', 'PerformanceAnalysis', 1, 0, 'x', 0,"
            " 'h', '', '{\"views\": 9000}', '[]', 1, 0, NULL, '{}')")
    h = run(learn_ctx, tracer_span).payload.history
    assert h.unreadable_reports == 1 and h.usable_samples == 0


# ---------------------------------------------------------- 4. the gate

def test_every_gate_states_its_bar_and_where_it_comes_from(learn_ctx,
                                                          tracer_span):
    p = run(learn_ctx, tracer_span).payload
    assert [g.gate for g in p.gates] == list(GATES)
    src = {g.gate: g.threshold_from for g in p.gates}
    assert src == {"sufficient_history": "policy",
                   "valid_nonzero_data": "architecture",
                   "cohort_size": "architecture",
                   "time_window": "architecture",
                   "holdout_validation": "policy",
                   "practical_effect": "policy",
                   "human_approval": "architecture"}
    assert f"at least {MIN_HISTORY}" in gate(p, "sufficient_history").required
    assert f"at least {MIN_COHORT} videos" in gate(p, "cohort_size").required
    assert f"{MIN_WINDOW_DAYS} days" in gate(p, "time_window").required
    assert f"{HOLDOUT_VIDEOS} most recent" in \
        gate(p, "holdout_validation").required
    assert "10%" in gate(p, "practical_effect").required
    assert "excludes zero" in gate(p, "practical_effect").required
    assert "approves" in gate(p, "human_approval").required
    assert "30 short" in gate(p, "sufficient_history").observed


@pytest.mark.parametrize("usable", [0, 5, 29, 30, 500])
def test_no_code_path_passes_a_gate(usable):
    h = history(performance_reports=max(usable, 1), usable_samples=usable,
                unavailable_reports=0,
                samples_needed=max(0, MIN_HISTORY - usable))
    gates = _gates(h)
    assert not [g for g in gates if g.state == "passed"]
    if usable < MIN_HISTORY:
        assert gates[0].state == "failed"


def test_the_gate_cannot_be_misreported():
    h = history()
    ok = dict(history=h)
    Learning(gates=every_gate(holdout_validation="not_evaluated"), **ok)
    with pytest.raises(ValidationError):            # a gate left out
        Learning(gates=every_gate()[:-1], **ok)
    with pytest.raises(ValidationError):            # out of order
        Learning(gates=list(reversed(every_gate())), **ok)
    with pytest.raises(ValidationError):            # a gate twice
        Learning(gates=every_gate() + every_gate()[:1], **ok)
    with pytest.raises(ValidationError):            # passing after a failure
        Learning(gates=every_gate(cohort_size="passed"), **ok)
    with pytest.raises(ValidationError):            # every gate passed
        Learning(gates=every_gate(**{g: "passed" for g in GATES}), **ok)
    with pytest.raises(ValidationError):            # approval of nothing
        Learning(gates=every_gate(**{g: "passed" for g in GATES[:-1]},
                                  human_approval="passed"), **ok)
    with pytest.raises(ValidationError):            # usable without a report
        Learning(gates=every_gate(),
                 history=history(performance_reports=0, usable_samples=1))


# --------------------- 5. the schema cannot carry a finding or a change

@pytest.mark.parametrize("kw", [
    {"learned": True}, {"configuration_changed": True},
    {"status": "learned"}, {"status": "proposal_ready"},
    {"insights": [{"claim": "short hooks win"}]},
    {"insight": "short hooks win"}, {"correlation": 0.62},
    {"effect_size": 0.18}, {"confidence_interval": [0.1, 0.3]},
    {"causal_status": "causal"}, {"causal_claim": "the hook drives views"},
    {"experiment": {"arm_a": "fast", "arm_b": "slow"}},
    {"proposal": {"key": "run_job.pass_threshold", "to": 70}},
    {"proposals": []}, {"applied_changes": ["x"]},
    {"code_change": "critic.py"}, {"views": 1200},
])
def test_the_schema_cannot_carry_a_finding_or_a_change(kw):
    base = dict(history=history(), gates=every_gate())
    Learning(**base)                               # the base itself is valid
    with pytest.raises(ValidationError):
        Learning(**{**base, **kw})


@pytest.mark.parametrize("kind,role", [
    ("CriticResult", "outcome"), ("Publishing", "outcome"),
    ("Title", "outcome"), ("Thumbnail", "outcome"),
    ("ContentStrategy", "outcome"), ("EditPlan", "outcome"),
    ("PerformanceAnalysis", "decision"), ("CriticResult", "decision"),
    ("ContentStrategy", "pre_publication"),
])
def test_nothing_made_before_publication_can_be_an_outcome(kind, role):
    with pytest.raises(ValidationError):
        LearningInput(kind=kind, role=role)


def test_a_config_target_is_a_named_key_not_a_patch():
    with pytest.raises(ValidationError):
        ConfigTarget(key="x", where="y", current="z", new="w")


def test_the_agent_is_deterministic_by_design():
    assert LearningAgent.uses_llm is False
    assert LearningAgent.agency == "deterministic"


# ----------------- 6. pre-publication data is never learned from

def test_each_input_is_classified_by_what_it_is(learn_ctx, tracer_span):
    p = run(learn_ctx, tracer_span).payload
    roles = {i.kind: i.role for i in p.inputs}
    assert roles == {"PerformanceAnalysis": "outcome",
                     "ContentStrategy": "decision", "EditPlan": "decision",
                     "CriticResult": "pre_publication",
                     "Publishing": "pre_publication"}
    by = {i.kind: i for i in p.inputs}
    assert by["PerformanceAnalysis"].summary == (
        "status unavailable: not_published, no_video_reference, "
        "no_analytics_provider")
    assert by["ContentStrategy"].summary == (
        "strategy minimal, pacing moderate, hook clip 02_lively.mp4")
    assert by["EditPlan"].summary == "opens on 02_lively.mp4"
    assert by["CriticResult"].summary == "verdict PASS"
    assert "never an outcome" in by["CriticResult"].why


def test_not_one_pre_publication_figure_reaches_the_report(learn_ctx,
                                                          tracer_span):
    """The critic said 97.3 and the cut scored 8.8 and 1.1 - and the report,
    decisions and all, says none of it."""
    art = run(learn_ctx, tracer_span)
    text = art.payload.model_dump_json() + json.dumps(
        [d.model_dump() for d in art.decisions])
    for leaked in ("97.3", "9.1", "8.8", "1.1", "23.5", "13000", "60000"):
        assert leaked not in text, leaked


def test_a_critic_score_never_changes_the_outcome(learn_ctx, tracer_span,
                                                  monkeypatch):
    # made twice for real - the cache would make this trivially true
    monkeypatch.setattr(learn_ctx.store, "get_by_inputs",
                        lambda *a, **kw: None)
    lo = run(learn_ctx, tracer_span, critic=a_critic(12.0))
    hi = run(learn_ctx, tracer_span, critic=a_critic(99.9), revision=1)
    assert lo.artifact_id != hi.artifact_id
    assert lo.payload == hi.payload and lo.decisions == hi.decisions


# --------------------------- 7. no model, no process, no network

class _Refuse:
    def __init__(self, what):
        self.what = what

    def __getattr__(self, name):
        raise AssertionError(f"the agent touched the {self.what} ({name})")


def test_it_never_calls_a_model_or_starts_a_process(learn_ctx, tracer_span):
    learn_ctx.llm = _Refuse("LLM")
    learn_ctx.pool = _Refuse("media pool")
    assert run(learn_ctx, tracer_span).payload.status == "insufficient_data"


def test_it_never_touches_the_network(learn_ctx, tracer_span, monkeypatch):
    attempts = []

    def refuse(self, *a, **kw):
        attempts.append(a)
        raise AssertionError("the learning agent opened a connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    p = run(learn_ctx, tracer_span).payload
    assert attempts == [] and p.status == "insufficient_data"


# ---------------------------------------------------------- 8. caching

def test_identical_inputs_reuse_the_report(learn_ctx, tracer_span):
    a = run(learn_ctx, tracer_span)
    b = run(learn_ctx, tracer_span, revision=1)
    assert a.artifact_id == b.artifact_id
    rows = [r for r in learn_ctx.tracer.for_job(learn_ctx.job.job_id)
            if r["node"] == "learning"]
    assert rows[-1]["cache_hits"] == 1 and rows[-1]["status"] == "skipped"


@pytest.mark.parametrize("change", [
    {"performance": None}, {"performance": a_performance(plan=None)},
    {"strategy": None}, {"plan": None}, {"critic": None},
    {"publishing": None}])
def test_the_report_is_remade_when_an_input_changes(learn_ctx, tracer_span,
                                                    change):
    a = run(learn_ctx, tracer_span)
    b = run(learn_ctx, tracer_span, revision=1, **change)
    assert a.inputs_hash != b.inputs_hash


def test_a_grown_history_is_read_again(learn_ctx, tracer_span):
    a = run(learn_ctx, tracer_span)
    other_job(learn_ctx, "newcomer")
    b = run(learn_ctx, tracer_span, revision=1)
    assert a.inputs_hash != b.inputs_hash
    assert b.payload.history.performance_reports == 1


# --------------------------------------------- 9. trace and decisions

def test_the_trace_records_what_was_read_and_the_fallback(learn_ctx,
                                                          tracer_span):
    art = run(learn_ctx, tracer_span)
    rows = [r for r in learn_ctx.tracer.for_job(learn_ctx.job.job_id)
            if r["node"] == "learning"]
    tools = [t["tool"] for r in rows for t in r["tool_calls"]]
    assert tools == ["learning.read_history", "learning.read_configuration",
                     "learning.evaluate_gates"]
    gates = rows[-1]["tool_calls"][2]
    assert gates["failed"] == ["sufficient_history", "valid_nonzero_data",
                               "cohort_size", "time_window"]
    assert rows[-1]["fallback_used"] == art.fallback_used
    assert art.trace_id == rows[-1]["trace_id"]
    assert art.artifact_id in rows[-1]["output_artifacts"]
    assert rows[-1]["tokens_in"] == 0 and rows[-1]["tokens_out"] == 0


def test_decisions_are_measured_policy_or_the_declared_fallback(
        learn_ctx, tracer_span):
    art = run(learn_ctx, tracer_span)
    assert {d.decided_by for d in art.decisions} == {"measurement", "policy",
                                                    "fallback"}
    by = {d.what: d for d in art.decisions}
    assert by["Outcome data"].decided_by == "measurement"
    assert by["Outcome data"].chose.startswith("none usable")
    assert by["Learning gate"].chose == "closed - 4 failed, 3 not evaluated"
    assert by["What was learned"].decided_by == "fallback"
    assert by["What was learned"].chose == "nothing - no insight, no proposal"
    assert "not a cause" in by["What was learned"].why
    assert by["Pre-publication data"].chose == \
        "not learned from: CriticResult, Publishing"
    assert by["Configuration"].chose == \
        "unchanged - nothing proposed, nothing applied"
    assert "no path to code" in by["Configuration"].why
    assert all(d.confidence == 1.0 for d in art.decisions)


def test_the_limitations_say_thresholds_are_not_findings(learn_ctx,
                                                         tracer_span):
    lims = " ".join(run(learn_ctx, tracer_span).payload.limitations)
    assert "thresholds, not findings" in lims
    assert "correlational" in lims and "not a cause" in lims
    assert "no versioned strategy profile" in lims


# ------------------------------------------ 10. reported through the app

def test_the_projection_has_no_place_for_a_finding(learn_ctx, tracer_span):
    from agents.compat import _project_learning
    p = run(learn_ctx, tracer_span).payload
    out = _project_learning(p)
    assert out["status"] == "insufficient_data"
    assert out["learned"] is False and out["configuration_changed"] is False
    assert out["usable_samples"] == 0 and out["samples_needed"] == MIN_HISTORY
    assert list(out["gates"]) == list(GATES)
    assert out["gates"]["human_approval"] == "not_evaluated"
    assert not set(out) & {"insights", "proposal", "proposals",
                           "correlation", "effect_size", "experiment"}
    json.dumps(out)
