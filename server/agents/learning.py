"""
Learning Agent - what the system has learned from how its videos performed,
and an explicit "insufficient data" for as long as there is nothing to learn
from.

WHERE IT SITS
-------------
    ... publishing -> performance_analysis -> learning -> END

The last step. It runs ONCE, after the performance report exists, so it can
never cost a render, trigger a revision or change any deliverable.

WHAT THE ARCHITECTURE ASKS OF IT
--------------------------------
Join OUTCOMES - measured performance of published videos - with the
production DECISIONS behind them, and turn what survives the evidence bar
into a CONFIGURATION proposal for a person to approve. Never code, never
applied by itself, never a causal claim. The bar is a gate, and every part
of it must hold:

    sufficient_history   enough published videos with measured performance
    valid_nonzero_data   every figure measured after publication, non-zero
    cohort_size          >= 12 videos in each cohort compared
    time_window          samples spanning >= 30 days, metrics mature (T+7d)
    holdout_validation   the pattern must hold on the most recent videos,
                         held out
    practical_effect     a difference worth acting on, its confidence
                         interval clear of zero
    human_approval       a person approves before anything changes

WHAT EXISTS
-----------
No outcome at all. Nothing publishes, and `PerformanceAnalysis.status` has
exactly one member, "unavailable". This agent reads that from the store -
this job's report and every other job's - and the gate fails at its first
check. The checks that need a pattern to test are not evaluated, because
there is none, and none is invented to test.

No gate can PASS here. Passing one takes measured outcomes, and evaluating
them is written when they exist - against their real fields, not guessed
now. Until then: nothing learned, nothing proposed, configuration unchanged.
The architecture calls that a success, and so does the artifact.

The architecture classes Learning as statistics-led, with a model only ever
phrasing a hypothesis the statistics already support. With no statistics
there is nothing to phrase, so no model is called: this is deterministic.
"""

from __future__ import annotations

import inspect
import time

from .artifacts import (Artifact, ConfigTarget, ContentStrategy, Cost,
                        CriticResult, Decision, EditPlan, JobSettings,
                        Learning, LearningGate, LearningHistory,
                        LearningInput, PerformanceAnalysis, Publishing)
from .base import Agent, AgentContext
from .trace import TraceSpan

MAX_STEPS = 6

# THE GATE'S THRESHOLDS. Two come from the architecture verbatim (12 per
# cohort, a 30-day range) and one from its snapshot schedule (metrics are
# mature at T+7d). The rest are stated POLICY: numbers chosen here, named
# here, and reported as policy on every gate that uses them. None of them is
# a finding.
MIN_COHORT = 12            # architecture: n >= 12 videos per cohort
MIN_COHORTS = 2            # a comparison needs two sides
MIN_WINDOW_DAYS = 30       # architecture: date_range >= 30 days
MIN_METRIC_AGE_DAYS = 7    # architecture: the T+7d snapshot
HOLDOUT_VIDEOS = 6         # policy: the most recent videos, held out
MIN_HISTORY = MIN_COHORT * MIN_COHORTS + HOLDOUT_VIDEOS   # policy: 30
MIN_EFFECT = 0.10          # policy: 10% of the baseline, CI clear of zero

THRESHOLDS = {"min_cohort": MIN_COHORT, "min_cohorts": MIN_COHORTS,
              "min_window_days": MIN_WINDOW_DAYS,
              "min_metric_age_days": MIN_METRIC_AGE_DAYS,
              "holdout_videos": HOLDOUT_VIDEOS, "min_history": MIN_HISTORY,
              "min_effect": MIN_EFFECT}


# ------------------------------------------------------- the real project

def _history(store, job_id: str) -> LearningHistory:
    """Every job in the shared store, and what its performance report says.

    The job being analysed is always part of its own history - counted
    whether or not it has written anything yet, so the count (and the cache
    key built on it) does not shift when this agent's own output lands.

    A report is a usable sample only if it carries measured, NON-ZERO
    performance. Both halves matter. "Measured" rules out the reports
    that say they know nothing - the only kind that existed until an
    operator could record an upload and drop a warehouse export beside
    it. "Non-zero" rules out a video that was published and genuinely
    got nothing: a row of zeroes is a true fact about that video and a
    useless one to compare against, and counting it would let thirty
    unwatched uploads open a gate that is there to keep conclusions
    honest.

    A report that is measured but all zero is counted separately rather
    than called unreadable: it was read perfectly well."""
    jobs = sorted(set(store.jobs()) | {job_id})
    reports = unavailable = unreadable = zero = usable = 0
    window = 0
    for job_id in jobs:
        try:
            art = store.get(job_id, "PerformanceAnalysis", PerformanceAnalysis)
        except Exception:                             # noqa: BLE001
            unreadable += 1
            continue
        if art is None:
            continue
        reports += 1
        payload = art.payload
        if payload.status == "unavailable":
            unavailable += 1
        elif any(m.value > 0 for m in payload.measured):
            usable += 1
            window = max(window, payload.window_days)
        else:
            zero += 1
    return LearningHistory(
        jobs_in_store=len(jobs), performance_reports=reports,
        unavailable_reports=unavailable, unreadable_reports=unreadable,
        zero_reports=zero, window_days=window,
        usable_samples=usable, samples_needed=max(0, MIN_HISTORY - usable))


def _fmt(v) -> str:
    if v is None:
        return "unset"
    return f"{v:g}" if isinstance(v, float) else str(v)


def _configurable() -> list[ConfigTarget]:
    """The configuration a proposal could ever name, read from the code
    that consumes it - an operator setting or a run parameter. A module
    constant is not on this list: changing it is changing code."""
    from . import runner          # lazy: the runner imports this module
    run = inspect.signature(runner.run_job).parameters
    make = inspect.signature(runner.make_job).parameters
    fields = JobSettings.model_fields
    out = []
    for key, where, src in (
            ("JobSettings.target_seconds", "an operator setting, per job",
             fields.get("target_seconds")),
            ("JobSettings.transition_duration",
             "an operator setting, per job", fields.get("transition_duration")),
            ("run_job.pass_threshold",
             "the critic's pass bar, per run (runner --threshold)",
             run.get("pass_threshold")),
            ("make_job.max_revisions",
             "the revision cap, per job (runner --max-revisions)",
             make.get("max_revisions"))):
        if src is not None:
            out.append(ConfigTarget(key=key, where=where,
                                    current=_fmt(src.default)))
    return out


def _gates(h: LearningHistory) -> list[LearningGate]:
    """The architecture's gate, against what the store actually holds.

    Each data gate either FAILS on a count that is measurably short, or is
    NOT EVALUATED. Nothing passes: see the module docstring."""
    u = h.usable_samples

    def short(ok: bool) -> str:
        return "not_evaluated" if ok else "failed"

    if u:
        none_measured = (f"{u} report(s) carry a measured, non-zero figure; "
                         f"{h.unavailable_reports} unavailable, "
                         f"{h.zero_reports} measured but all zero, "
                         f"{h.unreadable_reports} unreadable")
    else:
        none_measured = (f"{h.unavailable_reports} report(s) unavailable, "
                         f"{h.zero_reports} measured but all zero and "
                         f"{h.unreadable_reports} unreadable - not one "
                         f"carries a measured, non-zero figure")
    return [
        LearningGate(
            gate="sufficient_history", state=short(u >= MIN_HISTORY),
            required=(f"at least {MIN_HISTORY} published videos with "
                      f"measured performance ({MIN_COHORTS} cohorts of "
                      f"{MIN_COHORT} plus a holdout of {HOLDOUT_VIDEOS})"),
            observed=(f"{u} usable, of {h.performance_reports} performance "
                      f"report(s) across {h.jobs_in_store} job(s) in the "
                      f"store - {h.samples_needed} short"),
            threshold_from="policy"),
        LearningGate(
            gate="valid_nonzero_data", state=short(u > 0),
            required=("every figure measured after publication and non-zero "
                      "- never estimated, never taken before publication"),
            observed=none_measured, threshold_from="architecture"),
        LearningGate(
            gate="cohort_size", state=short(u >= MIN_COHORT * MIN_COHORTS),
            required=(f"at least {MIN_COHORT} videos in each of the "
                      f"{MIN_COHORTS} cohorts compared"),
            observed=f"{u} usable sample(s) - no cohort can be formed",
            threshold_from="architecture"),
        LearningGate(
            gate="time_window",
            state=short(u > 0 and h.window_days >= MIN_WINDOW_DAYS),
            required=(f"samples spanning at least {MIN_WINDOW_DAYS} days, "
                      f"each read at least {MIN_METRIC_AGE_DAYS} days after "
                      f"publication"),
            # A span is read from the export's own MIN(date)/MAX(date). A
            # warehouse total summed over all time carries none, and a
            # missing span is reported as missing rather than assumed to
            # be wide enough.
            observed=("no sample has a publication date - nothing was "
                      "published" if not u else
                      f"the widest export covers {h.window_days} day(s)"
                      + ("" if h.window_days else
                         "; the export carries no MIN(date)/MAX(date), so "
                         "no span can be read from it")),
            threshold_from="architecture"),
        LearningGate(
            gate="holdout_validation", state="not_evaluated",
            required=(f"a pattern found on the older videos must also hold "
                      f"on the {HOLDOUT_VIDEOS} most recent, held out"),
            observed="no pattern exists to validate, and none is invented "
                     "to test",
            threshold_from="policy"),
        LearningGate(
            gate="practical_effect", state="not_evaluated",
            required=(f"a difference of at least {MIN_EFFECT:.0%} of the "
                      f"baseline, with a bootstrap confidence interval that "
                      f"excludes zero"),
            observed="no effect was measured, so none is claimed",
            threshold_from="policy"),
        LearningGate(
            gate="human_approval", state="not_evaluated",
            required=("an operator approves a proposal before any "
                      "configuration changes - nothing applies itself"),
            observed="no proposal exists to approve, and approval is never "
                     "assumed",
            threshold_from="architecture"),
    ]


def _inputs(performance: PerformanceAnalysis | None,
            strategy: ContentStrategy | None, plan: EditPlan | None,
            critic: CriticResult | None,
            publishing: Publishing | None) -> list[LearningInput]:
    """What each artifact read IS to learning. Categorical facts only - not
    one figure is copied out of any of them."""
    out = []
    if performance is not None:
        out.append(LearningInput(
            kind="PerformanceAnalysis", role="outcome",
            summary=(f"status {performance.status}: "
                     + ", ".join(performance.reasons)),
            why=("the only outcome there could be - and it is unavailable, "
                 "so there is nothing to learn from")))
    if strategy is not None:
        out.append(LearningInput(
            kind="ContentStrategy", role="decision",
            summary=(f"strategy {strategy.strategy_type}, pacing "
                     f"{strategy.pacing}"
                     + (f", hook clip {strategy.hook.clip_id}"
                        if strategy.hook.clip_id else "")),
            why=("a production decision - one side of a comparison, never "
                 "an outcome; with no outcome to set it against it teaches "
                 "nothing")))
    if plan is not None:
        out.append(LearningInput(
            kind="EditPlan", role="decision",
            summary=f"opens on {plan.opening_clip}",
            why="the cut that shipped - a decision, never an outcome"))
    if critic is not None:
        out.append(LearningInput(
            kind="CriticResult", role="pre_publication",
            summary=f"verdict {critic.verdict}",
            why=("measured on the file before anyone watched it - a check, "
                 "never an outcome, so it is not learned from")))
    if publishing is not None:
        out.append(LearningInput(
            kind="Publishing", role="pre_publication",
            summary=(f"plan {publishing.status}, published "
                     f"{publishing.published}"),
            why=("a plan for the operator with published=False - there is "
                 "no publication behind it")))
    return out


class LearningAgent(Agent):
    id = "learning"
    produces = "Learning"
    uses_llm = False
    agency = "deterministic"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                performance: PerformanceAnalysis | None = None,
                strategy: ContentStrategy | None = None,
                plan: EditPlan | None = None,
                critic: CriticResult | None = None,
                publishing: Publishing | None = None,
                revision: int = 0) -> Artifact[Learning]:
        started = time.time()

        # ---- what there is to learn from, read now ------------------------
        history = _history(ctx.store, ctx.job.job_id)
        span.tool("learning.read_history", jobs=history.jobs_in_store,
                  reports=history.performance_reports,
                  usable=history.usable_samples)
        configurable = _configurable()
        span.tool("learning.read_configuration",
                  keys=[t.key for t in configurable])
        inputs = _inputs(performance, strategy, plan, critic, publishing)

        inputs_hash = self.hash_inputs(
            ctx.job.job_id, history.model_dump(),
            [i.model_dump() for i in inputs],
            [t.model_dump() for t in configurable], THRESHOLDS)
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Learning",
                                         inputs_hash, Learning)
        if cached is not None:
            span.cache(1)
            span.note("reusing the learning report: the history, this job's "
                      "inputs and the configuration are unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        gates = _gates(history)
        span.tool("learning.evaluate_gates",
                  failed=[g.gate for g in gates if g.state == "failed"],
                  not_evaluated=[g.gate for g in gates
                                 if g.state == "not_evaluated"])

        payload = Learning(
            performance_status=performance.status if performance else None,
            history=history, gates=gates, inputs=inputs,
            configurable=configurable,
            next_steps=self._next_steps(performance, history, configurable),
            limitations=self._limitations(configurable))

        why = (f"insufficient data: {history.usable_samples} usable "
               f"sample(s) of {MIN_HISTORY} needed; "
               f"{len(payload.failed_gates)} gate(s) failed")
        span.fallback(f"no learning ({why})")
        ctx.emit(f"learning: nothing learned - {why}; configuration "
                 f"unchanged, nothing proposed")
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, performance),
            confidence=0.0, degraded=True,
            fallback_used=f"no learning ({why})",
            cost=Cost(wall_s=round(time.time() - started, 2)))

    # ------------------------------------------------------- next steps

    @staticmethod
    def _next_steps(performance: PerformanceAnalysis | None,
                    h: LearningHistory,
                    configurable: list[ConfigTarget]) -> list[str]:
        steps = []
        if performance is None:
            steps.append("Make the performance report first - this job has "
                         "none, so there is not even an unavailable outcome "
                         "on record")
        steps.append("Nothing to approve: no proposal was made, and none "
                     "will be until the gate holds")
        steps.append(f"Learning needs at least {MIN_HISTORY} published "
                     f"videos with measured, non-zero performance, read at "
                     f"least {MIN_METRIC_AGE_DAYS} days after publication "
                     f"and spanning {MIN_WINDOW_DAYS}+ days - "
                     f"{h.usable_samples} exist")
        steps.append("Both missing links come first: publishing (see the "
                     "publishing plan) and reading performance (see the "
                     "performance report) - neither exists in this project")
        if configurable:
            steps.append("Configuration stays as it is ("
                         + ", ".join(t.key for t in configurable)
                         + ") - any future change is proposed for your "
                           "approval, never applied")
        return [s[:300] for s in steps][:MAX_STEPS]

    @staticmethod
    def _limitations(configurable: list[ConfigTarget]) -> list[str]:
        return [
            "nothing was learned: no video has a measured outcome, so the "
            "data gates fail and the rest cannot be evaluated",
            f"the gate's numbers are thresholds, not findings: {MIN_COHORT} "
            f"per cohort and a {MIN_WINDOW_DAYS}-day range from the "
            f"architecture, metrics {MIN_METRIC_AGE_DAYS} days old from its "
            f"snapshot schedule, and a {MIN_HISTORY}-video history, a "
            f"{HOLDOUT_VIDEOS}-video holdout and a {MIN_EFFECT:.0%} effect "
            f"floor as stated policy",
            "no versioned strategy profile exists to propose changes to; "
            "the configuration a proposal could name today is "
            + (", ".join(t.key for t in configurable) or "nothing"),
            "any future insight would be correlational and say so - a "
            "pattern across videos is not a cause",
        ]

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Learning,
                   performance: PerformanceAnalysis | None) -> list[Decision]:
        h = p.history
        out = [Decision(
            what="Outcome data",
            chose=(f"none usable - {h.usable_samples} of "
                   f"{h.performance_reports} performance report(s) across "
                   f"{h.jobs_in_store} job(s)"),
            why=((f"this job's performance is {performance.status} ("
                  + ", ".join(performance.reasons) + ")")
                 if performance else "this job has no performance report")
                + "; a performance report is the only outcome there is, and "
                  "every one in the store is unavailable - read from the "
                  "store on this run, not assumed",
            decided_by="measurement", confidence=1.0)]

        out.append(Decision(
            what="Learning gate",
            chose=(f"closed - {len(p.failed_gates)} failed, "
                   f"{len(p.unevaluated_gates)} not evaluated"),
            why=("; ".join(f"{g.gate}: {g.observed}" for g in p.gates
                           if g.state == "failed")[:300]),
            decided_by="policy", confidence=1.0))

        out.append(Decision(
            what="What was learned",
            chose="nothing - no insight, no proposal",
            why=("insufficient data; returning nothing is the correct result "
                 "- a pattern drawn from no outcomes would be invented, and "
                 "even with outcomes a correlation is not a cause"),
            decided_by="fallback", confidence=1.0))

        pre = [i.kind for i in p.inputs if i.role == "pre_publication"]
        out.append(Decision(
            what="Pre-publication data",
            chose=("not learned from: " + ", ".join(pre)) if pre
            else "none present",
            why=("each was produced before anyone could watch the video - "
                 "a check or a plan, never an outcome"),
            decided_by="policy", confidence=1.0))

        out.append(Decision(
            what="Configuration",
            chose="unchanged - nothing proposed, nothing applied",
            why=("Learning may only ever PROPOSE a configuration change ("
                 + (", ".join(t.key for t in p.configurable) or "none")
                 + "), for an operator to approve; it has no path to code "
                   "and applies nothing itself"),
            decided_by="policy", confidence=1.0))
        return out


__all__ = ["LearningAgent"]
