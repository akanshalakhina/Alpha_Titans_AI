"""Fill an ISOLATED store with enough measured outcomes to open the gates.

WHY
---
The Learning agent has sat at 0 usable samples of 30 since it was written,
so none of the code that counts, cohorts or windows those samples has ever
run against real numbers. This puts 30 of them somewhere safe and runs the
agent's own `_history` and `_gates` over them, so the data path can be
proved today instead of in however many months it takes to publish thirty
videos.

WHAT IT PROVES, AND WHAT IT STILL CANNOT
----------------------------------------
With `--approve` all seven gates open and a finding is produced: the
cohort comparison, the holdout check and the effect size are real
arithmetic over these rows, and the result is what `content_strategy`
and `editing_director` read on the next job.

Two honest limits remain. The numbers are SYNTHETIC - every report says
so in its own `limitations` - so what is proved is that the machinery
computes and carries a finding correctly, not that fast pacing wins.
And `human_approval` is not opened by data at all: without `--approve`
it stays shut however good the evidence looks, which is the point of
having it.

ISOLATION
---------
Everything lands under `data/_synthetic/`, including its own `state.db`.
The real shared store is never opened for writing, so there is no way for
a seeded row to be mistaken for a real one later, and `--flush` is a
directory removal rather than a surgical delete that might miss a row.

    python scripts/seed_synthetic_analytics.py --seed
    python scripts/seed_synthetic_analytics.py --verify
    python scripts/seed_synthetic_analytics.py --flush
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import (Artifact, ContentStrategy,      # noqa: E402
                              MeasuredMetric, PerformanceAnalysis)
from agents.learning import (MIN_COHORT, MIN_HISTORY,        # noqa: E402
                             MIN_METRIC_AGE_DAYS,
                             MIN_WINDOW_DAYS, _approval, _gates,
                             _history, _samples, approve, compare)
from agents.store import ArtifactStore                       # noqa: E402

HOME = ROOT / "data" / "_synthetic"
DB = HOME / "state.db"
CSV_FILE = HOME / "_synthetic_export.csv"
PREFIX = "synthetic-"

DAY = 86_400.0

# The signal planted in the numbers. Two cohorts of twelve, identical in
# every respect except the one thing under test, plus a holdout of six.
# A real comparison would have to recover this ratio; nothing in the
# pipeline reads it yet, so it exists to be looked for later.
COHORTS = [
    {"name": "fast-pacing", "pacing": "fast", "base_views": 9_000},
    {"name": "slow-pacing", "pacing": "slow", "base_views": 3_600},
]
EFFECT = 2.5                      # fast gets ~2.5x the views of slow


def _video_id(n: int) -> str:
    """Eleven characters, the shape a real YouTube id has."""
    return f"syn{n:08d}"[:11]


def _rows() -> list[dict]:
    """Thirty videos: 12 + 12 compared, 6 held out, spread over 45 days.

    Every date is at least `MIN_METRIC_AGE_DAYS` old, because a metric
    read sooner than that is not mature and the gate is right to say so.
    """
    now = time.time()
    out, n = [], 0
    for cohort in COHORTS:
        for i in range(MIN_COHORT):
            n += 1
            # 45 days back to 8 days back: wider than the 30-day gate and
            # entirely outside the 7-day maturity window
            first = now - (45 - (n % 37)) * DAY
            last = min(now - (MIN_METRIC_AGE_DAYS + 1) * DAY,
                       first + 30 * DAY)
            views = cohort["base_views"] + i * 137
            out.append({
                "title": f"{cohort['name']} build #{i + 1}",
                "video_id": _video_id(n),
                "cohort": cohort["name"], "pacing": cohort["pacing"],
                "views": views,
                "watch_time_minutes": round(views * 0.78, 1),
                "likes": int(views * 0.031),
                "subscribers_gained": int(views * 0.004),
                "first_day": time.strftime("%Y-%m-%d", time.gmtime(first)),
                "last_day": time.strftime("%Y-%m-%d", time.gmtime(last)),
            })
    # The holdout carries BOTH arms, and the same signal. `compare` sets
    # the most recent videos aside before measuring anything, then looks
    # for the pattern again in them - a holdout made entirely of some
    # third pacing could never agree or disagree, and the gate would then
    # fail for want of data rather than for want of a pattern.
    for i in range(MIN_HISTORY - MIN_COHORT * len(COHORTS)):
        n += 1
        cohort = COHORTS[i % len(COHORTS)]
        first = now - (20 - i) * DAY
        last = now - (MIN_METRIC_AGE_DAYS + 1) * DAY
        views = cohort["base_views"] + i * 70
        out.append({
            "title": f"holdout {cohort['name']} #{i + 1}",
            "video_id": _video_id(n),
            "cohort": "holdout", "pacing": cohort["pacing"], "views": views,
            "watch_time_minutes": round(views * 0.78, 1),
            "likes": int(views * 0.031),
            "subscribers_gained": int(views * 0.004),
            "first_day": time.strftime("%Y-%m-%d", time.gmtime(first)),
            "last_day": time.strftime("%Y-%m-%d", time.gmtime(last)),
        })
    return out


def _window_days(row: dict) -> int:
    a = time.mktime(time.strptime(row["first_day"], "%Y-%m-%d"))
    b = time.mktime(time.strptime(row["last_day"], "%Y-%m-%d"))
    return max(0, int(round((b - a) / DAY)) + 1)


def seed() -> int:
    if DB.exists():
        print(f"already seeded: {DB}\nrun --flush first")
        return 1
    HOME.mkdir(parents=True, exist_ok=True)
    rows = _rows()

    # the CSV an operator would have exported, so the reader is exercised
    # on the same shape it will meet in production
    with CSV_FILE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "title", "video_id", "views", "watch_time_minutes", "likes",
            "subscribers_gained", "first_day", "last_day"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    store = ArtifactStore(DB)
    for i, r in enumerate(rows, 1):
        job_id = f"{PREFIX}{i:03d}"
        # The DECISION half of the sample. A cohort is formed by joining
        # a job's outcome to its own ContentStrategy, so a report with no
        # strategy beside it cannot be put on either side of the
        # comparison - seeding one without the other would prove nothing.
        store.put(Artifact[ContentStrategy](
            job_id=job_id, kind="ContentStrategy",
            produced_by="content_strategy",
            inputs_hash=f"synthetic-strategy-{r['video_id']}",
            payload=ContentStrategy(
                strategy_type="reasoned", pacing=r["pacing"],
                central_topic=r["title"],
                core_angle=f"synthetic {r['cohort']} sample",
                limitations=["SYNTHETIC - seeded for cohort testing"])))
        store.put(Artifact[PerformanceAnalysis](
            job_id=job_id, kind="PerformanceAnalysis",
            produced_by="performance_analysis",
            inputs_hash=f"synthetic-{r['video_id']}",
            payload=PerformanceAnalysis(
                status="measured", provider="youtube_export", published=True,
                video_id=r["video_id"], window_days=_window_days(r),
                measured=[
                    MeasuredMetric(metric="views", value=float(r["views"]),
                                   source=CSV_FILE.name),
                    MeasuredMetric(metric="watch_time",
                                   value=float(r["watch_time_minutes"]),
                                   source=CSV_FILE.name),
                    MeasuredMetric(metric="likes", value=float(r["likes"]),
                                   source=CSV_FILE.name),
                    MeasuredMetric(metric="subscribers_gained",
                                   value=float(r["subscribers_gained"]),
                                   source=CSV_FILE.name),
                ],
                limitations=[f"SYNTHETIC - seeded by {pathlib.Path(__file__).name} "
                             f"into an isolated store; not a real video"])))
    (HOME / "manifest.json").write_text(
        json.dumps({"created": time.time(), "rows": len(rows),
                    "cohorts": {c["name"]: MIN_COHORT for c in COHORTS},
                    "effect": EFFECT, "prefix": PREFIX}, indent=2),
        encoding="utf-8")
    print(f"seeded {len(rows)} measured reports into {DB}")
    print(f"  signal: {COHORTS[0]['name']} ~{EFFECT}x the views of "
          f"{COHORTS[1]['name']}")
    return 0


def verify(approve_it: bool = False) -> int:
    """Run the agent's own history, comparison and gates over the seed."""
    if not DB.exists():
        print("nothing seeded; run --seed first")
        return 1
    store = ArtifactStore(DB)
    job = f"{PREFIX}001"
    h = _history(store, job)
    print(f"history: {h.usable_samples} usable of {MIN_HISTORY} needed, "
          f"{h.performance_reports} report(s), widest window "
          f"{h.window_days}d (gate wants {MIN_WINDOW_DAYS}d)")

    samples = _samples(store, job)
    finding = compare(samples) if h.usable_samples >= MIN_HISTORY else None
    print(f"samples with BOTH an outcome and a pacing: {len(samples)}")
    if finding:
        print(f"finding: {finding.treatment.value} "
              f"({finding.treatment.videos} videos, mean "
              f"{finding.treatment.mean_views:g}) vs "
              f"{finding.control.value} ({finding.control.videos}, "
              f"{finding.control.mean_views:g})")
        print(f"         lift {finding.lift:g}  effect {finding.effect:.0%}"
              f"  holdout lift {finding.holdout_lift}  agrees "
              f"{finding.holdout_agrees}")
    else:
        print("finding: none - nothing could be compared")

    if approve_it:
        approve(job)
    approved, approver = _approval(job)
    print()
    worst = 0
    for g in _gates(h, finding, approved, approver):
        mark = {"passed": "PASS", "failed": "FAIL"}.get(g.state, "----")
        print(f"  {mark}  {g.gate:22} {g.observed[:84]}")
        if g.state == "failed":
            worst = 1
    print()
    if not approved:
        print("human_approval is the one gate no amount of data opens.")
        print("Run again with --approve to see the whole chain close.")
    return worst


def flush() -> int:
    """Remove every synthetic artifact, or say exactly why it could not.

    `ArtifactStore` keeps its sqlite connection open for the life of the
    object, and Windows refuses to unlink an open file. Collecting first
    closes any store this process still holds; a file still locked after
    that belongs to another process - a running server, most likely - and
    saying so beats a bare PermissionError traceback.

    A half-removed directory would be the worst outcome: some synthetic
    rows gone, some left to be mistaken for real ones. So a failure
    reports what remains rather than pretending to have finished.
    """
    if not HOME.exists():
        print("nothing to flush")
        return 0
    import gc
    gc.collect()
    try:
        shutil.rmtree(HOME)
    except PermissionError as exc:
        print(f"could not remove {getattr(exc, 'filename', HOME)}: it is "
              f"open in another process.\nStop the server (or anything "
              f"else reading {HOME.name}) and run --flush again.")
        left = [p.name for p in HOME.rglob("*") if p.is_file()]
        print(f"still present: {', '.join(left) or '(nothing)'}")
        return 1
    print(f"removed {HOME}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--approve", action="store_true",
                    help="grant human_approval for this check")
    a = ap.parse_args()
    if a.flush:
        return flush()
    if a.seed:
        rc = seed()
        return rc or verify(a.approve)
    if a.verify:
        return verify(a.approve)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
