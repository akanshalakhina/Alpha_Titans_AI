"""How far along a production is, and roughly how long is left.

FROM WHAT WE HAVE MEASURED, NOT FROM A GUESS
--------------------------------------------
Every finished job leaves its own timings: the queue's events say when it
was queued, when the worker started fetching, when it began saving; the
graph trace says when each agent started and finished. The typical time
for every step is the median over the recent finished jobs, so the estimate
is "what the last few videos made here took", and it says how many that
was. With no history the defaults below are used; they are what the first
runs on this system measured, and the page labels such an estimate rough.

Two steps scale with the footage instead of being a fixed time: fetching
the clips (seconds per MB) and measuring them, `clip_analysis` (seconds per
MB too - an 80-minute clip took eight minutes, a one-minute clip seconds).

THE NUMBERS
-----------
`percent` is the share of the expected total already behind the job: the
steps that finished count their typical time, the one running counts the
time it has run so far, capped just under its typical time so the bar never
claims a step is done before it is. `remaining_s` is the typical time of
what is left; when a step overruns its typical time, the estimate stops
counting it down rather than going negative or promising "any moment".
"""

from __future__ import annotations

import statistics

# The graph's agents in running order (vision and research run side by
# side; both are counted, as both have to finish).
NODES = ["clip_analysis", "vision", "research", "content_strategy", "story",
         "voiceover", "music", "animation", "transition", "caption",
         "editing_director", "render", "critic", "thumbnail", "title",
         "publishing", "performance_analysis", "learning"]

# Seconds, from the first runs on this system (17-18 Oct 2026).
DEFAULT_NODE_S = {"clip_analysis": 60, "vision": 35, "research": 2,
                  "content_strategy": 4, "story": 5, "voiceover": 4,
                  "music": 12, "animation": 15, "transition": 10,
                  "caption": 8, "editing_director": 25, "render": 150,
                  "critic": 15, "thumbnail": 2, "title": 3, "publishing": 4,
                  "performance_analysis": 1, "learning": 1}
DEFAULTS = {"cold_start_s": 50.0,   # the worker was off: boot + claim
            "warm_start_s": 10.0,   # the worker was on: just the claim
            "fetch_s_per_mb": 0.08, "analysis_s_per_mb": 0.4,
            "save_s": 6.0}
HISTORY_JOBS = 10


def _median(xs, default):
    xs = [x for x in xs if x is not None and x >= 0]
    return float(statistics.median(xs)) if xs else float(default)


def _event_times(events: list[dict]) -> dict:
    """First time each state was entered, from the job's own audit trail."""
    out: dict = {}
    for e in events or []:
        out.setdefault(e.get("to") or e.get("to_state"), e.get("at"))
    return out


def _mb(job: dict) -> float:
    return sum(float(s.get("size") or 0) for s in job.get("sources") or []) / 1e6


def learn(finished: list[tuple[dict, list[dict], list[dict]]]) -> dict:
    """Typical timings from finished jobs: (job, events, trace rows)."""
    node: dict[str, list[float]] = {n: [] for n in NODES}
    cold, warm, fetch, analysis, save = [], [], [], [], []
    for job, events, trace in finished[:HISTORY_JOBS]:
        t = _event_times(events)
        mb = max(_mb(job), 1.0)
        start = (t["starting_vm"] if t.get("starting_vm") is not None
                 else t.get("queued"))
        if start is not None and t.get("downloading") is not None:
            gap = t["downloading"] - start
            (cold if gap > 25 else warm).append(gap)
        if t.get("downloading") is not None and t.get("processing") is not None:
            fetch.append((t["processing"] - t["downloading"]) / mb)
        if (t.get("uploading_results") is not None
                and t.get("completed") is not None):
            save.append(t["completed"] - t["uploading_results"])
        for r in trace or []:
            n, a, b = r.get("node"), r.get("started_at"), r.get("ended_at")
            if (n in node and a is not None and b is not None
                    and r.get("status") == "ok"):
                node[n].append(b - a)
                if n == "clip_analysis" and b - a > 10:
                    # a clip measured before is reused in seconds; only a
                    # real measurement says how long the footage takes
                    analysis.append((b - a) / mb)
    return {
        "jobs": len(finished[:HISTORY_JOBS]),
        "node_s": {n: _median(v, DEFAULT_NODE_S[n]) for n, v in node.items()},
        "cold_start_s": _median(cold, DEFAULTS["cold_start_s"]),
        "warm_start_s": _median(warm, DEFAULTS["warm_start_s"]),
        "fetch_s_per_mb": _median(fetch, DEFAULTS["fetch_s_per_mb"]),
        "analysis_s_per_mb": _median(analysis, DEFAULTS["analysis_s_per_mb"]),
        "save_s": _median(save, DEFAULTS["save_s"]),
    }


def estimate(job: dict, events: list[dict], worker_state: str | None,
             hist: dict, now: float) -> dict | None:
    """`{"percent", "remaining_s", "based_on"}`, or None where there is
    nothing honest to say (still uploading, waiting to retry, stopped)."""
    state = job.get("state")
    if state == "completed":
        return {"percent": 100, "remaining_s": 0, "based_on": hist["jobs"]}
    if state not in ("queued", "starting_vm", "downloading", "processing",
                     "uploading_results"):
        return None
    t = _event_times(events)
    mb = max(_mb(job), 1.0)
    node_s = dict(hist["node_s"])
    node_s["clip_analysis"] = min(1800.0, max(node_s["clip_analysis"],
                                              hist["analysis_s_per_mb"] * mb))
    start_s = (hist["warm_start_s"] if worker_state == "ready"
               and state == "queued" else hist["cold_start_s"])
    steps = ([("start", start_s), ("fetch", max(2.0, hist["fetch_s_per_mb"] * mb))]
             + [(n, node_s[n]) for n in NODES] + [("save", hist["save_s"])])
    total = sum(s for _, s in steps)

    # which steps are behind the job, and since when the current one runs
    progress = job.get("progress") or {}
    done_nodes = set(progress.get("nodes_done") or [])
    if state in ("queued", "starting_vm"):
        finished, current = set(), "start"
        since = t.get("queued")
    elif state == "downloading":
        finished, current = {"start"}, "fetch"
        since = t.get("downloading")
    elif state == "processing":
        finished = {"start", "fetch"} | done_nodes
        current = progress.get("current_node")
        since = progress.get("current_started_at")
        if since is None:
            since = t.get("processing")
    else:                                                # uploading_results
        finished = {"start", "fetch", *NODES}
        current = "save"
        since = t.get("uploading_results")

    done = sum(s for name, s in steps if name in finished)
    left = sum(s for name, s in steps
               if name not in finished and name != current)
    if current in dict(steps):
        expected = dict(steps)[current]
        ran = max(0.0, now - since) if since is not None else 0.0
        done += min(ran, 0.95 * expected)
        left += max(expected - ran, 0.0)
    percent = int(round(100 * done / total)) if total else 0
    return {"percent": max(1, min(99, percent)),
            "remaining_s": int(round(left)),
            "based_on": hist["jobs"]}
