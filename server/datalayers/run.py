"""Run the Data Layers team once: source -> staging -> AI -> analytics.

    cd server
    python -m datalayers.run                 # fetch, build, report
    python -m datalayers.run --no-ai         # rules instead of a model
    python -m datalayers.run --report        # print the last result only

On the VM the same command runs from cron as the machine's service account;
on a workstation it uses whoever is signed in to gcloud. Nothing here
writes to BigQuery: the warehouse is read, the layers are kept locally in
`data/layers/alpha_titans.db` until a dataset of our own is granted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
from datalayers import bigquery as BQ
from datalayers import layers as L

DB_PATH = Path(config.APP_ROOT) / "data" / "layers" / "alpha_titans.db"


def our_jobs(outputs: Path) -> list[dict]:
    """Every build linked to a YouTube video, with the decisions behind it.

    The link is the `published.json` the finished-video page records; a
    build without one cannot be matched, and is not guessed at."""
    from agents.artifacts import ContentStrategy, CriticResult, EditPlan
    from agents.store import store_for_build_dir
    from agents.tools import analytics_store as AS

    out = []
    for ref_file in sorted(outputs.glob(f"*/{AS.REFERENCE_FILE}")):
        build = ref_file.parent
        ref = AS.load_reference(build) or {}
        vid = ref.get("video_id")
        if not vid:
            continue
        decisions: dict = {}
        try:
            store = store_for_build_dir(build)
            job = build.name
            s = store.get(job, "ContentStrategy", ContentStrategy)
            p = store.get(job, "EditPlan", EditPlan)
            c = store.get(job, "CriticResult", CriticResult)
            if s:
                decisions["pacing"] = s.payload.pacing
            if p:
                decisions["shots"] = len(p.payload.segments)
                decisions["seconds"] = p.payload.total_duration
            if c:
                decisions["critic_score"] = c.payload.overall_score
        except Exception:                                   # noqa: BLE001
            pass
        out.append({"job_id": build.name, "video_id": vid,
                    "decisions": decisions})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default="myukti")
    ap.add_argument("--location", default=L.LOCATION)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    db = L.connect(a.db)
    if not a.report:
        def fetch():
            return BQ.query(L.SOURCE_SQL, a.project, a.location, BQ.token())
        llm = None
        if not a.no_ai:
            from agents.tools.llm_tools import LLMClient
            llm = LLMClient()
        result = L.run(db, fetch, llm=llm, jobs=our_jobs(config.OUTPUTS_DIR),
                       export_dir=Path(config.APP_ROOT) / "data" / "analytics")
        if not result["ok"]:
            print("stopped: a checker failed; the layers after it did not run")
            return 1

    rep = L.report(db)
    if a.json:
        print(json.dumps(rep, indent=2))
        return 0
    s = rep["summary"]
    print(f"\n{s['videos']} video(s), {s['matured']} with a full first week, "
          f"{s['linked_to_jobs']} linked to an Alpha Titans job")
    print(f"first-week views: {int(s['day7_views_total'])}   "
          f"average view: {s['avg_view_s']} s\n")
    print(f"{'#':>2}  {'views d7':>8}  {'avg view':>8}  {'hook':<12} title")
    for v in rep["videos"]:
        print(f"{v['rank_views']:>2}  {int(v['day7_views']):>8}  "
              f"{v['day7_avg_view_s']:>7.1f}s  {(v['hook_type'] or ''):<12} "
              f"{(v['title'] or '')[:60]}")
    if rep["groups"]:
        print("\ngroups (descriptive; too few videos to claim a cause):")
        for g in rep["groups"]:
            print(f"  {g['dimension']:<9} {g['value'] or '-':<12} {g['videos']:>2} video(s)  "
                  f"{g['avg_day7_views']:>7} views  {g['avg_day7_view_s']}s")
    print(f"\nLearning needs {s['learning_needs']} matured, linked videos "
          f"before it trusts a pattern.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
