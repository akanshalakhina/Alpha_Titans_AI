"""Cloud orchestration for Alpha Titans: a durable job queue, an on-demand
worker VM, and the reconciliation that keeps the two honest.

Infrastructure only. Nothing in this package sequences agents, renders video
or decides anything about an edit - the existing LangGraph graph and
`pipeline/build_video.py` do that, unchanged. This package decides WHEN that
work runs, on WHICH machine, and what happens when a machine goes away.

Off unless `AT_CLOUD_ENABLED=1`. With it off, importing the application
mounts no route, starts no thread and opens no database here.

See docs/cloud/ for the architecture decision record.
"""
