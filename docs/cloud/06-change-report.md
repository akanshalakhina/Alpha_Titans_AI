# G. Change report

Branch `dev-2`, cut from `main` at `f2c1ea3`. **Nothing is committed.**
Committing was not requested; when it is, attribution is Akansha only.

`dev-2` also carries the earlier, still-uncommitted fixes from before the
cloud work (EditingDirector truncation, the LLM retry budget, the vision 503
wording, moved-project downloads, research auth, the learning-gate guard).
They are listed separately below so the two can be reviewed, or committed,
apart.

## Cloud orchestration (this task)

### Added

| File | Purpose |
|---|---|
| `server/cloud/__init__.py` | package; inert unless `AT_CLOUD_ENABLED=1` |
| `server/cloud/config.py` | validated settings; no defaults for real infrastructure; secrets never echoed |
| `server/cloud/sanitize.py` | error categories, safe public messages, credential redaction |
| `server/cloud/queue.py` | durable SQLite queue: explicit state machine, idempotency, atomic claim, leases, fencing tokens, retries with backoff, audit trail |
| `server/cloud/gcp.py` | stdlib Google clients: no-redirect HTTP, metadata/gcloud tokens, Compute (get/start/stop of one instance), Drive (resumable chunked upload, verified resumable download) |
| `server/cloud/idtoken.py` | Google ID-token verification (RS256 by full-encoding comparison; `iss`/`aud`/`exp`/`email`) |
| `server/cloud/vm.py` | worker VM lifecycle: single-flight start, backoff and circuit, idle/shutdown decision, reconciliation against real VM status |
| `server/cloud/reconcile.py` | the recovery loop (thread plus an endpoint) |
| `server/cloud/paths.py` | filename and path safety; the engine's own format list |
| `server/cloud/api.py` | `/api/cloud/*`: browser routes (token) and worker routes (ID token) |
| `server/cloud/app.py` | wiring; mounts nothing unless enabled and correctly configured |
| `server/cloud/gcs.py` | Cloud Storage behind the Drive client's interface: same resumable protocol, private bucket, ranged streaming to the browser |
| `server/cloud/control_app.py` | the app the control-plane VM runs: `/cloud`, three static files, `/api/cloud/*`, `/healthz`, and nothing else (404 for every older route) |
| `server/cloud/worker.py` | the data plane: readiness, drain loop, heartbeat, download, pipeline subprocess with deadline and tree-kill, verified upload, cleanup, exit 75 to power off |
| `server/cloud/run_one.py` | runs the **existing** `make_job` / `run_job` / `resume_job` for one job; resume decided from the checkpoint; writes `youtube.txt` from the Title and Publishing artifacts |
| `frontend/cloud.html`, `frontend/cloud.js` | the page at `/cloud`, built from the studio's own shell, Create screen, production phases and finished-video screen (read from `main`, which was not changed) and styled by `studio.css`; server-confirmed upload progress; results come down through the browser's own download manager |
| `deploy/cloud/*` | systemd units (worker, control, quick tunnel), env examples, custom IAM role, cloudflared config |
| `docs/cloud/*` | findings, ADR, deployment guide, test report, cost analysis, this report, NVENC proposal |
| `tests/cloud/*` | 225 tests, with fakes for Compute, Drive and Cloud Storage |

About 3,750 lines of server code, 2,580 of tests and 570 of frontend.

### Modified for the cloud work

| File | Change | Why |
|---|---|---|
| `server/main.py` | one guarded block mounting `cloud.app.mount()` | the only hook; it does nothing when cloud is off (tested) |
| `server/agents/publishing.py` | the upload-endpoint marker narrowed from Google's generic upload path to YouTube's (still split, so the file never contains its own target); the capability sentence says "no **YouTube** upload" | the guard exists to catch channel publishing; a Drive client legitimately uses Drive's upload endpoint. No URL was disguised to get past it |
| `tests/agents/test_publishing.py` | the sentence assertion follows; two new tests: Drive is not mistaken for YouTube, and every YouTube spelling is still caught | |
| `tests/agents/test_title.py` | `test_nothing_in_the_project_publishes_anywhere` (a raw-text scan) forbids YouTube's upload path and `youtube/v3/videos` instead of any Google upload path | same reason, same guarantee: the full agents run caught this, which is what it is for |
| `server/agents/vision_agent.py` | circuit breaker: after one clip is refused through every retry for a **transport** reason, later clips in that run are not asked | brief section 10; ~154s of a 268s vision span was the second clip relearning a dead provider. Not triggered by a bad answer; the skipped clip's reason keeps the provider's status line, so it stays non-cacheable |

### Protected interfaces preserved

* `pipeline/build_video.py`: **not modified.** Its stage contracts are
  unchanged. NVENC is a proposal only.
* The agent graph: same nodes, same edges, the same critic → director cycle.
  The worker calls the existing entry points.
* `JobSettings`: the cloud upload validates against it, unchanged.
* Artifact store and `inputs_hash` caching: unchanged. Run on the worker,
  they are what make retries cheap.
* `/api/...` and `/api/v2/...`: unchanged by the cloud work.
* Zero new Python dependencies; zero npm.

## Earlier fixes on the same branch (uncommitted)

| File | Fix |
|---|---|
| `server/agents/tools/json_repair.py`, `llm_tools.py`, `editing_director.py` | a reply cut off at `max_tokens` was salvaged into a fragment and misread as "no segments"; now detected and widened; director budget 1600 → 2600 on measurement |
| `server/agents/tools/llm_tools.py` | the retry budget covers the whole call: in-flight time and every fallback model (the configured worst case was 46 minutes for one agent) |
| `server/agents/vision_agent.py` | a provider outage is reported as one, so it is not cached as a fact about the job |
| `server/main.py` | `/download` falls back to the build folder when the recorded absolute path predates the project move (27 of 35 builds recovered) |
| `server/agents/tools/search_tools.py` | Tavily auth sent as the current Bearer header as well as the legacy body field |
| `server/agents/learning.py`, `scripts/seed_synthetic_analytics.py` | seeded analytics can never become learning samples |
| `server/agents/store.py`, `animation.py`, `render_tools.py`, `select_tools.py` | transient-cache refusal, overlay thinning, one overlay per shot, top-up only from chosen clips |
| `pipeline/build_video.py.bak` | deleted: a stale backup of the protected file, not the file itself |
| tests across `tests/agents/`, `tests/backend/test_moved_project.py` | regression tests for each of the above |

## Where it stands

**Live:** a control plane and an on-demand worker on Google Cloud, a public
HTTPS page, two real productions end to end, the worker powering itself off
when idle. See the deployment guide and the test report.

**Not done, and why:**

1. **Nothing is committed.** Committing was not requested.
2. **The quick tunnel's address changes** whenever `cloudflared-quick`
   restarts. A fixed address needs a named tunnel and a domain on
   Cloudflare.
3. **Hardening the owner should decide** (listed in the deployment guide):
   SSH open to the internet through the default network rule, port 8000
   open to the whole VPC, no lifecycle rule on uploaded inputs.
4. **No GPU.** The free trial's GPU quota is 0 and quota is never requested
   here; running local models on a GPU needs a paid account and that quota.
   NVENC remains a proposal.
5. **Not yet exercised live:** a worker killed mid-render, and the control
   plane stopping a VM that never became ready (both tested against fakes).
