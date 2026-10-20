# A. Repository findings

Inspected on branch `dev-2` (cut from `main` at `f2c1ea3`), 16 Oct 2026.
Everything below was read from the source or observed on this machine.
Where something could not be verified it says so.

## What exists

| Concern | Where | What it actually does |
|---|---|---|
| HTTP API | `server/main.py` | FastAPI app. Legacy routes under `/api/...`, the agent layer's router under `/api/v2/...` (mounted inside a `try`, so the app starts without LangGraph installed). Serves `frontend/` same-origin. |
| Config | `server/config.py` | Stdlib `.env` reader; real environment variables win. Every path is relative to the app folder. |
| Agent graph | `server/agents/graph.py` | LangGraph `StateGraph`, 20 nodes (18 agents plus `finish`/`escalate`). See topology below. |
| Pipeline entry point | `server/agents/runner.py` | `make_job()` → `run_job()` / `resume_job()`. This is what `/api/v2/jobs/{id}/produce` calls, in a daemon thread. |
| Media engine | `pipeline/build_video.py` | Protected. Resumable stdlib + ffmpeg stages `probe → measure → plan → sheets → cut → join → music → mux`. |
| Agent state | `outputs/_agents/state.db` | One SQLite file shared by every job: artifacts plus traces. |
| Graph checkpoints | `outputs/<job>/checkpoints.db` | One LangGraph SQLite checkpointer per job, `thread_id = job_id`. `resume_job` continues from it. |
| Measurement cache | `outputs/_agents/measure_cache/` | Global, keyed on each file's sha256. Re-measuring the same footage is skipped across jobs. |
| Job layout | `uploads/<job>/`, `outputs/<job>/` | Sources in, build out: `run.log`, `job.json`, `manifest.json`, the final MP4. |
| Containers | `Dockerfile`, `docker-compose.yml` | `python:3.11-slim` + apt ffmpeg; volumes for `uploads/` and `outputs/`. |
| Public sharing | `share.bat` | Runs a `cloudflared` quick tunnel to the local server. |

### The real graph topology

```
clip_analysis ─┬─> vision ───┐
               └─> research ─┴─> content_strategy -> story -> voiceover -> music
    -> animation -> transition -> caption -> editing_director -> render -> critic
                                                  ^                          │
                                                  └──────── "revise" ────────┤
                                                         "finish" / "escalate"
                                                                             v
             thumbnail -> title -> publishing -> performance_analysis -> learning -> END
```

`vision` and `research` run in parallel and join at `content_strategy`.
`critic -> editing_director` is a real `add_conditional_edges` cycle, bounded
by `max_revisions`.

## Conflicts between the brief and the code

These matter more than anything else in this document.

### 1. Nothing in the pipeline uses a GPU

```
pipeline/build_video.py:865,910,1038   "-c:v", "libx264"          (CPU encoder)
server/extras.py:138                    WhisperModel(..., device="cpu")
LLMs                                    hosted (Groq, Gemini) - no local inference
```

No CUDA, no NVENC, no torch. On an L4 VM every job would leave the GPU idle
while paying accelerator-optimised rates for 4 vCPUs of `libx264` work. The
on-demand lifecycle built here is machine-agnostic, so it works unchanged on
either:

* **the existing L4 VM**, as briefed; or
* **a CPU-optimised VM**, which on today's code would render faster and cost
  less. That is a configuration change, not a code change.

Actually using the L4 means encoding with `h264_nvenc`, which touches the
protected `pipeline/build_video.py`. A change proposal is in
[`07-nvenc-proposal.md`](07-nvenc-proposal.md). **It is not implemented** and
needs your explicit approval.

### 2. There is no authentication anywhere

No route in `server/main.py` or `server/agents/api.py` checks a credential.
Upload, produce, delete and cancel are all open. Today that is contained by
the default `AT_HOST=127.0.0.1`, but:

* `docker-compose.yml` sets `AT_HOST=0.0.0.0`;
* `share.bat` publishes the server through a public `cloudflared` tunnel,
  **with every route unauthenticated**.

The cloud routes added here require a bearer token and refuse to start on a
non-loopback host without one. **The existing routes are still open.** Before
the control plane is exposed publicly, put the whole app behind an
authenticating proxy; see the deployment guide.

### 3. The operator's gcloud session has expired

```
gcloud config: project samriddhi-production, account akansha.lakhina@mysamriddhi.com
gcloud compute instances list -> "Reauthentication failed. cannot prompt during
                                  non-interactive execution"
```

So **the L4 VM's name, zone, machine type, service account and access scopes
could not be verified.** None of them are invented here; each is a required
setting that fails validation until it is supplied.

This also settles a design question. User credentials expire and need a human
to renew them, which is exactly what "no human should be required" rules out.
Nothing in the cloud path uses user credentials. Both machines authenticate as
service accounts through the GCE metadata server.

### 4. Two SQLite files would not share state

`outputs/_agents/state.db` lives on whichever machine runs the graph. If the
control plane and the worker are different machines, each has its own file.
So:

* the **job queue** lives in a separate SQLite file owned by the control plane,
  which is its **only writer**;
* the **worker never opens it**. It claims, heartbeats and completes over
  authenticated HTTP;
* the worker's own `state.db` and `checkpoints.db` are a **cache**. They make
  a retry cheap, but losing them loses no job. The queue and Drive are the
  record.

### 5. A service account cannot own Drive files

Service accounts have no Drive storage quota. An upload into a folder in
someone's *My Drive* fails with `storageQuotaExceeded`, even when the folder
is shared with the account. Outputs therefore have to go into a **Shared
Drive** with the service account as a member. `mysamriddhi.com` looks like a
Workspace domain, which would allow that, but this was not verified.

### 6. Drive-direct browser upload is unverified

Cloud Storage documents CORS for resumable sessions initiated server-side
(pass `Origin` at initiation). No Drive documentation found here states the
same. So the default upload mode **relays** chunks through the control plane,
which works by construction and keeps the session URI, a bearer capability,
off the browser. Direct mode is implemented but off, pending one real-browser
test.

### 7. The upload UI shows no byte progress

`frontend/studio.js` uploads with `fetch`, which exposes no upload progress.
The older `frontend/app.js` used `XMLHttpRequest.upload.onprogress`. The cloud
uploader reports bytes **acknowledged by the server**, not bytes handed to the
socket.

### 8. A guard test forbids any Google upload URL

`tests/agents/test_publishing.py::test_nothing_in_the_project_can_upload`
scans `server/` for `googleapis.com/upload`, which a Drive client must use. The
guard exists to catch a **YouTube** publishing path; the publishing agent's
claim is "the operator publishes". The marker is narrowed to YouTube, and a
test proves a YouTube upload URL is still caught. No URL was disguised to get
past it.

## Already true and preserved

* Models decide *what*; deterministic code decides *where*. The worker passes
  settings into the existing `JobSettings` schema and nothing else. No model
  output reaches a shell, and no request parameter reaches a VM name, project,
  zone or executable path.
* The LLM retry budget is bounded per call (this branch, `llm_tools.py`), so a
  rate limit cannot hold the worker for long. The worker also enforces a
  maximum job duration on top.
* `resume_job` plus the global measurement cache plus `inputs_hash` artifact
  caching mean a retried job does not redo finished work, provided the
  worker's disk survives. A stopped GCE VM keeps its persistent disk.

## Out of scope for this slice

* BigQuery analytics. It belongs to `performance_analysis` and `learning`, not
  to orchestration.
* YouTube publishing. Still the operator's step, by design.
* Moving any agent off the worker. Section 10 of the brief allows it only once
  artifacts can be shared safely. They cannot yet: they live in the worker's
  `state.db`.
