# B. Architecture decision record — on-demand worker

Status: **implemented on `dev-2`, not deployed.** Nothing here has run against
real Google Cloud resources. See the test report for exactly what was proven
and how.

## Decision

**D (a durable database-backed queue, an on-demand worker, periodic
reconciliation), deployed as B (a lightweight always-on control plane).**

The control plane runs on a **GCP e2-micro**, not on Render. The rest of this
document is why.

## The alternatives

| | A. Always-on L4 + local queue | B. Control plane + on-demand L4 | C. Cron starts VM, VM polls Drive | **D. DB queue + on-demand worker + reconciler** |
|---|---|---|---|---|
| Idle GPU cost | **~$516/month** at $0.707/h, whether or not anything runs | none | short bursts each tick, even with no work | none |
| Who owns job state | the GPU VM's disk | the control plane | **Drive folder listings**: no locks, no leases, no attempt counts | the control plane's queue |
| Crash recovery | restart, hope local state is intact | depends on the queue it uses | re-scan Drive; two VMs can process one file | lease expiry → retry with fencing |
| Duplicate work | n/a | depends | likely, with no claim primitive | atomic claim, fencing token |
| Complexity | lowest | medium | low, but brittle | medium |
| Fit for this repo | runs today, at the worst cost | B is a deployment shape, not a queue design | Drive is not a queue | uses the existing entry point unchanged |

**A** is rejected on cost alone. **C** is rejected because Drive has no atomic
claim, so two pollers can take the same file, and "has this been done?"
becomes a guess based on folder contents. **B and D are not rivals**: D is the
mechanism and B is where it runs. That is the brief's own preference, and
inspection supports it.

## Where the control plane runs

The queue needs a **single writer on a persistent disk**, with **no
long-lived credentials** and **no new Python dependencies**.

| Host | Persistent disk | Always on | Talks to GCP without an exported key, no new deps | Verdict |
|---|---|---|---|---|
| Render free | **no**: free web services cannot attach a disk | **no**: spins down after 15 min idle. Pinging to keep it awake is treated as abuse by Render staff | **no** | rejected: the queue would vanish on every restart |
| Render paid (~$7/mo + $0.25/GB disk) | yes | yes | **no**: needs an exported service-account key, and RS256 signing has no stdlib implementation. That means `cryptography` or `google-auth` (a new dependency) or hand-written private-key crypto | rejected |
| **GCP e2-micro** (always-free tier) | yes, 30 GB-month standard disk | yes | **yes**: the metadata server issues short-lived tokens over plain HTTP | **chosen** |
| Cloud Run + Firestore | no, so the queue moves to Firestore | scales to zero; reconciliation needs Cloud Scheduler | yes | viable later; a larger change |
| Operator's laptop | yes | no | no: user credentials expire, as happened during this inspection | rejected |

Vercel is not needed. The control plane already serves the vanilla-JS
frontend same-origin, so there is no CORS and no cross-origin token to
manage. Hosting the static files there later is possible but buys nothing for
this slice.

## Components

```
                   browser (vanilla JS, /static/cloud.html)
                        │  HTTPS, bearer token   (via an authenticating proxy)
                        v
 ┌──────────────── CONTROL PLANE: e2-micro, always on ──────────────────┐
 │  FastAPI (existing app)  +  server/cloud/  router                     │
 │    • upload sessions, chunk relay, completion check                   │
 │    • job queue        outputs/_cloud/queue.db   (SQLite, sole writer) │
 │    • VM manager       single-flight start / stop of ONE named VM      │
 │    • reconciler       in-process thread, every 60s                    │
 │  identity: its own service account via the metadata server            │
 └──────┬──────────────────────────────┬─────────────────────────┬──────┘
        │ Drive REST                   │ Compute REST            │ VPC-internal HTTP
        │ (resumable upload,           │ (instances.get/start/   │ claim / heartbeat /
        │  verify size + md5)          │  stop, ONE instance)    │ complete / fail / idle
        v                              v                         │ Google ID token (RS256)
 ┌──────────────┐          ┌───────────────────────────────────┐ │
 │ Shared Drive │<─────────│ WORKER VM (the L4), on demand     │<┘
 │  inputs/     │ download │  systemd: alpha-titans-worker     │
 │  outputs/<j> │  upload  │  python -m cloud.worker           │
 └──────────────┘          │    └ subprocess: cloud.run_one    │
                           │        └ agents.runner.run_job /  │
                           │          resume_job  (UNCHANGED)  │
                           │            └ build_video.py       │
                           │  local cache: outputs/_agents/,   │
                           │  checkpoints.db, measure_cache    │
                           │  exit 75 -> systemd powers off    │
                           └───────────────────────────────────┘
```

Four concerns, kept separate:

| Concern | Owner | Never does |
|---|---|---|
| Infrastructure orchestration | `server/cloud/` | sequence agents |
| Agent orchestration | the existing LangGraph graph | know about VMs, leases or Drive |
| Media orchestration | `pipeline/build_video.py`, unmodified | know about any of the above |
| UI orchestration | `frontend/cloud.js` + `/api/cloud/jobs/{id}` | invent progress |

The queue does not re-implement the graph. A job is one opaque unit of work,
"run the existing pipeline on these files with these settings". Stage-level
progress is **read from the worker's existing trace rows** and relayed, not
tracked a second time.

## Job lifecycle

```
            create session
                 │
             uploading ──(abandoned > 24h)──> failed
                 │ all files received AND verified in Drive (size, md5, mime)
                 v
   ┌───────── queued <──────────────┐
   │             │ VM start issued   │ start failed / VM stopped
   │             v                   │
   │        starting_vm ─────────────┘
   │             │ worker claims (atomic, fencing token)
   │             v
   ├──────> downloading ──> processing ──> uploading_results ──> completed
   │             │               │                 │
   │             └───────────────┴─────────────────┘
   │                 lease expired / retryable failure
   │                             │
   └──── retrying <──────────────┘    (not_before = now + backoff)
              │
              └── attempts exhausted / non-retryable ──> failed

   any non-terminal state ──(user cancel)──> cancelled
```

Every transition is checked against an explicit table (`queue.TRANSITIONS`)
and written to an append-only `job_events` audit table.

| Field | Purpose |
|---|---|
| `job_id`, `idempotency_key` (unique) | the same upload intent creates one job however often it is retried |
| `state`, `stage`, `progress_json` | overall state, the current graph node, nodes done, upload bytes |
| `attempt`, `max_attempts`, `not_before` | at-least-once delivery with bounded retries and backoff |
| `lease_owner`, `lease_expires_at`, `claim_token` | concurrency control; the token **fences** a zombie worker |
| `error_category`, `error_message` | sanitised; raw cloud errors never reach the browser |
| `worker_id`, `started_at`, `finished_at` | execution record |
| `settings_json`, `outputs_json`, `trace_json` | configuration in, Drive file IDs and the trace out |

**At-least-once, not exactly-once.** A worker can finish the pipeline and
crash before its `complete` call lands. The retry then finds every artifact
cached and every output already in Drive (uploads are skipped when a file
with the same name and md5 is already in the job's folder), so the second
pass is cheap and idempotent. Exactly-once is not promised, because it cannot
be.

## VM lifecycle

1. Upload completes → job `queued` → `request_start()` runs in the same
   request.
2. **Single flight.** `request_start` takes `BEGIN IMMEDIATE` on the queue
   database and moves the VM row to `starting` only if it is stopped, not
   started within the last 120s, and under 6 starts per hour. Only the caller
   that made that transition calls the Compute API, and it does so **after**
   committing, so a network call never holds the lock. Ten simultaneous
   uploads produce one `instances.start`.
3. The VM boots. systemd starts `alpha-titans-worker`. The worker checks
   ffmpeg, ffprobe, the pipeline imports, free disk and (optionally) the GPU,
   then calls `hello`.
4. It **drains the queue**: claim → download → run → upload → complete →
   claim again.
5. With nothing to claim, it asks the control plane `idle`. The control plane
   answers `shutdown` only if, **inside one transaction**, there is no
   claimable job, no active lease and the idle grace has elapsed. It marks the
   VM `stopping` in that same transaction, so a job queued a moment later is
   handled by the reconciler once the VM reaches `TERMINATED`, rather than
   being lost in a race.
6. The worker exits with code **75**. The unit's `ExecStopPost=+...` powers
   the machine off **only for that exit code**. An operator's
   `systemctl stop`, or a crash, does not.
7. The worker needs **no Compute IAM permission at all**. It cannot start or
   stop anything, including itself, except by exiting.

### Safeguards

| Risk | Guard |
|---|---|
| VM boots, worker never comes up | `STARTUP_TIMEOUT` (900s) with no `hello` → reconciler stops the VM, puts jobs back to `queued`, retries with backoff |
| Worker crash-loops | `Restart=on-failure`, `StartLimitBurst=5` in 15 min, `StartLimitAction=poweroff` |
| Worker dies mid-job | lease (300s, heartbeat every 30s) expires → `retrying` with backoff → resumed from the checkpoint |
| VM left running with no worker | reconciler stops a RUNNING VM that has had no active job, no claimable job and no worker contact for idle grace + 3 heartbeats |
| Start/stop flapping | jobs are committed **before** a start is requested; at least 120s between starts; at most 6 per hour, after which the start circuit opens and status says so |
| Zombie worker completes a re-claimed job | every worker call carries `claim_token`; a stale token gets `409 lease lost` |
| Provider outage holds the VM | LLM calls are capped at 150s each (`llm_tools`); the pipeline subprocess is killed at `MAX_JOB` (90 min); a `rate_limited` failure backs off 15 min, so the worker goes idle and powers off rather than waiting |
| Upload interrupted | job stays `uploading` and is never claimable; the browser resumes from the server-acknowledged offset |
| Output upload fails | `complete` is refused unless every declared output carries a Drive file ID verified by size and md5. The job retries; it is never marked complete |

## Batching

One activation drains the whole queue, **one job at a time**. Each job
already saturates the machine: libx264 uses every core, and the pipeline's
`MediaPool` parallelises inside a job. Running two at once would make both
slower and gain nothing.

Every activation pays a fixed overhead of **B** (boot to `hello`) plus **G**
(idle grace) at rate **R**. Processing N queued jobs in one activation instead
of N activations saves:

```
(N − 1) × (B + G) × R
```

With B = 120s (an assumption; **measure it on the real VM**), G = 180s and
R = $0.707/h, that is **$0.059 and 5 minutes of latency per avoided boot**.

**The idle grace** is a ski-rental problem. Idling costs G × R for certain;
stopping risks paying B again if a job arrives. Waiting exactly as long as a
boot costs (G = B) bounds the total at no more than twice the optimum,
whatever the arrival pattern. **Set `AT_CLOUD_IDLE_GRACE_S` to the measured
boot time.** The 180s default is a placeholder until it is measured.

The worker stops claiming once an activation has run `MAX_ACTIVATION`
(6h by default). It finishes the job in hand, powers off, and the reconciler
starts a fresh activation if anything is still queued. This bounds the damage
from a leak nobody noticed.

## Scheduling: what runs on a timer, and why only that

| Mechanism | Used for | Cost |
|---|---|---|
| Durable queue | **all** job triggering | free (SQLite on the e2-micro disk) |
| Start request in the upload-complete request | the immediate path | none |
| In-process reconciler thread, every 60s | recovery: expired leases, lost start requests, startup timeouts, orphaned VMs, abandoned uploads | free; the control plane is always on |
| systemd `Restart=always` on the control plane | recovery of the reconciler itself | free |
| Worker claim loop | draining, **only while the VM is already up** | none extra |
| Cloud Scheduler | not needed; would be a backup for the reconciler (3 jobs free per billing account) | free if used |
| Render cron | not used: $1/month minimum per job, and a sleeping free service runs no in-process scheduler | |

No agent has a cron job. LangGraph already sequences them.

## Security

* **Fixed operations only.** Project, zone and instance come from
  configuration, are validated against GCE's naming rules at startup, and are
  never taken from a request. Each Compute call is one of three hard-coded
  URL templates.
* **No exported keys anywhere.** Both machines use their attached service
  accounts through the metadata server.
* **Worker → control plane** uses a Google-signed ID token, minted per request
  on the worker by the metadata server and verified on the control plane:
  RS256 against Google's published keys, then `iss`, `aud`, `exp` and the
  expected service-account `email`. That is public-key verification, done by
  reconstructing the full PKCS#1 v1.5 encoding and comparing it, not by
  parsing the signature. The traffic also stays on the VPC's internal network.
* **Browser → control plane** needs `AT_CLOUD_API_TOKEN`. The cloud routes
  refuse to mount on a non-loopback host without it.
* **The Drive session URI is a bearer capability.** In relay mode it never
  leaves the control plane. It is stored in the queue database (which should
  be mode 0600) and is never logged; `sanitize.redact` strips it from any
  error text.
* **Errors are classified, then sanitised.** The browser sees a category and
  a fixed sentence. The redacted detail stays in the server log.

### Least-privilege IAM

| Identity | Needs |
|---|---|
| Control plane SA | a custom role with `compute.instances.get`, `compute.instances.start`, `compute.instances.stop` and `compute.zoneOperations.get`, **bound to the worker instance only** with an IAM condition on its resource name; Content manager on the Shared Drive |
| Worker SA | Content manager on the Shared Drive. **No Compute permissions.** |
| Both VMs | access scopes that include `https://www.googleapis.com/auth/drive`. `cloud-platform` does not include Drive, and scopes can only be changed while a VM is stopped |

## Amendment, 17 Oct 2026: what changed on contact with a real project

The design above held: one durable queue owned by the control plane, an
on-demand worker that drains it and powers itself off, leases, fencing and
a reconciler. Four things changed when it was deployed on a personal Google
account (`alphatitans-cloud-1017`):

| Decided above | Deployed | Why |
|---|---|---|
| Google Drive, Shared Drive | **Cloud Storage**, one private bucket (`AT_CLOUD_STORAGE=gcs`) | a personal account has no Shared Drives, and a service account owns no Drive storage, so its uploads fail with `storageQuotaExceeded`. GCS's resumable upload is the same protocol, so `gcs.py` subclasses the Drive client and only the addresses change. Drive stays available for Workspace |
| the main app with the cloud routes mounted | **`cloud.control_app`**, which serves only `/cloud`, its three static files, `/api/cloud/*` and `/healthz` | the older routes have no authentication; on a public address they must not exist at all, rather than be filtered by a proxy |
| Drive scope on both VMs; Content manager on the Shared Drive | `cloud-platform` scope; `roles/storage.objectAdmin` **on the bucket only** for each service account | the bucket is the only data either account can reach |
| an L4 worker | an **e2-standard-8** CPU worker | the free trial's GPU quota is 0, and quota is never requested by this project. The engine encodes with libx264 on the CPU anyway (findings, item 1) |
| a named cloudflared tunnel with Cloudflare Access | a **quick tunnel** for now | it needs no domain; its address changes on restart, so it is for testing |

Outputs reach the browser through the control plane, because a private
bucket has no public links: a one-time, 60-second download ticket hands the
file to the browser's own download manager, and a 30-minute view ticket
feeds the page's `<video>` and `<img>` with ranged requests. Neither puts
the operator token in a URL.

Also confirmed live: starting the worker needs only `compute.instances.get`,
`start` and `stop` on that one instance; `iam.serviceAccounts.actAs` was not
needed.
