# D. Deployment guide

This is what is **running now** (17 Oct 2026), how to reproduce it, and how
to operate it. Every value below was read back from the live project with
`gcloud ... describe` or from the VMs themselves, not copied from a plan.

## What is running

| | As deployed |
|---|---|
| Project | `alphatitans-cloud-1017`, a personal Google account on the free trial |
| Zone | `us-central1-a` |
| Control plane | `at-control`: e2-micro, 30 GB pd-standard, Ubuntu 22.04. Runs `cloud.control_app` and `cloudflared-quick` |
| Worker | `at-worker`: e2-standard-8 (8 vCPU, 31 GB RAM), 100 GB pd-balanced, Ubuntu 22.04, **CPU only**. `TERMINATED` whenever there is no work |
| Storage | bucket `gs://alphatitans-cloud-1017-media`: private, public access prevention **enforced**, uniform bucket-level access. Prefixes `inputs/` and `outputs/` |
| Service accounts | `at-control@…` and `at-worker@…`, both `cloud-platform` scope, both `roles/storage.objectAdmin` **on the bucket only** |
| VM control | custom role `atWorkerOperator` (`compute.instances.get/start/stop`), bound to `at-control` **on the `at-worker` instance only**. The worker account has no Compute permission; it powers itself off from inside |
| Public HTTPS | a cloudflared **quick tunnel**, `https://<random>.trycloudflare.com/cloud`. No inbound firewall rule |
| Worker → control plane | `http://10.128.0.3:8000` on the VPC, authenticated with a Google-signed ID token for `at-worker@…` |
| Browser → control plane | the operator token (`AT_CLOUD_API_TOKEN`), in a header only |

### Why Cloud Storage and not Drive

The brief asked for Drive. On a personal Google account it cannot work: a
service account owns no Drive storage, so its uploads fail with
`storageQuotaExceeded`, and the fix (a Shared Drive) exists only on Google
Workspace. Cloud Storage's resumable upload is the same protocol, so
`cloud/gcs.py` subclasses the Drive client and changes only addresses
(`AT_CLOUD_STORAGE=gcs`). Drive is still there for a Workspace account
(`AT_CLOUD_STORAGE=drive` plus a Shared Drive).

### Why a CPU worker

The free trial's GPU quota is 0, and this project never requests quota. An
e2-standard-8 produced the E2E video in about 7 minutes. Switching to a GPU
VM later is a settings change (`AT_GCP_WORKER_INSTANCE`, `AT_GCP_ZONE`), not
a code change.

## Reproducing it

The project was set up interactively with gcloud. These commands produce
the state the table above describes. Replace `PROJECT` and `BUCKET`.

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com iamcredentials.googleapis.com --project=PROJECT
```

```bash
gcloud iam service-accounts create at-control --display-name="Alpha Titans control plane" --project=PROJECT
```

```bash
gcloud iam service-accounts create at-worker --display-name="Alpha Titans worker" --project=PROJECT
```

```bash
gcloud storage buckets create gs://BUCKET --project=PROJECT --location=us-central1 --uniform-bucket-level-access --public-access-prevention
```

```bash
gcloud storage buckets add-iam-policy-binding gs://BUCKET --member=serviceAccount:at-control@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
```

```bash
gcloud storage buckets add-iam-policy-binding gs://BUCKET --member=serviceAccount:at-worker@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
```

```bash
gcloud compute instances create at-worker --project=PROJECT --zone=us-central1-a --machine-type=e2-standard-8 --boot-disk-size=100GB --boot-disk-type=pd-balanced --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --service-account=at-worker@PROJECT.iam.gserviceaccount.com --scopes=cloud-platform --tags=at-worker
```

```bash
gcloud compute instances create at-control --project=PROJECT --zone=us-central1-a --machine-type=e2-micro --boot-disk-size=30GB --boot-disk-type=pd-standard --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --service-account=at-control@PROJECT.iam.gserviceaccount.com --scopes=cloud-platform --tags=at-control
```

```bash
gcloud iam roles create atWorkerOperator --project=PROJECT --file=deploy/cloud/control-plane-role.yaml
```

```bash
gcloud compute instances add-iam-policy-binding at-worker --project=PROJECT --zone=us-central1-a --member=serviceAccount:at-control@PROJECT.iam.gserviceaccount.com --role=projects/PROJECT/roles/atWorkerOperator
```

`iam.serviceAccounts.actAs` turned out **not** to be needed: the control
plane starts the worker with the binding above alone.

### Installing the app on both VMs

The code is not committed, so it was copied as a tarball of the working tree
(without `outputs/`, `.env` or `.git`), unpacked to `/opt/alpha_titans`,
owned by a system user `alpha`. On each VM:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv fonts-dejavu-core
```

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin alpha
```

The worker gets the whole `requirements.txt` (LangGraph, NumPy, SciPy,
faster-whisper and the rest):

```bash
sudo -u alpha python3 -m venv /opt/alpha_titans/.venv && sudo -u alpha /opt/alpha_titans/.venv/bin/pip install -r /opt/alpha_titans/requirements.txt
```

The control plane runs no pipeline, so it gets only the web layer and
Pydantic (for `JobSettings`):

```bash
sudo -u alpha python3 -m venv /opt/alpha_titans/.venv && sudo -u alpha /opt/alpha_titans/.venv/bin/pip install fastapi "uvicorn[standard]" python-multipart pydantic
```

The worker also needs `ffmpeg` (`sudo apt-get install -y ffmpeg`; Ubuntu's
4.4.2) and the app's `.env` with the Groq and Gemini keys at
`/opt/alpha_titans/.env`, mode 0600, owner `alpha`. There is no search key,
so the research agent runs without web search.

Settings go in `/etc/alpha-titans/control.env` and `worker.env` (mode 0600),
filled in from the examples in `deploy/cloud/`. The non-secret values as
deployed:

| Control plane | Worker |
|---|---|
| `AT_HOST=0.0.0.0` (port 8000 is reachable only inside the VPC) | `AT_CLOUD_ROLE=worker` |
| `AT_CLOUD_PUBLIC_URL=http://10.128.0.3:8000` | `AT_CLOUD_PUBLIC_URL` = the same |
| `AT_CLOUD_STORAGE=gcs`, `AT_GCS_BUCKET=BUCKET` | `AT_CLOUD_STORAGE=gcs`, `AT_GCS_BUCKET=BUCKET` |
| `AT_CLOUD_UPLOAD_MODE=relay` (required with gcs) | `AT_CLOUD_POLL_S=15` |
| `AT_CLOUD_WORKER_AUTH=google_id_token` | `AT_CLOUD_WORKER_AUTH=google_id_token` |
| `AT_CLOUD_TOKEN_SOURCE=metadata` | `AT_CLOUD_TOKEN_SOURCE=metadata` |
| `AT_CLOUD_IDLE_GRACE_S=180` | `AT_CLOUD_REQUIRE_GPU=0`, `AT_CLOUD_MIN_FREE_GB=20` |
| | `AT_MUSIC_PROVIDER=lyria`, `AT_MUSIC_PROJECT=PROJECT` (AI-composed music, below) |
| `AT_CLOUD_API_TOKEN` (secret, 24+ characters; admin only since 18 Oct) | no secret of its own: it uses the VM's identity |
| `AT_CLOUD_OPEN_ACCESS=1`, `AT_CLOUD_MAX_JOBS_PER_DAY=20` (no sign-in, below) | |

Units, from `deploy/cloud/`:

```bash
sudo cp /opt/alpha_titans/deploy/cloud/alpha-titans-control.service /opt/alpha_titans/deploy/cloud/cloudflared-quick.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now alpha-titans-control cloudflared-quick
```

on the control plane (cloudflared from Cloudflare's `.deb` release), and

```bash
sudo cp /opt/alpha_titans/deploy/cloud/alpha-titans-worker.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable alpha-titans-worker
```

on the worker. Then stop the worker VM; from then on the control plane
starts it.

## No sign-in (open access)

Since 18 Oct 2026 the page asks for no token (`AT_CLOUD_OPEN_ACCESS=1`).
Anyone with the link can make videos, so three things stand in for a login:

* a job is reached only by its id, 48 random bits; the browser keeps the
  ids of the videos it made (or opened) and lists only those;
* at most `AT_CLOUD_MAX_JOBS_PER_DAY` new videos (20) in any 24 hours, so a
  shared link cannot spend the credit; the admin token is not capped;
* `POST /api/cloud/reconcile` and the full job list still need the token.

For real accounts, put Google sign-in in front with Cloudflare Access on a
named tunnel; that needs a domain.

## AI-composed music (Lyria 3)

Since 18 Oct 2026 the worker has the soundtrack composed by Google's Lyria 3
Pro (`lyria-3-pro-preview`, $0.08 a track) from the Music agent's direction;
see `server/agents/tools/ai_music.py`. It uses the worker's own identity, so
there is no key. Two changes to the project made it possible:

```bash
gcloud services enable aiplatform.googleapis.com --project=PROJECT
```

```bash
gcloud projects add-iam-policy-binding PROJECT --member=serviceAccount:at-worker@PROJECT.iam.gserviceaccount.com --role=roles/aiplatform.user --condition=None
```

Without `AT_MUSIC_PROVIDER=lyria`, or when Lyria refuses or fails, the
code-composed score is used as before.

## Finding the public address

A quick tunnel gets a **new random address every time `cloudflared-quick`
restarts** (a VM reboot included). Restarting `alpha-titans-control` does not
change it. To read the current one:

```bash
gcloud compute ssh at-control --zone=us-central1-a --command="sudo journalctl -u cloudflared-quick --no-pager | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1"
```

For a fixed address, use a named tunnel (`cloudflared.yml.example`) with
Cloudflare Access in front.

## Smoke test, as run

| Step | Check | Result on 17 Oct 2026 |
|---|---|---|
| Control plane up | `curl localhost:8000/healthz` on `at-control` | `{"ok":true,"cloud":true}` |
| Only the cloud is public | `/api/jobs`, `/api/v2/jobs`, `/docs`, `/static/studio.js` | 404 (also a test: `test_the_control_app_serves_only_the_cloud`) |
| Page | `https://…trycloudflare.com/cloud` | asks for the token, then the studio-style Create screen |
| Upload | 54.6 MB clip through the relay, 8 MiB chunks | confirmed chunk by chunk, about 21 s |
| VM starts | first job, worker stopped | `hello ready=True` 51 s after the start request |
| Job runs | the page walks the graph's phases | critic PASS, 852/852 frames, 28.4 s H.264 1080p30 + AAC |
| Outputs | video, thumbnail, YouTube sheet, report, plan, log | in `outputs/<job>/`, downloadable from the page |
| VM stops | idle grace 180 s | worker exits 75, the unit powers it off, `TERMINATED` |

Not yet exercised live: killing the worker mid-render to watch a lease
expire and the job resume. The tests cover it (`test_worker.py`), the live
system has not been made to do it.

## API

| Route | Who | Purpose |
|---|---|---|
| `GET /api/cloud/config` | anyone | enabled? upload mode, chunk size; no secret |
| `GET /api/cloud/options` | anyone | the style, pacing, music… values `JobSettings` accepts |
| `POST /api/cloud/uploads` | operator token | create (or find, by idempotency key) a job and its upload targets |
| `PUT /api/cloud/uploads/{job}/{idx}` | operator token | relay one chunk; `Content-Range` required |
| `GET /api/cloud/uploads/{job}/{idx}` | operator token | how much storage holds |
| `POST /api/cloud/uploads/{job}/complete` | operator token | verify every source, queue, start the worker |
| `GET /api/cloud/jobs?ids=a,b` | anyone (open access) | only the jobs named, which the browser remembers making; the full list needs the admin token |
| `GET /api/cloud/jobs/{job}` | anyone with the job id (open access) | status, progress, outputs, events, and `estimate`: percent done and seconds left, from the timings of recent finished jobs |
| `GET /api/cloud/jobs/{job}/trace` | operator token | the graph trace and its summary, once done |
| `GET /api/cloud/jobs/{job}/outputs/{n}` | operator token | stream one output (Range supported) |
| `POST /api/cloud/jobs/{job}/outputs/{n}/ticket` | operator token | a one-time, 60 s link for the browser's own download manager; `?mode=view` gives a reusable 30 min inline link for `<video>` and `<img>` |
| `GET /api/cloud/download/{ticket}` | the ticket | the file itself; no token in the URL |
| `POST /api/cloud/jobs/{job}/cancel` | operator token | cancel |
| `POST /api/cloud/reconcile` | operator token | one reconcile pass now |
| `POST /api/cloud/worker/{hello,claim,heartbeat,complete,fail,idle}` | worker ID token | the worker's side; VPC only |

## Operating

* **Logs:** `journalctl -u alpha-titans-control -f`; on the worker,
  `journalctl -u alpha-titans-worker -f`. Credentials are redacted before
  anything is logged.
* **Updating code:** copy the changed files into `/opt/alpha_titans`
  (`sudo install -o alpha -g alpha`), then
  `sudo systemctl restart alpha-titans-control`. For the worker, start the
  VM by hand, `sudo systemctl stop alpha-titans-worker` (a plain stop does
  not power it off: only exit code 75 does), install, start the unit again,
  and let it go idle. `run_one.py` is read fresh by each job.
* **Back up the queue:** `sqlite3 /opt/alpha_titans/outputs/_cloud/queue.db
  ".backup /path/queue-$(date +%F).db"`. Keep backups private.
* **Maintenance on the worker:** `sudo systemctl disable --now
  alpha-titans-worker` first.

## Hardening still to do

Found while reading the live project back; none of it has been changed,
because each is a security setting for the owner to decide.

* **SSH is open to the internet.** The default network's `default-allow-ssh`
  allows tcp:22 from `0.0.0.0/0` to both VMs. Narrow it to your own address
  or to IAP's range `35.235.240.0/20`.
* **Port 8000 is open to the whole VPC**, through `default-allow-internal`,
  not just from the worker. Only these two VMs exist, and every route
  authenticates, but a rule from tag `at-worker` to tag `at-control` on
  tcp:8000 is narrower.
* **Inputs are kept forever.** The bucket has no lifecycle rule. A rule
  deleting `inputs/` after, say, 30 days caps storage cost.
* **Quick tunnel.** Fine for testing; for anything lasting, a named tunnel
  with Cloudflare Access, so the operator token is a second lock rather than
  the only one.
