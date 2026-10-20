# F. Cost and lifecycle analysis

## As deployed (17 Oct 2026)

The analysis below was written for an L4 worker. What actually runs is a
**CPU** worker, because the free trial's GPU quota is 0:

| Resource | As deployed | Billed while idle? |
|---|---|---|
| Control plane | e2-micro, 30 GB pd-standard, `us-central1-a` | inside the always-free tier (see below) |
| Its external IPv4 | ephemeral, held for as long as the VM runs, which is always | probably yes: **E** per hour |
| Worker | e2-standard-8 (8 vCPU, 31 GB), on demand | no machine charge while `TERMINATED` |
| Worker disk | 100 GB pd-balanced, kept for checkpoints and caches | yes: **D** per GB-month × 100 |
| Worker external IPv4 | ephemeral: released when the VM stops | only while running |
| Storage | one private bucket, Standard class, `us-central1`; no lifecycle rule yet | yes: **S** per GB-month for what is kept |
| Public HTTPS | cloudflared quick tunnel | free |

Measured on the live system, replacing the assumptions below:

| | Measured |
|---|---|
| **B**, start request to worker ready | **51 s** |
| **G**, idle grace | 180 s (`AT_CLOUD_IDLE_GRACE_S`) |
| **Tj**, one 30 s video from one 57 MB clip | about **7 min 45 s** cold, about **4 min 30 s** when the worker's caches already held the clip |

So one cold activation for one video keeps the worker up about
51 s + 7 min 45 s + 180 s ≈ **11.6 minutes**, billed at the e2-standard-8
rate **C**. Read C, D, E and S from the billing console or the pricing
calculator; they were not verified here, so no dollar total is given.

Everything below this line is the original L4 analysis, kept because it
still describes what a GPU worker would cost.

---


Prices marked **verified** were checked on 16-17 Oct 2026 against Google's
pricing pages or consistent third-party mirrors of them. Anything else is a
parameter, because inventing it would make the totals look precise and be
wrong. Check the pricing calculator before budgeting.

## What runs when

| Resource | Runs | Billed while idle? |
|---|---|---|
| Control plane, e2-micro + 30 GB standard disk | **always** | inside the always-free tier in `us-central1`/`us-west1`/`us-east1` (**verified**: 1 non-preemptible e2-micro instance-month, 30 GB-month standard disk) |
| Its external IPv4 address (needed for the cloudflared tunnel's outbound connection) | always | **probably yes**: in-use external IPv4 addresses are charged and the free tier is reported not to cover them. Price: **E** per hour (not verified) |
| In-process reconciler | always, inside the control plane | no extra cost |
| Worker VM (`g2-standard-4` + 1 L4) | **only while there is work**, then the idle grace | no machine-hour charge while `TERMINATED`. Google documents that a stopped VM incurs no CPU charges; on G2 the GPU is part of the machine-type price. Confirm the GPU line on the GPU pricing page before relying on it |
| Worker boot disk | always (kept so checkpoints and caches survive) | **yes**: **D** per GB-month × disk size (not verified) |
| Cloud Scheduler | not used (3 jobs per billing account free, **verified**, if ever wanted as a backup) | |
| Shared Drive storage | always | counts against the Workspace plan's pooled storage |
| Groq, Gemini | per call | unchanged by this design; free tiers as today |

**L4 on-demand: `g2-standard-4` ≈ $0.707 per hour in `us-central1`
(verified)**, which is about **$516 per month** if left running.

## Cost per activation

```
B  = boot-to-ready seconds      (MEASURE on the real VM; assumed 120 below)
G  = idle grace seconds         (AT_CLOUD_IDLE_GRACE_S, default 180)
Tj = one job's processing time  (observed locally: 7-11 min for a 30 s cut)
R  = worker rate                ($0.707/h for the L4)

one activation, N jobs:   ( B + N × Tj + G ) × R
N separate activations:   N × ( B + Tj + G ) × R
saved by batching:        ( N − 1 ) × ( B + G ) × R
```

With B = 120s, G = 180s and Tj = 10 min:

| Jobs | One activation | Separate activations |
|---|---|---|
| 1 | 15 min = **$0.18** | the same |
| 3 | 35 min = **$0.41** | 45 min = $0.53 |
| 10 | 105 min = **$1.24** | 150 min = $1.77 |

Against always-on: **50 jobs a month** at 15 minutes each is 12.5 VM-hours,
about **$8.84**, plus the worker's disk (D × size) and the external IP
(E × 730 h). Always-on is $516 before disk.

## How needless activations are avoided

| Mechanism | Effect |
|---|---|
| Jobs are committed **before** a start is requested | a worker that boots always finds work; no boot-and-find-nothing |
| Single-flight start (`BEGIN IMMEDIATE`) | ten uploads, one `instances.start` |
| Draining the whole queue per activation | one boot amortised over every waiting job |
| Idle grace G = B (ski-rental rule) | idling never costs more than one wasted boot; total within 2× of the optimum |
| `shutdown` decided in one transaction with the queue | no window in which a job is stranded beside a powered-off VM, and none in which the VM idles because of a race |
| Retries back off, with a 15-minute floor for rate limits | the worker powers off instead of waiting out a provider's window on GPU time |
| Starts capped per hour (circuit) | a broken boot cannot loop at L4 prices |
| Reconciler stops unready, orphaned or overstaying VMs | a VM the worker failed to power off is stopped by the control plane |
| `MAX_JOB` and `MAX_ACTIVATION` | bounds on a job that hangs, and on an activation nobody noticed |
| LLM retry budget, 150s per call (this branch) | a provider outage cannot pin the worker for most of an hour, as the 46-minute configured worst case could |

## The bigger lever: the pipeline does not use the GPU

Every encode is `libx264` on the CPU (findings, item 1). On today's code the
L4 is a 4-vCPU machine at GPU prices. Two ways out, either of which is
worth more than any tuning above:

1. **Point the worker at a CPU-optimised VM.** Configuration only. Its
   hourly rate **C** was not verified here. Compare C against R for a
   machine with at least 8 vCPUs, since libx264 scales with cores.
2. **Adopt NVENC** ([proposal](07-nvenc-proposal.md)). This touches the
   protected engine, so it needs your approval. It would let the L4 earn its
   rate.

Until one of those happens, the honest summary is: **the lifecycle stops
the L4 costing money while idle, but not while it works.**

## Missing values

| Symbol | What | Where to find it |
|---|---|---|
| B | boot-to-ready time of the real worker VM | time from `instances.start` to the worker's `hello` in the control plane's log |
| D | persistent disk price per GB-month for the worker's disk type | Compute Engine disk pricing |
| E | external IPv4 price per hour | VPC network pricing |
| C | CPU worker hourly rate, if you switch | Compute Engine VM pricing |
| GPU while stopped | confirm no GPU charge in `TERMINATED` | GPU pricing page |
