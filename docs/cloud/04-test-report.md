# E. Test report

Run on Windows 11, Python 3.11, ffmpeg 8.1.1 (on `PATH` from its WinGet
folder, which the media tests need), Node 20, on branch `dev-2`. Every
number below is from a run that **completed**; where a run was repeated
after a fix, both are stated.

## Suites

| Suite | Command | Result |
|---|---|---|
| Cloud (new) | `python -m pytest tests/cloud -q` | **225 passed** (36.7 s, 17 Oct, after the studio page and the YouTube sheet) |
| Frontend | `node --test tests/frontend/` | **39 passed, 0 failed** (re-run 17 Oct) |
| Agents, first full run | `python -m pytest tests/agents -q` | **1934 passed, 2 skipped, 1 failed** (15 min 55 s) |
| Agents, the failure re-run after its fix | `python -m pytest tests/agents/test_title.py tests/agents/test_publishing.py -q` | **138 passed** |
| Agents, full re-run after the fix | `python -m pytest tests/agents -q` | **1935 passed, 2 skipped, 0 failed** (22 min 9 s) |
| Backend | `python -m pytest tests/backend -q` | **39 passed, 1 skipped** (6 min 12 s), the 8 moved-project tests included |

The agents and backend suites were not re-run after the last round of
changes; those touched only `server/cloud/`, `frontend/cloud.*` and
`tests/cloud/`, which the cloud suite covers.

### The one failure, and its fix

`tests/agents/test_title.py::test_nothing_in_the_project_publishes_anywhere`
scans the raw source of `server/` and `pipeline/` for any Google upload URL.
The new Drive client legitimately contains Drive's. My first narrowing of the
publishing agent's own marker had also put the joined string into
`publishing.py`, which the original split had deliberately avoided. Both
guards now name **YouTube's** upload path, and that file no longer contains
its own target. A test proves every YouTube spelling the guards were written
for is still caught. No URL was disguised to get past them.

## The brief's fifteen required tests

| # | Requirement | Where | Proven against |
|---|---|---|---|
| 1 | a completed upload creates exactly one job | `test_queue`: one key → one job, 12 concurrent creates → one job; `test_api`: complete ×3 → one start | real queue, real routes |
| 2 | an incomplete upload never enters the queue | `test_queue`, `test_api::test_complete_before_every_byte_is_refused` | real queue, real routes |
| 3 | repeated start requests → one start | `test_vm`: 10 sequential and 10 concurrent threads → `instances.start` once | real VM manager, fake Compute |
| 4 | simultaneous uploads → no duplicate claims | `test_queue`: 20 threads, 5 jobs → each claimed once | real SQLite under real contention |
| 5 | a failed VM start is retried safely | `test_vm`: failure → jobs back to `queued`, backoff, retry; unready VM stopped | real VM manager, fake Compute |
| 6 | VM crash → lease expiry → recovery | `test_queue`: lease expiry, retry, fencing of a zombie worker | real queue |
| 7 | a failed stage resumes without redoing cached work | `test_worker`: retry neither re-downloads nor re-uploads; `run_one` chooses run / resume / already-done from the checkpoint. The graph-level resume itself is covered by the existing agents suite (`test_resume*.py`) | real worker and routes |
| 8 | a failed output upload never completes a job | `test_queue` (five kinds of unverified output refused), `test_worker` (upload failure, md5 mismatch) | real queue and worker |
| 9 | no shutdown while a job is active | `test_vm::test_idle_never_says_shutdown_while_a_job_is_active`, `test_worker::test_the_control_plane_says_wait_while_a_job_is_running` | real VM manager |
| 10 | shutdown after the work and the grace | `test_worker::test_one_activation_drains_the_queue_then_powers_off`, `..._only_after_the_idle_grace` | real worker, real routes |
| 11 | a lost start request is recovered | `test_vm::test_a_queued_job_whose_start_request_was_lost_is_found` | real reconciler |
| 12 | rate limits: no endless retry, no prolonged GPU | `test_worker`: rate-limited → VM off before the retry; real subprocess killed at its deadline; timeouts retried once. LLM call budget: `tests/agents/test_retry_budget.py` | real subprocess, real tree-kill on Windows |
| 13 | hostile names cannot escape | `test_paths`, `test_worker` (`../`, `..\`, `C:\`, `con`, NUL, all-dots) | real code |
| 14 | no credential in logs, responses or assets | `test_security`: redaction of 7 credential shapes; a full production with every log line captured at DEBUG; frontend asset scan; errors return a fixed sentence | real code |
| 15 | the local path still works | `test_security::test_with_cloud_off_the_app_is_unchanged`, plus the agents, backend and frontend suites above | the real app |

## Live, on real Google Cloud

Project `alphatitans-cloud-1017`, as described in the deployment guide. Both
runs uploaded `IMG_0054.MOV` (57,276,988 bytes) through the **public** quick
tunnel URL with the same chunked relay protocol the page uses, then watched
the job through the API.

| | Job `a7168385e472` | Job `3bc59bcb37e7` |
|---|---|---|
| worker at the start | `TERMINATED`; the control plane started it | already running (started by hand to install new worker code) |
| upload | 7 chunks of 8 MiB, each acknowledged by the server, 22 s | the same, 21 s |
| start to ready | `hello ready=True` **51 s** after the start request | n/a |
| worker auth | Google ID token from the metadata server, verified against Google's **live** keys | the same |
| pipeline | critic **PASS 100**, 852/852 frames, 28.4 s, H.264 1080p30 + AAC | critic **PASS 100**, 864/864 frames, 28.8 s, H.264 1080p30 + AAC, -15.3 LUFS |
| queued to completed | about 7 min 50 s | about 4 min 30 s: the analysis came from the worker's caches of the same clip |
| outputs, md5-verified in the bucket | video, thumbnail, report, plan, log | the same plus `youtube.txt`; the trace now carries its summary |
| power off | idle grace 180 s, worker exit **75**, unit powered off, `TERMINATED` | told to shut down at 12:56:36 UTC, 180 s after its last upload; `TERMINATED` by the next reconcile at 12:58:03 |

The second job's `youtube.txt` was read back from the bucket. It lists the
title the operator gave, eleven checks against YouTube's limits (all ok),
and next steps without the local app's "nothing has left this machine" or
`/api/jobs/...` link; the thumbnail is named as it downloads.

### The page, in a browser

* Every screen (sign in, Create, My videos, in production, stopped,
  finished) rendered against the real control app on `localhost` with
  in-memory storage, using the first job's real video and thumbnail: no
  console errors, no sideways scroll at 375 px, the sidebar opens from the
  menu button.
* Through the public URL: the sign-in screen, styled by the studio's own
  stylesheet. The token is the operator's to enter, so nothing past it was
  driven there by me.
* Download tickets: `attachment` with the right type and length; a second
  use of the same ticket is 404. The player's view link answers ranged
  requests with 206.

## End to end on this machine, with fakes for Google

One run on this machine. The real app and cloud routes ran on `localhost`
with **in-memory Drive and Compute** (nothing touched Google), the **real
worker** talked real HTTP to it, and the worker ran the **real pipeline** as
a subprocess. Driven through `/cloud` in a real browser:

| Step | Observed |
|---|---|
| upload | your `IMG_0054.MOV`, 57,276,988 bytes: 7 chunk PUTs, each 200 with the server's acknowledged offset, then `complete` |
| queue and start | job `6d9c915958b3` queued; **one** start requested |
| worker | hello → claim → download → `processing`, with stage progress relayed from the worker's own trace rows (`story · 4 done`, `transition · 8 done`, `render · 11 done`, ...) |
| pipeline | critic **PASS 81.7/65**; 414 of 414 frames; 13.80 s against 13.80 s planned; vision answered this time |
| results | 5 deliverables uploaded and md5-verified (the 19.6 MB video, thumbnail, report, plan, log); completion accepted; the page showed links and the 18-stage trace |
| cleanup | downloaded sources and render media removed; logs, plans and the checkpoint kept |
| power off | after the idle grace the control plane answered `shutdown`; the worker exited **75**; the VM row went to `stopping` |

The video came out 13.8 s against a 15 s request. That is the existing
planner's rule (a shortfall under 2 s is not topped up), not the cloud path.

## Not yet exercised live

The live runs above replaced most of the earlier list (metadata tokens, ID
tokens against live keys, Compute start and get, the systemd poweroff, the
quick tunnel, boot time, and that `actAs` is not needed). Still unproven
against the real thing:

* a worker killed mid-render: lease expiry, retry and checkpoint resume
  (covered by tests against the real queue, not done on the live system);
* the control plane **stopping** a VM that never became ready, and the
  unit's `StartLimitAction=poweroff`;
* Google Drive as the store. The live system uses Cloud Storage, because a
  personal account has no Shared Drives; the Drive client is tested only
  against fakes and one real-socket 308 test;
* the direct browser-to-storage upload (off; relay is required with Cloud
  Storage);
* a named cloudflared tunnel and Cloudflare Access;
* a GPU worker: the free trial's GPU quota is 0.
