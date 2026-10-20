# Alpha Titans: handoff for the next agent

**As of 19 Oct 2026.** Read this first; it is the shortest accurate picture
of the project, what is live, and the rules the owner (Akansha) has set.
Deeper write-ups live in `docs/cloud/` (listed at the end).

---

## 1. What the project is

Alpha Titans turns raw phone footage into a finished, YouTube-ready video.
A person uploads clips and says what they want; a fixed graph of 18 agents
decides the edit; a video engine renders it; the person downloads the
video, a thumbnail and a YouTube upload sheet. **Nothing uploads to YouTube**:
publishing is always the person's own step.

It runs two ways:

| | Local studio | Cloud studio (live) |
|---|---|---|
| Page | `frontend/index.html` + `studio.js` | `frontend/cloud.html` + `cloud.js`, served at `/cloud` |
| Server | `server/main.py` (FastAPI) | `server/cloud/control_app.py` on a small always-on VM |
| Rendering | the same machine | an on-demand worker VM that switches itself off |

---

## 2. Rules the owner has set (do not break these)

- **`dev-2` is the only branch** (it replaced `main` on 20 Oct 2026). Commit
  or push only when the owner asks. When she does, **attribution is Akansha only**: no AI
  co-author lines in commits or PR descriptions.
- **`pipeline/build_video.py`, `pipeline/*.py` and `server/extras.py` are
  protected.** Change them only with the owner's explicit approval. (Two
  approved changes were made on 19 Oct; see section 8.)
- **Never put secrets** (tokens, API keys, passwords) in code, logs, URLs,
  chat or commits. `.env` and `/etc/alpha-titans/*.env` hold them.
- **Never request cloud quota increases** (GPU or CPU). That is the owner's
  decision.
- **Zero new Python dependencies, no npm, no frontend build step**, unless
  the owner approves.
- **Never fabricate results.** Say what was measured; label estimates.
- The owner writes in Hinglish/English; answer in clear, simple English.

---

## 3. Repository map

```
pipeline/build_video.py      PROTECTED video engine (ffmpeg): probe, measure,
                             plan, cut, join, master. Called as a library.
pipeline/bgm_stdlib.py       code-composed music (fallback when AI music fails)
server/main.py               local app (FastAPI); mounts the cloud routes if enabled
server/config.py             settings, loads the gitignored .env
server/extras.py             PROTECTED: music_stage, master_stage, prepare_track
server/agents/               the 18 agents + LangGraph graph (graph.py)
server/agents/tools/         llm_tools (Groq/Gemini client), select_tools (shot
                             windows), shot_check (Gemini picks shots),
                             ai_music (Lyria), render_tools, asset_library, ...
server/cloud/                cloud system: queue, VM manager, reconciler, worker,
                             GCS store, API, open-access, progress estimate
frontend/cloud.html|js       the cloud page (studio look, no build step)
tests/agents, tests/backend, tests/cloud, tests/frontend
deploy/cloud/                systemd units, env examples, IAM role
docs/cloud/                  findings, ADR, deployment, test report, cost, ...
```

---

## 4. The agent graph (in running order)

`clip_analysis -> (vision || research) -> content_strategy -> story ->
voiceover -> music -> animation -> transition -> caption -> editing_director
-> render -> critic -> [revise back to editing_director, max 2 times] ->
thumbnail -> title -> publishing -> performance_analysis -> learning`

| Agent | Does | Uses |
|---|---|---|
| clip_analysis | measures motion / frozen frames per clip | ffmpeg (engine) |
| vision | describes each clip from 4 frames | Gemini `gemini-3.1-flash-lite` |
| research | web facts; **off** (no search key, no topic) | Tavily/Brave if keyed |
| content_strategy, story, voiceover (text only), music (direction), animation, transition, caption, thumbnail, title (only if none given) | decisions | Groq `openai/gpt-oss-20b`; fallbacks `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `allam-2-7b` |
| editing_director | which clip, order, length; then **shot_check** asks Gemini to pick, for each shot, the best of 6 candidate windows (3-frame strips) | Groq + Gemini + code |
| render | cut, join, captions, music, master | engine + **Lyria 3 Pro** for music |
| critic | technical checks + first-3-second motion only (1 of 10 creative dimensions) | code |
| publishing | checks YouTube limits, writes `youtube.txt`; never uploads | code |
| performance_analysis, learning | report "unavailable" / "nothing learned": no published-video data exists | code |

Provider settings live in `.env` (`AT_LLM_*`, `AT_VISION_*`); one text model
serves all text agents (no per-agent model setting yet).

---

## 5. The cloud system (live)

**Flow:** browser uploads clips in 8 MiB chunks to the control VM, which relays
them to a private Cloud Storage bucket and records a job in a SQLite queue ->
"Create my video" queues it and starts the worker VM -> the worker claims the
job (lease + heartbeats, Google ID-token auth), downloads the clips, runs
`server/cloud/run_one.py` (the existing pipeline in a subprocess), uploads 6
files (video, thumbnail, youtube.txt, output.json, edit_plan.json, run.log),
and powers itself off 3 minutes after the last job (exit code 75 -> systemd
poweroff). The browser downloads through one-time ticket links.

**Live deployment** (Google Cloud project `alphatitans-cloud-1017`, zone
`us-central1-a`, owner's free trial with $300 credit):

| Thing | Value |
|---|---|
| Control VM | `at-control`, e2-micro, internal 10.128.0.3, external 136.111.113.165; services `alpha-titans-control` (uvicorn `cloud.control_app:app`, port 8000) and `cloudflared-quick` |
| Worker VM | `at-worker`, e2-standard-8 (8 CPUs, no GPU), on demand; ephemeral external IP changes on every start; service `alpha-titans-worker` |
| Storage | `gs://alphatitans-cloud-1017-media` (private), prefixes `inputs/`, `outputs/<job>/` |
| Public page | cloudflared quick tunnel, last `https://buttons-carlo-pipes-competing.trycloudflare.com/cloud`. **The URL changes whenever `cloudflared-quick` restarts**; read it with `sudo journalctl -u cloudflared-quick \| grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' \| tail -1` on the control VM |
| Code on VMs | `/opt/alpha_titans` (owner `alpha`, venv `.venv`); settings `/etc/alpha-titans/{control,worker}.env` |

**Access:** no sign-in (`AT_CLOUD_OPEN_ACCESS=1`). Each browser lists only
the job ids it keeps in localStorage (`at.cloud.jobs`); a job id (48 random
bits) is the key to that job. New jobs are capped at 20 per 24 h
(`AT_CLOUD_MAX_JOBS_PER_DAY`). The full job list and `POST /api/cloud/reconcile`
still need the admin token `AT_CLOUD_API_TOKEN` (stored only in control.env).

**Page behaviour:** clips upload as soon as they are added (percent per clip
and total); options are chosen meanwhile; "Create my video" unlocks at 100%
and sends the settings with `POST /uploads/{job}/complete`. Defaults:
Length Auto, **Orientation Horizontal**, **framing "fit"** (mixed clips shown
whole with blurred sides). The production screen shows "N% done / About M
minutes left" from `server/cloud/estimate.py` (medians of recent finished
jobs). Finished screen: player (9:16 frame for vertical videos), stats,
thumbnail, "Download everything", the YouTube sheet.

**AI music:** worker env `AT_MUSIC_PROVIDER=lyria`, `AT_MUSIC_PROJECT=...`;
Vertex AI API enabled; `roles/aiplatform.user` on the worker's service
account (no key; metadata-server identity). `ai_music.py` sends the Music
agent's direction + 3 frames of the cut to `lyria-3-pro-preview`
($0.08/track, from the credit). Lyria refuses ~1 in 4 requests **at random**
(`content_blocked`, not word-related); up to 3 tries, then the code-composed
score. It asks for duration + 4 s because returned lengths vary.

**Stock graphics** (star, arrow, brackets, badge) are **off**
(`AT_STOCK_GRAPHICS=1` re-enables): they were placed at a fixed spot, not on
objects.

**Costs:** worker ~$0.27/h list price (~$0.05/video), control VM free tier,
Lyria $0.08/track. Groq: **free plan** (1,000 requests and 200k tokens per
day per model, about 10 videos/day on the main model). Gemini key: free tier
exists, plan **not confirmed**; on the free tier Google may use the frames.
The free trial ends 90 days after sign-up or when $300 is spent; **the
project then stops** unless upgraded. Trial caps the project at 12 CPUs total.

---

## 6. Operating the live system

- SSH (from the owner's Windows machine, Git Bash):
  `ssh -i ~/.ssh/google_compute_engine -o UserKnownHostsFile=~/.ssh/known_hosts_alpha Akansha@<ip>`
- gcloud is at `C:\Users\Akansha\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin`
  (not on PATH; PowerShell works best). Active account is the owner's
  personal free-trial account; project `alphatitans-cloud-1017`. The
  `samriddhi-production` project is off limits.
- **Deploy a file:** `scp` to `/tmp`, then
  `sudo install -o alpha -g alpha -m 644 /tmp/x /opt/alpha_titans/<path>`.
  Control: `sudo systemctl restart alpha-titans-control` for Python changes
  (static files need no restart; restarting the control does **not** change
  the tunnel URL). Worker: start it (`gcloud compute instances start at-worker
  --zone us-central1-a`), install, done: agent code is loaded fresh per job;
  restart the worker service only for `server/cloud/worker.py`. Use
  `systemctl stop` (not kill) before maintenance; only exit 75 powers off.
- **Before restarting the control,** check no upload is in progress
  (admin `GET /api/cloud/jobs`).
- **Call the API without exposing the token:** on the control VM,
  `sudo bash -c 'set -a; . /etc/alpha-titans/control.env; curl -H "Authorization: Bearer $AT_CLOUD_API_TOKEN" localhost:8000/api/cloud/jobs'`.

---

## 7. Tests (Windows notes)

```bash
cd server && python -m pytest ../tests/cloud -q          # 247 tests
PATH="/c/Users/Akansha/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin:$PATH" python -m pytest tests/agents -q   # 1976 + 2 skipped, ~25 min
PATH="<same ffmpeg dir>:$PATH" python -m pytest tests/backend -q    # 39 + 1 skipped
node --test tests/frontend/                              # 39
```

- **ffmpeg is not on Python's PATH** on this machine: prepend its folder or
  every media test errors with `WinError 2`.
- Run `tests/agents` and `tests/cloud` **separately** (both have a
  `conftest.py`; together they clash at collection).
- `tests/agents/conftest.py` sets `AT_SHOT_CHECK=0` and
  `AT_STOCK_GRAPHICS=1` so older tests keep their behaviour;
  `test_shot_check.py` covers the new defaults.
- `test_graph.py::test_the_strategy_joins_vision_and_research_before_the_director`
  failed once in a full run (asyncio "no running event loop") and passes
  alone and with its file: treat as intermittent.
- Writing files from Bash heredocs mangles `\n` and backslashes; use a
  script file with raw strings, or the editor tools.

---

## 8. Recent changes (committed and pushed on `dev-2`)

- Cloud orchestration end to end (queue, VM lifecycle, reconciler, worker,
  GCS, ID-token auth, `control_app`), deployed live.
- Studio-style `/cloud` page; native downloads; YouTube sheet; vertical
  player; no-flicker previews; background upload with percentages; open
  access with per-browser lists and a daily cap; progress estimate.
- AI music with Lyria (with frames); shot check with Gemini; stock graphics
  off; one graphic per shot; default horizontal + fit.
- **Engine (approved 19 Oct):** `stage_cut` seeks before opening each clip
  (a shot 7.4 min in: 178.5 s -> 16.7 s, pixel-identical); `measure_once`
  decodes each clip once for both motion and freeze detection (identical
  results), with the old two-pass method as fallback.
- Earlier fixes: EditingDirector truncation, whole-call LLM retry budget,
  vision circuit breaker, moved-project download fallback, Tavily auth,
  seeded analytics refused by learning.

Live test jobs worth knowing: `9e797275158b` (old behaviour: hands
everywhere) vs `96f2431daba4` (shot check, same footage);
`35282d591ebd` (first Lyria music); `b5e72b2c68a4` (2 h of footage, ~22 min
before the engine fixes).

---

## 9. Open items and next steps

1. **YouTube analytics (proposed, waiting for the owner):** read-only OAuth
   (`youtube.readonly`, `yt-analytics.readonly`) granted once by the channel
   owner; "paste the YouTube link" on each video page to tie a video to its
   job; daily stats into `analytics_store`; show views, average view
   duration and retention drop-off per video; then let `learning.py` compare
   more decisions (today only pacing) under its existing gates (30 videos,
   12 per cohort, 30 days, T+7d metrics, holdout of 6, 10% effect, human
   approval). Never upload. Never store credentials in code.
2. Narration is written but never spoken (text-to-speech not added; Piper
   would be a new dependency, needs approval).
3. Graphics need object positions from Vision before stock graphics return.
4. Length "Auto" makes ~2 min (1 min vertical) and ignores a length in the
   brief (engine default).
5. The critic measures technical quality only; a visual critic is a
   candidate next step.
6. Hardening for the owner to approve: SSH open to the internet (default
   firewall rule), port 8000 open VPC-wide, no bucket lifecycle rule, quick
   tunnel instead of a named tunnel + domain.
7. Free trial end date: upgrade before 90 days or the project stops.

---

## 10. Further reading

- `docs/cloud/README.md`: index
- `docs/cloud/02-adr.md`: design decisions (+ amendment for what changed live)
- `docs/cloud/03-deployment.md`: how the live system was built and is operated
- `docs/cloud/04-test-report.md`, `05-cost.md`, `06-change-report.md`
- The owner's manager-facing doc (Claude Docs, private):
  https://claude.ai/code/artifact/8a12092f-2e66-411a-b897-086513e03924
