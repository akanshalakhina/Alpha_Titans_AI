# Alpha Titans

Alpha Titans turns raw phone footage into a finished, YouTube-ready video.

You upload your clips and say what the video should be about. A team of 18 AI
agents then decides the edit, and an ffmpeg engine renders it. You download
the video, a thumbnail and a YouTube upload sheet. **Nothing is ever posted
to YouTube for you.** Publishing is always your own step.

There are two ways to run it:

| | Local studio | Cloud studio (live) |
|---|---|---|
| Who it is for | one person, on their own computer | anyone with the link, no sign-in |
| Page | `frontend/index.html` + `studio.js` | `frontend/cloud.html` + `cloud.js`, at `/cloud` |
| Server | `server/main.py` (FastAPI) | `server/cloud/control_app.py`, on a small always-on VM |
| Rendering | the same computer | a worker VM that starts on demand and switches itself off |

---

## Contents

1. [What happens when you make a video](#1-what-happens-when-you-make-a-video)
2. [The agents and the models they use](#2-the-agents-and-the-models-they-use)
3. [The cloud studio](#3-the-cloud-studio)
4. [Repository layout](#4-repository-layout)
5. [Running it locally](#5-running-it-locally)
6. [Settings (environment variables)](#6-settings-environment-variables)
7. [Tests](#7-tests)
8. [Deployment](#8-deployment)
9. [Costs](#9-costs)
10. [Known limits](#10-known-limits)
11. [Troubleshooting](#11-troubleshooting)
12. [Further reading](#12-further-reading)

---

## 1. What happens when you make a video

1. **Upload.** Clips start uploading the moment you add them. Each clip shows
   its own percentage, and there is a total for all of them.
2. **Choose options.** While the upload runs, you pick length, orientation
   (horizontal by default), framing, music and the brief.
3. **Create.** "Create my video" unlocks at 100%. The job is queued and the
   agents start working.
4. **Watch progress.** The page shows "N% done, about M minutes left". The
   estimate comes from how long recent videos took.
5. **Download.** You get the finished video, a thumbnail, `youtube.txt`
   (title, description, tags) and the edit plan. "Download everything"
   saves all of them to your computer.

---

## 2. The agents and the models they use

The agents run as a fixed [LangGraph](https://github.com/langchain-ai/langgraph)
graph, in this order:

```
clip_analysis -> (vision || research) -> content_strategy -> story -> voiceover
-> music -> animation -> transition -> caption -> editing_director -> render
-> critic -> [revise: back to editing_director, at most 2 times]
-> thumbnail -> title -> publishing -> performance_analysis -> learning
```

| Agent | What it does | What it uses |
|---|---|---|
| clip_analysis | Measures motion and frozen frames in every clip. | ffmpeg (the engine) |
| vision | Describes what each clip shows, from 4 frames. | Gemini `gemini-3.1-flash-lite` |
| research | Looks up web facts. Off unless a search key is set. | Tavily / Brave (optional) |
| content_strategy, story, voiceover, music, animation, transition, caption, thumbnail, title | Make the creative decisions: angle, story order, narration text, music direction, captions, thumbnail and title. | Groq `openai/gpt-oss-20b`, with fallback models |
| editing_director | Picks which clip goes where and for how long. Then **shot check** shows Gemini 6 candidate moments per shot (3 frames each) and uses the one where the subject is clearly visible. | Groq + Gemini + code |
| render | Cuts, joins, adds captions and music, and masters the sound. | ffmpeg engine + **Lyria 3 Pro** for AI-composed music |
| critic | Checks the result technically (length, sound level, frames, first 3 seconds). Can send the edit back for a revision. | code |
| publishing | Checks YouTube's limits and writes `youtube.txt`. Never uploads. | code |
| performance_analysis, learning | Learn from how published videos perform. They do nothing yet, because there is no published-video data. | code |

**If a model is unavailable,** each agent falls back to a rule-based path and
marks its result as `degraded`. The video still gets made.

**Music:** Lyria gets the music direction and 3 frames of the cut. If Lyria
refuses (about 1 in 4 requests, at random), it is retried up to 3 times. After
that, a score composed in code (`pipeline/bgm_stdlib.py`) is used.

**Stock graphics** (stars, arrows, brackets) are off by default because they
were placed at a fixed spot rather than on objects. To turn them back on, set
`AT_STOCK_GRAPHICS=1`.

---

## 3. The cloud studio

```
Browser ──8 MiB chunks──> Control VM (always on, e2-micro)
                            │  SQLite job queue, upload relay, progress, download links
                            ├──> Cloud Storage bucket (private): inputs/, outputs/<job>/
                            └──> starts Worker VM (e2-standard-8, on demand)
                                     claims the job, downloads the clips, runs the
                                     18 agents + engine, uploads 6 files, then
                                     powers itself off 3 minutes after the last job
```

- **No sign-in.** Each browser keeps its own list of job ids. A job id is a
  48-bit random key, and only someone who has it can see that job.
- **Limit:** at most 20 new videos per 24 hours (`AT_CLOUD_MAX_JOBS_PER_DAY`).
- **Admin token:** the full job list and the reconcile endpoint still need
  the admin token (`AT_CLOUD_API_TOKEN`). That token is kept only in the
  control VM's settings file.
- **Worker security:** the worker proves who it is with a Google ID token,
  not a stored key. Jobs use leases and fencing tokens, so a crashed worker
  never corrupts a job.
- **Downloads** go through one-time ticket links. The bucket is never public.
- **Public address:** the page is served through a Cloudflare quick tunnel.
  Its address **changes whenever the tunnel restarts**.
  [docs/cloud/03-deployment.md](docs/cloud/03-deployment.md) explains how to
  read the current address.

Design decisions are in [docs/cloud/02-adr.md](docs/cloud/02-adr.md).

---

## 4. Repository layout

```
pipeline/build_video.py   the video engine (ffmpeg): probe, measure, plan, cut, join, master
pipeline/bgm_stdlib.py    code-composed music (fallback for AI music)
server/main.py            local app (FastAPI): API + studio page on one port
server/config.py          settings; reads .env
server/extras.py          music stage, mastering, track preparation
server/agents/            the 18 agents and the graph (graph.py)
server/agents/tools/      llm_tools (Groq/Gemini client), select_tools, shot_check,
                          ai_music (Lyria), render_tools, asset_library, ...
server/cloud/             cloud system: queue, VM manager, reconciler, worker,
                          storage, API, progress estimate, control_app
frontend/                 plain HTML/CSS/JS, no build step
tests/                    agents, backend, cloud, frontend
deploy/cloud/             systemd units, example settings files, IAM role
docs/                     AGENT_HANDOFF.md (start here) and docs/cloud/
```

---

## 5. Running it locally

**You need:**

- Python 3.9 or newer (developed on 3.11).
- ffmpeg 5 or newer, with ffprobe. ffmpeg is not a pip package, so install it
  with your system's package manager:

```bash
winget install Gyan.FFmpeg                        # Windows
brew install ffmpeg                               # macOS
sudo apt update && sudo apt install -y ffmpeg     # Debian / Ubuntu
```

**Install and start:**

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS / Linux: cp .env.example .env
run.bat                           # macOS / Linux: ./run.sh
```

Then open **http://127.0.0.1:8000**.

`run.sh` / `run.bat` check for Python, ffmpeg and the dependencies before
starting, and tell you exactly what is missing.

Other ways to start it:

```bash
python -m uvicorn main:app --app-dir server --reload   # reloads when you edit code
docker compose up --build                              # ffmpeg comes with the image
```

**Is it healthy?**

```bash
curl -s http://127.0.0.1:8000/api/health       # ffmpeg, ffprobe, pipeline: true
curl -s http://127.0.0.1:8000/api/v2/health    # reasoning_enabled, model, agent_count: 18
```

---

## 6. Settings (environment variables)

All settings are optional and go in `.env` (it is git-ignored, so **never
commit it**). Real environment variables override `.env`.
`.env.example` lists every setting with comments.

| Variable | What it does |
|---|---|
| `AT_HOST`, `AT_PORT` | Address and port (default `127.0.0.1:8000`). |
| `AT_FFMPEG`, `AT_FFPROBE` | Full paths, if ffmpeg is not on `PATH`. |
| `AT_LLM_API_KEY` | Turns the text agents on. Without it, every agent uses its rule-based fallback. |
| `AT_LLM_PROVIDER`, `AT_LLM_BASE_URL`, `AT_LLM_MODEL` | Which text model to use. For Groq: provider `openai`, Groq's OpenAI-compatible URL, model `openai/gpt-oss-20b`. |
| `AT_LLM_FALLBACK_MODELS` | Comma-separated models to try if the main one fails. |
| `AT_VISION_API_KEY`, `AT_VISION_PROVIDER`, `AT_VISION_BASE_URL`, `AT_VISION_MODEL` | The vision model (Gemini). Also used by shot check. |
| `AT_SEARCH_PROVIDER`, `AT_SEARCH_API_KEY` | Turns on web research (`tavily`, `brave`). |
| `AT_MUSIC_PROVIDER=lyria`, `AT_MUSIC_PROJECT` | AI music through Vertex AI. This needs a Google Cloud identity with `roles/aiplatform.user`. |
| `AT_SHOT_CHECK=0` | Turns off shot check (falls back to the busiest window). |
| `AT_STOCK_GRAPHICS=1` | Turns the stock graphics back on. |

The cloud settings (`AT_CLOUD_*`, `AT_GCP_*`) are documented in
[deploy/cloud/control.env.example](deploy/cloud/control.env.example) and
[deploy/cloud/worker.env.example](deploy/cloud/worker.env.example).

---

## 7. Tests

```bash
python -m pytest tests/agents -q      # ~2,000 tests, about 25 minutes
python -m pytest tests/backend -q
cd server && python -m pytest ../tests/cloud -q
node --test tests/frontend/
```

- **Run `tests/agents` and `tests/cloud` separately.** Each folder has its
  own `conftest.py`, and running them together makes the test collection
  clash.
- **ffmpeg must be on the `PATH` that Python sees.** If it is not, every
  media test fails with `WinError 2` or `FileNotFoundError`. On Windows,
  add the WinGet ffmpeg `bin` folder to `PATH` first.
- **CI:** GitHub Actions runs the suite on every push and pull request
  (`.github/workflows/tests.yml`).

---

## 8. Deployment

The live system runs on Google Cloud (Compute Engine + Cloud Storage +
Vertex AI). [docs/cloud/03-deployment.md](docs/cloud/03-deployment.md) covers
it step by step: creating the VMs, service accounts, bucket, systemd units,
tunnel, and how to update code.

In short:

- **Control VM:** runs `uvicorn cloud.control_app:app` as
  `alpha-titans-control`, plus `cloudflared-quick` for the public address.
- **Worker VM:** runs `alpha-titans-worker`. When it exits with code 75,
  systemd powers the VM off, so it costs nothing while idle.
- **Updating code:** copy the changed files to `/opt/alpha_titans`. Restart
  the control service for Python changes. Static files need no restart.
  Agent code is loaded fresh for every job.

---

## 9. Costs

| Item | Cost |
|---|---|
| Worker VM (e2-standard-8, only while rendering) | about $0.27/hour, about $0.05 per video |
| Control VM (e2-micro) | free tier |
| Lyria music | $0.08 per track |
| Groq text model | free plan: 1,000 requests and 200k tokens per day per model (about 10 videos/day) |
| Gemini vision | free tier available; paid plan not confirmed |

The Google Cloud project runs on a free trial ($300 credit for 90 days). **When
the trial ends the project stops** unless it is upgraded. Details are in
[docs/cloud/05-cost.md](docs/cloud/05-cost.md).

---

## 10. Known limits

- **Narration is not spoken yet.** The narration text is written, but no
  text-to-speech is used.
- **Learning has no data yet.** It needs statistics from published videos.
  A read-only YouTube Analytics connection is proposed.
- **Length "Auto"** makes about 2 minutes (1 minute for vertical) and
  ignores a length written in the brief.
- **The critic checks technical quality only,** not whether the video looks
  good.
- **Stock graphics are off** until the vision agent can tell where objects are.
- **The public address changes** whenever the tunnel restarts. A named
  tunnel with your own domain would fix this.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| "ffmpeg is not on PATH" | Install ffmpeg, or set `AT_FFMPEG` / `AT_FFPROBE` to full paths. |
| `reasoning_enabled: false` | `AT_LLM_API_KEY` is not set. The agents run on fallbacks. |
| Groq `429` errors | The free plan's daily or per-minute limit was reached. Read the response body for which one. Run one video at a time. |
| Gemini `503` | Google's side is busy. Vision and shot check fall back automatically. |
| Music is the code-composed score | Lyria refused 3 times or is not set up (`AT_MUSIC_PROVIDER`, `AT_MUSIC_PROJECT`). |
| Cloud page link stopped working | The tunnel restarted and has a new address. See section 3. |
| Very slow | Encoding is CPU-bound. Long clips take longest to measure and cut. Use Draft quality to check a cut first. |

---

## 12. Further reading

- [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md): the shortest accurate
  picture of the project. Start here.
- [docs/cloud/README.md](docs/cloud/README.md): index of the cloud
  write-ups (findings, design decisions, deployment, test report, cost,
  change report).
