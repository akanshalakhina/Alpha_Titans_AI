# Alpha Titans — Content Engine

Turn a folder of raw build footage into a finished, captioned, scored and
**verified** cut, from a browser.

This folder is the whole application. Copy it to another machine, install the
documented dependencies, run one script, open one URL.

---

## 1 · Project overview

Upload raw LEGO footage — builds and assembly, Technic, mechanisms, robotics,
experiments, close-ups, demonstrations — in any combination, and the engine
combines it into **one** finished video. It does what an editor would do, in
the order an editor would do it: look at every clip, find where the movement
actually is, choose the windows worth keeping, decide the running order, render
them onto a single canvas with the captions burnt in, crossfade between them,
lay audio against the real cut points, and then check the finished file rather
than assume it came out right.

**It handles the full range of LEGO shot types, and that is a design
constraint, not an afterthought.** A LEGO reel is not one kind of shot: hands
assembling a frame move a lot, a Technic gearbox turning over moves steadily, a
close-up of a finished model barely moves at all, and a demonstration can be
almost still while being the whole point of the video. The planner therefore
measures every clip against *the batch it arrived in* rather than against a
fixed motion figure — an absolute "below X is dead" cut-off would quietly throw
away your close-ups and your demonstrations, which it used to do (see §12).
Mixed orientations and mixed resolutions in the same batch are expected and are
fitted onto one canvas, blur-padded when a clip does not match it.

The work is owned by **five named agents** — Vision, Editor, Animation, Merge
and Music — and the UI groups the run and every decision under the agent that
made it. Only one of them calls a model; the badges say which. See
[§2 · The five agents](#the-five-agents).

> **What "AI" means here, precisely — there are two layers, and they are
> different.**
>
> **`server/vision.py` is a real vision model**, off by default. Set
> `AT_VISION_API_KEY` and it samples real frames from every clip with ffmpeg
> and sends them to a vision-capable API, which reports what the build is,
> close-up vs wide, assembly vs demonstration vs reveal, where in a build the
> shot sits, interesting moments, a hook score, duplicates across clips, and a
> caption where it is confident. Nothing in that file is canned; with no key it
> reports itself unavailable and the build carries on.
>
> **`server/director.py` is rule-based, and always runs.** It reads the vision
> result (when there is one) together with the measurements and decides which
> clips survive, their running order, where each is trimmed, how long each runs,
> which transition goes on each individual cut, the crossfade length and the
> music mood.
>
> Every decision from either layer lands in `edit_plan.json` with the reason it
> was chosen, is labelled in the UI as **saw it** / **decided** / **yours**, and
> genuinely drives the render. With no key the Director works from motion, dead
> air and recording timestamps alone — it has no idea a gearbox is a gearbox —
> so it writes no captions and sets no speed, and says so in the plan rather
> than pretending.

The **pipeline was not rewritten for this app.** `pipeline/build_video.py` is
still the source of truth for probing, motion measurement, cut planning,
segment rendering and transitions. The server shells out to it one stage at a
time — exactly as it is run by hand from a terminal — and reads its
`manifest.json` for everything the UI displays.

Two things follow from that, and they are the point of the design:

- **Nothing in the UI is mocked.** Clip durations and dimensions are ffprobe
  output. Motion figures are the pipeline's own measurements. The cut sheet is
  its plan. Progress is real stage transitions. The log is the pipeline's own
  stdout. There is no sample data and no simulated progress anywhere.
- **Where a capability does not exist, the UI says so.** See
  [Known limitation](#9--known-limitation-voice-transcription).

### The flow

```
   1 Upload Clips  →  2 Customize  →  3 AI Processing  →  4 Final Video
   drag in footage   script, music,   10 real stages,      player, download,
   reorder, inspect  speed, motion,    live status and     verification,
                     resolution        timing              cut sheet
```

---

## 2 · Architecture

### The five agents

The work is split into **five named agents**, each owning one job. An agent is
not a separate process or a separate model — it is a named owner for a group of
pipeline stages, the decisions those stages make, and the reasons behind them.
The split is declared once in `server/adapter.py` (`AGENTS` and `STAGE_AGENT`)
and everything else reads it from there: `/api/health` publishes it, the UI
groups the run and tags every decision by it, and the edit plan records which
agent made each call. Rename an agent or move a stage between agents in that
one table and the whole app follows.

| Agent | Owns | Stages | Model? |
|---|---|---|---|
| **Vision Agent** | Looks at real frames and reports what each clip actually shows | `vision` | **yes** — a real vision API |
| **Editor Agent** | Chooses which clips make the cut, their order, and where each is trimmed | `probe` · `measure` · `plan` · `direct` | no — measured + rule-based |
| **Animation Agent** | Picks a transition for every individual cut, its length, and the motion feel | `join` | no — rule-based |
| **Merge Agent** | Renders each segment onto one canvas and assembles them into a single file | `sheets` · `cut` · `master` | no — deterministic render |
| **Music Agent** | Scores the cut against its real transition marks and handles the source audio | `music` | no — synthesis + DSP |

**Only the Vision Agent calls a model.** The other four are deterministic, and
the UI says so on each one's badge rather than letting you assume otherwise.
That is the honest description: four of the five are good engineering with
their reasoning written down, not language models.

They are not independent — they run in a fixed order and hand work forward:

```
 Vision Agent ──► Editor Agent ──► Animation Agent ──► Merge Agent ──► Music Agent
  what each        which clips      which transition     one file       a score cut
  clip shows       survive, in      on each cut, and     on one         to the real
  (or nothing,     what order,      how long it runs     canvas         transition
  with no key)     trimmed where                                        marks
                          │                                                  │
                          └────────── every call, with its reason, ──────────┘
                                          into edit_plan.json
```

The Editor Agent is the one that consumes the Vision Agent's output: it feeds
what the model saw into the same rule-based Director that runs without it. With
no API key the Vision Agent reports itself unavailable, the Editor Agent works
from motion, dead air and recording timestamps alone, and the build completes —
that fallback is a supported mode, not a degraded one.

### The stack under the agents

```
Browser (vanilla ES modules, no build step)
    │  fetch + XHR (uploads)  ·  EventSource (live stage + log stream)
    ▼
FastAPI  (server/main.py)              ── 30 routes, no business logic
    │
    ├── adapter.py   the AGENTS registry and the STAGE_AGENT map;
    │                builds the exact CLI call for each pipeline stage,
    │                maps UI settings → CLI args + env overrides,
    │                streams the subprocess's stdout back line by line
    │
    ├── jobs.py      job store, stage runner thread, weighted progress,
    │                event log, state persisted to outputs/<id>/job.json
    │
    ├── vision.py    the VISION AGENT. Real frames -> a real vision model
    │                over HTTP. No API key = disabled, app runs as before.
    │
    ├── director.py  the EDITOR AGENT's judgement, and the ANIMATION
    │                AGENT's: reads the vision result and the measurements
    │                and decides what the pipeline does not - a transition
    │                per cut, the crossfade length, the mood - each with
    │                its reason, all into edit_plan.json
    │
    ├── extras.py    the MUSIC AGENT's score stage and source-audio rebuild,
    │                the MERGE AGENT's master/verify stage, plus
    │                script → captions.json and voice-note storage
    │
    └── config.py    every path resolved relative to this folder,
                     overridable by .env or real env vars
    │
    │  subprocess, one per stage
    ▼
pipeline/build_video.py   ← unchanged logic, the source of truth
pipeline/beatbed.py       ← music engine (numpy/scipy)
pipeline/bgm_stdlib.py    ← music engine (standard library only)
    │
    ▼
ffmpeg / ffprobe
```

**Why subprocesses.** A render is minutes of CPU. Running each stage as its own
process keeps it off the event loop, lets the server stream real stdout instead
of inventing progress, and means the pipeline's own resumability works — re-run
a stage and it skips finished work.

### How UI settings reach the pipeline

`build_video.py` was given a **21-line additive change**: four constants and the
canvas now read an environment override and **fall back to exactly their
previous values**. No logic was touched, and the command line behaves
identically when nothing is set (verified: `T=0.4 CRF=18 PRESET=fast`, same
XFADES list). `pipeline/build_video.py.bak` is the pre-change copy, for diffing.

| UI control | Mechanism |
|---|---|
| Resolution | `AT_CANVAS` (swapped for vertical) — segments render *at* target size, never upscaled afterwards |
| Quality | `AT_CRF` — draft 24 / standard 20 / high 18 / master 15 |
| Animation & transitions | `AT_XFADES` — style presets map to real ffmpeg xfade names |
| Transition length | `AT_XFADE_T` |
| Orientation | `--orient` |
| Target length | `--target` |
| Speed | post-join retime in the master stage; the transition marks are rescaled by the same factor so the score stays in sync |
| Music: generate | `beatbed.py` (beat/energetic/cinematic) or `bgm_stdlib.py` (lofi/calm/bright/upbeat), scored against the real transition marks |
| Music: library | the uploaded track looped/trimmed to length and faded |
| Music: silent | a silent AAC track |
| Running order | `--sequence hook\|recorded\|uploaded` — see below |
| Original audio | rebuilt in `extras.source_audio` from the planner's in/out points |
| Clip order | files are stored with a numeric prefix, so the pipeline's sorted glob follows the drag order |

---

## 3 · Folder structure

```
alpha_titans_app/
├── frontend/              the UI — no build step, no bundler, no CDN
│   ├── index.html         the four steps + the builds drawer
│   ├── styles.css         one theme, one spacing scale
│   └── app.js             all of the client logic (ES module)
├── server/
│   ├── main.py            FastAPI: REST + SSE + static file serving
│   ├── jobs.py            job store, stage runner, progress, event log
│   ├── adapter.py         the AGENTS registry; invokes build_video.py
│   ├── vision.py          the Vision Agent — real frames to a real model
│   ├── director.py        Editor + Animation judgement, and every reason
│   ├── extras.py          Music Agent score, Merge master, captions, voice
│   └── config.py          portable paths, .env reader, limits
├── pipeline/              the EXISTING pipeline, vendored unchanged
│   ├── build_video.py     probe · measure · plan · sheets · cut · join
│   ├── build_video.py.bak the pre-change copy, for diffing
│   ├── beatbed.py         full-kit music engine (needs numpy + scipy)
│   └── bgm_stdlib.py      standard-library music engine (no deps)
├── uploads/               raw clips, per job: uploads/<job_id>/
│   └── <job_id>/_voice/   voice notes
├── outputs/               builds, per job: outputs/<job_id>/
│   ├── manifest.json      the pipeline's own state
│   ├── edit_plan.json     every decision and why it was made
│   ├── vision.json        what the vision model saw (only if it ran)
│   ├── captions.json      written between plan and cut
│   ├── job.json           job state, survives a restart
│   ├── parts/ measure/ look/   pipeline working folders
│   ├── _posters/          cached clip thumbnails for the UI
│   └── <NAME>_<orient>.mp4     the finished video + thumbnail.jpg
├── data/music/            the music library (uploaded tracks)
├── requirements.txt
├── .env.example           copy to .env to change anything
├── run.sh / run.bat       start scripts, with dependency preflight
├── Dockerfile             includes ffmpeg — the app is useless without it
├── docker-compose.yml     uploads/outputs/music bind-mounted
└── README.md
```

Nothing outside this folder is read or written. `uploads/`, `outputs/` and
`data/music/` are the only directories that grow.

---

## 4 · Prerequisites

| | Version | Notes |
|---|---|---|
| **Python** | 3.9+ | developed and tested on 3.11 |
| **ffmpeg** | 5.0+ | tested on 6.1.1 — **required** |
| **ffprobe** | ships with ffmpeg | **required** |
| numpy + scipy | any recent | *optional* — enables the richer music engine |

ffmpeg is not a pip package:

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y ffmpeg
# macOS
brew install ffmpeg
# Windows
winget install Gyan.FFmpeg
```

Check: `ffmpeg -version` and `ffprobe -version` must both print.

Without numpy/scipy the app still runs — it falls back to `bgm_stdlib.py`,
which is pure standard library. `GET /api/health` reports exactly which
capabilities are live, and the Builds drawer shows it.

**Disk.** Renders are not small. A 1080p/high two-minute cut and its working
files run to a few hundred MB per build.

---

## 5 · Installation

```bash
cd alpha_titans_app

# a virtual environment is recommended but not required
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Optionally `cp .env.example .env` and edit. With no `.env` the defaults in
[Environment variables](#6--environment-variables) apply.

### Windows, exactly — from the ZIP to the first video

Open **PowerShell** and paste one block at a time.

```powershell
# 1. unzip it wherever you keep projects, and go in
cd $HOME\Downloads
Expand-Archive .\alpha_titans_app_deploy.zip -DestinationPath .
cd .\alpha_titans_app

# 2. ffmpeg — the app cannot render a frame without it
winget install Gyan.FFmpeg
#    close and reopen PowerShell here so the new PATH is picked up, then:
ffmpeg -version
ffprobe -version

# 3. python packages (a venv is optional; this is the plain route)
python -m pip install -r requirements.txt

# 4. start it
.\run.bat
```

Then open <http://127.0.0.1:8000> and work through the four steps. `Ctrl+C`
in that window stops the server.

`run.bat` refuses to start and tells you which one is missing if python,
ffmpeg, ffprobe or the Python dependencies are not there — it does not fail
later mid-render.

Two things that trip people up on Windows:

- **`ffmpeg is not recognized`** after installing it — the PATH is only read
  when a terminal starts. Close PowerShell, open a new one, `cd` back and
  retry. If it still fails, set `AT_FFMPEG` and `AT_FFPROBE` in `.env` to the
  full paths of the two `.exe` files.
- **`python is not recognized`** — install Python from python.org (not the
  Microsoft Store stub) and tick *Add python.exe to PATH* in the installer.

To change the port: `$env:AT_PORT=9000; .\run.bat`. To demo from another
machine on the same network: `$env:AT_HOST="0.0.0.0"; .\run.bat`.

---

## 6 · Environment variables

All optional. A `.env` next to this README is read at startup; real environment
variables always win over it. Relative paths resolve against this folder.

| Variable | Default | What it does |
|---|---|---|
| `AT_HOST` | `127.0.0.1` | bind address — `0.0.0.0` to accept from the network |
| `AT_PORT` | `8000` | port |
| `AT_FFMPEG` | `ffmpeg` | full path if it is not on `PATH` |
| `AT_FFPROBE` | `ffprobe` | full path if it is not on `PATH` |
| `AT_UPLOADS_DIR` | `uploads` | raw clips and voice notes |
| `AT_OUTPUTS_DIR` | `outputs` | builds and finished videos |
| `AT_MUSIC_DIR` | `data/music` | the music library |
| `AT_FRONTEND_DIR` | `frontend` | static files |
| `AT_PIPELINE_DIR` | `pipeline` | where `build_video.py` lives |
| `AT_MAX_CLIP_MB` | `2048` | per-clip upload cap |
| `AT_MAX_TRACK_MB` | `120` | per-track upload cap |
| `AT_MAX_VOICE_MB` | `40` | per-voice-note cap |
| `AT_CORS_ORIGINS` | *(empty)* | comma-separated; only needed if the UI is hosted elsewhere |

The pipeline's own overrides (`AT_CANVAS`, `AT_CRF`, `AT_XFADES`,
`AT_XFADE_T`) are set **per build** by the server from the UI settings — do not
set them in `.env` or you will pin every build to the same values.

---

## 7 · Running it

### Local development

The server serves the API **and** the UI on one port, so there is nothing
separate to start for the frontend and no CORS to configure.

```bash
./run.sh                    # macOS / Linux  → http://127.0.0.1:8000
run.bat                     # Windows
```

`run.sh` checks for python, ffmpeg, ffprobe and the Python dependencies before
it starts, and tells you exactly what is missing rather than failing later
mid-render.

Other ways in, all equivalent:

```bash
AT_PORT=9000 ./run.sh                                    # different port
AT_HOST=0.0.0.0 ./run.sh                                 # reachable on the LAN
python3 -m uvicorn main:app --app-dir server --reload     # auto-reload while editing
python3 server/main.py                                    # plain python
```

`--reload` watches `server/`; the frontend is plain static files, so a browser
refresh is enough after editing `frontend/`.

Then open **http://127.0.0.1:8000** and work through the four steps.

### Checking it is healthy

```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
```

`capabilities` should show `pipeline: true`, `ffmpeg: true`, `ffprobe: true`.
`beat_engine: false` just means numpy/scipy are absent and the standard-library
music engine will be used. `transcription: false` is expected — see
[section 9](#9--known-limitation-voice-transcription).

### API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | capabilities, the stage catalogue, option lists, limits |
| GET/POST | `/api/jobs` | list / create |
| GET/PATCH/DELETE | `/api/jobs/{id}` | snapshot / update settings / remove |
| GET/POST | `/api/jobs/{id}/clips` | list / upload (multiple) |
| DELETE | `/api/jobs/{id}/clips/{name}` | remove one clip |
| POST | `/api/jobs/{id}/clips/order` | reorder (rewrites the numeric prefixes) |
| GET | `/api/jobs/{id}/clip/{name}` | stream a source clip |
| GET | `/api/jobs/{id}/poster/{name}` | cached poster frame for a clip |
| POST | `/api/jobs/{id}/script` · `/script-file` | paste / upload the script |
| POST/GET | `/api/jobs/{id}/voice` · `/voice-file` | store / play a voice note |
| GET/POST | `/api/music` · DELETE/GET `/api/music/{name}` | the music library |
| POST | `/api/jobs/{id}/start` · `/cancel` · `/regenerate` | run control |
| GET | `/api/jobs/{id}/events` | **SSE**: stage transitions + pipeline stdout |
| GET | `/api/jobs/{id}/vision` · `/plan` | what the model saw · the edit plan |
| POST/GET | `/api/jobs/{id}/notes` | operator notes / the verified output figures |
| GET | `/api/jobs/{id}/file/{name}` · `/download` | artefacts |

All 30 are exercised by a sweep that reads the route list out of `main.py`
rather than a hand-written list, so a route added later shows up as untested
instead of quietly passing. Current result: **30 declared, 30 exercised, 29
returning 2xx** — the one non-2xx is `GET /notes` before a build has finished,
which is a correct 404.

---

## 8 · How the video pipeline works

Ten stages, each owned by one of the five agents. Six are the pipeline's own,
invoked unchanged; one is the optional vision pass; one is the edit planner;
two are the finishing pass. The UI shows all ten grouped under their agent,
with real status, elapsed time and the streamed stdout of each.

| # | Stage | Agent | What actually runs |
|---|---|---|---|
| 1 | **Clip Probe** | Editor | `build_video.py probe` — ffprobe every upload: rotation, fps, codec, audio. Drops iOS raw/edited duplicates. Chooses the canvas from whichever orientation dominates by total duration. |
| 2 | **Motion Analysis** | Editor | `build_video.py measure` — `freezedetect` + `signalstats` YDIF per clip, to find dead air and where the real movement is. Records both how *much* changes (`mean_ydif`) and how *often* anything changes at all (`live_share`). The slowest stage, and the one everything downstream depends on. |
| 3 | **Vision Analysis** | **Vision** | `vision.analyse` — samples real frames from every clip with ffmpeg and sends them to a vision-capable model over HTTP. Returns what the build is, close-up vs wide, assembly vs demonstration vs reveal, where in a build it sits, interesting moments, a hook score, duplicates across clips and a caption where it is confident. **Skipped entirely when `AT_VISION_API_KEY` is unset** — that fallback is a supported mode. |
| 4 | **Cut Planner** | Editor | `build_video.py plan` — scores candidate windows on measured motion, allocates the length budget across clips, and sets the running order (liveliest clip opens; the rest follow your upload order). |
| 5 | **Edit Plan** | Editor | `director.direct` — turns the measurements into the decisions the planner does not make: a transition chosen per cut from the pair of shots it joins, the crossfade length, and the music mood. Records every decision with its reason to `edit_plan.json`. |
| 6 | **Contact Sheets** | Merge | `build_video.py sheets` — a tiled preview per planned segment. |
| 7 | **Segment Render** | Merge | `build_video.py cut` — each segment onto the target canvas at the target resolution, caption burnt in, blur-pad for cross-orientation footage. |
| 8 | **Transitions** | Animation | `build_video.py join` — the xfade chain, and it records the real transition timings. |
| 9 | **Score** | Music | `extras.music_stage` — generates a track against those exact transition marks, or fits your uploaded one to length. Also rebuilds the footage's **own** audio when you ask for it (see below). Writes `_audio.json`. |
| 10 | **Master & Verify** | Merge | `extras.master_stage` — optional retime, mux, two-pass loudness normalisation, thumbnail, then **verification**. Writes `output.json`. |

Captions are written to `captions.json` between the planner and the cut, which is exactly
where the pipeline already expects to find them.

### Deciding what to keep, across every LEGO shot type

YDIF is an inter-frame *difference* metric, so its scale is a property of what
is in shot, not of whether the footage is any good. Hands assembling a frame
read high; a Technic gearbox turning over reads middling; a close-up of the
finished model, a locked-off demonstration or a slow orbit around a display
piece all read low while being the shots you most want in the reel. An
absolute "below X is dead" cut-off therefore encodes one kind of LEGO shot and
throws away the rest, so the planner does not use one:

- the floor is **relative to the batch you uploaded**, with an absolute floor
  low enough that only a genuinely static frame falls under it;
- the real test is `live_share` — the share of frames in which *anything*
  changes. A small subject on a still background has a tiny frame-average but
  changes in most frames; a lens cap, a frozen export or a held slate does not.
  That is what separates calm footage from dead footage;
- `freezedetect` output is kept and still steers window choice *inside* a clip,
  but it no longer rejects one: at its −38 dB threshold it calls a slow pan 98%
  frozen and a locked-off interview 100% frozen;
- runtime is shared by the **square root** of motion, not motion itself, so one
  frenetic clip cannot eat most of a mixed batch;
- if the rule would drop most of the upload, it is describing the footage
  rather than judging it — everything is kept and window scoring ranks it;
- and any clip that *is* left out is named, with the reason, on the processing
  step and again on the final screen. Nothing disappears quietly.

### The vision layer (optional)

Off unless `AT_VISION_API_KEY` is set. When on, it runs as stage 3 and feeds
the Director.

| Variable | Default | Purpose |
|---|---|---|
| `AT_VISION_API_KEY` | *(unset)* | **the switch.** Unset = disabled, and everything else works exactly as before |
| `AT_VISION_PROVIDER` | `anthropic` | `anthropic` or `openai` — both wire formats are implemented |
| `AT_VISION_MODEL` | provider default | any vision-capable model id |
| `AT_VISION_BASE_URL` | the provider's API | point at a gateway, proxy or self-hosted endpoint |
| `AT_VISION_FRAMES` | `4` | frames sampled per clip (1–8) |
| `AT_VISION_TIMEOUT` | `90` | seconds per request |
| `AT_VISION_MAX_CLIPS` | `12` | cap on clips analysed per build |

**Two passes, because it is cheaper and more accurate than one.** First, per
clip: N frames in, structured description out. Then one text-only pass over
those summaries — no images — to spot duplicates and nominate a hook. A
five-clip build at the default settings is six requests carrying 20 frames.

**Everything the model returns is treated as untrusted input.** Enum fields are
checked against a fixed list, numbers are range-clamped, timestamps are clamped
to the clip's real duration, and any clip name it mentions must be one you
actually uploaded. A model that returns nonsense degrades to the rule-based
path; it cannot corrupt the edit.

**What it is allowed to change, and what it is not.** It can nominate
duplicates (which become real `--drop` flags on the planner), nominate the
hook, shift a transition choice via shot type, and offer captions. It can only
fill caption slots **you left blank**, only above 60% confidence, and it never
overwrites a line you wrote. Your manual settings still beat both layers.

**Failure is designed for, not hoped against.** A bad key, a 500, an
unreachable host, a timeout, or a reply that is not usable JSON all log the
reason and fall through to the rule-based Director. Verified: all four produce
a build byte-identical to running with no key at all.

### Deciding the running order

A LEGO reel is a story — the frame goes down, the sub-assembly drops in, the
mechanism runs, the finished model gets shown. Three ways to sequence it, all
from real data, none of them guesswork:

| Mode | Order | Best for |
|---|---|---|
| **Hook first** *(default)* | the liveliest clip by measured motion opens, the rest follow your upload order, slow-motion lands last | a reel or short, where the first second decides whether anyone watches |
| **Build order** | chronological by the `creation_time` the camera wrote into each file | a build video shot as the build progressed — the recording order *is* the narrative order |
| **My order** | exactly the order you dragged them into on step 1 | when you already know the story you want |

**Build order** is worth explaining, because it is the one that does real work
for you. Phones and cameras stamp each file with when it was recorded, so
footage shot across a build session already carries its own running order —
even if you upload it in a jumble. The planner reads that stamp in `probe` and
sorts by it. Clips that have been re-encoded, screen-recorded or stripped by an
editor often have no stamp; when fewer than two clips carry one there is
nothing to sort, so it falls back to your upload order and the UI tells you
that before you build rather than after.

### The footage's own audio

The pipeline renders its segments video-only, which is right for a silent
montage and wrong for a tutorial, an interview or anything where someone is
talking. Rather than change the render, `extras.source_audio` reassembles the
same cut's original audio from the source files using the planner's own in/out
points, `acrossfade`d at exactly the `xfade` offsets the video used, so the two
stay in step. Clips with no audio track contribute silence of the right length
so a mixed batch lines up instead of drifting.

The **Original audio** control on the Customize step chooses:

| Setting | Result |
|---|---|
| Drop it *(default)* | music bed only — the footage's own audio is discarded |
| Keep + music | original audio over the score, score ducked beneath it via `sidechaincompress` |
| Keep it only | original audio, no score |

The default is off because a silent montage is the common case and wind noise
is not content. When any uploaded clip carries an audio track and the setting
is off, the UI says so before you build, rather than letting you find out from
a silent tutorial.

### Verification (stage 10)

The output screen shows four checks **measured on the finished file**:

- **Frame count** — frames are counted by *decoding*, not inferred from the
  container duration. This matters: an earlier truncated render still reported
  the right duration because the audio was full length while the video was 4s
  short. Counting frames caught it; reading metadata did not.
- **Black frames** — `blackdetect`.
- **Loudness** — `ebur128` integrated LUFS. A build you asked to be silent is
  judged against *that*, not against a loudness range it was never meant to hit.
- **Audio track** — present, and its codec.

---

## 9 · Known limitation: voice transcription

**Voice notes are recorded, stored, probed and played back — but not
transcribed.**

No offline speech-to-text engine could be installed in this environment (vosk
fails to build; the model hosts are blocked by the egress proxy). Rather than
show a control that quietly does nothing:

- `GET /api/health` reports `capabilities.transcription: false`
- the Customize step shows a standing notice saying transcription is unavailable
  and why
- `extras.store_voice_note()` returns `{"transcript": null,
  "transcript_status": "no_engine"}` — it never returns an empty string that
  could be mistaken for "nothing was said"

**To wire an engine in later**, `extras.store_voice_note()` is the single place
to do it. Install the engine, and have that function fill `transcript` and set
`transcript_status`; `preflight()` in `adapter.py` already probes for `vosk` and
`faster_whisper`, so `/api/health` and the UI notice will update themselves.

### Captions are yours, not the system's

Nothing in the pipeline decides what a caption should say. The system measures
motion; it does not recognise objects, read scenes or understand subject
matter, so it has no basis on which to describe your footage and does not
invent one. Captions come from the script you write on the Customize step,
spread one line per segment. Leave it blank and the cut renders clean.

### Other things that are honestly out of scope

The three agents specified in the original `AGENT_SPECS.md` (Technical QC,
Music & Sound Design, Content Judgment) are **not implemented** — they remain a
plan, as that document's own README states. This app exposes the pipeline that
*does* exist. `AGENT_SPECS` also requires the Content Judgment agent be
validated against the 31 hand-scored videos before it scores anything new, and
that validation has not been done.

---

## 10 · Deployment

### Docker (recommended — ffmpeg comes with it)

```bash
docker compose up --build         # → http://localhost:8000
```

`uploads/`, `outputs/` and `data/music/` are bind-mounted, so footage, renders
and the library survive an image rebuild. Or without compose:

```bash
docker build -t alpha-titans .
docker run -p 8000:8000 \
  -v "$PWD/uploads:/app/uploads" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/data/music:/app/data/music" \
  alpha-titans
```

The image has a `HEALTHCHECK` against `/api/health`, so an orchestrator will not
route traffic to a container whose pipeline is not answering.

**Give it CPU.** Encoding is the whole job; a 4K master render is CPU-bound.
`docker-compose.yml` sets a limit of 4 CPUs via `AT_CPUS` — raise it on a
bigger box.

### On a server directly

```bash
git clone <repo> && cd alpha_titans_app
sudo apt install -y ffmpeg python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set AT_HOST=0.0.0.0
./run.sh
```

As a systemd unit:

```ini
[Unit]
Description=Alpha Titans Content Engine
After=network.target

[Service]
Type=simple
User=titans
WorkingDirectory=/opt/alpha_titans_app
Environment=AT_HOST=0.0.0.0
ExecStart=/opt/alpha_titans_app/.venv/bin/python -m uvicorn main:app \
          --app-dir /opt/alpha_titans_app/server --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Behind nginx

Two things matter, and both are easy to get wrong:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;

    # SSE: the live stage/log stream dies without these
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;

    # uploads: clips are large
    client_max_body_size 2048m;
}
```

### Deployment notes

- **Run one instance per machine.** Job state lives in memory plus
  `outputs/<id>/job.json`, and renders are CPU-bound — a second worker would
  fight the first for cores without going any faster. If you scale, scale by
  machine, with `uploads/` and `outputs/` on shared storage.
- **There is no authentication.** Put it behind your own auth (an nginx
  `auth_basic`, an SSO proxy, or a private network) before exposing it.
- **Jobs survive a restart.** A job that was mid-run when the server stopped is
  marked failed rather than left looking alive.
- **`outputs/` grows.** Old builds are never deleted automatically; `DELETE
  /api/jobs/{id}` removes one, and the Builds drawer is the list.

---

## 11 · Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `./run.sh: Permission denied` | Some extractors drop the executable bit. `chmod +x run.sh`, or just run `python3 -m uvicorn main:app --app-dir server`. |
| `run.sh` says **ffmpeg is not on PATH** | Install it (section 4). If it is installed somewhere unusual, set `AT_FFMPEG` / `AT_FFPROBE` in `.env` to the full paths. |
| `run.sh` says **Python dependencies missing** | `pip install -r requirements.txt`. If you used a venv, activate it first. |
| **Port already in use** | `AT_PORT=9000 ./run.sh`, or stop whatever holds 8000. |
| Page loads but shows **"API unreachable"** | The static files are being served by something other than this server. Either serve the UI from this app (the default) or set `AT_CORS_ORIGINS`. |
| Health shows **`beat_engine: false`** | numpy/scipy are not installed. Not fatal — `bgm_stdlib.py` is used instead. `pip install numpy scipy` to get the full kit. |
| Health shows **`transcription: false`** | Expected. See [section 9](#9--known-limitation-voice-transcription). |
| **A clip is rejected on upload** | The reason is in the toast: not a video container, over the size cap, or ffprobe could not read it. Re-export it as H.264 MP4 and retry. |
| **Clip cards show a black thumbnail** | The poster endpoint could not extract a frame — usually a corrupt or zero-length file. Check `ffprobe <file>` directly. |
| Build fails at **Motion Analysis** | Almost always a clip ffmpeg cannot decode. Open the **Pipeline output** panel on step 3 — the pipeline names the file. |
| Build fails at **Segment Render** with a font error | The caption draw needs a font. In Docker this is `fonts-dejavu-core` (already in the Dockerfile); on a bare server `sudo apt install fonts-dejavu-core`. |
| **The finished video is silent** and loudness reads ≈ −70 LUFS | Either music was set to Silent (then the check reads "Silent, as requested" and this is correct), or a library track was selected but the file is missing from `data/music/`. |
| **Frame count check fails** | The render was truncated — usually the disk filled mid-encode. Check free space and regenerate. |
| **Progress stops updating but the build is still running** | The SSE stream was cut, typically by a proxy that buffers. See the nginx settings above. The build itself is unaffected; reload the page to reattach. |
| **Everything is very slow** | Encoding is CPU-bound. Use **Draft** quality and **720p** to check a cut, then regenerate at full quality. Motion Analysis and Segment Render are the two slow stages. |
| Want to **start clean** | Stop the server, delete the contents of `uploads/` and `outputs/`, restart. Nothing else holds state. |

---

## 12 · Tested end to end

### The vision layer, on vs off

Same five clips, same settings, the only difference being whether
`AT_VISION_API_KEY` was set:

| | vision **off** | vision **on** |
|---|---|---|
| clips kept | 5 of 5 | **4 of 5** — one dropped as a duplicate |
| running order | technic, assembly, robotics, closeup, demo | technic, assembly, closeup, demo |
| transitions | `fadewhite fade fade fade` | **`fadewhite wipeup fade`** |
| captions (no script given) | all blank | **4 suggested, ≥60% confidence** |
| rendered output | 904 frames | **912 frames — a different video** |

Both runs verified green. A **broken** vision endpoint (bad key) produced a
result byte-identical to the off run, and the vision stage finished `done`
rather than failing the build.

**What was verified, and what was not.** The integration is real: 20 genuine
JPEG frames were extracted by ffmpeg and sent as `POST /v1/messages` with
`x-api-key` and `anthropic-version` headers, correct `image` content blocks and
the right model id — the requests were captured and asserted on. Response
parsing, clamping, the effect on the edit plan and all four failure modes were
exercised. What could **not** be verified in the build environment is a live
model's judgement, because no API key was available and the sandbox's proxy
blocks the inference endpoints; that path was exercised against a local server
speaking the same wire protocol. Point `AT_VISION_BASE_URL` at the real API
with a real key and the same code path runs unchanged.

### Five LEGO shot types, uploaded in the wrong order

The test that matters for a LEGO reel. Five clips standing in for the shot
types a build session actually produces — assembly (busy), Technic gearbox
(steady), robotics demo (**vertical phone**, with motor audio), close-up of the
finished model (nearly still), demonstration to camera (stillest, with speech)
— **uploaded deliberately scrambled** and set to Build order:

| Uploaded as | Recorded at | Planner put it at |
|---|---|---|
| 1 close-up | 14:10 | **4** |
| 2 assembly | 09:02 | **1** |
| 3 demonstration | 16:30 | **5** |
| 4 robotics | 11:45 | **3** |
| 5 Technic | 10:20 | **2** |

The cut came out in build order — assembly → gearbox → motor running →
finished model → explanation — from the recording timestamps alone, with each
caption landing on its segment. Verified by pulling the frame at every
segment's predicted timestamp out of the finished MP4 and looking at it:
**904 of 904 frames**, −14.1 LUFS, AAC, vertical clip blur-padded onto the
horizontal canvas, original audio mixed under the score.

**All five were kept — and two of them only because of the threshold work in
§8.** The close-up measured `mean_ydif` **0.11** and the demonstration
**0.10**; the old absolute floor discarded anything under 0.5, so a plain
close-up of your finished model and a piece to camera were both being thrown
away. Those are not edge cases in LEGO footage, they are half the reel.

Switching the same five clips to **Hook first** produced a genuinely different
order (Technic opened, on 4.03 measured motion) — the sequencing is a real
decision, not a label.

### Six unlike clips in one batch

The test that matters for a general-purpose editor: six clips with nothing in
common — a travel pan (1920×1080, ambient audio), a product spin (1920×1080,
silent), a talking head (1920×1080, speech), an event reel (**1080×1920
vertical**, music), a b-roll macro (**1280×720**, silent) and a deliberately
dead slate — uploaded together, no per-clip hand-holding:

- all **five real clips** were combined into **one** 28.43s video; the dead
  slate was the only exclusion, named in the UI with its reason
- the vertical clip was blur-padded onto the horizontal canvas; the 720p clip
  was fitted alongside the 1080p ones
- the running order was chosen from measurement: liveliest clip opened
- each segment was confirmed **present in the finished MP4** by pulling the
  frame at its predicted timestamp out of the final file and looking at it
- **853 of 853 frames**, no black frames, −14.2 LUFS, AAC stereo
- audio verified by spectrum, not by trusting the setting: the talking head's
  180 Hz source tone is present and dominant in its window (energy 1722), the
  travel clip's pink noise carries through at 1–3 kHz (2130), and the window
  belonging to the genuinely silent product clip has neither (32 / 329) — music
  only, exactly as it should be

Three bugs surfaced only because this batch was heterogeneous, and all three
were the same mistake in different clothes — an absolute threshold tuned on one
kind of footage:

1. `mean_ydif < 0.5` **discarded the talking head and the travel pan.** Fixed by
   judging against the batch and by measuring how often anything changes.
2. `freeze_share >= 0.85` then discarded the talking head again, because
   `freezedetect` calls a locked-off interview 100% frozen. Fixed by dropping
   freeze share as a rejection criterion.
3. `good_video(min_bytes=20000)` declared a rendered segment **missing** and
   failed the join, because 2.5s of a low-detail talking head legitimately
   encodes to 7 KB. Fixed by asking ffprobe for a duration instead of weighing
   the file.

### Original three-clip regression

Three synthetic clips (two 1920×1080 landscape, one 1080×1920 vertical, 31.0s of
source) driven through the real pipeline from this folder — and, separately,
through the real UI in a browser:

- **8/8 stages green**, ~65s wall clock at 720p/draft
- 720p override honoured (`canvas [1280,720]`); vertical run correctly swapped
  the canvas to `720×1280`
- `--target 15` honoured: the planner allocated exactly 15.0s
- captions from the script burnt into the frame (visible in the thumbnail)
- **327 frames for 10.9s at 30fps — frame check passes exactly**
- re-run after the mixed-footage fixes: all three clips now make the cut where
  the old weighting had used only two, and every check stayed green
- generated score **−14.3 LUFS**; uploaded library track **−15.4 LUFS**;
  silent build **−70 LUFS** (correct, and reported as "Silent, as requested")
- speed `fast` → 14.9s became 11.03s; `slow` → longer, both frame-exact
- regenerate reuses the clips via hard links, carries the script and voice note
  across, and leaves the original build intact
- clip reorder rewrites the numeric prefixes and the planner follows it
- no JS console errors, no HTTP ≥ 400, no horizontal overflow at 1440px or 414px

### Bugs found while building this, and fixed

- **Cancel poisoned the job.** `cancel()` raised a `threading.Event` that
  `start()` never lowered, so after one cancel every later build on that job
  aborted at the first stage and reported itself cancelled — with a 200 from
  `/start`, which is how it stayed hidden. Found by a route sweep that starts,
  cancels and restarts in sequence. `start()` now clears the flag.
- **The agent roster never appeared on a fresh build.** `renderAgents()` was
  only reached from `openJob()`, so the five cards showed when you re-opened an
  old build and nowhere else. It now runs from `buildControls()`, which is on
  the boot path.
- `/start` resolved a library track name to a path but `/regenerate` did not, so
  regenerating with an uploaded track silently produced a **silent** video. The
  verification panel is what caught it, as −70 LUFS. Both paths now share
  `resolve_music()`.
- `.slider` and `.capprev` set `display`, which overrode the `hidden` attribute
  — the custom-speed slider showed even on "Normal". `[hidden] { display: none
  !important }` added.
- Clip cards rendered black: a browser will not reliably paint a frame for a
  `<video>` asked only for metadata. There is now a real server-side poster
  endpoint, cached per clip.
- Opening a finished build left the settings form empty; it now hydrates clips,
  script, voice note, captions **and** every control from that job's settings.
- `aresample` failed on channel layout during the mux → replaced with
  `aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo`.
- The `ebur128` regex was grabbing the first per-frame `I:` value (−70 LUFS)
  instead of the summary → it now parses after `"Summary:"`.
