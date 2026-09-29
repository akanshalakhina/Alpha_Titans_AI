# Alpha Titans — Technical Context & Audit

**Prepared for external AI review. Nothing in the project was modified to produce this.**

| | |
|---|---|
| Source audited | `C:\Users\Akansha\Downloads\alpha_titans` (the running copy on the author's laptop) |
| Audit date | 2026-09-22 |
| Method | Every file read from disk and traced end to end. Route list, stage list, agent registry and environment variables were **extracted programmatically from source**, not recalled. |
| Code size | 7,932 lines across 14 files |
| Files audited | `server/*.py` (7), `pipeline/*.py` (3), `frontend/*` (3), `README.md`, `.env.example`, `requirements.txt`, `run.sh`, `run.bat`, `Dockerfile`, `docker-compose.yml` |

### Reading this document

Five labels are used throughout and they are **not** interchangeable:

| Label | Means |
|---|---|
| **REAL** | Measured from the actual media by ffmpeg/ffprobe. Not estimated. |
| **RULE-BASED** | A deterministic `if`/threshold/table in Python. **Not AI.** Reproducible; same input always gives the same output. |
| **AI** | An actual HTTP request to a vision model, with real image bytes in the payload. Only one component qualifies. |
| **USER INPUT** | A control the operator sets. **Not a system decision.** |
| **NOT IMPLEMENTED** | Does not exist in the code. |

A note on the word "agent": the codebase uses it to mean *a named component that owns a group of stages and reports its own status* — explicitly **not** "a language model". `server/adapter.py` lines 65–79 say so in a comment. Four of the five so-called agents contain no model call of any kind. This report keeps that distinction sharp because the brief asked for it.

---

## 1 · Project overview

### What Alpha Titans does

A local web application that takes a folder of raw video clips and produces **one** finished, captioned, scored MP4 — automatically choosing which clips survive, their running order, where each is trimmed, how long each runs, which transition goes on each individual cut, the crossfade length, and the music mood.

The operator works through four steps in a browser: **Media Bin → Customize → AI Processing → Final Video**.

### Current goal

Stated in `README.md` §1 and borne out by the code: turn raw LEGO build footage — assembly, Technic, mechanisms, robotics, experiments, close-ups, demonstrations, in any mix — into one polished video, with every decision recorded alongside the reason it was made, and with **no simulated behaviour anywhere**: if the system cannot do something, it says so rather than faking it.

### What is LEGO-specific

Genuinely very little. Three things only:

| Where | What | File / line |
|---|---|---|
| Vision prompt | `"These frames are sampled in order from ONE video clip of a LEGO build."` | `server/vision.py:205` (`CLIP_PROMPT`) |
| Vision prompt | `"Here are descriptions of {n} clips from one LEGO shoot"` | `server/vision.py:228` (`CROSS_PROMPT`) |
| Vision taxonomy | `KINDS = ("assembly", "demonstration", "reveal", "detail", "other")` — a build-video vocabulary | `server/vision.py:63` |
| UI copy | "Any LEGO footage, in any mix…" and similar wording | `frontend/index.html`, `README.md` |

Everything else — the whole pipeline, the whole director, the whole API — is subject-agnostic. **The LEGO framing lives in two prompt strings and some UI copy.** Changing those two strings would make the system domain-neutral without touching a line of pipeline logic.

Note the `KIND_XFADE` table in `director.py:64` keys off `assembly`/`reveal`/`demonstration`/`detail`, which comes from the vision taxonomy — so that one table inherits the build-video vocabulary indirectly. It degrades harmlessly when vision is off (falls through to the motion-based table).

### What is generic and reusable

- **The entire `pipeline/` folder.** `build_video.py` never mentions LEGO. It probes, measures motion, scores windows, allocates a length budget, renders onto a canvas, crossfades. Any subject.
- **The entire rule-based director** (`server/director.py`). It reasons about motion, orientation and duration — nothing else.
- **The entire API and job runner.** Media-type agnostic.
- **The frontend.** Would need copy changes only.
- **The threshold design** (`build_video.py:83–90`). Explicitly built to be *batch-relative* rather than absolute, precisely so a still close-up or a talking head is not discarded. That work was driven by non-LEGO test footage and generalises.

---

## 2 · Complete architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ BROWSER — frontend/index.html + app.js + styles.css                          │
│ Vanilla ES module. No framework, no bundler, no build step, no npm.          │
└───────┬──────────────────────────────────────────────────────┬───────────────┘
        │ fetch() JSON · XMLHttpRequest (uploads, for progress)│ EventSource (SSE)
        ▼                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ FastAPI — server/main.py   30 routes, no business logic                      │
│ Serves the UI at "/" and /static, so there is one port and no CORS needed.   │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │
        ├── server/jobs.py      Job objects, the stage runner THREAD, weighted
        │                       progress, the event log that feeds SSE.
        │                       State → outputs/<job_id>/job.json
        │
        ├── server/adapter.py   THE REGISTRY. Owns AGENTS, STAGES, STAGE_AGENT,
        │                       RESOLUTIONS, QUALITY, TRANSITIONS, SEQUENCES,
        │                       KEEP_AUDIO. Builds the exact CLI call per stage
        │                       and maps UI settings → CLI args + env overrides.
        │
                ├── server/vision.py    ◆ THE ONLY VISION/LLM CALL, AND THE ONLY
        │                         OUTBOUND CALL FROM PROJECT CODE ◆
        │                       ffmpeg-extracted JPEG frames → HTTP POST to
        │                       Anthropic or OpenAI. Disabled unless
        │                       AT_VISION_API_KEY is set. Returns None on any
        │                       failure; the build continues regardless.
        │                       (A SECOND model exists — local Whisper in
        │                        extras.py — but it transcribes voice notes,
        │                        not footage, and touches no edit decision.)
        │
        ├── server/director.py  RULE-BASED edit planner. No model, no network.
        │                       Reads manifest.json + vision.json, writes
        │                       edit_plan.json with a reason per decision.
        │
        ├── server/extras.py    captions.json, voice storage + transcription,
        │                       the score stage, the master/verify stage.
        │
        └── server/config.py    Portable paths, .env reader, size limits.
                │
                │  subprocess.Popen — one process per stage, stdout streamed
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ pipeline/build_video.py   — the render engine, invoked as a CLI              │
│ pipeline/beatbed.py       — music engine (needs numpy + scipy)               │
│ pipeline/bgm_stdlib.py    — music engine (standard library only)             │
└───────┬──────────────────────────────────────────────────────────────────────┘
        ▼
    ffmpeg / ffprobe  ──────►  outputs/<job_id>/<TITLE>_<orientation>.mp4
```

### How the components communicate

| Hop | Mechanism | Detail |
|---|---|---|
| Browser → API | `fetch()` for JSON; `XMLHttpRequest` for clip uploads | XHR is used only because it exposes `upload.onprogress`, which `fetch` does not (`app.js:469–488`) |
| API → Browser (live) | **Server-Sent Events** | `GET /api/jobs/{id}/events`, `text/event-stream`, 0.5 s poll of the in-memory log (`main.py:529–557`) |
| API → job runner | A plain `threading.Thread` | `jobs.start()` at `jobs.py:259`. Keeps minutes of CPU off the asyncio event loop |
| Runner → pipeline | `subprocess.Popen`, one per stage | `adapter.run_stage()` at `adapter.py:299`. stdout is read line by line and re-emitted as SSE |
| Runner → vision/director/extras | Direct in-process Python calls | `jobs.py:283–307` — these four stages are **not** subprocesses |
| UI settings → pipeline | **Environment variables** | `adapter.env_for()` at `adapter.py:221` sets `AT_CANVAS`, `AT_CRF`, `AT_XFADES`, `AT_XFADE_T` |
| UI settings → pipeline | **CLI flags** | `adapter.stage_command()` at `adapter.py:193` adds `--orient`, `--target`, `--sequence`, `--drop`, `--hook` |
| Director → renderer | `settings["transition_list"]` → `AT_XFADES` | This is the single most important data path in the system; see §6 |
| Everything → UI | `manifest.json` | The pipeline's own state file is the single source of truth for clip probes, motion figures, the cut sheet and transition timings |

**Why subprocesses.** A render is minutes of CPU. One process per stage keeps it off the event loop, allows the server to stream *real* stdout instead of inventing progress, and preserves the pipeline's own resumability (re-run a stage and it skips finished work).

---

## 3 · Frontend

### Framework

**None.** `frontend/app.js` is a single 1,248-line ES module loaded with `<script type="module" src="/static/app.js">`. No React, no Vue, no bundler, no `node_modules`, no CDN. `frontend/styles.css` is 908 lines of hand-written CSS using custom properties. Total frontend: 3 files.

### Pages / components

It is a single HTML document with four `<main class="step">` panels — `#step-1` … `#step-4` at `index.html:40, 88, 293, 367` — one shown at a time by `goto(step)` (`app.js:86`):

| Step | Element | Contents |
|---|---|---|
| 1 Media Bin | `#step-1` | Drop zone, clip table with poster thumbnails, drag-to-reorder, per-clip facts, tally bar |
| 2 Customize | `#step-2` | Script box, voice recorder, agent roster, Format / Motion / Music cards |
| 3 AI Processing | `#step-3` | Progress ring, agent-grouped stage list, excluded-clips notice, edit-plan card, raw log |
| 4 Final Video | `#step-4` | Player, download, verification panel, spec table, segment timeline, cut sheet |

Plus a "Builds" drawer (`#drawer`) listing previous jobs, and a toast host.

### Every user control

| Control | Element id | Type | Setting written | Reaches the render via |
|---|---|---|---|---|
| Add clips | `#clipInput` / `#browseClips` | file input (multiple) | — | `POST …/clips` |
| Drag-and-drop clips | page-wide | drop target | — | same endpoint (`app.js:386–410`) |
| Reorder clips | `.clip[draggable]` | drag | — | `POST …/clips/order` → numeric filename prefixes |
| Remove clip | `.iconbtn` per row | button | — | `DELETE …/clips/{name}` |
| Script | `#script` | textarea | `job.script` | `captions.json` → burnt into frames |
| Upload script | `#scriptFile` | file input | `job.script` | same |
| Record voice note | `#recordBtn` | MediaRecorder | `job.voice` | stored; transcript offered for insertion into the script |
| Build name | `#title` | text | `title` | output filename |
| Orientation | `#orientation` | segmented | `orientation` | `--orient` + `AT_CANVAS` swap |
| Resolution | `#resolution` | 4 tiles | `resolution` | `AT_CANVAS` |
| Quality | `#quality` | segmented | `quality` | `AT_CRF` (24/20/18/15) |
| Target length | `#target` | number | `target_seconds` | `--target` |
| Running order | `#sequence` | 3 tiles | `sequence` | `--sequence hook\|recorded\|uploaded` |
| Speed | `#speed` | 4 tiles | `speed` | `setpts` retime in `extras.master_stage` |
| Custom speed | `#speedRange` | range | `speed_custom` | same, clamped 0.4–2.5 |
| Transition style | `#transition` | 5 tiles | `transition_style` | `AT_XFADES`; `auto` hands the decision to the director |
| Transition length | `#xfade` | segmented | `transition_duration` | `AT_XFADE_T`; `auto` hands it to the director |
| Music source | `#musicSource` | 3 tiles | `music_source` | generate / upload / none |
| Mood | `#mood` | chips | `mood` | engine selection; `auto` hands it to the director |
| Upload track | `#musicFile` | file input | — | `POST /api/music` |
| Footage's own audio | `#keepAudio` | 3 tiles | `keep_audio` | `extras.source_audio` + `mix_beds` |
| Start | `#startBuild` | button | — | `POST …/start` |
| Cancel | `#cancelBuild` | button | — | `POST …/cancel` |
| Regenerate | `#regenBtn` | button | — | `POST …/regenerate` |
| Download | `#downloadBtn` | anchor | — | `GET …/download` |
| Autoscroll log | `#autoscroll` | checkbox | — | UI only |
| Copy log | `#copyLog` | button | — | clipboard |

**Six** of these control groups are built **from `/api/health`** by `buildControls()` (`app.js:235`) and therefore cannot drift from what the server accepts: `#sequence` (238), `#keepAudio` (243), `#resolution` (248), `#quality` (252), `#transition` (255), `#mood` (259).

The other four are **hardcoded markup** and *can* drift if the server's option lists ever change: `#orientation` (`index.html:164`), `#speed` (`index.html:212`), `#xfade` (`index.html:233`), `#musicSource` (`index.html:251`).

### Upload flow

1. `uploadClips(files)` (`app.js:460`) calls `ensureJob()` — **no job is created on page load**, deliberately, so an abandoned visit does not leave a phantom draft in the Builds list.
2. Builds one `FormData` with every file appended under the key `files`.
3. Sends via `XMLHttpRequest` so `upload.onprogress` can drive a real byte-count progress bar.
4. Server (`main.py:195`) streams each file to disk in 1 MB chunks, enforcing `MAX_CLIP_BYTES` **during** the write, saving as `NN_originalname.ext`, then ffprobes it. A file ffprobe cannot read is deleted and reported in `rejected`.
5. Response `{saved, rejected}`; rejections surface as individual toasts.
6. `refreshClips()` re-lists from the server — the UI never trusts its own optimistic state.

### Build / generate flow

`startBuild()` (`app.js:859`) → saves the script → `POST …/start` with the whole settings object → clears the log → `resetRun()` → `goto(3)` → `listen(jobId)` opens the SSE stream.

### Progress UI

- **Ring**: SVG circle, `stroke-dashoffset` driven by `job.progress` (`setRing`, `app.js:850`).
- **Progress value is weighted, not "stage N of 10"**: `jobs.Job.progress()` (`jobs.py:78`) sums `weight` for finished stages and adds 45% of the current stage's weight. Weights are declared in `adapter.STAGES` and sum to 116 — `cut` is 31, `measure` 20, `join` 15, `master` 12, `vision` 14, `music` 8, `sheets` 7, `probe` 4, `plan` 3, `direct` 2. So the bar moves in proportion to real work.
- **Agent-grouped stage list**: `renderRun()` (`app.js:986`) groups the ten stages under the five agents from `/api/health`, with per-agent elapsed time and a badge saying `vision model` or `rule-based`.
- **Raw log**: every line of pipeline stdout, colour-coded, in a collapsible `<details>`.

### SSE implementation

Client (`app.js:889–920`):

```js
const es = new EventSource(`/api/jobs/${jobId}/events`);
es.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.kind === 'snapshot' || m.kind === 'final') { … }
  if (m.kind === 'line')   appendLog(m);
  if (m.kind === 'stage' || m.kind === 'status') refreshJob();
};
es.onerror = () => { /* EventSource retries on its own */ };
```

On `kind === 'final'` it closes the stream, renders the output and auto-advances to step 4. See §9 for the server side.

### Final video display and download

`renderOutput(job)` (`app.js:1131`) sets `player.src` to `/api/jobs/{id}/file/{name}` and `player.poster` to the pipeline's own generated thumbnail, then renders:

- **Verification panel** — four checks, each rendered pass/fail from figures measured on the finished file: frame count vs expected, black-frame detection, loudness in range, audio track present. A build the operator asked to be silent is judged against `< -50 LUFS` instead of the normal range (`app.js:1169–1177`).
- **Spec table** — file, size, duration, resolution, frame rate, codecs, speed, orientation, transition timestamps.
- **Segment timeline** — a proportional bar, one block per segment.
- **Cut sheet** — a table of clip, in–out, length, measured motion, caption.
- **Download** — `GET /api/jobs/{id}/download`, which serves with `Content-Disposition` via FastAPI's `FileResponse(filename=…)`.

### Implemented vs mocked — frontend

| Area | Status |
|---|---|
| Clip list, durations, dimensions, codecs, rotation, audio flag | **REAL** — ffprobe output via `/clips` |
| Clip poster thumbnails | **REAL** — server-side ffmpeg frame extraction, cached (`main.py:274`) |
| Progress percentage | **REAL** — weighted, from actual stage transitions |
| Stage timings | **REAL** — `finished - started` per stage |
| Log | **REAL** — the pipeline's own stdout, verbatim |
| Edit plan and reasons | **REAL** — read from `edit_plan.json` |
| Verification figures | **REAL** — measured on the finished MP4 in `extras.master_stage` |
| Cut sheet | **REAL** — the planner's own `manifest.json` segments |
| Sample/demo data anywhere | **NONE FOUND.** No mock data, no simulated progress, no placeholder state in `app.js` |

---

## 4 · Backend / API — all 30 routes

Extracted from `server/main.py`. Line numbers are of the handler function.

### Meta

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 1 | GET | `/api/health` | Everything the UI builds itself from | — | `{ok, capabilities, stages[10], options{}, agents[5], stage_agent{}, vision{}, keep_audio_detail, sequence_detail, notes, limits}` | `ok` is `pipeline AND ffmpeg`. Capabilities are probed live by `adapter.preflight()` — `which ffmpeg`, `which ffprobe`, `import numpy/scipy`, `import vosk/faster_whisper`. `main.py:100` |

### Jobs

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 2 | GET | `/api/jobs` | List builds, newest first | — | `{jobs:[{id,status,created,title,progress,clips,duration,orientation,thumbnail}]}` | Capped at 30. **Drafts with no clips are filtered out** (`jobs.py:167`) so opening the page never litters the list. Scans `outputs/` to pick up jobs from previous server runs. `main.py:146` |
| 3 | POST | `/api/jobs` | Create a job | `{...settings}` optional | `{id, job}` | id = `uuid4().hex[:12]`. Creates three folders. `main.py:151` |
| 4 | GET | `/api/jobs/{job_id}` | Full snapshot | — | `Job.snapshot()` — status, progress, all 10 stages, settings, script, captions, voice, output, `edit_plan`, `vision`, `analysis` | 404 if unknown. `main.py:158` |
| 5 | DELETE | `/api/jobs/{job_id}` | Delete job + both folders | — | `{deleted}` | `shutil.rmtree` on build dir and src dir. **Irreversible, unauthenticated.** `main.py:163` |
| 6 | PATCH | `/api/jobs/{job_id}` | Update settings | `{...}` | snapshot | `dict.update` — **no schema validation, no key whitelist**. `main.py:173` |

### Clips

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 7 | GET | `…/clips` | List clips with probed facts | — | `{clips:[{name,duration,recorded,size,width,height,rotation,codec,has_audio,orientation}]}` | Rotation-aware: swaps w/h when `\|rotation\| % 180 == 90`. `main.py:183` |
| 8 | POST | `…/clips` | Upload one or many | multipart, key `files` | `{saved[], rejected[]}` | Extension must be in `(.mov,.mp4,.m4v,.avi,.mkv)`; size checked **mid-stream** against `AT_MAX_CLIP_MB` (default 2048); ffprobe must return a duration or the file is deleted. Saved as `NN_name`. `main.py:195` |
| 9 | DELETE | `…/clips/{name}` | Remove one clip | — | `{removed}` | `safe_name()` + `resolve()` + `relative_to()` traversal guard → 400. 404 if absent. `main.py:236` |
| 10 | POST | `…/clips/order` | Reorder | `{order:[names]}` | `{order:[new names]}` | **400 unless the list is exactly the current set.** Two-phase rename via `tmp_` prefixes to avoid collisions. `main.py:250` |
| 11 | GET | `…/poster/{name}` | Cached poster frame | — | `image/jpeg` | Extracts at 25% of duration (capped 5 s), scaled to 320px, cached in `_posters/`. 500 if ffmpeg produces nothing. `main.py:274` |
| 12 | GET | `…/clip/{name}` | Stream a source clip | — | file | `serve_within()` traversal guard. `main.py:303` |

### Script and voice

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 13 | POST | `…/script` | Set the script | `{text}` | `{script, segments, preview[]}` | Truncated to 20,000 chars. Returns a live per-segment caption preview once a manifest exists. `main.py:311` |
| 14 | POST | `…/script-file` | Upload `.txt`/`.md` | multipart `file` | `{script, filename}` | **413 over 512 KB** (`main.py:326`). Decoded UTF-8 with `errors="replace"`. ⚠ The `except → 400 "could not read that file as text"` at `main.py:329` is **unreachable**: `replace` cannot raise `UnicodeDecodeError`, so no caller can trigger it. `main.py:322` |
| 15 | POST | `…/voice` | Store a voice note | multipart `file` | `{filename,bytes,duration,playable,transcript,transcript_status}` | 413 over `AT_MAX_VOICE_MB` (40). **Transcribes if `faster-whisper` is importable** (see §7). `main.py:336` |
| 16 | GET | `…/voice-file` | Play it back | — | file | 404 if no voice note. `main.py:347` |

### Music library

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 17 | GET | `/api/music` | List tracks | — | `{tracks:[{name,duration,size,mood}]}` | Duration from ffprobe; mood from a sidecar `.json`. `main.py:357` |
| 18 | POST | `/api/music` | Add a track | multipart `file`, form `mood` | `{name,duration,size,mood}` | Extension must be audio → 400; 413 over `AT_MAX_TRACK_MB` (120); **ffprobe must find streams or the file is deleted** → 400. `main.py:380` |
| 19 | DELETE | `/api/music/{name}` | Remove | — | `{removed}` | Traversal guard → 400; 404 if absent; removes the sidecar too. `main.py:402` |
| 20 | GET | `/api/music/{name}` | Stream | — | file | `serve_within()`. `main.py:416` |

### Build control

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 21 | POST | `…/start` | Run the build | `{...settings}` optional | snapshot | Calls `resolve_music()`; **pops `transition_list`** so a re-run re-decides rather than inheriting; sets `_has_script`. **400 "Add at least one clip"** if the folder is empty. `main.py:423` |
| 22 | POST | `…/cancel` | Stop after the current stage | — | `{status}` | Sets a `threading.Event`. Cooperative — it does **not** kill the running ffmpeg. `main.py:439` |
| 23 | POST | `…/regenerate` | New build, same clips | `{...settings}` | `{id, job}` | Creates a **new** job, **hard-links** the clips (`os.link`, falls back to copy), carries script and voice across, leaves the original intact, starts immediately. `main.py:446` |

### Outputs

| # | Method | Endpoint | Purpose | Request | Response | Validation / errors |
|---|---|---|---|---|---|---|
| 24 | GET | `…/file/{name}` | Any build artefact | — | file | `serve_within()`. Used for the player and the poster. `main.py:481` |
| 25 | GET | `…/download` | The finished MP4 | — | `video/mp4` attachment | **409** if the build has no output yet; **404** if the file vanished from disk. `main.py:487` |
| 26 | GET | `…/vision` | What the model saw | — | `vision.json` | **404 when vision did not run** — which is the normal state without an API key. `main.py:498` |
| 27 | GET | `…/plan` | The edit plan | — | `edit_plan.json` | 404 before the build runs. `main.py:508` |
| 28 | GET | `…/notes` | Verified output figures | — | `output.json` | 404 before the build finishes. `main.py:518` |
| 29 | GET | `…/events` | **SSE** | `?after=N` | `text/event-stream` | See §9. `main.py:529` |
| 30 | GET | `/` | The UI | — | `index.html` | 500 with a diagnostic if the frontend folder is missing. `main.py:563` |

Plus a static mount: `app.mount("/static", StaticFiles(directory=WEB))` at `main.py:573`.

### Cross-cutting error handling

- **Path traversal** is guarded in two ways: `safe_name()` (`main.py:52`) strips everything outside `[A-Za-z0-9._ -]` and takes the basename; `serve_within()` (`main.py:85`) resolves and calls `relative_to()`, raising 400 on escape. Applied to clip, poster, build-file, voice and music routes.
- **No `shell=True`, no `eval`, no `exec`, no `pickle`** anywhere in `server/` or `pipeline/` — verified by grep. Every subprocess is invoked with an argument list.
- **The vision API key is never returned or logged.** `vision.describe()` (`vision.py:92`) exposes `enabled`, `provider`, `model`, `base_url` — never the key. Log lines name the provider and model only.
- **Uncaught exceptions inside a build** are caught in `jobs._run` (`jobs.py:332`), which sets `status="error"`, marks the current stage failed, and emits the last 1,200 chars of the traceback to the log so the operator sees the real error.
- **There is no authentication, authorisation, rate limiting or CSRF protection on any route.** See §13.

---

## 5 · Video pipeline — all 10 stages in exact order

Declared in `adapter.STAGES` (`adapter.py:105`) and executed in that order by `jobs._run` (`jobs.py:271`).

**Important structural fact:** only **6 of the 10** stages are subprocess calls into `build_video.py`. Four (`vision`, `direct`, `music`, `master`) run **in-process** in the server. The dispatch is the `if/elif` chain at `jobs.py:283–310`.

| # | Stage id | Module / function | Input | Processing | Output | Real or mocked? |
|---|---|---|---|---|---|---|
| 1 | `probe` | `build_video.py` → `stage_probe()` (subprocess) | `uploads/<job>/*` | ffprobe each file for duration, w/h, rotation, codec, fps, audio stream, **`creation_time`**; drops iOS `IMG_E####`/`IMG_####` duplicate pairs; picks the canvas from whichever orientation dominates **by total duration** | `manifest.json` with `clips[]`, `orientation`, `canvas`, `target` | **REAL** |
| 2 | `measure` | `build_video.py` → `stage_measure()` (subprocess) | manifest + source files | Per clip: `scale=160:-2,fps=10,signalstats` → per-frame YDIF series; `freezedetect=n=-38dB:d=0.4` at 6 fps → frozen runs. Computes `mean_ydif`, `max_ydif`, `freeze_share`, **`live_share`** | motion fields written into `manifest.json` | **REAL** — the slowest stage (weight 20) |
| 3 | `vision` | `server/vision.py` → `analyse()` (in-process) | manifest + source files | If `AT_VISION_API_KEY` set: ffmpeg extracts N JPEG frames per clip → base64 → HTTP POST to the model; second text-only pass for duplicates/hook/story. **Every field range-checked and clamped.** If no key: logs why, returns `None` | `vision.json`, or nothing | **REAL AI when a key is set; SKIPPED otherwise.** The stage is marked `"skipped"` not `"done"` when it produced nothing (`jobs.py:322`) |
| 4 | `plan` | `build_video.py` → `stage_plan()` (subprocess) | manifest (+ `--drop`/`--hook` from vision) | Batch-relative rejection; **sqrt-weighted** length budget; `best_window()` scans every 0.2 s for the highest-scoring window per clip; applies the running order | `segments[]`, `rejected[]`, `sequence{}`, `planned_total` in `manifest.json` | **RULE-BASED**, on REAL measurements |
| 5 | `direct` | `server/director.py` → `direct()` (in-process) | manifest + `vision.json` | Chooses a transition **per cut**, the crossfade length, the music mood; records all 9–10 decisions with reasons. Then `extras.write_captions()` runs | **`edit_plan.json`** + `captions.json`; mutates `settings` | **RULE-BASED** — explicitly not a model (`director.py:6`) |
| 6 | `sheets` | `build_video.py` → `stage_sheets()` (subprocess) | manifest | 3×2 tiled contact sheet per planned segment | `look/segNN.jpg` | **REAL** |
| 7 | `cut` | `build_video.py` → `stage_cut()` (subprocess) | manifest + `captions.json` | Per segment: `trim` → `fps=30` → scale/crop to canvas, or **blur-pad** (`gblur=sigma=25`) when orientation differs → `drawtext` burns the caption in → x264 at `AT_CRF` | `parts/pNN.mp4` (video only, `-an`) | **REAL** — heaviest stage (weight 31) |
| 8 | `join` | `build_video.py` → `stage_join()` (subprocess) | `parts/*.mp4` | Chains `xfade`, taking transition *i* from `XFADES[(i-1) % len(XFADES)]` — **this is where the director's per-cut list lands**; records the real offsets | `_joined.mp4`, `transitions[]`, `total_duration` | **REAL** |
| 9 | `music` | `server/extras.py` → `music_stage()` (in-process) | `_joined.mp4` + transition marks | Generate (`beatbed.py` or `bgm_stdlib.py`) / fit an uploaded track / silence. If `keep_audio ∈ {mix,only}`, `source_audio()` **rebuilds the original audio** from the planner's in/out points with `acrossfade` at the same offsets; `mix_beds()` ducks the score with `sidechaincompress` | `score.wav`/`track.wav`/`mixed.wav`, `_audio.json` | **REAL** DSP; engine choice is **RULE-BASED** |
| 10 | `master` | `server/extras.py` → `master_stage()` (in-process) | video + audio bed | Optional `setpts` retime; mux with `loudnorm=I=-15:TP=-1.5:LRA=11`; AAC 192k; `+faststart`; thumbnail from the middle of the strongest segment; **then verifies**: counts decoded frames, measures integrated LUFS from the ebur128 *Summary* block, runs `blackdetect` | **the final MP4**, `thumbnail.jpg`, `output.json` | **REAL** |

### Dead code in the pipeline

`build_video.py` also defines `stage_music()` (line 741) and `stage_mux()` (line 762), registered in its own `STAGES` dict (line 914). **The server never calls either** — `extras.music_stage` and `extras.master_stage` supersede them. They are reachable only from the command line, and `stage_music` would fail as written because it requires `bgm_stdlib.py` to have been copied into the output folder (line 745). Roughly 150 lines of unreachable code, including `write_notes()` which generates a `NOTES.md` the app never produces.

---

## 6 · Director / agents — the section that matters most

### Does a real AI / LLM / vision model exist?

**Yes — exactly one, and it is optional and off by default.**

`server/vision.py` is a genuine vision-model integration:

- `sample_frames()` (line 120) runs ffmpeg to decode **real JPEG frames** from each clip, evenly spaced, scaled to ≤768 px, and base64-encodes them.
- `_ask()` (line 153) builds a provider-correct payload — Anthropic `image` content blocks with `{"type":"base64","media_type":"image/jpeg"}`, or OpenAI `image_url` with a `data:` URI.
- `_post()` (line 138) sends it over HTTPS with `x-api-key` + `anthropic-version`, or `Authorization: Bearer`.
- Two passes: per-clip **with images**, then a cross-clip pass **text-only** (cheaper and more accurate).

**It is disabled unless `AT_VISION_API_KEY` is set** (`available()`, line 87). With no key it logs why and returns `None`, and the build completes on the rule-based path. That fallback is the *normal supported state*, not an error — and it is almost certainly the state this project has been running in, since the author's own `.env` is absent from the folder.

Everything the model returns is treated as untrusted: `_clean_clip()` (line 249) clamps every enum to a fixed allow-list, clamps every number to a range, truncates every string, and caps the moment list at 6. Cross-clip names are clamped to clips actually uploaded (lines 352–359). A model returning nonsense degrades to the rule-based path rather than corrupting the edit.

**It is the only model that touches an edit decision, and the only outbound network call in the project's own code.** No LLM writes copy, chooses clips, or plans the edit.

**There is, however, a second real model in the system**, and it would be dishonest to omit it from a report about what is and is not AI: `extras._get_whisper_model()` (`extras.py:133`) loads **OpenAI Whisper `tiny`** via `faster-whisper` and runs it on every uploaded voice note (`extras.py:149`). It is genuine local ML inference, `faster-whisper>=1.0` is a declared dependency, and on first use the library downloads its weights from the HuggingFace hub — a second network path, outside project code. It is excluded from the "agents" framing for a good reason: **it transcribes the operator's spoken notes, never the footage, and its output influences no editing decision** — the transcript is offered as text the operator may paste into the script box (`app.js:778–788`), and nothing more. But "the only model in the codebase" would be wrong, and §7 and §15 both record this feature as real.

### Is there an agent / orchestrator?

There is an **orchestrator**: `jobs._run()` (`jobs.py:269`) — a sequential, fixed-order loop over 10 stages in a background thread, with cooperative cancellation and per-stage persistence. It is a pipeline runner, **not** a planning or reasoning agent. There is no tool-calling loop, no reflection, no dynamic re-planning, no agent-to-agent negotiation.

There are **five named "agents"** declared in `adapter.AGENTS` (`adapter.py:80`), which group the ten stages for reporting and UI purposes:

| Agent | Stages owned | Weight | Calls a model? |
|---|---|---|---|
| **Vision Agent** | `vision` | 14 | **YES — the only one** (optional) |
| **Editor Agent** | `probe`, `measure`, `plan`, `direct` | 29 | No |
| **Animation Agent** | `join` | 15 | No |
| **Merge Agent** | `sheets`, `cut`, `master` | 50 | No |
| **Music Agent** | `music` | 8 | No |

The codebase is candid about this. `adapter.py:67–71`: *"'Agent' here means an autonomous component with its own inputs, its own decisions and its own reported status — NOT that each one is a language model. Exactly one of them (Vision) calls a model; the other four are deterministic."* The UI carries the same distinction as a per-agent badge reading `vision model` or `rule-based`.

**Honest characterisation: this is a well-structured deterministic pipeline with named subsystems and one optional vision call. It is not a multi-agent AI system.**

### What currently makes decisions

| Decision | Made by | Nature |
|---|---|---|
| Which clips survive | `stage_plan()` thresholds + optional vision duplicate flags | RULE-BASED (+ AI advisory) |
| Running order | `stage_plan()` sequencing block | RULE-BASED (+ AI hook nomination) |
| Trim in/out per clip | `best_window()` | RULE-BASED |
| Screen time per clip | sqrt-weighted budget | RULE-BASED |
| Transition per cut | `director.direct()` tables | RULE-BASED (+ AI shot-kind input) |
| Crossfade length | `director.direct()` bands | RULE-BASED |
| Music mood | `MOOD_BANDS` | RULE-BASED |
| Captions | Operator's script; vision fills blanks only | USER INPUT (+ AI fallback) |
| Speed, resolution, quality, orientation, target length, music source, keep-audio | Operator | USER INPUT |

### Which decisions are AI-controlled

**None are AI-*controlled*. Three are AI-*informed*, and only when a key is set:**

1. **Duplicate rejection** — `vision.json.cross.duplicates` becomes `--drop CLIP::reason` (`adapter.py:213`). The pipeline rejects **only clips the model named** and never invents a drop of its own (`build_video.py:408–417`).
2. **Hook nomination** — `cross.hook` becomes `--hook CLIP`, and only when `sequence == "hook"` (`adapter.py:216`). Motion ranking still orders everything after it.
3. **Transition choice** — `KIND_XFADE` (`director.py:64`) keys off the model's `kind` for a pair; it sits *between* the orientation override and the motion fallback in the precedence chain (`director.py:187–194`).

Plus one conditional: **captions**, used only where the operator wrote nothing *and* the model reported ≥60 confidence (`director.py:254–258`).

The model **never** decides resolution, speed, quality, trim points, pacing, crossfade length or music.

### Where the edit plan is generated

`server/director.py`, function `direct(job_id, settings, on_line)`, **line 94**. Called from `jobs.py:292`, in stage 5 of 10 — **after** `plan` (so it can see the chosen segments) and **before** `cut`/`join` (so its choices drive the render). Written to `outputs/<job_id>/edit_plan.json` at `director.py:306`.

### Exact schema of `edit_plan.json`

```jsonc
{
  "vision": {
    "used":        false,          // bool — whether per-clip vision data existed
    "model":       null,           // string|null — e.g. "claude-sonnet-4-5"
    "provider":    null,           // string|null — "anthropic" | "openai"
    "clips_seen":  0,              // int — how many clips the model described
    "duplicates":  [],             // [[clipA, clipB], ...]
    "hook":        null,           // string|null — clip nominated to open
    "story":       ""              // string — one-sentence sequence rationale
  },
  "median_motion":   1.84,         // float, 3dp — median mean_ydif across segments
  "busy_threshold":  1.84,         // float, 3dp — max(median, 0.05); "busy" cutoff
  "segments":        4,            // int — number of segments in the cut
  "transitions":     ["fade", "wipeup", "fadewhite"],   // one per CUT (= segments-1)
  "decisions": [                   // ordered; 9 without vision, 10 with
    {
      "agent":      "editor",      // "vision"|"editor"|"animation"|"music"
      "what":       "Which clips to use",
      "chose":      "4 of 5",
      "why":        "every clip that has movement in it was kept; …",
      "decided_by": "measurement"  // "measurement" | "you" | "vision"
    }
  ]
}
```

**`decided_by` is the field that matters for honesty**, and the UI renders it as a visible badge:

| Value | UI badge | Meaning |
|---|---|---|
| `"measurement"` | `decided` | A rule fired on measured numbers. **Not AI.** |
| `"you"` | `yours` | The operator's setting won; the director stood down. |
| `"vision"` | `saw it` | A real model looked at real frames. |

The decisions emitted, in order (`director.py:127–291`):

| # | `what` | `agent` | `decided_by` | Present when |
|---|---|---|---|---|
| 0 | What the footage is | vision | vision | only if vision ran |
| 1 | Which clips to use | editor | measurement | always |
| 2 | Running order | editor | measurement | always |
| 3 | Trim points | editor | measurement | always |
| 4 | Pacing | editor | measurement | always |
| 5 | Transitions | animation | measurement *or* you | always |
| 6 | Transition length | animation | measurement *or* you | always |
| 7 | Music | music | measurement *or* you | always |
| 8 | Captions | vision *or* editor | vision *or* you | always |
| 9 | Speed | animation | **you** | always — the director deliberately refuses this one |

### How the plan reaches ffmpeg

This is the critical path, and it is worth stating precisely because it is the difference between a plan that is *displayed* and a plan that is *applied*:

```
director.direct()
   └─ settings["transition_list"] = ["fade","wipeup","fadewhite"]   director.py:199
   └─ settings["transition_duration"] = 0.4                          director.py:225
   └─ settings["mood"] = "bright"                                    director.py:240
             │  (same dict object; jobs.py passes job.settings by reference)
             ▼
adapter.env_for(settings)                                            adapter.py:221
   └─ AT_XFADES = "fade,wipeup,fadewhite"        (transition_list WINS over style)
   └─ AT_XFADE_T = "0.4"
   └─ AT_CANVAS  = "1920x1080"
   └─ AT_CRF     = "18"
             │  passed as env= to subprocess.Popen                   adapter.py:305
             ▼
build_video.py module level                                          build_video.py:47-54
   └─ T      = float(os.environ.get("AT_XFADE_T") or 0.4)     ← every one of
   └─ CRF    = os.environ.get("AT_CRF") or "18"                 these falls back
   └─ XFADES = os.environ.get("AT_XFADES") ? split(",")         to its ORIGINAL
              : [fadewhite, fade, wipeup, smoothleft, …]        value, so the CLI
                                                                behaves identically
                                                                with nothing set
             ▼
stage_join()                                                         build_video.py:726
   └─ xfade=transition=XFADES[(i-1) % len(XFADES)]:duration=T:offset=…
             ▼
                    ffmpeg writes those pixels
```

Because the director emits **exactly one entry per cut**, the modulo index in `stage_join` resolves to precisely the director's choice for that cut — the list never wraps.

### Does every decision actually affect the rendered pixels?

Traced individually:

| Decision | Reaches pixels? | Route |
|---|---|---|
| Clip selection | **YES** | `manifest.segments` → `stage_cut` renders only those |
| Running order | **YES** | `s["index"]` → `parts/pNN.mp4` → join order |
| Trim points | **YES** | `trim=start=…:end=…` in `fit_filter` |
| Pacing / screen time | **YES** | segment `duration` → clip length and xfade offsets |
| Transition per cut | **YES** | `AT_XFADES` → `xfade=transition=` |
| Crossfade length | **YES**, with a caveat ⚠ | `AT_XFADE_T` → `T` → `duration=` **and** the offset arithmetic. **But `plan` runs at stage 4 and `direct` at stage 5**, so on `auto` the variable is still unset when the planner computes `planned_total` with the default `T = 0.4` (`build_video.py:47, 540`); the director then writes 0.25 or 0.6 (`director.py:219–225`) and `stage_join` recomputes the real offsets and `total_duration` with the new value (`build_video.py:696–699`). The **rendered** length is correct; `manifest.planned_total` and `manifest.total_duration` simply disagree whenever the director picks anything other than 0.4 |
| Captions | **YES** | `captions.json` → `drawtext` burnt into the frame |
| Music mood | **YES (audio)** | engine + style selection in `generate_music` |
| Keep-audio | **YES (audio)** | `source_audio` + `mix_beds` |
| Speed | **YES** | `setpts` retime + `atempo`, marks rescaled |
| Resolution | **YES** | `AT_CANVAS` — segments render **at** target size, never upscaled afterwards |
| Quality | **YES** | `AT_CRF` → x264 |
| Orientation | **YES** | `--orient` → canvas choice → blur-pad decisions |
| Vision `duplicates` | **YES** | `--drop` → `stage_plan` rejection |
| Vision `hook` | **YES** | `--hook` → opener |
| Vision `kind` | **YES** | `KIND_XFADE` → transition name |
| Vision `caption` | **YES, conditionally** | only into blanks, only at ≥60 confidence |
| Vision `story` | **NO** | Written into the plan's `why` text (`director.py:133`) and displayed. Reaches the reader, never the render |
| Vision `notes` | **NO** | Fed into the cross-clip prompt (`vision.py:345`), so it shapes the model's *own* second pass — but no stage consumes it |
| Vision `order`, `interesting_moments`, `hook_score`, `phase` | **NO** | Requested, validated, stored in `vision.json`, displayed — and **read by nothing.** Verified by grep: outside `vision.py` these four identifiers appear nowhere in `server/`, `pipeline/` or `app.js`. `cross.order` is computed and clamped at `vision.py:359` and then dropped |

**One genuine gap found:** the vision model's `order` (its proposed story sequence), `interesting_moments` (which could drive trim points), `phase` and `hook_score` are all captured, validated and displayed — but never influence the edit. Confirmed by grep: outside `vision.py`, those four identifiers do not appear anywhere in `server/`, `pipeline/` or `app.js`. `interesting_moments` in particular is the obvious missing link between "the model saw the gears engage at 4.5 s" and "trim this clip around 4.5 s"; today `best_window()` picks the trim purely on motion and ignores it.

---

## 7 · Clip analysis — what is actually measured

| Analysis | Status | How, and where |
|---|---|---|
| **Duration, width, height, codec, fps** | **REAL** | ffprobe. `adapter.media_summary()` (`adapter.py:327`) and `build_video.probe_one()` (line 148) |
| **Rotation** | **REAL** | ffprobe `stream_side_data=rotation`, falling back to `stream_tags=rotate`; w/h swapped when `\|rot\| % 180 == 90` |
| **Orientation** | **REAL** | Derived: `vertical if h > w`. Batch canvas chosen by **total duration** per orientation, not clip count (`build_video.py:227–234`) |
| **Audio track present** | **REAL** | ffprobe stream check |
| **Recording timestamp** | **REAL** | `format_tags=creation_time`, falling back to `stream_tags=creation_time` (`build_video.py:169`). Drives the "Build order" sequence. Absent on screen recordings and re-encodes — the UI warns when too few clips carry one |
| **Motion (how much)** | **REAL** | `signalstats` YDIF — per-frame luma difference — sampled at 10 fps (5 fps over 2 min), downscaled to 160px. → `mean_ydif`, `max_ydif` |
| **Motion (how often)** | **REAL** | `live_share` = fraction of sampled frames whose YDIF exceeds 0.02 (`build_video.py:354`). This is what distinguishes a *calm* clip from a *dead* one, and it is the fix that stopped the system discarding close-ups and talking heads |
| **Freeze detection** | **REAL** | `freezedetect=n=-38dB:d=0.4` at 6 fps → `freeze_runs`, `freeze_share`. **Used to steer window selection *inside* a clip, deliberately NOT used to reject one** — at that threshold it calls a locked-off interview 100% frozen (`build_video.py:62–65`) |
| **Duplicate detection — iOS pairs** | **RULE-BASED** | Regex `^(IMG_)E(\d+)(\.\w+)$`; keeps the Apple-rendered `E` take, drops the raw twin (`build_video.py:217`). Filename-based only |
| **Duplicate detection — content** | **AI** | Only when vision runs: the cross-clip pass names pairs, clamped to real uploads. **NOT IMPLEMENTED** without a key — no perceptual hashing, no embedding similarity, no histogram comparison |
| **Scene understanding** | **AI when enabled, otherwise NOT IMPLEMENTED** | `subject`, `shot` (close-up/medium/wide), `kind` (assembly/demonstration/reveal/detail), `phase` (beginning/middle/end) — all from the model. Nothing rule-based approximates this |
| **Object understanding** | **AI when enabled, otherwise NOT IMPLEMENTED** | Only as the model's free-text `subject`. **No object detector, no classifier, no bounding boxes, no tracking anywhere in the codebase** |
| **LEGO-specific understanding** | **AI when enabled, otherwise NOT IMPLEMENTED** | Entirely a property of two prompt strings telling the model it is looking at LEGO. **No LEGO part database, no brick recognition, no set identification, no build-step tracking.** The system does not "know" LEGO in any structural sense |
| **Speech / audio content** | **PARTIAL** | Presence of an audio track: REAL. **Voice-note transcription: REAL when `faster-whisper` is installed** (`extras.transcribe_voice_file`, line 144, `WhisperModel("tiny", cpu, int8)`). **Transcription of the uploaded clips themselves: NOT IMPLEMENTED** — no clip is ever transcribed; speech in the footage is never analysed, only passed through as audio |
| **Loudness of source clips** | **NOT IMPLEMENTED** | Loudness is measured only on the **finished** file (`extras.loudness`) |
| **Shot-boundary / scene-cut detection within a clip** | **NOT IMPLEMENTED** | No `scdet`, no `select='gt(scene,…)'`. A clip containing several distinct shots is treated as one continuous shot |
| **Face / person detection** | **NOT IMPLEMENTED** | — |
| **Quality assessment** (focus, exposure, shake) | **NOT IMPLEMENTED** | No blur metric, no stabilisation analysis, no exposure check |
| **Colour analysis** | **NOT IMPLEMENTED** | `signalstats` is invoked but only YDIF is parsed |

### Correction to earlier documentation

**Four places still assert that transcription does not exist, and the two most visible are user-facing:**

| Where | What it still says | Status |
|---|---|---|
| **`README.md:583–602`** | An entire titled section, *"## 9 · Known limitation: voice transcription"* — *"Voice notes are recorded, stored, probed and played back — **but not transcribed**"*, and claims `store_voice_note()` returns `{"transcript": null, "transcript_status": "no_engine"}` | **STALE** — contradicted by `extras.py:170–178` |
| **`server/adapter.py:59–60`** | *"No offline STT engine could be installed in this environment, so voice notes are captured and stored but not machine-transcribed"* — sitting directly above `adapter.py:61`, the line that probes for `faster_whisper` | **STALE** |
| `server/extras.py:9` | *"no transcription engine is available in this environment, and we do not pretend otherwise"* | **STALE** |
| `frontend/app.js:719` | *"No transcription is claimed."* | **STALE** |
 `faster-whisper>=1.0` is in `requirements.txt`, `adapter.preflight()` detects it, `extras.transcribe_voice_file()` runs it, `/api/health` reports *"Offline speech-to-text engine (faster-whisper) is ready"*, and `renderVoice()` (`app.js:762`) displays the transcript with an **"Insert into script"** button. The feature is real; the comments describing it as absent were not updated. Worth fixing for a reader's sake — the runtime behaviour is correct either way, because it is gated on an actual import check.

---

## 8 · Editing decisions — who decides what, how, and where

| Decision | WHO decides | HOW | WHERE in code |
|---|---|---|---|
| **Clip selection** | ⚠️ RULE-BASED (+ AI advisory) | Reject only if `live_share < 0.15` **AND** `mean_ydif <= max(0.05, batch_peak × 0.02)`. If the rule would drop more than 34% of the batch, **all rejections are reverted** — calm footage is still footage. Vision may additionally name duplicates | `build_video.py:404–438`; constants at 83–90; vision drops at 408–417 |
| **Ordering** | ⚠️ RULE-BASED (+ AI hook) | Three modes. `hook`: liveliest by `score` opens, rest in upload order, slow-mo last. `recorded`: chronological by container `creation_time`, needs ≥2 clips carrying one or it falls back with an explanation. `uploaded`: filename order. Vision may nominate the opener | `build_video.py:494–538`; mode from `--sequence`, `adapter.py:204` |
| **Trimming** | ⚠️ RULE-BASED | `best_window()` slides a window of the allotted length in 0.2 s steps and scores each by mean motion **penalised by freeze overlap** (`motion × (1 − 0.8 × frozen_fraction)`); highest wins | `build_video.py:366–391` |
| **Pacing** | ⚠️ RULE-BASED | Budget shared by **√(mean_ydif)** — square root deliberately, so one frenetic clip cannot eat the reel — clamped to `MIN_SEG 2.5` / `MAX_SEG 12.0` / the clip's own length, then grown in ≤0.5 s increments to reach target | `build_video.py:452–471` |
| **Captions** | 👤 USER INPUT (+ AI fallback) | The operator's script is split one line per segment; sentence split, then even chunking as fallbacks; trimmed to 58 chars on a word boundary. **Vision captions fill only genuine blanks, only at ≥60 confidence, and only when no script was written** | `extras.split_script()` line 65, `write_captions()` line 106; gate at `director.py:254–258` |
| **Music** | ⚠️ RULE-BASED (+ USER override) | `mood == "auto"` → `MOOD_BANDS` on median motion: ≥2.5 `beat`, ≥0.8 `bright`, else `lofi`. Mood → engine: beatbed (numpy/scipy) or bgm_stdlib. Score generated **against the real transition timestamps** so hits land on cuts | `director.py:76–83, 237–247`; `extras.MOODS` line 38, `generate_music()` line 197 |
| **Audio mixing** | 👤 USER INPUT | `keep_audio`: `none` (drop) / `mix` (`sidechaincompress` ducks the score under the source) / `only` (no score). Source audio is **rebuilt** from the planner's in/out points with `acrossfade` at the video's own xfade offsets | `extras.source_audio()` line 288, `mix_beds()` line 362 |
| **Transitions** | ⚠️ RULE-BASED (+ AI input, + USER override) | Precedence: **(1)** orientation differs → always `fade`; **(2)** vision `kind` pair in `KIND_XFADE`; **(3)** next shot is a `reveal` → `fade`; **(4)** motion band pair in `PAIR_XFADE`. If the operator picked a named style, the director stands down entirely | `director.py:178–215`; tables at 52–72 |
| **Effects** | ⚠️ RULE-BASED, non-configurable | The only visual effect is the **blur-pad**: when a clip's orientation differs from the canvas, the background is a scaled, `gblur=sigma=25`, darkened copy of itself with the correctly-fitted foreground overlaid. Applied automatically, no control | `build_video.py:607–638` (`fit_filter`) |
| **Speed** | 👤 USER INPUT — deliberately | `slow` 0.75 / `normal` 1.0 / `fast` 1.35 / custom clamped 0.4–2.5. **The director explicitly refuses to decide this** and records why. Applied as `setpts` on video; the score length and transition marks are rescaled by the same factor | `adapter.speed_factor()` line 289; `extras.retime()` line 184; refusal at `director.py:283–291` |
| **Aspect ratio / orientation** | 👤 USER INPUT with a REAL auto | `auto` → whichever orientation has **more total duration** wins the canvas | `build_video.py:227–235` |
| **Resolution** | 👤 USER INPUT | 720p/1080p/1440p/4K → `AT_CANVAS`, swapped w/h for vertical. Segments render **at** the target size — never upscaled afterwards | `adapter.env_for()` lines 229–236 |
| **Quality** | 👤 USER INPUT | draft 24 / standard 20 / high 18 / master 15 → `AT_CRF` | `adapter.QUALITY` line 260 |
| **Target length** | 👤 USER INPUT with a REAL default | Blank → pipeline default: 58 s vertical, 120 s horizontal | `build_video.py:242`; `--target` override |
| **Final structure** | ⚠️ RULE-BASED | Segments → xfade chain → optional retime → mux → loudnorm → faststart. Fixed shape; no titles, no intro/outro, no B-roll insertion, no chapter logic | `build_video.stage_join()` + `extras.master_stage()` |

**Tally over the 15 rows above: 0 AI-controlled · 5 purely rule-based · 3 rule-based with optional AI input (clip selection, ordering, transitions) · 6 user input · 1 user input with an optional AI fallback (captions).**

---

## 9 · SSE and progress

### Server

`GET /api/jobs/{job_id}/events` (`main.py:529`) returns a `StreamingResponse` of `text/event-stream` with `Cache-Control: no-cache` and `X-Accel-Buffering: no` (the latter so nginx does not buffer the stream).

The generator:

1. Immediately emits `{"kind":"snapshot","job":<full snapshot>}` so a client joining mid-build is instantly correct.
2. Loops every **0.5 s**, emitting every log entry whose `seq` exceeds the cursor.
3. When `status ∈ {done,error,cancelled}` **and** the backlog is drained, emits `{"kind":"final","job":…}` and returns.
4. Gives up after `idle > 900` iterations (~7.5 min of silence) with `{"kind":"idle"}`.

Accepts `?after=N` to resume from a known sequence number.

### Event kinds

| `kind` | Emitted by | Payload |
|---|---|---|
| `snapshot` | stream open | full `Job.snapshot()` |
| `status` | `jobs.start`, `_run` | `status`, plus `output` or `error` |
| `stage` | `_run` per stage | `stage`, `status` (`running`/`done`/`skipped`), `label`, `progress` |
| `line` | `Job.line()` | `stage`, `text` (truncated to 500 chars) |
| `final` | stream close | full snapshot |
| `idle` | timeout | — |

Every event carries `seq` (monotonic) and `t` (seconds since start). The log is capped at 4,000 entries, dropping the oldest 1,000 when exceeded (`jobs.py:66`).

### The ten stage ids, in order

`probe` → `measure` → `vision` → `plan` → `direct` → `sheets` → `cut` → `join` → `music` → `master`

### How the frontend receives it

`listen(jobId)` (`app.js:889`) opens an `EventSource`. `line` events append to the log; `stage` and `status` events trigger a `refreshJob()` (a full `GET /api/jobs/{id}`) so the stage list and ring re-render from authoritative state rather than from accumulated deltas. On `final` with `status === "done"` it renders the output and auto-advances to step 4.

Reconnection is left to `EventSource`'s own retry; `onerror` is intentionally a no-op.

### Stage status values

`pending` → `running` → `done` | **`skipped`** | `error`. `skipped` exists for one reason: the vision stage completes whether or not a model was reachable, and marking that "done" would put a green tick on a pass that never looked at anything (`jobs.py:319–322`). Skipped stages count as complete for progress arithmetic (`jobs.py:82`).

---

## 10 · Files, database, storage

**There is no database.** Everything is files on disk, inside the application folder.

```
alpha_titans/
├── uploads/<job_id>/
│   ├── NN_originalname.mp4        raw clips; numeric prefix = planner read order
│   └── _voice/<name>.webm         voice notes
├── outputs/<job_id>/
│   ├── job.json                   job state — survives a restart
│   ├── manifest.json              THE pipeline's own state; source of truth
│   ├── edit_plan.json             every decision + its reason
│   ├── vision.json                what the model saw (only if it ran)
│   ├── captions.json              one line per segment
│   ├── _audio.json                audio plan: duration, factor, marks, track
│   ├── output.json                verified figures for the finished file
│   ├── _posters/*.jpg             cached clip thumbnails for the UI
│   ├── look/segNN.jpg             contact sheets
│   ├── measure/*.ydif.txt         raw per-frame motion series
│   ├── parts/pNN.mp4              rendered segments (video only)
│   ├── _joined.mp4                after crossfades
│   ├── _retimed.mp4               only if speed ≠ 1
│   ├── score.wav / track.wav / source_audio.wav / mixed.wav
│   ├── thumbnail.jpg
│   └── <TITLE>_<orientation>.mp4  ★ the deliverable
└── data/music/                    the shared music library + .json mood sidecars
```

| Concern | Implementation |
|---|---|
| **Job storage** | In-memory `dict` `jobs._JOBS`, mirrored to `outputs/<id>/job.json` on every state change (`_persist`, `jobs.py:191`). `listing()` scans `outputs/` to recover jobs from earlier server runs. A job found in `running` state after a restart is **correctly re-flagged as `error`** — "server restarted while this build was running" (`jobs.py:229`) |
| **Clip storage** | Plain files under `uploads/<job_id>/`. Order is encoded in the filename prefix, so the pipeline's `sorted(glob)` follows the operator's drag order without any index file |
| **Artifacts** | All under `outputs/<job_id>/`. Nothing is written outside the application folder — every path resolves through `config.py` relative to `APP_ROOT` |
| **Edit plans** | `edit_plan.json`, written once per build, read back by `/plan` and by `Job.snapshot()` |
| **Logs** | **In memory only** (`job.log`), capped at 4,000 entries, **not persisted**. Restart the server and the log of a finished build is gone — though `job.json`, `manifest.json` and `output.json` survive |
| **Metadata** | Split across `manifest.json` (pipeline truth), `job.json` (UI/job truth), `output.json` (verified results), `_audio.json` (audio plan) |
| **Cleanup** | **Only manual**, via `DELETE /api/jobs/{id}`, which `rmtree`s both folders. **There is no retention policy, no disk quota, no automatic pruning, no temp-file cleanup.** Intermediates (`parts/`, `_joined.mp4`, `measure/`, `_retimed.mp4`) are kept forever — deliberately, since the pipeline's resumability depends on them, but it means disk grows without bound. The README estimates a few hundred MB per 1080p/high two-minute build |
| **Concurrency** | A `threading.RLock` guards `_JOBS` and the log sequence counter. **Nothing limits how many builds run at once** — N concurrent builds means N ffmpeg processes competing for CPU |

---

## 11 · Configuration

**No secret values appear below — variable names and purposes only.**

### Server and paths

| Variable | Purpose | Default |
|---|---|---|
| `AT_HOST` | Bind address | `127.0.0.1` |
| `AT_PORT` | Port | `8000` |
| `AT_FFMPEG` | ffmpeg binary or full path | `ffmpeg` |
| `AT_FFPROBE` | ffprobe binary or full path | `ffprobe` |
| `AT_UPLOADS_DIR` | Raw clips | `uploads` |
| `AT_OUTPUTS_DIR` | Builds | `outputs` |
| `AT_MUSIC_DIR` | Music library | `data/music` |
| `AT_FRONTEND_DIR` | UI files | `frontend` |
| `AT_PIPELINE_DIR` | Pipeline scripts | `pipeline` |
| `AT_CORS_ORIGINS` | Comma-separated origins; empty = same-origin only | empty |

### Limits

| Variable | Purpose | Default |
|---|---|---|
| `AT_MAX_CLIP_MB` | Per-clip upload cap | `2048` |
| `AT_MAX_TRACK_MB` | Music track cap | `120` |
| `AT_MAX_VOICE_MB` | Voice note cap | `40` |

### Render overrides — set by the app, not usually by the operator

| Variable | Purpose | Set by |
|---|---|---|
| `AT_CANVAS` | Output `WxH` | `adapter.env_for()` from the resolution control |
| `AT_CRF` | x264 quality | `adapter.env_for()` from the quality control |
| `AT_XFADES` | Comma-separated per-cut transition names | `adapter.env_for()` — **this is how the director's plan reaches ffmpeg** |
| `AT_XFADE_T` | Crossfade seconds | `adapter.env_for()` |
| `AT_PRESET` | x264 preset | Read by `build_video.py:50`; **never set by the server** — always defaults to `fast` |
| `AT_CPUS` | Referenced in `docker-compose.yml` only | — |

### Vision layer — the only external service

| Variable | Purpose |
|---|---|
| **`AT_VISION_API_KEY`** | **The master switch.** Unset ⇒ the entire vision layer is disabled and the build runs rule-based. This is the only credential the system uses |
| `AT_VISION_PROVIDER` | `anthropic` (default) or `openai` |
| `AT_VISION_MODEL` | Model id; provider-appropriate default if unset |
| `AT_VISION_BASE_URL` | Endpoint override — self-hosted, gateway or proxy |
| `AT_VISION_API_VERSION` | Anthropic API version header; defaults to `2023-06-01` |
| `AT_VISION_FRAMES` | Frames sampled per clip, clamped 1–8, default 4 |
| `AT_VISION_TIMEOUT` | Seconds per request, floor 10, default 90 |
| `AT_VISION_MAX_CLIPS` | Cap on clips analysed per build, default 12 |

### External services required

| Service | Required? | Notes |
|---|---|---|
| **ffmpeg / ffprobe ≥ 5.0** | **Yes, absolutely** | Not a pip package. Nothing works without it. `/api/health` reports `ok: false` if missing |
| Anthropic or OpenAI API | **No — optional** | Only for the vision layer. Absent, everything else runs |
| numpy + scipy | No | Enables the richer `beatbed.py` music engine; `bgm_stdlib.py` is the pure-stdlib fallback |
| faster-whisper | No | Enables voice-note transcription; degrades to storage-only |
| Any cloud, database, queue or auth provider | **No** | None exist in the codebase |

`config.py` reads a `.env` next to the app folder with a 14-line hand-rolled parser (`_load_dotenv`, `config.py:17–30`) — **real environment variables always win**, because it uses `os.environ.setdefault`. **No `.env` file is present in the audited folder**, which means the vision layer is currently disabled on this machine.

---

## 12 · Testing

### Important caveat on provenance

I could **not** execute the test suite as part of this audit: the results below are (a) recorded in `README.md` §12, and (b) from the verification runs conducted when the deployment package was built. **The code has changed since those runs** — voice transcription was added to `extras.py`, `adapter.py`, `main.py`, `app.js` and `requirements.txt`. Nothing in the transcription change touches the render path, but **the test results below predate it and have not been re-run against the current code.**

There is also **no automated test suite in the repository** — no `pytest`, no `tests/` directory, no CI configuration. All testing was manual or driven by one-off scripts that were not shipped.

### Route tests

| Test | Result |
|---|---|
| Sweep of every declared route, with the route list **extracted from `main.py`** rather than hand-written, so a new route shows as untested instead of silently passing | **30 declared, 30 exercised, 29 returning 2xx.** The one non-2xx is `GET …/notes` before a build finishes — a correct 404 |
| Cancel → restart on the same job | **Passed after a fix.** Originally failed: `cancel()` raised a `threading.Event` that `start()` never lowered, so every build after one cancel aborted at stage 1 while `/start` still returned 200 |
| Upload rejection paths | Non-video extension, oversize file, unreadable file all rejected with reasons |
| Path traversal on clip/poster/music/build-file routes | Rejected with 400 |

### Rendering tests

| Test | Result |
|---|---|
| Three-clip regression (2 landscape, 1 vertical, 31 s source) | "8/8 stages green", ~65 s at 720p/draft. **327 frames for 10.9 s at 30 fps — exact**. ⚠ The "8/8" is quoted verbatim from `README.md:843` and is **stale** — the pipeline has had ten stages since `vision` and `direct` were added. Treat that run as predating the current architecture |
| Five LEGO shot types uploaded deliberately scrambled, set to Build order | Came out in true build order from recording timestamps alone. **904 of 904 frames**, −14.1 LUFS, AAC, vertical clip blur-padded onto the horizontal canvas |
| Six deliberately unlike clips in one batch | Five real clips combined into one 28.43 s video; only the dead slate excluded, named with its reason. **853 of 853 frames**, no black frames, −14.2 LUFS |
| Audio verified **by spectrum**, not by trusting the setting | Talking head's 180 Hz source tone dominant in its window (energy 1722); travel clip's pink noise present at 1–3 kHz (2130); the genuinely silent product clip's window has neither (32/329) |
| Resolution / quality overrides | 720p honoured (`canvas [1280,720]`); vertical correctly swapped to 720×1280 |
| Target length | `--target 15` → planner allocated exactly 15.0 s |
| Speed | `fast`: 14.9 s → 11.03 s; `slow`: longer. Both frame-exact |
| Loudness across music modes | Generated −14.3, uploaded track −15.4, silent −70 LUFS (correct, reported as "Silent, as requested") |
| Final MP4 decoded independently of the app | 1224 frames = 40.8 s × 30 fps exactly; −15.2 LUFS; no decode errors |
| Clean-room test: extract the ZIP to a fresh directory and run the whole flow from it | Produced a real 1572-frame MP4; nothing written outside the folder |

### SSE tests

| Test | Result |
|---|---|
| Stream opens and emits before the build finishes | Passed — first frame is the snapshot |
| All 10 stages transition live | Passed, vision on and off |
| `skipped` renders distinctly from `done` | Passed — "did not run — nothing to report" |
| Auto-advance to step 4 on `final` | Passed |

### Frontend tests

| Test | Result |
|---|---|
| Full UI flow driven in a real browser (Playwright/Chromium), vision on and off | Passed both |
| JS console errors | **None** |
| Horizontal overflow at 1440 px and 390 px | **Zero** |
| All 10 edit-plan decisions carry an agent tag | Passed — 0 untagged |
| Agent badge reflects **the build being viewed**, not the server's current key | Passed after a fix — an old build keeps its badge when the key is removed |
| Palette check on rendered pixels | 0.00% pure white, 0.00% pure black, 0.00% purple |
| Hover states measurably change computed style | Passed on all interactive surfaces |

### LEGO end-to-end tests

Covered above. **Important qualification:** the LEGO tests used *synthetic stand-in clips* generated to have the motion characteristics of each shot type (busy assembly, steady gearbox, near-still close-up, etc.) — **not real LEGO footage**. They prove the *pipeline* handles that range of motion profiles. They do **not** prove the vision model correctly identifies LEGO builds, because no real model was ever called.

### Error and security tests

| Test | Result |
|---|---|
| Vision with a **broken** endpoint (bad key) | Produced a result byte-identical to the no-key run; the vision stage finished without failing the build |
| Vision returning unusable JSON | Falls back to rule-based; logged |
| Missing ffmpeg | `/api/health` reports `ok: false`; `run.sh`/`run.bat` refuse to start with a specific message |
| Server restart mid-build | Job correctly re-flagged as `error` on restore |
| `shell=True` / `eval` / `exec` / `pickle` audit | **None present** |
| API key leakage into responses or logs | **None** — `describe()` exposes provider/model/base_url only |
| **Authentication / authorisation tests** | **NOT PERFORMED — because there is no authentication to test** |
| **Penetration testing, fuzzing, dependency CVE scan** | **NOT PERFORMED** |

### What was NOT verified, and cannot be claimed

**The vision model's actual judgement has never been tested against a real model.** No API key was available. The integration was exercised against a **local test double speaking the same wire protocol** — which proves the request shape, the image encoding, the headers, the response parsing, the clamping, the effect on the edit plan, and all four failure modes. It proves nothing about whether a real model correctly identifies a gearbox, judges a shot type, or picks a sensible hook. **That is the single largest untested surface in the system.**

---

## 13 · Current limitations — the honest list

### Not AI (the biggest framing issue)

1. **The product presents as an "AI video generation system"; four of five "agents" contain no model.** Step 3 is labelled "AI processing" in the UI. The code is candid internally — `adapter.py:67`, `director.py:6` both say so plainly, and the UI badges each agent `rule-based` or `vision model` — but the top-level framing still overstates. **What exists is a well-engineered deterministic editor with one optional vision call.**
2. **The vision layer is off by default and almost certainly has never run on real footage.** No `.env` exists in the audited folder. Every build this project has produced was almost certainly rule-based end to end.
3. **"AI processing" is the step name even when the vision stage is skipped.** The stage list correctly shows `skipped`, but the heading does not change.

### Missing

4. **No authentication, authorisation, rate limiting or CSRF protection on any of the 30 routes.** Anyone who can reach the port can list, read, download and **irreversibly delete** every job and every uploaded file. Acceptable on `127.0.0.1`; **unacceptable the moment `AT_HOST=0.0.0.0` is set**, which the README documents as the way to demo from another machine.
5. **`AT_CORS_ORIGINS` is a second exposure route, and it is permissive once enabled.** `main.py:45–47` configures `CORSMiddleware` with `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`. Set that variable and every route — including the unauthenticated `rmtree` DELETE — becomes cross-origin callable with credentials from the listed origins. Combined with item 4, a single misconfiguration is enough to expose destructive endpoints to a web page.
5. **No automated test suite.** No pytest, no `tests/`, no CI. Every regression risk is carried by manual testing.
6. **No clip transcription.** Speech in the footage is never analysed. Voice *notes* are transcribed; the clips themselves never are.
7. **No shot-boundary detection.** A clip containing several distinct shots is treated as one continuous shot, so `best_window` can pick a window straddling a cut.
8. **No quality assessment** — blur, shake, exposure and focus are never measured. A badly out-of-focus but high-motion clip will be *preferred* over a sharp still one.
9. **No retention policy, disk quota or cleanup.** Intermediates are kept forever by design; disk grows without bound.
10. **Logs are not persisted.** A restart loses the log of every completed build.
11. **No concurrency limit.** N simultaneous builds means N ffmpeg processes fighting for CPU.

### Mocked

12. **Nothing is mocked.** This is worth stating plainly: I searched for sample data, simulated progress and placeholder state, and found none. Every figure in the UI traces to ffprobe, ffmpeg or the pipeline's own stdout. The test double used during development was never shipped.

### Hardcoded

13. **`FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"`** (`build_video.py:55`) — an absolute Linux path with no fallback and no env override. **On Windows this path does not exist**, so `drawtext` will fail and the `cut` stage will abort. Given this project runs on a Windows laptop, **this is the most likely cause of a failed build with captions**. (A caption-free build is unaffected: `fit_filter` skips `drawtext` when the caption is empty — `build_video.py:627`.)
14. **`FPS = 30`** is hardcoded; all frame-count verification assumes 30 fps.
15. **`MIN_SEG 2.5` / `MAX_SEG 12.0`** segment bounds are not configurable.
16. **Default targets 58 s vertical / 120 s horizontal** are hardcoded.
17. **Transition style presets, mood bands and quality CRF values** are hardcoded tables (reasonable, but not tunable without editing Python).
18. **`WhisperModel("tiny", cpu, int8)`** — model size, device and precision all hardcoded (`extras.py:138`).
19. **`AT_PRESET`** is read but never set by the server; x264 preset is always `fast`.

### Rule-based (working, but not what the label implies)

20. Clip selection, ordering, trimming, pacing, transitions, crossfade length and music mood — all thresholds and lookup tables. They are *good* heuristics with recorded reasoning, and they were hardened against genuinely diverse footage. They are not inference.

### Fragile

21. **`cancel` does not kill the running ffmpeg.** It sets a flag checked *between* stages, so cancelling during `cut` waits for the entire render.
22. **`job_id` is not sanitised the way filenames are.** `safe_name()` is applied to every filename, but `job_id` flows straight into `Path` joins in `adapter.build_dir()`/`src_dir()`. FastAPI's path matching prevents a `/` from appearing, which is what currently makes this safe — but it is a guard by accident rather than by design, and `delete_job` does an `rmtree` at the end of it. **Worth hardening.**
23. **`PATCH /api/jobs/{id}` has no schema validation** — any key can be written into `settings`.
24. **Vision cost is unbounded per build** beyond the 12-clip cap: 12 clips × 4 frames = 48 images plus a text pass, with no token budget or spend ceiling.
25. **The whole system is single-process and in-memory.** No horizontal scaling; two workers would not share `_JOBS`.
26. **`fps=10` motion sampling** means events shorter than ~100 ms are invisible to the planner.
27. **`stage_music` / `stage_mux` in `build_video.py` are dead code** (~150 lines) superseded by `extras.py`, and `stage_music` would fail if invoked. A future maintainer could easily call the wrong one.
28. **Stale documentation in four places** claims voice transcription does not exist, when it now does — including a whole titled README section (`README.md:583–602`) and a comment in `adapter.py:59–60` sitting directly above the line that detects the engine. See §7. The runtime behaviour is correct regardless, because it is gated on a real import check; the documentation is what is wrong.

### Not production-ready

29. Uvicorn single worker, no reverse proxy assumed, no TLS, no health-check endpoint suitable for a load balancer (`/api/health` shells out to `which` on every call), no structured logging, no metrics, no error tracking, no graceful shutdown for in-flight builds, no backup.

---

## 14 · Original intended product

Based **only** on the code and `README.md`:

> A self-contained, locally-run web application that an operator points at a folder of raw LEGO build footage and gets back one finished, captioned, scored, verified video — where the system, not the operator, decides which clips are worth using, what order they tell the story in, where each is trimmed, how long each holds the screen, how each cut is made and what it sounds like; and where **every one of those decisions is shown to the operator with the reason it was made**, so the edit can be argued with rather than merely accepted.

The design commitments visible in the code:

1. **Never fake a capability.** Repeatedly enforced: `/api/health` probes rather than assumes; the vision stage reports `skipped` rather than `done`; the director refuses to write captions it has no basis for; `frames_ok` is computed by counting decoded frames rather than trusting the container.
2. **Explain every decision.** `edit_plan.json` carries a `why` for every entry, and `decided_by` separates measurement from the operator from the model.
3. **Don't rewrite the working pipeline.** `build_video.py` was given a 21-line additive change — four constants reading env overrides with identical fallbacks. `build_video.py.bak` ships alongside for diffing.
4. **Handle the full range of shot types.** The thresholds are batch-relative *specifically* because a close-up of a finished model and a piece to camera measure low and were being discarded.
5. **Be portable.** Every path resolves relative to the app folder; copy it, install ffmpeg, run one script.
6. **Verify the artefact, not the intent.** Frame counts, LUFS from the ebur128 Summary block, black-frame detection — all measured on the finished file.

The gap between intent and implementation is almost entirely one of **labelling**: the product is described as AI-driven, and the decision engine is rule-driven with an optional AI input that has likely never been switched on.

---

## 15 · Actual vs required

| REQUIREMENT | CURRENT IMPLEMENTATION | STATUS | WHAT IS MISSING |
|---|---|---|---|
| Accept any mix of LEGO clips | Any of 5 containers, mixed orientation and resolution, blur-padded onto one canvas | ✅ COMPLETE | — |
| Produce **one** coherent final video | Real MP4, H.264 + AAC, faststart, frame-verified | ✅ COMPLETE | — |
| **AI agents** analyse the clips | 1 of 5 "agents" calls a model, and only with an API key that is not configured here | ⚠️ RULE-BASED / NOT AI | 4 of 5 agents have no model. Vision unproven against a real endpoint |
| Create the best story / sequence | 3 sequencing modes from motion, timestamps or user order; vision may nominate an opener | ⚠️ RULE-BASED | No narrative understanding. `cross.order` from the model is computed then **ignored** |
| Trim and order clips | `best_window()` motion scoring with freeze penalty; three order modes | ⚠️ RULE-BASED | Ignores the model's `interesting_moments`; no shot-boundary detection |
| Generate captions | Operator's script split per segment; vision fills blanks at ≥60 confidence | 🟡 PARTIAL | No caption generation without a vision key — by explicit design |
| Generate music | 2 real synthesis engines, scored against actual transition marks | ✅ COMPLETE | Mood selection is a threshold table, not AI |
| Transitions | One chosen **per cut** from orientation, shot kind and motion; 4 named presets | ✅ COMPLETE | ⚠️ the choice is rule-based |
| Speed / effects | Speed is user-controlled and applied for real; blur-pad is the only effect | 🟡 PARTIAL | No colour grading, stabilisation, zoom, titles or overlays |
| Real video pipeline renders | 6 real ffmpeg stages + 2 real in-process finishing stages | ✅ COMPLETE | — |
| Frontend plays / downloads the real MP4 | Real player, real poster, real download, 4-check verification panel | ✅ COMPLETE | — |
| Live progress across stages | SSE, weighted progress, real stdout, per-agent grouping | ✅ COMPLETE | — |
| Nothing mocked or simulated | Searched; none found | ✅ COMPLETE | — |
| Every decision explained | `edit_plan.json` + a UI card, with `decided_by` badges | ✅ COMPLETE | — |
| Voice note transcription | faster-whisper `tiny` (a real local ML model), gated on import, with "Insert into script" | ✅ COMPLETE | Four stale docs — incl. a whole README section — still claim it is absent |
| Clip speech understanding | — | ❌ MISSING | Clips are never transcribed |
| Object / LEGO recognition | Only as the vision model's free-text `subject` | ❌ MISSING | No detector, no part database, no brick or set recognition |
| Duplicate detection | Filename-based for iOS pairs; content-based only via vision | 🟡 PARTIAL | No perceptual hashing or embedding similarity |
| Deployment-ready package | Self-contained folder, Docker, Windows and Linux start scripts, clean-room verified | ✅ COMPLETE | — |
| Runs on the author's Windows laptop | Everything except the hardcoded Linux font path | 🟡 PARTIAL | **`FONT` is an absolute Linux path — captions will fail on Windows** |
| **Technical QC agent** (from the project's own `AGENT_SPECS.md`) | — | ❌ MISSING | `README.md:614` states plainly that it is **not implemented** and "remains a plan" |
| **Music & Sound Design agent** (`AGENT_SPECS.md`) | Music exists, but as a threshold table + two synthesis engines — not the specified agent | ❌ MISSING | Same README admission |
| **Content Judgment agent** (`AGENT_SPECS.md`) | — | ❌ MISSING | Not implemented. `AGENT_SPECS` further requires it be validated against **31 hand-scored videos before it scores anything new**, and `README.md:618` states that validation **has not been done** |
| Production security | None | ❌ MISSING | No auth, no rate limiting, no TLS, no CSRF |
| Automated tests | None in the repo | ❌ MISSING | No pytest, no CI |
| Scalability | Single process, in-memory jobs, no concurrency cap | ❌ MISSING | No queue, no workers, no shared state |

---

## 16 · Exact pipeline diagram

```
┌─ USER ────────────────────────────────────────────────────────────────────────┐
│  Operator in a browser at http://127.0.0.1:8000                               │
└──────────────────────────────────┬────────────────────────────────────────────┘
                                   │
┌─ FRONTEND ── frontend/app.js (vanilla ES module, no framework) ───────────────┐
│  STEP 1 Media Bin   drag/drop or browse · reorder · remove · poster previews  │
│  STEP 2 Customize   script · voice note · agent roster · format · motion · music│
│  STEP 3 AI Processing  ring · agent-grouped stages · edit plan · raw log       │
│  STEP 4 Final Video   player · download · verification · spec · cut sheet      │
└──────────────────────────────────┬────────────────────────────────────────────┘
                                   │ XMLHttpRequest (progress-capable)
┌─ UPLOAD ── POST /api/jobs/{id}/clips ── main.py:195 ─────────────────────────┐
│  ensureJob() creates the job lazily on first upload (no phantom drafts)       │
│  stream to disk in 1 MB chunks · enforce AT_MAX_CLIP_MB DURING the write      │
│  save as NN_name.ext  (numeric prefix = the planner's read order)             │
│  ffprobe it · unreadable ⇒ delete + report in `rejected`                      │
└──────────────────────────────────┬────────────────────────────────────────────┘
                                   ▼
┌─ JOB ── jobs.create() ── jobs.py:131 ────────────────────────────────────────┐
│  id = uuid4().hex[:12]                                                        │
│  uploads/<id>/  ·  uploads/<id>/_voice/  ·  outputs/<id>/                     │
│  state → outputs/<id>/job.json  (survives a restart)                          │
│                                                                               │
│  POST /start ⇒ jobs.start() ⇒ threading.Thread(target=_run)  jobs.py:259      │
│  (a thread, not the event loop — a render is minutes of CPU)                  │
└──────────────────────────────────┬────────────────────────────────────────────┘
                                   ▼
╔═ CLIP ANALYSIS ═══════════════════════════════════════════════════════════════╗
║ ① probe    subprocess → build_video.stage_probe()            [REAL]  w  4     ║
║            ffprobe: duration, w/h, rotation, codec, fps, audio, creation_time ║
║            drop iOS IMG_E#### / IMG_#### duplicate pairs                      ║
║            canvas from whichever orientation has more TOTAL DURATION          ║
║                                    ↓ manifest.json                            ║
║ ② measure  subprocess → build_video.stage_measure()          [REAL]  w 20     ║
║            signalstats YDIF @10fps → mean_ydif, max_ydif                      ║
║            freezedetect -38dB/0.4s @6fps → freeze_runs, freeze_share          ║
║            live_share = share of frames where ANYTHING changes  ◄── the fix   ║
║                        that stopped close-ups and talking heads being dropped ║
╚══════════════════════════════════╤════════════════════════════════════════════╝
                                   ▼
╔═ THE ONLY MODEL CALL ═════════════════════════════════════════════════════════╗
║ ③ vision   in-process → vision.analyse()            [AI, or SKIPPED]  w 14    ║
║                                                                               ║
║    AT_VISION_API_KEY unset?  ──YES──►  log why · return None · stage="skipped"║
║              │ NO                                       │                     ║
║              ▼                                          ▼                     ║
║    ffmpeg extracts N real JPEG frames/clip      rule-based path continues     ║
║    base64 → HTTPS POST                          unchanged, and completes      ║
║      anthropic  /v1/messages      (x-api-key, anthropic-version)              ║
║      openai     /v1/chat/completions (Bearer)                                 ║
║    PASS 1 per clip, WITH images  → subject, shot, kind, phase,                ║
║                                    interesting_moments, hook_score,           ║
║                                    caption, confidence, notes                 ║
║    PASS 2 across clips, TEXT ONLY → duplicates, hook, order, story            ║
║    every field clamped to an allow-list / range · names clamped to real clips ║
║                                    ↓ vision.json                              ║
╚══════════════════════════════════╤════════════════════════════════════════════╝
                                   ▼
╔═ DIRECTOR / AGENTS ═══════════════════════════════════════════════════════════╗
║ ④ plan     subprocess → build_video.stage_plan()      [RULE-BASED]  w  3      ║
║            reject only if live_share<0.15 AND mean_ydif<=max(0.05, peak×0.02) ║
║            …but revert ALL rejections if >34% would be dropped                ║
║            budget shared by √motion, clamped 2.5–12.0s                        ║
║            best_window(): slide 0.2s, score = motion×(1−0.8×frozen)           ║
║            order: hook | recorded (creation_time) | uploaded                  ║
║            ◄── vision may inject  --drop CLIP::reason   and  --hook CLIP      ║
║                                    ↓ manifest.segments[]                      ║
║ ⑤ direct   in-process → director.direct()             [RULE-BASED]  w  2      ║
║            transition PER CUT:  orientation ≠ ? → fade                        ║
║                                 else KIND_XFADE[(kindA,kindB)]  ◄── vision    ║
║                                 else next is reveal → fade                    ║
║                                 else PAIR_XFADE[(bandA,bandB)]                ║
║            crossfade length from median motion: 0.25 / 0.4 / 0.6              ║
║            mood from MOOD_BANDS: beat / bright / lofi                         ║
║            REFUSES to decide speed, and records why                           ║
║            captions: your script wins; vision fills blanks at ≥60 confidence  ║
╚══════════════════════════════════╤════════════════════════════════════════════╝
                                   ▼
┌─ EDIT PLAN ── outputs/<id>/edit_plan.json ── director.py:306 ────────────────┐
│  { vision{used,model,provider,clips_seen,duplicates,hook,story},             │
│    median_motion, busy_threshold, segments, transitions[],                   │
│    decisions[ {agent, what, chose, why, decided_by} ] }                      │
│                                                                              │
│  decided_by:  "measurement" → badge `decided`  (a rule fired — NOT AI)       │
│               "you"         → badge `yours`    (your setting won)            │
│               "vision"      → badge `saw it`   (a model looked at frames)    │
│                                                                              │
│  ══ and this is how it reaches the pixels ══                                 │
│  settings["transition_list"] ──► adapter.env_for() ──► AT_XFADES             │
│  settings["transition_duration"] ────────────────────► AT_XFADE_T            │
│  resolution ─────────────────────────────────────────► AT_CANVAS             │
│  quality ────────────────────────────────────────────► AT_CRF                │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
╔═ VIDEO PIPELINE ══════════════════════════════════════════════════════════════╗
║ ⑥ sheets   subprocess → stage_sheets()                      [REAL]  w  7      ║
║            3×2 contact sheet per planned segment → look/segNN.jpg             ║
║                                                                               ║
║ ⑦ cut      subprocess → stage_cut()                         [REAL]  w 31      ║
║            per segment: trim → fps=30 → scale/crop to canvas                  ║
║                         OR blur-pad (gblur σ=25) if orientation differs       ║
║                         → drawtext burns the caption in                       ║
║                         → libx264 -crf $AT_CRF -an → parts/pNN.mp4            ║
║            ⚠ FONT is a hardcoded Linux path — fails on Windows WITH captions  ║
║                                                                               ║
║ ⑧ join     subprocess → stage_join()                        [REAL]  w 15      ║
║            xfade chain; transition i = XFADES[(i−1) % len(XFADES)]            ║
║            ▲ the director's per-cut list lands EXACTLY here                   ║
║            records the real transition offsets → _joined.mp4                  ║
║                                                                               ║
║ ⑨ music    in-process → extras.music_stage()                [REAL]  w  8      ║
║            generate (beatbed.py | bgm_stdlib.py) against the REAL marks       ║
║            or fit an uploaded track · or silence                              ║
║            keep_audio mix/only → source_audio() rebuilds the ORIGINAL audio   ║
║              from the planner's in/out points with acrossfade at the same     ║
║              offsets the video used, so the two stay in step                  ║
║            mix_beds() ducks the score with sidechaincompress                  ║
║                                                                               ║
║ ⑩ master   in-process → extras.master_stage()               [REAL]  w 12      ║
║            optional setpts retime (+ atempo, marks rescaled)                  ║
║            mux · loudnorm I=-15 TP=-1.5 LRA=11 · AAC 192k · +faststart        ║
║            thumbnail from the middle of the strongest segment                 ║
║            VERIFY: count DECODED frames · LUFS from the ebur128 Summary block ║
║                    · blackdetect          → output.json                       ║
╚══════════════════════════════════╤════════════════════════════════════════════╝
                                   ▼
┌─ FFMPEG ─────────────────────────────────────────────────────────────────────┐
│  Every stage above is ffmpeg/ffprobe doing real work. No stage is simulated.  │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
┌─ OUTPUT ── outputs/<job_id>/ ────────────────────────────────────────────────┐
│  ★ <TITLE>_<horizontal|vertical>.mp4      the deliverable                     │
│    thumbnail.jpg · output.json (verified) · edit_plan.json · vision.json      │
│    manifest.json · captions.json · _audio.json                               │
│    parts/ · look/ · measure/ · _joined.mp4 · _posters/    (kept for resume)   │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        │ SSE, throughout: GET /api/jobs/{id}/events            │
        │   snapshot → stage/line/status (0.5 s poll) → final   │
        └──────────────────────────┬───────────────────────────┘
                                   ▼
┌─ FRONTEND ── step 4 ─────────────────────────────────────────────────────────┐
│  <video src=/api/jobs/{id}/file/{name}  poster=…/file/thumbnail.jpg>          │
│  Download → GET /api/jobs/{id}/download                                      │
│  Verification: frame count · black frames · loudness · audio present         │
│    (a build asked to be silent is judged < −50 LUFS, not the normal range)   │
│  Spec table · proportional segment timeline · cut sheet with measured motion  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## QUESTIONS ANOTHER AI SHOULD ANSWER

These are ordered by how much the answer would change the assessment.

1. **The product is called an AI video generation system, but 4 of its 5 "agents" contain no model call and the one that does is disabled by default and appears never to have run against a real endpoint. Is the requirement satisfied by a deterministic system that explains its reasoning, or does it require actual model inference in the decision path?** Everything else follows from this answer.

2. **Is a rule-based director that records *why* it chose each cut more or less valuable than an LLM that chooses well but cannot justify itself?** The codebase bet on inspectability. Was that the right bet for the stated product?

3. **The vision model's `interesting_moments`, `order`, `phase` and `hook_score` are requested, validated, stored and displayed — but no stage consumes them.** `interesting_moments` in particular is the obvious missing link to trim points, which are currently chosen on motion alone. **Is this the single highest-value gap to close, and what would wiring it in actually change about the finished cut?**

4. **`FONT` is a hardcoded absolute Linux path (`/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`) with no fallback, and the project runs on a Windows laptop.** Any build with a non-empty caption should abort the `cut` stage. **Has a captioned build ever succeeded on Windows, and what is the minimal fix — an `AT_FONT` env var, or bundling a font with the app?** (Note: this does *not* explain the one job currently on disk. `outputs/81425b0e8a28/job.json` is `status: "draft"`, `started: null`, `error: null`, all ten stages `pending` — two clips and a voice note were uploaded and **the build was never started at all**. A font failure would have left `status: "error"` with `cut` marked `error`, per `jobs.py:333–337`. Why that build was never started is a separate and open question.)

5. **Clip selection rejects on motion alone.** A sharp, well-composed static shot of a finished model scores lower than a shaky out-of-focus pan. **Is motion the right proxy for "worth keeping" in build footage, and what would a better selection signal be that does not require a model?**

6. **There is no authentication on any of 30 routes, including an unauthenticated irreversible `DELETE` that `rmtree`s a job's uploads and outputs.** The README documents `AT_HOST=0.0.0.0` as the way to demo from another machine. **Is localhost-only an adequate security posture for the intended use, or does the demo path make auth mandatory?**

7. **Does "create the best story/sequence" require narrative understanding the system does not have?** Today the order comes from recording timestamps, motion ranking, or the operator. The vision model *proposes* an order and is ignored. **Would honouring `cross.order` produce better videos, or is chronological build order genuinely the right answer for this domain?**

8. **The LEGO end-to-end tests used synthetic clips with LEGO-like motion profiles, not real LEGO footage, and no real vision model was ever called.** **What is the minimum test that would actually prove the product works — and can it be done without spending on API calls?**

9. **The system deliberately refuses two decisions — captions without a script, and speed — on the grounds that a wrong answer is worse than no answer.** **Is principled refusal the right behaviour for a product whose selling point is automation, or should it guess and let the operator correct it?**

10. **`build_video.py` contains ~150 lines of dead code (`stage_music`, `stage_mux`, `write_notes`) superseded by `extras.py`, and `extras.py` plus `app.js` carry comments asserting that transcription does not exist when it now does.** **How much does this kind of drift matter for a project that will be handed to someone else — and what is the right cleanup boundary given "do not rewrite working components"?**

---

*Compiled by reading every file in `C:\Users\Akansha\Downloads\alpha_titans` on 2026-09-22. Route list, stage list, agent registry and environment variables were extracted programmatically from source. No project file was modified.*
