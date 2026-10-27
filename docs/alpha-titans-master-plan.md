# Alpha Titans - master plan

Architecture, build decisions, what runs where, the pipeline, and what we
need from the owner of the VM. 23 Oct 2026. Based on the Task Descriptions PDF
(62 tasks), the Getting Started guide for the VM, and the code on `dev-2`.
Nothing here is built yet. Where something is unverified, it says so.

## 0. The verdict in six lines

1. A hierarchy: one **Chief** agent, **nine teams**, each with a lead and 3-4
   workers. The 18 agents we have become workers; 15 new ones fill the gaps.
2. **Everything that touches footage, credentials or outputs runs on the VM.**
   The laptop is for building and testing only.
3. **No cloud control plane for this deployment.** The VM is woken by the
   guide's start command (or an owner-side schedule). A **Conductor** service
   on the VM owns a local queue and runs the Chief.
4. **One model gateway.** Agents that see footage use a model on the L4 by
   default. Hosted AI is used only where the owner approves it.
5. **Scores come from code, not from a language model.** Hard rule: breaking
   the honesty constraints scores zero.
6. **Human stops are structural.** The pipeline pauses at three points and
   cannot skip them. Publishing is always a private draft; the manager
   approves it himself.

## 1. What the two PDFs tell us

From the **VM guide**:
- `myyukti`, project `samriddhi-production`, `asia-south1-c`: 8 vCPU, 32 GB,
  NVIDIA L4, 271 GB free, Ubuntu desktop, Kdenlive, mpv, VLC, ffmpeg, python3,
  gcloud.
- Off when idle. We start it with `gcloud compute instances start`, connect
  through an IAP tunnel and remote desktop. **It stops itself after 30 minutes
  of no activity.** Files on the disk survive.
- Footage is in `gs://mysamriddhi-projects/P63_Alpha_Titans/clean/` (read with
  `gcloud storage`). Work goes under `/opt/alpha_titans/`.
- **The material is confidential and stays on the machine.** Exports go
  through Sachi.
- The guide says the VM has no internet. We were told it does. **Verify.**

From the **task list**:
- The VM is "the secure machine where pipeline work happens". It publishes to
  a private draft on the channel, pulls YouTube Analytics onto itself, and the
  manager reads numbers there, never in YouTube Studio.
- The manager **holds the credentials and approves publication**. We approve
  the machine's work at each stage. The two approvals must never blur (G5).
- The pipeline **must not run unattended start to finish** (G6).
- Recurring work: per build day (H1), a daily drift check (G2), weekly review
  (H2), quarterly research refresh (H5).
- The honesty rule: no title or thumbnail may promise motion the footage does
  not show. Breaking the constraints spec scores zero (F2).
- Faceless channel, footage arrives silent: every audio decision is ours.

What follows from that:
- A pipeline that waits for humans must **keep its state on disk**, because
  the VM may be off while it waits.
- A daily job on a VM that sleeps needs a **daily wake-up**.
- Confidential footage plus hosted AI is a **policy question**, so the design
  makes the model choice a setting, not a rewrite.

## 2. Architecture

```
 LAPTOP (build and test only)             COMPANY BUCKET                  YOUTUBE
  code, tests, fakes, prompts              incoming/<day>/  outputs/       private drafts,
          | git                                  |                         analytics
          v                                      v                            ^   |
 =========================== VM myyukti (L4; off when idle) ==================|===v======
  Conductor (systemd) -> local queue -> CHIEF graph -> nine teams
  Knowledge base (SQLite) . artifact store . traces . review screen (localhost:8000)
  Model gateway:  local model on the L4   |   hosted APIs (approved, non-footage agents)
 ======================================================================================
```

The Chief and its teams:

```
                                  CHIEF (Producer)
       plans the run . runs teams in order . owns the three human stops . retries
   +------+--------+--------+------+-------+--------+---------+---------+---------+
 INSIGHT FOOTAGE CREATIVE  SOUND   EDIT  RENDER  QUALITY  PUBLISH  LEARNING
   4       3        4       3      4      3       4        4        4      workers each
```

Built on branch `updated-at` (23 Oct 2026): the team structure
(`server/agents/teams.py`) and Mission Control (`/mission`). The teams are
ordered by when each starts in the graph, so Footage comes before Creative.
The checkers are derived at read time and are not yet gates; the three human
stops are planned.

| Team | Lead decides | Workers (existing / NEW) |
|---|---|---|
| Insight | what is worth making | Collector (NEW), Classifier (NEW), Opportunity (NEW), Patterns (NEW) |
| Creative | what the video says | Research, ContentStrategy, Story, Hook (NEW) |
| Footage | what usable moments exist | ClipAnalysis, Vision, ShotSelector (NEW) |
| Edit | how it is cut | EditingDirector, Animation, Transition, Caption |
| Sound | what it sounds like | Music, Voiceover (optional), SoundDesign (NEW) |
| Render | how files are made | Rendering, Look&Speed (NEW), Formats&EndCard (NEW) |
| Quality | whether it is good and allowed | Checker (the existing Critic), Features (NEW), Scorer (NEW), Calibrator (NEW) |
| Publish | how it ships safely | Title, Thumbnail, Publisher (extended), DriftWatch (NEW) |
| Learning | what happened and what changes | Metrics (NEW), Performance, Learning, KnowledgeBase (NEW) |

How it is built (LangGraph, already a dependency):
- Each team is a **subgraph**; the lead routes between workers; the subgraph
  is one node of the Chief's graph. The existing SQLite checkpointer, typed
  artifact store, traces and `base.Agent` contract carry over unchanged.
- The Vision-and-Research parallel branch and the bounded revise loop
  (Quality sends work back to Edit) stay.
- The three human stops use LangGraph `interrupt()`: after Creative (the
  brief), after Render (the cuts), after Quality (the score). Each approval is
  recorded with who and when.
- Rule kept from today's design: models decide what, code decides where.
  Leads are mostly deterministic routers.

Shared guardrails (a constraints spec every team reads):
- The honesty rule, enforced by the Checker as a hard failure.
- Publisher can only create **private** drafts. The code has no path that sets
  anything public. DriftWatch checks the channel daily and shouts on any
  public video nobody approved.

## 3. Build decisions and why

| # | Decision | Reason |
|---|---|---|
| 1 | Chief + nine teams | The manager asked for a hierarchy; every existing agent keeps a place |
| 2 | Footage, credentials and outputs stay on the VM | The guide says the material is confidential |
| 3 | Drop the separate control plane and worker VMs for this deployment. Reuse the queue and run-one code, run on the VM | Cross-project starting is blocked by org rules and the data would leave the project |
| 4 | Model gateway with a provider per agent | The data-approval question is open; this makes either answer a config change |
| 5 | Score = features + a small calibrated model, reasons from feature contributions | An LLM-invented number cannot be argued with or calibrated |
| 6 | Human stops via `interrupt()`, state on disk | Skipping must be impossible; the VM may be off while waiting |
| 7 | Private drafts only, plus a daily DriftWatch | The upload permission cannot prevent a public upload, so code enforces it and a check verifies it |
| 8 | Use the GPU for decode and encode (NVDEC/NVENC) | The pipeline uses it for nothing today; E10 asks how long a build day renders. Needs approval to change the protected engine |
| 9 | Every step resumable and idempotent | The 30-minute auto-off can cut a run |
| 10 | Daily wake-up, then drain and idle off | Metrics, drift check and intake need the VM on briefly each day |
| 11 | Knowledge base keeps features, scores, outcomes and every scoring version | G7 and G8 need history and a held-out set |
| 12 | A golden set in the tests (known videos, known scores) | Regression guard for the whole agent system |

## 4. What runs where

| Laptop | VM (myyukti) |
|---|---|
| Writing code and prompts | Everything that reads footage |
| The test suite, with fake models and fake APIs | All agents in production runs |
| Docs and recipes | Local model server and the GPU work |
| Developing the Insight collectors against public data | Credentials (YouTube token, API keys) in files only the service user can read |
| Nothing confidential, no credentials, no outputs | Outputs, knowledge base, review screen, Conductor, scheduled jobs |

Code reaches the VM with `git` from our repository (or the bucket if GitHub is
blocked).

## 5. How the pipeline runs, end to end

**Each morning (and on demand)**
1. The VM is started: by the guide's command, or by an owner-side schedule.
2. The Conductor boots with it and checks its queue.
3. **Intake:** new footage under `incoming/<build-day>/` becomes a job.
   **Measure:** pull yesterday's analytics (Metrics), run DriftWatch.

**For each build day**
4. **Footage:** probe and measure with GPU decode; vision descriptors; the
   ShotSelector keeps the moments that survive.
5. **Creative:** read the latest opportunity map and pattern sheet from the
   knowledge base; write the brief with a hook and beats. **STOP 1: approve the
   brief.**
6. **Edit, Sound, Render:** cut with speed ramps, colour and sound design;
   render three formats from one timeline: long landscape, vertical highlights,
   vertical short. **STOP 2: approve the cuts.**
7. **Quality:** mechanical checks, features, score 0-10 with reasons.
   Constraint breach = 0. Below the bar goes to the review queue, never to
   upload. **STOP 3: approve the score.**
8. **Publish:** title options, thumbnail candidates, **private draft** on the
   channel. The manager approves publication himself.
9. **Learning:** after the video has run, Metrics pulls the numbers; the
   Calibrator compares your sealed score, the machine's score and the real
   result; the scoring is retrained against a held-out set and versioned.

If the VM stops while a stop is pending, nothing is lost: the state is on disk
and the run resumes when the VM is next started and the approval is given.

**Weekly and quarterly:** the Insight team refreshes collection and the
opportunity map (H5); the weekly review proposes one recipe change (H2).

## 6. Which model does what

| Work | Model | Why |
|---|---|---|
| Vision on footage frames | Open vision-language model on the L4 | Frames are confidential |
| Creative, Edit, Sound planning (text from descriptors) | Open text model on the L4, or hosted if the owner approves | Text is derived from the footage |
| Insight (public comments and channels) | Hosted or local | Public data; no confidentiality issue |
| Scoring | No LLM for the number. Features + a calibrated model on CPU. An LLM may phrase the explanation | Auditable and calibratable |
| Encoding | ffmpeg with NVDEC/NVENC | Uses the L4 for real work |
| Speech to text | Not needed | The footage is silent |

One 7-14B open multimodal model fits in the L4's 24 GB. The existing code
already talks to an OpenAI-compatible endpoint, so pointing it at a local
server is a settings change. Risk: smaller models plan less well, and the
JSON-repair code was tuned on Groq's models. Test before trusting it.

## 7. Build order

0. **Regroup the 18 agents into nine teams.** Wiring only; all tests pass.
1. **VM bring-up, in parallel with 0:** the checks below, install, run the
   existing pipeline on one build day.
2. **Conductor and local queue; Quality team; checkpoints; review screen**
   (F1-F3, G5, G6).
3. **Metrics, Learning, KnowledgeBase** (B2, B3, G4, G7, G8).
4. **Publish** private drafts and DriftWatch (G1, G2). Needs the manager's
   credentials.
5. **Insight team** (C, D). Reuse the MySamriddhi collector.
6. **Production polish** (E5-E8): speed, colour, formats, end card.

Each phase ends with a test that can fail: for example, "one build day goes
from `incoming/` to a scored draft with no one touching it except the three
stops".

**VM bring-up checks** (run on the VM):
- `nvidia-smi` (the driver works)
- `ffmpeg -hide_banner -encoders | grep nvenc` (GPU encode is available)
- `python3 --version` (3.10 or newer for LangGraph)
- reach `api.groq.com` and `generativelanguage.googleapis.com` with `curl`
- `pip install` works, or the packages are installed for us

## 8. What we need from the owner

**Already in the guide (confirm they are active):** start/stop `myyukti`, the
IAP tunnel, read the footage bucket, write under `/opt/alpha_titans/`.

**Network (HTTPS, outbound from the VM)**
- `pypi.org`, `files.pythonhosted.org` (packages)
- `github.com` (our code)
- `huggingface.co` (one-time download of open model weights)
- `api.groq.com`, `generativelanguage.googleapis.com` (only if hosted AI is approved)
- `youtube.googleapis.com`, `youtubeanalytics.googleapis.com`,
  `oauth2.googleapis.com` (draft upload and analytics)
- `oauth.reddit.com`, `www.reddit.com`; `api.x.com`; `graph.facebook.com`
  (as each collector is built)

**On the VM**
- `sudo`, to install packages and services and to shut the VM down
- NVIDIA driver working, and ffmpeg with NVENC/NVDEC (or permission to install it)
- A web browser
- A background service that survives disconnects
- A way to keep the VM on while a job runs: a running job counts as activity,
  or an exemption

**Storage**
- Read the `incoming/` folder; write an `outputs/` folder in the same bucket
  (or Sachi collects, as the guide says)

**Scheduling**
- A daily (and per-build-day) VM start, created by the owner, or a start-only
  service account

**Publishing and analytics (manager)**
- Authorise the channel account for the pipeline: upload as private draft and
  read analytics; store the token on the VM
- Turn on the YouTube Data and Analytics APIs in a Google Cloud project
- Keep publication approval with the manager
- An alert route for the drift alarm (email or chat webhook), or show it on
  the review screen

**Access to existing work**
- Read the MySamriddhi Drive folder `3_Branding and Marketing/Autonomous
  Content Engine/1A_Content_Suggestions_Engine` (the collector to reuse)

**Review screen:** how the manager opens it: through the VM desktop, or an
IAP tunnel to the screen's port.

**Decisions only the owner can make**
- Which AI services may receive text or frames derived from footage; if none,
  we use local models only
- Whether finished videos may be viewed outside the VM
- The music source and licence (E9)
- The X access tier, Meta's Instagram app review, Reddit API credentials

## 9. Risks

- Hosted AI on confidential footage: unresolved until the owner answers.
- Local model quality: unproven on this pipeline. Mitigation: the golden set.
- YouTube upload quota: the default quota is limited and an upload costs
  many units. Check the current numbers before planning more than a few
  uploads a day.
- External approvals (Meta, X tier) take days to weeks. Start them early.
- The 30-minute idle rule vs long jobs: confirm how "activity" is measured.
- GPU encode needs a change to the protected engine: needs approval.
