# Alpha Titans - multi-agent architecture (proposal)

23 Oct 2026. Based on the Task Descriptions PDF (62 tasks) and the code on
branch `dev-2`. Nothing here is built yet; this is the plan.

## 1. The shape

One **Chief** agent on top, **nine teams** under it. Each team has one **lead**
and **three or four workers**. The 18 agents we already have do not change:
they become workers. New agents fill the gaps the task list asks for.

```
                              CHIEF (Producer)
        plans the run, runs the teams in order, stops for human approval
                                   |
  INSIGHT -> CREATIVE -> FOOTAGE -> EDIT -> SOUND -> RENDER -> QUALITY -> PUBLISH -> LEARNING
              ^ stop 1                                ^ stop 2     ^ stop 3    ^ manager approves
                                                                                 publication (separate)
     QUALITY can send the work back to EDIT (the existing bounded revise loop)
```

Today the graph is flat: 18 agents in one chain. The manager wants a
hierarchy, so the teams below wrap the same agents.

## 2. The teams

"existing" = already in `server/agents/`. "NEW" = to be built.

| # | Team (lead) | Workers | Tasks |
|---|---|---|---|
| 1 | **Insight** - what to make | Collector (NEW), Classifier (NEW), Opportunity (NEW: demand x gap x supply), Patterns (NEW: teardowns) | C1-C11, D1-D6 |
| 2 | **Creative** - what to say | Research (existing), ContentStrategy (existing), Story (existing), Hook (NEW) | E1, E4 |
| 3 | **Footage** - what we have | ClipAnalysis (existing), Vision (existing), ShotSelector (NEW) | E3 |
| 4 | **Edit** - how it is cut | EditingDirector (existing), Animation (existing), Transition (existing), Caption (existing) | E5 |
| 5 | **Sound** - what it sounds like | Music (existing), Voiceover (existing, optional: the channel is faceless), SoundDesign (NEW) | E9 |
| 6 | **Render** - make the files | Rendering (existing), Look&Speed (NEW: colour, speed ramps), Formats&EndCard (NEW: three outputs, end card, loop seam) | E6-E8, E10 |
| 7 | **Quality** - is it good | Checker (existing Critic), Features (NEW), Scorer (NEW), Calibrator (NEW) | E13, F1-F6 |
| 8 | **Publish** - ship it safely | Title (existing), Thumbnail (existing), Publisher (existing, extended from plan to private draft), DriftWatch (NEW) | E11, E12, G1-G3 |
| 9 | **Learning** - what happened | Metrics (NEW: YouTube Analytics), Performance (existing), Learning (existing), KnowledgeBase (NEW) | B2-B3, G4, G7-G9 |

Count: 18 existing + 15 new workers, 9 leads, 1 chief. Every one of the 18
existing agents has a place.

## 3. Where the project stands against the 62 tasks

| Stage | Tasks | In the code today | Left |
|---|---|---|---|
| A | A2-A8 (7) | Manual work. 6 approved, A2 has changes requested | Nothing to code. Fix A2's two missing columns |
| B | B2-B8 (7) | `analytics_store.py` reads exported rows only | B2/B3 live YouTube Analytics pull and baseline table. B4-B8 are documents |
| C | C1-C11 (10) | No collectors. `search_tools.py` does YouTube/Tavily/Brave search for research only | All of it: collectors (YouTube, Reddit, X, Instagram), classifier, opportunity map, supply count. C1, C9, C10 reuse MySamriddhi code |
| D | D1-D6 (6) | Nothing; this is analysis work | Human analysis; tooling optional |
| E | E1-E13 (13) | Most of the video pipeline: clip analysis, vision, strategy, story, editing director, animation, transition, caption, music, render, critic, title, thumbnail | Speed ramps and timelapse (E5), colour (E6), three formats from one timeline (E7), end card and loop seam (E8), sound design (E9), recipe config (E2), hook alternates (E4), timing report (E10) |
| F | F1-F6 (6) | The Critic checks the rendered file. It is not a 0-10 score | Features, the scoring gate, threshold calibration, parallel run log, accuracy tracking |
| G | G1-G9 (9) | Publishing writes a plan only (no upload, by design). Learning exists but has no real outcomes. Artifact store exists | Private-draft upload (G1), drift check (G2), other platforms (G3), cross-platform metrics (G4), review screen (G5), stop-and-ask checkpoints (G6), retraining with holdout (G7), knowledge base (G8), growth dashboard (G9) |
| H | H1-H5 (5) | Queue and VM start/stop built for the cloud platform | Per-build-day run, weekly review loop, one-change test, quarterly refresh |

## 4. How to build it (LangGraph)

- Each team is a LangGraph **subgraph**: a lead node plus the worker nodes.
  The lead routes between workers with conditional edges. The subgraph is
  added to the Chief's graph as one node.
- Kept as they are: the SQLite checkpoints, the typed artifact store, the
  traces, the `base.Agent` contract, the parallel Vision / Research branch,
  and the bounded critic-to-editor revise loop (now Quality to Edit).
- **Human checkpoints (G6):** the Chief pauses with LangGraph `interrupt()`
  after Creative, after Render and after Quality, and records who approved
  what. The manager's publication approval is separate. The pipeline never
  holds publishing credentials.
- **Rule kept from the current design:** models decide what, code decides
  where. Leads are mostly deterministic routers; only workers reason.
- **Where it runs:** the existing cloud worker wraps `run_job` unchanged, so
  the whole graph can run on the owner's VM.

## 5. Suggested build order

1. **Regroup the 18 agents into the nine teams.** Wiring only. It shows the
   manager the hierarchy and every existing test should keep passing.
2. **Quality team, checkpoints and review screen** (F1-F3, G5, G6): the gate.
3. **Learning and metrics** (B2, B3, G4, G7, G8): real numbers instead of
   "insufficient data".
4. **Publish** (G1, G2): needs the manager's YouTube credentials.
5. **Insight team** (C): reuse the MySamriddhi collector.
6. **Production polish** (E5-E8): speed, colour, formats, end card.

## 6. Open points

- The Publisher refuses to upload today, and a test guards that. G1 changes it
  on purpose, to private drafts only.
- Insight needs outside approvals: Reddit credentials, the X access tier,
  Meta's app review for Instagram.
- Hosted AI (Groq, Gemini) or open models on the L4 is still undecided.
- Voiceover stays optional: the channel is faceless.
