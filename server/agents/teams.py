"""
The agent hierarchy: one Chief, nine teams, and what sits under each team.

WHY THIS FILE EXISTS
--------------------
The orchestration graph (`graph.py`) is flat: eighteen agents in one chain. The
brief for the system is a HIERARCHY - every step is a team, and a team is
several agents working together, with something that checks their result
before it is handed on. This module is the single place that says which agent
belongs to which team.

It describes the structure. It does not run anything, and it does not change
how the graph executes: the proven graph, its checkpoints, retries and revise
loop are untouched. Everything the Mission Control screen shows is READ from
the trace the graph already writes, grouped by the teams declared here.

THE THREE ROLES IN A TEAM
-------------------------
  worker    does the work. Nineteen of them already exist (one per agent) and
            are listed here with the graph node that runs them. Others are
            PLANNED: named so the shape is visible, but with no node and no
            code. They are shown as "planned", never as working.
  lead      the team's coordinator. Deterministic: it reports the team's state
            and what is left. It makes no model call and is declared as such.
  checker   verifies the team's result before the next team relies on it.
            Deterministic and read-only: see `mission.py`. In this version its
            verdict is DERIVED when the screen is read, not a gate that stops
            the run. Making it a gate is a later step, and is said so wherever
            it is shown.

HONESTY RULE
------------
A worker is `built` only if a graph node runs it. `test_teams.py` fails the
moment this file claims a node the graph does not have, an agent the registry
does not have, or leaves one of the eighteen agents unplaced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    key: str                       # stable id, e.g. "clip_analysis"
    label: str                     # what the screen calls it
    does: str                      # one plain line
    node: str | None = None        # graph/trace node when BUILT, else None
    agent_id: str | None = None    # id in agent_registry() when BUILT
    # The run cannot continue without these four: every other node has a
    # `*_failed` key in the graph state and degrades instead of stopping it.
    essential: bool = False

    @property
    def built(self) -> bool:
        return self.node is not None


@dataclass(frozen=True)
class Team:
    id: str
    label: str
    phase: str
    purpose: str
    workers: tuple[Worker, ...]

    @property
    def lead_label(self) -> str:
        return f"{self.label} Lead"

    @property
    def checker_label(self) -> str:
        return f"{self.label} Checker"

    @property
    def built_workers(self) -> tuple[Worker, ...]:
        return tuple(w for w in self.workers if w.built)


def _w(key, label, does, *, node=None, agent=None, essential=False) -> Worker:
    return Worker(key=key, label=label, does=does, node=node,
                  agent_id=agent, essential=essential)


def _built(key, label, does, *, agent=None, essential=False) -> Worker:
    """A worker the graph runs: node name == key, agent id defaults to it."""
    return _w(key, label, does, node=key, agent=agent or key,
              essential=essential)


def _planned(key, label, does) -> Worker:
    return _w(key, label, does)


# Ordered by when each team's first built worker starts in the graph, so the
# screen reads left to right in the order things happen.
TEAMS: tuple[Team, ...] = (
    Team("insight", "Insight", "Understand",
         "What is worth making: demand, unmet need and how crowded it is.",
         (
             _planned("collector", "Collector",
                      "Pulls comments and channel data from allowed sources "
                      "into the source layer."),
             _planned("classifier", "Classifier",
                      "Tags every comment with a topic and what it signals: "
                      "confusion, a request, a complaint."),
             _planned("opportunity", "Opportunity",
                      "Ranks topics by demand, unmet need and how many "
                      "channels already serve them."),
             _planned("patterns", "Patterns",
                      "Turns teardowns of the best videos into a pattern "
                      "sheet."),
         )),
    Team("footage", "Footage", "Prepare",
         "What usable footage exists, and which moments are worth keeping.",
         (
             _built("clip_analysis", "ClipAnalysis",
                    "Measures every clip: length, motion, orientation, "
                    "duplicates.", essential=True),
             _built("vision", "Vision",
                    "Looks at frames and says what is happening in each "
                    "moment."),
             _planned("shot_selector", "ShotSelector",
                      "Decides which moments of a whole build day survive."),
         )),
    Team("creative", "Creative", "Plan",
         "What the video says: the idea, the story and the opening.",
         (
             _built("research", "Research",
                    "Finds what the world says about the subject, from real "
                    "sources only."),
             _built("content_strategy", "ContentStrategy",
                    "Decides what the video should communicate."),
             _built("story", "Story",
                    "Shapes the narrative and its beats."),
             _planned("hook", "Hook",
                      "Writes and tests the first three seconds, with "
                      "alternates."),
         )),
    Team("sound", "Sound", "Make",
         "What the video sounds like.",
         (
             _built("voiceover", "Voiceover",
                    "Writes what is said over the story (optional for a "
                    "faceless channel)."),
             _built("music", "Music",
                    "Decides what the score does, beat by beat."),
             _planned("sound_design", "SoundDesign",
                      "Adds the click-and-snap sounds and the payoff cues."),
         )),
    Team("edit", "Edit", "Make",
         "How the video is cut and dressed.",
         (
             _built("animation", "Animation",
                    "Decides how the cut looks on the canvas."),
             _built("transition", "Transition",
                    "Chooses how one beat gives way to the next."),
             _built("caption", "Caption",
                    "Chooses which words appear on screen."),
             _built("editing_director", "EditingDirector",
                    "Chooses the clips, their order and their length.",
                    essential=True),
         )),
    Team("render", "Render", "Make",
         "Turning the plan into real video files.",
         (
             _built("render", "Rendering",
                    "Cuts, joins and masters the video with ffmpeg.",
                    agent="rendering", essential=True),
             _planned("look_speed", "Look&Speed",
                      "Colour treatment, speed ramps and timelapse."),
             _planned("formats", "Formats&EndCard",
                      "Three formats from one timeline, an end card and a "
                      "loop seam."),
         )),
    Team("quality", "Quality", "Judge",
         "Whether the video is good and allowed to ship.",
         (
             _built("critic", "Critic",
                    "Inspects the rendered file and decides pass, revise or "
                    "escalate.", essential=True),
             _built("visual_review", "VisualReviewer",
                    "Watches the finished video shot by shot and scores "
                    "0-10 whether each shot shows the subject; weak shots "
                    "go back to be re-cut."),
             _planned("features", "Features",
                      "Turns the video into measurable features."),
             _planned("calibrator", "Calibrator",
                      "Compares your score, the machine's and the real "
                      "result."),
         )),
    Team("publish", "Publish", "Ship and learn",
         "Getting the video ready to ship, safely.",
         (
             _built("thumbnail", "Thumbnail",
                    "Picks the frame that represents the video."),
             _built("title", "Title",
                    "Writes honest title options."),
             _built("publishing", "Publishing",
                    "Prepares the publishing plan. Never publishes."),
             _planned("drift_watch", "DriftWatch",
                      "Daily check for public videos nobody approved."),
         )),
    Team("learning", "Learning", "Ship and learn",
         "What happened, and what should change.",
         (
             _built("performance_analysis", "Performance",
                    "Reports how the video performed, or says it cannot "
                    "know."),
             _built("learning", "Learning",
                    "Learns from real outcomes and proposes changes for "
                    "approval."),
             _planned("metrics", "Metrics",
                      "Pulls analytics from the warehouse."),
             _planned("knowledge_base", "KnowledgeBase",
                      "Keeps every video's features, score and outcome."),
         )),
)

CHIEF = {
    "label": "Chief",
    "does": ("Runs the teams in order, the revise loop and the retries. The "
             "three human approval stops are planned, not built."),
}

_BY_ID = {t.id: t for t in TEAMS}
_BY_NODE = {w.node: (t, w) for t in TEAMS for w in t.workers if w.node}


def team(team_id: str) -> Team | None:
    return _BY_ID.get(team_id)


def team_of_node(node: str) -> Team | None:
    hit = _BY_NODE.get(node)
    return hit[0] if hit else None


def worker_for_node(node: str) -> Worker | None:
    hit = _BY_NODE.get(node)
    return hit[1] if hit else None


def built_nodes() -> set[str]:
    return set(_BY_NODE)


def hierarchy() -> dict:
    """The static structure, JSON-ready. No job, no state."""
    return {
        "chief": dict(CHIEF),
        "teams": [{
            "id": t.id, "label": t.label, "phase": t.phase,
            "purpose": t.purpose,
            "lead": {"label": t.lead_label,
                     "does": "Reports the team's state and what is left. "
                             "Deterministic: it makes no model call."},
            "checker": {"label": t.checker_label,
                        "does": "Verifies the team's result before the next "
                                "team relies on it. Read-only."},
            "workers": [{"key": w.key, "label": w.label, "does": w.does,
                         "node": w.node, "built": w.built,
                         "essential": w.essential} for w in t.workers],
        } for t in TEAMS],
        "counts": {
            "teams": len(TEAMS),
            "built_workers": len(_BY_NODE),
            "planned_workers": sum(1 for t in TEAMS for w in t.workers
                                   if not w.built),
        },
    }


__all__ = ["CHIEF", "TEAMS", "Team", "Worker", "built_nodes", "hierarchy",
           "team", "team_of_node", "worker_for_node"]
