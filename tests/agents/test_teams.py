"""The agent hierarchy must describe the system that actually exists.

`agents/teams.py` says which agent belongs to which team. It is a claim about
the graph and the registry, so these tests compare it to both. The moment it
names a node the graph does not have, an agent the registry does not have, or
leaves one of the real agents unplaced, one of these fails.

A worker with no node is PLANNED. It may be listed so the shape is visible,
but nothing may present it as working: `test_mission.py` covers that.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents import teams as T                          # noqa: E402
from agents.base import agent_registry                 # noqa: E402

# The order the graph actually runs the agents in (vision and research run in
# parallel after clip_analysis; research is listed where it first starts).
GRAPH_ORDER = ["clip_analysis", "research", "vision", "content_strategy",
               "story", "voiceover", "music", "animation", "transition",
               "caption", "editing_director", "render", "visual_review",
               "critic",
               "thumbnail", "title", "publishing", "performance_analysis",
               "learning"]


def _graph_nodes() -> set[str]:
    from langgraph.graph import StateGraph
    from agents.graph import GraphState, _wire
    return set(_wire(StateGraph(GraphState)).nodes)


def _workers():
    return [w for t in T.TEAMS for w in t.workers]


# ------------------------------------------------------------ the shape

def test_there_are_nine_teams_with_three_or_four_workers_each():
    assert len(T.TEAMS) == 9
    for t in T.TEAMS:
        assert 3 <= len(t.workers) <= 4, (
            f"{t.id} has {len(t.workers)} workers; a team is a lead with "
            f"three or four agents under it")


def test_ids_keys_and_nodes_are_unique():
    assert len({t.id for t in T.TEAMS}) == len(T.TEAMS)
    keys = [w.key for w in _workers()]
    assert len(set(keys)) == len(keys), "two workers share a key"
    nodes = [w.node for w in _workers() if w.node]
    assert len(set(nodes)) == len(nodes), "two workers claim one node"


def test_a_planned_worker_has_no_node_and_no_agent():
    for w in _workers():
        if not w.built:
            assert w.node is None and w.agent_id is None, (
                f"{w.key} is planned but names a node or agent")


def test_a_built_worker_names_both_a_node_and_an_agent():
    for w in _workers():
        if w.built:
            assert w.agent_id, f"{w.key} is built but has no agent id"


# --------------------------------------------- claims match the system

def test_every_built_worker_is_a_real_graph_node():
    nodes = _graph_nodes()
    missing = {w.node for w in _workers() if w.node} - nodes
    assert not missing, f"teams.py names nodes the graph lacks: {missing}"


def test_every_traced_graph_node_belongs_to_a_team():
    """The nodes that run an agent are exactly the ones with a policy."""
    from agents.graph import POLICIES
    assert set(POLICIES) == T.built_nodes(), (
        f"graph nodes with a policy but no team: "
        f"{set(POLICIES) - T.built_nodes()}; teams naming a node with no "
        f"policy: {T.built_nodes() - set(POLICIES)}")


def test_every_registered_agent_is_placed_exactly_once():
    reg = set(agent_registry())
    placed = [w.agent_id for w in _workers() if w.agent_id]
    assert sorted(placed) == sorted(reg), (
        f"unplaced: {reg - set(placed)}; unknown: {set(placed) - reg}; "
        f"duplicated: {sorted({a for a in placed if placed.count(a) > 1})}")


def test_the_essential_workers_are_the_four_the_run_cannot_lose():
    """Every other node has a `*_failed` key in the graph state and degrades
    instead of stopping the run; these four do not."""
    ess = {w.node for w in _workers() if w.essential}
    assert ess == {"clip_analysis", "editing_director", "render", "critic"}


def test_teams_are_ordered_by_when_they_first_start_in_the_graph():
    first = []
    for t in T.TEAMS:
        built = [GRAPH_ORDER.index(w.node) for w in t.built_workers]
        if built:
            first.append(min(built))
    assert first == sorted(first), (
        "the screen reads left to right in the order things happen, so a "
        "team that starts later cannot sit before one that starts earlier")


# ------------------------------------------------------------ the lookups

def test_lookups_round_trip():
    for w in _workers():
        if w.node:
            assert T.worker_for_node(w.node) is w
            assert T.team_of_node(w.node).workers.count(w) == 1
    assert T.worker_for_node("finish") is None
    assert T.team_of_node("no_such_node") is None


def test_hierarchy_counts_match_the_workers():
    h = T.hierarchy()
    built = sum(1 for w in _workers() if w.built)
    assert h["counts"]["teams"] == 9
    assert h["counts"]["built_workers"] == built == len(agent_registry())
    assert h["counts"]["planned_workers"] == len(_workers()) - built
    for t in h["teams"]:
        assert t["lead"]["label"].endswith("Lead")
        assert t["checker"]["label"].endswith("Checker")


def test_the_chief_does_not_claim_the_approval_stops_exist():
    """The three human stops are planned. The chief's description must say
    so, or the screen would imply a safeguard that is not there."""
    assert "planned" in T.CHIEF["does"].lower()
