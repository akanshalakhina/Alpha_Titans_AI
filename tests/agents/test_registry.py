"""The agent registry, and the health endpoint that answers from it.

WHY THIS FILE EXISTS
--------------------
`/api/v2/health` used to answer from a literal list:

    "agents": ["clip_analysis", "editing_director", "rendering", "critic"],
    "phase": "2.4",

Fourteen agents later it was still saying four. Nothing failed, because
nothing compared that list to reality. These tests are that comparison: the
registry is derived by walking the package, and the endpoint must report
exactly what the registry found - so the only way to change the answer is to
add or remove an agent.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.artifacts import ArtifactKind          # noqa: E402
from agents.base import agent_registry             # noqa: E402

try:
    from typing import get_args
except ImportError:                                 # pragma: no cover
    from typing_extensions import get_args          # type: ignore

AGENT_DIR = ROOT / "server" / "agents"


# ------------------------------------------------------- the registry

def test_registry_finds_every_agent_in_the_package():
    reg = agent_registry()
    on_disk = set()
    for f in AGENT_DIR.glob("*.py"):
        src = f.read_text(encoding="utf-8", errors="replace")
        # a concrete agent declares its own id
        for m in re.finditer(r'\n\s{4}id\s*=\s*"([^"]+)"', src):
            on_disk.add(m.group(1))
    missing = on_disk - set(reg)
    assert not missing, f"agents on disk the registry did not find: {missing}"


def test_every_agent_produces_a_distinct_artifact_kind():
    reg = agent_registry()
    kinds = [a["produces"] for a in reg.values()]
    dupes = {k for k in kinds if kinds.count(k) > 1}
    assert not dupes, f"two agents claim the same artifact kind: {dupes}"


def test_every_produced_kind_is_a_declared_artifact_kind():
    declared = set(get_args(ArtifactKind))
    for aid, a in agent_registry().items():
        assert a["produces"] in declared, (
            f"{aid} produces {a['produces']!r}, which is not an ArtifactKind")


def test_registry_entries_are_real_agent_subclasses():
    reg = agent_registry()
    assert reg, "registry is empty - the package walk found nothing"
    for aid, a in reg.items():
        assert a["id"] == aid
        assert a["agency"] in {"reasoning", "deterministic", "hybrid"}
        assert isinstance(a["uses_llm"], bool)


@pytest.mark.parametrize("aid", sorted(agent_registry()))
def test_uses_llm_matches_what_the_module_actually_does(aid):
    """A reasoning agent must call the model; a deterministic one must not.

    This is the drift that matters most: an agent that advertises itself as
    reasoning while never calling `ctx.llm` is a claim the code does not
    back up.
    """
    a = agent_registry()[aid]
    src = (AGENT_DIR / f"{a['module']}.py").read_text(encoding="utf-8",
                                                      errors="replace")
    calls = bool(re.search(r"ctx\.llm\.(complete|complete_json)", src))
    assert calls == a["uses_llm"], (
        f"{aid}: uses_llm={a['uses_llm']} but "
        f"{'no ' if not calls else ''}ctx.llm call found in {a['module']}.py")


# --------------------------------------------------- the health endpoint

@pytest.fixture(scope="module")
def health():
    from fastapi.testclient import TestClient
    import main                                       # noqa: WPS433
    r = TestClient(main.app).get("/api/v2/health")
    assert r.status_code == 200
    return r.json()


def test_health_reports_exactly_the_registry(health):
    assert sorted(health["agents"]) == sorted(agent_registry())
    assert health["agent_count"] == len(agent_registry())


def test_health_counts_reasoning_agents_from_the_registry(health):
    expected = sum(1 for a in agent_registry().values() if a["uses_llm"])
    assert health["reasoning_agents"] == expected


def test_health_does_not_carry_a_hardcoded_agent_list():
    """Regression guard: the literal list must not come back."""
    src = (AGENT_DIR / "api.py").read_text(encoding="utf-8")
    assert '"clip_analysis", "editing_director"' not in src
    assert not re.search(r'"phase"\s*:\s*"[\d.]+"', src), (
        "a hardcoded phase string reappeared in api.py")


def test_health_states_whether_reasoning_is_actually_enabled(health,
                                                             monkeypatch):
    """With no key the endpoint must say so, and say why."""
    if not health["reasoning_enabled"]:
        assert health["model"] is None
        assert health["why_no_model"], "disabled but no reason given"
    else:
        assert health["model"], "enabled but no model named"


def test_health_reports_the_configured_model_when_a_key_is_set(monkeypatch):
    from fastapi.testclient import TestClient
    import main                                       # noqa: WPS433
    monkeypatch.setenv("AT_LLM_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AT_LLM_MODEL", "claude-test-model")
    d = TestClient(main.app).get("/api/v2/health").json()
    assert d["reasoning_enabled"] is True
    assert d["model"] == "claude-test-model"
    assert d["provider"] == "anthropic"
    assert d["why_no_model"] is None
