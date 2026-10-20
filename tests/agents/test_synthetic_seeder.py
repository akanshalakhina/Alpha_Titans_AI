"""The seeder must satisfy every data gate, and claim nothing more.

The Learning agent has never once run against real outcomes, so none of
the code that counts, cohorts or windows them has been exercised. The
seeder puts thirty measured reports somewhere isolated so that path can
be proved today rather than in however long it takes to publish thirty
videos.

Two things it must not do: leak into the real store, and overstate what
it proves. Enough samples opens the four DATA gates and nothing else -
`holdout_validation`, `practical_effect` and `human_approval` stay
unevaluated because no comparison is implemented, and `Learning.learned`
is `Literal[False]`, so there is nowhere to record a finding even if one
were computed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.learning import (MIN_COHORT, MIN_HISTORY,        # noqa: E402
                             MIN_METRIC_AGE_DAYS,
                             MIN_WINDOW_DAYS, _gates, _history)
from agents.store import ArtifactStore                       # noqa: E402


def _seeder():
    path = ROOT / "scripts" / "seed_synthetic_analytics.py"
    spec = importlib.util.spec_from_file_location("seeder_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def seeder(tmp_path, monkeypatch):
    """The real script, pointed somewhere disposable."""
    mod = _seeder()
    home = tmp_path / "_synthetic"
    monkeypatch.setattr(mod, "HOME", home)
    monkeypatch.setattr(mod, "DB", home / "state.db")
    monkeypatch.setattr(mod, "CSV_FILE", home / "_synthetic_export.csv")
    return mod


# ================================================= what the rows look like

def test_there_are_exactly_as_many_rows_as_the_gate_asks_for(seeder):
    assert len(seeder._rows()) == MIN_HISTORY


def test_two_full_cohorts_plus_a_holdout(seeder):
    from collections import Counter
    c = Counter(r["cohort"] for r in seeder._rows())
    assert c["fast-pacing"] == MIN_COHORT and c["slow-pacing"] == MIN_COHORT
    assert c["holdout"] == MIN_HISTORY - 2 * MIN_COHORT


def test_every_metric_is_mature(seeder):
    """A metric read sooner than T+7d is not mature, and the gate is
    right to say so - seeding immature rows would prove nothing."""
    import time
    cutoff = time.time() - MIN_METRIC_AGE_DAYS * seeder.DAY
    for r in seeder._rows():
        last = time.mktime(time.strptime(r["last_day"], "%Y-%m-%d"))
        assert last <= cutoff, r["video_id"]


def test_the_window_is_wide_enough_for_the_gate(seeder):
    assert max(seeder._window_days(r) for r in seeder._rows()) \
        >= MIN_WINDOW_DAYS


def test_the_signal_is_actually_in_the_numbers(seeder):
    """A comparison that is written later has to be able to recover it."""
    rows = seeder._rows()
    fast = [r["views"] for r in rows if r["cohort"] == "fast-pacing"]
    slow = [r["views"] for r in rows if r["cohort"] == "slow-pacing"]
    ratio = (sum(fast) / len(fast)) / (sum(slow) / len(slow))
    assert ratio == pytest.approx(seeder.EFFECT, rel=0.15), ratio


def test_every_video_id_is_distinct_and_the_right_shape(seeder):
    ids = [r["video_id"] for r in seeder._rows()]
    assert len(set(ids)) == len(ids)
    assert all(len(v) == 11 for v in ids)


# ========================================================= seeding for real

def _seeded(seeder):
    """Seed, then run the agent's own history and comparison over it.

    `include_seeded=True` is the same opt-in `--verify` uses, and the
    reason it has to be asked for: without it the agent refuses these
    rows outright, which is the behaviour the production path wants and
    the one thing that would make this file untestable.
    """
    from agents.learning import _samples, compare
    store = ArtifactStore(seeder.DB)
    job = "synthetic-001"
    h = _history(store, job)
    return h, compare(_samples(store, job, include_seeded=True))


def test_seeding_opens_every_gate_except_approval(seeder, capsys):
    assert seeder.seed() == 0
    capsys.readouterr()
    h, finding = _seeded(seeder)

    assert h.usable_samples == MIN_HISTORY and h.samples_needed == 0
    assert h.zero_reports == 0 and h.unavailable_reports == 0
    assert h.window_days >= MIN_WINDOW_DAYS
    assert finding is not None, "30 samples with a pacing each must compare"

    states = {g.gate: g.state for g in _gates(h, finding, False, "")}
    for gate in ("sufficient_history", "valid_nonzero_data", "cohort_size",
                 "time_window", "holdout_validation", "practical_effect"):
        assert states[gate] == "passed", (gate, states[gate])
    assert states["human_approval"] == "failed",         "no amount of data opens this one"


def test_the_planted_signal_is_the_one_recovered(seeder, capsys):
    seeder.seed()
    capsys.readouterr()
    _h, finding = _seeded(seeder)
    assert finding.treatment.value == "fast"
    assert finding.lift == pytest.approx(seeder.EFFECT, rel=0.2)
    assert finding.holdout_agrees, "the holdout carries both arms"


def test_approval_closes_the_chain(seeder, capsys):
    from agents.learning import _approval, approve
    seeder.seed()
    capsys.readouterr()
    h, finding = _seeded(seeder)
    approve("synthetic-001")
    ok, who = _approval("synthetic-001")
    states = {g.gate: g.state for g in _gates(h, finding, ok, who)}
    assert all(st == "passed" for st in states.values())
    assert who == "operator"


def test_the_dev_override_never_passes_for_a_person(monkeypatch):
    from agents.learning import _approval
    monkeypatch.setenv("AT_LEARNING_AUTO_APPROVE", "1")
    ok, who = _approval("never-approved-by-hand")
    assert ok and who == "auto_dev_override",         "a reader of the artifact must see the word, not a name"


def test_it_claims_nothing_about_the_three_it_cannot_reach(seeder, capsys):
    seeder.seed()
    capsys.readouterr()
    states = {g.gate: g.state for g in
              _gates(_history(ArtifactStore(seeder.DB), "synthetic-001"))}
    for gate in ("holdout_validation", "practical_effect",
                 "human_approval"):
        assert states[gate] == "not_evaluated", \
            f"{gate} cannot pass: no comparison is implemented"


def test_the_cohort_message_stops_lying_once_samples_exist(seeder, capsys):
    """'no cohort can be formed' was true of zero samples and is a lie
    about thirty."""
    seeder.seed()
    capsys.readouterr()
    h, finding = _seeded(seeder)
    g = next(g for g in _gates(h, finding, False, "")
             if g.gate == "cohort_size")
    assert "fast: 12 video(s) against slow: 12" in g.observed


def test_the_export_csv_is_written_in_the_real_shape(seeder, capsys):
    seeder.seed()
    capsys.readouterr()
    from agents.tools import analytics_store as AS
    ex = AS.read_export(seeder.HOME)
    assert len(ex.rows) == MIN_HISTORY and not ex.skipped


# ============================================================ isolation

def test_nothing_is_written_to_the_real_store(seeder, capsys):
    from agents.store import shared_db_path
    real = pathlib.Path(shared_db_path())
    before = real.stat().st_mtime if real.exists() else None
    seeder.seed()
    capsys.readouterr()
    after = real.stat().st_mtime if real.exists() else None
    assert before == after, "the production store was touched"


def test_seeding_twice_is_refused_rather_than_doubled(seeder, capsys):
    seeder.seed()
    capsys.readouterr()
    assert seeder.seed() == 1
    assert "already seeded" in capsys.readouterr().out


def test_flush_restores_the_prior_state_exactly(seeder, capsys):
    seeder.seed()
    capsys.readouterr()
    assert seeder.HOME.exists()
    assert seeder.flush() == 0
    capsys.readouterr()
    assert not seeder.HOME.exists(), "a seeded row must not survive a flush"


def test_a_locked_store_is_named_rather_than_half_removed(seeder, capsys,
                                                          monkeypatch):
    """Windows refuses to unlink an open sqlite file. Half a flush is the
    worst outcome - some synthetic rows gone, some left to be mistaken
    for real ones - so it reports what remains instead."""
    seeder.seed()
    capsys.readouterr()

    def _locked(path):
        raise PermissionError(32, "in use", str(seeder.DB))
    monkeypatch.setattr(seeder.shutil, "rmtree", _locked)

    assert seeder.flush() == 1
    out = capsys.readouterr().out
    assert "open in another process" in out and "state.db" in out
    assert seeder.HOME.exists(), "nothing was removed"


def test_flushing_nothing_is_harmless(seeder, capsys):
    assert seeder.flush() == 0
    assert "nothing to flush" in capsys.readouterr().out


def test_verify_before_seeding_says_so(seeder, capsys):
    assert seeder.verify() == 1
    assert "nothing seeded" in capsys.readouterr().out


def test_every_seeded_report_says_it_is_synthetic(seeder, capsys):
    """Anyone reading one of these later must not mistake it for a real
    video's numbers."""
    from agents.artifacts import PerformanceAnalysis
    seeder.seed()
    capsys.readouterr()
    store = ArtifactStore(seeder.DB)
    for i in (1, 15, MIN_HISTORY):
        art = store.get(f"synthetic-{i:03d}", "PerformanceAnalysis",
                        PerformanceAnalysis)
        assert any("SYNTHETIC" in line for line in art.payload.limitations)
