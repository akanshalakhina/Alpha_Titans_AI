"""The readable account of a build must outlive the build.

WHAT WAS BEING LOST
-------------------
`ctx.emit` appends to a plain list on the runtime. The CLI passes
`on_line` and sees every line as it happens; the server passed nothing,
so twenty minutes of lines accumulated in memory and went out with the
object. The trace database kept the structured record - per node,
status, tokens, errors, fallbacks - but not the sentences a person
reads: "performance: measured from oct.csv - 10432 view(s)".

That is the half somebody wants at nine the next morning, and it is
exactly the half a crash takes with it.

WHAT THIS IS NOT
----------------
Not a replacement for the trace. The trace is queryable and structured
and stays the source of truth; this is the narrative beside it. And not
a global logging framework: the 46 `print()` calls in this project are
all in CLI entry points - `runner`, `compat`, `migrate` - where printing
to stdout is correct, and routing them through `logging` would make
those tools worse, not better.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from agents.runner import RUN_LOG, _RunLog                   # noqa: E402


def _lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# ===================================================== it is written down

def test_every_line_reaches_the_file(tmp_path):
    log = _RunLog(tmp_path / RUN_LOG)
    log.append("clip_analysis: 2 clip(s) measured")
    log.append("learning: fast over slow: lift 2.24")

    got = _lines(tmp_path / RUN_LOG)
    assert len(got) == 2
    assert got[0].endswith("clip_analysis: 2 clip(s) measured")
    assert "lift 2.24" in got[1]


def test_each_line_is_stamped(tmp_path):
    """Knowing WHEN a step happened is most of what the log is for - a
    twenty-minute build with no times cannot be read for where it
    stalled."""
    import re
    log = _RunLog(tmp_path / RUN_LOG)
    log.append("render: starting")
    assert re.match(r"^\d{2}:\d{2}:\d{2} render: starting$",
                    _lines(tmp_path / RUN_LOG)[0])


def test_it_is_still_a_list(tmp_path):
    """The runtime reads `rt.log` as a list elsewhere; writing a file is
    something it does AS WELL, not instead."""
    log = _RunLog(tmp_path / RUN_LOG)
    log.append("a")
    log.append("b")
    assert list(log) == ["a", "b"] and len(log) == 2


def test_it_still_forwards_to_a_live_watcher(tmp_path):
    """The CLI's `on_line` must keep working - that is the live progress
    an operator watches while a build runs."""
    seen = []
    log = _RunLog(tmp_path / RUN_LOG, seen.append)
    log.append("story: 4 beats")
    assert seen == ["story: 4 beats"]
    assert _lines(tmp_path / RUN_LOG)[0].endswith("story: 4 beats")


def test_a_run_with_no_watcher_is_still_written(tmp_path):
    """This is the server's case, and the one that was losing everything."""
    log = _RunLog(tmp_path / RUN_LOG)
    log.append("music: upbeat at 129.1 BPM")
    assert _lines(tmp_path / RUN_LOG)


def test_the_folder_is_created_if_it_is_not_there(tmp_path):
    log = _RunLog(tmp_path / "outputs" / "job01" / RUN_LOG)
    log.append("first line of a brand new build")
    assert (tmp_path / "outputs" / "job01" / RUN_LOG).exists()


def test_it_is_flushed_as_it_goes(tmp_path):
    """A run that crashes is exactly the one whose log is wanted, so the
    lines must be on disk before the end rather than at it."""
    log = _RunLog(tmp_path / RUN_LOG)
    log.append("clip_analysis: starting")
    assert _lines(tmp_path / RUN_LOG), "readable mid-run, not only after"


def test_a_resumed_build_adds_to_the_log_it_had(tmp_path):
    """Resuming continues one build; a second log, or a truncated one,
    would hide the half that was already done."""
    _RunLog(tmp_path / RUN_LOG).append("before the server stopped")
    _RunLog(tmp_path / RUN_LOG).append("after it was resumed")
    assert len(_lines(tmp_path / RUN_LOG)) == 2


# ======================================== and it never stops a build

def test_a_path_that_cannot_be_written_is_survived(tmp_path):
    log = _RunLog(tmp_path / RUN_LOG)
    log._path = pathlib.Path("Z:/no/such/drive/run.log")
    log.append("this must not raise")
    assert list(log) == ["this must not raise"], \
        "a build must not die because a log line could not be written"


def test_it_stops_trying_after_a_write_fails(tmp_path):
    log = _RunLog(tmp_path / RUN_LOG)
    log._path = pathlib.Path("Z:/no/such/drive/run.log")
    log.append("one")
    assert log._path is None, "no point retrying a path that cannot work"
    log.append("two")
    assert list(log) == ["one", "two"]


def test_a_watcher_that_throws_does_not_stop_the_file(tmp_path):
    def _boom(_line):
        raise RuntimeError("the browser went away")
    log = _RunLog(tmp_path / RUN_LOG, _boom)
    log.append("render: 40%")
    assert _lines(tmp_path / RUN_LOG), "the file is written regardless"


# ============================== the prints that should NOT become logging

@pytest.mark.parametrize("module", ["runner", "compat", "migrate"])
def test_the_printing_modules_are_command_line_entry_points(module):
    """Pinning why 46 `print()` calls were left alone.

    Each of these is a CLI. Printing to stdout is what a CLI does, and
    routing it through `logging` would mean a tool whose output
    disappears unless a handler is configured.
    """
    src = (ROOT / "server" / "agents" / f"{module}.py").read_text(
        encoding="utf-8")
    assert "argparse" in src and '__main__' in src, module
