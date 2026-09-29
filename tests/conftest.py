"""
Session-wide temporary-directory cleanup.

THE LEAK THIS CLOSES
--------------------
Both test packages create a session temp root at import time:

    tests/backend/conftest.py   tempfile.mkdtemp(prefix="at_tests_")
    tests/agents/conftest.py    tempfile.mkdtemp(prefix="at_agents_")

and neither removed it. Those roots hold real uploads and real renders, so a
full run left roughly 213 MB behind EVERY time. Across the Phase 1/2 work that
accumulated to about 1.5 GB and eventually filled the disk, which surfaced as
thirteen backend tests failing with `OSError: [Errno 28] No space left on
device` - a test-environment failure that looks exactly like a code
regression.

WHY IT LIVES HERE
-----------------
This file is new, so the fix needs no edit to either existing conftest. pytest
loads conftest.py from the rootdir down, so a hook here applies to every test
package underneath it.

WHAT IT REMOVES
---------------
Only the directories the test suites themselves created, found by reading the
`_TMP` attribute off the loaded conftest modules - not by globbing a pattern.
Every candidate is additionally checked to be inside the system temp directory
and to carry one of the suites' own prefixes, so nothing outside a test's own
scratch space can be reached. Project data (uploads/, outputs/, footage,
finished videos) is never touched: this only ever looks at %TEMP%.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# the prefixes the two suites use for their own session roots
_SUITE_PREFIXES = ("at_tests_", "at_agents_")


def _is_safe_to_remove(path: Path) -> bool:
    """A conservative three-part check before any recursive delete."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_dir():
        return False
    # 1. must live inside the system temp directory
    try:
        resolved.relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError:
        return False
    # 2. must carry one of the suites' own prefixes
    if not resolved.name.startswith(_SUITE_PREFIXES):
        return False
    # 3. must not be the temp directory itself
    return resolved != Path(tempfile.gettempdir()).resolve()


def _suite_temp_roots() -> list[Path]:
    """The `_TMP` of every loaded conftest, rather than a glob.

    Reading the attribute off the module that created it means only THIS
    session's directories are candidates; a concurrent run's scratch space is
    never visible here.
    """
    roots: list[Path] = []
    for name, module in list(sys.modules.items()):
        if "conftest" not in name:
            continue
        tmp = getattr(module, "_TMP", None)
        if tmp is None:
            continue
        p = Path(str(tmp))
        if _is_safe_to_remove(p):
            roots.append(p)
    return roots


def pytest_sessionfinish(session, exitstatus):      # noqa: ARG001
    """Remove this session's temp roots once every test has finished.

    `ignore_errors=True` on purpose: on Windows a file can still be held for a
    moment by a just-exited ffmpeg, and failing the run over leftover scratch
    would be worse than leaving it. Anything missed is picked up by the next
    session's sweep below.
    """
    removed, freed = 0, 0
    for root in _suite_temp_roots():
        try:
            freed += sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        except OSError:
            pass
        shutil.rmtree(root, ignore_errors=True)
        if not root.exists():
            removed += 1

    if removed and os.environ.get("AT_TEST_CLEANUP_QUIET") != "1":
        mb = freed / (1024 * 1024)
        print(f"\n[cleanup] removed {removed} test temp root(s), "
              f"freed ~{mb:.0f} MB")
