"""
Import bridge to the existing, unmodified code.

`pipeline/build_video.py` is a CLI script, not a package, and the `server/`
modules are imported flat (uvicorn is started with `--app-dir server`). This
module makes both importable from anywhere - a pytest run at the repo root, a
graph node, or the runner - WITHOUT moving, renaming or editing a single
existing file.

Importing build_video.py is safe: its `main()` is guarded by
`if __name__ == "__main__"`, and its module-level work is reading four
environment variables and probing for a font file.

One caveat that is deliberately surfaced rather than hidden: build_video.py
reads AT_XFADE_T, AT_CRF, AT_PRESET and AT_XFADES at IMPORT time into module
constants. In-process callers therefore see whatever was set when this module
was first imported. That is fine for Phase 1 because the render itself runs in
a SUBPROCESS with a freshly-built environment (see render_tools.py), which is
the same mechanism server/adapter.py already uses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

APP_ROOT = Path(__file__).resolve().parents[3]
SERVER_DIR = APP_ROOT / "server"
PIPELINE_DIR = APP_ROOT / "pipeline"
BUILD_VIDEO_PY = PIPELINE_DIR / "build_video.py"


def _ensure_server_on_path() -> None:
    p = str(SERVER_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_script(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:          # pragma: no cover
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The existing pipeline, loaded as a module. NOT modified, NOT copied.
build_video: ModuleType = _load_script("at_build_video", BUILD_VIDEO_PY)

_ensure_server_on_path()

import adapter as adapter_mod          # noqa: E402  existing, unmodified
import config as config_mod            # noqa: E402  existing, unmodified
import extras as extras_mod            # noqa: E402  existing, unmodified


def pipeline_path() -> Path:
    return BUILD_VIDEO_PY


__all__ = ["build_video", "adapter_mod", "config_mod", "extras_mod",
           "APP_ROOT", "SERVER_DIR", "PIPELINE_DIR", "pipeline_path"]
