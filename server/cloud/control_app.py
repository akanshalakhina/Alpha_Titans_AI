"""The CONTROL PLANE's web app: the cloud routes and page, and nothing else.

    uvicorn cloud.control_app:app --app-dir server --host 0.0.0.0 --port 8000

The full app (`main:app`) also carries the local studio's routes - upload,
produce, delete - and none of them has authentication. They also render
video, which a 1 GB e2-micro must never be asked to do. Publishing that app
would publish all of it, so the control plane serves this one instead:
`/cloud`, the three static files that page loads, `/api/cloud/*` and a
health check. Anything else is a 404, by construction rather than by a rule
somebody has to remember to configure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

import config

from .app import mount
from .sanitize import redact

PUBLIC_STATIC = frozenset({"cloud.js", "styles.css", "studio.css"})


class _Redacting(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), 2000)
        record.args = ()
        return True


def _logging() -> None:
    root = logging.getLogger()
    if any(isinstance(f, _Redacting) for h in root.handlers
           for f in h.filters):
        return
    handler = logging.StreamHandler()
    handler.addFilter(_Redacting())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def build(env=None, **overrides) -> FastAPI:
    _logging()
    web = Path(config.FRONTEND_DIR)
    app = FastAPI(title="Alpha Titans control plane", docs_url=None,
                  redoc_url=None, openapi_url=None)
    ctx = mount(app, host=config.HOST, web_dir=web, env=env, **overrides)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/cloud")

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"ok": True, "cloud": ctx is not None}

    @app.get("/static/{name}", include_in_schema=False)
    def static(name: str):
        if name not in PUBLIC_STATIC:
            raise HTTPException(404)
        return FileResponse(web / name, headers={"Cache-Control": "no-cache"})

    return app


app = build()
