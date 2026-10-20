"""Wiring the cloud path into the existing FastAPI app.

`mount()` is the only thing `server/main.py` calls, and it does nothing at
all unless `AT_CLOUD_ENABLED=1`: no route, no thread, no database. The local
path - upload, produce, download - is untouched either way.

When enabled but misconfigured, the problems are logged (never a secret's
value) and only `/api/cloud/config` is mounted, reporting them. A
half-configured control plane that silently accepted uploads it could never
process would be worse than one that says what is missing.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from dataclasses import dataclass

from .config import CloudConfig

log = logging.getLogger("alpha.cloud")


@dataclass
class CloudContext:
    cfg: CloudConfig
    queue: object
    drive: object
    vm: object
    reconciler: object
    worker_auth: object                  # callable(token) -> claims
    validate_settings: object            # callable(dict) -> dict


def _settings_validator():
    """The SAME schema the local `produce` route accepts, so a cloud job and
    a local job cannot be asked for different things."""
    try:
        from agents.artifacts import JobSettings
    except Exception:                                 # noqa: BLE001
        JobSettings = None

    def validate(raw: dict) -> dict:
        from .sanitize import CloudError
        if not isinstance(raw, dict):
            raise CloudError("invalid", "settings must be an object")
        if JobSettings is None:
            raise CloudError("internal", "the agent layer is not installed")
        try:
            return JobSettings(**raw).model_dump(mode="json")
        except Exception as exc:                      # noqa: BLE001
            first = getattr(exc, "errors", lambda: [{}])()[0]
            where = ".".join(str(x) for x in first.get("loc", ())) or "settings"
            raise CloudError("invalid", f"{where}: "
                             f"{first.get('msg', 'invalid')}") from None
    return validate


def build_context(cfg: CloudConfig, *, http=None, tokens=None, drive=None,
                  compute=None, worker_auth=None, clock=time.time
                  ) -> CloudContext:
    from .gcp import Compute, Drive, Http, token_source
    from .idtoken import GoogleIdVerifier
    from .queue import JobQueue
    from .reconcile import Reconciler
    from .vm import VmManager

    http = http or Http()
    if drive is None or compute is None:
        tokens = tokens or token_source(cfg.token_source)
    if drive is None:
        if cfg.storage == "gcs":
            from .gcs import GcsStore
            drive = GcsStore(http, tokens, cfg.gcs_bucket)
        else:
            drive = Drive(http, tokens, cfg.shared_drive)
    compute = compute or Compute(http, tokens, cfg.project, cfg.zone,
                                 cfg.instance)
    queue = JobQueue(cfg.queue_db, clock=clock, lease_s=cfg.lease_s,
                     max_attempts=cfg.max_attempts,
                     retry_base_s=cfg.retry_base_s,
                     rate_limit_backoff_s=cfg.rate_limit_backoff_s)
    vm = VmManager(queue, compute, cfg, clock=clock)
    if worker_auth is None:
        if cfg.worker_auth == "google_id_token":
            worker_auth = GoogleIdVerifier(cfg.public_url,
                                           cfg.worker_sa_email,
                                           http=http, clock=clock).verify
        else:
            expected = cfg.worker_token.encode()

            def worker_auth(tok: str):
                from .sanitize import CloudError
                if not hmac.compare_digest(tok.encode(), expected):
                    raise CloudError("auth", "bad worker token")
                return {"email": "shared-token-worker"}
    return CloudContext(cfg=cfg, queue=queue, drive=drive, vm=vm,
                        reconciler=Reconciler(queue, vm, cfg, clock=clock),
                        worker_auth=worker_auth,
                        validate_settings=_settings_validator())


def mount(app, host: str, web_dir=None, env=None, **overrides):
    """Attach the cloud routes to `app` if, and only if, enabled and
    correctly configured. Returns the context, or None."""
    cfg = CloudConfig.from_env(env)
    if not cfg.enabled:
        return None
    problems = cfg.problems("control", host)
    if problems:
        for p in problems:
            log.error("cloud path not started: %s", p)
        from fastapi import APIRouter
        r = APIRouter(prefix="/api/cloud")

        @r.get("/config")
        def misconfigured():
            return {"enabled": False, "problems": problems}

        app.include_router(r)
        return None

    from .api import build_router, page_route
    ctx = build_context(cfg, **overrides)
    app.include_router(build_router(ctx))
    if web_dir is not None:
        app.include_router(page_route(web_dir))
    if (env if env is not None else os.environ).get(
            "AT_CLOUD_RECONCILE_THREAD", "1") != "0":
        ctx.reconciler.start()
    log.info("cloud control plane mounted: %s", {
        k: v for k, v in cfg.summary().items()
        if k in ("role", "upload_mode", "project", "zone", "instance",
                 "worker_auth", "token_source", "api_token")})
    app.state.cloud = ctx
    return ctx
