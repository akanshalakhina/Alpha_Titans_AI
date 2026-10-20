"""Shared fixtures for the cloud tests.

Nothing here touches a real Google Cloud resource. Compute Engine and Drive
are replaced by in-memory fakes that keep the same contracts, the clock is
a number the test moves, and the worker talks to the REAL control-plane
routes through FastAPI's TestClient - so the queue, the VM manager, the API
and the worker client are all the production code.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_TMP = pathlib.Path(tempfile.mkdtemp(prefix="at_cloud_"))
os.environ.setdefault("AT_UPLOADS_DIR", str(_TMP / "uploads"))
os.environ.setdefault("AT_OUTPUTS_DIR", str(_TMP / "outputs"))

from fakes import (API_TOKEN, WORKER_TOKEN, Clock, ClientHttp,  # noqa: E402
                   FakeCompute, FakeDrive, cloud_env)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def compute():
    return FakeCompute()


@pytest.fixture
def drive():
    return FakeDrive()


@pytest.fixture
def env(tmp_path):
    return cloud_env(tmp_path)


@pytest.fixture
def cfg(env):
    from cloud.config import CloudConfig
    return CloudConfig.from_env(env)


@pytest.fixture
def queue(cfg, clock):
    from cloud.queue import JobQueue
    return JobQueue(cfg.queue_db, clock=clock, lease_s=cfg.lease_s,
                    max_attempts=cfg.max_attempts,
                    retry_base_s=cfg.retry_base_s,
                    rate_limit_backoff_s=cfg.rate_limit_backoff_s)


@pytest.fixture
def vm(queue, compute, cfg, clock):
    from cloud.vm import VmManager
    return VmManager(queue, compute, cfg, clock=clock)


@pytest.fixture
def plane(env, drive, compute, clock):
    """The real control plane: a FastAPI app with the cloud router mounted,
    over the fakes. Returns (client, ctx)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cloud.app import mount

    app = FastAPI()
    ctx = mount(app, host="127.0.0.1", env=env, drive=drive,
                compute=compute, clock=clock)
    assert ctx is not None, "the test configuration must be valid"
    with TestClient(app) as client:
        yield client, ctx


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {API_TOKEN}"}


@pytest.fixture
def worker_cp(plane):
    """The production worker client, its HTTP routed into the test app."""
    from cloud.worker import ControlPlane
    client, _ctx = plane
    return ControlPlane("http://cp.test", lambda: f"Bearer {WORKER_TOKEN}",
                        ClientHttp(client), "test-host:boot0001")
