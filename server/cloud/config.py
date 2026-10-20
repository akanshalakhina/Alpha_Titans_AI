"""Configuration for the cloud path, read from the environment and validated.

Every value that names real infrastructure - project, zone, instance, Drive
folders, the worker's service account - has NO default. A missing or
malformed one is reported by `problems()` and the component refuses to start,
rather than guessing a production value. They are only ever read from here;
nothing a request carries can reach them.

Secrets (`AT_CLOUD_API_TOKEN`, `AT_CLOUD_WORKER_TOKEN`) are read but never
echoed: `summary()` reports only whether each one is set.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

# GCE naming rules. Checked so a typo fails at startup with a clear message
# instead of as a 400 from the Compute API at 3am, and so nothing that is not
# a plain resource name can ever be spliced into a URL template.
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_ZONE = re.compile(r"^[a-z]+-[a-z]+[0-9]+-[a-z]$")
_INSTANCE = re.compile(r"^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$")
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,200}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_PREFIX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,60}$")
_SA_EMAIL = re.compile(
    r"^[a-z0-9-]+@([a-z0-9-]+\.iam\.gserviceaccount\.com"
    r"|developer\.gserviceaccount\.com)$")

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _flag(env, key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _num(env, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(env.get(key, default))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


@dataclass(frozen=True)
class CloudConfig:
    enabled: bool = False
    role: str = "control"                    # control | worker

    # ---- where things are (no defaults for real infrastructure)
    queue_db: Path = Path("outputs/_cloud/queue.db")
    public_url: str = ""                     # control plane base URL
    project: str = ""
    zone: str = ""
    instance: str = ""
    drive_input_folder: str = ""         # Drive folder id, or GCS prefix
    drive_output_folder: str = ""        # Drive folder id, or GCS prefix
    shared_drive: bool = True
    storage: str = "drive"               # drive | gcs
    gcs_bucket: str = ""

    # ---- identity
    token_source: str = "metadata"           # metadata | gcloud
    api_token: str = field(default="", repr=False)
    # No sign-in for the page: anyone with the link may make videos. Each
    # browser sees only the jobs it made (a job id is a 48-bit capability),
    # new jobs are capped per day, and admin routes still need the token.
    open_access: bool = False
    max_jobs_per_day: int = 20
    worker_auth: str = "google_id_token"     # google_id_token | shared_token
    worker_sa_email: str = ""
    worker_token: str = field(default="", repr=False)

    # ---- upload
    upload_mode: str = "relay"               # relay | direct
    max_upload_bytes: int = 2048 * 1024 * 1024
    max_files_per_job: int = 8

    # ---- timing (seconds), each clamped to a sane range
    lease_s: float = 300.0
    heartbeat_s: float = 30.0
    startup_timeout_s: float = 900.0
    idle_grace_s: float = 180.0
    max_job_s: float = 5400.0
    max_activation_s: float = 6 * 3600.0
    reconcile_s: float = 60.0
    min_start_interval_s: float = 120.0
    max_starts_per_hour: int = 6
    queue_timeout_s: float = 24 * 3600.0
    upload_timeout_s: float = 24 * 3600.0
    retry_base_s: float = 60.0
    rate_limit_backoff_s: float = 900.0
    max_attempts: int = 3

    # ---- worker host checks
    require_gpu: bool = False
    min_free_bytes: int = 5 * 1024 ** 3
    keep_media: bool = False             # keep renders after upload
    poll_s: float = 15.0                 # idle worker's claim interval

    @classmethod
    def from_env(cls, env=None, app_root: Path | None = None) -> "CloudConfig":
        env = os.environ if env is None else env
        root = app_root or Path(__file__).resolve().parents[2]
        db = env.get("AT_CLOUD_DB") or "outputs/_cloud/queue.db"
        db_path = Path(db)
        if not db_path.is_absolute():
            db_path = (root / db_path).resolve()
        max_mb = int(_num(env, "AT_MAX_CLIP_MB", 2048, 1, 64 * 1024))
        storage = (env.get("AT_CLOUD_STORAGE") or "drive").strip().lower()
        if storage == "gcs":
            # in a bucket the two "folders" are object-name prefixes
            inputs = (env.get("AT_GCS_INPUT_PREFIX") or "inputs").strip("/ ")
            outputs = (env.get("AT_GCS_OUTPUT_PREFIX")
                       or "outputs").strip("/ ")
        else:
            inputs = (env.get("AT_DRIVE_INPUT_FOLDER_ID") or "").strip()
            outputs = (env.get("AT_DRIVE_OUTPUT_FOLDER_ID") or "").strip()
        return cls(
            enabled=_flag(env, "AT_CLOUD_ENABLED"),
            role=(env.get("AT_CLOUD_ROLE") or "control").strip().lower(),
            queue_db=db_path,
            public_url=(env.get("AT_CLOUD_PUBLIC_URL") or "").rstrip("/"),
            project=(env.get("AT_GCP_PROJECT") or "").strip(),
            zone=(env.get("AT_GCP_ZONE") or "").strip(),
            instance=(env.get("AT_GCP_WORKER_INSTANCE") or "").strip(),
            drive_input_folder=inputs,
            drive_output_folder=outputs,
            shared_drive=_flag(env, "AT_DRIVE_SHARED_DRIVE", True),
            storage=storage,
            gcs_bucket=(env.get("AT_GCS_BUCKET") or "").strip(),
            token_source=(env.get("AT_CLOUD_TOKEN_SOURCE")
                          or "metadata").strip().lower(),
            api_token=env.get("AT_CLOUD_API_TOKEN") or "",
            open_access=_flag(env, "AT_CLOUD_OPEN_ACCESS", False),
            max_jobs_per_day=int(_num(env, "AT_CLOUD_MAX_JOBS_PER_DAY", 20,
                                      1, 10000)),
            worker_auth=(env.get("AT_CLOUD_WORKER_AUTH")
                         or "google_id_token").strip().lower(),
            worker_sa_email=(env.get("AT_CLOUD_WORKER_SA_EMAIL")
                             or "").strip().lower(),
            worker_token=env.get("AT_CLOUD_WORKER_TOKEN") or "",
            upload_mode=(env.get("AT_CLOUD_UPLOAD_MODE")
                         or "relay").strip().lower(),
            max_upload_bytes=max_mb * 1024 * 1024,
            max_files_per_job=int(_num(env, "AT_CLOUD_MAX_FILES", 8, 1, 64)),
            lease_s=_num(env, "AT_CLOUD_LEASE_S", 300, 60, 3600),
            heartbeat_s=_num(env, "AT_CLOUD_HEARTBEAT_S", 30, 5, 600),
            startup_timeout_s=_num(env, "AT_CLOUD_STARTUP_TIMEOUT_S", 900,
                                   120, 3600),
            idle_grace_s=_num(env, "AT_CLOUD_IDLE_GRACE_S", 180, 0, 3600),
            max_job_s=_num(env, "AT_CLOUD_MAX_JOB_S", 5400, 300, 6 * 3600),
            max_activation_s=_num(env, "AT_CLOUD_MAX_ACTIVATION_S",
                                  6 * 3600, 600, 24 * 3600),
            reconcile_s=_num(env, "AT_CLOUD_RECONCILE_S", 60, 10, 3600),
            min_start_interval_s=_num(env, "AT_CLOUD_MIN_START_INTERVAL_S",
                                      120, 0, 3600),
            max_starts_per_hour=int(_num(env, "AT_CLOUD_MAX_STARTS_PER_HOUR",
                                         6, 1, 60)),
            queue_timeout_s=_num(env, "AT_CLOUD_QUEUE_TIMEOUT_S", 86400,
                                 600, 30 * 86400),
            upload_timeout_s=_num(env, "AT_CLOUD_UPLOAD_TIMEOUT_S", 86400,
                                  600, 7 * 86400),
            retry_base_s=_num(env, "AT_CLOUD_RETRY_BASE_S", 60, 1, 3600),
            rate_limit_backoff_s=_num(env, "AT_CLOUD_RATE_LIMIT_BACKOFF_S",
                                      900, 60, 24 * 3600),
            max_attempts=int(_num(env, "AT_CLOUD_MAX_ATTEMPTS", 3, 1, 10)),
            require_gpu=_flag(env, "AT_CLOUD_REQUIRE_GPU"),
            min_free_bytes=int(_num(env, "AT_CLOUD_MIN_FREE_GB", 5, 0,
                                    10_000) * 1024 ** 3),
            keep_media=_flag(env, "AT_CLOUD_KEEP_MEDIA"),
            poll_s=_num(env, "AT_CLOUD_POLL_S", 15, 1, 300),
        )

    # ------------------------------------------------------------ checking

    def problems(self, role: str | None = None, host: str = "127.0.0.1"
                 ) -> list[str]:
        """Everything that would stop `role` from working, in plain words.
        Never includes a secret's value."""
        role = role or self.role
        out: list[str] = []
        if role not in ("control", "worker"):
            out.append(f"AT_CLOUD_ROLE must be 'control' or 'worker', "
                       f"not {role!r}")
            return out
        if self.token_source not in ("metadata", "gcloud"):
            out.append("AT_CLOUD_TOKEN_SOURCE must be 'metadata' or 'gcloud'")
        if self.worker_auth not in ("google_id_token", "shared_token"):
            out.append("AT_CLOUD_WORKER_AUTH must be 'google_id_token' or "
                       "'shared_token'")
        if self.upload_mode not in ("relay", "direct"):
            out.append("AT_CLOUD_UPLOAD_MODE must be 'relay' or 'direct'")
        if self.storage not in ("drive", "gcs"):
            out.append("AT_CLOUD_STORAGE must be 'drive' or 'gcs'")
        elif self.storage == "gcs":
            if not _BUCKET.match(self.gcs_bucket or ""):
                out.append("AT_GCS_BUCKET is not set or not a valid bucket "
                           "name")
            if self.upload_mode != "relay":
                out.append("Cloud Storage supports relay uploads only")
            for key, val in (("AT_GCS_INPUT_PREFIX", self.drive_input_folder),
                             ("AT_GCS_OUTPUT_PREFIX",
                              self.drive_output_folder)):
                if not _PREFIX.match(val or ""):
                    out.append(f"{key} is not a valid prefix")

        if role == "control":
            checks = [("AT_GCP_PROJECT", self.project, _PROJECT),
                      ("AT_GCP_ZONE", self.zone, _ZONE),
                      ("AT_GCP_WORKER_INSTANCE", self.instance, _INSTANCE)]
            if self.storage == "drive":
                checks += [("AT_DRIVE_INPUT_FOLDER_ID",
                            self.drive_input_folder, _DRIVE_ID),
                           ("AT_DRIVE_OUTPUT_FOLDER_ID",
                            self.drive_output_folder, _DRIVE_ID)]
            for key, val, rx in checks:
                if not val:
                    out.append(f"{key} is not set")
                elif not rx.match(val):
                    out.append(f"{key} is not a valid value")
            if (host not in LOOPBACK and not self.api_token
                    and not self.open_access):
                out.append("AT_CLOUD_API_TOKEN must be set before the cloud "
                           "routes are served on a non-loopback host "
                           f"({host})")
            if self.api_token and len(self.api_token) < 24:
                out.append("AT_CLOUD_API_TOKEN is too short (24+ characters)")
        if role == "worker" and not self.public_url:
            out.append("AT_CLOUD_PUBLIC_URL (the control plane's address) "
                       "is not set")

        if self.worker_auth == "google_id_token":
            if role == "control" and not _SA_EMAIL.match(
                    self.worker_sa_email or ""):
                out.append("AT_CLOUD_WORKER_SA_EMAIL must name the worker "
                           "VM's service account")
            if role == "control" and not self.public_url:
                out.append("AT_CLOUD_PUBLIC_URL is the ID-token audience and "
                           "must be set")
        elif len(self.worker_token) < 32:
            out.append("AT_CLOUD_WORKER_TOKEN must be 32+ characters in "
                       "shared_token mode")
        return out

    def summary(self) -> dict:
        """Safe to log or return to an operator: secrets appear only as
        whether they are set."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in ("api_token", "worker_token"):
                out[f.name] = "set" if v else "unset"
            elif isinstance(v, Path):
                out[f.name] = str(v)
            else:
                out[f.name] = v
        return out
