"""In-memory stand-ins for Compute Engine and Drive, plus test plumbing.

They keep the CONTRACTS the production code relies on - a resumable session
that holds only the bytes it acknowledged, an md5 computed by the "server",
an instance whose status moves when it is started and stopped - and they
can be told to fail, so the failure paths are exercised rather than assumed.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import threading
import time
import urllib.parse
from pathlib import Path

from cloud.gcp import CHUNK_ALIGN, DRIVE_API, DRIVE_UPLOAD, Response
from cloud.sanitize import CloudError

API_TOKEN = "op-token-" + "a1b2c3d4" * 4             # 41 chars, test only
WORKER_TOKEN = "worker-token-" + "f00dfeed" * 5       # 53 chars, test only
INPUT_FOLDER = "input_folder_000001"
OUTPUT_FOLDER = "output_folder_00001"


def cloud_env(tmp_path: Path, **extra) -> dict:
    env = {
        "AT_CLOUD_ENABLED": "1",
        "AT_CLOUD_ROLE": "control",
        "AT_CLOUD_DB": str(tmp_path / "queue.db"),
        "AT_CLOUD_PUBLIC_URL": "http://cp.test",
        "AT_GCP_PROJECT": "test-project-123",
        "AT_GCP_ZONE": "us-central1-a",
        "AT_GCP_WORKER_INSTANCE": "at-worker",
        "AT_DRIVE_INPUT_FOLDER_ID": INPUT_FOLDER,
        "AT_DRIVE_OUTPUT_FOLDER_ID": OUTPUT_FOLDER,
        "AT_CLOUD_API_TOKEN": API_TOKEN,
        "AT_CLOUD_WORKER_AUTH": "shared_token",
        "AT_CLOUD_WORKER_TOKEN": WORKER_TOKEN,
        "AT_CLOUD_RECONCILE_THREAD": "0",
        "AT_CLOUD_MIN_FREE_GB": "0",
    }
    env.update(extra)
    return env


class Clock:
    """A clock that moves only when told to."""

    def __init__(self, start: float = 1_800_000_000.0):
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, s: float) -> None:
        with self._lock:
            self.now += s

    def sleep(self, s: float) -> None:
        self.advance(s)


# --------------------------------------------------------------- compute

class FakeCompute:
    def __init__(self, status: str = "TERMINATED"):
        self._status = status
        self.calls: list[str] = []
        self.fail_start = 0             # this many start() calls raise
        self.fail_status = False
        self.start_delay = 0.0          # widen race windows
        self._lock = threading.Lock()

    def status(self) -> str:
        if self.fail_status:
            raise CloudError("transient", "status unavailable")
        return self._status

    def start(self) -> dict:
        if self.start_delay:
            time.sleep(self.start_delay)
        with self._lock:
            self.calls.append("start")
            if self.fail_start:
                self.fail_start -= 1
                raise CloudError("transient", "HTTP 503: backend error "
                                              "ya29.SECRETTOKEN")
            self._status = "RUNNING"
        return {"name": "op-start"}

    def stop(self) -> dict:
        with self._lock:
            self.calls.append("stop")
            self._status = "TERMINATED"
        return {"name": "op-stop"}

    def set(self, status: str) -> None:
        self._status = status

    def starts(self) -> int:
        return self.calls.count("start")


# ----------------------------------------------------------------- drive

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()               # noqa: S324


class FakeDrive:
    """The high-level Drive interface the API and worker use."""

    def __init__(self):
        self.files: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.fail_upload = 0            # worker uploads that raise
        self.corrupt_upload = False     # worker uploads that fail verify
        self.fail_download = 0
        self.downloads: list[str] = []
        self.uploads: list[str] = []
        self.short_ack = 0              # chunks acknowledged short of sent
        self._lock = threading.Lock()

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._ids):012d}"

    def add(self, data: bytes, name: str, parent: str = INPUT_FOLDER,
            mime: str = "video/mp4") -> dict:
        fid = self._new_id("file_")
        meta = {"id": fid, "name": name, "mimeType": mime,
                "size": str(len(data)), "md5Checksum": _md5(data),
                "parents": [parent], "trashed": False}
        self.files[fid] = {**meta, "data": data}
        return meta

    def _meta(self, fid: str) -> dict:
        f = self.files[fid]
        return {k: v for k, v in f.items() if k != "data"}

    # --- the interface

    def start_session(self, name, mime, size, parent, origin=None) -> str:
        with self._lock:
            n = len(self.sessions) + 1
            uri = (f"https://upload.fake/drive/v3/files?uploadType="
                   f"resumable&upload_id=SECRETSESSION{n:04d}")
            self.sessions[uri] = {"name": name, "mime": mime,
                                  "size": int(size), "parent": parent,
                                  "buf": bytearray()}
            return uri

    def put_chunk(self, uri, data, offset, total):
        with self._lock:
            s = self.sessions.get(uri)
            if s is None:
                raise CloudError("verification", "upload session expired")
            if offset != len(s["buf"]):
                return False, len(s["buf"]), None
            take = data
            if self.short_ack and len(data) > CHUNK_ALIGN:
                self.short_ack -= 1
                take = data[:len(data) - CHUNK_ALIGN]
            s["buf"] += take
            if len(s["buf"]) == s["size"]:
                meta = self.add(bytes(s["buf"]), s["name"], s["parent"],
                                s["mime"])
                s["file"] = meta["id"]
                return True, -1, meta
            return False, len(s["buf"]), None

    def query(self, uri, total):
        s = self.sessions.get(uri)
        if s is None:
            raise CloudError("verification", "upload session expired")
        if s.get("file"):
            return True, -1, self._meta(s["file"])
        return False, len(s["buf"]), None

    def get(self, file_id):
        if file_id not in self.files:
            raise CloudError("not_found", "HTTP 404")
        return self._meta(file_id)

    def find(self, parent, name):
        return [self._meta(i) for i, f in self.files.items()
                if parent in f["parents"] and f["name"] == name
                and not f["trashed"]]

    def create_folder(self, name, parent):
        for i, f in self.files.items():
            if (f["name"] == name and parent in f["parents"]
                    and f["mimeType"].endswith("folder")):
                return i
        fid = self._new_id("fold_")
        self.files[fid] = {"id": fid, "name": name, "parents": [parent],
                           "mimeType": "application/vnd.google-apps.folder",
                           "size": "0", "md5Checksum": "", "trashed": False,
                           "data": b""}
        return fid

    def trash(self, file_id):
        self.files[file_id]["trashed"] = True

    def upload(self, path, name, mime, parent, on_progress=None, retries=5):
        if self.fail_upload:
            self.fail_upload -= 1
            raise CloudError("transient", "HTTP 503 while uploading")
        data = Path(path).read_bytes()
        meta = self.add(data, name, parent, mime)
        self.uploads.append(name)
        if self.corrupt_upload:
            self.trash(meta["id"])
            raise CloudError("verification", f"{name}: md5 mismatch")
        return meta

    def download(self, file_id, dest, size, md5, on_progress=None,
                 retries=5):
        if self.fail_download:
            self.fail_download -= 1
            raise CloudError("transient", "HTTP 503 while downloading")
        data = self.files[file_id]["data"]
        if len(data) != size or (md5 and _md5(data) != md5):
            raise CloudError("verification", "download did not match")
        d = Path(dest)
        # the real client's contract (test_drive.py): a destination that is
        # already there and verified is not fetched again
        if d.exists() and d.stat().st_size == size and (
                not md5 or _md5(d.read_bytes()) == md5):
            return d
        d.write_bytes(data)
        self.downloads.append(file_id)
        return Path(dest)


# ----------------------------------------------------- drive, at HTTP level

class DriveHttp:
    """Speaks Drive's REST protocol, for testing the REAL `gcp.Drive`
    client: the resumable session, 308 with `Range`, ranged media GETs."""

    def __init__(self):
        self.files: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.ack_short = 0          # 308s acknowledging less than was sent
        self.fail_puts = 0          # PUTs that 503
        self.ignore_range = False   # media GETs that ignore Range
        self.corrupt_md5 = False    # the server reports a different md5
        self.cut_stream_at = None   # a media GET that ends early
        self._n = itertools.count(1)

    def request(self, method, url, headers=None, body=None, timeout=None):
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.calls.append((method, url))
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if method == "POST" and url.startswith(DRIVE_UPLOAD):
            uri = f"https://up.fake/s/{next(self._n)}?upload_id=SECRETX"
            meta = json.loads(body)
            self.sessions[uri] = {"meta": meta, "buf": bytearray(),
                                  "size": int(headers[
                                      "x-upload-content-length"])}
            return Response(200, {"location": uri}, b"{}")
        if method == "PUT" and url in self.sessions:
            s = self.sessions[url]
            rng = headers.get("content-range", "")
            if rng.startswith("bytes */"):
                return self._state(s)
            if self.fail_puts:
                self.fail_puts -= 1
                return Response(503, {}, b'{"error":"backend"}')
            span, total = rng[len("bytes "):].split("/")
            start = int(span.split("-")[0])
            if start != len(s["buf"]):
                return self._state(s)
            data = body
            if self.ack_short and len(data) > CHUNK_ALIGN:
                self.ack_short -= 1
                data = data[:CHUNK_ALIGN]
            s["buf"] += data
            return self._state(s)
        if method == "GET" and u.path.startswith("/drive/v3/files/"):
            fid = u.path.rsplit("/", 1)[1]
            if fid not in self.files:
                return Response(404, {}, b'{"error":"notFound"}')
            return Response(200, {}, json.dumps(self._meta(fid)).encode())
        if method == "GET" and url.startswith(DRIVE_API + "?"):
            return Response(200, {}, json.dumps({"files": []}).encode())
        if method == "PATCH":
            fid = u.path.rsplit("/", 1)[1]
            self.files[fid]["trashed"] = True
            return Response(200, {}, b"{}")
        return Response(400, {}, b'{"error":"unexpected"}')

    def _state(self, s):
        if len(s["buf"]) >= s["size"]:
            if "file" not in s:
                fid = f"fileid{next(self._n):08d}"
                data = bytes(s["buf"])
                self.files[fid] = {"data": data, "name": s["meta"]["name"],
                                   "trashed": False}
                s["file"] = fid
            return Response(200, {}, json.dumps(self._meta(s["file"])
                                                ).encode())
        if not s["buf"]:
            return Response(308, {}, b"")
        return Response(308, {"range": f"bytes=0-{len(s['buf']) - 1}"}, b"")

    def _meta(self, fid):
        f = self.files[fid]
        md5 = _md5(f["data"]) if not self.corrupt_md5 else "0" * 32
        return {"id": fid, "name": f["name"], "size": str(len(f["data"])),
                "md5Checksum": md5, "mimeType": "video/mp4",
                "trashed": f["trashed"], "parents": ["p"]}

    def stream(self, method, url, headers=None, timeout=None):
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        fid = urllib.parse.urlparse(url).path.rsplit("/", 1)[1]
        data = self.files[fid]["data"]
        start, status = 0, 200
        if "range" in headers and not self.ignore_range:
            start = int(headers["range"].split("=")[1].rstrip("-"))
            status = 206
        chunk = data[start:]
        if self.cut_stream_at is not None:
            chunk = chunk[:self.cut_stream_at]
            self.cut_stream_at = None
        return status, {}, io.BytesIO(chunk)


class Tokens:
    def access_token(self):
        return "ya29.TEST-ACCESS-TOKEN"


# ------------------------------------------------------- http -> test app

class ClientHttp:
    """The production `Http` interface, answered by a FastAPI TestClient,
    so the production worker client talks to the production routes."""

    def __init__(self, client):
        self.client = client

    def request(self, method, url, headers=None, body=None, timeout=None):
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        r = self.client.request(method, path, headers=headers or {},
                                content=body)
        return Response(r.status_code,
                        {k.lower(): v for k, v in r.headers.items()},
                        r.content)
