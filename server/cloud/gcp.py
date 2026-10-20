"""Google Cloud over plain HTTPS, stdlib only.

No client library: `google-api-python-client`, `google-auth` and friends
would be new dependencies, and nothing here needs them. These are a handful
of documented REST calls.

IDENTITY
--------
`MetadataTokens` asks the GCE metadata server for the attached service
account's token. That token is short-lived, nothing is exported and there is
no key file. It is the only token source meant for production.
`GcloudTokens` shells out to an installed gcloud for development on an
operator's machine; user credentials expire and need a human to renew them,
so production never uses it.

FIXED OPERATIONS
----------------
`Compute` can do exactly three things - get, start, stop - to exactly one
instance, named at construction from validated configuration. Nothing a
request carries reaches its URLs.

DRIVE
-----
Resumable uploads in 256 KiB-aligned chunks, honouring the `Range` the server
acknowledges rather than assuming every byte sent arrived. Downloads resume
from a `.part` file and are verified against Drive's own size and md5 before
they are renamed into place. A transfer that cannot be verified is never
treated as done.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import _INSTANCE, _PROJECT, _ZONE
from .sanitize import CloudError, category_for_status, classify, redact

# ------------------------------------------------------------------ http


@dataclass
class Response:
    status: int
    headers: dict = field(default_factory=dict)      # lower-cased keys
    body: bytes = b""

    def json(self) -> dict:
        try:
            return json.loads(self.body or b"{}")
        except ValueError:
            return {}

    def raise_for_status(self, what: str) -> "Response":
        if self.status >= 400:
            text = self.body[:2000].decode("utf-8", "replace")
            raise CloudError(category_for_status(self.status, text),
                             f"{what}: HTTP {self.status}: {text}",
                             self.status)
        return self


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Drive answers an incomplete resumable upload with `308 Resume
    Incomplete`, which is not a redirect. Following it - urllib does, for
    308, since 3.11 - would lose the `Range` header that says how much
    arrived. Nothing here needs a redirect followed, so none ever is."""

    def redirect_request(self, *args, **kwargs):
        return None


class Http:
    """One place every outbound call goes through: a timeout always, no
    redirects, errors turned into `Response`s rather than raised."""

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method: str, url: str, headers: dict | None = None,
                body: bytes | None = None, timeout: float | None = None
                ) -> Response:
        req = urllib.request.Request(url, data=body, method=method,
                                     headers=headers or {})
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as r:
                return Response(r.status, _lower(r.headers), r.read())
        except urllib.error.HTTPError as exc:
            try:
                data = exc.read()
            except Exception:                         # noqa: BLE001
                data = b""
            return Response(exc.code, _lower(exc.headers or {}), data)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise classify(exc) from None

    def stream(self, method: str, url: str, headers: dict | None = None,
               timeout: float | None = None):
        """An open response for large bodies. The caller must close it.
        Returns (status, headers, fileobj)."""
        req = urllib.request.Request(url, method=method,
                                     headers=headers or {})
        try:
            r = self._opener.open(req, timeout=timeout or self.timeout)
            return r.status, _lower(r.headers), r
        except urllib.error.HTTPError as exc:
            data = b""
            try:
                data = exc.read()
            except Exception:                         # noqa: BLE001
                pass
            raise CloudError(category_for_status(exc.code, data.decode(
                "utf-8", "replace")), f"HTTP {exc.code}: "
                f"{data[:500].decode('utf-8', 'replace')}", exc.code
            ) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise classify(exc) from None


def _lower(headers) -> dict:
    try:
        items = headers.items()
    except AttributeError:                            # pragma: no cover
        items = []
    return {str(k).lower(): v for k, v in items}


# ---------------------------------------------------------------- tokens

METADATA = "http://metadata.google.internal/computeMetadata/v1"
_SA = METADATA + "/instance/service-accounts/default"


def _jwt_exp(token: str) -> float:
    """The `exp` claim of a JWT, without verifying it - used only to decide
    when OUR OWN freshly minted token needs replacing."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        import base64
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:                                 # noqa: BLE001
        return 0.0


class MetadataTokens:
    """Short-lived credentials for the VM's attached service account."""

    def __init__(self, http: Http | None = None, clock=time.time):
        self.http = http or Http(timeout=10.0)
        self.clock = clock
        self._access: tuple[str, float] | None = None
        self._ids: dict[str, tuple[str, float]] = {}

    def _get(self, url: str) -> Response:
        r = self.http.request("GET", url, {"Metadata-Flavor": "Google"})
        return r.raise_for_status("metadata server")

    def access_token(self) -> str:
        now = self.clock()
        if self._access and self._access[1] - 60 > now:
            return self._access[0]
        d = self._get(_SA + "/token").json()
        tok = d.get("access_token")
        if not tok:
            raise CloudError("auth", "metadata server returned no token")
        self._access = (tok, now + float(d.get("expires_in", 300)))
        return tok

    def id_token(self, audience: str) -> str:
        """A Google-signed ID token naming this VM's service account,
        minted for `audience` (the control plane's URL)."""
        now = self.clock()
        hit = self._ids.get(audience)
        if hit and hit[1] - 60 > now:
            return hit[0]
        q = urllib.parse.urlencode({"audience": audience, "format": "full"})
        r = self._get(_SA + "/identity?" + q)
        tok = r.body.decode("ascii", "replace").strip()
        if tok.count(".") != 2:
            raise CloudError("auth", "metadata server returned no ID token")
        self._ids[audience] = (tok, _jwt_exp(tok) or now + 300)
        return tok

    def email(self) -> str:
        r = self._get(_SA + "/email")
        return r.body.decode("ascii", "replace").strip()


class GcloudTokens:
    """Development only: an operator's installed gcloud. Never production -
    those credentials expire and need a person to renew them."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self._access: tuple[str, float] | None = None
        exe = shutil.which("gcloud")
        self.exe = exe

    def _run(self, *args: str) -> str:
        if not self.exe:
            raise CloudError("auth", "gcloud is not installed")
        try:
            out = subprocess.run([self.exe, *args], capture_output=True,
                                 text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise classify(exc) from None
        if out.returncode != 0:
            # stderr can carry an account name, never a token; still redact
            raise CloudError("auth", "gcloud: " + redact(out.stderr, 300))
        return out.stdout.strip()

    def access_token(self) -> str:
        now = self.clock()
        if self._access and self._access[1] > now:
            return self._access[0]
        tok = self._run("auth", "print-access-token")
        self._access = (tok, now + 600)
        return tok

    def id_token(self, audience: str) -> str:
        return self._run("auth", "print-identity-token",
                         f"--audiences={audience}")

    def email(self) -> str:
        return self._run("config", "get-value", "account")


def token_source(kind: str):
    if kind == "metadata":
        return MetadataTokens()
    if kind == "gcloud":
        return GcloudTokens()
    raise CloudError("invalid", f"unknown token source {kind!r}")


# --------------------------------------------------------------- compute

class Compute:
    """get / start / stop of ONE instance. The names are validated here
    as well as in config, so even a caller bypassing config cannot put
    anything but a plain resource name into the URL."""

    API = "https://compute.googleapis.com/compute/v1"

    def __init__(self, http: Http, tokens, project: str, zone: str,
                 instance: str):
        for label, val, rx in (("project", project, _PROJECT),
                               ("zone", zone, _ZONE),
                               ("instance", instance, _INSTANCE)):
            if not rx.match(val or ""):
                raise CloudError("invalid", f"bad {label} name")
        self.http, self.tokens = http, tokens
        self._base = (f"{self.API}/projects/{project}/zones/{zone}"
                      f"/instances/{instance}")

    def _call(self, method: str, suffix: str = "") -> dict:
        r = self.http.request(method, self._base + suffix, {
            "Authorization": f"Bearer {self.tokens.access_token()}",
            "Content-Type": "application/json"},
            b"" if method == "POST" else None, timeout=30)
        return r.raise_for_status(f"compute {method} {suffix or 'get'}"
                                  ).json()

    def status(self) -> str:
        """PROVISIONING, STAGING, RUNNING, STOPPING, STOPPED, SUSPENDING,
        SUSPENDED, REPAIRING or TERMINATED - as Compute Engine reports it."""
        return str(self._call("GET").get("status", "UNKNOWN"))

    def start(self) -> dict:
        return self._call("POST", "/start")

    def stop(self) -> dict:
        return self._call("POST", "/stop")


# ----------------------------------------------------------------- drive

DRIVE_API = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"

# Every chunk but the last must be a multiple of 256 KiB, or the upload
# server rejects it. 8 MiB = 32 of them: big enough that a 450 MB clip is ~57
# round trips, small enough that a retried chunk costs seconds.
CHUNK_ALIGN = 256 * 1024
CHUNK = 32 * CHUNK_ALIGN
_RANGE = re.compile(r"bytes=0-(\d+)")
FILE_FIELDS = "id,name,mimeType,size,md5Checksum,parents,trashed"


def drive_link(file_id: str) -> str:
    """A view link. Not a capability: opening it still needs Drive access
    to the file, which this project never grants publicly."""
    return f"https://drive.google.com/file/d/{file_id}/view"


def _q_escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def md5_of(path: Path, upto: int | None = None) -> str:
    h = hashlib.md5()                                  # noqa: S324 - integrity
    with open(path, "rb") as f:
        left = upto
        while True:
            n = CHUNK if left is None else min(CHUNK, left)
            if n == 0:
                break
            block = f.read(n)
            if not block:
                break
            h.update(block)
            if left is not None:
                left -= len(block)
    return h.hexdigest()


class Drive:
    def __init__(self, http: Http, tokens, shared_drive: bool = True,
                 sleep=time.sleep):
        self.http, self.tokens = http, tokens
        self.shared = shared_drive
        self.sleep = sleep

    # ---------------------------------------------------------- helpers

    def _auth(self, extra: dict | None = None) -> dict:
        h = {"Authorization": f"Bearer {self.tokens.access_token()}"}
        h.update(extra or {})
        return h

    def _qs(self, **params) -> str:
        if self.shared:
            params["supportsAllDrives"] = "true"
        return urllib.parse.urlencode(params)

    # --------------------------------------------------------- metadata

    def get(self, file_id: str) -> dict:
        if not re.match(r"^[A-Za-z0-9_-]{10,200}$", file_id or ""):
            raise CloudError("invalid", "malformed Drive file id")
        r = self.http.request("GET", f"{DRIVE_API}/{file_id}?"
                              + self._qs(fields=FILE_FIELDS), self._auth())
        return r.raise_for_status("drive get").json()

    def find(self, parent: str, name: str) -> list[dict]:
        q = (f"'{_q_escape(parent)}' in parents and name = "
             f"'{_q_escape(name)}' and trashed = false")
        extra = {"includeItemsFromAllDrives": "true"} if self.shared else {}
        r = self.http.request("GET", DRIVE_API + "?" + self._qs(
            q=q, fields=f"files({FILE_FIELDS})", pageSize="10", **extra),
            self._auth())
        return r.raise_for_status("drive find").json().get("files", [])

    def create_folder(self, name: str, parent: str) -> str:
        """Find-or-create, so a retried call cannot make two folders."""
        found = [f for f in self.find(parent, name)
                 if f.get("mimeType") == FOLDER_MIME]
        if found:
            return found[0]["id"]
        body = json.dumps({"name": name, "mimeType": FOLDER_MIME,
                           "parents": [parent]}).encode()
        r = self.http.request("POST", DRIVE_API + "?" + self._qs(
            fields="id"), self._auth({"Content-Type": "application/json"}),
            body)
        return r.raise_for_status("drive create folder").json()["id"]

    def trash(self, file_id: str) -> None:
        """Trash, never delete: a mistaken call stays recoverable."""
        r = self.http.request("PATCH", f"{DRIVE_API}/{file_id}?"
                              + self._qs(fields="id"),
                              self._auth({"Content-Type":
                                          "application/json"}),
                              json.dumps({"trashed": True}).encode())
        r.raise_for_status("drive trash")

    # ------------------------------------------------ resumable upload

    def start_session(self, name: str, mime: str, size: int, parent: str,
                      origin: str | None = None) -> str:
        """Open a resumable upload and return its session URI.

        THE URI IS A BEARER CAPABILITY: whoever holds it can write this
        upload with no other credential. Never log it, never return it to a
        browser unless the direct mode was deliberately enabled."""
        body = json.dumps({"name": name, "parents": [parent],
                           "mimeType": mime}).encode()
        headers = self._auth({"Content-Type": "application/json; "
                                              "charset=UTF-8",
                              "X-Upload-Content-Type": mime,
                              "X-Upload-Content-Length": str(int(size))})
        if origin:
            headers["Origin"] = origin
        r = self.http.request("POST", DRIVE_UPLOAD + "?" + self._qs(
            uploadType="resumable", fields=FILE_FIELDS), headers, body)
        r.raise_for_status("drive start upload")
        uri = r.headers.get("location")
        if not uri or not uri.startswith("https://"):
            raise CloudError("internal", "drive returned no session")
        return uri

    def _parse(self, r: Response) -> tuple[bool, int, dict | None]:
        if r.status in (200, 201):
            return True, -1, r.json()
        if r.status == 308:
            m = _RANGE.match(r.headers.get("range", ""))
            return False, (int(m.group(1)) + 1) if m else 0, None
        if r.status in (404, 410):
            raise CloudError("verification", "upload session expired",
                             r.status)
        r.raise_for_status("drive chunk")
        raise CloudError("internal", f"unexpected HTTP {r.status}")

    def put_chunk(self, session_uri: str, data: bytes, offset: int,
                  total: int) -> tuple[bool, int, dict | None]:
        """Send bytes [offset, offset+len). Returns (done, next_offset,
        file_metadata_when_done). `next_offset` is what the SERVER says it
        holds, which may be less than was sent."""
        end = offset + len(data) - 1
        if len(data) == 0 or end >= total:
            raise CloudError("invalid", "chunk outside the declared size")
        if end + 1 < total and len(data) % CHUNK_ALIGN:
            raise CloudError("invalid", "non-final chunk not 256 KiB "
                                        "aligned")
        r = self.http.request("PUT", session_uri, {
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes {offset}-{end}/{total}"}, data,
            timeout=120)
        return self._parse(r)

    def query(self, session_uri: str, total: int
              ) -> tuple[bool, int, dict | None]:
        """How much of this upload the server holds."""
        r = self.http.request("PUT", session_uri, {
            "Content-Length": "0", "Content-Range": f"bytes */{total}"},
            b"", timeout=30)
        return self._parse(r)

    def upload(self, path: Path, name: str, mime: str, parent: str,
               on_progress=None, retries: int = 5) -> dict:
        """Upload a local file and VERIFY it: the file Drive reports must
        have this file's size and md5, or it is trashed and this raises."""
        path = Path(path)
        size = path.stat().st_size
        if size == 0:
            raise CloudError("invalid", f"{name} is empty")
        local_md5 = md5_of(path)
        uri = self.start_session(name, mime, size, parent)
        offset, failures, meta = 0, 0, None
        with open(path, "rb") as f:
            while meta is None:
                f.seek(offset)
                data = f.read(CHUNK)
                try:
                    done, nxt, meta = self.put_chunk(uri, data, offset, size)
                except CloudError as exc:
                    if not exc.retryable or failures >= retries:
                        raise
                    failures += 1
                    self.sleep(min(30.0, 2.0 ** failures))
                    done, nxt, meta = self.query(uri, size)
                if not done:
                    offset = nxt
                    if on_progress:
                        on_progress(offset, size)
        if on_progress:
            on_progress(size, size)
        self._verify(meta, size, local_md5, name)
        return meta

    def _verify(self, meta: dict, size: int, md5: str, name: str) -> None:
        got_size = int(meta.get("size") or -1)
        got_md5 = (meta.get("md5Checksum") or "").lower()
        if got_size != size or (got_md5 and got_md5 != md5):
            try:
                self.trash(meta["id"])
            except Exception:                         # noqa: BLE001
                pass
            raise CloudError("verification",
                             f"{name}: Drive holds {got_size} bytes / md5 "
                             f"{got_md5 or '?'}, expected {size} / {md5}")
        if not got_md5:
            raise CloudError("verification",
                             f"{name}: Drive reported no md5 to verify")

    # ----------------------------------------------------------- download

    def _media_url(self, file_id: str) -> str:
        return f"{DRIVE_API}/{file_id}?" + self._qs(alt="media")

    def link(self, file_id: str) -> str | None:
        """A link a person can open, or None when the bytes must be
        streamed through the control plane instead."""
        return drive_link(file_id)

    def download(self, file_id: str, dest: Path, size: int,
                 md5: str | None, on_progress=None,
                 retries: int = 5) -> Path:
        """Fetch a file to `dest`, resuming a `.part` left by an earlier
        attempt, and verify size and md5 before moving it into place. An
        existing, verified `dest` is not fetched again."""
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        if dest.exists() and dest.stat().st_size == size and (
                not md5 or md5_of(dest) == md5.lower()):
            return dest
        failures = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            if have == size:
                break
            headers = self._auth()
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                status, _h, body = self.http.stream(
                    "GET", self._media_url(file_id), headers, timeout=120)
                try:
                    if have and status != 206:
                        have = 0               # range ignored: start over
                    with open(part, "ab" if have else "wb") as out:
                        while True:
                            block = body.read(1024 * 1024)
                            if not block:
                                break
                            out.write(block)
                            if on_progress:
                                on_progress(out.tell(), size)
                finally:
                    body.close()
            except CloudError as exc:
                if not exc.retryable or failures >= retries:
                    raise
                failures += 1
                self.sleep(min(30.0, 2.0 ** failures))
                continue
            if part.stat().st_size >= size:
                break
            failures += 1
            if failures > retries:
                raise CloudError("verification", f"{dest.name}: download "
                                 f"stopped at {part.stat().st_size} of "
                                 f"{size} bytes")
        got = part.stat().st_size
        if got != size or (md5 and md5_of(part) != md5.lower()):
            part.unlink(missing_ok=True)
            raise CloudError("verification", f"{dest.name}: downloaded "
                             f"{got} bytes did not match Drive's record")
        os.replace(part, dest)
        return dest
