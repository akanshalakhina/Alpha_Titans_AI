"""Google Cloud Storage as the file store, behind the Drive client's interface.

WHY THIS EXISTS
---------------
Drive was the brief's storage, and it needs a Shared Drive: a service account
owns no Drive storage, so its uploads into an ordinary My Drive folder fail
with `storageQuotaExceeded`. Shared Drives exist only on Google Workspace.
On a personal Google account there is no such thing, so outputs could never
be written. Cloud Storage has no such limit: a private bucket, with access
granted to each service account by IAM, and nothing ever public.

SAME PROTOCOL, DIFFERENT ADDRESSES
----------------------------------
GCS's JSON API resumable upload is Drive's protocol: a POST opens a session,
chunks go up with `Content-Range`, the server answers `308` with the
`Range` it holds, then `200` with the object. So this SUBCLASSES the Drive
client and changes only URLs and the shape of the metadata it returns. The
chunking, retries and verification are the same code, already tested.

Names: a "folder" is an object-name prefix and a "file id" is the object's
full name (`outputs/<job>/<file>`). GCS reports md5 as base64; it is turned
into hex here so every comparison stays the same as for Drive.

There is no public link. Outputs are streamed to an authenticated browser
by the control plane, so the bucket never needs to be readable by anyone.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import urllib.parse

from .gcp import Drive
from .sanitize import CloudError

GCS_API = "https://storage.googleapis.com/storage/v1/b"
GCS_UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b"

_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
# object names we create are `prefix/job_NN_name.ext`; refuse anything else
_OBJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/ -]{0,900}$")


def check_bucket(name: str) -> str:
    if not _BUCKET.match(name or ""):
        raise CloudError("invalid", "bad bucket name")
    return name


def check_object(name: str) -> str:
    if (not _OBJECT.match(name or "") or ".." in name or "//" in name
            or name.endswith("/")):
        raise CloudError("invalid", "bad object name")
    return name


def _hex_md5(b64: str | None) -> str:
    if not b64:
        return ""
    try:
        return base64.b64decode(b64).hex()
    except (binascii.Error, ValueError):
        return ""


def normalise(obj: dict) -> dict:
    """A GCS object resource, shaped like the Drive metadata the rest of
    the code reads: id, name, size, md5Checksum (hex), mimeType, parents."""
    name = obj.get("name", "")
    prefix, _, base = name.rpartition("/")
    return {"id": name, "name": base or name,
            "size": str(obj.get("size", "")),
            "md5Checksum": _hex_md5(obj.get("md5Hash")),
            "mimeType": obj.get("contentType", ""),
            "parents": [prefix] if prefix else [], "trashed": False}


class GcsStore(Drive):
    def __init__(self, http, tokens, bucket: str, sleep=None):
        kw = {"sleep": sleep} if sleep else {}
        super().__init__(http, tokens, shared_drive=False, **kw)
        self.bucket = check_bucket(bucket)

    def _obj_url(self, name: str) -> str:
        return (f"{GCS_API}/{self.bucket}/o/"
                + urllib.parse.quote(check_object(name), safe=""))

    def _media_url(self, file_id: str) -> str:
        return self._obj_url(file_id) + "?alt=media"

    def link(self, file_id: str) -> str | None:
        return None                    # streamed by the control plane

    # ---------------------------------------------------------- metadata

    def get(self, file_id: str) -> dict:
        r = self.http.request("GET", self._obj_url(file_id), self._auth())
        return normalise(r.raise_for_status("storage get").json())

    def find(self, parent: str, name: str) -> list[dict]:
        try:
            return [self.get(f"{parent}/{name}")]
        except CloudError as exc:
            if exc.category == "not_found":
                return []
            raise

    def create_folder(self, name: str, parent: str) -> str:
        """A prefix, not a thing: nothing to create, nothing to duplicate."""
        return check_object(f"{parent}/{name}")

    def trash(self, file_id: str) -> None:
        # No trash in GCS, and deleting is not this code's business: a
        # re-upload under the same name replaces the object anyway.
        return None

    # ------------------------------------------------- resumable upload

    def start_session(self, name: str, mime: str, size: int, parent: str,
                      origin: str | None = None) -> str:
        """Same contract as Drive's: returns a session URI that is a bearer
        capability - never logged, never sent to a browser."""
        obj = check_object(f"{parent}/{name}")
        headers = self._auth({"Content-Type": "application/json; "
                                              "charset=UTF-8",
                              "X-Upload-Content-Type": mime,
                              "X-Upload-Content-Length": str(int(size))})
        if origin:
            headers["Origin"] = origin
        url = (f"{GCS_UPLOAD}/{self.bucket}/o?"
               + urllib.parse.urlencode({"uploadType": "resumable",
                                         "name": obj}))
        r = self.http.request("POST", url, headers,
                              json.dumps({"contentType": mime}).encode())
        r.raise_for_status("storage start upload")
        uri = r.headers.get("location")
        if not uri or not uri.startswith("https://"):
            raise CloudError("internal", "storage returned no session")
        return uri

    def _parse(self, r):
        done, nxt, meta = super()._parse(r)
        return done, nxt, (normalise(meta) if done and meta else meta)

    # ------------------------------------------------------ streaming out

    def open_media(self, file_id: str, rng: str | None = None):
        """(status, headers, stream) for passing an object on to a browser.
        `rng` is the browser's own Range header, forwarded so a video
        player can seek; storage answers 206 with the Content-Range."""
        h = self._auth()
        if rng:
            h["Range"] = rng
        return self.http.stream("GET", self._media_url(file_id), h,
                                timeout=120)
