"""Error classification and redaction.

Two rules hold everywhere in this package:

  * A raw exception from a cloud call never reaches a browser. The browser
    gets a CATEGORY and a fixed sentence written for a person; the detail
    stays in the server log, redacted.
  * Nothing that grants access is ever logged: OAuth access tokens, ID
    tokens, a resumable-upload session URI (whoever holds it can write to
    that upload with no other credential), a private key, an Authorization
    header, an API key in a query string.

`redact` is deliberately greedy. A log line made less readable costs
nothing; a token in a log costs a rotation and an incident.
"""

from __future__ import annotations

import re
import socket
import urllib.error

# Google OAuth access tokens: "ya29." then base64url-ish.
_ACCESS_TOKEN = re.compile(r"ya29\.[A-Za-z0-9_\-.]+")
# Any JWT (ID tokens, signed assertions): three base64url segments, header
# always starts with eyJ ('{"').
_JWT = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*")
# Authorization / proxy headers, whatever scheme.
_AUTH_HEADER = re.compile(r"(?i)(authorization|proxy-authorization)"
                          r"\s*[:=]\s*[^\r\n,;]+")
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-.~+/=]+")
# A resumable session URI carries the capability in `upload_id`.
_UPLOAD_ID = re.compile(r"(?i)(upload_id=)[^&\s\"'<>]+")
# key=..., access_token=..., token=... in a URL.
_QS_SECRET = re.compile(
    r"(?i)([?&](?:key|access_token|token|sig|signature|code)=)"
    r"[^&\s\"'<>]+")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.S)
_PRIVATE_KEY_JSON = re.compile(r'("private_key"\s*:\s*")[^"]*(")')
_MARK = "[REDACTED]"


def redact(text, limit: int = 600) -> str:
    """Remove anything that grants access, then truncate."""
    if text is None:
        return ""
    s = str(text)
    s = _PRIVATE_KEY.sub(_MARK, s)
    s = _PRIVATE_KEY_JSON.sub(r"\1" + _MARK + r"\2", s)
    s = _AUTH_HEADER.sub(lambda m: m.group(1) + ": " + _MARK, s)
    s = _BEARER.sub("Bearer " + _MARK, s)
    s = _ACCESS_TOKEN.sub(_MARK, s)
    s = _JWT.sub(_MARK, s)
    s = _UPLOAD_ID.sub(r"\1" + _MARK, s)
    s = _QS_SECRET.sub(r"\1" + _MARK, s)
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


# ------------------------------------------------------------- categories

# category -> (retryable, what a person is told)
CATEGORIES: dict[str, tuple[bool, str]] = {
    "auth": (False, "The server could not authenticate to Google Cloud. "
                    "An operator needs to check its service account."),
    "permission": (False, "The server is not allowed to do that in Google "
                          "Cloud. An operator needs to check its permissions."),
    "not_found": (False, "A file or resource this job needs was not found."),
    "invalid": (False, "The upload or its settings were not accepted."),
    "quota": (True, "A storage or API quota was reached. The job will be "
                    "retried later."),
    "rate_limited": (True, "A provider is rate-limiting requests. The job "
                           "will be retried after a pause."),
    "transient": (True, "A temporary problem occurred. The job will be "
                        "retried."),
    "timeout": (True, "The job took longer than allowed."),
    "lease_expired": (True, "The worker stopped responding. The job will be "
                            "resumed."),
    "worker_unavailable": (True, "The video worker could not be started. "
                                 "It will be tried again."),
    "pipeline": (False, "The video could not be produced from these clips."),
    "verification": (True, "A transfer could not be verified and will be "
                           "repeated."),
    "cancelled": (False, "The job was cancelled."),
    "daily_cap": (False, "Today's limit of new videos has been reached. "
                         "Please try again tomorrow."),
    "internal": (False, "An unexpected error occurred."),
}


def public_message(category: str) -> str:
    return CATEGORIES.get(category, CATEGORIES["internal"])[1]


def is_retryable(category: str) -> bool:
    return CATEGORIES.get(category, CATEGORIES["internal"])[0]


class CloudError(Exception):
    """A classified failure. `detail` is redacted on construction, so
    holding one can never leak a credential into a log."""

    def __init__(self, category: str, detail: str = "",
                 status: int | None = None):
        self.category = category if category in CATEGORIES else "internal"
        self.detail = redact(detail)
        self.status = status
        super().__init__(f"{self.category}: {self.detail}")

    @property
    def retryable(self) -> bool:
        return is_retryable(self.category)

    def public(self) -> dict:
        return {"category": self.category,
                "message": public_message(self.category)}


def category_for_status(status: int, body: str = "") -> str:
    """HTTP status from a Google API -> category."""
    low = (body or "").lower()
    if status == 401:
        return "auth"
    if status == 403:
        if "ratelimit" in low or "rate limit" in low or "userratelimit" in low:
            return "rate_limited"
        if "quota" in low or "storagequota" in low:
            return "quota"
        return "permission"
    if status == 404:
        return "not_found"
    if status in (400, 411, 413, 416, 422):
        return "invalid"
    if status == 409:
        return "transient"       # e.g. an operation already in progress
    if status == 429:
        return "rate_limited"
    if status in (500, 502, 503, 504):
        return "transient"
    return "internal"


def classify(exc: BaseException) -> CloudError:
    """Any exception -> a CloudError with a category and a redacted detail."""
    if isinstance(exc, CloudError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:                             # noqa: BLE001
            body = ""
        return CloudError(category_for_status(exc.code, body),
                          f"HTTP {exc.code}: {body}", exc.code)
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return CloudError("timeout", f"{type(exc).__name__}: {exc}")
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        return CloudError("transient", f"{type(exc).__name__}: {exc}")
    return CloudError("internal", f"{type(exc).__name__}: {exc}")
