"""Read rows from BigQuery with the standard library only.

WHY NOT THE GOOGLE CLIENT LIBRARY
---------------------------------
The project adds no Python dependency without approval, and a read-only
query needs two REST calls: `jobs.query` to run it and `getQueryResults` to
page through the answer. That is all this file does.

WHO IS ASKING
-------------
`ai_music.access_token` already solves "whose identity": on the VM it is the
machine's own service account (the metadata server); on a workstation it is
whoever is signed in to `gcloud`. Nothing is read from a key file, and the
token is never logged or written anywhere.

Values come back as strings; they are converted by the result's own schema,
so an INTEGER is an int and a TIMESTAMP is epoch seconds - never guessed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://bigquery.googleapis.com/bigquery/v2/projects/{project}"


class BigQueryError(RuntimeError):
    """A query that could not run. The message never carries the token."""


def _convert(value, kind: str):
    if value is None:
        return None
    if kind in ("INTEGER", "INT64"):
        return int(value)
    if kind in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        return float(value)
    if kind in ("BOOLEAN", "BOOL"):
        return value in (True, "true", "True")
    if kind == "TIMESTAMP":
        return float(value)            # BigQuery sends epoch seconds
    return value                       # STRING, DATE, DATETIME stay text


def _rows(payload: dict) -> list[dict]:
    fields = (payload.get("schema") or {}).get("fields") or []
    out = []
    for r in payload.get("rows") or []:
        cells = r.get("f") or []
        out.append({f["name"]: _convert(c.get("v"), f.get("type", "STRING"))
                    for f, c in zip(fields, cells)})
    return out


def _call(url: str, token: str, body: dict | None, http, timeout: float):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    try:
        with http(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read())["error"]["message"]
        except Exception:                                   # noqa: BLE001
            msg = exc.reason
        raise BigQueryError(f"BigQuery answered {exc.code}: {msg}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise BigQueryError(f"BigQuery could not be reached: {exc}") from None


def query(sql: str, project: str, location: str, token: str,
          http=urllib.request.urlopen, timeout: float = 60.0,
          max_wait_s: float = 300.0) -> tuple[list[dict], dict]:
    """Run `sql`, return (rows, info). info: job id, bytes billed, schema.

    Read-only by construction: the caller passes SELECT text and nothing
    here creates a table or a dataset."""
    if not token:
        raise BigQueryError("no Google identity: sign in with gcloud on a "
                            "workstation, or run on the VM")
    base = API.format(project=urllib.parse.quote(project))
    first = _call(f"{base}/queries", token,
                  {"query": sql, "useLegacySql": False, "location": location,
                   "timeoutMs": int(timeout * 1000)}, http, timeout + 15)
    job = (first.get("jobReference") or {}).get("jobId", "")
    payload, started = first, time.time()
    rows: list[dict] = []
    while True:
        if payload.get("jobComplete") is False:
            if time.time() - started > max_wait_s:
                raise BigQueryError(f"query {job} still running after "
                                    f"{max_wait_s:.0f}s")
            payload = _call(
                f"{base}/queries/{job}?location={location}"
                f"&timeoutMs={int(timeout * 1000)}", token, None, http,
                timeout + 15)
            continue
        rows += _rows(payload)
        page = payload.get("pageToken")
        if not page:
            break
        payload = _call(f"{base}/queries/{job}?location={location}"
                        f"&pageToken={urllib.parse.quote(page)}",
                        token, None, http, timeout + 15)
    info = {"job_id": job,
            "bytes_billed": int(first.get("totalBytesBilled") or 0),
            "schema": [f["name"] for f in
                       (first.get("schema") or {}).get("fields") or []]}
    return rows, info


def token(http=urllib.request.urlopen) -> str | None:
    """The VM's service account, else the signed-in gcloud user."""
    from agents.tools.ai_music import access_token
    return access_token(http)


__all__ = ["BigQueryError", "query", "token"]
