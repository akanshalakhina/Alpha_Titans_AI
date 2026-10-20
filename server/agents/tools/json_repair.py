"""Getting a JSON value out of a reply that was supposed to be JSON.

WHY THIS EXISTS
---------------
`vision._json_from` finds the first balanced `{...}` and parses it. That
is right most of the time and has two blind spots, both of which a
smaller model walks into:

  * it counts braces WITHOUT regard for string literals, so a caption
    like `"text": "a { brace"` throws the depth count off and the whole
    object is lost;
  * when the first candidate fails to parse it `break`s - it never tries
    the next `{` in the reply. A model that says "Sure! Here's `{}`:"
    before the real answer therefore produces nothing.

Observed on 14 Oct, after the daily quota pushed the chain onto a smaller
model: `story reply unusable: not a JSON object`. The Story agent fell
back to its deterministic path and the whole narrative was lost to a
formatting slip, not to a failure of reasoning.

`vision.py` is a protected file, so this does not change `_json_from`. It
TRIES it first - unchanged behaviour whenever it works - and only then
applies repairs, each one reported by name so the trace says what had to
be done rather than hiding it.

WHAT IS NOT DONE HERE
---------------------
No guessing at missing fields, no inventing values, no "it probably
meant". A reply that cannot be parsed returns None, and the caller falls
back deterministically and marks the artifact degraded. Repair means
recovering what the model actually wrote, never supplying what it did
not.
"""

from __future__ import annotations

import json
import re
from typing import Any

import vision as _vision

# ```json ... ``` or ``` ... ``` anywhere in the reply, the fenced body
# captured. Non-greedy so several fences do not merge into one.
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)

# `, }` or `, ]` - the single commonest malformation in generated JSON.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _balanced(text: str, start: int) -> str | None:
    """The substring from `start` to its matching close, strings honoured.

    The scanner tracks whether it is inside a string literal and skips the
    character after a backslash, so braces and brackets in content cannot
    unbalance the count. That is the part `_json_from` gets wrong.
    """
    opener = text[start]
    closer = {"{": "}", "[": "]"}[opener]
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _candidates(text: str):
    """Every plausible JSON payload in the reply, best first, each with
    the index it starts at.

    Fenced blocks come first because a model that fences its answer means
    the fence; then every `{` or `[` in document order, so a chatty
    preamble containing an example object does not stop the real one from
    being found.

    The index is what tells an answer apart from a leftover - see
    `_lost_envelope`.
    """
    for m in _FENCE.finditer(text):
        raw = m.group(1)
        body = raw.strip()
        if body:
            yield body, m.start(1) + (len(raw) - len(raw.lstrip()))
    for i, ch in enumerate(text):
        if ch in "{[":
            got = _balanced(text, i)
            if got:
                yield got, i


def _lost_envelope(text: str, start: int) -> bool:
    """True when the value at `start` was only reachable because the
    reply's OUTERMOST value never closed.

    A model whose answer is cut off at the output ceiling leaves the
    envelope open, and the scan above then walks past it and returns the
    last complete thing inside it - a single segment, one beat, one
    caption. That object is valid JSON and is not the answer, and
    nothing downstream can tell the difference by looking at it.

    Position can. The real answer starts at the reply's first JSON
    character; what is left after a cut starts later. Only the FIRST
    opener is examined: a `{` that appears inside a string literal
    earlier in the reply would never balance either, and treating that
    as loss would buy a second call for a reply that was always whole.
    """
    for i, ch in enumerate(text):
        if ch in "{[":
            return i < start and _balanced(text, i) is None
    return False


def parse(text: str) -> tuple[Any | None, str]:
    """Return `(value, how)`. `how` names the route, for the trace.

    "clean" means the reply needed nothing doing to it. Anything else is
    a repair that was necessary, and worth seeing in the trace: a model
    that always needs its commas fixed is a prompt problem, and nobody
    can notice that if the repair is silent.
    """
    if not text or not text.strip():
        return None, "empty"

    # 1. fenced blocks, then every balanced value in document order.
    #
    #    This runs BEFORE `_json_from` on purpose. On a reply that holds a
    #    broken object followed by a good one, `_json_from` fails on the
    #    first, abandons the object branch entirely and falls through to
    #    its array branch - which then returns the list NESTED INSIDE the
    #    answer instead of the answer. Every agent reads its reply as a
    #    mapping, so that surfaces as a `TypeError` or a silently empty
    #    result rather than an honest parse failure.
    #
    #    Scanning in document order with string literals honoured returns
    #    the outermost value at the earliest position that parses, which
    #    is the same answer on every well-formed reply and the right one
    #    on the malformed ones.
    first_clean = None
    for cand, at in _candidates(text):
        try:
            value = json.loads(cand)
        except json.JSONDecodeError:
            repaired = _TRAILING_COMMA.sub(r"\1", cand)
            if repaired == cand:
                continue
            try:
                return json.loads(repaired), "trailing-comma"
            except json.JSONDecodeError:
                continue
        how = ("clean" if cand.strip() == text.strip()
               else "fragment" if _lost_envelope(text, at)
               else "extracted")
        # a mapping is what every caller wants; keep a bare list in case
        # nothing better turns up rather than returning it over an object
        if isinstance(value, dict):
            return value, how
        if first_clean is None:
            first_clean = (value, how)

    # 2. what every other caller already does, as a safety net
    got = _vision._json_from(text)
    if isinstance(got, dict):
        return got, "extracted"
    if first_clean is not None:
        return first_clean
    if isinstance(got, list):
        return got, "extracted"

    # 3. the whole reply, commas repaired - a bare object with no prose
    #    around it that only a trailing comma spoiled
    repaired = _TRAILING_COMMA.sub(r"\1", text.strip())
    try:
        value = json.loads(repaired)
    except json.JSONDecodeError:
        return None, "unparseable"
    return (value, "trailing-comma") if isinstance(value, (dict, list)) \
        else (None, "not-an-object")


def complain(text: str, want: str = "a JSON object") -> str:
    """One line a model can act on, for the retry prompt.

    Quotes what actually came back, trimmed, because "your reply was not
    JSON" tells a model nothing it can correct.
    """
    head = " ".join((text or "").split())[:180]
    return (f"Your previous reply could not be parsed as {want}. "
            f"It began: {head!r}. "
            f"Reply with {want} and nothing else - no prose, no markdown "
            f"fences, no trailing commas.")


def with_complaint(prompt: str, text: str, want: str = "a JSON object") -> str:
    """The original prompt, with the complaint appended.

    The prompt is repeated in full rather than replaced: the model needs
    the task as much as the correction, and a bare "that was not JSON"
    with no question attached produces an apology, not an answer.
    """
    return prompt + "\n\n" + complain(text, want)
