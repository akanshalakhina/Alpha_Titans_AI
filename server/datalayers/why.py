"""Why Writer: the lessons that survived, said in a few plain sentences.

The Pattern Miner can produce many small findings; a person reading them
wants the few that matter and what to try next. This agent writes that
summary - and is held to the lessons it was given:

* it sees ONLY lessons that held with each group's best video removed;
* every sentence must cite the id of the lesson it rests on, and a sentence
  that cites nothing, or an id it was not given, is dropped;
* with no model, or nothing usable back, the summary is the strongest
  lessons' own wording - never an invented reason.

It explains; it decides nothing. Lessons still need a person's approval
before any video uses them.
"""

from __future__ import annotations

import re
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS why_summary (
  created_at REAL, written_by TEXT, text TEXT, lesson_ids TEXT);
"""

PROMPT = """You are summarising what this YouTube channel's own numbers \
suggest about why some of its LEGO build videos do better than others.

Use ONLY these findings. Each held when the best video of each group was \
removed. Their labels say how many videos stand behind them.

{lessons}

Write at most 4 short sentences for a busy manager, most useful first. \
Plain English, no jargon. Each sentence must end with the id of the \
finding it rests on, in square brackets, like [{example}]. Say "early \
signal" when a finding's label says so. Do not add any reason, number or \
fact that is not in the findings. Name each measure exactly as the finding \
does (first-week views, % of the video watched) - never a broader word like \
"engagement" or "performance" - and do not repeat one finding in several \
sentences. Then one sentence starting "Try next:" \
suggesting one change to test on the next videos, also ending with the id \
it rests on.

Answer with JSON only: {{"sentences": ["...", "..."]}}"""

_ORDER = {"strong": 0, "medium": 1, "early signal": 2}


def write(db, llm=None) -> list[dict]:
    """Why Writer's step. Never raises; always leaves a summary row."""
    from datalayers import lessons as LS
    db.executescript(SCHEMA)
    usable = [l for l in LS.listing(db) if l["stable"]
              and l["status"] != "rejected"]
    usable.sort(key=lambda l: (_ORDER.get(l["confidence"], 9),
                               -(l["lift"] or 0)))
    usable = usable[:12]
    ids = {l["id"] for l in usable}
    text, by, cited = "", "rules", []
    if not usable:
        text = ("No pattern has held up yet. More videos, and the published "
                "video files, are needed before the numbers can say why.")
    else:
        if llm is not None and llm.available():
            listing = "\n".join(f"[{l['id']}] ({l['confidence']}) {l['text']}"
                                for l in usable)
            try:
                raw, reply = llm.complete_json(
                    PROMPT.format(lessons=listing, example=usable[0]["id"]),
                    max_tokens=700)
                sents = raw.get("sentences") if isinstance(raw, dict) else None
                keep = []
                for snt in sents or []:
                    snt = str(snt).strip()
                    refs = re.findall(r"\[([0-9a-f]{10})\]", snt)
                    if refs and all(r in ids for r in refs):
                        keep.append(snt)
                        cited += refs
                if keep:
                    text = "\n".join(keep[:5])
                    by = ((reply.meta or {}).get("model") if reply else "") \
                        or "llm"
            except Exception:                               # noqa: BLE001
                pass
        if not text:
            top = usable[:3]
            text = "\n".join(f"{l['text']} [{l['id']}]" for l in top)
            cited = [l["id"] for l in top]
    db.execute("INSERT INTO why_summary VALUES (?,?,?,?)",
               (time.time(), by, text, ",".join(dict.fromkeys(cited))))
    db.commit()
    return [{"name": "summary_written", "ok": True,
             "detail": f"{len(text.splitlines())} sentence(s) from "
                       f"{len(usable)} lesson(s) that held up, written by {by}"},
            {"name": "only_cited_findings", "ok": True,
             "detail": "every sentence names the lesson it rests on; any that "
                       "did not were dropped"}]


def latest(db) -> dict | None:
    db.executescript(SCHEMA)
    r = db.execute("SELECT * FROM why_summary ORDER BY created_at DESC "
                   "LIMIT 1").fetchone()
    return dict(r) if r else None


__all__ = ["latest", "write"]
