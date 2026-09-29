"""
The edit-planning layer: turns measurements into an explicit, inspectable plan.

WHAT THIS IS, PLAINLY
---------------------
This is a rule-based director, NOT a language model and NOT an LLM agent. It
cannot see your footage or understand that a gearbox is a gearbox. Every
decision below is derived from numbers the pipeline actually measured -
per-frame motion, freeze coverage, orientation, duration - and every one
records the reason it chose what it chose, so the plan can be argued with.

WHY IT EXISTS
-------------
Before this module the pipeline decided four things (which clips survive, their
order, where each is trimmed, how long each runs) and simply passed the rest
through from the UI: one transition style cycled blindly over every cut, a
music mood picked from a dropdown, a fixed crossfade length. Those are dials,
not decisions. This module makes them decisions, and writes the whole lot -
including the four the planner already made - into one `edit_plan.json`.

WHAT IT DELIBERATELY DOES NOT DECIDE
------------------------------------
* Captions. It has no scene understanding, so it has no basis to describe your
  build and will not invent one. Captions come from your script.
* Speed. A defensible rule would need to know whether anyone is talking; a
  speed-up that ruins a demonstration's audio is worse than leaving it alone.

Runs after `plan` (so it can see the chosen segments) and before `cut`/`join`
(so its choices actually drive the render).
"""

from __future__ import annotations

import json
from pathlib import Path

import vision as _vision
from adapter import build_dir, read_manifest

# ---------------------------------------------------------------- transitions
# Chosen per cut from the PAIR of shots being joined, rather than one list
# cycled blindly. The names are real ffmpeg xfade transitions.
#
#   busy  -> busy   a white flash reads as a beat cut and keeps energy up
#   busy  -> calm   a plain dissolve lets the eye settle into the quiet shot
#   calm  -> busy   a directional wipe lifts you into the action
#   calm  -> calm   a gentle dissolve; anything harder would jar
#
# Orientation change is handled first and overrides all of the above: a wipe
# travelling across a blur-padded letterbox edge looks like a glitch, so those
# cuts always dissolve.
PAIR_XFADE = {
    ("busy", "busy"): ("fadewhite", "both shots are busy - a flash keeps the energy up"),
    ("busy", "calm"): ("fade", "the motion drops here, so the cut settles rather than snaps"),
    ("calm", "busy"): ("wipeup", "motion lifts here, so the cut lifts with it"),
    ("calm", "calm"): ("fade", "both shots are calm - a gentle dissolve suits them"),
}
ORIENT_XFADE = ("fade", "the two shots are different orientations, so a dissolve "
                        "avoids wiping across the blur-padded edge")

# When the vision pass ran, what a shot IS beats how much it moves. Cutting
# into a reveal or a close-up wants to settle, not snap; leaving assembly for a
# demonstration is a change of gear and can take a harder cut.
KIND_XFADE = {
    ("assembly", "reveal"):        ("fade", "assembly gives way to the reveal, so the cut settles into it"),
    ("assembly", "demonstration"): ("wipeup", "assembly hands over to the demonstration - a wipe marks the change of gear"),
    ("demonstration", "reveal"):   ("fade", "the demonstration resolves into the reveal, so the cut is gentle"),
    ("detail", "reveal"):          ("fade", "a detail shot opening onto the reveal wants a soft dissolve"),
    ("assembly", "assembly"):      ("fadewhite", "two assembly shots in a row - a flash keeps the build moving"),
}
REVEAL_IN = ("fade", "the next shot is the reveal, so the cut settles into it "
                     "rather than snapping")

# Mood from the batch's overall energy. Thresholds are on mean YDIF, the same
# figure the planner budgets with.
MOOD_BANDS = [
    (2.5, "beat", "busy footage overall (median motion %.2f) - a full kit with "
                  "impacts on the cuts"),
    (0.8, "bright", "moderate motion (median %.2f) - a brighter bed without a "
                    "heavy kit"),
    (0.0, "lofi", "calm footage (median %.2f) - a soft bed that stays out of "
                  "the way"),
]


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def direct(job_id: str, settings: dict, on_line) -> dict:
    """Produce the edit plan. Returns it, and writes edit_plan.json."""
    m = read_manifest(job_id)
    if not m:
        raise RuntimeError("no manifest - the planner has not run")
    segs = m.get("segments") or []
    if not segs:
        raise RuntimeError("the planner selected no segments")

    clips = {c["name"]: c for c in m.get("clips") or []}
    motions = [float(s.get("mean_ydif") or 0.0) for s in segs]
    med = _median(motions)
    # "busy" is relative to this batch, for the same reason the rejection rule
    # is: absolute motion figures mean different things for different subjects.
    busy_at = max(med, 0.05)

    def band(s):
        return "busy" if float(s.get("mean_ydif") or 0) >= busy_at else "calm"

    decisions = []

    # ---- 0: what, if anything, the vision pass contributed ------------------
    vis = _vision.read(job_id) or {}
    vclips = vis.get("clips") or {}
    vcross = vis.get("cross") or {}
    if vclips:
        seen = []
        for sg in segs:
            v = vclips.get(sg["clip"])
            if v:
                seen.append("%s: %s (%s, %s shot, conf %d)"
                            % (sg["clip"].split("_", 1)[-1], v["subject"],
                               v["kind"], v["shot"], v["confidence"]))
        decisions.append({
            "agent": "vision",
            "what": "What the footage is",
            "chose": "%s / %s" % (vis.get("provider", "?"), vis.get("model", "?")),
            "why": ("a vision model looked at %d real frames from each clip:\n  "
                    % vis.get("frames_per_clip", 0)) + "\n  ".join(seen)
                   + (("\n  story: " + vcross["story"]) if vcross.get("story") else ""),
            "decided_by": "vision",
        })

    # ---- 1-4: record what the planner already decided, with its reasoning ---
    kept, dropped = len(segs), len(m.get("rejected") or [])
    decisions.append({
        "agent": "editor",
        "what": "Which clips to use",
        "chose": f"{kept} of {kept + dropped}",
        "why": ("every clip that has movement in it was kept; "
                + ("; ".join(f"{r['clip']} left out - {r['why']}"
                             for r in (m.get("rejected") or []))
                   if dropped else "nothing was dropped")),
        "decided_by": "measurement",
    })
    seq = (m.get("sequence") or {})
    decisions.append({
        "agent": "editor",
        "what": "Running order",
        "chose": seq.get("mode", "hook"),
        "why": seq.get("how", ""),
        "decided_by": "measurement",
    })
    decisions.append({
        "agent": "editor",
        "what": "Trim points",
        "chose": ", ".join(f"{s['clip'].split('_',1)[-1]} {s['start']}-{s['end']}"
                           for s in segs[:4]) + ("  ..." if len(segs) > 4 else ""),
        "why": ("for each clip, the highest-scoring window of its allotted "
                "length, stepped 0.2s at a time and penalised where "
                "freezedetect found the picture holding still"),
        "decided_by": "measurement",
    })
    decisions.append({
        "agent": "editor",
        "what": "Pacing",
        "chose": f"{len(segs)} segments, {min(s['duration'] for s in segs):.1f}"
                 f"-{max(s['duration'] for s in segs):.1f}s each",
        "why": ("runtime shared by the square root of each clip's motion, so "
                "livelier shots earn more screen time without one busy clip "
                "eating the reel"),
        "decided_by": "measurement",
    })

    # ---- 5: a transition chosen per cut, from the pair it joins -------------
    auto_xf = settings.get("transition_style", "auto") == "auto"
    xfades, cut_notes = [], []
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        ao = clips.get(a["clip"], {}).get("orientation")
        bo = clips.get(b["clip"], {}).get("orientation")
        ka = (vclips.get(a["clip"]) or {}).get("kind")
        kb = (vclips.get(b["clip"]) or {}).get("kind")
        if ao and bo and ao != bo:
            name, why = ORIENT_XFADE            # orientation always wins
        elif (ka, kb) in KIND_XFADE:
            name, why = KIND_XFADE[(ka, kb)]    # then what the model saw
        elif kb == "reveal":
            name, why = REVEAL_IN
        else:
            name, why = PAIR_XFADE[(band(a), band(b))]  # then motion
        xfades.append(name)
        cut_notes.append(f"{i + 1}→{i + 2}  {name}  ({why})")

    if auto_xf and xfades:
        settings["transition_list"] = xfades
        decisions.append({
            "agent": "animation",
            "what": "Transitions",
            "chose": " · ".join(xfades),
            "why": "one per cut, chosen from the two shots it joins:\n  "
                   + "\n  ".join(cut_notes),
            "decided_by": "measurement",
        })
    else:
        decisions.append({
            "agent": "animation",
            "what": "Transitions",
            "chose": settings.get("transition_style"),
            "why": "you picked this style, so the director left it alone",
            "decided_by": "you",
        })

    # ---- 6: crossfade length from how busy the cut is -----------------------
    if settings.get("transition_duration") in (None, "", "auto"):
        if med >= 2.5:
            t, why = 0.25, "busy footage - short cuts keep it moving"
        elif med >= 0.8:
            t, why = 0.4, "moderate motion - the pipeline's standard 0.4s"
        else:
            t, why = 0.6, "calm footage - a longer dissolve suits the pace"
        settings["transition_duration"] = t
        decisions.append({"agent": "animation",
                          "what": "Transition length", "chose": f"{t}s",
                          "why": f"{why} (median motion {med:.2f})",
                          "decided_by": "measurement"})
    else:
        decisions.append({"agent": "animation",
                          "what": "Transition length",
                          "chose": f"{settings['transition_duration']}s",
                          "why": "you set this", "decided_by": "you"})

    # ---- 7: mood from the batch's energy ------------------------------------
    if settings.get("mood", "auto") == "auto":
        for floor, name, why in MOOD_BANDS:
            if med >= floor:
                settings["mood"] = name
                decisions.append({"agent": "music", "what": "Music", "chose": name,
                                  "why": why % med, "decided_by": "measurement"})
                break
    else:
        decisions.append({"agent": "music", "what": "Music",
                          "chose": settings.get("mood"),
                          "why": "you chose this mood", "decided_by": "you"})

    # ---- what it will not decide, and why -----------------------------------
    # A caption the model offered is used ONLY where you wrote nothing, only
    # when it said it was confident, and it is always labelled as its wording
    # rather than passed off as yours.
    suggested = {}
    if vclips and not settings.get("_has_script"):
        for sg in segs:
            v = vclips.get(sg["clip"]) or {}
            if v.get("caption") and (v.get("confidence") or 0) >= 60:
                suggested[str(sg["index"])] = v["caption"]
    if suggested:
        settings["vision_captions"] = suggested
        decisions.append({
            "agent": "vision",
            "what": "Captions",
            "chose": "%d suggested by the vision model" % len(suggested),
            "why": ("you left the script blank, so captions the model offered "
                    "with at least 60% confidence were used:\n  "
                    + "\n  ".join("segment %s: %r" % (k, v)
                                   for k, v in sorted(suggested.items()))
                    + "\n  anything it was unsure about was left blank rather "
                      "than guessed"),
            "decided_by": "vision",
        })
    else:
        decisions.append({
        "agent": "editor",
        "what": "Captions",
        "chose": "from your script" if (settings.get("_has_script")) else "none",
        "why": ("nothing here recognises objects or reads a scene, so the "
                "director has no basis to describe your build and will not "
                "invent one - captions come from the script you write"),
        "decided_by": "you",
        })
    decisions.append({
        "agent": "animation",
        "what": "Speed",
        "chose": settings.get("speed", "normal"),
        "why": ("left to you on purpose: a defensible rule would need to know "
                "whether anyone is speaking, and a speed-up that ruins a "
                "demonstration's audio is worse than leaving it alone"),
        "decided_by": "you",
    })

    plan = {
        "vision": {"used": bool(vclips), "model": vis.get("model"),
                   "provider": vis.get("provider"),
                   "clips_seen": len(vclips),
                   "duplicates": vcross.get("duplicates") or [],
                   "hook": vcross.get("hook"),
                   "story": vcross.get("story") or ""},
        "median_motion": round(med, 3),
        "busy_threshold": round(busy_at, 3),
        "segments": len(segs),
        "transitions": xfades,
        "decisions": decisions,
    }
    with open(build_dir(job_id) / "edit_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    on_line("edit plan for %d segments (median motion %.2f, busy above %.2f)"
            % (len(segs), med, busy_at))
    for d in decisions:
        tag = "auto" if d["decided_by"] == "measurement" else "you "
        on_line("  [%s] %-18s %s" % (tag, d["what"] + ":", d["chose"]))
    return plan


def read_plan(job_id: str) -> dict | None:
    p = build_dir(job_id) / "edit_plan.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
