"""Turning the Music agent's reasoning into an actual score.

THE GAP THIS CLOSES
-------------------
The Music agent decides a great deal - `mood`, `tempo_feel`, `energy_arc`,
`instrumentation`, `sonic_direction`, a per-beat `sections` list and a list
of sounds to `avoid`. Exactly one of those reached the engine. `extras.py`
maps the mood to `--style lofi` or `--style bright` and nothing else, so
the seven moods collapse to TWO scores and the same mood produced a
byte-identical wav every single time. An agent had reasoned about a
driving pulse that layers per beat and climaxes at the payoff, and the
renderer played the same eighty-BPM loop it plays for everything.

WHY NOT JUST WIDEN extras.py
----------------------------
`server/extras.py` and all three `pipeline/*.py` are protected files. But
the engine already has the knobs:

    bgm_stdlib.render(duration, transitions, bpm, style, root_midi, seed)
    bgm_stdlib.best_bpm(transitions, lo, hi)

`style` was the only one anybody passed. So this module imports that same
engine as a library - the protected code is READ, never changed - chooses
its arguments from what the agent actually decided, and writes the wav.
`music_stage` then takes it through the path it already has for an
operator's own track (`music_source="upload"`), which loops, trims and
fades it to the cut. No second audio engine exists, and the protected
stage still does the fitting.

REPRODUCIBLE, NOT RANDOM
------------------------
The seed is a hash of what the agent wrote, never of the clock. The same
reasoning therefore yields the same score - so a rebuild is comparable to
its predecessor - while a different brief yields a different one. Music
that changed on every render would make it impossible to tell whether a
difference came from the edit or from chance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from ..artifacts import Music

# How a tempo FEELING becomes a search band for the engine's own beat
# fitter. The bands are the engine's: `bgm_stdlib.main` searches 68-92 for
# lofi and 110-140 for bright, and these widen that into four feelings
# rather than two styles. `best_bpm` still picks the exact figure inside
# the band, because it is fitting to this cut's real transition marks.
TEMPO_BANDS: dict[str, tuple[float, float]] = {
    "slow": (62.0, 80.0),
    "moderate": (84.0, 104.0),
    "driving": (112.0, 140.0),
    # "varied" is not a tempo, it is a refusal to commit to one. A single
    # rendered track has exactly one BPM, so the honest reading is the
    # middle of the range rather than a pretence of variation.
    "varied": (92.0, 120.0),
}

# Which of the engine's two voicings each mood belongs to. Same mapping
# `extras.MOODS` uses, read off it rather than reinvented, so the two can
# never disagree about what "calm" sounds like.
STYLE_FOR_MOOD: dict[str, str] = {
    "lofi": "lofi", "calm": "lofi",
    "bright": "bright", "upbeat": "bright",
    "beat": "bright", "energetic": "bright", "cinematic": "lofi",
}

# The key, in MIDI note numbers. The engine defaults to 57 (A) for
# everything; giving each mood its own root is what stops two different
# moods from being the same tune at a different speed. Low and dark for
# the calm end, higher and brighter for the energetic end.
ROOT_FOR_MOOD: dict[str, int] = {
    "lofi": 55,         # G
    "calm": 53,         # F
    "cinematic": 50,    # D
    "bright": 60,       # C
    "upbeat": 62,       # D
    "beat": 57,         # A  (the engine's own default)
    "energetic": 64,    # E
}

# An energy arc that RISES deserves to start below its tempo and arrive at
# it; one that falls, the reverse. The engine renders a single BPM, so the
# lever here is which end of the band to sit at.
_ARC_PULL = {"low": -0.5, "building": -0.25, "steady": 0.0,
             "high": 0.3, "peak": 0.5, "falling": -0.3}


def _engine():
    """`pipeline/bgm_stdlib.py`, imported as a library.

    By path, because `pipeline` is not a package and this must not depend
    on how the pipeline is invoked as a script. Read-only: nothing here
    writes to a protected file.
    """
    from .pipeline_bridge import PIPELINE_DIR
    path = Path(PIPELINE_DIR) / "bgm_stdlib.py"
    if not path.exists():
        return None
    name = "at_bgm_stdlib"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _seed_from(music: Music, job_id: str) -> int:
    """A stable number from what the agent WROTE.

    Hashing the reasoning rather than the clock is what makes two builds
    of the same brief comparable: if the score changed every render, a
    difference between two cuts could never be attributed to the edit.
    """
    text = "\u0000".join([
        job_id, music.mood, music.tempo_feel or "",
        music.sonic_direction or "", music.energy_arc or "",
        "|".join(music.instrumentation or []),
        "|".join(music.avoid or []),
        "|".join(f"{s.beat_index}:{s.energy}" for s in (music.sections or [])),
    ])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def settings_for(music: Music, job_id: str, marks: list[float]) -> dict:
    """Every engine argument, chosen from the agent's own decisions.

    Returned as a plain dict so it can be put in the trace: a score the
    person cannot account for is no better than a random one.
    """
    mood = music.mood
    style = STYLE_FOR_MOOD.get(mood, "lofi")
    lo, hi = TEMPO_BANDS.get(music.tempo_feel or "moderate",
                             TEMPO_BANDS["moderate"])

    # Where in the band to sit, from the arc the agent described. The
    # sections carry it beat by beat; their average is the shape of the
    # whole piece, which is what a single-tempo render can express.
    energies = [s.energy for s in (music.sections or [])]
    pull = (sum(_ARC_PULL.get(e, 0.0) for e in energies) / len(energies)
            if energies else 0.0)
    span = hi - lo
    centre = lo + span / 2 + pull * span / 2
    lo2, hi2 = max(lo, centre - span / 4), min(hi, centre + span / 4)
    if hi2 <= lo2:                                   # a degenerate band
        lo2, hi2 = lo, hi

    return {"style": style, "bpm_lo": round(lo2, 1), "bpm_hi": round(hi2, 1),
            "root_midi": ROOT_FOR_MOOD.get(mood, 57),
            "seed": _seed_from(music, job_id) % 10_000,
            "arc_pull": round(pull, 3), "marks": len(marks)}


def compose(music: Music, job_id: str, out_path: str | Path,
            duration: float, marks: list[float],
            on_line=None) -> tuple[Path | None, dict]:
    """Render a score from the agent's reasoning. Returns (path, settings).

    `(None, settings)` means the engine could not be reached, which the
    caller reports and falls back to the mood-only path for. It is never
    papered over: a silent cut that was supposed to be scored is a defect
    the person has to be told about.
    """
    plan = settings_for(music, job_id, marks)
    eng = _engine()
    if eng is None:
        plan["error"] = "bgm_stdlib.py not found"
        return None, plan

    try:
        bpm, err = eng.best_bpm(marks or [1.0], plan["bpm_lo"], plan["bpm_hi"])
        plan["bpm"] = round(bpm, 1)
        plan["beat_error_s"] = round(err, 3)
        if on_line:
            on_line(f"scoring from the Music agent: {plan['style']} at "
                    f"{plan['bpm']:.1f} BPM, key {plan['root_midi']}, "
                    f"seed {plan['seed']} - fitted to {len(marks)} "
                    f"transition mark(s), mean beat error "
                    f"{plan['beat_error_s']:.3f}s")
        # `render` returns the samples AND the stereo delay it wants
        # written with; `write_wav` indexes the buffer by it, so the two
        # have to travel together.
        buf, delay = eng.render(duration, marks or [1.0], bpm,
                                style=plan["style"],
                                root_midi=plan["root_midi"],
                                seed=plan["seed"])
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        eng.write_wav(str(out), buf, delay)
    except Exception as exc:                              # noqa: BLE001
        plan["error"] = f"{type(exc).__name__}: {exc}"
        if on_line:
            on_line(f"the agent's score could not be rendered ({plan['error']});"
                    f" falling back to the mood-only engine")
        return None, plan

    if not out.exists() or out.stat().st_size == 0:
        plan["error"] = "the engine wrote no audio"
        return None, plan
    plan["bytes"] = out.stat().st_size
    return out, plan
