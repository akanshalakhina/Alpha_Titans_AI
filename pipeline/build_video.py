#!/usr/bin/env python3
"""
ALPHA TITANS build engine - generalises the A14_BUILD process to any clip folder.

Needs ffmpeg/ffprobe and the python3 standard library. No pip, no sudo.

Every stage is a separate invocation and is RESUMABLE: work already done is
skipped. That matters because each remote shell call has a time limit and
background jobs do not survive it, so long renders are done in small pieces.

Stages, in order:

  probe    ffprobe every clip; drop iOS raw/edited duplicates; pick orientation
  measure  freezedetect + signalstats YDIF per clip
  plan     choose keep windows from the measurements, build the cut sheet
  sheets   contact sheet per planned segment, so captions can describe the shot
  cut      render each segment onto the target canvas, caption burnt in
  join     xfade the segments together, record transition timings
  music    generate a track synced to those transitions (bgm_stdlib.py)
  mux      loudness-normalise, mux, thumbnail, verify, write NOTES.md

State lives in <out>/manifest.json. Captions live in <out>/captions.json and
are written between `sheets` and `cut`.

Usage:
  python3 build_video.py probe   --src DIR --out DIR [--orient auto|vertical|horizontal]
  python3 build_video.py measure --out DIR
  python3 build_video.py plan    --out DIR [--target SECONDS]
  python3 build_video.py sheets  --out DIR
  python3 build_video.py cut     --out DIR
  python3 build_video.py join    --out DIR
  python3 build_video.py music   --out DIR [--style lofi|bright]
  python3 build_video.py mux     --out DIR
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# These four are the only render knobs the UI needs to drive. They read an
# environment override and fall back to exactly their previous values, so the
# command line behaves identically when nothing is set. No logic below changes.
T = float(os.environ.get("AT_XFADE_T") or 0.4)      # crossfade length, seconds
FPS = 30
CRF = os.environ.get("AT_CRF") or "18"
PRESET = os.environ.get("AT_PRESET") or "fast"
XFADES = ([x for x in os.environ["AT_XFADES"].split(",") if x]
          if os.environ.get("AT_XFADES") else
          ["fadewhite", "fade", "wipeup", "smoothleft", "fade", "fadewhite",
           "wipeleft", "smoothup", "fade", "fadewhite"])
def _default_font():
    candidates = [
        os.environ.get("AT_FONT"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

FONT = _default_font()


VIDEO_EXT = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# window scoring / selection
MIN_SEG = 2.5
MAX_SEG = 12.0
# freezedetect's own output stays in the manifest and still steers window
# selection inside a clip, but it is NOT used to reject one: at its -38dB noise
# threshold it calls a slow pan 98% frozen and a locked-off interview 100%
# frozen, both of which are perfectly good footage.
#
# YDIF is an inter-frame DIFFERENCE metric, so its scale is a property of the
# subject, not of the quality of the footage. Hands assembling something read
# high; a talking head, an interview, a locked-off product shot, a screen
# recording or a slow landscape pan all read low while being perfectly good
# material. An absolute "below X is dead" cut-off therefore encodes one kind of
# subject and silently discards every other kind, so the thresholds below are
# relative to the set of clips the user actually uploaded:
#
#   ABS_DEAD_YDIF   an absolute floor low enough that only a genuinely static
#                   frame falls under it (lens cap, frozen export, slate)
#   REL_DEAD_SHARE  and, relative to the liveliest clip in this batch, anything
#                   under this share of it is dead *for this batch*
#   MAX_DROP_SHARE  never discard more than this share of the upload: if the
#                   rule wants to drop most of it, the footage is simply calm
#                   overall and window scoring should rank it rather than the
#                   rejection rule throwing it away
ABS_DEAD_YDIF = 0.05
REL_DEAD_SHARE = 0.02
MAX_DROP_SHARE = 0.34

# Per-frame noise floor for "did anything change here at all", and the share of
# a clip's frames that has to clear it before the clip counts as alive.
LIVE_YDIF = 0.02
LIVE_SHARE_FLOOR = 0.15


# ----------------------------------------------------------------- helpers

def sh(cmd, check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace")
    if check and p.returncode != 0:
        sys.exit("FAILED: %s\n%s" % (" ".join(cmd[:8]), out[-2500:]))
    return p.returncode, out


def hms(t):
    return "%02d:%02d:%06.3f" % (int(t // 3600), int(t % 3600 // 60), t % 60)


def load(out):
    with open(os.path.join(out, "manifest.json")) as f:
        return json.load(f)


def save(out, m):
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)


def esc(text):
    """Escape text for drawtext."""
    return (text.replace("\\", "\\\\").replace(":", "\\:")
                .replace("'", "").replace("%", ""))


def esc_path(path):
    """Escape a file path for use inside FFmpeg filter strings.
    FFmpeg unescapes once in filtergraph parsing and once in filter option splitting,
    so colons on Windows drive letters require '\\\\:' and forward slashes."""
    return str(path).replace("\\", "/").replace(":", "\\\\:")


def good_video(path, min_bytes=1024):
    """True if the file exists and ffprobe reads a real duration from it.

    The byte floor only catches an empty or half-written file. It deliberately
    does NOT try to judge a render by its size: how large a segment encodes is a
    property of the footage, not of whether the encode worked. A few seconds of
    a talking head, a screen recording, a title card or any other low-detail
    shot compresses to a handful of kilobytes, and an earlier 20 KB floor
    declared exactly those segments "missing" and failed the join. Whether
    ffprobe can read a positive duration is the question that actually matters.
    """
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        return False
    rc, o = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", path], check=False)
    if rc != 0 or o.strip() in ("", "N/A"):
        return False
    try:
        return float(o.strip()) > 0.0
    except ValueError:
        return False


# ------------------------------------------------------------------- probe

def probe_one(path):
    def q(args):
        rc, o = sh(["ffprobe", "-v", "error"] + args + [path], check=False)
        return o.strip()

    dur = q(["-show_entries", "format=duration", "-of", "csv=p=0"])
    wh = q(["-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "csv=p=0"])
    rot = q(["-select_streams", "v:0", "-show_entries",
             "stream_side_data=rotation", "-of", "csv=p=0"])
    if not re.search(r"-?\d", rot or ""):
        rot = q(["-show_entries", "stream_tags=rotate", "-of", "csv=p=0"])
    codec = q(["-select_streams", "v:0", "-show_entries", "stream=codec_name",
               "-of", "csv=p=0"])
    rate = q(["-select_streams", "v:0", "-show_entries",
              "stream=r_frame_rate", "-of", "csv=p=0"])
    audio = q(["-select_streams", "a", "-show_entries", "stream=codec_name",
               "-of", "csv=p=0"])
    # When the clip was actually recorded. Phones and cameras write this into
    # the container; it is the one honest signal for build order, because
    # footage of a build shot in sequence IS the build's narrative order.
    shot = q(["-show_entries", "format_tags=creation_time", "-of", "csv=p=0"])
    if not shot:
        shot = q(["-select_streams", "v:0", "-show_entries",
                  "stream_tags=creation_time", "-of", "csv=p=0"])
    shot = (shot or "").splitlines()[0].strip() if shot else ""

    try:
        w, h = [int(x) for x in wh.splitlines()[0].split(",")[:2]]
    except Exception:
        return None
    m = re.search(r"(-?\d+)", rot or "")
    rotation = int(m.group(1)) if m else 0
    dw, dh = (h, w) if abs(rotation) % 180 == 90 else (w, h)
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den or 1)
    except Exception:
        fps = 0.0
    return dict(name=os.path.basename(path), duration=float(dur or 0),
                coded=[w, h], rotation=rotation, display=[dw, dh],
                codec=codec, fps=round(fps, 3),
                audio=(audio.splitlines()[0] if audio else None),
                recorded=shot or None,
                mtime=round(os.path.getmtime(path), 3),
                orientation="vertical" if dh > dw else "horizontal")


def stage_probe(args):
    os.makedirs(args.out, exist_ok=True)
    files = sorted(p for p in glob.glob(os.path.join(args.src, "*"))
                   if p.lower().endswith(VIDEO_EXT))
    if not files:
        sys.exit("No video files in %s" % args.src)

    clips, bad = [], []
    for p in files:
        info = probe_one(p)
        if info is None or info["duration"] <= 0:
            bad.append(os.path.basename(p))
            continue
        clips.append(info)

    # iOS pairs: IMG_E1234 is Apple's rendered/edited take of raw IMG_1234.
    # Keep the E version (already has the slow-motion baked in), drop the raw
    # twin so the same moment is not used twice.
    by_name = {c["name"]: c for c in clips}
    dropped = {}
    for c in clips:
        m = re.match(r"^(IMG_)E(\d+)(\.\w+)$", c["name"], re.I)
        if not m:
            continue
        raw = "%s%s%s" % (m.group(1), m.group(2), m.group(3))
        if raw in by_name:
            slow = c["duration"] > by_name[raw]["duration"] * 1.5
            c["slowmo"] = bool(slow)
            dropped[raw] = "duplicate of %s%s" % ("", c["name"])
    kept = [c for c in clips if c["name"] not in dropped]

    vert = [c for c in kept if c["orientation"] == "vertical"]
    horz = [c for c in kept if c["orientation"] == "horizontal"]
    vdur = sum(c["duration"] for c in vert)
    hdur = sum(c["duration"] for c in horz)

    orient = args.orient
    if orient == "auto":
        orient = "vertical" if vdur > hdur else "horizontal"
    canvas = [1080, 1920] if orient == "vertical" else [1920, 1080]
    if os.environ.get("AT_CANVAS"):
        try:
            cw, chh = (int(v) for v in os.environ["AT_CANVAS"].split("x"))
            canvas = [cw, chh]
        except ValueError:
            pass
    target = 58.0 if orient == "vertical" else 120.0

    m = dict(src=os.path.abspath(args.src), out=os.path.abspath(args.out),
             orientation=orient, canvas=canvas, target=target,
             clips=kept, dropped=dropped, unreadable=bad,
             totals=dict(vertical_clips=len(vert), horizontal_clips=len(horz),
                         vertical_seconds=round(vdur, 2),
                         horizontal_seconds=round(hdur, 2),
                         kept_seconds=round(sum(c["duration"] for c in kept), 2)))
    save(args.out, m)

    print("SOURCE   %s" % m["src"])
    print("CLIPS    %d readable, %d kept after removing %d duplicate raw takes"
          % (len(clips), len(kept), len(dropped)))
    for name, why in sorted(dropped.items()):
        print("   dropped %-16s %s" % (name, why))
    if bad:
        print("   unreadable: %s" % ", ".join(bad))
    print("ORIENT   %s of source (%d vertical / %.1fs, %d horizontal / %.1fs)"
          % (orient, len(vert), vdur, len(horz), hdur))
    print("CANVAS   %dx%d, target length %.0fs" % (canvas[0], canvas[1], target))
    print("")
    print("%-18s %8s %11s %6s %8s %s"
          % ("clip", "dur", "display", "fps", "codec", "note"))
    for c in kept:
        print("%-18s %8.2f %11s %6.1f %8s %s"
              % (c["name"], c["duration"], "%dx%d" % tuple(c["display"]),
                 c["fps"], c["codec"], "slow-mo" if c.get("slowmo") else ""))


# ----------------------------------------------------------------- measure

def parse_meta(path, key):
    vals, idx = {}, None
    if not os.path.exists(path):
        return vals
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            mm = re.match(r"frame:(\d+)\s", line)
            if mm:
                idx = int(mm.group(1))
                continue
            if idx is None:
                continue
            mm = re.match(r"lavfi\.%s=([-0-9.eE]+)" % re.escape(key),
                          line.strip())
            if mm:
                try:
                    vals[idx] = float(mm.group(1))
                except ValueError:
                    pass
    return vals


def freeze_runs(path, duration):
    rc, o = sh(["ffmpeg", "-nostdin", "-v", "info", "-i", path,
                "-vf", "scale=320:-2,fps=6,freezedetect=n=-38dB:d=0.4",
                "-map", "0:v", "-f", "null", "-"], check=False)
    runs, start = [], None
    for line in o.splitlines():
        mm = re.search(r"freeze_start:\s*([0-9.]+)", line)
        if mm:
            start = float(mm.group(1))
        mm = re.search(r"freeze_end:\s*([0-9.]+)", line)
        if mm and start is not None:
            runs.append([start, float(mm.group(1))])
            start = None
    # A clip that stays frozen to the last frame gets a freeze_start with no
    # matching freeze_end. Without this the whole run is lost and the clip
    # reports as having no dead time at all.
    if start is not None and duration > start:
        runs.append([start, duration])
    return runs


def measure_once(src, rate, stats, duration):
    """Motion series AND freeze runs from ONE decode of the clip.

    The two used to be separate ffmpeg runs, so every frame of every clip
    was decoded twice - for a 35-minute iPhone HEVC clip that was most of a
    seven-minute measuring step (job b5e72b2c68a4, 19 Oct 2026). One decode
    feeds both filters through `split`. Returns the freeze runs, or None if
    the combined run failed, in which case the caller measures the old way.
    """
    graph = ("[0:v]split=2[m][f];"
             "[m]scale=160:-2,fps=%g,signalstats,metadata=print:file=%s[mo];"
             "[f]scale=320:-2,fps=6,freezedetect=n=-38dB:d=0.4[fo]"
             % (rate, esc_path(stats)))
    rc, o = sh(["ffmpeg", "-nostdin", "-v", "info", "-i", src,
                "-filter_complex", graph,
                "-map", "[mo]", "-f", "null", "-",
                "-map", "[fo]", "-f", "null", "-"], check=False)
    if rc != 0 or not os.path.exists(stats):
        return None
    runs, start = [], None
    for line in o.splitlines():
        mm = re.search(r"freeze_start:\s*([0-9.]+)", line)
        if mm:
            start = float(mm.group(1))
        mm = re.search(r"freeze_end:\s*([0-9.]+)", line)
        if mm and start is not None:
            runs.append([start, float(mm.group(1))])
            start = None
    if start is not None and duration > start:
        runs.append([start, duration])
    return runs


def stage_measure(args):
    m = load(args.out)
    vdir = os.path.join(args.out, "measure")
    os.makedirs(vdir, exist_ok=True)
    todo = [c for c in m["clips"] if "mean_ydif" not in c]
    if not todo:
        print("All %d clips already measured." % len(m["clips"]))
        return
    print("Measuring %d clip(s) (%d already done)."
          % (len(todo), len(m["clips"]) - len(todo)))

    for c in todo:
        src = os.path.join(m["src"], c["name"])
        rate = 10.0 if c["duration"] <= 120 else 5.0
        stats = os.path.join(vdir, c["name"] + ".ydif.txt")
        if os.path.exists(stats):
            os.remove(stats)
        runs = measure_once(src, rate, stats, c["duration"])
        if runs is None:                       # the two-pass way, as before
            if os.path.exists(stats):
                os.remove(stats)
            sh(["ffmpeg", "-nostdin", "-v", "error", "-i", src,
                "-vf", "scale=160:-2,fps=%g,signalstats,metadata=print:file=%s"
                       % (rate, esc_path(stats)), "-f", "null", "-"])
            runs = freeze_runs(src, c["duration"])
        series = parse_meta(stats, "signalstats.YDIF")
        vals = [series[i] for i in sorted(series)]
        frozen = sum(b - a for a, b in runs)

        c["sample_fps"] = rate
        c["ydif"] = [round(v, 2) for v in vals]
        c["mean_ydif"] = round(sum(vals) / len(vals), 2) if vals else 0.0
        c["max_ydif"] = round(max(vals), 2) if vals else 0.0
        c["freeze_runs"] = [[round(a, 2), round(b, 2)] for a, b in runs]
        c["freeze_share"] = round(frozen / c["duration"], 3) if c["duration"] else 0
        # How OFTEN anything changes, as opposed to how MUCH. A locked-off
        # interview or a screen recording has a small subject on a still
        # background: the whole-frame average is tiny, but nearly every frame
        # differs from the one before it. A genuinely dead clip - lens cap,
        # frozen export, held slate - is flat on this measure too, which is
        # what makes it the honest way to tell the two apart.
        c["live_share"] = (round(sum(1 for v in vals if v > LIVE_YDIF) / len(vals), 3)
                           if vals else 0.0)
        print("  %-18s mean %6.2f  max %6.2f  moving %5.1f%% of frames  "
              "frozen %5.1f%%  (%d runs)"
              % (c["name"], c["mean_ydif"], c["max_ydif"],
                 100 * c["live_share"], 100 * c["freeze_share"], len(runs)))
        save(args.out, m)
    print("Done. Run `plan` next.")


# -------------------------------------------------------------------- plan

def window_score(c, a, b):
    """Mean motion over [a,b], penalised for overlapping frozen stretches."""
    rate = c["sample_fps"]
    i0, i1 = int(a * rate), max(int(b * rate), int(a * rate) + 1)
    vals = c["ydif"][i0:i1]
    if not vals:
        return 0.0
    motion = sum(vals) / len(vals)
    overlap = 0.0
    for fa, fb in c["freeze_runs"]:
        overlap += max(0.0, min(b, fb) - max(a, fa))
    return motion * (1.0 - 0.8 * (overlap / (b - a)))


def best_window(c, length):
    """Highest scoring window of the given length, stepped by 0.2s."""
    dur = c["duration"]
    length = min(length, dur)
    best = (0.0, 0.0, -1.0)
    t = 0.0
    while t + length <= dur + 1e-6:
        s = window_score(c, t, t + length)
        if s > best[2]:
            best = (round(t, 2), round(t + length, 2), round(s, 3))
        t += 0.2
    return best


def stage_plan(args):
    m = load(args.out)
    if any("mean_ydif" not in c for c in m["clips"]):
        sys.exit("Run `measure` first.")
    target = args.target or m["target"]

    # Only reject what is genuinely dead, judged against this batch rather than
    # against an absolute motion figure (see the constants above for why).
    # `best_window` already steers away from the frozen stretches inside a clip,
    # so the bar here only has to catch footage with nothing in it at all.
    peak = max((c["mean_ydif"] for c in m["clips"]), default=0.0)
    dead_at = max(ABS_DEAD_YDIF, peak * REL_DEAD_SHARE)

    # The vision pass, when it ran, can nominate clips as redundant. They are
    # rejected with its reason rather than a motion reason, and only ever
    # clips it named - this never invents a drop of its own.
    vision_drop = {d.split("::", 1)[0]: (d.split("::", 1)[1] if "::" in d else
                                         "the vision pass flagged it as redundant")
                   for d in (getattr(args, "drop", None) or [])}

    usable, rejected = [], []
    for c in m["clips"]:
        if c["name"] in vision_drop:
            rejected.append((c, vision_drop[c["name"]]))
            continue
        # live_share was added after freeze_share; fall back for old manifests
        live = c.get("live_share")
        if live is None:
            live = 0.0 if c["mean_ydif"] <= dead_at else 1.0
        if live < LIVE_SHARE_FLOOR and c["mean_ydif"] <= dead_at:
            rejected.append((c, "nothing changes in it (only %.0f%% of frames "
                                "differ at all)" % (100 * live)))
        else:
            usable.append(c)

    # A rule that wants to throw away most of what was uploaded is describing
    # the footage, not judging it. Calm footage is still footage: put it back
    # and let window scoring decide how much screen time each clip earns.
    if rejected and len(rejected) > max(1, int(len(m["clips"]) * MAX_DROP_SHARE)):
        print("NOTE: the motion floor would have dropped %d of %d clips, which "
              "means this\n      batch is simply calm rather than empty. "
              "Keeping all of them and letting\n      the window scoring rank "
              "them instead." % (len(rejected), len(m["clips"])))
        usable = list(m["clips"])
        rejected = []

    if not usable:
        sys.exit("Every clip is a still frame. Nothing worth cutting.")

    # Share the budget out by motion, but never below MIN_SEG or above MAX_SEG,
    # and never more than the clip actually has.
    #
    # The weight is the square root of the motion figure rather than the figure
    # itself. Raw YDIF spans a wide range across unlike footage - a frenetic
    # clip can measure forty times a calm one - and weighting by it directly
    # lets one busy clip eat most of the runtime. The square root keeps the
    # ordering (livelier clips still earn more time) while compressing the
    # ratio, which is what makes a mixed batch cut into something coherent.
    weights = {c["name"]: max(c["mean_ydif"], 0.25) ** 0.5 for c in usable}
    wsum = sum(weights.values())
    lengths = {}
    for c in usable:
        want = target * weights[c["name"]] / wsum
        lengths[c["name"]] = max(MIN_SEG, min(MAX_SEG, want, c["duration"]))

    # If that undershoots the target, grow the clips that have room.
    for _ in range(80):
        totalled = sum(lengths.values())
        if totalled >= target - 0.5:
            break
        room = [c for c in usable
                if lengths[c["name"]] < min(MAX_SEG, c["duration"]) - 0.05]
        if not room:
            break
        add = min(0.5, (target - totalled) / len(room))
        for c in room:
            lengths[c["name"]] = min(MAX_SEG, c["duration"],
                                     lengths[c["name"]] + add)

    segs = []
    for c in usable:
        a, b, score = best_window(c, lengths[c["name"]])
        segs.append(dict(clip=c["name"], start=a, end=b,
                         duration=round(b - a, 2), score=score,
                         orientation=c["orientation"],
                         slowmo=bool(c.get("slowmo")),
                         mean_ydif=c["mean_ydif"],
                         reason="best-scoring window of %.1fs" % (b - a)))

    # ---- running order ------------------------------------------------
    # Three ways to sequence a build, all driven by real data:
    #
    #   hook      the default. The liveliest clip opens, because a reel that
    #             starts on the busiest moment holds a viewer; the rest follow
    #             the order they were uploaded in, and slow-motion lands last.
    #   recorded  chronological by the recording timestamp the camera wrote
    #             into the file. For footage shot as the build progressed this
    #             IS the story: frame, then sub-assembly, then mechanism, then
    #             the finished model running.
    #   uploaded  strictly the order you dragged them into, nothing moved.
    by_name = {c["name"]: c for c in m["clips"]}

    def shot_key(seg):
        c = by_name.get(seg["clip"], {})
        return (0, c["recorded"]) if c.get("recorded") else (1, "")

    mode = getattr(args, "sequence", None) or "hook"
    have_times = sum(1 for c in usable if c.get("recorded"))

    if mode == "recorded" and have_times >= 2:
        ordered = sorted(segs, key=lambda s: (shot_key(s), s["clip"]))
        how = "chronological by recording time (%d of %d clips carry one)" % (
            have_times, len(usable))
    elif mode == "recorded":
        ordered = sorted(segs, key=lambda s: s["clip"])
        how = ("upload order - only %d clip(s) carry a recording timestamp, "
               "so there is nothing to sort chronologically" % have_times)
    elif mode == "uploaded":
        ordered = sorted(segs, key=lambda s: s["clip"])
        how = "exactly the order you uploaded them in"
    else:
        slow = [s for s in segs if s["slowmo"]]
        rest = [s for s in segs if not s["slowmo"]]
        rest.sort(key=lambda s: -s["score"])
        forced = getattr(args, "hook", None)
        picked = [x for x in rest if x["clip"] == forced]
        if picked:
            # the vision pass nominated an opener; motion ranking still
            # decides everything after it
            hook = picked[:1]
            middle = sorted([x for x in rest if x is not hook[0]],
                            key=lambda s: s["clip"])
            how = ("%s opens - the vision pass picked it as the hook; the rest "
                   "follow upload order%s" % (forced, ", slow-motion last"
                                              if slow else ""))
        else:
            hook = rest[:1]
            middle = sorted(rest[1:], key=lambda s: s["clip"])
            how = "liveliest clip opens, then upload order%s" % (
                ", slow-motion last" if slow else "")
        ordered = hook + middle + slow

    for i, s in enumerate(ordered, 1):
        s["index"] = i
    m["sequence"] = {"mode": mode, "how": how}

    total = round(sum(s["duration"] for s in ordered) - (len(ordered) - 1) * T, 2)
    m["segments"] = ordered
    m["rejected"] = [dict(clip=c["name"], why=why) for c, why in rejected]
    m["planned_total"] = total
    save(args.out, m)

    print("TARGET   %.0fs on a %dx%d canvas" % (target, *m["canvas"]))
    print("ORDER    %s" % how)
    print("PLANNED  %.1fs from %d segments (%d crossfades of %.1fs)"
          % (total, len(ordered), len(ordered) - 1, T))
    if rejected:
        print("")
        print("EXCLUDED")
        for c, why in rejected:
            print("   %-18s %s" % (c["name"], why))
    print("")
    print("%-3s %-18s %-15s %6s %8s %s"
          % ("#", "clip", "in-out", "dur", "motion", "orientation"))
    for s in ordered:
        print("%-3d %-18s %-15s %6.1f %8.1f %s%s"
              % (s["index"], s["clip"], "%.1f-%.1f" % (s["start"], s["end"]),
                 s["duration"], s["mean_ydif"], s["orientation"],
                 "  slow-mo" if s["slowmo"] else ""))
    if total < target * 0.75:
        print("")
        print("NOTE: %.1fs is well short of the %.0fs target. The footage in "
              "this folder does not\n      contain more than this without "
              "padding it with static or repeated shots."
              % (total, target))
    print("")
    print("Next: `sheets`, look at them, write captions.json, then `cut`.")


# ------------------------------------------------------------------ sheets

def stage_sheets(args):
    m = load(args.out)
    if not m.get("segments"):
        sys.exit("Run `plan` first.")
    look = os.path.join(args.out, "look")
    os.makedirs(look, exist_ok=True)
    made = 0
    for s in m["segments"]:
        dest = os.path.join(look, "seg%02d.jpg" % s["index"])
        if os.path.exists(dest) and os.path.getsize(dest) > 3000:
            continue
        src = os.path.join(m["src"], s["clip"])
        n = 6
        step = max((s["end"] - s["start"]) / n, 0.2)
        vf = ("fps=1/%g,scale=-2:220,tile=3x2:margin=4:padding=3:color=white"
              % step)
        sh(["ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", hms(s["start"]), "-t", "%.2f" % (s["end"] - s["start"]),
            "-i", src, "-vf", vf, "-frames:v", "1", dest])
        made += 1
        print("  wrote look/seg%02d.jpg  (%s %.1f-%.1f)"
              % (s["index"], s["clip"], s["start"], s["end"]))
    print("%d sheet(s) written, %d already there." % (made, len(m["segments"]) - made))
    cj = os.path.join(args.out, "captions.json")
    if not os.path.exists(cj):
        with open(cj, "w") as f:
            json.dump(["" for _ in m["segments"]], f, indent=2)
        print("Wrote an empty captions.json - one line per segment, in order.")


# --------------------------------------------------------------------- cut

# ----------------------------------------------------------- overlays
#
# A graphic laid over a segment: a sticker, a logo, a lower third.
#
# THE RULE THIS KEEPS
# -------------------
# The agent layer names an asset, a segment and an ANCHOR. It never names
# a pixel or a second. Everything below turns those words into exact
# numbers from measurements this file already has - the canvas size, the
# segment's own duration, and the size ffmpeg reports for the asset at
# render time. A model cannot put a sticker three seconds into a clip
# that is two seconds long, because it is not given the vocabulary to
# say so.
#
# Nine anchors, as ffmpeg overlay expressions. `W`/`H` are the canvas,
# `w`/`h` the scaled overlay, and `%(m)d` the margin - so the position is
# resolved against the real sizes rather than assumed.
ANCHORS = {
    "top-left":      "x=%(m)d:y=%(m)d",
    "top-center":    "x=(W-w)/2:y=%(m)d",
    "top-right":     "x=W-w-%(m)d:y=%(m)d",
    "middle-left":   "x=%(m)d:y=(H-h)/2",
    "center":        "x=(W-w)/2:y=(H-h)/2",
    "middle-right":  "x=W-w-%(m)d:y=(H-h)/2",
    "bottom-left":   "x=%(m)d:y=H-h-%(m)d",
    "bottom-center": "x=(W-w)/2:y=H-h-%(m)d",
    "bottom-right":  "x=W-w-%(m)d:y=H-h-%(m)d",
}

# Margin and default size, both as a share of the canvas WIDTH so a
# sticker sits the same distance in on 720p and 4k.
OVERLAY_MARGIN_PCT = 4.0
OVERLAY_SCALE_PCT = 18.0

# How long a graphic holds, as a share of its segment. A closed
# vocabulary: the agent picks a word, this file picks the seconds.
HOLDS = {
    "full":  (0.00, 1.00),
    "open":  (0.00, 0.45),
    "close": (0.55, 1.00),
}

# Never right at the cut - a graphic that appears in the same frame as a
# crossfade reads as a glitch.
OVERLAY_EDGE = 0.15


def _read_overlays(out_dir):
    """`overlays.json`, grouped by segment index. Missing means none.

    Written by the agent layer next to `captions.json`, in the same
    shape: semantic only. Each entry names an `asset` relative to
    `assets/`, a `segment_index`, an `anchor` and optionally
    `scale_pct` and `hold`. A malformed file is reported and ignored
    rather than failing a render that was otherwise fine - a build
    losing its stickers is a disappointment; a build losing its video is
    a defect.
    """
    path = os.path.join(out_dir, "overlays.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError("expected a list")
    except (OSError, ValueError) as exc:
        print("WARNING: overlays.json unreadable (%s); rendering without "
              "graphics." % exc)
        return {}

    assets = os.path.join(out_dir, "assets")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("segment_index"))
        except (TypeError, ValueError):
            print("WARNING: overlay with no segment_index, skipping.")
            continue
        name = str(row.get("asset") or "").strip()
        # a name only - never a path the caller chose, so nothing outside
        # the build's own assets folder can be read
        name = os.path.basename(name)
        if not name:
            print("WARNING: overlay with no asset name, skipping.")
            continue
        out.setdefault(index, []).append({
            "path": os.path.join(assets, name),
            "anchor": row.get("anchor"),
            "scale_pct": row.get("scale_pct"),
            "hold": row.get("hold"),
        })
    return out


def overlay_chain(seg, canvas, overlays, first_input=1):
    """Turn semantic overlay requests into inputs and a filter fragment.

    Returns `(inputs, fragment, out_label)`, where `inputs` are extra
    `-i` arguments in the order the fragment refers to them.

    With no overlays it returns `([], "", None)` and the caller appends
    nothing - the filtergraph for a segment without graphics is then
    byte-identical to the one this file has always produced.
    """
    if not overlays:
        return [], "", None
    cw, ch = canvas
    margin = max(8, int(round(cw * OVERLAY_MARGIN_PCT / 100.0)))
    dur = float(seg["end"]) - float(seg["start"])

    inputs, parts, idx = [], [], first_input
    for ov in overlays:
        path = ov.get("path")
        if not path or not os.path.exists(path):
            print("WARNING: overlay asset missing, skipping: %s" % (path,))
            continue
        anchor = ANCHORS.get(ov.get("anchor"), ANCHORS["top-right"])
        pct = float(ov.get("scale_pct") or OVERLAY_SCALE_PCT)
        pct = min(max(pct, 2.0), 60.0)            # neither invisible nor a wall
        width = max(16, int(round(cw * pct / 100.0)))
        lo, hi = HOLDS.get(ov.get("hold"), HOLDS["full"])
        a = round(dur * lo + (OVERLAY_EDGE if lo == 0.0 else 0.0), 2)
        b = round(dur * hi - (OVERLAY_EDGE if hi == 1.0 else 0.0), 2)
        if b <= a:                                 # a segment too short to hold it
            print("WARNING: segment %s is too short for an overlay, skipping."
                  % seg.get("index"))
            continue
        inputs += ["-i", path]
        parts.append((idx, width, anchor % {"m": margin}, a, b))
        idx += 1

    if not parts:
        return [], "", None

    frag, prev = "", None
    for n, (i, width, pos, a, b) in enumerate(parts):
        src = prev or "%s"                         # filled in by the caller
        label = "[vo%d]" % n
        frag += (";[%d:v]scale=%d:-1[ov%d]"
                 ";%s[ov%d]overlay=%s:enable='between(t\\,%.2f\\,%.2f)'%s"
                 % (i, width, n, src, n, pos, a, b, label))
        prev = label
    return inputs, frag, prev


def fit_filter(seg, canvas, crop=None, overlays=None):
    """Filter chain putting one segment on the canvas, captioned."""
    cw, ch = canvas
    pre = "trim=start=%f:end=%f,setpts=PTS-STARTPTS" % (seg["start"], seg["end"])
    if crop:
        pre += ",crop=%d:%d:%d:%d" % tuple(crop)
    pre += ",fps=%d" % FPS

    same = (seg["orientation"] == ("vertical" if ch > cw else "horizontal"))
    if same and not crop:
        body = ("[0:v]%s,scale=%d:%d:force_original_aspect_ratio=increase,"
                "crop=%d:%d,setsar=1[v]" % (pre, cw, ch, cw, ch))
    else:
        body = ("[0:v]%s,split=2[fg][bg];"
                "[bg]scale=%d:%d:force_original_aspect_ratio=increase,"
                "crop=%d:%d,gblur=sigma=25,eq=brightness=-0.08[bgb];"
                "[fg]scale=%d:%d:force_original_aspect_ratio=decrease[fgs];"
                "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
                % (pre, cw, ch, cw, ch, cw, ch))

    def _with_overlays(chain, label):
        """Append the graphics, if any were asked for.

        Returns `(chain, label, inputs)`. When nothing was asked for the
        chain and label come back exactly as they went in, so a build
        with no overlays produces the filtergraph it always has.
        """
        ins, frag, out = overlay_chain(seg, canvas, overlays)
        if not frag:
            return chain, label, []
        return chain + (frag % label), out, ins

    text = seg.get("caption") or ""
    if not text:
        return _with_overlays(body, "[v]")
    # vertical output keeps captions clear of the platform UI
    y = "h-470" if ch > cw else "h-160"
    size = 56 if ch > cw else 48
    dur = seg["end"] - seg["start"]
    body += (";[v]drawtext=fontfile=%s:text='%s':fontsize=%d:fontcolor=white:"
             "borderw=3:bordercolor=black@0.7:x=(w-text_w)/2:y=%s:"
             "enable='between(t\\,%.2f\\,%.2f)'[vc]"
             % (esc_path(FONT), esc(text), size, y, 0.25, max(dur - 0.35, 0.5)))
    return _with_overlays(body, "[vc]")


def stage_cut(args):
    m = load(args.out)
    if not m.get("segments"):
        sys.exit("Run `plan` first.")
    cj = os.path.join(args.out, "captions.json")
    caps = []
    if os.path.exists(cj):
        with open(cj) as f:
            caps = json.load(f)
    if len(caps) < len(m["segments"]) or not all(str(c).strip() for c in caps):
        print("WARNING: captions.json is missing or incomplete - segments will "
              "render without captions.")
    for s, cap in zip(m["segments"], caps + [""] * len(m["segments"])):
        s["caption"] = str(cap).strip()
    save(args.out, m)

    # Graphics, beside the captions and read the same way. Absent is the
    # normal case and must stay completely silent: with no overlays.json
    # every filtergraph below is the one this file has always built.
    overlays = _read_overlays(args.out)

    parts = os.path.join(args.out, "parts")
    os.makedirs(parts, exist_ok=True)
    pending = []
    for s in m["segments"]:
        dest = os.path.join(parts, "p%02d.mp4" % s["index"])
        if good_video(dest):
            continue
        pending.append((s, dest))
    if not pending:
        print("All %d segments already rendered." % len(m["segments"]))
        return
    print("Rendering %d segment(s); %d already done."
          % (len(pending), len(m["segments"]) - len(pending)))
    for s, dest in pending:
        src = os.path.join(m["src"], s["clip"])
        # Seek BEFORE opening the clip. `trim` alone decodes from second 0
        # up to the in-point, so a shot at 19 minutes decoded 19 minutes
        # of video first - about nine minutes of cutting for six shots of
        # long footage (job b5e72b2c68a4). Input seeking jumps to the
        # nearest keyframe and decodes only from there; the trim then
        # works relative to that point, one second before the shot.
        lead = max(0.0, s["start"] - 1.0)
        rel = dict(s, start=s["start"] - lead, end=s["end"] - lead)
        fc, out_label, extra = fit_filter(rel, m["canvas"], s.get("crop"),
                                          overlays.get(s["index"]))
        sh(["ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", "%.3f" % lead, "-i", src] + extra +
           ["-filter_complex", fc, "-map", out_label, "-an",
            "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
            "-profile:v", "high", "-pix_fmt", "yuv420p", dest])
        print("  p%02d  %-18s %.1f-%.1f  %s"
              % (s["index"], s["clip"], s["start"], s["end"],
                 s["caption"] or "(no caption)"))
    print("Done. Run `cut` again if it stopped early, then `join`.")


# -------------------------------------------------------------------- join

ROLLS = ("intro", "outro")


def _roll_source(out_dir, which):
    """An `intro.*` / `outro.*` in the build's assets folder, or None."""
    assets = os.path.join(out_dir, "assets")
    if not os.path.isdir(assets):
        return None
    for name in sorted(os.listdir(assets)):
        stem, ext = os.path.splitext(name)
        if stem.lower() == which and ext.lower() in (
                ".mp4", ".mov", ".m4v", ".webm"):
            return os.path.join(assets, name)
    return None


def _normalise_roll(src, dest, canvas, on_missing=print):
    """Bring a pre- or post-roll onto the cut's own canvas and frame rate.

    A roll that is 25fps, or a different aspect, or carries its own
    audio cannot simply be concatenated: ffmpeg's concat demuxer needs
    every input to agree, and a mismatch shows up as a stutter or a
    silent failure rather than an error.

    VIDEO ONLY, deliberately. The score is mixed across the whole cut in
    `stage_mux`, so a roll's own soundtrack would fight it. Dropping it
    here is a decision worth seeing rather than a surprise later.
    """
    cw, ch = canvas
    if good_video(dest):
        return dest
    rc, out = sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src,
                  "-vf", ("scale=%d:%d:force_original_aspect_ratio=decrease,"
                          "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=%d"
                          % (cw, ch, cw, ch, FPS)),
                  "-an", "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
                  "-profile:v", "high", "-pix_fmt", "yuv420p", dest])
    if rc != 0 or not good_video(dest):
        on_missing("WARNING: could not normalise %s (%s); continuing without "
                   "it." % (os.path.basename(src), (out or "").strip()[-160:]))
        return None
    return dest


def _duration_of(path):
    rc, out = sh(["ffprobe", "-v", "error", "-show_entries",
                  "format=duration", "-of", "csv=p=0", path])
    try:
        return round(float((out or "").strip()), 3)
    except ValueError:
        return 0.0


def _attach_rolls(out_dir, m, joined, canvas):
    """Concatenate an intro and/or outro around the finished body.

    Runs AFTER the crossfade chain and leaves it untouched: the rolls are
    butt-joined, not faded. An intro that dissolves into itself reads as
    a mistake, and a logo that bleeds into the first shot is worse than
    no logo at all.

    Nothing happens when no roll is present, which is the normal case.

    THE PART THAT IS EASY TO GET WRONG
    ----------------------------------
    `m["transitions"]` are the beat marks `stage_music` scores against.
    Prepending an intro moves every cut later by exactly the intro's
    length, so the marks are shifted by that amount. Leaving them alone
    would key the whole score to moments that no longer exist.
    """
    found = [(w, _roll_source(out_dir, w)) for w in ROLLS]
    found = [(w, src) for w, src in found if src]
    if not found:
        return

    ready = []
    for which, src in found:
        dest = os.path.join(out_dir, "_%s_norm.mp4" % which)
        got = _normalise_roll(src, dest, canvas)
        if got:
            ready.append((which, got))
    if not ready:
        return

    intro = next((p for w, p in ready if w == "intro"), None)
    outro = next((p for w, p in ready if w == "outro"), None)
    order = ([intro] if intro else []) + [joined] + ([outro] if outro else [])

    listing = os.path.join(out_dir, "_rolls.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        for p in order:
            fh.write("file '%s'\n" % p.replace("'", "'\\''"))
    stitched = os.path.join(out_dir, "_withrolls.mp4")
    rc, out = sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat",
                  "-safe", "0", "-i", listing, "-c", "copy", stitched])
    if rc != 0 or not good_video(stitched):
        print("WARNING: could not attach intro/outro (%s); the cut is "
              "unchanged." % (out or "").strip()[-160:])
        return

    lead = _duration_of(intro) if intro else 0.0
    if lead:
        m["transitions"] = [round(t + lead, 3) for t in m.get("transitions", [])]
    m["total_duration"] = _duration_of(stitched)
    m["rolls"] = {w: os.path.basename(p) for w, p in ready}
    save(out_dir, m)
    os.replace(stitched, joined)
    print("Attached %s -> %.1fs%s"
          % (" + ".join(w for w, _ in ready), m["total_duration"],
             ("; transition marks shifted %.2fs" % lead) if lead else ""))


def stage_join(args):
    m = load(args.out)
    parts = [os.path.join(args.out, "parts", "p%02d.mp4" % s["index"])
             for s in m["segments"]]
    missing = [p for p in parts if not good_video(p)]
    if missing:
        sys.exit("Missing rendered segments: %s\nRun `cut` again."
                 % ", ".join(os.path.basename(p) for p in missing))

    durs = [s["duration"] for s in m["segments"]]
    offsets, cum = [], 0.0
    for i, d in enumerate(durs[:-1], start=1):
        cum += d
        offsets.append(round(cum - i * T, 3))
    total = round(sum(durs) - len(offsets) * T, 3)
    m["transitions"] = offsets
    m["total_duration"] = total
    save(args.out, m)

    joined = os.path.join(args.out, "_joined.mp4")
    if good_video(joined):
        rc, o = sh(["ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "csv=p=0", joined])
        if abs(float(o.strip()) - total) < 1.0:
            print("_joined.mp4 already built (%.1fs)." % total)
            print("transitions: %s" % offsets)
            return

    if len(parts) == 1:
        print("Only one segment - nothing to crossfade, copying it through.")
        sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", parts[0],
            "-c:v", "copy", joined])
        print("Wrote _joined.mp4  %.1fs" % total)
        # a one-segment cut still gets its intro and outro
        _attach_rolls(args.out, m, joined, canvas=m["canvas"])
        return

    inputs = []
    for p in parts:
        inputs += ["-i", p]
    chain, prev = [], "[0:v]"
    for i in range(1, len(parts)):
        label = "[vout]" if i == len(parts) - 1 else "[v%d]" % i
        chain.append("%s[%d:v]xfade=transition=%s:duration=%.1f:offset=%.3f%s"
                     % (prev, i, XFADES[(i - 1) % len(XFADES)], T,
                        offsets[i - 1], label))
        prev = label
    print("Joining %d segments with %d crossfades -> %.1fs ..."
          % (len(parts), len(offsets), total))
    sh(["ffmpeg", "-nostdin", "-v", "error", "-y"] + inputs +
       ["-filter_complex", ";".join(chain), "-map", "[vout]",
        "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
        "-profile:v", "high", "-pix_fmt", "yuv420p", joined])
    print("Wrote _joined.mp4  %.1fs   transitions at %s" % (total, offsets))
    _attach_rolls(args.out, m, joined, canvas=m["canvas"])


# ------------------------------------------------------------------- music

def stage_music(args):
    m = load(args.out)
    if not m.get("transitions"):
        sys.exit("Run `join` first.")
    gen = os.path.join(args.out, "bgm_stdlib.py")
    if not os.path.exists(gen):
        sys.exit("bgm_stdlib.py is not in %s - copy it there first." % args.out)
    wav = os.path.join(args.out, "bgm.wav")
    cmd = [sys.executable, gen, wav, "%.2f" % m["total_duration"]] + \
          ["%.3f" % t for t in m["transitions"]]
    if args.style:
        cmd += ["--style", args.style]
    rc, o = sh(cmd)
    print(o.strip())
    m["music"] = dict(file="bgm.wav", style=args.style or "lofi",
                      log=o.strip().splitlines()[:2])
    save(args.out, m)


# --------------------------------------------------------------------- mux

def stage_mux(args):
    m = load(args.out)
    joined = os.path.join(args.out, "_joined.mp4")
    wav = os.path.join(args.out, "bgm.wav")
    if not good_video(joined):
        sys.exit("Run `join` first.")
    tag = os.path.basename(args.out.rstrip("/\\")).replace("_BUILD", "")
    kind = "short" if m["orientation"] == "vertical" else "longform"
    final = os.path.join(args.out, "ALPHA_TITANS_%s_%s.mp4" % (tag, kind))

    if os.path.exists(wav):
        sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", joined, "-i", wav,
            "-filter_complex",
            "[1:a]loudnorm=I=-16:TP=-1.5:LRA=9,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final])
    else:
        print("No bgm.wav - muxing silence instead.")
        sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", joined,
            "-f", "lavfi", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000", "-shortest",
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final])

    # thumbnail from the middle of the strongest segment
    best = max(m["segments"], key=lambda s: s["score"])
    at = 0.0
    for s in m["segments"]:
        if s["index"] == best["index"]:
            at += s["duration"] / 2
            break
        at += s["duration"] - T
    cw, ch = (1280, 720) if m["orientation"] == "horizontal" else (1080, 1920)
    thumb = os.path.join(args.out, "thumb_%dx%d.jpg" % (cw, ch))
    sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "%.2f" % at,
        "-i", final, "-frames:v", "1",
        "-vf", "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,"
               "drawtext=fontfile=%s:text='ALPHA TITANS':fontsize=%d:"
               "fontcolor=white:borderw=4:bordercolor=black@0.7:x=40:y=h-120"
               % (cw, ch, cw, ch, esc_path(FONT), 64 if cw > ch else 72), thumb])

    checks = {}
    rc, o = sh(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_name,width,height,"
                "r_frame_rate,sample_rate", "-of", "default=nw=1", final])
    checks["probe"] = o.strip()
    rc, o = sh(["ffmpeg", "-nostdin", "-v", "info", "-i", final,
                "-af", "ebur128", "-f", "null", "-"], check=False)
    # ebur128 prints a running per-frame "I:" the whole way through, starting at
    # -70 LUFS before any signal. Only the value in the trailing Summary block
    # is the integrated loudness, so read that one and not the first match.
    tail = o.split("Summary:")[-1] if "Summary:" in o else ""
    mm = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    checks["lufs"] = mm.group(1) if mm else "unknown"
    rc, o = sh(["ffmpeg", "-nostdin", "-v", "error", "-i", final,
                "-vf", "blackdetect=d=0.3", "-f", "null", "-"], check=False)
    checks["black"] = o.strip() or "none"
    m["final"] = os.path.basename(final)
    m["checks"] = checks
    save(args.out, m)
    write_notes(args.out, m)

    print("FINAL    %s" % final)
    print(checks["probe"])
    print("loudness %s LUFS   black frames: %s" % (checks["lufs"], checks["black"]))
    print("thumb    %s" % thumb)
    print("notes    %s" % os.path.join(args.out, "NOTES.md"))


def write_notes(out, m):
    L = []
    tag = os.path.basename(out.rstrip("/\\"))
    L.append("# %s\n" % tag)
    L.append("Source: `%s`\n" % m["src"])
    L.append("Output: **%s** - %dx%d, %.1fs, %s\n"
             % (m.get("final", "(not built)"), m["canvas"][0], m["canvas"][1],
                m.get("total_duration", 0),
                "vertical short" if m["orientation"] == "vertical"
                else "horizontal long form"))

    L.append("\n## Source clips\n")
    if m["dropped"]:
        L.append("Dropped %d raw take(s) that duplicate an Apple-rendered "
                 "version of the same moment: %s.\n"
                 % (len(m["dropped"]), ", ".join(sorted(m["dropped"]))))
    L.append("\n| clip | duration | display | fps | codec | note |")
    L.append("|---|---|---|---|---|---|")
    for c in m["clips"]:
        L.append("| %s | %.2fs | %dx%d | %.0f | %s | %s |"
                 % (c["name"], c["duration"], c["display"][0], c["display"][1],
                    c["fps"], c["codec"], "slow-mo" if c.get("slowmo") else ""))

    L.append("\n## Measurements\n")
    L.append("freezedetect at -38dB/0.4s, and signalstats YDIF "
             "(frame-to-frame luma difference) as the motion measure.\n")
    L.append("\n| clip | mean motion | peak | frozen | freeze runs |")
    L.append("|---|---|---|---|---|")
    for c in m["clips"]:
        L.append("| %s | %.1f | %.1f | %.0f%% | %d |"
                 % (c["name"], c.get("mean_ydif", 0), c.get("max_ydif", 0),
                    100 * c.get("freeze_share", 0), len(c.get("freeze_runs", []))))

    if m.get("rejected"):
        L.append("\n### Excluded\n")
        for r in m["rejected"]:
            L.append("- **%s** - %s" % (r["clip"], r["why"]))

    L.append("\n## Cut sheet\n")
    L.append("| # | clip | in-out | duration | caption |")
    L.append("|---|---|---|---|---|")
    for s in m["segments"]:
        L.append("| %d | %s | %.1f-%.1f | %.1fs | %s |"
                 % (s["index"], s["clip"], s["start"], s["end"],
                    s["duration"], s.get("caption") or "-"))
    if m.get("transitions"):
        L.append("\n%d crossfades of %.1fs at %s seconds.\n"
                 % (len(m["transitions"]), T,
                    ", ".join("%.1f" % t for t in m["transitions"])))

    if m.get("music"):
        L.append("\n## Audio\n")
        L.append("Generated by `bgm_stdlib.py` (pure standard library, no "
                 "installs) with an accent placed on every transition "
                 "timestamp. Nothing sampled, so there is no licence or "
                 "Content ID exposure.\n")
        for line in m["music"].get("log", []):
            L.append("\n    %s" % line)

    if m.get("checks"):
        L.append("\n\n## Verification\n")
        L.append("- Loudness: %s LUFS (target -16)" % m["checks"].get("lufs"))
        L.append("- Black frames: %s" % m["checks"].get("black"))
        for line in m["checks"].get("probe", "").splitlines():
            L.append("- %s" % line)

    target = m.get("target", 0)
    tot = m.get("total_duration", 0)
    if target and tot < target * 0.75:
        L.append("\n## On the length\n")
        L.append("Target was about %.0fs; this is %.1fs. After removing "
                 "duplicate takes and the clips that measured as static, that "
                 "is what the folder actually contains. Getting to the target "
                 "would mean repeating shots or leaving dead time in, so it "
                 "was left at the honest length.\n" % (target, tot))

    with open(os.path.join(out, "NOTES.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# -------------------------------------------------------------------- main

STAGES = dict(probe=stage_probe, measure=stage_measure, plan=stage_plan,
              sheets=stage_sheets, cut=stage_cut, join=stage_join,
              music=stage_music, mux=stage_mux)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--src", help="folder of source clips (probe only)")
    ap.add_argument("--out", required=True, help="the _BUILD folder")
    ap.add_argument("--orient", default="auto",
                    choices=["auto", "vertical", "horizontal"])
    ap.add_argument("--target", type=float, help="override target length")
    ap.add_argument("--sequence", choices=["hook", "recorded", "uploaded"],
                    default="hook", help="running order for the cut")
    ap.add_argument("--drop", action="append", default=[],
                    help="CLIP[::reason] to exclude (repeatable)")
    ap.add_argument("--hook", help="force this clip to open the cut")
    ap.add_argument("--style", help="music style: lofi or bright")
    args = ap.parse_args()
    if args.stage == "probe" and not args.src:
        ap.error("probe needs --src")
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
