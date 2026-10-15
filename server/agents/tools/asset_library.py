"""The graphics a job can actually put on screen.

WHY THIS EXISTS
---------------
The renderer can lay an image over a segment now, but the Animation
agent's prompt still ends with:

    Do NOT name an overlay, a graphic, a template or a font. There is no
    asset library in this project and you cannot refer to one.

which was true and, for an empty job, still is. A model cannot name a
file that does not exist, and inviting it to try is how you get
`lower_third_blue.png` in an artifact and nothing on screen.

So the prompt is written from the folder. When a job has overlay images
the agent is given their exact names and may choose only from them; when
it has none the sentence above stands unchanged. Nothing here ever
widens what the model may say beyond what is on disk.

THE DEFAULTS
------------
Four plain shapes are generated on demand so a fresh install can use
overlays without the operator going to find artwork. They are drawn here
in pure standard library - zlib and struct write the PNG - because
Pillow is not in `requirements.txt`, and something that ships as a
default must not depend on a package that happens to be installed.

They are deliberately plain: white, soft-edged, no colour and no
branding, so they read on any footage and look like punctuation rather
than somebody else's design.
"""

from __future__ import annotations

import math
import pathlib
import struct
import zlib

# What the renderer can overlay. `.jpg` is excluded on purpose: a graphic
# without an alpha channel is a rectangle stuck on the picture.
OVERLAY_SUFFIXES = (".png", ".webp")

# Reserved names - these are pre/post-roll video, not overlays.
ROLL_STEMS = ("intro", "outro")

# Generated at this size and scaled by the renderer to a share of the
# canvas, so one file serves 720p and 4k alike.
SIZE = 256


# ----------------------------------------------------------- PNG writing

def _png(path: pathlib.Path, width: int, height: int, rgba: bytearray) -> None:
    """Write 8-bit RGBA, non-interlaced. zlib and struct, nothing else."""
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)                               # filter type 0: none
        raw += rgba[y * stride:(y + 1) * stride]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
                     + chunk(b"IEND", b""))


def _canvas(n: int = SIZE) -> bytearray:
    return bytearray(n * n * 4)


def _put(buf: bytearray, n: int, x: int, y: int, a: float) -> None:
    """White pixel at `a` opacity, kept at the strongest value written.

    Taking the max rather than overwriting lets overlapping strokes build
    a shape without a later one punching a hole in an earlier one.
    """
    if not (0 <= x < n and 0 <= y < n) or a <= 0:
        return
    i = (y * n + x) * 4
    v = min(255, max(0, int(round(a * 255))))
    if v <= buf[i + 3]:
        return
    buf[i] = buf[i + 1] = buf[i + 2] = 255
    buf[i + 3] = v


def _disc(buf, n, cx, cy, r, soft=2.0):
    for y in range(max(0, int(cy - r - soft)), min(n, int(cy + r + soft) + 1)):
        for x in range(max(0, int(cx - r - soft)),
                       min(n, int(cx + r + soft) + 1)):
            d = math.hypot(x - cx, y - cy)
            _put(buf, n, x, y, min(1.0, max(0.0, (r - d) / soft + 0.5)))


def _bar(buf, n, x0, y0, x1, y1, w):
    """A soft-edged line from (x0,y0) to (x1,y1), `w` wide."""
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) * 2 + 1
    for s in range(steps + 1):
        t = s / steps
        _disc(buf, n, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, w / 2.0)


def _sparkle(n=SIZE) -> bytearray:
    """A four-point star - 'look here', without an arrow's direction."""
    buf, c = _canvas(n), n / 2.0
    for dx, dy in ((1, 0), (0, 1)):
        for sign in (1, -1):
            for i in range(int(n * 0.42)):
                t = i / (n * 0.42)
                _disc(buf, n, c + dx * sign * i, c + dy * sign * i,
                      max(1.0, n * 0.055 * (1 - t) ** 1.6))
    _disc(buf, n, c, c, n * 0.07)
    return buf


def _arrow_focus(n=SIZE) -> bytearray:
    """A chevron pointing down-left, for picking one thing out of many."""
    buf = _canvas(n)
    w = n * 0.11
    _bar(buf, n, n * 0.74, n * 0.22, n * 0.30, n * 0.66, w)
    _bar(buf, n, n * 0.30, n * 0.66, n * 0.60, n * 0.70, w)
    _bar(buf, n, n * 0.30, n * 0.66, n * 0.34, n * 0.36, w)
    return buf


def _badge_step(n=SIZE) -> bytearray:
    """A filled disc to sit a step number against."""
    buf = _canvas(n)
    _disc(buf, n, n / 2.0, n / 2.0, n * 0.40, soft=3.0)
    return buf


def _frame_highlight(n=SIZE) -> bytearray:
    """Four corner brackets - a frame that does not hide the picture."""
    buf = _canvas(n)
    m, arm, w = n * 0.10, n * 0.26, n * 0.055
    for sx in (0, 1):
        for sy in (0, 1):
            x = m if sx == 0 else n - m
            y = m if sy == 0 else n - m
            _bar(buf, n, x, y, x + (arm if sx == 0 else -arm), y, w)
            _bar(buf, n, x, y, x, y + (arm if sy == 0 else -arm), w)
    return buf


DEFAULTS = {
    "sparkle.png": (_sparkle, "a four-point star, for drawing the eye"),
    "arrow_focus.png": (_arrow_focus, "a chevron, for picking one thing out"),
    "badge_step.png": (_badge_step, "a filled disc, for a step marker"),
    "frame_highlight.png": (_frame_highlight,
                            "four corner brackets, framing without hiding"),
}


# ------------------------------------------------------------- the folder

def assets_dir(build_dir) -> pathlib.Path | None:
    """`<build>/assets`, or None when there is no build to look in.

    None is a real case, not a guard: the prompt is also rendered in
    tests and in the degraded path where no job directory exists, and
    those must read as "no assets" rather than raising.
    """
    if build_dir in (None, ""):
        return None
    return pathlib.Path(build_dir) / "assets"


def ensure_defaults(build_dir) -> list[str]:
    """Write any missing default graphic. Returns what it created.

    An operator's own file of the same name is never overwritten: their
    artwork beats ours, and silently replacing it would be the kind of
    surprise that costs an afternoon.
    """
    d = assets_dir(build_dir)
    if d is None:
        return []
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    made = []
    for name, (draw, _why) in DEFAULTS.items():
        path = d / name
        if path.exists():
            continue
        try:
            _png(path, SIZE, SIZE, draw())
            made.append(name)
        except OSError:                                  # pragma: no cover
            pass
    return made


def available(build_dir) -> list[str]:
    """Overlay graphics this job can use, by name, sorted.

    Pre- and post-roll videos are excluded: they are a different feature
    with a different vocabulary, and offering `intro.mp4` as a sticker
    would be offering something that cannot work.
    """
    d = assets_dir(build_dir)
    if d is None or not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in OVERLAY_SUFFIXES:
            continue
        if p.stem.lower() in ROLL_STEMS:
            continue
        out.append(p.name)
    return out


def describe(build_dir) -> str:
    """The prompt paragraph about graphics, written from the folder.

    With no assets this returns the sentence the prompt has always
    carried, word for word. That is the point: the model is told what is
    there, and when nothing is there it is told not to invent any.
    """
    names = available(build_dir)
    if not names:
        return ("- Do NOT name an overlay, a graphic, a template or a font. "
                "There is no asset\n  library in this project and you cannot "
                "refer to one.")
    lines = ["- A graphic MAY be placed over a beat, using `overlay`. These "
             "files exist and\n  are the ONLY ones you may name:"]
    for n in names:
        why = DEFAULTS.get(n, (None, ""))[1]
        lines.append(f"    - {n}" + (f"  ({why})" if why else ""))
    lines.append("  Name one with `\"overlay\": {\"asset\": \"<file>\", "
                 "\"anchor\": \"<anchor>\",\n"
                 "  \"scale_pct\": <2-60>, \"hold\": \"full|open|close\"}`. "
                 "You choose WHERE in words\n"
                 "  only - the exact pixels and seconds are computed from the "
                 "canvas and the\n  shot's measured length. Leave `overlay` "
                 "out when a beat needs no graphic;\n"
                 "  most do not, and a sticker on every shot is noise.")
    return "\n".join(lines)
