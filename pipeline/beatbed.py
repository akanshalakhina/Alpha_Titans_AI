#!/usr/bin/env python3
"""
ALPHA TITANS - instrumental beat engine.

Everything is synthesised from oscillators and filtered noise. No samples, no
loops, no third-party recordings, so the result carries no licence, no
attribution requirement and nothing for Content ID to match.

Written around the edit rather than under it - the lesson from the earlier
work, where fitting a finished track to a finished cut never lined up better
than about 5 frames. Here the arrangement is built FROM the scene-change list:

  * tempo is searched to fit the scene changes as closely as a constant tempo can
  * an impact lands exactly on every scene change, independent of the beat grid
  * a riser fills the 1.5s before each big section so the change is telegraphed
  * sections gate instruments on and off, so the energy has an actual shape
    instead of looping one bed for 38 seconds

Hooks, in both senses of the word:
  * a rhythmic hook - a syncopated kick pattern that repeats from bar one
  * a melodic hook - an 8-note pentatonic motif, stated filtered, then plain,
    then doubled an octave up in the drop, which is what makes a listener
    recognise it the second time

    python3 beatbed.py out.wav 37.8 1.4 5.0 10.4 14.8 19.0 24.4 30.2 35.4
"""

import argparse
import sys

import numpy as np
from scipy import signal as sg

SR = 48000
PENT = [0, 3, 5, 7, 10]              # minor pentatonic


# ------------------------------------------------------------------ helpers

def hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def lp(x, cut, order=4):
    sos = sg.butter(order, min(cut, SR * 0.49) / (SR / 2), "low", output="sos")
    return sg.sosfilt(sos, x)


def hp(x, cut, order=2):
    sos = sg.butter(order, max(cut, 12) / (SR / 2), "high", output="sos")
    return sg.sosfilt(sos, x)


def bp(x, lo, hi, order=2):
    sos = sg.butter(order, [max(lo, 12) / (SR / 2), min(hi, SR * 0.49) / (SR / 2)],
                    "band", output="sos")
    return sg.sosfilt(sos, x)


def ad(n, atk, dec, shape=2.0):
    """Attack/decay envelope."""
    a = max(1, int(atk * SR))
    t = np.arange(n)
    e = np.empty(n)
    a = min(a, n)
    e[:a] = np.linspace(0.0, 1.0, a)
    if n > a:
        e[a:] = np.exp(-np.linspace(0, 1, n - a) * dec) ** shape
    return e


def fftconv(x, ir):
    n = len(x) + len(ir) - 1
    N = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:len(x)]
    return y


def reverb_ir(seconds=0.55, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 1, n) * 7.0)
    ir = lp(ir, 4200.0)
    ir[:int(0.004 * SR)] = 0.0
    return ir / (np.max(np.abs(ir)) or 1.0) * 0.5


# ----------------------------------------------------------------- voices

def v_kick(dur=0.50, amp=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 118.0 * np.exp(-t * 30.0) + 47.0
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7.0)
    click = np.random.default_rng(1).standard_normal(n) * np.exp(-t * 420.0) * 0.25
    return amp * np.tanh((body + click) * 1.5) * 0.9


def v_snare(dur=0.26, amp=0.7, seed=2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    noise = bp(rng.standard_normal(n), 900, 7200) * ad(n, 0.001, 9.0)
    tone = (np.sin(2 * np.pi * 186 * t) + 0.6 * np.sin(2 * np.pi * 278 * t)) \
        * np.exp(-t * 26.0)
    return amp * (noise * 0.85 + tone * 0.35)


def v_clap(dur=0.34, amp=0.55, seed=5):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for k, off in enumerate((0.0, 0.011, 0.022, 0.033)):
        i = int(off * SR)
        if i >= n:
            break
        seglen = n - i
        burst = bp(rng.standard_normal(seglen), 1100, 5200)
        burst *= ad(seglen, 0.0008, 40.0 if k < 3 else 7.0)
        out[i:] += burst * (0.6 if k < 3 else 1.0)
    return amp * out / 2.2


def v_hat(dur=0.07, amp=0.22, open_=False, seed=7):
    n = int(dur * (3.4 if open_ else 1.0) * SR)
    rng = np.random.default_rng(seed)
    x = hp(rng.standard_normal(n), 7000)
    return amp * x * ad(n, 0.0005, 14.0 if open_ else 46.0)


def v_shaker(dur=0.10, amp=0.10, seed=11):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    return amp * bp(rng.standard_normal(n), 4200, 11000) * ad(n, 0.004, 26.0)


def v_sub(freq, dur, amp=0.85):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)
    return amp * np.tanh(x * 1.3) * ad(n, 0.006, 2.6, 1.0)


def v_bass(freq, dur, amp=0.55):
    """Filtered saw - carries the groove above the sub."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    saw = 2.0 * ((freq * t) % 1.0) - 1.0
    saw += 2.0 * (((freq * 1.005) * t % 1.0) - 0.5) * 0.4
    env = ad(n, 0.004, 3.2, 1.0)
    cut = 320 + 900 * env
    return amp * lp(saw * env, float(np.mean(cut)))


def v_pluck(freq, dur, amp=0.45, bright=1.0):
    """Short mallet/pluck - the hook voice."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * freq * t)
         + 0.45 * bright * np.sin(4 * np.pi * freq * t) * np.exp(-t * 9.0)
         + 0.18 * bright * np.sin(6.01 * np.pi * freq * t) * np.exp(-t * 16.0))
    return amp * x * ad(n, 0.002, 7.0, 1.2)


def v_bell(freq, dur, amp=0.30):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * freq * t)
         + 0.5 * np.sin(2 * np.pi * freq * 2.76 * t)
         + 0.25 * np.sin(2 * np.pi * freq * 5.4 * t))
    return amp * x * ad(n, 0.001, 4.0, 1.0)


def v_pad(freqs, dur, amp=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k, f in enumerate(freqs):
        det = 1.0 + 0.0022 * np.sin(2 * np.pi * (0.08 + 0.013 * k) * t)
        x += np.sin(2 * np.pi * f * t * det) / len(freqs)
    swell = 0.5 + 0.5 * np.sin(np.linspace(-np.pi / 2, np.pi * 1.5, n))
    return amp * lp(x, 2600.0) * swell


def v_riser(dur=1.5, amp=0.40, seed=13):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    step = 2048
    for i in range(0, n, step):
        j = min(n, i + step)
        f = 400 + 6200 * (i / n) ** 1.7
        out[i:j] = bp(noise[i:j], f * 0.7, min(f * 1.6, SR * 0.45))
    tone = np.sin(2 * np.pi * np.cumsum(180 + 900 * t ** 2) / SR) * 0.35
    return amp * (out * 0.9 + tone) * (t ** 1.6)


def v_impact(dur=1.4, amp=0.85, seed=17):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    boom = np.sin(2 * np.pi * (72 * np.exp(-t * 3.4) + 38) * t) * np.exp(-t * 3.2)
    air = lp(rng.standard_normal(n), 3000) * np.exp(-t * 9.0) * 0.35
    return amp * np.tanh((boom + air) * 1.2)


# ------------------------------------------------------------------- track

class Track:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.buses = {k: np.zeros(self.n) for k in
                      ("drum", "bass", "music", "fx")}

    def add(self, bus, sig, at, gain=1.0):
        i = int(at * SR)
        if i < 0:
            sig = sig[-i:]
            i = 0
        j = min(self.n, i + len(sig))
        if j > i:
            self.buses[bus][i:j] += sig[:j - i] * gain


def best_bpm(marks, lo=86.0, hi=100.0, step=0.05):
    """Constant tempo whose beat grid best fits the scene changes."""
    best = None
    n = int(round((hi - lo) / step)) + 1
    for i in range(n):
        bpm = lo + i * step
        beat = 60.0 / bpm
        err = sum(abs(m - round(m / beat) * beat) for m in marks)
        if best is None or err < best[1]:
            best = (round(bpm, 2), err)
    return best[0], best[1] / max(1, len(marks))


# --------------------------------------------------------------- arrangement

def sections(marks, total):
    """Name each span between scene changes, and give it an energy level."""
    edges = [0.0] + list(marks) + [total]
    names = ["hook", "intro", "groove", "hookA", "hookB", "build", "drop",
             "break", "outro"]
    out = []
    for i in range(len(edges) - 1):
        out.append(dict(name=names[i] if i < len(names) else "extra",
                        t0=edges[i], t1=edges[i + 1]))
    return out


# 16-step patterns, one bar
K_BASE = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]     # rhythmic hook
K_DROP = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
S_BASE = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
H_BASE = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
H_DROP = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]

# melodic hook: 16 eighth-note slots, pentatonic degrees or None for a rest
HOOK = [0, None, 3, 5, None, 3, 7, None, 5, None, 3, 0, None, 3, None, None]
PROG = [0, -4, 3, -2]            # Am - F - C - G, as root offsets


def build(total, marks, hits=(), whoosh=(), thud=(), root=57, seed=9):
    bpm, err = best_bpm(marks)
    beat = 60.0 / bpm
    bar = beat * 4
    step = beat / 4                    # 16ths
    tr = Track(total)
    secs = sections(marks, total)

    def sec_at(t):
        for s in secs:
            if s["t0"] <= t < s["t1"]:
                return s["name"]
        return "outro"

    # An obstacle accent landing ~100ms after a pattern kick reads as a flam,
    # not as a hit - the first knock measured QUIETER than the bar before it.
    # The action accents therefore take priority: any pattern drum inside this
    # window of one is dropped.
    action = sorted(list(hits) + list(thud))

    def collides(at, win=0.17):
        return any(abs(at - a) < win for a in action)

    rng = np.random.default_rng(seed)
    nbars = int(total / bar) + 2

    for b in range(nbars):
        t0 = b * bar
        if t0 >= total:
            break
        name = sec_at(t0)
        chord_root = root + PROG[b % len(PROG)]

        drums_on = name in ("intro", "groove", "hookA", "hookB", "build", "drop")
        full = name in ("groove", "hookA", "hookB", "build", "drop")
        drop = name == "drop"

        # ---- drums
        if drums_on:
            kick = K_DROP if drop else K_BASE
            for i, on in enumerate(kick):
                if not on:
                    continue
                at = t0 + i * step
                if at >= total:
                    break
                # intro is half-time: only the downbeats
                if name == "intro" and i not in (0, 10):
                    continue
                if collides(at):
                    continue
                tr.add("drum", v_kick(amp=1.0 if i == 0 else 0.86), at)
            if full:
                for i, on in enumerate(S_BASE):
                    if on and t0 + i * step < total and not collides(t0 + i * step):
                        tr.add("drum", v_snare(seed=b * 3 + i), t0 + i * step,
                               0.9 if not drop else 1.0)
                        if drop:
                            tr.add("drum", v_clap(seed=b + i), t0 + i * step, 0.8)
                hats = H_DROP if drop else H_BASE
                for i, on in enumerate(hats):
                    if not on or t0 + i * step >= total:
                        continue
                    if collides(t0 + i * step, 0.10):
                        continue
                    op = (i == 14 and not drop) or (drop and i in (7, 15))
                    tr.add("drum", v_hat(open_=op, seed=b * 16 + i),
                           t0 + i * step, 0.9)
            if name in ("build", "drop"):
                for i in range(0, 16, 2):
                    at = t0 + i * step + step * 0.5
                    if at < total and not collides(at, 0.10):
                        tr.add("drum", v_shaker(seed=b * 8 + i), at, 0.8)
            # a fill on the bar before the drop
            if name == "build" and t0 + bar >= next(
                    (s["t0"] for s in secs if s["name"] == "drop"), 1e9) - bar:
                for i in range(12, 16):
                    at = t0 + i * step
                    if at < total:
                        tr.add("drum", v_snare(seed=99 + i), at, 0.5 + 0.15 * (i - 12))

        # ---- bass
        if name in ("intro", "groove", "hookA", "hookB", "build", "drop"):
            tr.add("bass", v_sub(hz(chord_root - 24), bar * 0.55), t0,
                   0.9 if not drop else 1.0)
            if full:
                pushes = (0, 4, 6, 10, 12, 14) if name in ("build", "drop") \
                    else (6, 10, 14)
                for i in pushes:
                    at = t0 + i * step
                    if at < total and not collides(at, 0.22):
                        tr.add("bass", v_bass(hz(chord_root - 12), step * 1.8),
                               at, 0.85 if name in ("build", "drop") else 0.7)

        # ---- pad
        if name in ("hook", "intro", "hookA", "hookB", "build", "drop", "break",
                    "outro"):
            voicing = [hz(chord_root), hz(chord_root + 7), hz(chord_root + 12),
                       hz(chord_root + 14)]
            g = 0.7 if name in ("break", "hook") else 0.45
            tr.add("music", v_pad(voicing, bar * 1.02), t0, g)

        # ---- melodic hook
        play_hook = name in ("hookA", "hookB", "build", "drop", "break")
        if name == "intro" and b % 2 == 1:
            play_hook = True                     # a filtered taste of it early
        if play_hook:
            for i, deg in enumerate(HOOK):
                if deg is None:
                    continue
                at = t0 + i * (beat / 2)
                if at >= total:
                    break
                m = chord_root + 12 + PENT[deg % len(PENT)] + (12 if deg >= 10 else 0)
                dur = beat * (0.9 if name != "break" else 1.6)
                gain = dict(intro=0.35, hookA=0.85, hookB=0.9, build=0.95,
                            drop=1.0).get(name, 0.8)
                if name == "break":
                    gain = 0.6
                    if i % 2:
                        continue                 # halve the density
                tr.add("music", v_pluck(hz(m), dur, bright=1.0 if name != "break"
                                        else 0.4), at, gain)
                if name == "hookB" and i % 4 == 0:
                    tr.add("music", v_bell(hz(m + 12), beat * 1.4), at, 0.35)
                if drop:
                    tr.add("music", v_pluck(hz(m + 12), dur * 0.7, amp=0.30),
                           at, 0.9)

    # ---- impacts exactly on the scene changes, plus risers into the big ones
    big = {s["t0"] for s in secs if s["name"] in ("drop", "hookA", "break")}
    for m in marks:
        tr.add("fx", v_impact(), m, 1.15 if m in big else 0.9)
        tr.add("fx", v_sub(hz(root - 24), 0.7), m, 0.7)
        deg = PENT[int(m) % len(PENT)]
        tr.add("music", v_bell(hz(root + 24 + deg), 1.6), m, 0.30)
    for m in sorted(big):
        tr.add("fx", v_riser(1.5), m - 1.5, 0.85)
    # ---- action accents for the obstacle run
    for h in hits:
        tr.add("drum", v_kick(amp=1.0), h)
        tr.add("drum", v_clap(amp=0.72, seed=int(h * 100) % 97), h)
        tr.add("drum", v_hat(open_=True, seed=int(h * 31) % 71), h, 0.8)
        tr.add("fx", v_impact(0.5), h, 0.38)
    for w in whoosh:
        tr.add("fx", v_riser(0.55, seed=int(w * 10) % 91), w - 0.55, 0.55)
        tr.add("drum", v_hat(open_=True, seed=int(w * 7) % 83), w, 0.7)
    for th in thud:
        tr.add("fx", v_impact(1.1), th, 0.95)
        tr.add("fx", v_sub(hz(root - 24), 0.55), th, 0.85)
        tr.add("drum", v_kick(amp=1.0), th)
        tr.add("drum", v_snare(amp=0.9, seed=int(th * 13) % 79), th, 1.0)
        tr.add("drum", v_hat(open_=True, seed=int(th * 17) % 67), th, 0.7)

    # opening hook: grab attention in the first beat
    tr.add("fx", v_impact(1.8), 0.0, 1.35)
    tr.add("fx", v_sub(hz(root - 24), 1.1), 0.0, 1.0)
    tr.add("drum", v_kick(amp=1.0), 0.0, 1.0)
    for i in range(6):
        tr.add("drum", v_hat(seed=200 + i), i * (step * 0.75), 0.26 + 0.05 * i)
    tr.add("fx", v_riser(1.25), max(0.0, marks[0] - 1.25), 0.7)

    return tr, bpm, err, secs


def mixdown(tr, total, marks=(), short=()):
    n = int(total * SR)
    drum = tr.buses["drum"][:n]
    bass = tr.buses["bass"][:n]
    music = tr.buses["music"][:n]
    fx = tr.buses["fx"][:n]

    # sidechain: duck bass and pad under the kick so the low end stays clean
    key = np.abs(lp(drum, 160.0))
    key = lp(key, 12.0)
    key /= (np.max(key) or 1.0)
    duck = 1.0 - 0.55 * key
    bass = bass * duck
    music = music * (1.0 - 0.18 * key)

    ir = reverb_ir()
    wet = fftconv(music * 0.55 + drum * 0.12, ir)
    mix = drum * 0.95 + bass * 0.95 + music * 0.85 + fx * 0.8 + wet * 0.22

    # Carve a brief dip immediately before each scene change. Without it the
    # impact is masked inside the busy sections and the change is not felt -
    # two of the eight measured as an energy DROP before this was added.
    if len(marks) or len(short):
        gate = np.ones(n)
        rel = int(0.010 * SR)
        # scene changes get a longer hole; the obstacle knocks get a short one
        # so a dense drop section does not sound gappy around every impact
        for group, pre_s, depth in ((marks, 0.075, 0.30),
                                    (short, 0.045, 0.45)):
            pre = int(pre_s * SR)
            for m in group:
                i = int(m * SR)
                a = max(0, i - pre)
                if i <= a:
                    continue
                gate[a:i] = np.minimum(gate[a:i],
                                       np.linspace(1.0, depth, i - a) ** 1.4)
                bb = min(n, i + rel)
                if bb > i:
                    gate[i:bb] = np.minimum(gate[i:bb],
                                            np.linspace(depth, 1.0, bb - i))
        mix = mix * gate

    mix = hp(mix, 27.0)
    mix = lp(mix, 16500.0)
    # glue: slow RMS compression, then soft clip
    rms = np.sqrt(lp(mix ** 2, 6.0).clip(1e-9))
    gain = 1.0 / (1.0 + 2.2 * np.maximum(rms - 0.16, 0.0))
    mix = mix * gain
    mix = np.tanh(mix * 1.28) / 1.28

    peak = np.max(np.abs(mix)) or 1.0
    mix *= (10 ** (-1.6 / 20.0)) / peak

    # narrow stereo: tiny delay on the music side only, kept mono at the bottom
    d = int(0.010 * SR)
    side = np.concatenate([np.zeros(d), (music * 0.5)[:-d]]) * 0.5
    left = mix + side * 0.5
    right = mix - side * 0.5
    low = lp(mix, 140.0) * 0.5
    left = left * 0.94 + low * 0.06
    right = right * 0.94 + low * 0.06
    st = np.stack([left, right], axis=1)
    st /= (np.max(np.abs(st)) or 1.0) / (10 ** (-1.4 / 20.0))

    # short fades so nothing clicks
    fi, fo = int(0.02 * SR), int(0.5 * SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]
    st[-fo:] *= (np.linspace(1, 0, fo) ** 1.4)[:, None]
    return st.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("total", type=float)
    ap.add_argument("marks", nargs="+", type=float)
    ap.add_argument("--hits", nargs="*", type=float, default=[],
                    help="obstacle knocks - short percussive accents")
    ap.add_argument("--whoosh", nargs="*", type=float, default=[],
                    help="jump launches - a short riser into each")
    ap.add_argument("--thud", nargs="*", type=float, default=[],
                    help="landings - the big low hits")
    a = ap.parse_args()

    tr, bpm, err, secs = build(a.total, a.marks, a.hits, a.whoosh,
                               a.thud)
    audio = mixdown(tr, a.total, a.marks,
                    list(a.hits) + list(a.thud))
    import soundfile as sf
    sf.write(a.out, audio, SR, subtype="PCM_16")

    print("tempo      %.2f BPM  (mean beat error %.3fs = %.1f frames at 30fps)"
          % (bpm, err, err * 30))
    print("impacts    exactly on %s" % ", ".join("%.2f" % m for m in a.marks))
    if a.hits:
        print("knocks     %s" % ", ".join("%.2f" % h for h in a.hits))
    if a.whoosh:
        print("launches   %s" % ", ".join("%.2f" % w for w in a.whoosh))
    if a.thud:
        print("landings   %s" % ", ".join("%.2f" % t for t in a.thud))
    print("sections   %s" % "  ".join("%s@%.1f" % (s["name"], s["t0"])
                                      for s in secs))
    print("wrote %s  %.2fs stereo %d Hz" % (a.out, len(audio) / SR, SR))


if __name__ == "__main__":
    main()
