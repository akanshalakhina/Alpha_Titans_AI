#!/usr/bin/env python3
"""
Original background music generator, PURE STANDARD LIBRARY.

Runs on myyukti with nothing installed. No numpy, no scipy, no soundfile, no
pip, no sudo. Uses wave, math, array and random only.

Synthesised from oscillators and filtered noise, so the output carries no
licence, no attribution requirement and nothing for Content ID to match.

Two functions, same interface as the numpy versions:
    best_bpm(transitions, lo, hi)      -> (bpm, mean_error_seconds)
    render(duration, transitions, bpm) -> list of (left, right) float samples

Usage:
    python3 bgm_stdlib.py out.wav 39.3 2.2 5.2 9.0 15.6 23.2 30.8
    python3 bgm_stdlib.py out.wav 39.3 2.2 5.2 --style bright

Styles: lofi (default, mellow, ~80 BPM) or bright (~126 BPM).
Takes roughly 20 to 40 seconds for a 40 second track. That is fine.
"""

import array
import math
import random
import sys
import wave

SR = 48000
PENT = [0, 3, 5, 7, 10]          # minor pentatonic, nothing can clash


def best_bpm(transitions, lo=68.0, hi=92.0, step=0.1):
    """Constant tempo whose beat grid best fits the given cut points."""
    best = None
    n = int(round((hi - lo) / step)) + 1
    for i in range(n):
        bpm = lo + i * step
        beat = 60.0 / bpm
        err = 0.0
        for t in transitions:
            err += abs(t - round(t / beat) * beat)
        if best is None or err < best[1]:
            best = (round(bpm, 2), err)
    return best[0], best[1] / max(1, len(transitions))


def hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def add(buf, sig, at):
    i = int(at * SR)
    if i < 0:
        sig = sig[-i:]
        i = 0
    n = len(buf)
    for k, v in enumerate(sig):
        j = i + k
        if j >= n:
            break
        buf[j] += v


def rhodes(freq, dur, amp=0.20):
    """Soft electric piano: slow attack, long decay, three detuned voices."""
    n = int(dur * SR)
    out = [0.0] * n
    dets = (1.0, 1.0022, 0.9979)
    weights = (1.0, 0.6, 0.6)
    for det, w in zip(dets, weights):
        f = freq * det
        w1 = 2.0 * math.pi * f / SR
        w2 = w1 * 2.0
        w4 = w1 * 4.0
        for i in range(n):
            t = i / SR
            env = math.exp(-t * 1.5) * (1.0 - math.exp(-t * 26.0))
            if env < 1e-4:
                break
            s = (math.sin(w1 * i)
                 + 0.30 * math.sin(w2 * i) * math.exp(-t * 4.5)
                 + 0.08 * math.sin(w4 * i) * math.exp(-t * 9.0))
            out[i] += w * s * env
    return [v * amp / 2.2 for v in out]


def bass(freq, dur, amp=0.30):
    n = int(dur * SR)
    w1 = 2.0 * math.pi * freq / SR
    out = [0.0] * n
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 1.1) * (1.0 - math.exp(-t * 40.0))
        if env < 1e-4:
            break
        out[i] = amp * env * (math.sin(w1 * i) + 0.14 * math.sin(2 * w1 * i))
    return out


def kick(dur=0.42, amp=0.30):
    n = int(dur * SR)
    out = [0.0] * n
    ph = 0.0
    for i in range(n):
        t = i / SR
        f = 105.0 * math.exp(-t * 26.0) + 46.0
        ph += 2.0 * math.pi * f / SR
        out[i] = amp * math.sin(ph) * math.exp(-t * 7.5)
    return out


def brush(dur=0.22, amp=0.055, seed=0):
    """Noise burst, one-pole high pass then low pass, percussive envelope."""
    rng = random.Random(seed)
    n = int(dur * SR)
    raw = [rng.gauss(0.0, 1.0) for _ in range(n)]
    hp, prev_in, prev_out = [], 0.0, 0.0
    a = 0.93
    for x in raw:
        y = a * (prev_out + x - prev_in)
        hp.append(y)
        prev_in, prev_out = x, y
    lp, y = [], 0.0
    b = 0.35
    for x in hp:
        y += b * (x - y)
        lp.append(y)
    return [amp * v * math.exp(-(i / SR) * 16.0) for i, v in enumerate(lp)]


def hat(dur=0.07, amp=0.011, seed=0):
    rng = random.Random(seed)
    n = int(dur * SR)
    prev = 0.0
    out = []
    for i in range(n):
        x = rng.gauss(0.0, 1.0)
        d = x - prev
        prev = x
        out.append(amp * d * math.exp(-(i / SR) * 55.0))
    return out


def pad(freqs, dur, amp=0.075):
    n = int(dur * SR)
    out = [0.0] * n
    for k, f in enumerate(freqs):
        w = 2.0 * math.pi * f / SR
        wd = 2.0 * math.pi * (0.07 + 0.011 * k) / SR
        for i in range(n):
            drift = 1.0 + 0.0016 * math.sin(wd * i)
            out[i] += math.sin(w * i * drift) / len(freqs)
    ws = 2.0 * math.pi * 0.045 / SR
    return [amp * v * (0.55 + 0.45 * math.sin(ws * i - math.pi / 2))
            for i, v in enumerate(out)]


def vinyl(n, seed=11):
    """Steady noise floor plus sparse surface crackle."""
    rng = random.Random(seed)
    out = [0.0] * n
    y = 0.0
    for i in range(n):
        x = rng.gauss(0.0, 1.0)
        y += 0.25 * (x - y)
        out[i] = y * 0.010
    for _ in range(int(n / SR * 26)):
        i = rng.randrange(0, max(1, n - 300))
        w = rng.randrange(40, 260)
        for k in range(w):
            if i + k < n:
                out[i + k] += rng.gauss(0.0, 1.0) * math.exp(-k / w * 7.0) \
                    * rng.uniform(0.015, 0.050)
    return lowpass(out, 5200.0, poles=2)


def lowpass(buf, cutoff, poles=4):
    """Cascaded one-pole low pass. Four poles gives roughly 24 dB per octave,
    which is what actually removes the digital edge. One pole is far too gentle."""
    a = 1.0 - math.exp(-2.0 * math.pi * cutoff / SR)
    for _ in range(poles):
        y = 0.0
        for i, x in enumerate(buf):
            y += a * (x - y)
            buf[i] = y
    return buf


def render(duration, transitions, bpm, style="lofi", root_midi=57, seed=7):
    n = int(duration * SR) + SR
    buf = [0.0] * n
    beat = 60.0 / bpm
    bar = beat * 4
    swing = beat * (0.18 if style == "lofi" else 0.0)

    prog = [(0, [0, 7, 12, 14, 16]), (-4, [0, 7, 12, 16, 19]),
            (3, [0, 7, 11, 14, 16]), (-2, [0, 7, 10, 14, 17])]
    chord_len = bar * 2

    # pad bed
    t, ci = 0.0, 0
    while t < duration:
        rt = prog[ci % len(prog)][0]
        base = root_midi + rt
        add(buf, pad([hz(base - 12), hz(base), hz(base + 7)],
                     min(chord_len + 0.5, duration - t + 0.5)), t)
        t += chord_len
        ci += 1

    rng = random.Random(seed)
    nbars = int(duration / bar) + 2
    for b in range(nbars):
        t0 = b * bar
        if t0 > duration:
            break
        rt, ivs = prog[b % len(prog)]
        base = root_midi + rt
        for hit, gain in ((0.0, 1.0), (beat * 2 + swing, 0.62)):
            at = t0 + hit
            if at > duration:
                continue
            for k, iv in enumerate(ivs[:4]):
                add(buf, rhodes(hz(base + iv), 2.6, 0.105 * gain),
                    at + rng.uniform(-0.010, 0.014))
        add(buf, bass(hz(base - 24), 1.7, 0.30), t0)
        add(buf, kick(amp=0.28), t0)
        add(buf, kick(amp=0.19), t0 + beat * 2 + swing * 0.5)
        add(buf, brush(seed=seed + b), t0 + beat * 2)
        for e in range(8):
            if e % 2 == 1:
                at = t0 + e * (beat / 2) + swing
                if at <= duration:
                    add(buf, hat(seed=seed + b * 8 + e), at)

    # a soft accent exactly on every edit transition
    for i, tt in enumerate([0.0] + list(transitions)):
        rt, ivs = prog[i % len(prog)]
        base = root_midi + rt
        for iv in ivs[:3]:
            add(buf, rhodes(hz(base + iv + 12), 2.2, 0.070), tt)

    buf = buf[:int(duration * SR)]
    v = vinyl(len(buf), seed + 3)
    for i in range(len(buf)):
        buf[i] += v[i]

    if style == "lofi":
        buf = lowpass(buf, 3100.0, poles=4)

    fi = int((1.1 if style == "lofi" else 0.45) * SR)
    for i in range(min(fi, len(buf))):
        buf[i] *= (i / fi) ** 1.6
    fo = int((3.2 if style == "lofi" else 2.6) * SR)
    for i in range(min(fo, len(buf))):
        buf[-1 - i] *= (i / fo) ** 1.3

    peak = max(abs(x) for x in buf) or 1.0
    g = (10.0 ** (-14.0 / 20.0)) / peak
    buf = [math.tanh(x * g * 1.4) / 1.4 for x in buf]

    d = int(0.012 * SR)
    return buf, d


def write_wav(path, buf, delay):
    n = len(buf)
    data = array.array("h")
    for i in range(n):
        l = buf[i]
        r = buf[i - delay] if i >= delay else 0.0
        data.append(max(-32767, min(32767, int(l * 32767))))
        data.append(max(-32767, min(32767, int(r * 0.97 * 32767))))
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    style = "lofi"
    args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--style":
            style = sys.argv[i + 1]
            i += 2
        else:
            args.append(sys.argv[i])
            i += 1
    out = args[0]
    duration = float(args[1])
    trans = [float(x) for x in args[2:]]
    lo, hi = (68.0, 92.0) if style == "lofi" else (110.0, 140.0)
    bpm, err = best_bpm(trans, lo, hi)
    print("style %s   tempo %.1f BPM   mean beat error %.3f s (%.1f frames)"
          % (style, bpm, err, err * 30))
    print("rendering %.1f s, pure stdlib, this takes 20 to 40 seconds ..."
          % duration)
    buf, delay = render(duration, trans, bpm, style)
    write_wav(out, buf, delay)
    print("wrote %s  %.2f s stereo %d Hz 16-bit" % (out, len(buf) / SR, SR))


if __name__ == "__main__":
    main()
