"""Sounds the robot plays, all synthesised here from scratch (no recordings, no samples, no existing melodies), so they are free
to play at a public booth.

The spin jingle: a few seconds of upbeat music the robot plays while it spins.

Made here from scratch (no recording, no sample, no existing melody), so it is free to play at a public booth: a rising whoosh,
a four-on-the-floor kick, off-beat hats and a bright arpeggio over A minor, F, C, G at 150 BPM, four bars (6.4 s).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

RATE = 16000          # the robot's speaker stream: 16 kHz mono 16-bit
BPM = 150.0
BEAT = 60.0 / BPM
BARS = 4
CHORDS = ((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62))  # MIDI: A minor, F, C, G (one bar each)
PATTERN = (0, 1, 2, 1, 0, 1, 2, 3)  # arpeggio, eighth notes; 3 = the chord root an octave up


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _pluck(midi: float, seconds: float) -> np.ndarray:
    """A bright synth pluck: a square-ish tone (odd harmonics) that darkens as it decays."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f = _hz(midi)
    tone = sum(np.sin(2 * np.pi * f * k * t) / k * np.exp(-t * k * 6) for k in (1, 3, 5, 7))
    return tone * _env(n, 0.004, 0.18)


def _kick(seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f = 45 + 110 * np.exp(-t * 30)  # the pitch drops fast: the "thump"
    return np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(n, 0.001, 0.12)


def _hat(seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * RATE)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # crude high-pass: keeps the hiss, drops the rumble
    return noise * _env(n, 0.001, 0.025)


def _bass(midi: float, seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f = _hz(midi - 12)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * _env(n, 0.005, 0.25)


def _whoosh(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """A rising sweep: noise filtered ever brighter, getting louder, for the start of the spin."""
    n = int(seconds * RATE)
    noise = rng.standard_normal(n)
    out, y = np.empty(n), 0.0
    for i in range(n):  # one-pole low-pass whose cutoff rises from dull to bright
        a = 0.02 + 0.5 * (i / n) ** 2
        y += a * (noise[i] - y)
        out[i] = y
    t = np.arange(n) / RATE
    sweep = np.sin(2 * np.pi * np.cumsum(300 + 1500 * (t / seconds) ** 2) / RATE)
    return (out * 1.5 + 0.25 * sweep) * np.linspace(0.0, 1.0, n) ** 1.5


def _add(buf: np.ndarray, sound: np.ndarray, at: float, gain: float) -> None:
    i = int(at * RATE)
    j = min(len(buf), i + len(sound))
    if i < j:
        buf[i:j] += gain * sound[: j - i]


@lru_cache(maxsize=1)
def spin_jingle() -> bytes:
    """The jingle as 16 kHz mono 16-bit PCM (what the robot's speaker stream takes). Always the same; made once."""
    rng = np.random.default_rng(7)
    intro = 2 * BEAT  # half a bar of whoosh, then the beat comes in
    total = intro + BARS * 4 * BEAT + 0.4
    buf = np.zeros(int(total * RATE))
    _add(buf, _whoosh(intro, rng), 0.0, 0.35)
    for bar, chord in enumerate(CHORDS[:BARS]):
        t0 = intro + bar * 4 * BEAT
        for beat in range(4):
            _add(buf, _kick(0.3), t0 + beat * BEAT, 0.9)
            _add(buf, _hat(0.08, rng), t0 + (beat + 0.5) * BEAT, 0.25)
            _add(buf, _bass(chord[0] - (12 if beat % 2 else 0), BEAT * 0.9), t0 + beat * BEAT, 0.45)
        for k, p in enumerate(PATTERN):
            note = chord[0] + 12 if p == 3 else chord[p]
            _add(buf, _pluck(note + 12, BEAT * 0.6), t0 + k * BEAT / 2, 0.3)
    # the last hit: the whole chord, letting ring
    end = intro + BARS * 4 * BEAT
    for note in CHORDS[0]:
        _add(buf, _pluck(note + 12, 0.4), end - BEAT / 2, 0.25)
    fade = int(0.15 * RATE)
    buf[-fade:] *= np.linspace(1.0, 0.0, fade)
    buf *= 0.7 / max(1e-9, float(np.max(np.abs(buf))))  # loud on the small speaker, never clipping
    return (buf * 32767).astype("<i2").tobytes()


def _honk(seconds: float) -> np.ndarray:
    """One squeeze of a rubber-bulb bicycle horn: a buzzy reed tone that jumps up in pitch as the bulb is squeezed and sags as
    it empties, through a narrow "nasal" resonance."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f = 330 + 90 * (1 - np.exp(-t * 60)) - 40 * (t / seconds) ** 2  # squeeze: up fast; the bulb empties: down a little
    f = f * (1 + 0.012 * np.sin(2 * np.pi * 23 * t))                    # the reed flutters
    phase = 2 * np.pi * np.cumsum(f) / RATE
    reed = sum(np.sin(k * phase) / k ** 0.7 for k in range(1, 14))      # many harmonics: buzzy, not pure
    y, out = 0.0, np.empty(n)                                           # resonance around 1.1 kHz (the horn's bell)
    r, w = 0.97, 2 * np.pi * 1100 / RATE
    a1, a2 = 2 * r * np.cos(w), -r * r
    y1 = y2 = 0.0
    for i in range(n):
        y = reed[i] + a1 * y1 + a2 * y2
        y2, y1 = y1, y
        out[i] = y
    mix = 0.35 * reed / np.max(np.abs(reed)) + out / np.max(np.abs(out))
    attack, release = int(0.012 * RATE), int(0.05 * RATE)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return np.tanh(2.2 * mix * env)  # pushed into soft clipping, like the real thing


@lru_cache(maxsize=1)
def horn() -> bytes:
    """A bicycle horn, honk-honk (16 kHz mono 16-bit PCM, about 0.6 s)."""
    buf = np.zeros(int(0.62 * RATE))
    _add(buf, _honk(0.22), 0.0, 1.0)
    _add(buf, _honk(0.26), 0.32, 1.0)
    buf *= 0.75 / max(1e-9, float(np.max(np.abs(buf))))
    return (buf * 32767).astype("<i2").tobytes()


SOUNDS = {"spin": spin_jingle, "horn": horn}  # what LiveBridge.sound can play


def wav_bytes(pcm: bytes) -> bytes:
    """A playable WAV of the jingle (to listen to it on the laptop)."""
    import io
    import wave

    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return b.getvalue()
