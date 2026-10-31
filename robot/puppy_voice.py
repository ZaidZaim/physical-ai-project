#!/usr/bin/env python3
"""Runs ON the PuppyPi (Raspberry Pi). The ears and mouth: listens through the WonderEcho Pro microphone, sends what it hears to the
brain, and plays the reply (and any greeting the brain wants to say) through the WonderEcho Pro speaker.

Stdlib only; uses ALSA's `arecord` and `aplay` (present on the Pi image). The Pi does no speech recognition or synthesis: the laptop
does the understanding (transcription, the Graph Wizard conversation, text-to-speech) and returns a WAV to play.

Usage on the robot (it finds the WonderEcho Pro and measures the room noise by itself):
  python3 puppy_voice.py --brain http://<laptop-ip>:8000
  python3 puppy_voice.py --brain http://<laptop-ip>:8000 --test-mic       # prints the loudness it hears
  Override if needed: --mic plughw:1,0 --speaker plughw:1,0 --threshold 600   (cards: arecord -l ; aplay -l)

The robot only speaks when the brain tells it to (a greeting, or the answer to someone who just spoke) and it never listens while it
is talking. Audio leaves the robot only when someone speaks: nothing is recorded to disk.
"""
import argparse
import array
import io
import json
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
import wave

RATE, CHUNK_S = 16000, 0.03
MIN_VOICE_S = 0.35  # at least this much actual loud sound before a recording is sent: a click plus silence is not speech
DEFAULT_THRESHOLD = 800.0  # used when the start-up room measurement was not quiet
PREFER = re.compile(r"wonder|echo|hiwonder|usb|uac", re.I)  # the WonderEcho Pro is a USB sound card
SKIP = re.compile(r"bcm2835|vc4|hdmi|headphones", re.I)       # the Pi's own outputs (no microphone, not the robot speaker)


def pick_card(listing):
    """'plughw:N,D' for the WonderEcho Pro (or another USB sound card) in `arecord -l` / `aplay -l` output, else None."""
    cards = re.findall(r"^card (\d+): ([^\n]*?), device (\d+):", listing or "", re.M)
    good = [c for c in cards if PREFER.search(c[1])] or [c for c in cards if not SKIP.search(c[1])]
    return "plughw:%s,%s" % (good[0][0], good[0][2]) if good else None


def find_device(tool):
    try:
        out = subprocess.run([tool, "-l"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    return pick_card(out)


def auto_threshold(levels):
    """Speech threshold from a couple of seconds of room noise: well above the noise, never absurdly low or high. If the 'quiet'
    measurement was loud (someone talked, the speaker played), it is not room noise: fall back to a safe default."""
    quiet = sorted(levels)[int(len(levels) * 0.9)] if levels else 0.0
    if quiet > 2000:
        return DEFAULT_THRESHOLD
    return min(4000.0, max(300.0, quiet * 3.0))


def rms(buf):
    a = array.array("h", buf)
    return (sum(x * x for x in a) / max(1, len(a))) ** 0.5


def wav_bytes(pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return b.getvalue()


def post_audio(brain, wav, token, encounter):
    b = uuid.uuid4().hex
    parts = []
    if encounter:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="encounter_id"\r\n\r\n{encounter}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="s.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + wav + f"\r\n--{b}--\r\n".encode())
    req = urllib.request.Request(brain + "/api/voice/turn", data=b"".join(parts), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={b}")
    if token:
        req.add_header("X-Brain-Token", token)
    return urllib.request.urlopen(req, timeout=60)


def get(brain, path, token):
    req = urllib.request.Request(brain + path)
    if token:
        req.add_header("X-Brain-Token", token)
    return urllib.request.urlopen(req, timeout=30)


class Voice:
    def __init__(self, a):
        self.a = a
        # set from the end of a visitor's sentence until the reply has been played, and while a greeting plays: the microphone is
        # ignored meanwhile. Listening while the brain still thinks sent the next sentence without the conversation id (a new
        # conversation that forgot the visitor) and made two replies fight over the speaker ("Device or resource busy").
        self.talking = threading.Event()
        self.speaker = threading.Lock()  # one sound at a time
        self.encounter, self.last_turn = None, 0.0

    def play(self, wav):
        with self.speaker:
            for attempt in (1, 2):  # the USB card can still be closing from the previous sound
                r = subprocess.run(["aplay", "-q", "-D", self.a.speaker, "-"], input=wav, timeout=120, stderr=subprocess.PIPE)
                if r.returncode == 0:
                    break
                print("speaker busy, retrying:", (r.stderr or b"").decode(errors="ignore").strip()[:120])
                time.sleep(0.5)
            time.sleep(0.4)  # let the room's echo die down before listening again

    def greetings(self):
        """Poll the brain for something it wants to say (someone walked up)."""
        while True:
            try:
                if not self.talking.is_set():
                    r = get(self.a.brain, "/api/voice/next", self.a.token)
                    if r.status == 200:
                        print("say:", urllib.parse.unquote(r.headers.get("X-Reply", "")))
                        self.talking.set()
                        try:
                            self.play(r.read())
                        finally:
                            self.talking.clear()
            except Exception as e:
                print("greeting poll:", e)
                time.sleep(3)
            time.sleep(1.0)

    def listen(self):
        mic = subprocess.Popen(["arecord", "-q", "-D", self.a.mic, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
                               stdout=subprocess.PIPE)
        n = int(RATE * CHUNK_S) * 2
        floor, speech, quiet, started, voiced = 0.0, bytearray(), 0, False, 0
        end_quiet = int(self.a.silence / CHUNK_S)
        if self.a.threshold is None and not self.a.test_mic:  # measure the room first: stay quiet for two seconds
            print("Measuring room noise for 2 seconds: please stay quiet...")
            levels = []
            for _ in range(int(2.0 / CHUNK_S)):
                b = mic.stdout.read(n)
                if not b or len(b) < n:
                    raise SystemExit("microphone stopped: check --mic (arecord -l)")
                levels.append(rms(b))
            self.a.threshold = auto_threshold(levels)
            noisy = self.a.threshold == DEFAULT_THRESHOLD and levels and sorted(levels)[int(len(levels) * 0.9)] > 2000
            print("Speech threshold set to %.0f (room noise ~%.0f%s). Ready: talk to me!" % (
                self.a.threshold, max(levels) if levels else 0, ": too loud to be the room, using the default" if noisy else ""))
        while True:
            buf = mic.stdout.read(n)
            if not buf or len(buf) < n:
                raise SystemExit("microphone stopped: check --mic (arecord -l)")
            if self.talking.is_set():
                speech, quiet, started, voiced = bytearray(), 0, False, 0
                continue
            level = rms(buf)
            if self.a.test_mic:
                print("level %6.0f %s" % (level, "#" * int(level / 200)))
                continue
            floor = level if floor == 0 else (floor * 0.98 + level * 0.02 if not started else floor)
            loud = level > max(self.a.threshold, floor * 3)
            if loud:
                started, quiet, voiced = True, 0, voiced + 1
            elif started:
                quiet += 1
            if started:
                speech += buf
                if quiet >= end_quiet or len(speech) > RATE * 2 * self.a.max_s:
                    pcm, speech, quiet, started, loud_s, voiced = bytes(speech), bytearray(), 0, False, voiced * CHUNK_S, 0
                    if loud_s >= MIN_VOICE_S:  # footsteps, clicks, a cough: not sent (each became a "Thanks for watching!" phantom)
                        self.talking.set()  # stop listening until the answer has been played
                        print("thinking...")
                        threading.Thread(target=self.turn, args=(pcm,), daemon=True).start()

    def turn(self, pcm):
        if time.time() - self.last_turn > self.a.forget_after:
            self.encounter = None  # a new person: the brain starts a fresh conversation (and asks for consent again)
        try:
            r = post_audio(self.a.brain, wav_bytes(pcm), self.a.token, self.encounter)
            if r.status != 200:
                print("heard nothing clear", ("(" + urllib.parse.unquote(r.headers.get("X-Heard", "")) + ")") if r.headers.get("X-Heard") else "")
                return
            self.encounter = r.headers.get("X-Encounter") or self.encounter
            self.last_turn = time.time()
            print("heard:", urllib.parse.unquote(r.headers.get("X-Heard", "")), "| say:", urllib.parse.unquote(r.headers.get("X-Reply", "")))
            self.play(r.read())
        except Exception as e:
            print("turn failed:", e)
        finally:
            self.talking.clear()  # listen again


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--mic", default="auto", help="ALSA capture device, e.g. plughw:1,0 (auto: the WonderEcho Pro / USB card)")
    ap.add_argument("--speaker", default="auto", help="ALSA playback device, e.g. plughw:1,0 (auto: the WonderEcho Pro / USB card)")
    ap.add_argument("--threshold", type=float, default=None, help="loudness (16-bit RMS) that counts as speech (default: measured at start)")
    ap.add_argument("--silence", type=float, default=1.8, help="seconds of quiet that end a sentence (people pause to think)")
    ap.add_argument("--max-s", type=float, default=15.0, help="longest single utterance")
    ap.add_argument("--forget-after", type=float, default=120.0, help="seconds of quiet after which the next voice is treated as a new visitor")
    ap.add_argument("--test-mic", action="store_true")
    a = ap.parse_args()
    a.brain = a.brain.rstrip("/")
    for name, tool in (("mic", "arecord"), ("speaker", "aplay")):
        if getattr(a, name) == "auto":
            dev = find_device(tool)
            setattr(a, name, dev or "default")
            print("%s: %s" % (name, dev or "default (no USB sound card found: is the WonderEcho Pro plugged in? see `%s -l`)" % tool))
    try:  # can the robot reach the laptop at all? (Windows Firewall blocks it on a Public network unless a rule allows it)
        get(a.brain, "/api/booth", a.token).read()
        print("laptop reachable:", a.brain)
    except Exception as e:
        raise SystemExit("cannot reach the laptop at %s (%s): is the app running there, and does the Windows Firewall allow port 8000?" % (a.brain, e))
    v = Voice(a)
    if not a.test_mic:
        threading.Thread(target=v.greetings, daemon=True).start()
    v.listen()


if __name__ == "__main__":
    main()
