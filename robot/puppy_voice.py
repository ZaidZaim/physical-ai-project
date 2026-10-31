#!/usr/bin/env python3
"""Runs ON the PuppyPi (Raspberry Pi). The ears and mouth: listens through the WonderEcho Pro microphone, sends what it hears to the
brain, and plays the reply (and any greeting the brain wants to say) through the WonderEcho Pro speaker.

Stdlib only; uses ALSA's `arecord` and `aplay` (present on the Pi image). The Pi does no speech recognition or synthesis: the laptop
does the understanding (transcription, the Graph Wizard conversation, text-to-speech) and returns a WAV to play.

Usage on the robot:
  arecord -l ; aplay -l                       # find the WonderEcho Pro card, e.g. "card 1: ... USB Audio"
  python3 puppy_voice.py --brain http://<laptop-ip>:8000 --mic plughw:1,0 --speaker plughw:1,0
  python3 puppy_voice.py --brain http://<laptop-ip>:8000 --test-mic       # prints the loudness it hears: tune --threshold

The robot only speaks when the brain tells it to (a greeting, or the answer to someone who just spoke) and it never listens while it
is talking. Audio leaves the robot only when someone speaks: nothing is recorded to disk.
"""
import argparse
import array
import io
import json
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
import wave

RATE, CHUNK_S = 16000, 0.03


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
        self.talking = threading.Event()  # set while the speaker is playing: the microphone is ignored
        self.encounter, self.last_turn = None, 0.0

    def play(self, wav):
        self.talking.set()
        try:
            subprocess.run(["aplay", "-q", "-D", self.a.speaker, "-"], input=wav, timeout=120)
        finally:
            time.sleep(0.4)  # let the room's echo die down before listening again
            self.talking.clear()

    def greetings(self):
        """Poll the brain for something it wants to say (someone walked up)."""
        while True:
            try:
                if not self.talking.is_set():
                    r = get(self.a.brain, "/api/voice/next", self.a.token)
                    if r.status == 200:
                        print("say:", urllib.parse.unquote(r.headers.get("X-Reply", "")))
                        self.play(r.read())
            except Exception as e:
                print("greeting poll:", e)
                time.sleep(3)
            time.sleep(1.0)

    def listen(self):
        mic = subprocess.Popen(["arecord", "-q", "-D", self.a.mic, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
                               stdout=subprocess.PIPE)
        n = int(RATE * CHUNK_S) * 2
        floor, speech, quiet, started = 0.0, bytearray(), 0, False
        end_quiet = int(self.a.silence / CHUNK_S)
        while True:
            buf = mic.stdout.read(n)
            if not buf or len(buf) < n:
                raise SystemExit("microphone stopped: check --mic (arecord -l)")
            if self.talking.is_set():
                speech, quiet, started = bytearray(), 0, False
                continue
            level = rms(buf)
            if self.a.test_mic:
                print("level %6.0f %s" % (level, "#" * int(level / 200)))
                continue
            floor = level if floor == 0 else (floor * 0.98 + level * 0.02 if not started else floor)
            loud = level > max(self.a.threshold, floor * 3)
            if loud:
                started, quiet = True, 0
            elif started:
                quiet += 1
            if started:
                speech += buf
                if quiet >= end_quiet or len(speech) > RATE * 2 * self.a.max_s:
                    pcm, speech, quiet, started = bytes(speech), bytearray(), 0, False
                    if len(pcm) > RATE * 2 * 0.5:  # ignore clicks and coughs shorter than half a second
                        threading.Thread(target=self.turn, args=(pcm,), daemon=True).start()

    def turn(self, pcm):
        if time.time() - self.last_turn > self.a.forget_after:
            self.encounter = None  # a new person: the brain starts a fresh conversation (and asks for consent again)
        try:
            r = post_audio(self.a.brain, wav_bytes(pcm), self.a.token, self.encounter)
            if r.status != 200:
                return  # nothing intelligible
            self.encounter = r.headers.get("X-Encounter") or self.encounter
            self.last_turn = time.time()
            print("heard:", urllib.parse.unquote(r.headers.get("X-Heard", "")), "| say:", urllib.parse.unquote(r.headers.get("X-Reply", "")))
            self.play(r.read())
        except Exception as e:
            print("turn failed:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--mic", default="default", help="ALSA capture device, e.g. plughw:1,0")
    ap.add_argument("--speaker", default="default", help="ALSA playback device, e.g. plughw:1,0")
    ap.add_argument("--threshold", type=float, default=600.0, help="loudness (16-bit RMS) that counts as speech; see --test-mic")
    ap.add_argument("--silence", type=float, default=0.9, help="seconds of quiet that end a sentence")
    ap.add_argument("--max-s", type=float, default=15.0, help="longest single utterance")
    ap.add_argument("--forget-after", type=float, default=120.0, help="seconds of quiet after which the next voice is treated as a new visitor")
    ap.add_argument("--test-mic", action="store_true")
    a = ap.parse_args()
    v = Voice(a)
    if not a.test_mic:
        threading.Thread(target=v.greetings, daemon=True).start()
    v.listen()


if __name__ == "__main__":
    main()
