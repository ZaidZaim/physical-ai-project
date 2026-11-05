#!/usr/bin/env python3
"""Runs ON the PuppyPi (Raspberry Pi). The ears and mouth: listens through the WonderEcho Pro microphone, sends what it hears to the
brain, and plays the reply (and any greeting the brain wants to say) through the WonderEcho Pro speaker.

Stdlib only; uses ALSA's `arecord` and `aplay` (present on the Pi image). The Pi does no speech recognition or synthesis: the laptop
does the understanding (transcription, the Graph Wizard conversation, text-to-speech) and returns a WAV to play.

Usage on the robot (it finds the WonderEcho Pro and measures the room noise by itself):
  At boot it starts by itself: puppy-voice.service runs start_voice_auto.sh, which finds the laptop on the current Wi-Fi.
  By hand: python3 puppy_voice.py --brain http://<laptop-ip>:8000
  python3 puppy_voice.py --brain http://<laptop-ip>:8000 --test-mic       # prints the loudness it hears
  Override if needed: --mic plughw:1,0 --speaker plughw:1,0 --threshold 600   (cards: arecord -l ; aplay -l)

The robot only speaks when the brain tells it to (a greeting, or the answer to someone who just spoke) and it never listens while it
is talking. Audio leaves the robot only when someone speaks: nothing is recorded to disk.

GPT-Live (when the laptop has LIVE_VOICE=true; asked at start): the same, but over one WebSocket. Outside a conversation the robot
still sends only what someone said; once the laptop has opened a call (after "Hey PuppyPi") it streams the microphone and plays the
answer as it arrives, until the call ends. --classic forces the old turn by turn mode.
"""
import argparse
import array
import base64
import collections
import io
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
import wave

RATE, CHUNK_S = 16000, 0.03
MIN_VOICE_S = 0.35  # at least this much actual loud sound before a recording is sent: a click plus silence is not speech
KEEP_TAIL_S = 0.3   # trailing silence kept: the rest is cut before sending (less to upload and transcribe, fewer invented words)


def trim_tail(pcm, quiet_chunks):
    """Drop the trailing silence that ended the sentence, keeping KEEP_TAIL_S of it."""
    drop = max(0, quiet_chunks - int(KEEP_TAIL_S / CHUNK_S)) * int(RATE * CHUNK_S) * 2
    return pcm[:len(pcm) - drop] if drop else pcm
DEFAULT_THRESHOLD = 800.0  # used when the start-up room measurement was not quiet
PREFER = re.compile(r"wonder|echo|hiwonder|usb|uac", re.I)  # the WonderEcho Pro is a USB sound card
SKIP = re.compile(r"bcm2835|vc4|hdmi|headphones", re.I)       # the Pi's own outputs (no microphone, not the robot speaker)


def pick_card(listing):
    """'plughw:N,D' for the WonderEcho Pro (or another USB sound card) in `arecord -l` / `aplay -l` output, else None."""
    cards = re.findall(r"^card (\d+): ([^\n]*?), device (\d+):", listing or "", re.M)
    good = [c for c in cards if PREFER.search(c[1])] or [c for c in cards if not SKIP.search(c[1])]
    return "plughw:%s,%s" % (good[0][0], good[0][2]) if good else None


AEC_ARGS = "noise_suppression=1 high_pass_filter=1 analog_gain_control=0 digital_gain_control=1"


def setup_echo_cancel():
    """Record and play through PulseAudio's WebRTC echo canceller (module-echo-cancel): it removes most of the robot's own voice
    from its microphone and suppresses the hall's noise. Measured on this robot with speech at booth volume: its own voice 24260 ->
    2976 (RMS), a quiet room 188 instead of 1000-2500. True when it is on; then the mic and speaker are ALSA's "pulse" device."""
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())  # the systemd service has no session environment

    def pactl(*args):
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10)
    try:
        if pactl("info").returncode != 0:
            print("echo cancelling: no PulseAudio running")
            return False
        names = lambda kind: [l.split("\t")[1] for l in pactl("list", "short", kind).stdout.splitlines() if "\t" in l]
        usb = lambda items: [n for n in items if PREFER.search(n) and not n.endswith(".monitor") and not n.startswith("ec_")]
        src, snk = usb(names("sources")), usb(names("sinks"))
        if not src or not snk:
            print("echo cancelling: the WonderEcho Pro is not in PulseAudio")
            return False
        pactl("unload-module", "module-echo-cancel")  # from an earlier start
        r = pactl("load-module", "module-echo-cancel", "aec_method=webrtc", "source_name=ec_source", "sink_name=ec_sink",
                  "source_master=" + src[0], "sink_master=" + snk[0], "rate=16000", "channels=1", 'aec_args="%s"' % AEC_ARGS)
        if r.returncode != 0:
            print("echo cancelling: PulseAudio refused it:", (r.stderr or r.stdout).strip()[:120])
            return False
        pactl("set-default-source", "ec_source")
        pactl("set-default-sink", "ec_sink")
        print("echo cancelling: on (WebRTC: the robot's own voice and the hall's noise are taken out of the microphone)")
        return True
    except Exception as e:
        print("echo cancelling: not available (%s)" % e)
        return False


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


class MiniWS:
    """A minimal WebSocket client (ws:// only, stdlib): the Pi image has no websocket package and the voice client stays stdlib only."""

    def __init__(self, url, headers=None, timeout=10.0):
        u = urllib.parse.urlparse(url)
        host, port = u.hostname, u.port or 80
        path = (u.path or "/") + ("?" + u.query if u.query else "")
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
               "Sec-WebSocket-Version: 13\r\n" % (path, host, port, key))
        for k, v in (headers or {}).items():
            req += "%s: %s\r\n" % (k, v)
        self.sock.sendall((req + "\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("the laptop closed the connection during the handshake")
            resp += chunk
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n")[0].decode(errors="ignore")
        if " 101 " not in status + " ":
            raise ConnectionError("no WebSocket: " + status)
        self.sock.settimeout(None)
        self.lock = threading.Lock()
        self.close_code = None

    @staticmethod
    def _mask(data, mask):
        if not data:
            return data
        m = (mask * (len(data) // 4 + 1))[:len(data)]
        return (int.from_bytes(data, "big") ^ int.from_bytes(m, "big")).to_bytes(len(data), "big")

    def _send(self, opcode, data):
        n = len(data)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        with self.lock:
            self.sock.sendall(head + mask + self._mask(data, mask))

    def send_bytes(self, data):
        self._send(0x2, data)

    def send_text(self, text):
        self._send(0x1, text.encode())

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("the laptop closed the connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        """(opcode, payload) of the next whole message: 1 text, 2 binary. Answers pings; raises ConnectionError on close."""
        parts, first_op = [], None
        while True:
            b1, b2 = self._read(2)
            op, n = b1 & 0x0F, b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b2 & 0x80 else None
            data = self._read(n)
            if mask:
                data = self._mask(data, mask)
            if op == 0x9:
                self._send(0xA, data)
                continue
            if op == 0xA:
                continue
            if op == 0x8:
                self.close_code = struct.unpack(">H", data[:2])[0] if len(data) >= 2 else 1005
                raise ConnectionError("closed by the laptop (%s)" % self.close_code)
            if op != 0x0:
                first_op = op
            parts.append(data)
            if b1 & 0x80:
                return first_op, b"".join(parts)

    def close(self):
        try:
            self._send(0x8, struct.pack(">H", 1000))
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


class LiveOff(Exception):
    """The laptop does not do GPT-Live calls (LIVE_VOICE is off): use the classic turns."""


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
                    pcm, speech, quiet, started, loud_s, voiced = trim_tail(bytes(speech), quiet), bytearray(), 0, False, voiced * CHUNK_S, 0
                    if loud_s >= MIN_VOICE_S:  # footsteps, clicks, a cough: not sent (each became a "Thanks for watching!" phantom)
                        self.talking.set()  # stop listening until the answer has been played
                        print("thinking...")
                        threading.Thread(target=self.turn, args=(pcm,), daemon=True).start()

    def measure(self, mic, n):
        """The speech threshold from two quiet seconds of room noise (unless --threshold was given)."""
        if self.a.threshold is not None:
            return
        print("Measuring room noise for 2 seconds: please stay quiet...")
        levels = []
        for _ in range(int(2.0 / CHUNK_S)):
            b = mic.stdout.read(n)
            if not b or len(b) < n:
                raise SystemExit("microphone stopped: check --mic (arecord -l)")
            levels.append(rms(b))
        self.a.threshold = auto_threshold(levels)
        print("Speech threshold set to %.0f (room noise ~%.0f). Ready: say Hey PuppyPi!" % (self.a.threshold, max(levels) if levels else 0))

    def live(self):
        """GPT-Live mode: one WebSocket to the laptop. Raises LiveOff when the laptop says it is not doing live calls."""
        url = "ws" + self.a.brain[len("http"):] + "/api/voice/live"
        try:
            ws = MiniWS(url, {"X-Brain-Token": self.a.token} if self.a.token else None)
        except ConnectionError as e:
            if "404" in str(e) or "403" in str(e):
                raise LiveOff(str(e))
            raise
        print("GPT-Live: connected to", url)
        ws.send_text(json.dumps({"type": "hello", "aec": bool(getattr(self.a, "aec", False))}))  # may it hear visitors while it talks?
        # stderr: between sentences the stream pauses and aplay prints "underrun!!!" (harmless; it filled the log)
        player = subprocess.Popen(["aplay", "-q", "-D", self.a.speaker, "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "1"],
                                  stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        mic = subprocess.Popen(["arecord", "-q", "-D", self.a.mic, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
                               stdout=subprocess.PIPE)
        state = {"call": False, "play_until": 0.0, "error": None}

        def receive():
            try:
                while True:
                    op, data = ws.recv()
                    if op == 2 and data[:1] == b"A":
                        pcm = data[1:]
                        player.stdin.write(pcm)
                        player.stdin.flush()
                        if rms(pcm[:len(pcm) // 2 * 2]) > 200:
                            state["play_until"] = max(state["play_until"], time.time()) + len(pcm) / (RATE * 2.0)
                    elif op == 1:
                        m = json.loads(data.decode())
                        if m.get("type") == "call":
                            state["call"] = bool(m.get("open"))
                            print("call", "open: talking to GPT-Live" if state["call"] else "closed: say Hey PuppyPi to start again")
                        elif m.get("type") == "say":
                            print("say:", m.get("text", ""))
            except Exception as e:
                state["error"] = e

        threading.Thread(target=receive, daemon=True).start()
        n = int(RATE * CHUNK_S) * 2
        try:
            self.measure(mic, n)
            floor, speech, quiet, started, voiced = 0.0, bytearray(), 0, False, 0
            end_quiet = int(self.a.live_silence / CHUNK_S)
            pre = collections.deque(maxlen=int(0.3 / CHUNK_S))  # the start of the first word, from before it was loud enough
            streaming = False  # this sentence is being streamed (b"S"): GPT-Live hears it while it is said
            report, peak = time.time() + 10, 0.0
            while True:
                if state["error"] is not None:
                    if ws.close_code == 4001:
                        raise LiveOff("the laptop has LIVE_VOICE off")
                    raise ConnectionError(str(state["error"]))
                buf = mic.stdout.read(n)
                if not buf or len(buf) < n:
                    raise SystemExit("microphone stopped: check --mic (arecord -l)")
                if state["call"]:  # in a call: stream everything (the laptop mutes it while the robot talks)
                    ws.send_bytes(b"M" + buf)
                    speech, quiet, started, voiced, streaming = bytearray(), 0, False, 0, False
                    continue
                if time.time() < state["play_until"] + 0.5:  # a greeting or the end of a goodbye is playing: that is not a visitor
                    speech, quiet, started, voiced, streaming = bytearray(), 0, False, 0, False
                    continue
                level = rms(buf)  # outside a call: only what someone says is sent (the laptop checks the wake word)
                peak = max(peak, level)
                if time.time() > report:  # what the microphone hears: the first thing to check when "Hey PuppyPi" is not heard
                    print("mic: loudest %.0f in the last 10 s (speech counts above %.0f; room ~%.0f)" % (
                        peak, max(self.a.threshold, floor * self.a.live_ratio), floor))
                    report, peak = time.time() + 10, 0.0
                floor = level if floor == 0 else (floor * 0.98 + level * 0.02 if not started else floor)
                # GPT-Live decides whether it was the wake word, so speech may count more readily than in the classic mode: in the
                # loud hall the room was ~2400 and a visitor close by 3800 to 7000, below the classic 3x the room.
                loud = level > max(self.a.threshold, floor * self.a.live_ratio)
                if loud:
                    if not started:
                        speech = bytearray(b"".join(pre))
                    started, quiet, voiced = True, 0, voiced + 1
                elif started:
                    quiet += 1
                if not started:
                    pre.append(buf)
                    continue
                speech += buf
                if streaming:
                    ws.send_bytes(b"S" + buf)
                elif voiced * CHUNK_S >= MIN_VOICE_S:  # real speech, not a click: stream it from its start
                    ws.send_bytes(b"S" + bytes(speech))
                    streaming = True
                    print("someone is speaking: streaming it to the laptop")
                if quiet >= end_quiet or len(speech) > RATE * 2 * self.a.max_s:
                    if streaming:
                        ws.send_bytes(b"E")
                    speech, quiet, started, voiced, streaming = bytearray(), 0, False, 0, False
                    pre.clear()
        finally:
            for proc in (mic, player):
                try:
                    proc.kill()
                except Exception:
                    pass
            ws.close()

    def turn(self, pcm):
        if time.time() - self.last_turn > self.a.forget_after:
            self.encounter = None  # a new person: the brain starts a fresh conversation (and asks for consent again)
        try:
            t_sent = time.time()
            r = post_audio(self.a.brain, wav_bytes(pcm), self.a.token, self.encounter)
            if r.status != 200:
                print("heard nothing clear", ("(" + urllib.parse.unquote(r.headers.get("X-Heard", "")) + ")") if r.headers.get("X-Heard") else "")
                return
            self.encounter = r.headers.get("X-Encounter") or self.encounter
            self.last_turn = time.time()
            print("heard:", urllib.parse.unquote(r.headers.get("X-Heard", "")), "| say:", urllib.parse.unquote(r.headers.get("X-Reply", "")),
                  "| brain " + urllib.parse.unquote(r.headers.get("X-Timing", "")) if r.headers.get("X-Timing") else "",
                  "| round trip %.1fs" % (time.time() - t_sent))
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
    ap.add_argument("--silence", type=float, default=1.5, help="seconds of quiet that end a sentence (people pause to think)")
    ap.add_argument("--max-s", type=float, default=15.0, help="longest single utterance")
    ap.add_argument("--forget-after", type=float, default=120.0, help="seconds of quiet after which the next voice is treated as a new visitor")
    ap.add_argument("--test-mic", action="store_true")
    ap.add_argument("--classic", action="store_true", help="never use GPT-Live calls, even when the laptop offers them")
    ap.add_argument("--no-echo-cancel", dest="echo_cancel", action="store_false",
                    help="use the sound card directly instead of PulseAudio's WebRTC echo canceller")
    ap.add_argument("--live-ratio", type=float, default=1.7,
                    help="GPT-Live mode: speech counts above this many times the room noise (the classic mode uses 3)")
    ap.add_argument("--live-silence", type=float, default=0.6,
                    help="GPT-Live mode: quiet that ends the wake sentence (the rest of the question streams into the call)")
    a = ap.parse_args()
    a.brain = a.brain.rstrip("/")
    a.aec = a.echo_cancel and a.mic == "auto" and a.speaker == "auto" and not a.test_mic and setup_echo_cancel()
    if a.aec:
        a.mic = a.speaker = "pulse"  # ALSA's PulseAudio device: its defaults are now the echo-cancelled source and sink
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
    live = False
    if not a.test_mic and not a.classic:
        try:
            live = bool(json.loads(get(a.brain, "/api/voice/mode", a.token).read().decode()).get("live"))
        except Exception:
            live = False  # an older laptop app: no /api/voice/mode
    while live:
        try:
            v.live()
        except LiveOff as e:
            print("GPT-Live is off on the laptop (%s): classic turns" % e)
            live = False
        except (ConnectionError, OSError) as e:
            print("GPT-Live connection lost (%s): reconnecting in 3 s" % e)
            time.sleep(3)
    if not a.test_mic:
        threading.Thread(target=v.greetings, daemon=True).start()
    v.listen()


if __name__ == "__main__":
    main()
