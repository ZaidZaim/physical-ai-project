"""GPT-Live: a booth conversation as one full-duplex voice stream (OpenAI gpt-live-1), instead of record, transcribe, chat, speak.

Measured on this laptop: about 1 s from the end of a question to the first sound, against 5 to 7 s for the classic pipeline (1.5 s
waiting for silence on the robot, then transcription, the chat model and the whole reply synthesised before anything plays).

How it runs:
- The robot keeps one WebSocket to the laptop (/api/voice/live). Outside a call it sends only what someone said (one utterance per
  message, as before); during a call it streams its microphone, and it plays whatever audio comes back.
- The laptop keeps the OpenAI key and the booth rules. It checks each utterance for "Hey PuppyPi" the cheap way (transcription and the
  same wake check as before), and only then opens a GPT-Live call. The call closes on goodbye, the Booth tab's End conversation, or
  WAKE_WINDOW_S of quiet: GPT-Live is billed per minute.
- Tricks, questions about the room and a new visitor are function tools of GPT-Live's delegated backend (LIVE_BACKEND_MODEL).
- The live transcripts go into the same Encounter as a classic conversation, so the anonymous booth notes are made the same way.
- The robot's microphone hears its own speaker, so while it talks (and briefly after) silence is sent instead: no barge-in yet.

Robot frames, binary: b"U" + PCM (one utterance, outside a call), b"M" + PCM (microphone stream, during a call). Laptop frames:
binary b"A" + PCM (play this), text {"type": "call", "open": bool} and {"type": "say", "text": ...}. PCM is mono 16-bit 16 kHz.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import threading
import time
import uuid
import wave

import numpy as np

from .booth import TOOLS, says_goodbye
from .voice import fix_terms, heard_nothing
from .wizard import system_prompt

RATE = 16000
BPS = RATE * 2              # bytes per second of 16-bit mono audio
URL = "wss://api.openai.com/v1/live/sessions"
LOUD_OUT = 200.0            # output audio louder than this (RMS) is the robot talking; GPT-Live also streams near-silence
ECHO_TAIL_S = 0.6           # the microphone stays muted this long after the robot's last audible sound (its player buffers)
BOT_TURN_GAP_S = 1.2        # no new transcript for this long: the turns so far are committed (GPT-Live sends no end-of-turn event)
MAX_CALL_S = 600.0          # a safety net: no call runs longer than ten minutes

BACKEND = ("You are the backend of {robot}, a robot dog at the Neo4j booth. Use the tools: do_trick when someone asks for a trick, "
           "ask_world for anything about the room or what the robot can see, new_visitor when a different person has taken over. "
           "Reply with one short plain sentence the voice can say. Never ask for or keep personal data.")


def _plain(s: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in (s or "").lower()).split())


def merge_turns(pieces: list[tuple[float, int, str, str]]) -> list[dict]:
    """Transcript pieces (start_ms, n, role, text) -> turns. Ordered by start_ms, each run of one speaker is a turn. A scrap of at
    most two words inside the other side's speech (the robot's "I'd" before the visitor's last word) joins its own speaker's next
    turn, so the visitor's sentence stays whole. Sound labels such as "[hum]" are dropped."""
    runs: list[dict] = []
    for _, _, role, text in sorted(pieces):
        text = re.sub(r"\[[^\]]*\]", " ", text)
        if runs and runs[-1]["role"] == role:
            runs[-1]["content"] += text
        elif text.strip():
            runs.append({"role": role, "content": text})
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            r = runs[i]
            if len(r["content"].split()) <= 2 and runs[i - 1]["role"] == runs[i + 1]["role"] != r["role"]:
                later = next((x for x in runs[i + 2:] if x["role"] == r["role"]), None)
                if later is not None:
                    later["content"] = r["content"] + " " + later["content"]
                else:
                    earlier = next((x for x in reversed(runs[:i - 1]) if x["role"] == r["role"]), None)
                    if earlier is not None:
                        earlier["content"] += " " + r["content"]
                runs[i - 1]["content"] += runs[i + 1]["content"]
                del runs[i:i + 2]
                changed = True
                break
    return [{"role": x["role"], "content": fix_terms(" ".join(x["content"].split()))} for x in runs if x["content"].strip()]


def live_tools() -> list[dict]:
    """The booth's chat tools in the Responses format GPT-Live's backend takes (flat: type, name, description, parameters)."""
    return [{"type": "function", **t["function"]} for t in TOOLS]


def pcm_rms(pcm: bytes) -> float:
    a = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float32)
    return float(np.sqrt((a * a).mean())) if a.size else 0.0


def louder_pcm(pcm: bytes, gain: float) -> bytes:
    """VOICE_GAIN for a stream: the same soft limiter as voice.louder, but without normalising each chunk (that would pump)."""
    if gain <= 1.0 or len(pcm) < 2:
        return pcm
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    y = np.tanh(gain * x) / np.tanh(gain) * 0.97
    return (y * 32767).astype("<i2").tobytes()


def pcm_to_wav(pcm: bytes, rate: int = RATE) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


def wav_to_pcm16k(wav: bytes) -> bytes:
    """A WAV (the classic voice, 24 kHz) as the robot stream's 16 kHz mono PCM."""
    with wave.open(io.BytesIO(wav)) as w:
        rate, ch, frames = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != RATE and x.size:
        x = np.interp(np.arange(0, x.size, rate / RATE), np.arange(x.size), x)
    return x.astype("<i2").tobytes()


async def openai_connect(s):
    """A WebSocket to GPT-Live. Nothing is billed until session.start is sent on it."""
    import websockets

    return await websockets.connect(URL, additional_headers={"Authorization": "Bearer " + s.openai_api_key}, max_size=None, open_timeout=15)


class LiveCall:
    """One GPT-Live conversation: opened by the wake word, closed on goodbye or quiet."""

    def __init__(self, bridge: "LiveBridge", first_pcm: bytes, first_text: str):
        self.b, self.brain = bridge, bridge.brain
        self.s, self.booth = self.brain.s, self.brain.booth
        self.enc = self.booth._encounter(None, self.brain.place_info().get("name"))
        text = fix_terms(first_text)
        self.enc.messages.append({"role": "user", "content": text})  # the wake utterance: the conversation is open from now on
        self.enc.turns += 1
        self.first, self.pending, self.starting, self.open = first_pcm, bytearray(), True, True
        self.wake_text, self.last_heard = text, text
        # The transcript as GPT-Live sends it: pieces (start_ms, n, role, text) on one timeline for both sides. Their text arrives in
        # bursts and the wake clip's transcript still arrives while the robot already answers, so turns are never cut by arrival:
        # the pieces are ordered by start_ms and each run of one speaker is a turn (see turns()).
        self.pieces: list[tuple[float, int, str, str]] = []
        self.published = ""   # the last robot turn shown / noted, so a turn is published again only when it grew
        self.committed_at = 0.0
        now = time.monotonic()
        self.opened = self.heard_at = self.bot_at = self.delta_at = now
        self.close_at: float | None = None
        self.ws = None
        self.closed_reason = ""

    # ---- the OpenAI side
    def session_config(self) -> dict:
        s, enc = self.s, self.enc
        state = "\nSTATE OF THIS CONVERSATION\n- They just said your wake word: answer what they asked, or say hi if that was all."
        if enc.greeted:
            state += f"\n- You already greeted them out loud with: \"{enc.greeted}\" Do not greet again; carry on from there."
        if s.claw_machine:
            state += f"\n- When they leave, invite them to try the claw machine to win {s.claw_prize}."
        return {
            "model": s.live_model,
            "instructions": system_prompt(s.robot_name, s.event_name, state, s.voice_language, self.booth._booth_info(), live=True),
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": s.live_voice_name}},
            "delegation": {"type": "responses", "responses": {
                "model": s.live_backend_model, "instructions": BACKEND.format(robot=s.robot_name),
                "tools": live_tools(), "tool_choice": "auto", "parallel_tool_calls": False}},
        }

    async def send(self, ev: dict) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps(ev))
        except Exception:  # GPT-Live has closed the call (goodbye, expiry): microphone frames still in flight are dropped
            self.open = False

    async def mic(self, pcm: bytes) -> None:
        if self.open and not self.starting and pcm:
            await self.send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm[:len(pcm) // 2 * 2]).decode()})

    async def run(self, ws=None) -> None:
        """ws: a connection opened while the wake word was checked (saves about a second); else one is opened here."""
        watch = None
        t0 = time.monotonic()
        try:
            self.ws = ws if ws is not None else await openai_connect(self.s)
            await self.send({"type": "session.start", "event_id": "start", "session": self.session_config()})
            while True:
                m = json.loads(await asyncio.wait_for(self.ws.recv(), 15))
                if m.get("type") == "session.started":
                    break
                if m.get("type") == "error":
                    raise RuntimeError((m.get("error") or {}).get("message", "GPT-Live refused the session"))
            print("live call open (%.1fs): %r" % (time.monotonic() - t0, self.last_heard), flush=True)
            self.starting = False
            await self.mic(self.first)            # what they said with the wake word: answered first
            if self.pending:                      # what they said while the call was opening
                await self.mic(bytes(self.pending))
                self.pending.clear()
            watch = asyncio.create_task(self._watch())
            async for raw in self.ws:
                if await self.on_event(json.loads(raw)) == "closed":
                    break
        except Exception as e:
            print("live call failed:", type(e).__name__, str(e)[:200], flush=True)
            self.booth._publish("error", {"error": f"GPT-Live call failed: {e}"[:200]})
        finally:
            self.open = False
            if watch:
                watch.cancel()
            self.commit()
            self.enc.ended = True  # the next visitor says the wake word again
            try:
                await self.ws.close()
            except Exception:
                pass
            print("live call closed after %.0fs (%s)" % (time.monotonic() - self.opened, self.closed_reason or "connection ended"), flush=True)

    async def on_event(self, m: dict) -> str | None:
        t, now = m.get("type"), time.monotonic()
        if t == "session.output_audio.delta":
            await self.b.play(louder_pcm(base64.b64decode(m.get("delta") or ""), self.s.voice_gain))
        elif t in ("session.input_transcript.delta", "session.output_transcript.delta"):
            user = t == "session.input_transcript.delta"
            last = self.pieces[-1][0] if self.pieces else 0.0
            start = m.get("start_ms")
            self.pieces.append((float(start) if start is not None else last + 1.0, len(self.pieces), "user" if user else "assistant", m.get("delta") or ""))
            self.delta_at = now
            if user:
                self.heard_at = now
            else:
                self.bot_at = now
        elif t == "response.event":
            ev = m.get("event") or {}
            item = ev.get("item") or {}
            if ev.get("type") in ("response.created", "response.output_item.done", "response.failed", "error"):
                print("live backend:", ev.get("type"), item.get("type") or "", item.get("name") or "", flush=True)
            if ev.get("type") == "response.output_item.done" and item.get("type") == "function_call":
                asyncio.create_task(self._tool(m.get("delegation_id"), item))
        elif t == "error":
            err = m.get("error") or {}
            print("GPT-Live error:", err.get("code"), str(err.get("message"))[:200], flush=True)
        elif t == "session.closed":
            return "closed"
        return None

    async def _tool(self, delegation_id: str | None, item: dict) -> None:
        name = item.get("name", "")
        try:
            args = json.loads(item.get("arguments") or "{}")
            res = await asyncio.to_thread(self.booth._tool, self.enc, name, args)
        except Exception as e:
            args, res = {}, {"error": str(e)[:300]}
        print("live tool %s %s -> %s" % (name, json.dumps(args), json.dumps(res, default=str)[:160]), flush=True)
        await self.send({"type": "response.item.create",
                         "item": {"type": "function_call_output", "call_id": item.get("call_id"), "output": json.dumps(res, default=str)[:4000]}})
        await self.send({"type": "response.create"})

    # ---- turns, rebuilt from the timeline
    def turns(self) -> list[dict]:
        out = merge_turns(self.pieces)
        if not out or out[0]["role"] != "user":  # GPT-Live did not transcribe the wake clip (yet): what the wake check heard
            out.insert(0, {"role": "user", "content": self.wake_text})
        return out

    def commit(self) -> None:
        """The conversation so far into the Encounter (the booth notes read it); publish and note the robot's latest turn."""
        enc, turns = self.enc, self.turns()
        enc.messages[:] = turns
        enc.turns = sum(1 for x in turns if x["role"] == "user")
        enc.last = time.monotonic()
        users = [x["content"] for x in turns if x["role"] == "user"]
        self.last_heard = users[-1] if users else self.wake_text
        if any(says_goodbye(u) for u in users[1:] or users[-1:]) and not enc.closing:
            enc.closing = True
        if turns[-1]["role"] != "assistant" or turns[-1]["content"] == self.published:
            return
        reply = self.published = turns[-1]["content"]
        heard = turns[-2]["content"] if len(turns) > 1 else ""
        self.booth._last_said = time.monotonic()
        self.booth._publish("converse", {"encounter_id": enc.id, "heard": heard, "reply": reply, "tools": [], "live": True})
        print("live: heard %r | said %r" % (heard[:160], reply[:200]), flush=True)
        if self.booth.extract_fn:  # the anonymous note, as for a classic conversation (in the background: never delays the call)
            lines = turns[-16:]
            transcript = "\n".join(("Visitor: " if x["role"] == "user" else "Robot: ") + x["content"] for x in lines)
            args = (enc, enc.id, transcript, [x["content"] for x in lines if x["role"] == "user"])
            if self.booth.learn_in_background:
                threading.Thread(target=self.booth._learn, args=args, daemon=True).start()
            else:
                self.booth._learn(*args)
        if enc.closing:
            self.close_at = time.monotonic() + 0.5  # once the goodbye has been played

    def tick(self) -> str | None:
        """Housekeeping, about four times a second: finish a quiet robot turn; return a reason when the call should close."""
        now = time.monotonic()
        if self.pieces and self.delta_at > self.committed_at and now - self.delta_at > BOT_TURN_GAP_S:
            self.committed_at = now
            self.commit()
        if self.close_at and now > self.close_at and now > self.b.audible_until + ECHO_TAIL_S:
            return "goodbye"
        if self.enc.ended:
            return "ended from the Booth tab"
        if now - max(self.heard_at, self.bot_at, self.b.audible_until) > self.s.wake_window_s:
            return "quiet for %.0fs" % self.s.wake_window_s
        if now - self.opened > MAX_CALL_S:
            return "ten minutes"
        return None

    async def _watch(self) -> None:
        while self.open:
            await asyncio.sleep(0.25)
            reason = self.tick()
            if reason:
                await self.close(reason)
                return

    async def close(self, reason: str) -> None:
        if not self.open:
            return
        self.closed_reason = reason
        try:
            await self.send({"type": "session.close", "event_id": "close-" + uuid.uuid4().hex[:6]})
        except Exception:
            pass
        loop = asyncio.get_running_loop()
        loop.call_later(5.0, lambda: asyncio.ensure_future(self.ws.close()) if self.ws is not None else None)


class LiveBridge:
    """One robot connected to /api/voice/live."""

    def __init__(self, brain):
        self.brain = brain
        self.ws = None
        self.call: LiveCall | None = None
        self.checking = False
        self.play_end = 0.0
        self.audible_until = 0.0

    def busy(self) -> bool:
        """The robot is talking (its microphone hears its own speaker) or moving (it hears its servos)."""
        return time.monotonic() < self.audible_until + ECHO_TAIL_S or self.brain.drive.moving

    async def play(self, pcm: bytes) -> None:
        if not pcm or self.ws is None:
            return
        for i in range(0, len(pcm), 32000):
            await self.ws.send_bytes(b"A" + pcm[i:i + 32000])
        now = time.monotonic()
        self.play_end = max(self.play_end, now) + len(pcm) / BPS
        if pcm_rms(pcm) > LOUD_OUT:
            self.audible_until = self.play_end

    async def tell(self, msg: dict) -> None:
        if self.ws is not None:
            await self.ws.send_text(json.dumps(msg))

    async def serve(self, ws) -> None:
        self.ws = ws
        ticker = asyncio.create_task(self._tick())
        try:
            while True:
                m = await ws.receive()
                if m.get("type") == "websocket.disconnect":
                    break
                data = m.get("bytes")
                if not data:
                    continue
                kind, pcm = data[:1], data[1:]
                call = self.call
                if kind == b"M" and call is not None and call.open:
                    await call.mic(bytes(len(pcm)) if self.busy() else pcm)
                elif kind == b"U":
                    if call is not None and call.starting:
                        call.pending += pcm
                    elif call is None and not self.checking:
                        self.checking = True
                        asyncio.create_task(self._utterance(pcm))
        finally:
            ticker.cancel()
            if self.call is not None:
                await self.call.close("the robot disconnected")

    async def _utterance(self, pcm: bytes) -> None:
        """Outside a call: is this someone saying the wake word (or answering a greeting)? Then open a call."""
        try:
            if self.busy() or not self.brain.listening or not self.brain.voice:
                return
            warm = asyncio.create_task(openai_connect(self.brain.s))  # meanwhile: connected by the time we know
            text = await asyncio.to_thread(self.brain.voice.transcribe, pcm_to_wav(pcm), "speech.wav")
            ok = not heard_nothing(text) and self.brain.booth.addressed(text, None)[0]
            if not ok:
                warm.add_done_callback(lambda f: f.exception() is None and asyncio.ensure_future(f.result().close()))
                return
            print("live: woken by %r" % text, flush=True)
            self.call = LiveCall(self, pcm, text)
            self.brain.on_wake()  # look up at the visitor while the call opens
            try:
                ws = await warm
            except Exception:
                ws = None  # LiveCall.run opens its own
            asyncio.create_task(self._run_call(self.call, ws))
        except Exception as e:
            print("live wake check failed:", e, flush=True)
        finally:
            self.checking = False

    async def _run_call(self, call: LiveCall, ws=None) -> None:
        await self.tell({"type": "call", "open": True})
        try:
            await call.run(ws)
        finally:
            if self.call is call:
                self.call = None
            try:
                await self.tell({"type": "call", "open": False})
            except Exception:
                pass

    async def _tick(self) -> None:
        """Every second: the head-levelling clock the classic client's polls gave, and greetings or cheers waiting to be said."""
        while True:
            await asyncio.sleep(1.0)
            try:
                self.brain.booth_tick()
                if self.call is None and not self.busy() and self.brain.voice:
                    it = self.brain.booth.outbox.next()
                    if it:
                        wav = await asyncio.to_thread(self.brain.voice.speak, it["text"])
                        await self.tell({"type": "say", "text": it["text"]})
                        await self.play(wav_to_pcm16k(wav))
            except Exception as e:
                print("live tick:", e, flush=True)
