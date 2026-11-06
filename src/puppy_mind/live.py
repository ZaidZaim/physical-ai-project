"""GPT-Live: a booth conversation as one full-duplex voice stream (OpenAI gpt-live-1), instead of record, transcribe, chat, speak.

Measured on this laptop: about 1 s from the end of a question to the first sound, against 5 to 7 s for the classic pipeline (1.5 s
waiting for silence on the robot, then transcription, the chat model and the whole reply synthesised before anything plays).

How it runs:
- The robot keeps one WebSocket to the laptop (/api/voice/live). Outside a call it sends only what someone said (one utterance per
  message, as before); during a call it streams its microphone, and it plays whatever audio comes back.
- The laptop keeps the OpenAI key and the booth rules. Every utterance opens a GPT-Live session at once (on a connection kept ready),
  but nothing it says is played and nothing is kept until GPT-Live's own live transcript shows the wake word ("Hey PuppyPi", or an
  answer to a greeting): then the held answer plays and the call goes on. Otherwise the session is closed after a second or two
  (billed per second: a fraction of a cent). Checking the wake word with a separate transcription first made every first answer
  1 to 2 s slower (seen at the booth).
- A call closes on goodbye, the Booth tab's End conversation, or WAKE_WINDOW_S of quiet: GPT-Live is billed per minute.
- Tricks, questions about the room and a new visitor are function tools of GPT-Live's delegated backend (LIVE_BACKEND_MODEL).
- The live transcripts go into the same Encounter as a classic conversation, so the anonymous booth notes are made the same way.
- The robot's microphone hears its own speaker, so while it talks (and briefly after) silence is sent instead: no barge-in yet.
- What the robot says on its own (the camera greeting, the claw machine cheers and invites) is spoken by GPT-Live too, in the same
  voice as the calls: a short session told to say exactly that line (LiveBridge.say). If that fails, the classic voice says it.

Robot frames, binary: b"S" + PCM (someone is speaking, outside a call: streamed from the start of the sentence, since GPT-Live
listens at real-time speed and a sentence sent only once it ended cost its whole length again), b"E" (that sentence ended),
b"U" + PCM (a whole utterance at once: older robot clients), b"M" + PCM (microphone stream, during a call). Laptop frames:
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

from .booth import TOOLS, has_wake_word, robot_says_goodbye, says_goodbye
from .voice import fix_terms
from .wizard import system_prompt

RATE = 16000
BPS = RATE * 2              # bytes per second of 16-bit mono audio
URL = "wss://api.openai.com/v1/live/sessions"
LOUD_OUT = 200.0            # output audio louder than this (RMS) is the robot talking; GPT-Live also streams near-silence
ECHO_TAIL_S = 0.6           # the microphone stays muted this long after the robot's last audible sound (its player buffers)
BOT_TURN_GAP_S = 1.2        # no new transcript for this long: the turns so far are committed (GPT-Live sends no end-of-turn event)
MAX_CALL_S = 600.0          # a safety net: no call runs longer than ten minutes
PROBE_AFTER_SPEECH_S = 2.0  # a sentence without the wake word: its session is closed this long after the sentence ended
PROBE_AFTER_ANSWER_S = 2.0  # ... or this long after GPT-Live began answering it (at the booth its transcript of "Hey, Puppy"
                            # came more than 0.7 s after the answer began: the call was dropped while the robot said "Hi there")
PROBE_NO_TRANSCRIPT_S = 4.0  # ... but while NO transcript has come yet, wait this long: at the robot (2026-11-06) the first "Hey
                            # PuppyPi" after a quiet spell was dropped and the second worked; its session had to (re)connect, and
                            # the 2 s were counted from the end of the sentence, not from when the session could hear it
PROBE_WORDS = 6             # ... or as soon as this many words were heard with nothing like "hey" / "puppy" / "pi" among them:
                            # people say the wake word first. At the busy booth checking chatter cost more than the calls
                            # (9.3 of 17 minutes, ~4.9 s a sentence); most of it was listening to the rest of other people's sentences
WAKEISH = {"hey", "hi", "hello", "hallo", "ok", "okay", "puppy", "puppypi", "pi", "pie", "pai", "poppy", "puppi", "papi", "bobby"}
PROBE_MAX_S = 20.0          # ... and a long monologue without it after this long
TRAIL_SILENCE_S = 1.5       # after the robot says a sentence ended, a probe is fed this much silence: GPT-Live transcribes a
                            # turn only once it HEARS the pause after it, and the robot sends nothing between sentences. Without
                            # it the first "Hey PuppyPi" after a quiet spell often got no transcript at all (robot, 2026-11-06)
SOUND_QUIET_S = 0.8         # a sound (the spin jingle) waits until no speech has been sent for this long and all of it has played
SOUND_WAIT_S = 6.0          # ... but no longer than this
SPARE_MAX_AGE_S = 45.0      # the connection kept ready for the next call is renewed this often (an idle one may be dropped)
class LiveUsage:
    """What GPT-Live has cost since the app started: session seconds of conversations, of wake-word checks (every sentence the
    robot hears opens a short session) and of lines it spoke on its own. Billed per second of session; the backend model's tokens
    for tricks and room questions are extra and not counted here."""

    def __init__(self):
        from datetime import datetime, timezone
        self.since = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.seconds = {"calls": 0.0, "checks": 0.0, "lines": 0.0}
        self.count = {"calls": 0, "checks": 0, "lines": 0}

    def add(self, kind: str, seconds: float) -> None:
        self.seconds[kind] += max(0.0, seconds)
        self.count[kind] += 1

    def report(self, price_per_min: float) -> dict:
        minutes = sum(self.seconds.values()) / 60
        return {"since": self.since, "minutes": round(minutes, 2), "usd": round(minutes * price_per_min, 2),
                "price_per_min": price_per_min, "count": dict(self.count),
                "minutes_by_kind": {k: round(v / 60, 2) for k, v in self.seconds.items()}}


USAGE = LiveUsage()  # one per app run (shown in the Booth tab)

OVERLAP_MS = 100            # speech resumed within this of a scrap's end overlaps it (see merge_turns)

SAY_INSTRUCTIONS = ("You are {robot}, a friendly robot dog at the Neo4j booth, and right now you are only the voice for one line, "
                    "not in a conversation. When you get a line, say it out loud exactly as written, word for word, warm and lively, "
                    "then stop. Add nothing, answer nothing, ask nothing.")
SAY_SETUP_S = 6.0           # connecting and starting a line's session: longer, and the classic voice says it instead
SAY_TIMEOUT_S = 15.0        # speaking a line, counted once its session has started
SAY_DONE_QUIET_S = 0.8      # the line is over once its words were all said and the voice has been quiet this long

STOP_WORDS = {"stop", "halt", "freeze"}  # heard in a call: the robot stops at once, before GPT-Live has even answered

BACKEND = ("You are the backend of {robot}, a robot dog at the Neo4j booth. Use the tools: do_trick when someone asks for a trick, "
           "move when someone asks you to walk, come closer, come back, back up, turn or spin (never sideways: it cannot) (faster = speed fast "
           "or sprint), routine when one sentence asks for several things ('come here and sit', 'walk over and take a nap', 'show them "
           "what you can do', 'wake up' = stand then stretch), do_trick for sit / take a nap (lie_down) / stand up (stand) / "
           "shake hands or give paw (shake_hands) / stretch / push-ups or exercise (push-up), "
           "stop_moving the moment anyone says stop, wait or freeze, "
           "ask_world for anything about the room or what the robot can see, new_visitor when a different person has taken over. "
           "Reply with one short plain sentence the voice can say. Never ask for or keep personal data.")


def _plain(s: str) -> str:
    """Lowercase words only: to compare what GPT-Live said with the line it was given."""
    return " ".join("".join(c if c.isalnum() else " " for c in (s or "").lower()).split())


def _says_all(said: str, line: str) -> bool:
    """Every word of `line` is in `said`, in order (GPT-Live may add a word or two, or say "Graph Wizard." for "Graph Wizard!")."""
    words, heard = _plain(line).split(), iter(_plain(said).split())
    return bool(words) and all(w in heard for w in words)  # `in` on an iterator moves it on: the words must come in order


def merge_turns(pieces: list[tuple]) -> list[dict]:
    """Transcript pieces (start_ms, n, role, text[, end_ms]) -> turns. Ordered by start_ms, each run of one speaker is a turn.
    A scrap of at most two words that OVERLAPS the other side's speech (the robot's "I'd" said over the visitor's last word) joins
    its own speaker's next turn, so the visitor's sentence stays whole. A short answer in its own pause ("GenAI, please", "Okay")
    stays a turn of its own. Sound labels such as "[hum]" are dropped."""
    runs: list[dict] = []
    for p in sorted(pieces, key=lambda p: (p[0], p[1])):
        start, role, text = p[0], p[2], re.sub(r"\[[^\]]*\]", " ", p[3])
        end = p[4] if len(p) > 4 and p[4] is not None else start + 200
        if runs and runs[-1]["role"] == role:
            runs[-1]["content"] += text
            runs[-1]["end"] = max(runs[-1]["end"], end)
        elif text.strip():
            runs.append({"role": role, "content": text, "start": start, "end": end})
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            prev, r, nxt = runs[i - 1], runs[i], runs[i + 1]
            overlaps = nxt["start"] < r["end"] + OVERLAP_MS  # the other side carries on while the scrap is still being said
            if len(r["content"].split()) <= 2 and prev["role"] == nxt["role"] != r["role"] and overlaps:
                later = next((x for x in runs[i + 2:] if x["role"] == r["role"]), None)
                if later is not None:
                    later["content"] = r["content"] + " " + later["content"]
                else:
                    earlier = next((x for x in reversed(runs[:i - 1]) if x["role"] == r["role"]), None)
                    if earlier is not None:
                        earlier["content"] += " " + r["content"]
                prev["content"] += nxt["content"]
                prev["end"] = max(prev["end"], nxt["end"])
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
    """One GPT-Live session for one utterance onwards: a probe until the wake word shows in its transcript, then a call."""

    def __init__(self, bridge: "LiveBridge", first_pcm: bytes):
        self.b, self.brain = bridge, bridge.brain
        self.s, self.booth = self.brain.s, self.brain.booth
        self.enc = None              # made when the wake word is confirmed: an utterance that was not for the robot leaves nothing
        self.confirmed = False
        self.held: list[bytes] = []  # GPT-Live's answer while unconfirmed: played at once on the wake word, else dropped
        self.first, self.pending, self.starting, self.open = first_pcm, bytearray(), True, True
        self.wake_text = self.last_heard = ""
        # The transcript as GPT-Live sends it: pieces (start_ms, n, role, text, end_ms) on one timeline for both sides. Their text
        # arrives in bursts and the first utterance's transcript still arrives while the robot already answers, so turns are never
        # cut by arrival: the pieces are ordered by start_ms and each run of one speaker is a turn (see merge_turns).
        self.pieces: list[tuple] = []
        self.published = ""   # the last robot turn shown / noted, so a turn is published again only when it grew
        self.committed_at = 0.0
        now = time.monotonic()
        self.opened = self.heard_at = self.bot_at = self.delta_at = now
        self.started_at: float | None = None
        self.drop_detail = ""  # why a probe was dropped (timing only, never the words)
        self.answer_at: float | None = None
        self.speech_end_at: float | None = None  # the robot said the sentence ended (b"E")
        self.robot_audio_at = 0.0  # the last microphone frame from the robot (the trailing silence never cuts into speech)
        self.robot_audio_s = 0.0   # how much the robot sent, and how loud: logged when a probe is dropped (never the words)
        self.robot_peak = 0.0
        self.not_for_me = False  # its first words have nothing like the wake word: the check ends now (see PROBE_WORDS)
        self.close_at: float | None = None
        self.ws = None
        self.closed_reason = ""

    # ---- the OpenAI side
    def session_config(self) -> dict:
        s = self.s
        state = ("\nSTATE OF THIS CONVERSATION\n- Someone just said your wake word (or answered your greeting): answer what they asked, "
                 "or say hi if that was all.")
        g = self.booth._recent_greeting
        if g and time.monotonic() - g[0] < 120:
            state += f"\n- You already greeted them out loud with: \"{g[1]}\" Do not greet again; carry on from there."
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
        if self.enc is not None and self.enc.closing:  # they said goodbye: the hall's chatter must not keep the call going
            pcm = bytes(len(pcm))
        if self.open and not self.starting and pcm:
            await self.send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm[:len(pcm) // 2 * 2]).decode()})

    async def run(self, ws=None) -> None:
        """ws: a connection kept ready (saves up to a second and a half); else one is opened here."""
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
            self.started_at = time.monotonic()
            self.starting = False
            await self.mic(self.first)            # the utterance: GPT-Live transcribes it (the wake check) and answers it
            if self.pending:                      # what was said while the session was opening
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
            if self.started_at is not None:
                USAGE.add("calls" if self.confirmed else "checks", time.monotonic() - self.started_at)
            if self.confirmed:
                self.commit()
                self.enc.ended = True  # the next visitor says the wake word again
                print("live call closed after %.0fs (%s)" % (time.monotonic() - self.opened, self.closed_reason or "connection ended"), flush=True)
            try:
                await self.ws.close()
            except Exception:
                pass

    async def on_event(self, m: dict) -> str | None:
        t, now = m.get("type"), time.monotonic()
        if t == "session.output_audio.delta":
            raw = base64.b64decode(m.get("delta") or "")
            item = (louder_pcm(raw, self.s.voice_gain), pcm_rms(raw) > LOUD_OUT)  # talking or not: judged before the gain
            if self.confirmed:
                await self.b.play(*item)
            else:
                self.held.append(item)
        elif t in ("session.input_transcript.delta", "session.output_transcript.delta"):
            user = t == "session.input_transcript.delta"
            last = self.pieces[-1][0] if self.pieces else 0.0
            start, end = m.get("start_ms"), m.get("end_ms")
            start = float(start) if start is not None else last + 1.0
            self.pieces.append((start, len(self.pieces), "user" if user else "assistant", m.get("delta") or "",
                                float(end) if end is not None else None))
            self.delta_at = now
            if user:
                self.heard_at = now
                # "stop" must not wait for the backend's tool call (a second or two): stop any walk the moment it is heard
                if self.confirmed and STOP_WORDS & set(_plain(m.get("delta") or "").split()) and self.booth.stop_fn:
                    print("live: heard stop, robot stopped:", self.booth.stop_fn(), flush=True)
            else:
                self.bot_at = now
                self.answer_at = self.answer_at or now
            if not self.confirmed and not self.closed_reason:  # a session already being closed is never confirmed after all
                heard = " ".join(x["content"] for x in merge_turns(self.pieces) if x["role"] == "user")
                # only "Hey PuppyPi" opens a call: right after a camera greeting any sentence used to count as an answer to it,
                # and in the busy hall bystanders' chatter ("to have ...") opened calls nobody had asked for
                words = _plain(heard).split()
                if len(words) >= PROBE_WORDS and not WAKEISH & set(words[:PROBE_WORDS]) and not has_wake_word(heard, self.s.wake_word):
                    self.not_for_me = True
                if heard and has_wake_word(heard, self.s.wake_word):
                    await self.confirm(heard)
        elif t == "response.event":
            ev = m.get("event") or {}
            item = ev.get("item") or {}
            if ev.get("type") in ("response.created", "response.output_item.done", "response.failed", "error"):
                print("live backend:", ev.get("type"), item.get("type") or "", item.get("name") or "", flush=True)
            if ev.get("type") == "response.output_item.done" and item.get("type") == "function_call" and self.confirmed:
                asyncio.create_task(self._tool(item))
        elif t == "error":
            err = m.get("error") or {}
            print("GPT-Live error:", err.get("code"), str(err.get("message"))[:200], flush=True)
        elif t == "session.closed":
            return "closed"
        return None

    async def confirm(self, heard: str) -> None:
        """The wake word (or an answer to a greeting) is in the transcript: this is a conversation. Play what is held, keep notes."""
        self.confirmed = True
        self.wake_text = self.last_heard = fix_terms(heard)
        self.enc = self.booth._encounter(None, self.brain.place_info().get("name"))
        self.enc.messages.append({"role": "user", "content": self.wake_text})  # the conversation is open from now on
        self.enc.turns += 1
        print("live: woken by %r (%.1fs after the session started; robot sent %.1fs of audio, loudest RMS %.0f)" % (
            self.wake_text, time.monotonic() - (self.started_at or self.opened), self.robot_audio_s, self.robot_peak), flush=True)
        self.brain.on_wake()  # look up at the visitor
        await self.b.tell({"type": "call", "open": True})
        held, self.held = self.held, []
        for pcm, loud in held:
            await self.b.play(pcm, loud)

    async def _tool(self, item: dict) -> None:
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
        if not out or out[0]["role"] != "user":  # GPT-Live did not transcribe the wake utterance (yet): what confirmed it
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
        if len(turns) > 2 and robot_says_goodbye(reply) and not enc.closing:  # it said goodbye itself: the call ends after it
            enc.closing = True
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
        if enc.closing and self.close_at is None:  # once, when the goodbye is answered: later replies never push it back
            self.close_at = time.monotonic() + 0.5  # (and the goodbye has been played: see tick)

    def tick(self) -> str | None:
        """Housekeeping, about four times a second: end a probe that was not for the robot, commit quiet turns, and return a
        reason when the call should close."""
        now = time.monotonic()
        if not self.confirmed:
            heard_any = any(p[2] == "user" and p[3].strip() for p in self.pieces)
            wait = PROBE_AFTER_SPEECH_S if heard_any else PROBE_NO_TRANSCRIPT_S
            # the wait counts from when the session could hear it (it may still have been connecting when the sentence ended)
            sentence_over = (self.speech_end_at is not None and self.started_at is not None
                             and now - max(self.speech_end_at, self.started_at) > wait)
            if (self.not_for_me or sentence_over
                    or (self.answer_at and now - self.answer_at > PROBE_AFTER_ANSWER_S) or now - self.opened > PROBE_MAX_S):
                # why, and how long things took; never what was said (bystanders)
                ready = "never ready" if self.started_at is None else "ready after %.1fs" % (self.started_at - self.opened)
                self.drop_detail = "%s; session %s; %s; robot sent %.1fs of audio, loudest RMS %.0f" % (
                    "words but no wake word" if heard_any else "no transcript came", ready,
                    "the sentence ended %.1fs ago" % (now - self.speech_end_at) if self.speech_end_at else "still talking",
                    self.robot_audio_s, self.robot_peak)
                return "no wake word"
            return None
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
        self.held = []
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
        self.saying: asyncio.Task | None = None  # a greeting or cheer being spoken by GPT-Live
        self.last_said = ""
        self.play_end = 0.0
        self.audible_until = 0.0
        self.spare: asyncio.Task | None = None  # a GPT-Live connection kept ready for the next utterance
        self.robot_aec = False  # the robot said its echo canceller is on (its hello)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.robot_music = False  # the robot plays music in a stream of its own (its hello says so; newer robot clients)
        self.loud_sent_at = 0.0  # when speech was last sent to the robot (a sound waits for the answer to end)
        self.spare_at = 0.0

    def mute_mic(self) -> bool:
        """In a call: send silence instead of the microphone? While the robot talks, unless LIVE_BARGE_IN is on and the robot's
        echo canceller removes its own voice (then visitors can interrupt it). NOT while it moves: muting the servo noise also
        muted "Stop!", which is exactly when it is said (robot, 2026-11-06)."""
        return time.monotonic() < self.audible_until + ECHO_TAIL_S and not (self.brain.s.live_barge_in and self.robot_aec)

    def busy(self) -> bool:
        """The robot is talking (its microphone hears its own speaker) or moving (it hears its servos). A line still being prepared
        does not count: a visitor who speaks first is heard, and the line gives way (see say)."""
        return time.monotonic() < self.audible_until + ECHO_TAIL_S or self.brain.drive.moving

    async def _say_setup(self, text: str):
        """A GPT-Live session ready to say `text`: connected, started, the line handed over."""
        s = self.brain.s
        ws = await self._take_spare() or await openai_connect(s)
        try:
            await ws.send(json.dumps({"type": "session.start", "event_id": "say", "session": {
                "model": s.live_model, "instructions": SAY_INSTRUCTIONS.format(robot=s.robot_name),
                "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": s.live_voice_name}}}}))
            while True:
                m = json.loads(await ws.recv())
                if m.get("type") == "session.started":
                    break
                if m.get("type") == "error":
                    raise RuntimeError((m.get("error") or {}).get("message", "GPT-Live refused the session"))
            await ws.send(json.dumps({"type": "session.commentary.append", "event_id": "say-" + uuid.uuid4().hex[:8],
                                      "delegation_id": None, "content": text}))
            return ws
        except BaseException:
            self._close_later(ws)
            raise

    def _close_later(self, ws) -> None:
        """End a session without waiting for it (busy() must clear as soon as the line has played)."""
        async def close():
            try:
                await asyncio.wait_for(ws.send(json.dumps({"type": "session.close", "event_id": "say-close"})), 2)
                await asyncio.wait_for(ws.close(), 3)
            except BaseException:
                pass
        asyncio.ensure_future(close())

    async def say(self, text: str) -> str:
        """Speak one line in GPT-Live's voice (the voice of the calls) on the robot's speaker. Returns "said", "skipped" (a visitor
        spoke before any of it played: the conversation goes first) or "failed" (nothing was heard: the classic voice says it)."""
        s, ws, feeder = self.brain.s, None, None
        heard, played, t0, started = False, False, time.monotonic(), None
        try:
            try:  # connecting and starting have their own short limit: a slow link falls back to the classic voice at once
                ws = await asyncio.wait_for(self._say_setup(text), SAY_SETUP_S)
            except asyncio.TimeoutError:
                print("live say: GPT-Live took over %.0fs to start" % SAY_SETUP_S, flush=True)
                return "failed"

            async def silence():  # GPT-Live runs on its input stream: with no audio coming in it never speaks (measured)
                chunk = base64.b64encode(bytes(BPS // 50)).decode()
                while True:
                    await ws.send(json.dumps({"type": "session.input_audio.append", "audio": chunk}))
                    await asyncio.sleep(0.02)
            feeder = asyncio.create_task(silence())
            said, first_loud, last_loud, done_at = "", None, None, None
            longest = 2.0 + 0.45 * len(text.split())  # a generous estimate of how long the line takes to say
            started = time.monotonic()  # the speaking limits count from here, not from the connection (and the billing)
            while time.monotonic() - started < SAY_TIMEOUT_S:
                now = time.monotonic()
                if not played and self.call is not None:
                    return "skipped"  # a visitor spoke while the line was being prepared: their conversation goes first
                if done_at is None and heard and _says_all(said, text):
                    done_at = now  # every word of the line is in the transcript (its audio may still be arriving)
                if done_at is not None and last_loud is not None and now - last_loud > SAY_DONE_QUIET_S:
                    break  # the first pause after the last word: the line is over
                if first_loud is not None and now - first_loud > longest:
                    break  # it carries on past the line without a pause: stop there
                if heard and last_loud is not None and now - last_loud > 2.5:
                    break  # it stopped early (it reworded the line): what was said is what it said
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 0.25))
                except asyncio.TimeoutError:
                    continue
                kind = m.get("type")
                if kind == "session.output_audio.delta":
                    raw = base64.b64decode(m.get("delta") or "")
                    loud = pcm_rms(raw) > LOUD_OUT
                    if loud and not played and self.call is not None:
                        return "skipped"
                    await self.play(louder_pcm(raw, s.voice_gain), loud)  # every chunk of the line plays, to its last word
                    if loud:
                        played = heard = True
                        last_loud = time.monotonic()
                        first_loud = first_loud or last_loud
                elif kind == "session.output_transcript.delta":
                    said += m.get("delta") or ""
                elif kind == "session.closed":
                    break
                elif kind == "error":
                    err = m.get("error") or {}
                    print("GPT-Live error while speaking:", err.get("code"), str(err.get("message"))[:160], flush=True)
            self.last_said = said.strip()
            print("live say (%.1fs): %r" % (time.monotonic() - t0, self.last_said[:120] or "(nothing)"), flush=True)
        except Exception as e:
            print("live say failed:", type(e).__name__, str(e)[:160], flush=True)
        finally:
            if feeder is not None:
                feeder.cancel()
            if ws is not None:
                self._close_later(ws)
            if started is not None:
                USAGE.add("lines", time.monotonic() - started)
        return "said" if heard else "failed"

    async def _say_item(self, text: str) -> None:
        """A greeting or cheer from the outbox: GPT-Live's voice, or the classic voice if GPT-Live could not say it."""
        try:
            await self.tell({"type": "say", "text": text})
            if await self.say(text) == "failed" and self.brain.voice and self.call is None:
                wav = await asyncio.to_thread(self.brain.voice.speak, text)
                await self.play(wav_to_pcm16k(wav), True)
        except Exception as e:  # the robot went away mid-line: nothing to say it to
            print("live say:", type(e).__name__, str(e)[:120], flush=True)

    async def play(self, pcm: bytes, loud: bool | None = None) -> None:
        """Send audio to the robot's speaker. loud: whether it is speech (the microphone is muted until it has played)."""
        if not pcm or self.ws is None:
            return
        for i in range(0, len(pcm), 32000):
            await self.ws.send_bytes(b"A" + pcm[i:i + 32000])
        now = time.monotonic()
        self.play_end = max(self.play_end, now) + len(pcm) / BPS
        if loud if loud is not None else pcm_rms(pcm) > LOUD_OUT:
            self.audible_until = self.play_end
            self.loud_sent_at = now

    async def tell(self, msg: dict) -> None:
        if self.ws is not None:
            await self.ws.send_text(json.dumps(msg))

    def _renew_spare(self) -> None:
        old = self.spare
        if old is not None:
            old.add_done_callback(lambda f: (not f.cancelled() and f.exception() is None) and asyncio.ensure_future(f.result().close()))
        self.spare, self.spare_at = asyncio.create_task(openai_connect(self.brain.s)), time.monotonic()

    async def _take_spare(self):
        """The ready connection (a fresh one is started for the next time); None if it failed, and the call opens its own."""
        spare, self.spare = self.spare, None
        self.spare, self.spare_at = asyncio.create_task(openai_connect(self.brain.s)), time.monotonic()
        if spare is None:
            return None
        try:
            ws = await spare
            return ws if getattr(ws, "close_code", None) is None else None
        except Exception:
            return None

    def sound(self, name: str) -> threading.Event | None:
        """Play a named sound (jingle.SOUNDS: "spin", "horn") on the robot's speaker; safe to call from any thread (tools run in
        threads). Returns an event set when the sound starts (None: it cannot be played). Not counted as speech: the microphone
        stays open while it plays, so "Stop!" is still heard mid-spin."""
        from .jingle import SOUNDS
        if name not in SOUNDS or self.ws is None or self.loop is None:
            return None
        ev = threading.Event()
        if self.robot_music:  # mixed with its voice and played at once: the spin starts with it, even mid-sentence
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._sound_now(name, ev)))
        elif name == "horn":  # an older robot client: a short honk just joins the speech queue
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._queue_sound(name, ev)))
        else:  # an older robot client queues music behind its speech: wait until it has finished talking
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._sound_when_quiet(ev)))
        return ev

    async def _sound_now(self, name: str, ev: threading.Event) -> None:
        """b"B": music, replacing music still playing; b"C": a short effect (the horn), played over whatever plays."""
        from .jingle import SOUNDS
        try:
            await self.ws.send_bytes((b"C" if name == "horn" else b"B") + SOUNDS[name]())
            print("%s: playing now, beside the voice" % ("horn" if name == "horn" else "spin music"), flush=True)
        finally:
            ev.set()

    async def _queue_sound(self, name: str, ev: threading.Event) -> None:
        from .jingle import SOUNDS
        try:
            await self.play(SOUNDS[name](), False)
        finally:
            ev.set()

    async def _sound_when_quiet(self, ev: threading.Event) -> None:
        from .jingle import spin_jingle
        t0 = time.monotonic()
        end = t0 + SOUND_WAIT_S
        try:
            while time.monotonic() < end:
                now = time.monotonic()
                if now - self.loud_sent_at > SOUND_QUIET_S and now >= self.play_end - 0.1:
                    break  # the answer has stopped coming and the speaker has played it all
                await asyncio.sleep(0.05)
            now = time.monotonic()
            print("spin music: waited %.1fs for the robot to finish talking%s; speaker queue %.1fs, playing now" % (
                now - t0, " (gave up waiting)" if now >= end else "", max(0.0, self.play_end - now)), flush=True)
            await self.play(spin_jingle(), False)
        finally:
            ev.set()

    async def serve(self, ws) -> None:
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self._renew_spare()
        from .jingle import SOUNDS
        await asyncio.to_thread(lambda: [make() for make in SOUNDS.values()])  # made once (about a second), never mid-trick
        self.brain.sound_fn = self.sound
        ticker = asyncio.create_task(self._tick())
        try:
            while True:
                m = await ws.receive()
                if m.get("type") == "websocket.disconnect":
                    break
                data = m.get("bytes")
                if not data:
                    try:  # the robot's hello: is its echo canceller on?
                        hello = json.loads(m.get("text") or "{}")
                        if hello.get("type") == "hello":
                            self.robot_aec = bool(hello.get("aec"))
                            self.robot_music = bool(hello.get("music"))  # it plays b"B" music beside its voice, at once
                            print("robot echo cancelling: %s%s" % ("on" if self.robot_aec else "off", ", visitors may interrupt it"
                                  if self.robot_aec and self.brain.s.live_barge_in else ""), flush=True)
                    except ValueError:
                        pass
                    continue
                kind, pcm = data[:1], data[1:]
                call = self.call
                if kind == b"E":
                    if call is not None and not call.confirmed and call.speech_end_at is None:
                        call.speech_end_at = time.monotonic()
                        asyncio.create_task(self._trail(call))
                elif call is not None and call.open:
                    call.robot_audio_at, call.robot_audio_s = time.monotonic(), call.robot_audio_s + len(pcm) / BPS
                    call.robot_peak = max(call.robot_peak, pcm_rms(pcm))
                    if call.starting:
                        call.pending += pcm
                    elif kind in (b"M", b"S") or not call.confirmed:  # a probe hears the rest of the sentence as it is said
                        await call.mic(bytes(len(pcm)) if self.mute_mic() else pcm)
                elif kind in (b"S", b"U") and not self.busy() and self.brain.listening:
                    # self.call is set HERE, before the next frame is read: set inside the task, the next pieces of the same sentence
                    # each opened a session of their own (seen at the booth: 3 to 4 per sentence, the wake word split between them)
                    self.call = LiveCall(self, pcm)
                    if kind == b"U":  # a whole sentence at once: GPT-Live hears it at real-time speed, so it "ends" that much later
                        self.call.speech_end_at = time.monotonic() + len(pcm) / BPS
                    asyncio.create_task(self._probe(self.call))
        finally:
            ticker.cancel()
            if getattr(self.brain, "sound_fn", None) == self.sound:
                self.brain.sound_fn = None
            if self.call is not None:
                await self.call.close("the robot disconnected")
            if self.spare is not None:
                self._renew_spare()  # closes the spare when it is ready
                self.spare.cancel()

    async def _probe(self, call: LiveCall) -> None:
        """Someone speaks outside a call: open a GPT-Live session at once; it becomes a call if the wake word is in it."""
        print("live: someone speaks: listening for the wake word", flush=True)
        try:
            await call.run(await self._take_spare())
        finally:
            if not call.confirmed:  # what a bystander said is never printed
                print("live: no wake word (%s): dropped%s" % (call.closed_reason or "session ended",
                                                               " [%s]" % call.drop_detail if call.drop_detail else ""), flush=True)
            if self.call is call:
                self.call = None
            if call.confirmed:
                try:
                    await self.tell({"type": "call", "open": False})
                except Exception:
                    pass

    async def _trail(self, call: LiveCall) -> None:
        """The sentence ended: feed the probe the pause after it (see TRAIL_SILENCE_S), once its session can hear."""
        step = 0.02
        try:
            while call.starting and call.open:
                await asyncio.sleep(step)
            chunk, sent, t_end = bytes(int(BPS * step) // 2 * 2), 0.0, time.monotonic() + TRAIL_SILENCE_S + 5.0
            while sent < TRAIL_SILENCE_S and call.open and not call.confirmed and time.monotonic() < t_end:
                if time.monotonic() - call.robot_audio_at > 3 * step:  # the robot is not sending: never mixed into real speech
                    await call.mic(chunk)
                    sent += step
                await asyncio.sleep(step)
        except Exception:  # the session went away: nothing to feed
            pass

    async def _tick(self) -> None:
        """Every second: the head-levelling clock the classic client's polls gave, greetings or cheers waiting to be said, and a
        fresh ready connection now and then."""
        while True:
            await asyncio.sleep(1.0)
            try:
                self.brain.booth_tick()
                if self.call is None and time.monotonic() - self.spare_at > SPARE_MAX_AGE_S:
                    self._renew_spare()
                if self.call is None and not self.busy() and (self.saying is None or self.saying.done()):
                    it = self.brain.booth.outbox.next()
                    if it:  # spoken beside this clock (a line takes a few seconds), never two at once (busy() while it runs)
                        self.saying = asyncio.create_task(self._say_item(it["text"]))
            except Exception as e:
                print("live tick:", e, flush=True)
