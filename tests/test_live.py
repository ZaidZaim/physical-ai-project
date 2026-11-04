"""GPT-Live calls (live.py) and the robot's streaming client, without the network: transcript turns, the booth notes, audio helpers,

the switch, and the robot's stdlib WebSocket framing."""

import asyncio

import base64

import dataclasses

import importlib.util

import io

import socket

import threading

import wave



import numpy as np

import pytest

from fastapi.testclient import TestClient



from puppy_mind import server

from puppy_mind.booth import BoothHost

from puppy_mind.devfake import FakeMemory

from puppy_mind.live import LiveCall, live_tools, louder_pcm, merge_turns, wav_to_pcm16k

from puppy_mind.wizard import BOOTH, _ADDED_QUESTION, system_prompt

from tests.test_booth import FakeChat, S





def pieces(*items):

    """(start_ms, role, text) -> transcript pieces; each word-ish piece lasts 200 ms like GPT-Live's."""

    return [(float(ms), i, role, text, float(ms) + 200) for i, (ms, role, text) in enumerate(items)]





def test_turns_follow_the_timeline_not_the_arrival_order():

    # the wake clip's transcript arrives while the robot already answers: ordered by start_ms it is still one sentence

    p = pieces((5200, "assistant", " GraphRAG"), (600, "user", " Hey PuppyPi,"), (1800, "user", " what is"), (5400, "assistant", " uses graphs."),

               (2600, "user", " GraphRAG?"))

    assert merge_turns(p) == [{"role": "user", "content": "Hey PuppyPi, what is GraphRAG?"},

                              {"role": "assistant", "content": "GraphRAG uses graphs."}]





def test_a_scrap_of_the_robot_inside_the_visitors_sentence_joins_its_answer_and_sound_labels_go():

    p = pieces((0, "user", " Can you wave at"), (900, "assistant", " I'd"), (1000, "user", " me"), (1200, "assistant", " love to [hum] wave."),

               (5000, "user", " Thanks, bye"))

    assert merge_turns(p) == [{"role": "user", "content": "Can you wave at me"}, {"role": "assistant", "content": "I'd love to wave."},

                              {"role": "user", "content": "Thanks, bye"}]





def test_a_short_answer_in_its_own_pause_stays_a_turn():

    # seen at the booth: "GenAI, please" (an answer to the robot's question) was glued onto the next turn

    p = pieces((0, "assistant", " Cypher basics or GenAI stuff?"), (2000, "user", " GenAI, please"), (3500, "assistant", " Try the GraphRAG course."))

    assert [x["content"] for x in merge_turns(p)] == ["Cypher basics or GenAI stuff?", "GenAI, please", "Try the GraphRAG course."]





class FakeBridge:

    def __init__(self, booth):

        woke = []

        self.woke = woke

        self.brain = type("B", (), {"s": booth.s, "booth": booth, "place_info": lambda self: {"name": "Conference hall"},

                                    "on_wake": lambda self: woke.append(1)})()

        self.audible_until = 0.0

        self.played, self.told = [], []



    async def play(self, pcm, loud=None):

        self.played.append(pcm)



    async def tell(self, msg):

        self.told.append(msg)





def _ev(call, t, ms, delta):

    asyncio.run(call.on_event({"type": f"session.{t}_transcript.delta", "start_ms": ms, "end_ms": ms + 200, "delta": delta}))





def _booth(**kw):

    seen = []

    booth = BoothHost(dataclasses.replace(S, claw_machine=True), FakeMemory(S), client=FakeChat([]),

                      extract_fn=lambda tr: seen.append(tr) or {"topics": ["GraphRAG"], "questions": ["what is GraphRAG"]}, **kw)

    booth.learn_in_background = booth.write_in_background = False

    return booth, seen





def test_a_live_call_starts_on_the_wake_word_in_gpt_lives_own_transcript_and_plays_what_it_held():

    booth, seen = _booth()

    bridge = FakeBridge(booth)

    call = LiveCall(bridge, b"\x00\x00" * 1600)

    asyncio.run(call.on_event({"type": "session.output_audio.delta", "delta": base64.b64encode(b"\x10\x00" * 800).decode()}))

    assert call.held and not bridge.played and call.enc is None  # nothing plays and nothing is kept before the wake word

    _ev(call, "input", 600, " Hey Puppy Pi, what is GraphRAG?")

    assert call.confirmed and bridge.woke and bridge.told == [{"type": "call", "open": True}] and len(bridge.played) == 1

    assert booth.conversation_active()

    _ev(call, "output", 4000, " GraphRAG retrieves with graphs.")

    call.commit()

    assert [m["role"] for m in call.enc.messages] == ["user", "assistant"]

    assert "what is GraphRAG" in seen[-1] and call.enc.notes["topics"] and booth.mem.list_conversations()

    _ev(call, "input", 9000, " Thanks, bye!")

    _ev(call, "output", 10000, " Bye! Try the claw machine.")

    call.commit()

    assert call.enc.closing and call.close_at is not None

    assert call.tick() is None  # the goodbye is still being played (0.5 s)

    call.close_at -= 1

    assert call.tick() == "goodbye"





def test_an_utterance_without_the_wake_word_is_dropped_unheard_and_unkept():

    booth, seen = _booth()

    bridge = FakeBridge(booth)

    call = LiveCall(bridge, b"\x00\x00" * 1600)

    call.started_at = call.opened

    _ev(call, "input", 600, " so I told him the booth opens at nine")

    _ev(call, "output", 3000, " That sounds like a plan!")

    asyncio.run(call.on_event({"type": "session.output_audio.delta", "delta": base64.b64encode(b"\x10\x00" * 800).decode()}))

    assert not call.confirmed and not bridge.played and not bridge.told and call.enc is None

    call.answer_at -= 1

    assert call.tick() == "no wake word"

    assert not booth.encounters and not seen and not booth.conversation_active()





def test_the_robot_voice_is_louder_and_the_classic_voice_becomes_the_stream_format():

    quiet = (np.sin(np.linspace(0, 50, 1600)) * 3000).astype("<i2").tobytes()

    loud = np.frombuffer(louder_pcm(quiet, 2.5), dtype="<i2")

    assert np.abs(loud).max() > 2 * 3000 and np.abs(loud).max() < 32767 and louder_pcm(quiet, 1.0) == quiet

    b = io.BytesIO()

    with wave.open(b, "wb") as w:

        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(bytes(48000))

    assert len(wav_to_pcm16k(b.getvalue())) == 32000  # one second at 16 kHz





def test_live_instructions_ask_their_own_question_and_say_when_to_delegate():

    assert _ADDED_QUESTION in BOOTH  # the classic bullet that live mode swaps out

    live = system_prompt("PuppyPi", "the conference", "", live=True)

    assert "<|SpawnThinking|>" in live and "is added automatically" not in live

    assert "is added automatically" in system_prompt("PuppyPi", "the conference", "")

    names = {t["name"] for t in live_tools()}

    assert names == {"do_trick", "new_visitor", "ask_world"} and all(t["type"] == "function" and "parameters" in t for t in live_tools())





@pytest.fixture()

def client(monkeypatch):

    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")

    monkeypatch.delenv("ROSBRIDGE_URL", raising=False)

    with TestClient(server.app, client=("127.0.0.1", 5000)) as c:

        yield c





def test_live_voice_is_off_unless_switched_on_and_the_robot_is_told_so(client):

    server.brain.s = dataclasses.replace(server.brain.s, live_voice=False)

    assert client.get("/api/voice/mode").json()["live"] is False

    with client.websocket_connect("/api/voice/live") as ws:

        with pytest.raises(Exception) as e:

            ws.receive_bytes()

    assert getattr(e.value, "code", None) == 4001  # the robot falls back to the classic turns





def _robot_module():

    spec = importlib.util.spec_from_file_location("puppy_voice", "robot/puppy_voice.py")

    m = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(m)

    return m





def test_the_robots_websocket_client_frames_messages_both_ways():

    pv = _robot_module()

    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)

    port, got = srv.getsockname()[1], {}



    def laptop():  # a tiny server: handshake, read one masked frame, answer with an unmasked binary frame and a close

        c, _ = srv.accept()

        req = b""

        while b"\r\n\r\n" not in req:

            req += c.recv(1024)

        c.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")

        h = c.recv(2); n = h[1] & 0x7F

        if n == 126:

            n = int.from_bytes(c.recv(2), "big")

        mask = c.recv(4); data = b""

        while len(data) < n:

            data += c.recv(n - len(data))

        got["frame"] = bytes(b ^ mask[i % 4] for i, b in enumerate(data))

        reply = b"A" + bytes(300)

        c.sendall(bytes([0x82, 126]) + len(reply).to_bytes(2, "big") + reply + bytes([0x88, 2]) + (4001).to_bytes(2, "big"))

        c.close()

    threading.Thread(target=laptop, daemon=True).start()

    ws = pv.MiniWS("ws://127.0.0.1:%d/api/voice/live" % port)

    ws.send_bytes(b"U" + bytes(range(200)))

    op, data = ws.recv()

    assert op == 2 and data == b"A" + bytes(300)

    with pytest.raises(ConnectionError):

        ws.recv()

    assert ws.close_code == 4001 and got["frame"] == b"U" + bytes(range(200))



def test_the_pieces_of_one_sentence_open_one_session_not_one_each(monkeypatch):
    # seen at the booth: each streamed piece that arrived before the first check started opened its own session (3 to 4 per
    # sentence), the wake word was split between them and never heard
    from puppy_mind import live

    opened = []

    async def fake_run(self, ws=None):
        opened.append(self)
        self.starting = False
        await asyncio.sleep(0.05)

    monkeypatch.setattr(live.LiveCall, "run", fake_run)
    booth, _ = _booth()
    brain = type("B", (), {"s": booth.s, "booth": booth, "listening": True, "voice": None, "place_info": lambda self: {},
                           "drive": type("D", (), {"moving": False})(), "booth_tick": lambda self: None})()
    frames = [{"type": "websocket.receive", "bytes": b"S" + bytes(640)} for _ in range(4)] + [{"type": "websocket.disconnect"}]

    class WS:
        async def receive(self):
            await asyncio.sleep(0)
            return frames.pop(0)

        async def send_bytes(self, b):
            pass

        async def send_text(self, s):
            pass

    bridge = live.LiveBridge(brain)
    monkeypatch.setattr(bridge, "_renew_spare", lambda: None)

    async def no_spare():
        return None
    monkeypatch.setattr(bridge, "_take_spare", no_spare)

    async def go():
        await bridge.serve(WS())
        await asyncio.sleep(0.1)
    asyncio.run(go())
    assert len(opened) == 1


def test_the_robots_own_goodbye_ends_the_call():
    booth, _ = _booth()
    call = LiveCall(FakeBridge(booth), b"\x00\x00" * 1600)
    _ev(call, "input", 600, " Hey PuppyPi, how do I start with Aura?")
    _ev(call, "output", 3000, " Try AuraDB Free and the GraphAcademy course.")
    _ev(call, "input", 8000, " Great, that helps a lot, I will")
    _ev(call, "output", 10000, " Awesome. Good luck, and try the claw machine!")
    call.commit()
    assert call.enc.closing and call.close_at is not None
