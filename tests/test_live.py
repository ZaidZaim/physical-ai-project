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
    return [(float(ms), i, role, text) for i, (ms, role, text) in enumerate(items)]


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


class FakeBridge:
    def __init__(self, booth):
        self.brain = type("B", (), {"s": booth.s, "booth": booth, "place_info": lambda self: {"name": "Conference hall"}})()
        self.audible_until = 0.0
        self.played = []

    async def play(self, pcm):
        self.played.append(pcm)


def test_a_live_call_fills_the_encounter_notes_the_topic_and_closes_on_goodbye():
    seen = []
    booth = BoothHost(dataclasses.replace(S, claw_machine=True), FakeMemory(S), client=FakeChat([]),
                      extract_fn=lambda tr: seen.append(tr) or {"topics": ["GraphRAG"], "questions": ["what is GraphRAG"]})
    booth.learn_in_background = booth.write_in_background = False
    call = LiveCall(FakeBridge(booth), b"\x00\x00" * 1600, "Hey PuppyPi, what is GraphRAG?")
    assert booth.conversation_active() and call.enc.messages[0]["content"] == "Hey PuppyPi, what is GraphRAG?"

    def ev(t, ms, delta):
        asyncio.run(call.on_event({"type": f"session.{t}_transcript.delta", "start_ms": ms, "delta": delta}))
    ev("output", 4000, " GraphRAG retrieves with graphs.")
    ev("input", 600, " Hey PuppyPi, what is GraphRAG?")  # its own transcript of the wake clip: no second turn
    call.commit()
    assert [m["role"] for m in call.enc.messages] == ["user", "assistant"]
    assert "what is GraphRAG" in seen[-1] and call.enc.notes["topics"] and booth.mem.list_conversations()
    ev("input", 9000, " Thanks, bye!")
    ev("output", 10000, " Bye! Try the claw machine.")
    call.commit()
    assert call.enc.closing and call.close_at is not None
    assert call.tick() is None  # the goodbye is still being played (0.5 s)
    call.close_at -= 1
    assert call.tick() == "goodbye"


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
