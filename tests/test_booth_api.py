"""The booth API: insights stay on this machine, the voice endpoints round-trip, the wake word and the trick."""
from types import SimpleNamespace as NS

import pytest
from urllib.parse import unquote
from fastapi.testclient import TestClient

from puppy_mind import server
from tests.test_booth import FakeChat


class FakeVoice:
    def __init__(self):
        self.filenames = []

    def transcribe(self, audio, filename="speech.wav"):
        self.filenames.append(filename)
        return audio.decode(errors="ignore")[-40:].strip() if len(audio) > 44 else ""

    def speak(self, text):
        return b"RIFFfake-wav:" + text.encode()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    monkeypatch.delenv("ROSBRIDGE_URL", raising=False)
    with TestClient(server.app, client=("127.0.0.1", 5000)) as c:
        server.brain.booth.client = FakeChat([])
        server.brain.voice = FakeVoice()
        server.brain.drive.action_busy_s = 0.05  # the wake trick finishes at once in tests
        server.brain.booth.write_in_background = False
        yield c


def test_booth_insight_endpoints_and_removing_a_note(client):
    server.brain.mem.save_conversation({"id": "c1", "topics": ["graphrag", "agent memory"], "need": "chat over manuals",
                                        "familiarity": "curious", "questions": ["what is graphrag"]}, place="Booth", ts="2026-11-02T10:00:00Z")
    i = client.get("/api/booth/insights").json()
    assert i["conversations"] == 1 and i["needs"][0]["need"] == "chat over manuals" and i["needs"][0]["id"] == "c1"
    assert {n["id"] for n in client.get("/api/booth/graph").json()["nodes"]} == {"robot", "conv:c1", "topic:graphrag", "topic:agent memory"}
    assert client.get("/api/booth/conversations").json()["conversations"][0]["place"] == "Booth"
    assert client.delete("/api/booth/conversations/c1").status_code == 200
    assert client.delete("/api/booth/conversations/c1").status_code == 404 and client.get("/api/booth/insights").json()["conversations"] == 0
    for gone in ("/api/people", "/api/people/graph", "/api/people/export.csv"):  # lead generation is on hold: no people endpoints
        assert client.get(gone).status_code == 404, gone


def test_booth_data_is_not_served_to_other_machines_without_token(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    with TestClient(server.app, client=("192.168.1.50", 5000)) as c:
        server.brain.booth.client = FakeChat([])
        for method, path in (("get", "/api/booth/insights"), ("get", "/api/booth/graph"), ("get", "/api/booth/conversations"),
                             ("post", "/api/converse")):
            r = getattr(c, method)(path, **({"json": {"text": "hi"}} if method == "post" else {}))
            assert r.status_code == 403, path
        server.brain.s = type(server.brain.s)(**{**server.brain.s.__dict__, "brain_token": "s3cret"})
        assert c.get("/api/booth/insights", headers={"X-Brain-Token": "s3cret"}).status_code == 200
        assert c.get("/api/booth/insights", params={"token": "s3cret"}).status_code == 200
        assert c.get("/api/booth/insights", headers={"X-Brain-Token": "wrong"}).status_code == 403


def test_aura_booth_queries_never_pass_a_parameter_named_like_the_query_helper_argument():
    """The Aura implementation calls Memory._read(q, **params): a Cypher parameter named `q` crashed an endpoint on the real
    database (the in-RAM twin has no such helper, so the other tests could not see it). Drive every method through the same
    helper signature."""
    from puppy_mind.booth_graph import BoothGraph

    class Stub(BoothGraph):
        def __init__(self):
            self.calls = []

        def _read(self, q, **p):
            self.calls.append(p)
            return []

        _write = _read

    s = Stub()
    s.save_conversation({"id": "c1", "topics": ["x"], "enjoyed": ["y"], "questions": ["z"]}, place="Booth", ts="2026-11-02T10:00:00Z")
    s.list_conversations()
    s.booth_insights()
    s.booth_graph()
    s.delete_conversation("c1")
    assert s.calls and all("q" not in p for p in s.calls)


def test_converse_endpoint(client):
    server.brain.booth.client = FakeChat([["Hello, I'm PuppyPi!"]])
    r = client.post("/api/converse", json={"text": "hi"}).json()
    assert r["reply"].startswith("Hello, I'm PuppyPi!") and r["encounter_id"] and "consent" not in r


def test_voice_turn_returns_audio_and_headers_and_204_on_silence(client):
    client.post("/api/listen")  # the Booth tab is open
    server.brain.booth.client = FakeChat([["Hello there"]])
    r = client.post("/api/voice/turn", files={"file": ("s.wav", b"RIFF" + b"\0" * 60 + b"Hey PuppyPi, hello robot", "audio/wav")})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content.startswith(b"RIFFfake-wav:Hello there")
    assert "think" in unquote(r.headers["x-timing"])  # where the time went, for the robot's log
    assert r.headers["x-encounter"] and "hello%20robot" in r.headers["x-heard"] and r.headers["x-reply"].startswith("Hello%20there")
    assert client.post("/api/voice/turn", files={"file": ("s.wav", b"tiny", "audio/wav")}).status_code == 204


def test_voice_next_plays_a_waiting_greeting_once(client):
    assert client.get("/api/voice/next").status_code == 204
    server.brain.booth.outbox.say("Hi there!")
    r = client.get("/api/voice/next")
    assert r.status_code == 200 and r.content.endswith(b"Hi there!") and client.get("/api/voice/next").status_code == 204


def test_voice_endpoints_need_the_robot_token_when_one_is_set(client):
    server.brain.s = type(server.brain.s)(**{**server.brain.s.__dict__, "brain_token": "s3cret"})
    assert client.get("/api/voice/next").status_code == 401
    assert client.get("/api/voice/next", headers={"X-Brain-Token": "s3cret"}).status_code == 204


def test_booth_mode_and_tricks_endpoints(client):
    b = client.get("/api/booth").json()
    assert b["on"] is False and "boxing" in b["tricks"]
    assert server.brain.drive.request_action("boxing")["reason"] == "booth mode is off"  # automatic tricks need booth mode
    b = client.post("/api/booth", json={"on": True, "look_up_deg": 12}).json()
    assert b["on"] is True and b["look_up_deg"] == 12
    assert client.get("/api/drive/cmd").json()["pitch"] > 0
    r = client.post("/api/gesture", json={"name": "boxing"}).json()
    assert r["ok"] and client.get("/api/drive/cmd").json()["action"] == "boxing"


def test_booth_controls_are_local_only(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    with TestClient(server.app, client=("192.168.1.50", 5000)) as c:
        assert c.post("/api/booth", json={"on": True}).status_code == 403
        assert c.post("/api/gesture", json={"name": "wave"}).status_code == 403


def test_speak_here_endpoint_returns_wav_and_is_local_only(client, monkeypatch):
    r = client.post("/api/voice/say", json={"text": "Hello there"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content.endswith(b"Hello there")
    with TestClient(server.app, client=("192.168.1.50", 5000)) as c:
        server.brain.voice = FakeVoice()
        assert c.post("/api/voice/say", json={"text": "hi"}).status_code == 403


def test_hold_to_talk_webm_reaches_transcription_with_its_format(client):
    server.brain.booth.client = FakeChat([["Hi! What's your name?"]])
    r = client.post("/api/voice/turn", files={"file": ("speech.webm", bytes(64) + b"hello robot", "audio/webm")},
                    data={"require_wake": "false"})  # Hold to talk needs no wake word
    assert r.status_code == 200 and server.brain.voice.filenames[-1] == "speech.webm"


def test_phrases_the_speech_model_invents_for_silence_never_reach_the_wizard(client):
    from puppy_mind.voice import heard_nothing

    assert heard_nothing("Thank you for watching. Please subscribe to my channel and click the bell.")
    assert heard_nothing("Thanks for watching!") and heard_nothing("you") and heard_nothing("")
    assert not heard_nothing("Hi, I'm Sam from a startup") and not heard_nothing("Thank you") and not heard_nothing("Yes")
    assert not heard_nothing("Docs are on neo4j.com")
    server.brain.booth.client = FakeChat([])  # the Wizard must not be called at all
    r = client.post("/api/voice/turn", files={"file": ("speech.webm", bytes(64) + b"Thanks for watching!", "audio/webm")})
    assert r.status_code == 204 and "watching" in r.headers["x-heard"].lower()


def test_recordings_made_while_the_robot_moves_are_dropped_without_transcribing(client):
    # seen on the robot: the boxing trick's servo noise came through as "Thanks for watching!" just before each greeting
    server.brain.drive.set_booth(True, 10)
    assert server.brain.drive.request_action("boxing")["ok"]
    before = len(server.brain.voice.filenames)
    r = client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + b"hello robot", "audio/wav")})
    assert r.status_code == 204 and "moving" in r.headers["x-heard"] and len(server.brain.voice.filenames) == before


def test_the_robot_answers_only_after_hey_puppypi_then_follow_ups_need_none(client):
    import dataclasses

    server.brain.s = dataclasses.replace(server.brain.s, wake_action="")  # about the wake word, not the trick
    client.post("/api/listen")  # the Booth tab is open
    server.brain.booth.client = FakeChat([["Hi! What would you like to know?"], ["GraphRAG uses a graph."]])
    post = lambda said, enc=None: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")},
                                              data={"encounter_id": enc} if enc else {})
    r = post("so I told him about the party")  # hall chatter: not answered, and not echoed back
    from urllib.parse import unquote
    assert r.status_code == 204 and "party" not in unquote(r.headers["x-heard"]) and "Hey PuppyPi" in unquote(r.headers["x-heard"])
    r = post("Hey Puppy Pi!")
    assert r.status_code == 200
    enc = r.headers["x-encounter"]
    r = post("What is GraphRAG?", enc)  # the conversation is open: no wake word needed
    assert r.status_code == 200 and r.headers["x-encounter"] == enc


def test_after_the_conversation_went_quiet_the_wake_word_starts_a_fresh_one(client):
    import dataclasses

    server.brain.s = dataclasses.replace(server.brain.s, wake_action="")  # about the wake word, not the trick
    client.post("/api/listen")  # the Booth tab is open
    server.brain.booth.client = FakeChat([["Hi!"], ["Hello again, who are you?"]])
    post = lambda said, enc=None: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")},
                                              data={"encounter_id": enc} if enc else {})
    enc = post("Hey PuppyPi").headers["x-encounter"]
    server.brain.booth.encounters[enc].last -= 999  # long quiet: the visitor left
    assert post("what about vector search", enc).status_code == 204
    r = post("Hey PuppyPi, I'm new here", enc)
    assert r.status_code == 200 and r.headers["x-encounter"] != enc  # most likely someone else: a new conversation


def test_no_trick_by_default_when_greeting():
    from puppy_mind.config import Settings

    assert Settings().greet_action == "" or __import__("os").getenv("GREET_ACTION")


def test_the_booth_tab_is_optional_the_robot_listens_and_hears_the_goodbye_without_it(client):
    # seen at the booth: the tab went to the background mid-conversation, the robot stopped listening and missed the goodbye
    import dataclasses

    server.brain.s = dataclasses.replace(server.brain.s, listen_always=True, wake_action="")
    server.brain.booth.s = server.brain.s
    server.brain._listen_until = 0  # no tab has pinged
    server.brain.booth.client = FakeChat([["Hi!"], ["Goodbye!"]])
    post = lambda said, enc=None: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")},
                                              data={"encounter_id": enc} if enc else {})
    r = post("Hey PuppyPi")
    assert r.status_code == 200
    client.get("/api/voice/next")  # the robot's next poll: booth mode (tricks, tilt) comes on without the tab
    assert server.brain.drive.booth
    r = post("Thanks, I need to run.", r.headers["x-encounter"])
    assert r.status_code == 200 and not server.brain.booth.conversation_active()


def test_with_listen_always_off_the_robot_listens_only_while_the_booth_tab_is_open(client):
    import dataclasses

    server.brain.s = dataclasses.replace(server.brain.s, listen_always=False)
    server.brain.booth.client = FakeChat([["Hi!"]])
    post = lambda said: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")})
    r = post("Hey PuppyPi")
    assert r.status_code == 204 and "not listening" in unquote(r.headers["x-heard"])
    st = client.post("/api/listen").json()
    assert st["state"] == "listening" and st["wake_word"] == "Hey PuppyPi" and server.brain.drive.booth
    server.brain._listen_until = 0  # the tab was closed 15 s ago
    client.get("/api/voice/next")  # the robot's next poll
    assert not server.brain.drive.booth  # booth mode (tricks, tilt) is off again


def test_hey_puppypi_does_the_trick_then_looks_up_then_levels_when_quiet(client):
    import math

    server.brain.booth.client = FakeChat([["Hi! What would you like to know?"]])
    client.post("/api/listen")
    assert server.brain.drive.current()["pitch"] == 0.0
    r = client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + b"Hey PuppyPi!", "audio/wav")})
    assert r.status_code == 200
    d = server.brain.drive.current()
    assert d["action"] == "boxing" and d["action_seq"] == 1  # the trick came first
    assert math.isclose(d["pitch"], math.radians(25))  # then it looks up at the visitor, before it speaks
    assert client.post("/api/listen").json()["state"] == "conversation"
    for e in server.brain.booth.encounters.values():
        e.last -= 999  # the conversation went quiet
    client.get("/api/voice/next")
    assert server.brain.drive.current()["pitch"] == 0.0 and client.post("/api/listen").json()["state"] == "listening"



def test_unclear_audio_inside_a_conversation_gets_a_question_back_outside_silence(client):
    # seen at the booth: a quiet answer came out as "Thanks for watching!", was dropped, and the conversation lapsed
    import dataclasses

    server.brain.s = dataclasses.replace(server.brain.s, wake_action="")
    server.brain.booth.client = FakeChat([["Hi! What would you like to know?"]])
    client.post("/api/listen")
    post = lambda said, enc=None: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")},
                                              data={"encounter_id": enc} if enc else {})
    assert post("Thanks for watching!").status_code == 204  # no conversation: hall noise stays unanswered
    enc = post("Hey PuppyPi").headers["x-encounter"]
    r = post("Thanks for watching!", enc)
    assert r.status_code == 200 and unquote(r.headers["x-reply"]) == "Sorry, I didn't catch that. Could you say it again?"
    assert r.headers["x-encounter"] == enc
    assert post("Thanks for watching!", enc).status_code == 204  # not every few seconds


def test_camera_analysis_pauses_while_a_visitor_talks(client):
    # seen on a phone hotspot: a frame upload every 5 s made replies take 13-22 s instead of 6
    import dataclasses

    server.brain.s = dataclasses.replace(server.brain.s, wake_action="", listen_always=True, watch_interval_s=0)
    server.brain.booth.s = server.brain.s
    server.brain.perception = object()  # something to analyse with (never called: the check comes first)
    server.brain.booth.client = FakeChat([["Hi!"], ["Bye!"]])
    post = lambda said, enc=None: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")},
                                              data={"encounter_id": enc} if enc else {})
    enc = post("Hey PuppyPi").headers["x-encounter"]
    assert server.brain.submit_frame(b"jpeg") is False and server.brain.frame == b"jpeg"  # shown, not analysed
    post("Thanks!", enc)
    assert not server.brain.booth.conversation_active()
    server.brain.perception = None


def test_the_robot_can_download_its_boot_service_and_start_script(client):
    # the robot starts its voice client by itself at boot: puppy-voice.service runs start_voice_auto.sh (fetched from the laptop)
    for name in ("start_voice_auto.sh", "puppy-voice.service", "puppy_voice.py"):
        r = client.get(f"/robot/{name}")
        assert r.status_code == 200 and b"\r\n" not in r.content, name  # bash on the robot breaks on Windows line endings
    assert b"ExecStart=/bin/bash /home/pi/start_voice_auto.sh" in client.get("/robot/puppy-voice.service").content
    assert client.get("/robot/.env").status_code == 404 and client.get("/robot/server.py").status_code == 404


def test_trick_buttons_play_one_after_another_and_the_big_ones_only_from_the_buttons(client):
    from puppy_mind.drive import BIG_TRICKS, MANUAL_GAP_S, TRICKS

    b = client.get("/api/booth").json()
    assert {"sit", "stand", "lie_down", "stretch", "shake_head"} <= set(b["tricks"]) and set(b["big_tricks"]) == set(BIG_TRICKS)
    drive = server.brain.drive
    drive.set_booth(False)
    r = client.post("/api/gesture", json={"name": "sit"}).json()
    assert r["ok"] and drive.booth  # a button turns booth mode on (it used to say "booth mode is off")
    assert client.post("/api/gesture", json={"name": "wave"}).json()["reason"] == "a trick is already running"
    drive.action_done()  # the robot reports the trick finished
    assert client.post("/api/gesture", json={"name": "wave"}).json()["reason"] == "the last trick is still settling"
    drive._action_until -= MANUAL_GAP_S + 0.1
    drive._moving_until = 0
    assert client.post("/api/gesture", json={"name": "moonwalk"}).json()["ok"]  # no 60 s cooldown for buttons
    # automatic tricks (the wake word, a visitor asking) keep the cooldown and never do the big ones
    drive._action_until = drive._moving_until = 0
    assert drive.request_action("wave")["reason"] == "tricks are rate-limited"
    assert drive.request_action("push-up")["reason"] == "unknown trick"
    from puppy_mind.booth import TOOLS
    enum = next(t for t in TOOLS if t["function"]["name"] == "do_trick")["function"]["parameters"]["properties"]["name"]["enum"]
    assert enum == list(TRICKS) and not set(enum) & set(BIG_TRICKS)
