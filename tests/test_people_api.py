"""The people API: personal data stays on this machine, the voice endpoints round-trip, the CSV is safe for spreadsheets."""
from types import SimpleNamespace as NS

import pytest
from urllib.parse import unquote
from fastapi.testclient import TestClient

from puppy_mind import server
from puppy_mind.people import merge_person
from tests.test_people import FakeChat, call


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
        server.brain.people.client = FakeChat([])
        server.brain.voice = FakeVoice()
        server.brain.drive.action_busy_s = 0.05  # the wake trick finishes at once in tests
        server.brain.people.write_in_background = False
        yield c


def seed(client, **facts):
    p = merge_person(None, facts, facts.pop("name", "Ana Berg") if "name" in facts else "Ana Berg")
    server.brain.mem.save_person(p, encounter_id="e1", place="Booth", summary="talked graphs", ts="2026-10-31T10:00:00+00:00")
    return p["person_id"]


def test_people_endpoints_roundtrip_followup_and_forget(client):
    pid = seed(client, company="Acme", interests=["graphrag"], wants_follow_up=True)
    d = client.get("/api/people").json()
    assert d["counts"]["people"] == 1 and d["people"][0]["name"] == "Ana Berg"
    assert client.get("/api/people", params={"q": "graphrag"}).json()["people"][0]["person_id"] == pid
    assert client.get("/api/people", params={"q": "nobody"}).json()["people"] == []
    assert client.get(f"/api/people/{pid}").json()["encounters"][0]["place"] == "Booth"
    g = client.get("/api/people/graph").json()
    assert {n["kind"] for n in g["nodes"]} == {"Person", "Company", "Theme"}
    assert client.post(f"/api/people/{pid}/followup", json={"done": True}).status_code == 200
    assert client.get(f"/api/people/{pid}").json()["followed_up"] is True
    assert client.delete(f"/api/people/{pid}").status_code == 200
    assert client.get(f"/api/people/{pid}").status_code == 404 and client.delete(f"/api/people/{pid}").status_code == 404


def test_csv_export_neutralises_spreadsheet_formulas(client):
    seed(client, company="=HYPERLINK(\"http://evil\")", role="CTO", interests=["a", "b"])
    r = client.get("/api/people/export.csv")
    assert r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("name,company,role") and "'=HYPERLINK" in lines[1] and ",a; b," in lines[1]


def test_personal_data_is_not_served_to_other_machines_without_token(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    with TestClient(server.app, client=("192.168.1.50", 5000)) as c:
        server.brain.people.client = FakeChat([])
        for method, path in (("get", "/api/people"), ("get", "/api/people/export.csv"), ("get", "/api/people/graph"),
                             ("post", "/api/converse")):
            r = getattr(c, method)(path, **({"json": {"text": "hi"}} if method == "post" else {}))
            assert r.status_code == 403, path
        server.brain.s = type(server.brain.s)(**{**server.brain.s.__dict__, "brain_token": "s3cret"})
        assert c.get("/api/people", headers={"X-Brain-Token": "s3cret"}).status_code == 200
        assert c.get("/api/people", params={"token": "s3cret"}).status_code == 200
        assert c.get("/api/people", headers={"X-Brain-Token": "wrong"}).status_code == 403


def test_converse_endpoint(client):
    server.brain.people.client = FakeChat([["Hello, I'm PuppyPi!"]])
    r = client.post("/api/converse", json={"text": "hi"}).json()
    assert r["reply"] == "Hello, I'm PuppyPi!" and r["encounter_id"] and r["consent"] is None


def test_voice_turn_returns_audio_and_headers_and_204_on_silence(client):
    client.post("/api/listen")  # the People tab is open
    server.brain.people.client = FakeChat([["Hello there"]])
    r = client.post("/api/voice/turn", files={"file": ("s.wav", b"RIFF" + b"\0" * 60 + b"Hey PuppyPi, hello robot", "audio/wav")})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == b"RIFFfake-wav:Hello there"
    assert r.headers["x-encounter"] and "hello%20robot" in r.headers["x-heard"] and r.headers["x-reply"] == "Hello%20there"
    assert client.post("/api/voice/turn", files={"file": ("s.wav", b"tiny", "audio/wav")}).status_code == 204


def test_voice_next_plays_a_waiting_greeting_once(client):
    assert client.get("/api/voice/next").status_code == 204
    server.brain.people.outbox.say("Hi there!")
    r = client.get("/api/voice/next")
    assert r.status_code == 200 and r.content.endswith(b"Hi there!") and client.get("/api/voice/next").status_code == 204


def test_voice_endpoints_need_the_robot_token_when_one_is_set(client):
    server.brain.s = type(server.brain.s)(**{**server.brain.s.__dict__, "brain_token": "s3cret"})
    assert client.get("/api/voice/next").status_code == 401
    assert client.get("/api/voice/next", headers={"X-Brain-Token": "s3cret"}).status_code == 204


def test_booth_mode_and_tricks_endpoints(client):
    b = client.get("/api/booth").json()
    assert b["on"] is False and "boxing" in b["tricks"]
    assert client.post("/api/gesture", json={"name": "boxing"}).json()["reason"] == "booth mode is off"
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


def test_aura_people_queries_never_pass_a_parameter_named_like_the_query_helper_argument():
    """The Aura implementation calls Memory._read(q, **params): a Cypher parameter named `q` crashed /api/people on the real
    database (the in-RAM twin has no such helper, so the other tests could not see it). Drive every method through the same
    helper signature."""
    from puppy_mind.people_graph import PeopleGraph

    class Stub(PeopleGraph):
        def __init__(self):
            self.calls = []

        def _read(self, q, **p):
            self.calls.append(p)
            return [{"people": 0, "hot": 0, "warm": 0, "encounters": 0}] if "count {" in q else []

        _write = _read

    s = Stub()
    s.list_people("hot", "graph")
    s.list_people()
    s.find_people("ana")
    s.get_person("psn-1")
    s.people_graph()
    s.people_counts()
    s.set_followed_up("psn-1", True)
    s.forget_person("psn-1")
    s.save_person({"person_id": "psn-1", "name": "Ana", "company": "Acme", "interests": ["x"], "enjoyed": ["y"]},
                  encounter_id="e", place="Booth", summary="", ts="2026-10-31T10:00:00+00:00")
    assert s.calls and all("q" not in p for p in s.calls)


def test_speak_here_endpoint_returns_wav_and_is_local_only(client, monkeypatch):
    r = client.post("/api/voice/say", json={"text": "Hello there"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content.endswith(b"Hello there")
    with TestClient(server.app, client=("192.168.1.50", 5000)) as c:
        server.brain.voice = FakeVoice()
        assert c.post("/api/voice/say", json={"text": "hi"}).status_code == 403


def test_hold_to_talk_webm_reaches_transcription_with_its_format(client):
    server.brain.people.client = FakeChat([["Hi! What's your name?"]])
    r = client.post("/api/voice/turn", files={"file": ("speech.webm", bytes(64) + b"hello robot", "audio/webm")},
                    data={"require_wake": "false"})  # Hold to talk needs no wake word
    assert r.status_code == 200 and server.brain.voice.filenames[-1] == "speech.webm"


def test_phrases_the_speech_model_invents_for_silence_never_reach_the_wizard(client):
    from puppy_mind.voice import heard_nothing

    assert heard_nothing("Thank you for watching. Please subscribe to my channel and click the bell.")
    assert heard_nothing("Thanks for watching!") and heard_nothing("you") and heard_nothing("")
    assert not heard_nothing("Hi, I'm Zaid from Neo4j") and not heard_nothing("Thank you") and not heard_nothing("Yes")
    assert not heard_nothing("Docs are on neo4j.com")
    server.brain.people.client = FakeChat([])  # the Wizard must not be called at all
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
    client.post("/api/listen")  # the People tab is open
    server.brain.people.client = FakeChat([["Hi! What would you like to know?"], ["GraphRAG uses a graph."]])
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
    client.post("/api/listen")  # the People tab is open
    server.brain.people.client = FakeChat([["Hi!"], ["Hello again, who are you?"]])
    post = lambda said, enc=None: client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + said.encode(), "audio/wav")},
                                              data={"encounter_id": enc} if enc else {})
    enc = post("Hey PuppyPi").headers["x-encounter"]
    server.brain.people.encounters[enc].last -= 999  # long quiet: the visitor left
    assert post("what about vector search", enc).status_code == 204
    r = post("Hey PuppyPi, I'm new here", enc)
    assert r.status_code == 200 and r.headers["x-encounter"] != enc  # most likely someone else: a new conversation


def test_no_trick_by_default_when_greeting():
    from puppy_mind.config import Settings

    assert Settings().greet_action == "" or __import__("os").getenv("GREET_ACTION")


def test_next_step_endpoint_and_export(client):
    pid = seed(client, company="Acme")
    assert client.post(f"/api/people/{pid}/next_step", json={"text": "Book a demo with Ana"}).status_code == 200
    assert client.get(f"/api/people/{pid}").json()["next_step"] == "Book a demo with Ana"
    assert "next_step" in client.get("/api/people/export.csv").text.splitlines()[0]
    assert client.get("/api/people/graph").json()["nodes"][0].get("score") is not None


def test_the_robot_listens_only_while_the_people_tab_is_open(client):
    server.brain.people.client = FakeChat([["Hi!"]])
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

    server.brain.people.client = FakeChat([["Hi! What would you like to know?"]])
    client.post("/api/listen")
    assert server.brain.drive.current()["pitch"] == 0.0
    r = client.post("/api/voice/turn", files={"file": ("s.wav", bytes(64) + b"Hey PuppyPi!", "audio/wav")})
    assert r.status_code == 200
    d = server.brain.drive.current()
    assert d["action"] == "boxing" and d["action_seq"] == 1  # the trick came first
    assert math.isclose(d["pitch"], math.radians(25))  # then it looks up at the visitor, before it speaks
    assert client.post("/api/listen").json()["state"] == "conversation"
    for e in server.brain.people.encounters.values():
        e.last -= 999  # the conversation went quiet
    client.get("/api/voice/next")
    assert server.brain.drive.current()["pitch"] == 0.0 and client.post("/api/listen").json()["state"] == "listening"
