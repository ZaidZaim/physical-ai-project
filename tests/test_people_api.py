"""The people API: personal data stays on this machine, the voice endpoints round-trip, the CSV is safe for spreadsheets."""
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from puppy_mind import server
from puppy_mind.people import merge_person
from tests.test_people import FakeChat, call


class FakeVoice:
    def transcribe(self, audio):
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
    server.brain.people.client = FakeChat([["Hello there"]])
    r = client.post("/api/voice/turn", files={"file": ("s.wav", b"RIFF" + b"\0" * 60 + b"hello robot", "audio/wav")})
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
