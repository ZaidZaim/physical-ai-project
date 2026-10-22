"""Brain.observe_scene -> ingest -> NAMS, with the in-RAM memory and a fake NAMS (no network)."""
import dataclasses

import httpx

from puppy_mind.brain import Brain
from puppy_mind.config import get_settings
from puppy_mind.models import Detection, Relation, Scene
from puppy_mind.nams import Nams
from tests.test_nams import FakeNams


def D(i, label, color, zone="center"):
    return Detection(idx=i, label=label, color=color, description="", zone=zone, confidence=0.9)


def test_observation_flows_to_nams_and_missing_objects_become_possibly_gone(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="", miss_threshold=2)
    brain = Brain(s, need_openai=False)
    fake = FakeNams()
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), base_url="https://nams.test")
    brain.nams = Nams(dataclasses.replace(s, nams_api_key="k"), memory=brain.mem, client=client)
    try:
        full = Scene(summary="table, laptop", room_type="unknown", room_confidence=0.0, objects=[D(0, "table", "brown"), D(1, "laptop", "silver")],
                     relations=[Relation(subject=1, predicate="ON", object=0, confidence=0.9)])
        brain.observe_scene(full, location="Booth")
        empty_table = Scene(summary="table only", room_type="unknown", room_confidence=0.0, objects=[D(0, "table", "brown")], relations=[])
        brain.observe_scene(empty_table, location="Booth")
        brain.observe_scene(empty_table, location="Booth")  # laptop missed twice => possibly gone
        assert brain.nams.flush(5)
        types = {(e["name"], e["type"]) for c in fake.calls_to("POST", "/v1/entities/bulk") for e in c["entities"]}
        assert any(n.startswith("silver laptop ") and t == "Object" for n, t in types) and any(n.startswith("Booth ") and t == "Location" for n, t in types) and ("PuppyPi", "Agent") in types
        rels = [r["relationshipType"] for c in fake.calls_to("POST", "/v1/relationships/bulk") for r in c["relationships"]]
        assert "ON" in rels and "OBSERVED" in rels
        descs = [b["description"] for m, p, b in fake.calls if m == "PUT"]
        assert len(descs) == 1 and "possibly gone" in descs[0]
        logs = fake.calls_to("POST", r"/v1/conversations/[^/]+/messages")
        assert len(logs) == 3 and "Possibly gone: the silver laptop" in logs[-1]["content"]
        assert brain.nams.status()["synced_observations"] == 3
    finally:
        brain.nams.close()
        brain.close()


def test_merging_places_in_the_brain_tells_nams(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="")
    brain = Brain(s, need_openai=False)
    fake = FakeNams()
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), base_url="https://nams.test")
    brain.nams = Nams(dataclasses.replace(s, nams_api_key="k"), memory=brain.mem, client=client)
    try:
        brain.observe_scene(Scene(summary="stairs", room_type="garage", room_confidence=0.9, objects=[D(0, "stairs", "brown"), D(1, "cabinet", "black")], relations=[]), location="Garage")
        brain.observe_scene(Scene(summary="door", room_type="hallway", room_confidence=0.9, objects=[D(0, "door", "brown")], relations=[]), location="Hallway")
        assert brain.nams.flush(5)
        garage = next(v for (nm, t), v in fake.entities.items() if nm.startswith("Garage ") and t == "Location")
        hallway = next(v for (nm, t), v in fake.entities.items() if nm.startswith("Hallway ") and t == "Location")
        moved = {o["id"] for o in brain.mem.objs.values() if o["loc"] == "Garage"}
        fake.calls.clear()
        assert brain.merge_places("Garage", "Hallway")
        assert brain.nams.flush(5)
        rels = [r for c in fake.calls_to("POST", "/v1/relationships/bulk") for r in c["relationships"]]
        assert {r["sourceId"] for r in rels} == {brain.nams._links[o]["id"] for o in moved} and {r["targetId"] for r in rels} == {hallway}
        assert [p.rsplit("/", 1)[-1] for m, p, b in fake.calls if m == "DELETE"] == [garage]
        assert not brain.merge_places("Garage", "Hallway")  # gone now: nothing else is sent
        assert not brain.merge_places("Nowhere", "Hallway") and "Nowhere" not in brain.mem.places
    finally:
        brain.close()
