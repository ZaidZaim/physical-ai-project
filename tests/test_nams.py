"""NAMS layer against a fake NAMS server that follows the real OpenAPI shapes (verified live separately)."""
import dataclasses
import json
import re

import httpx

from puppy_mind import nams as nams_mod
from puppy_mind.config import get_settings
from puppy_mind.nams import Nams, narrative


class FakeNams:
    def __init__(self, mode="ok"):
        self.mode, self.calls, self.entities, self.n = mode, [], {}, 0
        self.headers_seen = {}

    def _id(self, p):
        self.n += 1
        return f"{p}-{self.n}"

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.headers_seen = dict(req.headers)
        if self.mode == "401":
            return httpx.Response(401, json={"error": "invalid api key"})
        if self.mode == "500":
            return httpx.Response(500, text="boom")
        path, body = req.url.path, (json.loads(req.content) if req.content else {})
        self.calls.append((req.method, path, body))
        if path == "/v1/conversations":
            return httpx.Response(201, json={"id": self._id("conv"), "metadata": body.get("metadata")})
        if re.fullmatch(r"/v1/conversations/[^/]+/messages", path):
            return httpx.Response(201, json={"id": self._id("msg"), **body})
        if path == "/v1/entities/bulk":
            results = []
            for i, e in enumerate(body["entities"]):
                key = (e["name"], e["type"])
                if key in self.entities:
                    results.append({"index": i, "merged_into": self.entities[key], "resolution": "merged"})
                else:
                    self.entities[key] = self._id("ent")
                    results.append({"index": i, "id": self.entities[key]})
            return httpx.Response(200, json={"created": len(results), "failed": 0, "merged": 0, "results": results})
        if path == "/v1/relationships/bulk":
            return httpx.Response(200, json={"created": len(body["relationships"]), "failed": 0, "reasserted": 0, "results": []})
        if req.method == "PUT" and path.startswith("/v1/entities/"):
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/reasoning/steps":
            return httpx.Response(201, json={"id": self._id("step"), **body})
        if path == "/v1/reasoning/tool-calls":
            return httpx.Response(201, json={"id": self._id("tc"), "toolName": body["toolName"]})
        if path == "/v1/entities/search":
            return httpx.Response(200, json={"entities": [{"name": "red backpack", "type": "PhysicalObject", "description": "d", "score": 0.9, "id": "x"}]})
        if path == "/v1/messages/search":
            return httpx.Response(200, json={"messages": [{"role": "user", "content": "x" * 900, "createdAt": "t", "score": 0.8, "metadata": {"kind": "observation"}}]})
        return httpx.Response(404, json={"error": path})

    def calls_to(self, method, pattern):
        return [b for m, p, b in self.calls if m == method and re.fullmatch(pattern, p)]


class FakeAura:
    def __init__(self):
        self.links = []

    def nams_links(self):
        return {}

    def set_nams_links(self, links):
        self.links += links


def make(fake: FakeNams, aura=None):
    s = dataclasses.replace(get_settings(), nams_api_key="nams_test", nams_workspace_id="ws-1", nams_base_url="https://nams.test")
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), base_url="https://nams.test",
                          headers={"Authorization": "Bearer nams_test", "X-Workspace-Id": "ws-1"})
    return Nams(s, memory=aura, client=client), s


def obs(objects, relations=(), missed=(), ts="2026-10-14T09:00:00Z", oid="obs-1"):
    return {"session": "sess", "ts": ts, "location": "Booth", "summary": "A table.", "robot": "PuppyPi", "observation_id": oid,
            "objects": objects, "relations": list(relations), "missed": [dict(m) for m in missed]}


def O(i, label, color="unknown", new=True):
    return {"id": i, "label": label, "color": color, "description": "", "zone": "center", "conf": 0.9, "new": new}


def test_observation_creates_entities_relationships_narrative_and_links_back_to_aura():
    fake, aura = FakeNams(), FakeAura()
    n, _ = make(fake, aura)
    n.submit_observation(obs([O("o1", "table", "brown"), O("o2", "bottle", "blue")], [{"a": "o2", "b": "o1", "predicate": "ON"}]))
    assert n.flush(5)
    ents = {(e["name"], e["type"]) for call in fake.calls_to("POST", "/v1/entities/bulk") for e in call["entities"]}
    assert {("brown table", "PhysicalObject"), ("blue bottle", "PhysicalObject"), ("PuppyPi", "Robot"), ("Booth", "Place")} <= ents
    rels = [r for call in fake.calls_to("POST", "/v1/relationships/bulk") for r in call["relationships"]]
    kinds = sorted(r["relationshipType"] for r in rels)
    assert kinds.count("OBSERVED") == 2 and kinds.count("LOCATED_AT") == 2 and kinds.count("ON") == 1
    on = [r for r in rels if r["relationshipType"] == "ON"][0]
    assert on["properties"]["observation_id"] == "obs-1" and on["properties"]["location"] == "Booth"
    log = fake.calls_to("POST", r"/v1/conversations/[^/]+/messages")[0]
    assert log["role"] == "system" and "blue bottle on brown table" in log["content"]
    assert log["metadata"]["kind"] == "observation"
    assert {l["id"] for l in aura.links} == {"o1", "o2"}  # Aura learns which NAMS entity mirrors each object
    assert n.status()["synced_observations"] == 1 and n.auth == "ok"
    n.close()


def test_identical_looking_objects_get_distinct_entities():
    fake = FakeNams()
    n, _ = make(fake)
    n.submit_observation(obs([O("c1", "chair", "black"), O("c2", "chair", "black")]))
    assert n.flush(5)
    names = sorted(e["name"] for call in fake.calls_to("POST", "/v1/entities/bulk") for e in call["entities"] if e["type"] == "PhysicalObject")
    assert names == ["black chair", "black chair #2"]
    n.close()


def test_possibly_gone_updates_description_once_then_back_in_view():
    fake = FakeNams()
    n, _ = make(fake)
    n.submit_observation(obs([O("l1", "laptop", "silver")], oid="a"))
    n.submit_observation(obs([], missed=[{"id": "l1", "label": "laptop", "misses": 3, "status": "possibly_gone"}], oid="b", ts="T2"))
    n.submit_observation(obs([], missed=[{"id": "l1", "label": "laptop", "misses": 4, "status": "possibly_gone"}], oid="c", ts="T3"))
    n.submit_observation(obs([O("l1", "laptop", "silver", new=False)], oid="d", ts="T4"))
    assert n.flush(5)
    puts = [b["description"] for m, p, b in fake.calls if m == "PUT"]
    assert len(puts) == 2 and "possibly gone since T2" in puts[0] and "not proof" in puts[0] and "back in view" in puts[1]
    n.close()


def test_invalid_key_switches_the_layer_off_and_stops_retrying():
    fake = FakeNams("401")
    n, _ = make(fake)
    n.submit_observation(obs([O("o1", "table")]))
    n.flush(5)
    assert n.enabled is False and n.auth == "invalid" and "401" in n.status()["last_error"]
    before = len(fake.calls)
    n.submit_observation(obs([O("o2", "chair")]))  # ignored: no more traffic
    n.flush(2)
    assert len(fake.calls) == before
    assert "not available" in n.recall("x")["error"]


def test_server_errors_are_counted_not_raised():
    fake = FakeNams("500")
    n, _ = make(fake)
    n.submit_observation(obs([O("o1", "table")]))
    assert n.flush(5)
    assert n.status()["failed"] >= 1 and n.enabled is True and n.status()["synced_observations"] == 1
    n.close()


def test_chat_and_reasoning_are_recorded_with_provenance():
    fake = FakeNams()
    n, _ = make(fake)
    n.message("sess", "user", "what changed?")
    n.message("sess", "assistant", "a backpack appeared", tools="what_changed")
    n.reasoning("sess", "what changed?", [{"tool": "what_changed", "args": {}, "result": {"summary": ["x"]}}], "a backpack appeared")
    assert n.flush(5)
    assert len(fake.calls_to("POST", "/v1/conversations")) == 1  # one chat conversation per session
    step = fake.calls_to("POST", "/v1/reasoning/steps")[0]
    assert "what changed?" in step["actionTaken"] and "what_changed" in step["reasoning"]
    tc = fake.calls_to("POST", "/v1/reasoning/tool-calls")[0]
    assert tc["toolName"] == "what_changed" and tc["stepId"].startswith("step-")
    n.close()


def test_recall_returns_trimmed_entities_and_messages():
    fake = FakeNams()
    n, _ = make(fake)
    r = n.recall("backpack")
    assert r["entities"][0]["name"] == "red backpack"
    assert len(r["messages"][0]["content"]) == 400 and r["messages"][0]["kind"] == "observation"
    n.close()


def test_narrative_is_readable_and_careful_about_missing_objects():
    p = obs([O("t", "table", "brown", new=False), O("b", "backpack", "red")], [{"a": "b", "b": "t", "predicate": "NEXT_TO"}],
            [{"id": "l", "label": "laptop", "color": "silver", "misses": 3, "status": "possibly_gone"}])
    text = narrative(p)
    assert "New: red backpack." in text and "red backpack next to brown table" in text
    assert "Possibly gone: the silver laptop" in text and "not proof" in text


def test_real_constructor_sends_bearer_and_workspace_headers(monkeypatch):
    seen = {}
    real = httpx.Client

    def spy(**kw):
        seen.update(kw)
        return real(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), **kw)

    monkeypatch.setattr(nams_mod.httpx, "Client", spy)
    s = dataclasses.replace(get_settings(), nams_api_key="nams_abc", nams_workspace_id="ws-9", nams_base_url="https://memory.example/")
    n = Nams(s)
    assert seen["headers"]["Authorization"] == "Bearer nams_abc" and seen["headers"]["X-Workspace-Id"] == "ws-9"
    assert seen["base_url"] == "https://memory.example"
    n.close()
