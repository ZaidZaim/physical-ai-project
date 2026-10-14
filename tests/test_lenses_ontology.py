"""Memory lenses, ontology view, demo guard, and the invariant that we never reuse labels owned by NAMS."""
import dataclasses
import re
from pathlib import Path

import pytest

from puppy_mind import ontology
from puppy_mind.config import get_settings
from puppy_mind.devfake import FakeMemory

# Labels NAMS creates in a shared database (seen via its idx_* indexes and its documented node types).
NAMS_LABELS = {"Entity", "Conversation", "Message", "AgentStep", "ToolCall", "Observation", "Reflection", "Skill",
               "SkillVersion", "SkillComponent", "SkillStep", "DistillationRun", "Alias", "PipelineMeta"}
SRC = Path(__file__).parents[1] / "src" / "puppy_mind"


def test_we_never_write_labels_that_nams_owns():
    """The Aura database can be shared with NAMS; a label clash would corrupt both sides (and reset() could delete NAMS data)."""
    cypher_label = re.compile(r"\(\w*:([A-Z][A-Za-z]+)")
    src = (SRC / "graph.py").read_text(encoding="utf-8")
    a, b = src.index("    def _migrate_labels"), src.index("    def init_schema")
    src = src[:a] + src[b:]  # the one-time migration must name the OLD label (:Observation) to rename it; nothing else may
    used = set(cypher_label.findall(src))
    assert not (used & NAMS_LABELS), used & NAMS_LABELS
    for label, *_ in ontology.MEMORY_TYPES:
        assert label not in NAMS_LABELS, label


def test_ontology_covers_three_memory_layers_and_edges_point_at_real_types():
    o = ontology.build({"Object": 3, "Sighting": 5})
    ids = {n["id"] for n in o["nodes"]}
    assert {n["memory"] for n in o["nodes"]} == {"short", "long", "reasoning"}
    assert {"Agent", "Object", "Location", "Sighting", "Utterance", "ReasoningStep", "Lookup"} <= ids
    for e in o["edges"]:
        assert e["source"] in ids and e["target"] in ids, e
    po = next(n for n in o["nodes"] if n["id"] == "Object")
    assert po["aura_label"] == "Object" and po["count"] == 3
    assert next(n for n in o["nodes"] if n["id"] == "Person")["pii"] is True


def make_fake():
    s = dataclasses.replace(get_settings(), miss_threshold=3)
    m = FakeMemory(s)
    from puppy_mind.models import Detection, Relation, Scene
    d = lambda i, l, c: Detection(idx=i, label=l, color=c, description="", zone="center", confidence=0.9)
    m.ingest(Scene(summary="t", room_type="unknown", room_confidence=0.0, objects=[d(0, "table", "brown"), d(1, "bottle", "blue")], relations=[Relation(subject=1, predicate="ON", object=0, confidence=0.9)]),
             session_id="s", location="Booth")
    oid = next(o["id"] for o in m.objs.values() if o["label"] == "bottle")
    m.record_exchange("s", "what is on the table?", "a bottle", [{"tool": "related_objects", "args": {}, "result": {}}], [oid])
    return m, oid


def test_lenses_separate_memory_types_and_share_entities_as_context():
    m, oid = make_fake()
    kinds = lambda lens: {n["kind"] for n in m.graph_snapshot(lens=lens)["nodes"]}
    assert kinds("long") == {"Object", "Location"}
    assert {"Sighting", "Utterance", "Dialogue"} <= kinds("short")
    assert {"ReasoningStep", "Lookup"} <= kinds("reasoning") and "Utterance" not in kinds("reasoning")
    assert kinds("long") <= kinds("all") and {"ReasoningStep", "Sighting"} <= kinds("all")
    snap = m.graph_snapshot(lens="reasoning")
    mem = {n["kind"]: n["memory"] for n in snap["nodes"]}
    assert mem["ReasoningStep"] == "reasoning" and mem["Object"] == "long"  # colour = memory type, even as context
    assert any(e["type"] == "RELIED_ON" and e["target"] == oid for e in snap["edges"])  # provenance: answer -> entity
    s = m.memory_summary()
    assert s["short"]["messages"] == 2 and s["reasoning"] == {"steps": 1, "tool_calls": 1} and s["long"]["objects"] == 2


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from puppy_mind import server
    from puppy_mind.brain import Brain

    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="")
    monkeypatch.setattr(server, "Brain", lambda: Brain(s, need_openai=False))
    with TestClient(server.app) as c:
        yield c


def test_api_lenses_ontology_and_demo_guard(client, monkeypatch):
    assert client.get("/api/graph?lens=bad").status_code == 400
    for lens in ("long", "short", "reasoning", "all"):
        assert client.get(f"/api/graph?lens={lens}").status_code == 200
    o = client.get("/api/ontology").json()
    assert {n["memory"] for n in o["nodes"]} == {"short", "long", "reasoning"}
    assert client.get("/api/state").json()["memory"] == "fake"
    assert client.post("/api/demo/replay?delay=0.01").status_code == 200  # allowed on throwaway memory
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "0")  # pretend this is the real graph
    r = client.post("/api/demo/replay?delay=0.01")
    assert r.status_code == 403 and "real graph" in r.json()["detail"]
