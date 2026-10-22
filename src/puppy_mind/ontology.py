"""The schema the app actually uses, as one picture: domain ontology + the three agent-memory layers.

Memory layers (context-graph model, as in the Neo4j agent-memory design):
  short-term  what is happening now: the observation stream and the conversation (Session, Sighting, Dialogue, Utterance)
  long-term   what is known: entities and facts (Robot, Place, PhysicalObject, Person) and the spatial relationships between them
  reasoning   how answers were reached: ReasoningStep -> Lookup, with RELIED_ON links to the entities that mattered

The domain part comes from ontology/puppy_pi.json (the same file `nams-setup` sends to NAMS). Aura stores PhysicalObject as :Object and
Place as :Location; counts below are live from Aura.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ONTOLOGY_FILE = Path(__file__).parents[2] / "ontology" / "puppy_pi.json"
AURA_LABEL = {"Agent": "Robot", "Location": "Place", "Object": "Item"}  # ontology type -> label stored in Aura (NAMS owns :Location, so our places are :Place)

# agent-memory types (not in the domain file): label, memory layer, description, properties
MEMORY_TYPES = [
    ("Session", "short", "One run of the app; groups the observations and conversation of that run.", ["id", "started_at"]),
    ("Sighting", "short", "One analysed camera frame: what was seen at a place at a time.", ["id", "ts", "summary", "source", "n_objects"]),
    ("Dialogue", "short", "A chat with the robot (one per session).", ["id", "started_at"]),
    ("Utterance", "short", "One question or answer, chained in order with FOLLOWED_BY.", ["id", "role", "content", "ts"]),
    ("ReasoningStep", "reasoning", "One reasoning step: the action taken and why, with its result.", ["id", "action", "reasoning", "result", "ts"]),
    ("Lookup", "reasoning", "One memory lookup made during a step: tool name, input, output.", ["id", "name", "input", "output", "seq"]),
]
MEMORY_EDGES = [
    ("ADJACENT_TO", "Location", "Location", "long", "the robot went from one place to the other (learned, with a count)"),
    ("POSSIBLY_MOVED_FROM", "Object", "Object", "long", "HYPOTHESIS: a look-alike appeared here after the original stopped being seen elsewhere"),
    ("RAN", "Agent", "Session", "short", "the robot ran this session (a sighting's robot is reached through its session)"),
    ("CURRENTLY_IN", "Agent", "Location", "long", "where the robot is now (one edge, moved as it moves; carries heading)"),
    ("DETECTED", "Sighting", "Object", "short", "this frame showed the object (with confidence and bearing)"),
    ("MISSED", "Sighting", "Object", "short", "expected here but not detected (not proof it is gone)"),
    ("AT", "Sighting", "Location", "short", "where the frame was taken"),
    ("IN_SESSION", "Sighting", "Session", "short", ""),
    ("IN_SESSION", "Dialogue", "Session", "short", ""),
    ("HAS_UTTERANCE", "Dialogue", "Utterance", "short", ""),
    ("FOLLOWED_BY", "Utterance", "Utterance", "short", "order of the conversation"),
    ("ABOUT", "Utterance", "Object", "short", "an answer talked about this object"),
    ("HAS_REASONING", "Dialogue", "ReasoningStep", "reasoning", ""),
    ("PRODUCED", "ReasoningStep", "Utterance", "reasoning", "the step that produced this answer"),
    ("MADE_LOOKUP", "ReasoningStep", "Lookup", "reasoning", "a lookup made while reasoning"),
    ("RELIED_ON", "ReasoningStep", "Object", "reasoning", "this object informed the answer (provenance)"),
]
SPATIAL = ("ON", "INSIDE", "NEXT_TO", "NEAR", "IN_FRONT_OF", "BEHIND")


def build(counts: dict[str, int] | None = None, rel_counts: dict[str, int] | None = None, nams: dict | None = None) -> dict[str, Any]:
    spec = json.loads(ONTOLOGY_FILE.read_text(encoding="utf-8"))["ontology"]
    counts, rel_counts = counts or {}, rel_counts or {}
    nodes: list[dict] = []
    for t in spec["entity_types"]:
        nodes.append({
            "id": t["label"], "label": t["label"], "memory": "long", "pole": t.get("pole_type"),
            "description": t.get("description", ""), "properties": [p["name"] for p in t.get("properties", [])],
            "aura_label": AURA_LABEL.get(t["label"], t["label"]), "count": counts.get(AURA_LABEL.get(t["label"], t["label"]), 0),
            "pii": bool(t.get("pii_sensitive")),
        })
    for label, layer, desc, props in MEMORY_TYPES:
        nodes.append({"id": label, "label": label, "memory": layer, "pole": None, "description": desc, "properties": props,
                      "aura_label": label, "count": counts.get(label, 0), "pii": False})
    edges: list[dict] = []
    for r in spec["relationships"]:
        edges.append({"type": r["type"], "source": r["source"], "target": r["target"], "memory": "long",
                      "description": r.get("description", ""), "count": rel_counts.get(r["type"])})
    for t, a, b, layer, desc in MEMORY_EDGES:
        edges.append({"type": t, "source": a, "target": b, "memory": layer, "description": desc, "count": None})
    return {
        "source": str(ONTOLOGY_FILE.name),
        "layers": {"short": "Short-term memory", "long": "Long-term memory", "reasoning": "Reasoning memory"},
        "nodes": nodes, "edges": edges,
        "nams": nams or {},
    }
