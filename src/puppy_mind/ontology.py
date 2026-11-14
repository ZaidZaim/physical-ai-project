"""The schema the app actually uses, as one picture: domain ontology + the memory it is filed under.

Three memory layers, as in the Neo4j agent-memory model, each holding the memory types of cognitive science (see docs/MEMORY.md):
  short-term  what is happening now
                working     where the robot is and which way it faces ((Robot)-[:CURRENTLY_IN]->(Place))
                episodic    the stream of what happens: camera sightings and the conversation (Session, Sighting, Dialogue, Utterance)
  long-term   what is known
                semantic    objects, places, where things are and what they are on, which rooms connect (Robot, Place, Object, Topic
                            ...). Every fact says how it was learned (seen / told / inferred) and fades unless confirmed (knowledge.py)
                procedural  know-how learned by doing: how to walk from one room to the next (Route), how each trick goes in each
                            room (Trick)
                episodic    what is kept of an episode: the anonymous note of one booth conversation (Event)
  reasoning   how answers were reached: ReasoningStep -> Lookup, with RELIED_ON links to the entities that mattered (provenance)
Spatial memory is not a separate box: place runs through all of them (where an episode happened, where a thing is, how to get there).

The domain part comes from ontology/puppy_pi.json (the same file `nams-setup` sends to NAMS), so NAMS and Aura share one vocabulary: the
POLE+O types. NAMS writes an extracted entity's type as a label into the same database, so where both could hold the same kind of node
the app uses its own label (AURA_LABEL): Object is :Item, Location :Place, Agent :Robot, Organization :Company, Topic :Theme, Event
:Encounter. NAMS also owns :Skill (its own procedural memory for agents), so the robot's tricks are :Trick. The app never writes a
Person (booth conversations are anonymous). Counts below are live from Aura.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ONTOLOGY_FILE = Path(__file__).parents[2] / "ontology" / "puppy_pi.json"
# ontology (NAMS) type -> label stored in Aura. Never the same label as NAMS would write for that type, or the two would share nodes and
# our uniqueness constraints would reject NAMS's writes (NAMS owns :Location, so our places are :Place, and so on).
AURA_LABEL = {"Agent": "Robot", "Location": "Place", "Object": "Item", "Organization": "Company", "Topic": "Theme", "Event": "Encounter"}

# the three memory layers (a lens shows one, or all) and the memory types inside them
LAYERS = {"short": "Short-term memory", "long": "Long-term memory", "reasoning": "Reasoning memory"}
MEMORY_TYPE_NAMES = {"working": "working", "episodic": "episodic", "semantic": "semantic", "procedural": "procedural", "reasoning": "reasoning"}
LAYER_OF = {"working": "short", "episodic": "short", "semantic": "long", "procedural": "long", "reasoning": "reasoning"}
LENSES = ("long", "short", "reasoning", "all")
_TYPE_LENS = {"semantic": "long", "procedural": "long", "working": "short", "episodic": "short"}  # a memory type's name shows its layer


def lens_name(lens: str | None) -> str:
    """A lens name: long, short, reasoning or all (a memory type's name shows the layer it is in)."""
    lens = (lens or "long").strip().lower()
    return _TYPE_LENS.get(lens, lens)


# domain types and relationships (ontology/puppy_pi.json) are long-term and semantic, except the booth's conversation notes: an
# episode, kept long-term (anonymous)
DOMAIN_TYPE = {"Event": "episodic"}
DOMAIN_EDGE_TYPE = {"HELD_AT": "episodic", "HOSTED": "episodic", "DISCUSSED": "episodic", "ENJOYED": "episodic"}

# agent-memory types (not in the domain file): label, memory type (its layer is LAYER_OF[type]), description, properties
MEMORY_TYPES = [
    ("Session", "episodic", "One run of the app; groups the observations and conversation of that run.", ["id", "startedAt"]),
    ("Sighting", "episodic", "One analysed camera frame: what was seen at a place at a time.", ["id", "ts", "summary", "source", "objectCount"]),
    ("Dialogue", "episodic", "A chat with the robot (one per session).", ["id", "startedAt"]),
    ("Utterance", "episodic", "One question or answer, chained in order with FOLLOWED_BY.", ["id", "role", "content", "ts"]),
    ("Route", "procedural", "How the robot walked from one place to the next: its moves (the quickest walk and the latest), how often.",
     ["routeId", "steps", "seconds", "walks", "lastSteps", "lastWalked"]),
    ("Trick", "procedural", "A trick the robot can do (shake hands, push-ups ...); how it went per place is on PRACTISED_AT.", ["name"]),
    ("ReasoningStep", "reasoning", "One reasoning step: the action taken and why, with its result.", ["id", "action", "reasoning", "result", "ts"]),
    ("Lookup", "reasoning", "One memory lookup made during a step: tool name, input, output.", ["id", "name", "input", "output", "seq"]),
]
MEMORY_EDGES = [
    ("CURRENTLY_IN", "Agent", "Location", "working", "where the robot is now (one edge, moved as it moves; carries heading)"),
    ("ADJACENT_TO", "Location", "Location", "semantic", "the robot went from one place to the other (learned, with a count)"),
    ("POSSIBLY_MOVED_FROM", "Object", "Object", "semantic", "INFERRED, a hypothesis: a look-alike appeared here after the original stopped being seen elsewhere"),
    ("RAN", "Agent", "Session", "episodic", "the robot ran this session (a sighting's robot is reached through its session)"),
    ("DETECTED", "Sighting", "Object", "episodic", "this frame showed the object (with confidence and bearing)"),
    ("MISSED", "Sighting", "Object", "episodic", "expected here but not detected (not proof it is gone)"),
    ("TAKEN_AT", "Sighting", "Location", "episodic", "where the frame was taken"),
    ("IN_SESSION", "Sighting", "Session", "episodic", ""),
    ("IN_SESSION", "Dialogue", "Session", "episodic", ""),
    ("HAS_UTTERANCE", "Dialogue", "Utterance", "episodic", ""),
    ("FOLLOWED_BY", "Utterance", "Utterance", "episodic", "order of the conversation"),
    ("ABOUT", "Utterance", "Object", "episodic", "an answer talked about this object"),
    ("STARTS_AT", "Route", "Location", "procedural", "where the route begins"),
    ("ENDS_AT", "Route", "Location", "procedural", "where the route arrives"),
    ("PRACTISED_AT", "Trick", "Location", "procedural", "how the trick went in this place: attempts, how many went wrong (seen or told)"),
    ("HAS_REASONING", "Dialogue", "ReasoningStep", "reasoning", ""),
    ("PRODUCED", "ReasoningStep", "Utterance", "reasoning", "the step that produced this answer"),
    ("MADE_LOOKUP", "ReasoningStep", "Lookup", "reasoning", "a lookup made while reasoning"),
    ("RELIED_ON", "ReasoningStep", "Object", "reasoning", "this object informed the answer (provenance)"),
]
# how a fact was learned: on LOCATED_AT, the spatial relations and POSSIBLY_MOVED_FROM (property `source`)
PROVENANCE = {"seen": "the camera detected it", "told": "a person said so (weaker until seen)", "inferred": "worked out: a hypothesis"}
SPATIAL = ("ON", "INSIDE", "NEXT_TO", "NEAR", "IN_FRONT_OF", "BEHIND")


def build(counts: dict[str, int] | None = None, rel_counts: dict[str, int] | None = None, nams: dict | None = None) -> dict[str, Any]:
    spec = json.loads(ONTOLOGY_FILE.read_text(encoding="utf-8"))["ontology"]
    counts, rel_counts = counts or {}, rel_counts or {}
    nodes: list[dict] = []
    for t in spec["entity_types"]:
        nodes.append({
            "id": t["label"], "label": t["label"], "memory": "long", "memory_type": DOMAIN_TYPE.get(t["label"], "semantic"),
            "pole": t.get("pole_type"),
            "description": t.get("description", ""), "properties": [p["name"] for p in t.get("properties", [])],
            "aura_label": AURA_LABEL.get(t["label"], t["label"]), "count": counts.get(AURA_LABEL.get(t["label"], t["label"]), 0),
            "pii": bool(t.get("pii_sensitive")),
        })
    for label, layer, desc, props in MEMORY_TYPES:
        nodes.append({"id": label, "label": label, "memory": LAYER_OF[layer], "memory_type": layer, "pole": None, "description": desc, "properties": props,
                      "aura_label": label, "count": counts.get(label, 0), "pii": False})
    edges: list[dict] = []
    for r in spec["relationships"]:
        edges.append({"type": r["type"], "source": r["source"], "target": r["target"], "memory": "long", "memory_type": DOMAIN_EDGE_TYPE.get(r["type"], "semantic"),
                      "description": r.get("description", ""), "count": rel_counts.get(r["type"])})
    for t, a, b, layer, desc in MEMORY_EDGES:
        edges.append({"type": t, "source": a, "target": b, "memory": LAYER_OF[layer], "memory_type": layer, "description": desc, "count": None})
    return {
        "source": str(ONTOLOGY_FILE.name),
        "layers": dict(LAYERS), "memory_types": {t: LAYER_OF[t] for t in MEMORY_TYPE_NAMES}, "provenance": dict(PROVENANCE),
        "nodes": nodes, "edges": edges,
        "nams": nams or {},
    }
