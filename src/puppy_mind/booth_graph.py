"""Booth memory in the graph: what visitors asked about and need, never who they are.

Model (camelCase keys, every MERGE key has a uniqueness constraint)
  (Encounter {id, ts, event, need, familiarity, questions, summary, turns})   one anonymous booth conversation
  (Encounter)-[:DISCUSSED]->(Theme {name})     a topic the visitor raised or asked about
  (Encounter)-[:ENJOYED]->(Theme)              something they enjoyed at the event
  (Encounter)-[:HELD_AT]->(Place)

No Person, Company, name, email or job title is ever written: lead generation (which kept those after a spoken yes) is on hold for
compliance and parked on branch parked/booth-lead-generation.

`BoothGraph` is the Aura implementation (mixed into graph.Memory); `FakeBooth` is the in-RAM twin for PUPPY_FAKE_MEMORY / tests.
Both expose the same methods and return the same shapes (snake_case).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

BOOTH_CONSTRAINTS = [
    "CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (n:Encounter) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (n:Theme) REQUIRE n.name IS UNIQUE",
]
BOOTH_LABELS = ("Encounter", "Theme")  # deletion order for reset()
FAMILIARITY = ("none", "curious", "user", "expert")

_RETURN = """
    e.id AS id, toString(e.ts) AS ts, e.event AS event, e.need AS need, e.familiarity AS familiarity,
    coalesce(e.questions, []) AS questions, e.summary AS summary, coalesce(e.turns, 0) AS turns,
    [(e)-[:DISCUSSED]->(t:Theme) | t.name] AS topics, [(e)-[:ENJOYED]->(t:Theme) | t.name] AS enjoyed,
    head([(e)-[:HELD_AT]->(pl:Place) | pl.name]) AS place
"""


def _insights(convs: list[dict]) -> dict[str, Any]:
    """Counts the booth team reads at a glance. Pure: the same for Aura and RAM."""
    topics = Counter(t for c in convs for t in c["topics"])
    enjoyed = Counter(t for c in convs for t in c["enjoyed"])
    fam = Counter(c["familiarity"] for c in convs if c.get("familiarity") in FAMILIARITY)
    return {
        "conversations": len(convs), "questions": sum(len(c["questions"]) for c in convs), "topics": len(topics),
        "top_topics": [{"name": n, "count": k} for n, k in topics.most_common(15)],
        "top_enjoyed": [{"name": n, "count": k} for n, k in enjoyed.most_common(8)],
        "familiarity": {f: fam.get(f, 0) for f in FAMILIARITY},
        "needs": [{"id": c["id"], "need": c["need"], "ts": c["ts"], "topics": c["topics"]} for c in convs if c.get("need")][:30],
        "recent_questions": [q for c in convs for q in c["questions"]][:30],
    }


def _graph(convs: list[dict], robot: str = "PuppyPi", limit: int = 60) -> dict[str, Any]:
    """The booth as a graph: the robot HOSTED each (anonymous) conversation, which DISCUSSED topics and ENJOYED highlights. Topics
    are shared, so the ones many conversations raised become hubs. The latest `limit` conversations."""
    convs = convs[:limit]
    count = Counter(t for c in convs for t in [*c["topics"], *c["enjoyed"]])
    nodes = [{"id": "robot", "kind": "Robot", "label": robot}]
    edges: list[dict] = []
    for c in convs:
        cid = "conv:" + c["id"]
        when = (c.get("ts") or "")[11:16]
        nodes.append({"id": cid, "kind": "Conversation", "conv_id": c["id"], "label": (c.get("need") or f"conversation {when}")[:48],
                      "title": c.get("summary") or c.get("need") or "", "ts": c.get("ts"), "familiarity": c.get("familiarity"),
                      "need": c.get("need"), "summary": c.get("summary"), "questions": c.get("questions") or [],
                      "topics": c.get("topics") or [], "enjoyed": c.get("enjoyed") or [], "turns": c.get("turns") or 0})
        edges.append({"source": "robot", "target": cid, "type": "HOSTED"})
        edges += [{"source": cid, "target": "topic:" + t, "type": "DISCUSSED"} for t in c["topics"]]
        edges += [{"source": cid, "target": "topic:" + t, "type": "ENJOYED"} for t in c["enjoyed"] if t not in c["topics"]]
    nodes += [{"id": "topic:" + t, "kind": "Topic", "label": t, "count": k} for t, k in count.items()]
    return {"nodes": nodes, "edges": edges}


class BoothGraph:
    """Mixin for graph.Memory (uses its _read/_write)."""

    def init_booth_schema(self) -> None:
        from neo4j.exceptions import ClientError

        for q in BOOTH_CONSTRAINTS:
            try:
                self._write(q)
            except ClientError as e:
                if "already exist" not in (e.message or "").lower():
                    raise

    def save_conversation(self, c: dict[str, Any], *, place: str | None, ts: str) -> None:
        """One statement, one round trip (from the booth each Aura round trip took ~1 s). The lists replace what was there: the
        record is the whole conversation so far."""
        self._write(
            """
            MERGE (e:Encounter {id:$id}) ON CREATE SET e.ts = datetime($ts)
            SET e.event = $event, e.need = $need, e.familiarity = $familiarity, e.questions = $questions, e.summary = $summary,
                e.turns = $turns
            WITH e
            CALL (e) {
              OPTIONAL MATCH (pl:Place {name:$place})
              WITH e, pl WHERE pl IS NOT NULL
              MERGE (e)-[:HELD_AT]->(pl)
            }
            CALL (e) {
              MATCH (e)-[r:DISCUSSED|ENJOYED]->(:Theme)
              DELETE r
            }
            CALL (e) {
              UNWIND $topics AS t
              MERGE (th:Theme {name:t})
              MERGE (e)-[:DISCUSSED]->(th)
            }
            CALL (e) {
              UNWIND $enjoyed AS t
              MERGE (th:Theme {name:t})
              MERGE (e)-[:ENJOYED]->(th)
            }
            RETURN e.id AS id
            """,
            id=c["id"], ts=ts, event=c.get("event") or "", need=c.get("need") or None, familiarity=c.get("familiarity") or None,
            questions=list(c.get("questions") or []), summary=(c.get("summary") or "")[:500], turns=int(c.get("turns") or 0),
            place=place or "", topics=[t for t in c.get("topics") or [] if t], enjoyed=[t for t in c.get("enjoyed") or [] if t],
        )
        self._write("MATCH (t:Theme) WHERE NOT (t)<-[:DISCUSSED|ENJOYED]-() DELETE t")

    def list_conversations(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._read(f"MATCH (e:Encounter) RETURN {_RETURN} ORDER BY e.ts DESC LIMIT $limit", limit=limit)

    def delete_conversation(self, conv_id: str) -> bool:
        rows = self._write("MATCH (e:Encounter {id:$id}) DETACH DELETE e RETURN 1 AS done", id=conv_id)
        self._write("MATCH (t:Theme) WHERE NOT (t)<-[:DISCUSSED|ENJOYED]-() DELETE t")
        return bool(rows)

    def booth_insights(self) -> dict[str, Any]:
        return _insights(self.list_conversations())

    def booth_graph(self) -> dict[str, Any]:
        return _graph(self.list_conversations(), getattr(getattr(self, "s", None), "robot_name", "PuppyPi"))

    def set_conversation_nams(self, conv_id: str, nams_id: str) -> None:
        self._write("MATCH (e:Encounter {id:$id}) SET e.namsId = $n", id=conv_id, n=nams_id)

    def conversation_nams_id(self, conv_id: str) -> str | None:
        rows = self._read("MATCH (e:Encounter {id:$id}) RETURN e.namsId AS n", id=conv_id)
        return rows[0]["n"] if rows else None


class FakeBooth:
    """In-RAM twin of BoothGraph (dev + tests)."""

    def _convs(self) -> dict[str, dict]:
        if not hasattr(self, "_booth"):
            self._booth: dict[str, dict] = {}
        return self._booth

    def init_booth_schema(self) -> None: ...

    def save_conversation(self, c: dict[str, Any], *, place: str | None, ts: str) -> None:
        old = self._convs().get(c["id"])
        self._convs()[c["id"]] = {
            "id": c["id"], "ts": old["ts"] if old else ts, "event": c.get("event") or "", "need": c.get("need") or None,
            "familiarity": c.get("familiarity") or None, "questions": list(c.get("questions") or []),
            "summary": (c.get("summary") or "")[:500], "turns": int(c.get("turns") or 0), "place": place,
            "topics": [t for t in c.get("topics") or [] if t], "enjoyed": [t for t in c.get("enjoyed") or [] if t],
            **({"nams_id": old["nams_id"]} if old and old.get("nams_id") else {})}

    def list_conversations(self, limit: int = 200) -> list[dict[str, Any]]:
        out = sorted((dict(c, topics=list(c["topics"]), enjoyed=list(c["enjoyed"]), questions=list(c["questions"]))
                      for c in self._convs().values()), key=lambda c: c["ts"], reverse=True)
        return out[:limit]

    def delete_conversation(self, conv_id: str) -> bool:
        return self._convs().pop(conv_id, None) is not None

    def booth_insights(self) -> dict[str, Any]:
        return _insights(self.list_conversations())

    def booth_graph(self) -> dict[str, Any]:
        return _graph(self.list_conversations(), getattr(getattr(self, "s", None), "robot_name", "PuppyPi"))

    def set_conversation_nams(self, conv_id: str, nams_id: str) -> None:
        if conv_id in self._convs():
            self._convs()[conv_id]["nams_id"] = nams_id

    def conversation_nams_id(self, conv_id: str) -> str | None:
        return (self._convs().get(conv_id) or {}).get("nams_id")
