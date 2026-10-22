"""Aura-side view of NAMS's own data (the Aura database is shared with the NAMS workspace; verified live 2026-10-14).

NAMS writes (:Entity), (:Conversation)-[:HAS_MESSAGE]->(:Message), (:Conversation)-[:HAS_STEP]->(:AgentStep)-[:USED_TOOL]->(:ToolCall)
and Entity-[:RELATED_TO {predicate}]->Entity into the database. This module only READS those nodes and adds two kinds of
relationship *from our nodes to theirs*:
    (:Object)-[:MIRRORED_BY]->(:Entity)        which NAMS entity mirrors our world-model object
    (:AgentStep)-[:RELIED_ON]->(:Object)       which objects an answer's reasoning relied on (provenance NAMS cannot record itself)
It never creates, merges or deletes a node carrying a NAMS label (a test enforces that), so NAMS's data stays NAMS's.

Property names used here were read from live nodes: Entity{name,type,description,workspaceId,needsReview}, Message{content,role,
createdAt}, Conversation{workspaceId}, AgentStep{actionTaken,reasoning,createdAt}, ToolCall{toolName}.
"""
from __future__ import annotations

import re
import time
from typing import Any

DUPLICATES_OF_OUR_FACTS = {"on", "inside", "next_to", "near", "in_front_of", "behind", "located_at"}  # already drawn from the world model


def display_name(name: str) -> str:
    """'black chair 3f2a9c' -> 'black chair' (the id suffix keeps NAMS from merging look-alikes; it is noise on screen)."""
    return re.sub(r"\s[0-9a-f]{6}(-[0-9a-f]{4})*$", "", name or "")


def _ws(mem) -> str:
    return getattr(mem.s, "nams_workspace_id", "") or ""


def link_entities(mem) -> int:
    """(:Object)-[:MIRRORED_BY]->(:Entity) for every object we have mirrored into NAMS. Idempotent; skips entities NAMS has not written yet."""
    rows = mem._write(
        """
        MATCH (o:Item) WHERE o.nams_id IS NOT NULL
        MATCH (e:Entity {id: o.nams_id})
        MERGE (o)-[:MIRRORED_BY]->(e)
        RETURN count(*) AS n
        """
    )
    return rows[0]["n"] if rows else 0


def link_step(mem, step_id: str, object_ids: list[str], attempts: int = 5) -> int:
    """(:AgentStep {id})-[:RELIED_ON]->(:Object). NAMS creates the step asynchronously, so retry briefly."""
    if not object_ids:
        return 0
    for i in range(attempts):
        rows = mem._write(
            """
            MATCH (st:AgentStep {id: $sid})
            UNWIND $ids AS oid
            MATCH (o:Item {id: oid})
            MERGE (st)-[:RELIED_ON]->(o)
            RETURN count(*) AS n
            """,
            sid=step_id, ids=object_ids,
        )
        if rows and rows[0]["n"]:
            return rows[0]["n"]
        time.sleep(1.0)
    return 0


def summary(mem) -> dict[str, int]:
    ws = _ws(mem)
    r = mem._read(
        """
        RETURN count { (e:Entity) WHERE $ws = '' OR e.workspaceId = $ws } AS entities,
               count { (c:Conversation) WHERE $ws = '' OR c.workspaceId = $ws } AS conversations,
               count { (:Conversation)-[:HAS_MESSAGE]->(m:Message) } AS messages,
               count { (:Conversation)-[:HAS_STEP]->(st:AgentStep) } AS steps,
               count { (:AgentStep)-[:USED_TOOL]->(t:ToolCall) } AS tool_calls
        """,
        ws=ws,
    )
    return r[0] if r else {}


def entities(mem, limit: int = 60) -> dict[str, list]:
    ws = _ws(mem)
    ents = mem._read(
        """
        MATCH (e:Entity) WHERE $ws = '' OR e.workspaceId = $ws
        RETURN e.id AS id, e.name AS name, e.type AS type, coalesce(e.needsReview, false) AS review, e.description AS description
        ORDER BY e.createdAt DESC LIMIT $limit
        """,
        ws=ws, limit=limit,
    )
    ids = [e["id"] for e in ents]
    rel = mem._read("MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids "
                    "RETURN a.id AS a, b.id AS b, coalesce(r.predicate, 'related') AS predicate", ids=ids)
    mirror = mem._read("MATCH (o:Item)-[:MIRRORED_BY]->(e:Entity) WHERE e.id IN $ids RETURN o.id AS a, e.id AS b", ids=ids)
    return {"entities": ents, "related": rel, "mirrored": mirror}


def conversation(mem, limit: int = 12) -> dict[str, list]:
    ws = _ws(mem)
    msgs = mem._read(
        """
        MATCH (c:Conversation)-[:HAS_MESSAGE]->(m:Message) WHERE $ws = '' OR c.workspaceId = $ws
        WITH c, m ORDER BY m.createdAt DESC LIMIT $limit
        RETURN c.id AS conv, m.id AS id, m.role AS role, left(m.content, 70) AS text, toString(m.createdAt) AS ts
        ORDER BY ts
        """,
        ws=ws, limit=limit,
    )
    return {"messages": msgs}


def reasoning(mem, limit: int = 8) -> dict[str, list]:
    ws = _ws(mem)
    steps = mem._read(
        """
        MATCH (c:Conversation)-[:HAS_STEP]->(st:AgentStep) WHERE $ws = '' OR c.workspaceId = $ws
        WITH st ORDER BY st.createdAt DESC LIMIT $limit
        RETURN st.id AS id, left(st.actionTaken, 60) AS action, st.reasoning AS reasoning ORDER BY st.createdAt
        """,
        ws=ws, limit=limit,
    )
    ids = [s["id"] for s in steps]
    tools = mem._read("MATCH (st:AgentStep)-[:USED_TOOL]->(t:ToolCall) WHERE st.id IN $ids RETURN st.id AS a, t.id AS id, t.toolName AS name", ids=ids)
    relied = mem._read("MATCH (st:AgentStep)-[:RELIED_ON]->(o:Item) WHERE st.id IN $ids RETURN st.id AS a, o.id AS b", ids=ids)
    return {"steps": steps, "tools": tools, "relied": relied}


def add_to_snapshot(mem, lens: str, node, edge, show_entities: bool = True, nodes: dict | None = None) -> None:
    """Add NAMS-stored nodes to a lens snapshot via the caller's node()/edge() helpers. Never raises: NAMS data is optional."""
    try:
        if lens in ("long", "all"):
            d = entities(mem)
            if show_entities:
                for e in d["entities"]:
                    node("ent:" + e["id"], "Entity", display_name(e["name"]), "long", source="nams", detail=f"{e['name']} ({e['type']}): {e['description'] or ''}", review=e["review"])
                for r in d["related"]:
                    if r["predicate"] not in DUPLICATES_OF_OUR_FACTS:  # e.g. the robot observed X: information only NAMS holds
                        edge("ent:" + r["a"], "ent:" + r["b"], r["predicate"], "long")
                for r in d["mirrored"]:
                    edge(r["a"], "ent:" + r["b"], "MIRRORED_BY", "long")
            elif nodes is not None:  # entities hidden: just mark our own objects that NAMS also stores (one node per thing, not two)
                for r in d["mirrored"]:
                    if r["a"] in nodes:
                        nodes[r["a"]]["in_nams"] = True
        if lens in ("short", "all"):
            d = conversation(mem)
            prev: dict[str, str] = {}
            for m in d["messages"]:
                node("nmsg:" + m["id"], "Utterance", ("Q: " if m["role"] == "user" else "A: ") + (m["text"] or ""), "short", source="nams")
                node("nconv:" + m["conv"], "Dialogue", "conversation", "short", source="nams")
                edge("nconv:" + m["conv"], "nmsg:" + m["id"], "HAS_UTTERANCE", "short")
                if m["conv"] in prev:
                    edge(prev[m["conv"]], "nmsg:" + m["id"], "FOLLOWED_BY", "short")
                prev[m["conv"]] = "nmsg:" + m["id"]
        if lens in ("reasoning", "all"):
            d = reasoning(mem)
            for st in d["steps"]:
                node("nstep:" + st["id"], "ReasoningStep", st["action"] or "step", "reasoning", source="nams", detail=st["reasoning"])
            for t in d["tools"]:
                node("ntool:" + t["id"], "Lookup", t["name"] or "tool", "reasoning", source="nams")
                edge("nstep:" + t["a"], "ntool:" + t["id"], "MADE_LOOKUP", "reasoning")
            ids = sorted({r["b"] for r in d["relied"]})
            if ids:
                for n in mem._read("MATCH (o:Item) WHERE o.id IN $ids RETURN o.id AS id, o.label AS label, o.color AS color, o.status AS status", ids=ids):
                    node(n["id"], "Object", n["label"], "long", color=n["color"], status=n["status"])
            for r in d["relied"]:
                edge("nstep:" + r["a"], r["b"], "RELIED_ON", "reasoning")
    except Exception:  # shared-database extras must never break the world-model view
        return
