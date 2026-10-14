"""Question answering: an LLM that can ONLY learn about the world by calling graph tools.

Grounding contract (enforced by the prompt and by returning the tool trace as evidence):
every claim must come from a tool result, carry its timestamp, and express uncertainty.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from .config import Settings
from .graph import Memory
from .nams import Nams

SYSTEM = """You are {robot}, a quadruped robot with a camera and a persistent memory graph.
You answer questions about your surroundings and what you observed earlier, ONLY from tool results.

Rules:
- Never answer from general knowledge about the scene. If tools return nothing, say you have no
  observation of it.
- Cite evidence: give timestamps (and observation ids when helpful) in plain language.
- Distinguish: currently visible (last observation) vs remembered (older) vs possibly_gone.
  A missed detection is NOT proof something was removed; say "not seen in the last N observations".
- Objects have identities (ids like obj-xxxx). Two same-label objects are different things.
- Relations (ON, NEXT_TO, ...): "currently" false means it ended. IMPORTANT: `until` is when memory CONCLUDED it ended; it is
  NOT the last time it was seen. The last time a relation was actually observed is `last_seen_so` (use that, and the object's
  `last_seen`). ended_by=possibly_gone means the object stopped being detected; say "last seen on the table at <last_seen_so>".
- "Where did you last see X?": use find_object and answer with relation_history (e.g. on the table) plus last_seen/zone/location.
- "Was X there at the start / first": call first_observation and check whether X is in it; if not, say it appeared later and give
  X's first_seen from find_object.
- "What changed" / "what new objects appeared" (no time given) means since the FIRST observation: call what_changed WITHOUT `since`
  and narrate every line of its `summary` (new objects, possibly gone, relation changes). "Summarise what you remember": location_summary.
- Place names are matched case-insensitively.
- For EVERY question about the world, call a tool again. Never answer from your earlier replies in this chat: the memory
  may have changed, and earlier answers may not have covered this object. "No observation" is only valid after a tool returned nothing.
- For "what changed", call what_changed with a sensible `since` (e.g. the session start or the
  first observation's timestamp) and also first_observation if comparing to the beginning.
- Be concise and conversational. Current UTC time: {now}.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "current_view",
        "description": "Objects/relations in the most recent observation (what the robot sees right now), optionally for one location.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "find_object",
        "description": "Find remembered objects by label (substring) and optional colour: status, last seen time, last location, active relations.",
        "parameters": {"type": "object", "properties": {"label": {"type": "string"}, "color": {"type": "string"}}, "required": ["label"]}}},
    {"type": "function", "function": {
        "name": "related_objects",
        "description": "Objects that are (or were, with include_ended) ON/INSIDE/NEXT_TO/NEAR/IN_FRONT_OF/BEHIND a target label, e.g. what is on the table.",
        "parameters": {"type": "object", "properties": {
            "target_label": {"type": "string"},
            "predicate": {"type": "string", "enum": ["ON", "INSIDE", "NEXT_TO", "NEAR", "IN_FRONT_OF", "BEHIND"]},
            "include_ended": {"type": "boolean"}}, "required": ["target_label"]}}},
    {"type": "function", "function": {
        "name": "object_history",
        "description": "Every sighting (time, location) and relation history of one object id.",
        "parameters": {"type": "object", "properties": {"object_id": {"type": "string"}}, "required": ["object_id"]}}},
    {"type": "function", "function": {
        "name": "first_observation",
        "description": "The very first observation the robot made (optionally at a location): the baseline for 'when you first arrived'.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "what_changed",
        "description": "What changed versus a baseline: new objects, possibly-gone objects, relation changes, plus ready-made `summary` sentences. Omit `since` to compare with the FIRST observation (the default for 'what changed' / 'what new objects appeared').",
        "parameters": {"type": "object", "properties": {"since": {"type": "string", "description": "optional ISO-8601 UTC timestamp"}, "location": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "location_summary",
        "description": "Everything remembered about a place in one call: every object with first/last seen, status, relations and the observation time span. Use for 'summarise what you remember'.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "timeline",
        "description": "Most recent observations (id, time, location, summary).",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "read_cypher",
        "description": "Last resort: read-only Cypher. Schema: (Observation)-[:DETECTED|MISSED]->(Object), (Observation)-[:AT]->(Location), (Object)-[:LOCATED_AT]->(Location), (Object)-[:ON|INSIDE|NEXT_TO|NEAR|IN_FRONT_OF|BEHIND {active,since,until}]->(Object). Object{id,label,color,status,first_seen,last_seen}. Observation{id,ts,summary}.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


class Agent:
    def __init__(self, s: Settings, mem: Memory, nams: Nams):
        self.s, self.mem, self.nams = s, mem, nams
        self.client = OpenAI(api_key=s.openai_api_key)
        self.history: dict[str, list[dict]] = {}  # per-session short-term turns (also mirrored to NAMS)

    def _call(self, name: str, a: dict) -> Any:
        m = self.mem
        return {
            "current_view": lambda: m.current_view(a.get("location")),
            "find_object": lambda: m.find_object(a["label"], a.get("color")),
            "related_objects": lambda: m.related_objects(
                a["target_label"], a.get("predicate", "ON"), a.get("include_ended", False)),
            "object_history": lambda: m.object_history(a["object_id"]),
            "first_observation": lambda: m.first_observation(a.get("location")),
            "what_changed": lambda: m.changes_since(a.get("since"), a.get("location")),
            "timeline": lambda: m.timeline(int(a.get("limit", 10))),
            "location_summary": lambda: m.location_summary(a.get("location")),
            "read_cypher": lambda: m.read_cypher(a["query"]),
        }[name]()

    def ask(self, question: str, session_id: str) -> dict[str, Any]:
        sysmsg = SYSTEM.format(robot=self.s.robot_name, now=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        turns = self.history.setdefault(session_id, [])
        msgs: list[dict] = [{"role": "system", "content": sysmsg}, *turns[-6:],
                            {"role": "user", "content": question}]
        trace: list[dict] = []
        answer = ""
        for _ in range(6):
            r = self.client.chat.completions.create(
                model=self.s.chat_model, messages=msgs, tools=TOOLS, temperature=0)
            msg = r.choices[0].message
            if not msg.tool_calls:
                answer = msg.content or ""
                break
            msgs.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                try:
                    res = self._call(tc.function.name, args)
                except Exception as e:
                    res = {"error": str(e)[:300]}
                trace.append({"tool": tc.function.name, "args": args, "result": res})
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(res, default=str)[:12000]})
        else:
            answer = "I couldn't finish reasoning about that within my step budget."
        turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        self.nams.message(session_id, "user", question)
        self.nams.message(session_id, "assistant", answer, tools=",".join(t["tool"] for t in trace))
        self.nams.reasoning(session_id, question, trace, answer)
        return {"answer": answer, "evidence": trace}
