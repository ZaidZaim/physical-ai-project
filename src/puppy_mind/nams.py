"""NAMS (Neo4j Agent Memory Service) client: conversation + reasoning-trace memory.

Division of labour
  * Aura graph (graph.py)  = the robot's *world model*: objects, places, spatial relations, history.
  * NAMS                    = the robot's *agent memory*: who asked what (short-term), what it
                              learned about the people/goals (long-term), and how it reasoned
                              (tool calls and steps), all in a separate, managed workspace graph.

Optional and fail-open: if NAMS_API_KEY is unset or the service errors, the robot keeps working.
Endpoints verified against https://memory.neo4jlabs.com/openapi.json (Swagger 2, Bearer nams_ key).
"""
from __future__ import annotations

import json
import logging

import httpx

from .config import Settings

log = logging.getLogger("puppy_mind.nams")


class Nams:
    def __init__(self, s: Settings):
        self.enabled = bool(s.nams_api_key)
        self._conv: dict[str, str] = {}  # session_id -> NAMS conversation id
        if self.enabled:
            self.http = httpx.Client(
                base_url=s.nams_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {s.nams_api_key}"},
                timeout=15,
            )

    def _post(self, path: str, body: dict) -> dict | None:
        if not self.enabled:
            return None
        try:
            r = self.http.post(path, json=body)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # never break the robot because memory service hiccuped
            log.warning("NAMS %s failed: %s", path, e)
            return None

    def conversation(self, session_id: str) -> str | None:
        if session_id not in self._conv:
            r = self._post("/v1/conversations", {"metadata": {"session": session_id, "app": "puppy-mind"}})
            if not r:
                return None
            self._conv[session_id] = r["id"]
        return self._conv[session_id]

    def message(self, session_id: str, role: str, content: str, **meta: str) -> None:
        cid = self.conversation(session_id)
        if cid:
            self._post(f"/v1/conversations/{cid}/messages",
                       {"role": role, "content": content, "metadata": {k: str(v) for k, v in meta.items()}})

    def reasoning(self, session_id: str, question: str, tool_trace: list[dict], answer: str) -> None:
        """Record the agent's tool-call chain as a reasoning step + tool calls."""
        cid = self.conversation(session_id)
        if not cid:
            return
        step = self._post(
            "/v1/reasoning/steps",
            {
                "conversationId": cid,
                "actionTaken": f"answer: {question[:200]}",
                "reasoning": "Queried the Aura world-model graph via tools: "
                + ", ".join(t["tool"] for t in tool_trace),
                "result": answer[:2000],
            },
        )
        for t in tool_trace:
            self._post(
                "/v1/reasoning/tool-calls",
                {
                    "toolName": t["tool"],
                    "input": json.dumps(t["args"])[:4000],
                    "output": json.dumps(t["result"], default=str)[:4000],
                    "status": "success",
                    "stepId": (step or {}).get("id"),
                },
            )

    def remember_object(self, label: str, color: str, location: str, object_id: str) -> None:
        """Mirror notable world-model facts into NAMS long-term memory (POLE+O: Object/Location)."""
        self._post("/v1/entities", {
            "name": f"{color + ' ' if color not in ('', 'unknown') else ''}{label}",
            "type": "OBJECT",
            "description": f"Seen by the robot at {location}",
            "properties": {"graph_id": object_id, "location": location},
        })

    def context(self, session_id: str) -> dict | None:
        cid = self._conv.get(session_id)
        if not (self.enabled and cid):
            return None
        try:
            r = self.http.get(f"/v1/conversations/{cid}/context")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("NAMS context failed: %s", e)
            return None
