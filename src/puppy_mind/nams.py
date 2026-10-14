"""NAMS (Neo4j Agent Memory Service) as the robot's *semantic and episodic* memory.

Division of labour (see docs/MEMORY.md)
  Aura (graph.py)  authoritative physical world model: typed, time-aware facts (what is ON what, since/until,
                   possibly_gone). Exact queries. NAMS cannot update or delete relationships, so it cannot hold this.
  NAMS (this file) the story and the meaning around it:
      * entities   Agent (the robot) / Location / Object (+ Person only with consent), linked by ON / NEXT_TO / LOCATED_AT / OBSERVED
      * narrative  every observation is written as a message to an observation-log conversation; NAMS itself builds
                   Observations and Reflections from message windows (the "story of the day")
      * dialogue   questions and answers (chat conversation), reasoning steps and tool calls (provenance)
      * recall     semantic search over entities and past messages, exposed to the agent as a tool
  Cross-link: each Aura Object stores nams_id / nams_name.

Everything here is best-effort and ASYNCHRONOUS: calls go through a background worker so the robot never waits on the
network, failures are counted and never raised, and a rejected API key (401) switches the layer off instead of retrying forever.

API shapes verified against https://memory.neo4jlabs.com/openapi.json (Bearer key + X-Workspace-Id header).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable

import httpx

from .config import Settings

log = logging.getLogger("puppy_mind.nams")

T_ROBOT, T_PLACE, T_OBJECT, T_PERSON = "Agent", "Location", "Object", "Person"  # types of the workspace's ACTIVE ontology (verified live)
SPATIAL = ("ON", "INSIDE", "NEXT_TO", "NEAR", "IN_FRONT_OF", "BEHIND")


def object_name(label: str, color: str) -> str:
    return f"{color} {label}" if color and color not in ("", "unknown") else label


def narrative(p: dict) -> str:
    """One observation as a sentence block NAMS can summarise. Pure function (unit-tested)."""
    seen = ", ".join(object_name(o["label"], o["color"]) for o in p["objects"]) or "nothing recognisable"
    lines = [f"[{p['ts']}] {p['robot']} observed {p['location']}: {p['summary'] or 'no description'}", f"Seen: {seen}."]
    new = [object_name(o["label"], o["color"]) for o in p["objects"] if o["new"]]
    if new:
        lines.append("New: " + ", ".join(new) + ".")
    names = {o["id"]: object_name(o["label"], o["color"]) for o in p["objects"]}
    rel = [f"{names.get(r['a'], r['a'])} {r['predicate'].replace('_', ' ').lower()} {names.get(r['b'], r['b'])}" for r in p["relations"]]
    if rel:
        lines.append("Spatial: " + "; ".join(rel) + ".")
    for m in p["missed"]:
        who = object_name(m.get("label", "object"), m.get("color", ""))
        if m.get("status") == "possibly_gone":
            lines.append(f"Possibly gone: the {who} (not seen in {m['misses']} consecutive observations; a missed detection is not proof it was removed).")
        else:
            lines.append(f"Not seen this time: the {who} (miss {m['misses']}).")
    return " ".join(lines)


class Nams:
    def __init__(self, s: Settings, memory=None, client: httpx.Client | None = None):
        self.s, self.mem = s, memory
        self.enabled = bool(s.nams_api_key) or client is not None
        self.auth = "unchecked"  # unchecked | ok | invalid
        self.stats = {"ok": 0, "failed": 0, "synced_observations": 0, "last_error": ""}
        self._conv: dict[tuple[str, str], str] = {}
        self._links: dict[str, dict] = {}      # aura object id -> {"id": nams entity id, "name": str}
        self._names: dict[str, str] = {}       # nams entity name -> aura object id (collision avoidance)
        self._fixed: dict[tuple[str, str], str] = {}
        self._info: dict[str, dict] = {}       # aura object id -> {"label","color"}
        self._gone: set[str] = set()
        self._flagged: set[str] = set()       # NAMS entity ids it flagged as possible duplicates
        self._q: queue.Queue = queue.Queue(maxsize=1000)
        self._stop = threading.Event()
        self.http = client
        if self.enabled and client is None:
            headers = {"Authorization": f"Bearer {s.nams_api_key}"}
            if s.nams_workspace_id:
                headers["X-Workspace-Id"] = s.nams_workspace_id
            self.http = httpx.Client(base_url=s.nams_base_url.rstrip("/"), headers=headers, timeout=20)
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._load_links()
            self._thread = threading.Thread(target=self._work, daemon=True, name="nams-sync")
            self._thread.start()
            self._submit(self._req, "GET", "/v1/status")  # validates the key now: 401 flips auth to 'invalid'

    # ------------------------------------------------------------------ plumbing
    def _req(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> Any | None:
        if not self.enabled:
            return None
        try:
            r = self.http.request(method, path, json=body, params=params)
        except Exception as e:  # network error: count, never raise
            self.stats["failed"] += 1
            self.stats["last_error"] = f"{type(e).__name__}: {e}"[:160]
            return None
        if r.status_code == 401:
            self.auth, self.enabled = "invalid", False  # stop hammering a rejected key
            self.stats["failed"] += 1
            self.stats["last_error"] = "401 invalid API key: NAMS sync switched off (create a new key in the NAMS dashboard, update .env, restart)"
            log.warning(self.stats["last_error"])
            return None
        if r.status_code >= 300:
            self.stats["failed"] += 1
            self.stats["last_error"] = f"{method} {path} -> {r.status_code} {r.text[:140]}"
            log.warning(self.stats["last_error"])
            return None
        self.auth = "ok"
        self.stats["ok"] += 1
        try:
            return r.json()
        except ValueError:
            return {}

    def _submit(self, fn: Callable, *a) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait((fn, a))
        except queue.Full:
            self.stats["failed"] += 1
            self.stats["last_error"] = "NAMS queue full: dropping updates"

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                fn, a = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.enabled:
                    fn(*a)
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["last_error"] = f"{type(e).__name__}: {e}"[:160]
                log.exception("NAMS task failed")
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until queued updates are sent (tests, CLI, shutdown)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        self.flush(5)
        self._stop.set()

    def status(self) -> dict:
        return {"enabled": self.enabled, "auth": self.auth, "workspace": self.s.nams_workspace_id or None,
                "queued": self._q.unfinished_tasks, **self.stats}

    # ------------------------------------------------------------------ entities & relationships
    def _load_links(self) -> None:
        try:
            if self.mem is not None and hasattr(self.mem, "nams_links"):
                for oid, v in self.mem.nams_links().items():
                    self._links[oid] = v
        except Exception as e:
            log.warning("could not load NAMS links from Aura: %s", e)

    @staticmethod
    def _id_of(res: dict) -> str | None:
        return res.get("id") or res.get("merged_into") or res.get("duplicate_of")

    def _create_entities(self, items: list[dict]) -> list[dict]:
        """Bulk-create. Returns one dict per input item: {"id", "resolution", "duplicate_of"} (id None on error).

        NAMS resolves duplicates by name similarity (auto-merge at >= 0.92, review flag from 0.78 on, measured live), so the
        caller must treat resolution == "merged" as an identity conflict. Per-item errors are surfaced, never swallowed.
        """
        if not items:
            return []
        out = self._req("POST", "/v1/entities/bulk", {"entities": items})
        res: list[dict] = [{"id": None, "resolution": "failed", "duplicate_of": None} for _ in items]
        for r in (out or {}).get("results", []):
            i = r.get("index")
            if not (isinstance(i, int) and 0 <= i < len(items)):
                continue
            if r.get("error"):
                self.stats["failed"] += 1
                self.stats["last_error"] = f"entity {items[i]['name']!r}: {r['error']}"[:200]
                continue
            res[i] = {"id": r.get("id") or r.get("merged_into") or r.get("duplicate_of"),
                      "resolution": r.get("resolution") or "created", "duplicate_of": r.get("duplicate_of")}
            if r.get("resolution") == "merged":
                res[i]["id"] = None  # merged into someone else's entity: never mirror 1:1 onto it
                res[i]["merged_into"] = r.get("merged_into")
        return res

    def _fixed_entity(self, kind: str, name: str, type_: str, description: str) -> str | None:
        key = (kind, name)
        if key not in self._fixed:
            item = {"name": name, "type": type_, "description": description}
            if type_ == T_ROBOT:
                item["properties"] = {"agent_id": name}  # the Agent type in the active ontology requires agent_id
            r = self._create_entities([item])
            if r and r[0]["id"]:
                self._fixed[key] = r[0]["id"]
        return self._fixed.get(key)

    @staticmethod
    def _instance_name(label: str, color: str, object_id: str, salt: str = "") -> str:
        """Readable AND unique: 'black chair 3f2a9c'. Colour + a short id keep NAMS from merging distinct look-alikes
        (measured live: 'chair 3f2a9c' without the colour merged; 'black chair 3f2a9c' stayed separate)."""
        short = object_id.removeprefix("obj-")[:6]
        return f"{object_name(label, color)} {short}{salt}"

    def _ensure_objects(self, objs: list[dict], place: str, ts: str) -> None:
        """Create NAMS entities for objects we have not linked yet and record the link in Aura (1:1, never onto a merged entity)."""
        todo = [o for o in objs if o["id"] not in self._links]
        if not todo:
            return
        salts = [""] * len(todo)
        links: list[dict] = []
        pending = list(range(len(todo)))
        for attempt in range(3):
            items = []
            for i in pending:
                o = todo[i]
                name = self._instance_name(o["label"], o["color"], o["id"], salts[i])
                items.append({"name": name, "type": T_OBJECT, "description": f"{object_name(o['label'], o['color'])}, first observed at {place} on {ts}",
                              "properties": {"object_id": o["id"], "label": o["label"], "color": o["color"], "first_seen": ts, "location": place}})
            results = self._create_entities(items)
            retry = []
            for i, item, r in zip(pending, items, results):
                if r["id"]:
                    o = todo[i]
                    self._links[o["id"]] = {"id": r["id"], "name": item["name"]}
                    links.append({"id": o["id"], "nams_id": r["id"], "name": item["name"]})
                    if r["resolution"] == "review_pending":
                        self._flagged.add(r["id"])
                elif r["resolution"] == "merged":
                    salts[i] += f"-{uuid.uuid4().hex[:4]}"  # NAMS thought it was a duplicate: retry with a more distinct name
                    retry.append(i)
            pending = retry
            if not pending:
                break
        if links:
            self._recheck_flags_later()
        if pending:
            self.stats["last_error"] = f"{len(pending)} object(s) could not be mirrored without being merged by NAMS"
        if links and self.mem is not None and hasattr(self.mem, "set_nams_links"):
            try:
                self.mem.set_nams_links(links)
            except Exception as e:
                log.warning("could not save NAMS links to Aura: %s", e)
        self._reject_flags()

    def _reject_flags(self, force: bool = False) -> None:
        """Our world model knows these objects are different instances: reject NAMS's 'possible duplicate' flags between them,
        so nobody merges them by accident in the review queue. Only pairs where BOTH sides are objects we mirror are touched."""
        if not self._flagged and not force:
            return
        mine = {v["id"] for v in self._links.values()}
        out = self._req("GET", "/v1/reviews", params={"status": "pending", "limit": 200})
        for p in (out or {}).get("pairs", []):
            if p.get("sourceId") in mine and p.get("targetId") in mine:
                r = self._req("POST", f"/v1/reviews/{p['pairId']}/decision",
                              {"action": "reject", "sourceId": p["sourceId"], "targetId": p["targetId"]})
                if r is not None:
                    self.stats["rejected_flags"] = self.stats.get("rejected_flags", 0) + 1
        self._flagged.clear()

    def _recheck_flags_later(self, seconds: float = 6.0) -> None:
        """NAMS raises duplicate flags a moment AFTER the bulk response: look again shortly (never blocks the worker)."""
        t = threading.Timer(seconds, lambda: self._submit(self._reject_flags, True))
        t.daemon = True
        t.start()

    def _link_mirrors(self) -> None:
        """Shared database: connect our Object nodes to the Entity nodes NAMS wrote for them (best effort, idempotent)."""
        try:
            if self.mem is not None and hasattr(self.mem, "mirror_links"):
                self.mem.mirror_links()
        except Exception as e:
            log.warning("could not link mirrored entities: %s", e)

    def _relationships(self, rels: list[dict]) -> None:
        if rels:
            self._req("POST", "/v1/relationships/bulk", {"relationships": rels})

    # ------------------------------------------------------------------ observation sync (async)
    def submit_observation(self, p: dict) -> None:
        """p = {session, ts, location, summary, robot, observation_id, objects[], relations[], missed[]}."""
        self._submit(self._sync_observation, p)

    def _sync_observation(self, p: dict) -> None:
        robot = self._fixed_entity("robot", p["robot"], T_ROBOT, "Hiwonder PuppyPi quadruped robot")
        place = self._fixed_entity("place", p["location"], T_PLACE, f"Place observed by {p['robot']}")
        for o in p["objects"]:
            self._info[o["id"]] = {"label": o["label"], "color": o["color"]}
        self._ensure_objects(p["objects"], p["location"], p["ts"])
        props = {"observed_at": p["ts"], "observation_id": p["observation_id"], "location": p["location"]}
        rels = []
        for o in p["objects"]:
            link = self._links.get(o["id"])
            if not link:
                continue
            if robot:
                rels.append({"relationshipType": "OBSERVED", "sourceId": robot, "targetId": link["id"],
                             "confidence": float(o.get("conf", 0.9)), "properties": props})
            if place and o["new"]:
                rels.append({"relationshipType": "LOCATED_AT", "sourceId": link["id"], "targetId": place, "properties": props})
        for r in p["relations"]:
            a, b = self._links.get(r["a"]), self._links.get(r["b"])
            if a and b and r["predicate"] in SPATIAL:
                rels.append({"relationshipType": r["predicate"], "sourceId": a["id"], "targetId": b["id"], "properties": props})
        self._relationships(rels)
        self._link_mirrors()

        # status changes live in the entity description (NAMS entities update name/description only)
        for m in p["missed"]:
            self._info.setdefault(m["id"], {"label": m.get("label", "object"), "color": ""})
            m.setdefault("color", self._info[m["id"]]["color"])
            link = self._links.get(m["id"])
            if link and m.get("status") == "possibly_gone" and m["id"] not in self._gone:
                self._gone.add(m["id"])
                self._req("PUT", f"/v1/entities/{link['id']}",
                          {"description": f"{link['name']}: possibly gone since {p['ts']} (missed {m['misses']} observations in a row at {p['location']}); not proof of removal"})
        for o in p["objects"]:
            link = self._links.get(o["id"])
            if link and o["id"] in self._gone:
                self._gone.discard(o["id"])
                self._req("PUT", f"/v1/entities/{link['id']}", {"description": f"{link['name']}: back in view at {p['location']} on {p['ts']}"})

        conv = self.conversation(p["session"], "observations")
        if conv:
            self._req("POST", f"/v1/conversations/{conv}/messages",
                      {"role": "system", "content": narrative({**p, "robot": p["robot"]}),
                       "metadata": {"kind": "observation", "observation_id": p["observation_id"], "location": p["location"], "ts": p["ts"]}})
        self.stats["synced_observations"] += 1

    def sync_snapshot(self, objects: list[dict], relations: list[dict], location: str, robot: str) -> dict:
        """Backfill an existing Aura world model (objects + active spatial relations) into NAMS."""
        ts = objects[0]["last_seen"] if objects else ""
        robot_id = self._fixed_entity("robot", robot, T_ROBOT, "Hiwonder PuppyPi quadruped robot")
        place_id = self._fixed_entity("place", location, T_PLACE, f"Place observed by {robot}")
        objs = [{"id": o["id"], "label": o["label"], "color": o.get("color") or "unknown"} for o in objects]
        self._ensure_objects(objs, location, ts)
        rels = []
        for o in objs:
            link = self._links.get(o["id"])
            if link and robot_id:
                rels.append({"relationshipType": "OBSERVED", "sourceId": robot_id, "targetId": link["id"]})
            if link and place_id:
                rels.append({"relationshipType": "LOCATED_AT", "sourceId": link["id"], "targetId": place_id})
        for r in relations:
            a, b = self._links.get(r["a"]), self._links.get(r["b"])
            if a and b and r["predicate"] in SPATIAL:
                rels.append({"relationshipType": r["predicate"], "sourceId": a["id"], "targetId": b["id"],
                             "properties": {"since": r.get("since") or ""}})
        self._relationships(rels)
        self._link_mirrors()
        self._reject_flags(force=True)
        return {"objects": len(objs), "relationships": len(rels), "linked": sum(1 for o in objs if o["id"] in self._links)}

    # ------------------------------------------------------------------ conversations (async)
    def conversation(self, session_id: str, kind: str) -> str | None:
        key = (session_id, kind)
        if key not in self._conv:
            r = self._req("POST", "/v1/conversations",
                          {"metadata": {"session": session_id, "kind": kind, "app": "puppy-mind"}})
            if not r or "id" not in r:
                return None
            self._conv[key] = r["id"]
        return self._conv[key]

    def message(self, session_id: str, role: str, content: str, **meta: str) -> None:
        self._submit(self._message, session_id, role, content, {k: str(v) for k, v in meta.items()})

    def _message(self, session_id: str, role: str, content: str, meta: dict) -> None:
        cid = self.conversation(session_id, "chat")
        if cid:
            self._req("POST", f"/v1/conversations/{cid}/messages", {"role": role, "content": content, "metadata": meta})

    def reasoning(self, session_id: str, question: str, tool_trace: list[dict], answer: str, object_ids: list[str] | None = None) -> None:
        self._submit(self._reasoning, session_id, question, tool_trace, answer, object_ids or [])

    def _reasoning(self, session_id: str, question: str, tool_trace: list[dict], answer: str, object_ids: list[str]) -> None:
        cid = self.conversation(session_id, "chat")
        if not cid:
            return
        step = self._req("POST", "/v1/reasoning/steps", {
            "conversationId": cid, "actionTaken": f"answer: {question[:200]}",
            "reasoning": "Queried memory with tools: " + (", ".join(t["tool"] for t in tool_trace) or "none"),
            "result": answer[:2000]})
        for t in tool_trace:
            self._req("POST", "/v1/reasoning/tool-calls", {
                "toolName": t["tool"], "input": json.dumps(t["args"])[:4000],
                "output": json.dumps(t["result"], default=str)[:4000], "status": "success",
                **({"stepId": step["id"]} if step and step.get("id") else {})})
        if step and step.get("id") and object_ids and self.mem is not None and hasattr(self.mem, "link_reasoning"):
            try:  # provenance NAMS cannot store: which objects this answer relied on
                self.mem.link_reasoning(step["id"], object_ids)
            except Exception as e:
                log.warning("could not link reasoning step to objects: %s", e)

    # ------------------------------------------------------------------ recall (sync: the agent waits for it)
    def recall(self, query: str, limit: int = 5) -> dict:
        if not self.enabled:
            return {"error": "NAMS memory is not available (" + (self.stats["last_error"] or "no API key") + ")"}
        ents = self._req("POST", "/v1/entities/search", {"query": query, "limit": limit}) or {}
        msgs = self._req("POST", "/v1/messages/search", {"query": query, "limit": limit}) or {}
        return {
            "entities": [{"name": e.get("name"), "type": e.get("type"), "description": e.get("description"), "score": e.get("score")}
                         for e in ents.get("entities", [])],
            "messages": [{"role": m.get("role"), "content": (m.get("content") or "")[:400], "at": m.get("createdAt"),
                          "kind": (m.get("metadata") or {}).get("kind"), "score": m.get("score")}
                         for m in msgs.get("messages", [])],
        }

    def context(self, session_id: str, kind: str = "chat") -> dict | None:
        cid = self._conv.get((session_id, kind))
        return self._req("GET", f"/v1/conversations/{cid}/context") if cid else None
