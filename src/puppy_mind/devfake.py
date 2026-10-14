"""In-memory stand-in for graph.Memory. DEV ONLY (PUPPY_FAKE_MEMORY=1): lets you exercise the live UI,
video stream and event bus without Aura. It reuses the real matching logic but is NOT a substitute for
testing the Cypher in graph.py against Aura. Nothing persists.
"""
from __future__ import annotations

import uuid
from typing import Any

from .graph import now_iso
from .matching import Known, Seen, assign
from .models import Scene


class FakeMemory:
    def __init__(self, s):
        self.s = s
        self.objs: dict[str, dict] = {}
        self.rels: dict[tuple, dict] = {}
        self.obs: list[dict] = []
        self.msgs: list[dict] = []
        self.steps: list[dict] = []

    def close(self): ...
    def init_schema(self): ...
    def start_session(self, session_id=None): return session_id or "fake-session"

    def ingest(self, scene: Scene, *, session_id, location=None, source="camera", ts=None, image_ref=None, count_misses=True) -> dict[str, Any]:
        ts, loc = ts or now_iso(), location or self.s.default_location
        dets = {d.idx: d for d in scene.objects if d.confidence >= self.s.min_confidence}
        anchors = {i: set() for i in dets}
        for r in scene.relations:
            if r.subject in dets and r.object in dets:
                anchors[r.subject].add((r.predicate, dets[r.object].label))
        knowns = [
            Known(o["id"], o["label"], o["color"], o["zone"],
                  frozenset((p, self.objs[b]["label"]) for (a, p, b), r in self.rels.items() if a == o["id"] and r["active"]))
            for o in self.objs.values() if o["loc"] == loc
        ]
        mp = assign(knowns, [Seen(i, d.label, d.color, d.zone, d.confidence, frozenset(anchors[i])) for i, d in dets.items()])
        ids, rows = {}, []
        for i, d in dets.items():
            new = mp[i] is None
            oid = mp[i] or f"obj-{uuid.uuid4().hex[:8]}"
            ids[i] = oid
            o = self.objs.setdefault(oid, {"id": oid, "label": d.label, "color": d.color, "loc": loc, "status": "present", "miss": 0})
            o.update(zone=d.zone, status="present", miss=0, last_seen=ts)
            if d.color != "unknown":
                o["color"] = d.color
            rows.append({"id": oid, "label": d.label, "color": o["color"], "description": d.description,
                         "zone": d.zone, "conf": d.confidence, "new": new})
        cur = set()
        for r in scene.relations:
            if r.subject in ids and r.object in ids:
                k = (ids[r.subject], r.predicate, ids[r.object])
                cur.add(k)
                self.rels.setdefault(k, {"since": ts})["active"] = True
        vis = set(ids.values())
        for (a, p, b), r in self.rels.items():
            if r["active"] and (a, p, b) not in cur and a in vis and b in vis:
                r["active"], r["until"] = False, ts
        missed = []
        for o in (self.objs.values() if count_misses else []):
            if o["loc"] == loc and o["id"] not in vis and o["status"] in ("present", "possibly_gone"):
                o["miss"] += 1
                if o["miss"] >= self.s.miss_threshold:
                    o["status"] = "possibly_gone"
                missed.append({"id": o["id"], "label": o["label"], "misses": o["miss"], "status": o["status"]})
        gone = {m["id"] for m in missed if m["status"] == "possibly_gone"}
        for (a, p, b), r in self.rels.items():
            if a in gone and r["active"]:
                r["active"], r["until"] = False, ts
        oid = f"obs-{uuid.uuid4().hex[:10]}"
        self.obs.insert(0, {"id": oid, "ts": ts, "location": loc, "summary": scene.summary, "n": len(rows)})
        return {"observation_id": oid, "ts": ts, "location": loc, "missed": missed, "objects": rows,
                "new_objects": [f"{r['label']} ({r['id']})" for r in rows if r["new"]],
                "matched_objects": [f"{r['label']} ({r['id']})" for r in rows if not r["new"]],
                "relations": len(cur), "relation_list": [{"a": a, "predicate": p, "b": b} for a, p, b in cur]}

    def stats(self):
        return {"objects": len(self.objs), "observations": len(self.obs), "locations": len({o["loc"] for o in self.objs.values()}),
                "sessions": 1, "present": sum(o["status"] == "present" for o in self.objs.values()),
                "possibly_gone": sum(o["status"] == "possibly_gone" for o in self.objs.values())}

    def nams_links(self): return {}
    def set_nams_links(self, links): ...
    def current_view(self, location=None): return {"observation": None}
    def timeline(self, limit=20): return self.obs[:limit]

    # ---- conversation + reasoning memory (RAM)
    def record_exchange(self, session_id, question, answer, trace, object_ids, ts=None):
        ts = ts or now_iso()
        uid = uuid.uuid4().hex[:8]
        self.msgs += [{"id": f"msg-{uid}-q", "role": "user", "text": question, "conv": session_id, "ts": ts},
                      {"id": f"msg-{uid}-a", "role": "assistant", "text": answer, "conv": session_id, "ts": ts}]
        self.steps.append({"id": f"step-{uid}", "action": f"answer: {question[:60]}", "reasoning": "Queried memory with tools: "
                           + ", ".join(t["tool"] for t in trace), "tools": [(f"tc-{uid}-{i}", t["tool"]) for i, t in enumerate(trace)],
                           "ids": [i for i in object_ids if i in self.objs]})
        return f"step-{uid}"

    def label_counts(self):
        return {"Object": len(self.objs), "Location": len({o["loc"] for o in self.objs.values()}), "Robot": 1, "Sighting": len(self.obs),
                "Session": 1, "Dialogue": 1 if self.msgs else 0, "Utterance": len(self.msgs), "ReasoningStep": len(self.steps),
                "Lookup": sum(len(s["tools"]) for s in self.steps)}

    def rel_counts(self):
        out = {}
        for (a, p, b), r in self.rels.items():
            out[p] = out.get(p, 0) + 1
        return out

    def memory_summary(self):
        c = self.label_counts()
        return {"short": {"observations": c["Sighting"], "messages": c["Utterance"], "conversations": c["Dialogue"]},
                "long": {"objects": c["Object"], "places": c["Location"], "relationships": len(self.rels)},
                "reasoning": {"steps": c["ReasoningStep"], "tool_calls": c["Lookup"]}}

    def graph_snapshot(self, limit=300, lens="long"):
        nodes, edges = {}, []

        def node(i, kind, label, memory, **x):
            nodes.setdefault(i, {"id": i, "kind": kind, "label": label, "memory": memory, **x})

        def edge(a, b, t, m):
            if a in nodes and b in nodes:
                edges.append({"source": a, "target": b, "type": t, "memory": m})

        def entity(ids=None):
            for o in self.objs.values():
                if ids is None or o["id"] in ids:
                    node(o["id"], "Object", o["label"], "long", color=o["color"], status=o["status"])
            for l in {o["loc"] for o in self.objs.values()}:
                node(f"loc:{l}", "Location", l, "long")

        if lens in ("long", "all"):
            entity()
            for (a, p, b), r in self.rels.items():
                if r["active"]:
                    edge(a, b, p, "long")
            for o in self.objs.values():
                edge(o["id"], f"loc:{o['loc']}", "LOCATED_AT", "long")
        if lens in ("short", "all"):
            recent = list(reversed(self.obs[:12]))
            for o in recent:
                node(o["id"], "Sighting", o["ts"][11:19] + f" · {o['n']} seen", "short", detail=o["summary"])
            for a, b in zip(recent, recent[1:]):
                edge(a["id"], b["id"], "THEN", "short")
            prev = None
            for m in self.msgs[-12:]:
                node(m["id"], "Utterance", ("Q: " if m["role"] == "user" else "A: ") + m["text"][:70], "short")
                node("conv:" + m["conv"], "Dialogue", "conversation", "short")
                edge("conv:" + m["conv"], m["id"], "HAS_UTTERANCE", "short")
                if prev:
                    edge(prev, m["id"], "FOLLOWED_BY", "short")
                prev = m["id"]
        if lens in ("reasoning", "all"):
            used = set()
            for st in self.steps[-8:]:
                node(st["id"], "ReasoningStep", st["action"], "reasoning", detail=st["reasoning"])
                for tid, name in st["tools"]:
                    node(tid, "Lookup", name, "reasoning")
                    edge(st["id"], tid, "MADE_LOOKUP", "reasoning")
                used |= set(st["ids"])
            entity(used)
            for st in self.steps[-8:]:
                for i in st["ids"]:
                    edge(st["id"], i, "RELIED_ON", "reasoning")
        return {"nodes": list(nodes.values()), "edges": edges, "lens": lens}
