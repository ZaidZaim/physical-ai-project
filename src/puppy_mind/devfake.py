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

    def close(self): ...
    def init_schema(self): ...
    def start_session(self, session_id=None): return session_id or "fake-session"

    def ingest(self, scene: Scene, *, session_id, location=None, source="camera", ts=None, image_ref=None) -> dict[str, Any]:
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
        for o in self.objs.values():
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

    def current_view(self, location=None): return {"observation": None}
    def timeline(self, limit=20): return self.obs[:limit]

    def graph_snapshot(self, limit=300):
        nodes = [{"id": o["id"], "kind": "Object", "label": o["label"], "color": o["color"], "status": o["status"]} for o in self.objs.values()]
        locs = {o["loc"] for o in self.objs.values()}
        nodes += [{"id": f"loc:{l}", "kind": "Location", "label": l, "color": None, "status": None} for l in locs]
        edges = [{"source": a, "target": b, "type": p} for (a, p, b), r in self.rels.items() if r["active"]]
        edges += [{"source": o["id"], "target": f"loc:{o['loc']}", "type": "LOCATED_AT"} for o in self.objs.values()]
        return {"nodes": nodes, "edges": edges}
