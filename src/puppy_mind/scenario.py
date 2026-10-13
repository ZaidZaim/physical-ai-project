"""Compact scripted scenarios: replay a 'booth day' into memory without a camera or OpenAI.

Used for (a) testing graph logic against real Aura, (b) the repeatable 20-question evaluation,
(c) demoing before the robot is ready. Format (JSON):

{"location": "Booth", "steps": [
  {"t_min": 0, "summary": "...", "objects": ["table:brown:center", "laptop:silver:center"],
   "relations": ["1 ON 0"]}]}

object = label:color:zone[:confidence]   relation = "<subject idx> <PREDICATE> <object idx>"
t_min = minutes after the scenario start (the scenario is placed so the last step is 'now').
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Detection, Relation, Scene


def _scene(step: dict) -> Scene:
    objs = []
    for i, spec in enumerate(step["objects"]):
        p = spec.split(":")
        objs.append(Detection(
            idx=i, label=p[0], color=p[1] if len(p) > 1 else "unknown", description="",
            zone=p[2] if len(p) > 2 else "unknown", confidence=float(p[3]) if len(p) > 3 else 0.9))
    rels = []
    for r in step.get("relations", []):
        a, pred, b = r.split()
        rels.append(Relation(subject=int(a), predicate=pred, object=int(b), confidence=0.9))
    return Scene(summary=step.get("summary", ""), objects=objs, relations=rels)


def replay(brain, path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = data["steps"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=max(s["t_min"] for s in steps))
    out = []
    for st in steps:
        ts = (start + timedelta(minutes=st["t_min"])).isoformat()
        out.append(brain.observe_scene(_scene(st), source="scenario", location=data.get("location"), ts=ts))
    return out
