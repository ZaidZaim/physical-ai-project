"""Real-scene evaluation: a TOUR is what the robot saw on a walk through the home, frame by frame, in order.

  record   while the robot runs (--record DIR): every analysed frame's picture and what perception made of it
  export   from Aura: the frames already stored as Sightings (no pictures; detections, room type, the room the robot chose)
  replay   feed a tour through the real room tracker and object matching on throwaway in-RAM memory (no Aura, no OpenAI,
           deterministic) and compare the rooms it ends up with to what the robot decided at the time

Tours describe someone's home, so they belong in the git-ignored tours/ folder, not in the repository.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Detection, Relation, Scene

ROOM_CONF_UNKNOWN = 0.8  # exported frames from before roomConfidence was stored: assume a confident room type when one was given


# ------------------------------------------------------------------ recording
class TourRecorder:
    """Appends one JSON line per analysed frame (and saves its picture) to DIR/tour.jsonl."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.count = 0

    def add(self, scene: Scene, jpeg: bytes | None, *, ts: str, session: str, place: str | None = None, note: str = "") -> None:
        with self._lock:
            self.count += 1
            image = None
            if jpeg:
                image = f"frames/{self.count:05d}.jpg"
                (self.dir / image).write_bytes(jpeg)
            rec = {"ts": ts, "session": session, "place": place, "image": image, "note": note, "scene": scene.model_dump()}
            with (self.dir / "tour.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ export from Aura
def export(mem, session: str | None = None) -> list[dict[str, Any]]:
    """The stored Sightings as tour frames, oldest first. Read-only."""
    rows = mem._read(
        """
        MATCH (s:Sighting)-[:IN_SESSION]->(se:Session)
        WHERE NOT se:Entity AND ($session IS NULL OR se.id = $session)
        OPTIONAL MATCH (s)-[:TAKEN_AT]->(p:Place)
        CALL (s) {
          OPTIONAL MATCH (s)-[d:DETECTED]->(o:Item)
          WITH o, d ORDER BY o.itemId
          RETURN collect(CASE WHEN o IS NULL THEN NULL ELSE
                   {id: o.itemId, label: o.label, color: coalesce(o.color, 'unknown'), zone: coalesce(d.zone, 'unknown'),
                    confidence: coalesce(d.confidence, 0.9), description: coalesce(o.description, '')} END) AS objects
        }
        CALL (s) {
          OPTIONAL MATCH (a:Item)-[r]->(b:Item) WHERE r.lastObs = s.id AND type(r) IN ['ON','INSIDE','NEXT_TO','NEAR','IN_FRONT_OF','BEHIND']
          RETURN collect(CASE WHEN r IS NULL THEN NULL ELSE {a: a.itemId, b: b.itemId, predicate: type(r), confidence: coalesce(r.confidence, 0.8)} END) AS rels
        }
        RETURN toString(s.ts) AS ts, se.id AS session, p.name AS place, s.summary AS summary, coalesce(s.roomType, 'unknown') AS room,
               s.roomConfidence AS room_conf, s.source AS source, objects, rels
        ORDER BY s.ts
        """, session=session)
    frames = []
    for r in rows:
        idx = {o["id"]: i for i, o in enumerate(r["objects"])}
        room_conf = r["room_conf"] if r["room_conf"] is not None else (ROOM_CONF_UNKNOWN if r["room"] not in ("unknown", "other", "") else 0.0)
        scene = Scene(
            summary=r["summary"] or "", room_type=r["room"], room_confidence=room_conf,
            objects=[Detection(idx=i, label=o["label"], color=o["color"], description=o["description"], zone=o["zone"]
                               if o["zone"] in ("left", "center", "right") else "unknown", confidence=o["confidence"])
                     for i, o in enumerate(r["objects"])],
            relations=[Relation(subject=idx[x["a"]], predicate=x["predicate"], object=idx[x["b"]], confidence=x["confidence"])
                       for x in r["rels"] if x["a"] in idx and x["b"] in idx])
        frames.append({"ts": r["ts"], "session": r["session"], "place": r["place"], "image": None, "note": r["source"] or "",
                       "scene": scene.model_dump()})
    return frames


def save(frames: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in frames), encoding="utf-8")
    return p


def load(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "tour.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------ replay
def replay(frames: list[dict], settings, *, verbose: bool = False) -> dict[str, Any]:
    """Feed the frames through Brain on in-RAM memory, in order. A new session means the robot was restarted: the tracker forgets
    where it is (as it does after a real start). Returns the room it chose per frame and a summary."""
    from .brain import Brain

    prev = os.environ.get("PUPPY_FAKE_MEMORY")
    os.environ["PUPPY_FAKE_MEMORY"] = "1"  # never Aura
    try:
        brain = Brain(dataclasses.replace(settings, nams_api_key="", auto_place=True), need_openai=False)
    finally:
        if prev is None:
            os.environ.pop("PUPPY_FAKE_MEMORY", None)
        else:
            os.environ["PUPPY_FAKE_MEMORY"] = prev
    brain.nams.enabled = False
    rows, last_session = [], None
    try:
        for i, f in enumerate(frames, 1):
            if f.get("session") != last_session:
                brain.tracker.set_current(None)
                brain.location = settings.default_location
                last_session = f.get("session")
            scene = Scene.model_validate(f["scene"])
            res = brain.observe_scene(scene, ts=f.get("ts"))
            chosen = res.get("location") if "skipped" not in res else None
            rows.append({"i": i, "ts": f.get("ts"), "session": f.get("session"), "was": f.get("place"), "now": chosen,
                         "status": res.get("skipped") or "stored", "room_type": scene.room_type,
                         "objects": [o.label for o in scene.objects]})
            if verbose:
                r = rows[-1]
                print(f"{i:3d} {str(r['ts'])[11:19]} was={str(r['was']):<12} now={str(r['now']):<12} {r['room_type']:<12} "
                      f"{', '.join(r['objects'])[:60]}{'   [' + r['status'] + ']' if r['status'] != 'stored' else ''}")
        rooms = sorted(brain.mem.places)
        stored = [r for r in rows if r["status"] == "stored"]
        agree = sum(1 for r in stored if r["was"] == r["now"])
        return {"frames": len(rows), "stored": len(stored), "held_or_skipped": len(rows) - len(stored), "filed": len(brain.mem.obs), "rooms": rooms,
                "rooms_recorded": sorted({r["was"] for r in rows if r["was"]}),
                "agreement": round(agree / len(stored), 3) if stored else None,
                "transitions": Counter((a["now"], b["now"]) for a, b in zip(stored, stored[1:]) if a["now"] != b["now"]),
                "rows": rows}
    finally:
        brain.close()


def default_tour_path() -> Path:
    return Path("tours") / f"tour-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
