import json, warnings, logging
logging.disable(logging.CRITICAL)
from puppy_mind.config import get_settings
from puppy_mind.graph import Memory
m = Memory(get_settings())
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, detail)

st = m.stats(); print("stats:", st)
check("5 objects stored", st["objects"] == 5, st["objects"])
check("laptop is possibly_gone, others present", (st["possibly_gone"], st["present"]) == (1, 4), (st["possibly_gone"], st["present"]))

lap = m.find_object("laptop")
check("laptop: status possibly_gone, 3 consecutive misses", lap and lap[0]["status"] == "possibly_gone" and lap[0]["consecutive_misses"] == 3, lap[0]["status"] if lap else None)
check("laptop has no active relations (ended when gone)", lap and lap[0]["active_relations"] == [])
check("laptop location is Booth", lap and lap[0]["location"] == "Booth")

on = m.related_objects("table", "ON")
labels = sorted(r["label"] for r in on)
check("currently ON table: only the bottle", labels == ["bottle"], labels)
hist = m.related_objects("table", "ON", include_ended=True)
check("history shows laptop was ON table (ended)", any(r["label"] == "laptop" and r["currently"] is False for r in hist), [(r["label"], r["currently"]) for r in hist])

bp = m.find_object("backpack", "red")
check("red backpack found, next to chair", bp and any(r["predicate"] == "NEXT_TO" and "chair" in r["target"] for r in bp[0]["active_relations"]), bp[0]["active_relations"] if bp else None)

first = m.first_observation("Booth")
check("first observation had the laptop and no backpack", "laptop" in [o["label"] for o in first["objects"]] and "backpack" not in [o["label"] for o in first["objects"]], sorted(o["label"] for o in first["objects"]))
cur = m.current_view("Booth")
check("current view has backpack, no laptop", "backpack" in [o["label"] for o in cur["objects"]] and "laptop" not in [o["label"] for o in cur["objects"]], sorted(o["label"] for o in cur["objects"]))

ch = m.changes_since(None, "booth")  # default baseline = first observation; location is case-insensitive
check("changes (default baseline): backpack is new", [o["label"] for o in ch["appeared"]] == ["backpack"], [o["label"] for o in ch["appeared"]])
check("changes: laptop possibly gone", [o["label"] for o in ch["possibly_gone"]] == ["laptop"])
text = " ".join(ch["summary"]).lower()
check("summary mentions backpack NEW, laptop POSSIBLY GONE, and the ended laptop-on-table relation",
      "new: a red backpack" in text and "possibly gone: the silver laptop" in text and "relation ended: the silver laptop was on the brown table" in text, ch["summary"][:1])
check("explicit since = first obs gives the same answer", [o["label"] for o in m.changes_since(first["ts"], "Booth")["appeared"]] == ["backpack"])
check("find_object understands 'red backpack'", bool(m.find_object("red backpack")) and m.find_object("red backpack")[0]["label"] == "backpack")
check("find_object keeps ended relations in history", any(r["predicate"] == "ON" and r["currently"] is False and r["ended_by"] == "possibly_gone" for r in m.find_object("laptop")[0]["relation_history"]))

check("next-to is symmetric: chair has the backpack next to it AND vice versa",
      [x["label"] for x in m.related_objects("chair", "NEXT_TO")] == ["backpack"] and [x["label"] for x in m.related_objects("backpack", "NEXT_TO")] == ["chair"])
check("related_objects understands a colour word ('red backpack')", [x["label"] for x in m.related_objects("red backpack", "ANY")] == ["chair"])
check("find_object colour word picks the right one", all(x["color"] == "blue" for x in m.find_object("blue bottle")))

h = m.object_history(lap[0]["id"])
check("laptop history: 3 sightings", len(h["object"]["sightings"]) == 3, len(h["object"]["sightings"]))
check("timeline has 6 observations", len(m.timeline(20)) == 6)
g = m.graph_snapshot()
check("graph snapshot: 5 objects + 1 location", len(g["nodes"]) == 6, len(g["nodes"]))
bad = m.read_cypher("CREATE (n:Hack) RETURN n")
check("read_cypher blocks writes", bad and "error" in bad[0])
check("read_cypher runs reads", m.read_cypher("MATCH (o:Item) RETURN count(o) AS n")[0]["n"] == 5)
m.close()
print("\nALL PASS" if ok else "\nSOME FAILED")
