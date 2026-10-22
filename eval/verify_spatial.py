"""Live check of the spatial-memory queries against the configured Aura database, in an ISOLATED namespace.

Everything it writes uses place names starting with 'zz-spatial-test-' and made-up object labels ('zzchair', ...), and it deletes all
of it at the end (also if a check fails). Real observations and places are never matched, linked or modified, so it is safe to run
on the database that holds real robot memory.

    uv run python eval/verify_spatial.py
"""
import logging

from puppy_mind.config import get_settings
from puppy_mind.graph import Memory
from puppy_mind.models import Detection, Scene

logging.disable(logging.CRITICAL)
P = "zz-spatial-test-"
KITCHEN, LIVING, THIRD = P + "kitchen", P + "living", P + "third"
SRC = "spatial-test"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, detail)


def scene(room, *objs):
    ds = [Detection(idx=i, label=l, color=c, description="", zone=z, confidence=0.9) for i, (l, c, z) in enumerate(objs)]
    return Scene(summary="test", room_type=room, room_confidence=0.9, objects=ds, relations=[])


def cleanup(m):
    m._write("MATCH (s:Sighting {source: $src}) OPTIONAL MATCH (s)-[:DETECTED]->(o:Item) WHERE o.label STARTS WITH 'zz' DETACH DELETE o", src=SRC)
    m._write("MATCH (s:Sighting {source: $src}) DETACH DELETE s", src=SRC)
    m._write("MATCH (o:Item) WHERE o.label STARTS WITH 'zz' DETACH DELETE o")
    m._write("MATCH (l:Place) WHERE l.name STARTS WITH $p DETACH DELETE l", p=P)
    m._write("MATCH (n) WHERE (n:Location OR n:Entity) AND n.name STARTS WITH $p DETACH DELETE n", p=P)


m = Memory(get_settings())
try:
    cleanup(m)
    sid = m.start_session("zz-spatial-test-session")
    ing = lambda place, room, *objs, **kw: m.ingest(scene(room, *objs), session_id=sid, location=place, source=SRC, room_type=room, **kw)

    m.ensure_place(KITCHEN, "kitchen", True)
    m.ensure_place(LIVING, "living room", True)
    k1 = ing(KITCHEN, "kitchen", ("zzfridge", "white", "left"), ("zzchair", "brown", "center"), ("zzstove", "black", "right"))
    l1 = ing(LIVING, "living room", ("zzsofa", "grey", "left"), ("zzchair", "brown", "center"), ("zztv", "black", "right"))
    ch = m._read("MATCH (o:Item {label:'zzchair'})-[:LOCATED_AT]->(l:Place) RETURN o.id AS id, l.name AS place ORDER BY l.name")
    check("the same kind of chair in two places is two different objects", len(ch) == 2 and ch[0]["id"] != ch[1]["id"], [c["place"][-8:] for c in ch])

    places = {p["name"]: p for p in m.list_places() if p["name"].startswith(P)}
    check("list_places gives each place its own objects and room type",
          {o["label"] for o in places[KITCHEN]["objects"]} == {"zzfridge", "zzchair", "zzstove"} and places[LIVING]["room_type"] == "living room")

    stove = m._read("MATCH (o:Item {label:'zzstove'}) RETURN o.id AS id")[0]["id"]
    fr = lambda: m._read("MATCH (o:Item {id:$i}) RETURN o.miss_count AS miss, o.status AS status", i=stove)[0]

    r = ing(KITCHEN, "kitchen", ("zzchair", "brown", "center"))  # one corner of the kitchen: overlap 1/3 < 0.5
    check("a frame showing only a corner does NOT count as evidence of absence", r["misses_counted"] is False and fr()["miss"] == 0, f"overlap {r['overlap']}")
    r = ing(KITCHEN, "kitchen", ("zzfridge", "white", "left"), ("zzchair", "brown", "center"))  # overlap 2/3
    check("a frame showing most of the place DOES count a miss for what is absent", r["misses_counted"] is True and fr()["miss"] == 1, f"overlap {r['overlap']}")
    for _ in range(2):
        ing(KITCHEN, "kitchen", ("zzfridge", "white", "left"), ("zzchair", "brown", "center"))
    check("after 3 real misses the stove is possibly gone (never on one miss)", fr()["status"] == "possibly_gone")

    r = ing(LIVING, "living room", ("zzstove", "black", "right"), ("zzsofa", "grey", "left"), ("zztv", "black", "center"))
    mv = [x for x in r["moved_hypotheses"] if x["label"] == "zzstove"]
    check("a look-alike appearing elsewhere is recorded as a HYPOTHESIS (possibly moved), not asserted",
          len(mv) == 1 and mv[0]["old_id"] == stove and 0.5 <= mv[0]["confidence"] <= 0.6, mv)
    check("the original is still possibly gone (hypothesis did not rewrite history)", fr()["status"] == "possibly_gone")

    m.note_transition(LIVING, KITCHEN)
    m.note_transition(KITCHEN, LIVING)
    ov = {p["name"]: p for p in m.places_overview() if p["name"].startswith(P)}
    check("transitions build an undirected adjacency map with counts", ov[KITCHEN]["adjacent"] == [{"place": LIVING, "times": 2}] and ov[LIVING]["adjacent"] == [{"place": KITCHEN, "times": 2}], ov[KITCHEN]["adjacent"])

    m.ensure_place(THIRD, "kitchen", True)
    ing(THIRD, "kitchen", ("zzfridge", "white", "center"))
    check("rename refuses an existing name, accepts a new one", m.rename_place(THIRD, KITCHEN) is False and m.rename_place(THIRD, P + "pantry") is True)
    check("merge folds one place into another and removes it", m.merge_places(P + "pantry", KITCHEN) is True
          and not m._read("MATCH (l:Place {name:$n}) RETURN l", n=P + "pantry")
          and m._read("MATCH (o:Item {label:'zzfridge'})-[:LOCATED_AT]->(:Place {name:$k}) RETURN count(o) AS c", k=KITCHEN)[0]["c"] == 2)
    check("every place has a stable id", all(p["id"] for p in m.places_overview() if p["name"].startswith(P)))

    # --- anchors: absence counts only where the thing it rested on is in view
    from puppy_mind.models import Relation

    ANCHOR = P + "anchor"
    m.ensure_place(ANCHOR, "office", True)
    sc = Scene(summary="t", room_type="office", room_confidence=0.9, relations=[Relation(subject=1, predicate="ON", object=0, confidence=0.9)],
               objects=[Detection(idx=i, label=l, color=c, description="", zone="center", confidence=0.9)
                        for i, (l, c) in enumerate([("zzdesk", "brown"), ("zzmug", "white"), ("zzlamp", "black"), ("zzmonitor", "black")])])
    m.ingest(sc, session_id=sid, location=ANCHOR, source=SRC, room_type="office")
    for _ in range(3):
        r = ing(ANCHOR, "office", ("zzdesk", "brown", "center"))  # a quarter of the place in view
    st = {x["label"]: x for x in m._read("MATCH (o:Item)-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.label AS label, o.status AS status, o.miss_count AS miss", p=ANCHOR)}
    check("partial view: the mug that was ON the desk is missed (its anchor is in view)", st["zzmug"]["status"] == "possibly_gone", st["zzmug"])
    check("partial view: the lamp and monitor (no anchor in view) are just out of frame, not missing", st["zzlamp"]["miss"] == 0 and st["zzmonitor"]["miss"] == 0, (st["zzlamp"]["miss"], st["zzmonitor"]["miss"]))

    # --- orientation: bearings are stored, and an object outside the view cone is never charged with a miss
    ORI = P + "orient"
    m.ensure_place(ORI, "living room", True)
    ing(ORI, "living room", ("zztable", "brown", "center"), ("zzsofa", "grey", "left"), ("zzlamp", "black", "right"), heading=90.0)
    b = {x["label"]: x["bearing"] for x in m._read("MATCH (o:Item)-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.label AS label, o.last_bearing AS bearing", p=ORI)}
    fov = get_settings().camera_fov_deg
    check("bearings are stored per object (left is counter-clockwise: heading + fov/3)", b["zztable"] == 90.0 and abs(b["zzsofa"] - (90 + fov / 3)) < 0.2 and abs(b["zzlamp"] - (90 - fov / 3)) < 0.2, b)
    for _ in range(4):
        ing(ORI, "living room", ("zzwindow", "white", "center"), ("zzcabinet", "white", "left"), heading=270.0)  # turned around
    st = {x["label"]: x["miss"] for x in m._read("MATCH (o:Item)-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.label AS label, o.miss_count AS miss", p=ORI)}
    check("objects behind the robot (outside the view cone) are not missing", st["zztable"] == 0 and st["zzsofa"] == 0 and st["zzlamp"] == 0, st)
    for _ in range(3):
        ing(ORI, "living room", ("zzsofa", "grey", "center"), ("zzlamp", "black", "right"), heading=90.0)  # looking at the table's spot: it is gone
    t = m._read("MATCH (o:Item {label:'zztable'})-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.status AS status", p=ORI)[0]["status"]
    check("but an object missing from where the robot is looking does count", t == "possibly_gone", t)
    # --- duplicates: the same box seen twice (never together) merges; two boxes in one frame never do
    DUP = P + "dup"
    m.ensure_place(DUP, "storage", True)
    for lab in ("zzbox", "zzbox"):
        pass
    ids = []
    for zone in ("left", "right"):  # two sightings, each showing one brown box, plus a bag that is always present
        m.ingest(scene("storage", ("zzbox", "brown", zone), ("zzcrate", "black", "center")), session_id=sid, location=DUP, source=SRC, room_type="storage")
    box_ids = [r["id"] for r in m._read("MATCH (o:Item {label:'zzbox'})-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.id AS id", p=DUP)]
    if len(box_ids) == 1:  # matching already unified them: force a second copy so the merge itself is tested
        m._write("MATCH (l:Place {name:$p}) CREATE (o:Item {id:'zz-dup-2', label:'zzbox', color:'brown', seen_count:1, miss_count:0, status:'present', first_seen:datetime()})-[:LOCATED_AT]->(l)", p=DUP)
    pairs = m.find_duplicates(DUP)
    check("two never-co-occurring look-alikes are reported as duplicates", len(pairs) == 1 and pairs[0]["label"] == "zzbox", pairs)
    merged = m.consolidate_duplicates(DUP)
    left_boxes = m._read("MATCH (o:Item {label:'zzbox'})-[:LOCATED_AT]->(:Place {name:$p}) RETURN count(o) AS c", p=DUP)[0]["c"]
    check("consolidate merges them into one object", len(merged) == 1 and left_boxes == 1, (len(merged), left_boxes))
    both = ing(DUP, "storage", ("zzcup", "white", "left"), ("zzcup", "white", "right"))
    cups = m._read("MATCH (o:Item {label:'zzcup'})-[:LOCATED_AT]->(:Place {name:$p}) RETURN count(o) AS c", p=DUP)[0]["c"]
    check("two cups in ONE frame are two objects and are never merged", cups == 2 and not [d for d in m.find_duplicates(DUP) if d["label"] == "zzcup"], cups)
    # --- schema: NAMS's :Location entities must never show up as our places; legacy :Location places migrate to :Place
    m._write("CREATE (:Entity:Location {id:'zz-ent-1', name:'zz-spatial-test-floor'})")
    m._write("CREATE (:Location {id:'plc-zzlegacy', name:'zz-spatial-test-legacy', room_type:'kitchen'})")
    m.init_schema()  # runs the label migration
    names = {p["name"] for p in m.list_places()}
    check("a NAMS :Entity:Location entity is not one of our places", "zz-spatial-test-floor" not in names, sorted(n for n in names if n.startswith(P))[:3])
    check("a legacy :Location place (plc- id) was migrated to :Place", "zz-spatial-test-legacy" in names)
    m.set_robot_place(KITCHEN, 90.0)
    m.set_robot_place(LIVING, 10.0)
    cur = m._read("MATCH (:Robot)-[c:CURRENTLY_IN]->(p:Place) WHERE p.name STARTS WITH $p RETURN p.name AS place, c.heading AS h", p=P)
    check("the robot is currently in exactly one place, with its heading", len(cur) == 1 and cur[0]["place"] == LIVING and cur[0]["h"] == 10.0, cur)
finally:
    cleanup(m)
    left = m._read("MATCH (n) WHERE (n:Item AND n.label STARTS WITH 'zz') OR (n:Place AND n.name STARTS WITH $p) OR (n:Sighting AND n.source = $s) RETURN count(n) AS c", p=P, s=SRC)[0]["c"]
    print(f"cleanup: {left} test nodes left")
    m.close()
print("\nALL PASS" if ok else "\nSOME FAILED")
