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
    m._write("MATCH (c:Dialogue) WHERE c.id STARTS WITH 'zz-' OPTIONAL MATCH (c)-[:HAS_UTTERANCE|HAS_REASONING]->(x) "
             "OPTIONAL MATCH (x)-[:MADE_LOOKUP]->(t) DETACH DELETE c, x, t")
    m._write("MATCH (s:Session) WHERE s.id STARTS WITH 'zz-' DETACH DELETE s")


m = Memory(get_settings())
# the checks move the robot's CURRENTLY_IN edge onto a test place: remember the real one and put it back at the end
robot_at = m._read("MATCH (:Robot {name:$r})-[c:CURRENTLY_IN]->(p:Place) RETURN p.name AS place, c.heading AS h", r=m.s.robot_name)
try:
    cleanup(m)
    sid = m.start_session("zz-spatial-test-session")
    check("starting a session writes nothing until something happens in it",
          not m._read("MATCH (s:Session {id:$s}) RETURN s", s=sid))
    ing = lambda place, room, *objs, **kw: m.ingest(scene(room, *objs), session_id=sid, location=place, source=SRC, room_type=room, **kw)

    m.ensure_place(KITCHEN, "kitchen", True)
    m.ensure_place(LIVING, "living room", True)
    k1 = ing(KITCHEN, "kitchen", ("zzfridge", "white", "left"), ("zzchair", "brown", "center"), ("zzstove", "black", "right"))
    l1 = ing(LIVING, "living room", ("zzsofa", "grey", "left"), ("zzchair", "brown", "center"), ("zztv", "black", "right"))
    fresh = m._read("MATCH (s:Sighting {source:$src})-[:TAKEN_AT]->(l:Place)<-[:LOCATED_AT]-(o:Item) "
                    "RETURN [k IN keys(s) + keys(l) + keys(o) WHERE k CONTAINS '_'] AS snake LIMIT 1", src=SRC)
    check("fresh writes use camelCase keys (before any migration runs)", fresh and not fresh[0]["snake"], fresh)
    ch = m._read("MATCH (o:Item {label:'zzchair'})-[:LOCATED_AT]->(l:Place) RETURN o.itemId AS id, l.name AS place ORDER BY l.name")
    check("the same kind of chair in two places is two different objects", len(ch) == 2 and ch[0]["id"] != ch[1]["id"], [c["place"][-8:] for c in ch])

    places = {p["name"]: p for p in m.list_places() if p["name"].startswith(P)}
    check("list_places gives each place its own objects and room type",
          {o["label"] for o in places[KITCHEN]["objects"]} == {"zzfridge", "zzchair", "zzstove"} and places[LIVING]["room_type"] == "living room")

    stove = m._read("MATCH (o:Item {label:'zzstove'}) RETURN o.itemId AS id")[0]["id"]
    fr = lambda: m._read("MATCH (o:Item {itemId:$i}) RETURN o.missCount AS miss, o.status AS status", i=stove)[0]

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
    st = {x["label"]: x for x in m._read("MATCH (o:Item)-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.label AS label, o.status AS status, o.missCount AS miss", p=ANCHOR)}
    check("partial view: the mug that was ON the desk is missed (its anchor is in view)", st["zzmug"]["status"] == "possibly_gone", st["zzmug"])
    check("partial view: the lamp and monitor (no anchor in view) are just out of frame, not missing", st["zzlamp"]["miss"] == 0 and st["zzmonitor"]["miss"] == 0, (st["zzlamp"]["miss"], st["zzmonitor"]["miss"]))

    # --- orientation: bearings are stored, and an object outside the view cone is never charged with a miss
    ORI = P + "orient"
    m.ensure_place(ORI, "living room", True)
    ing(ORI, "living room", ("zztable", "brown", "center"), ("zzsofa", "grey", "left"), ("zzlamp", "black", "right"), heading=90.0)
    b = {x["label"]: x["bearing"] for x in m._read("MATCH (o:Item)-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.label AS label, o.lastBearing AS bearing", p=ORI)}
    fov = get_settings().camera_fov_deg
    check("bearings are stored per object (left is counter-clockwise: heading + fov/3)", b["zztable"] == 90.0 and abs(b["zzsofa"] - (90 + fov / 3)) < 0.2 and abs(b["zzlamp"] - (90 - fov / 3)) < 0.2, b)
    for _ in range(4):
        ing(ORI, "living room", ("zzwindow", "white", "center"), ("zzcabinet", "white", "left"), heading=270.0)  # turned around
    st = {x["label"]: x["miss"] for x in m._read("MATCH (o:Item)-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.label AS label, o.missCount AS miss", p=ORI)}
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
    box_ids = [r["id"] for r in m._read("MATCH (o:Item {label:'zzbox'})-[:LOCATED_AT]->(:Place {name:$p}) RETURN o.itemId AS id", p=DUP)]
    if len(box_ids) == 1:  # matching already unified them: force a second copy so the merge itself is tested
        m._write("MATCH (l:Place {name:$p}) CREATE (o:Item {itemId:'zz-dup-2', label:'zzbox', color:'brown', seenCount:1, missCount:0, status:'present', firstSeen:datetime()})-[:LOCATED_AT]->(l)", p=DUP)
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
    # --- key migration: what older versions stored (snake_case keys, bare id, :AT) is renamed; a NAMS node with one of our labels is not
    m._write("CREATE (:Item {id:'zz-legacy-item', label:'zzlegacy', seen_count:2, first_seen:datetime()})")
    m._write("MATCH (k:Place {name:$k}) CREATE (:Sighting {id:'zz-legacy-obs', ts:datetime(), source:$src, n_objects:0})-[:AT]->(k)", k=KITCHEN, src=SRC)
    m._write("CREATE (:Session:Entity {id:'zz-nams-session', name:'zz-spatial-test-nams', started_at:'kept'})")
    m._write("CREATE (:Session {id:'zz-empty-session', startedAt:datetime()})")  # what every app start used to leave behind
    m._write("MATCH (r:Robot {name:$r}), (o:Item {label:'zzlegacy'}) CREATE (r)-[:OBSERVED {first:datetime()}]->(o)", r=m.s.robot_name)
    m.init_schema()  # runs the label and key migrations
    names = {p["name"] for p in m.list_places()}
    check("a NAMS :Entity:Location entity is not one of our places", "zz-spatial-test-floor" not in names, sorted(n for n in names if n.startswith(P))[:3])
    check("a legacy :Location place (plc- id) was migrated to :Place", "zz-spatial-test-legacy" in names)
    lp = m._read("MATCH (l:Place {name:'zz-spatial-test-legacy'}) RETURN l.placeId AS id, l.roomType AS room, l.id AS old_id, l.room_type AS old_room")[0]
    check("legacy place keys renamed: id -> placeId, room_type -> roomType", lp == {"id": "plc-zzlegacy", "room": "kitchen", "old_id": None, "old_room": None}, lp)
    li = m._read("MATCH (o:Item {label:'zzlegacy'}) RETURN o.itemId AS id, o.seenCount AS seen, o.firstSeen IS NOT NULL AS first, o.id AS old_id, o.seen_count AS old_seen")[0]
    check("legacy item keys renamed: id -> itemId, seen_count -> seenCount, first_seen -> firstSeen",
          li == {"id": "zz-legacy-item", "seen": 2, "first": True, "old_id": None, "old_seen": None}, li)
    lo = m._read("MATCH (s:Sighting {id:'zz-legacy-obs'}) RETURN s.objectCount AS n, count{ (s)-[:TAKEN_AT]->() } AS taken, count{ (s)-[:AT]->() } AS at")[0]
    check("legacy sighting: n_objects -> objectCount, AT -> TAKEN_AT", lo == {"n": 0, "taken": 1, "at": 0}, lo)
    ns = m._read("MATCH (s:Session:Entity {id:'zz-nams-session'}) RETURN s.started_at AS old, s.startedAt AS new")[0]
    check("a NAMS :Session:Entity node is left untouched by the key migration", ns == {"old": "kept", "new": None}, ns)
    check("empty sessions are removed; the session with sightings stays (and the NAMS one is never touched)",
          [r["id"] for r in m._read("MATCH (s:Session) WHERE s.id STARTS WITH 'zz-' RETURN s.id AS id ORDER BY id")] == ["zz-nams-session", sid])
    m.record_exchange("zz-chat-session", "what is here?", "a zzlegacy", [{"tool": "current_view", "args": {}, "result": {}}], [])
    chat = m._read("MATCH (:Robot {name:$r})-[:RAN]->(s:Session {id:'zz-chat-session'})<-[:IN_SESSION]-(c:Dialogue) "
                   "RETURN count{ (c)-[:HAS_UTTERANCE]->() } AS u", r=m.s.robot_name)
    check("a chat before any sighting creates its own session (with the robot's RAN edge)", chat == [{"u": 2}], chat)
    check("the robot has no OBSERVED edges (DETECTED carries that, with evidence)",
          not m._read("MATCH (:Robot)-[o:OBSERVED]->() RETURN o LIMIT 1"))
    check("no snake_case keys are left on this app's nodes", not m._read(
        "MATCH (n) WHERE (n:Item OR n:Place OR n:Sighting OR n:Session OR n:Dialogue) AND NOT n:Entity "
        "AND any(k IN keys(n) WHERE k CONTAINS '_') RETURN labels(n) AS l, [k IN keys(n) WHERE k CONTAINS '_'] AS k LIMIT 5"))
    m.set_robot_place(KITCHEN, 90.0)
    m.set_robot_place(LIVING, 10.0)
    cur = m._read("MATCH (:Robot)-[c:CURRENTLY_IN]->(p:Place) WHERE p.name STARTS WITH $p RETURN p.name AS place, c.heading AS h", p=P)
    check("the robot is currently in exactly one place, with its heading", len(cur) == 1 and cur[0]["place"] == LIVING and cur[0]["h"] == 10.0, cur)
finally:
    cleanup(m)
    if robot_at and not robot_at[0]["place"].startswith(P):
        m.set_robot_place(robot_at[0]["place"], robot_at[0]["h"])
    left = m._read("MATCH (n) WHERE (n:Item AND n.label STARTS WITH 'zz') OR (n:Place AND n.name STARTS WITH $p) OR (n:Sighting AND n.source = $s) RETURN count(n) AS c", p=P, s=SRC)[0]["c"]
    print(f"cleanup: {left} test nodes left")
    m.close()
print("\nALL PASS" if ok else "\nSOME FAILED")
