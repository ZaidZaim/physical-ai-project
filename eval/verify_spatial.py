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
    m._write("MATCH (s:Sighting {source: $src}) OPTIONAL MATCH (s)-[:DETECTED]->(o:Object) WHERE o.label STARTS WITH 'zz' DETACH DELETE o", src=SRC)
    m._write("MATCH (s:Sighting {source: $src}) DETACH DELETE s", src=SRC)
    m._write("MATCH (o:Object) WHERE o.label STARTS WITH 'zz' DETACH DELETE o")
    m._write("MATCH (l:Location) WHERE l.name STARTS WITH $p DETACH DELETE l", p=P)


m = Memory(get_settings())
try:
    cleanup(m)
    sid = m.start_session("zz-spatial-test-session")
    ing = lambda place, room, *objs, **kw: m.ingest(scene(room, *objs), session_id=sid, location=place, source=SRC, room_type=room, **kw)

    m.ensure_place(KITCHEN, "kitchen", True)
    m.ensure_place(LIVING, "living room", True)
    k1 = ing(KITCHEN, "kitchen", ("zzfridge", "white", "left"), ("zzchair", "brown", "center"), ("zzstove", "black", "right"))
    l1 = ing(LIVING, "living room", ("zzsofa", "grey", "left"), ("zzchair", "brown", "center"), ("zztv", "black", "right"))
    ch = m._read("MATCH (o:Object {label:'zzchair'})-[:LOCATED_AT]->(l:Location) RETURN o.id AS id, l.name AS place ORDER BY l.name")
    check("the same kind of chair in two places is two different objects", len(ch) == 2 and ch[0]["id"] != ch[1]["id"], [c["place"][-8:] for c in ch])

    places = {p["name"]: p for p in m.list_places() if p["name"].startswith(P)}
    check("list_places gives each place its own objects and room type",
          {o["label"] for o in places[KITCHEN]["objects"]} == {"zzfridge", "zzchair", "zzstove"} and places[LIVING]["room_type"] == "living room")

    stove = m._read("MATCH (o:Object {label:'zzstove'}) RETURN o.id AS id")[0]["id"]
    fr = lambda: m._read("MATCH (o:Object {id:$i}) RETURN o.miss_count AS miss, o.status AS status", i=stove)[0]

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
          and not m._read("MATCH (l:Location {name:$n}) RETURN l", n=P + "pantry")
          and m._read("MATCH (o:Object {label:'zzfridge'})-[:LOCATED_AT]->(:Location {name:$k}) RETURN count(o) AS c", k=KITCHEN)[0]["c"] == 2)
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
    st = {x["label"]: x for x in m._read("MATCH (o:Object)-[:LOCATED_AT]->(:Location {name:$p}) RETURN o.label AS label, o.status AS status, o.miss_count AS miss", p=ANCHOR)}
    check("partial view: the mug that was ON the desk is missed (its anchor is in view)", st["zzmug"]["status"] == "possibly_gone", st["zzmug"])
    check("partial view: the lamp and monitor (no anchor in view) are just out of frame, not missing", st["zzlamp"]["miss"] == 0 and st["zzmonitor"]["miss"] == 0, (st["zzlamp"]["miss"], st["zzmonitor"]["miss"]))
finally:
    cleanup(m)
    left = m._read("MATCH (n) WHERE (n:Object AND n.label STARTS WITH 'zz') OR (n:Location AND n.name STARTS WITH $p) OR (n:Sighting AND n.source = $s) RETURN count(n) AS c", p=P, s=SRC)[0]["c"]
    print(f"cleanup: {left} test nodes left")
    m.close()
print("\nALL PASS" if ok else "\nSOME FAILED")
