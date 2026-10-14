"""Neo4j (Aura) contextual memory: ingestion with entity matching + temporal queries.

Model
  (Robot)-[:OBSERVED]->(Object)
  (Sighting)-[:CAPTURED_BY]->(Robot), -[:IN_SESSION]->(Session), -[:AT]->(Location)
  (Sighting)-[:DETECTED {confidence, zone}]->(Object)
  (Sighting)-[:MISSED]->(Object)           evidence of absence (not proof of removal)
  (Object)-[:LOCATED_AT {first_seen,last_seen,count}]->(Location)
  (Object)-[:ON|INSIDE|NEXT_TO|NEAR|IN_FRONT_OF|BEHIND {active,since,until,last_confirmed}]->(Object)

Object.status: present | possibly_gone (>= MISS_THRESHOLD consecutive misses at the same location)
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import ClientError

from .config import Settings
from .matching import Known, Seen, assign
from .models import EXCLUSIVE_PREDICATES, PREDICATES, Scene

REL_ALT = "|".join(PREDICATES)
_WRITE_WORDS = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|load|foreach|call\s+(apoc|dbms))\b", re.I
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Memory:
    def __init__(self, s: Settings):
        self.s = s
        self.driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password), notifications_min_severity="OFF")
        self.db = s.neo4j_database

    def close(self) -> None:
        self.driver.close()

    # ---- plumbing
    def _read(self, q: str, **p) -> list[dict[str, Any]]:
        recs, _, _ = self.driver.execute_query(
            q, p, database_=self.db, routing_=RoutingControl.READ
        )
        return [r.data() for r in recs]

    def _write(self, q: str, **p) -> list[dict[str, Any]]:
        recs, _, _ = self.driver.execute_query(q, p, database_=self.db)
        return [r.data() for r in recs]

    # ---- setup
    def _migrate_labels(self) -> None:
        """One-time rename of camera-frame nodes from :Observation to :Sighting.

        This database can be shared with NAMS, which owns the labels Entity, Conversation, Message, AgentStep, ToolCall,
        Observation, Reflection ... so everything this app owns uses its own labels. Only nodes that look like ours
        (they carry n_objects and source) are touched.
        """
        self._write("MATCH (n:Observation) WHERE n.n_objects IS NOT NULL AND n.source IS NOT NULL SET n:Sighting REMOVE n:Observation")
        self._write("MATCH (l:Location) WHERE l.id IS NULL SET l.id = 'plc-' + substring(randomUUID(), 0, 8)")  # every place gets a stable id
        for q in ("DROP CONSTRAINT obs_id IF EXISTS", "DROP INDEX obs_ts IF EXISTS"):
            try:
                self._write(q)
            except ClientError:
                pass

    def init_schema(self) -> None:
        self._migrate_labels()
        for q in [
            "CREATE CONSTRAINT object_id IF NOT EXISTS FOR (n:Object) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT sighting_id IF NOT EXISTS FOR (n:Sighting) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (n:Location) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT robot_name IF NOT EXISTS FOR (n:Robot) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (n:Location) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT dialogue_id IF NOT EXISTS FOR (n:Dialogue) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (n:Utterance) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT reasoning_step_id IF NOT EXISTS FOR (n:ReasoningStep) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT lookup_id IF NOT EXISTS FOR (n:Lookup) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX object_label IF NOT EXISTS FOR (n:Object) ON (n.label)",
            "CREATE INDEX sighting_ts IF NOT EXISTS FOR (n:Sighting) ON (n.ts)",
        ]:
            try:
                self._write(q)
            except ClientError as e:  # an equivalent index/constraint already exists: fine, never fatal
                if "already exist" not in (e.message or "").lower():
                    raise
        self._write("MERGE (:Robot {name:$n})", n=self.s.robot_name)

    def reset(self) -> None:
        """Delete all memory THIS APP owns. Never touches labels owned by NAMS (Entity, Conversation, Message, ...)."""
        for lbl in ("Sighting", "Object", "Session", "Location", "Robot", "Dialogue", "Utterance", "ReasoningStep", "Lookup"):
            self._write(f"MATCH (n:{lbl}) DETACH DELETE n")

    def start_session(self, session_id: str | None = None) -> str:
        sid = session_id or f"session-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        self._write(
            "MERGE (r:Robot {name:$r}) "
            "MERGE (s:Session {id:$sid}) ON CREATE SET s.started_at=datetime($ts) "
            "MERGE (r)-[:RAN]->(s)",
            sid=sid, ts=now_iso(), r=self.s.robot_name,
        )
        return sid

    # ---- ingestion
    def ingest(
        self,
        scene: Scene,
        *,
        session_id: str,
        location: str | None = None,
        source: str = "camera",
        ts: str | None = None,
        image_ref: str | None = None,
        count_misses: bool = True,
        room_type: str | None = None,
        place_confidence: float | None = None,
        heading: float | None = None,
    ) -> dict[str, Any]:
        ts = ts or now_iso()
        location = location or self.s.default_location
        obs_id = f"obs-{uuid.uuid4().hex[:10]}"

        dets = {d.idx: d for d in scene.objects if d.confidence >= self.s.min_confidence}
        rels = [
            r for r in scene.relations
            if r.subject in dets and r.object in dets and r.subject != r.object
            and r.predicate in PREDICATES and r.confidence >= self.s.min_confidence
        ]

        # 1) known objects at this location (+ their active relations as identity anchors)
        rows = self._read(
            """
            MATCH (o:Object)-[:LOCATED_AT]->(:Location {name:$loc})
            OPTIONAL MATCH (o)-[r]->(b:Object) WHERE type(r) IN $preds AND r.active
            RETURN o.id AS id, o.label AS label, coalesce(o.color,'unknown') AS color,
                   coalesce(o.last_zone,'unknown') AS zone, o.status AS status,
                   collect(DISTINCT [type(r), b.label]) AS anchors
            """,
            loc=location, preds=list(PREDICATES),
        )
        seen_anchor: dict[int, set] = {i: set() for i in dets}
        for r in rels:
            seen_anchor[r.subject].add((r.predicate, dets[r.object].label.lower()))
        knowns = [
            Known(
                id=r["id"], label=r["label"], color=r["color"], zone=r["zone"],
                anchors=frozenset((p, l) for p, l in r["anchors"] if p and l),
            )
            for r in rows
        ]
        seens = [
            Seen(
                idx=i, label=d.label.lower(), color=d.color.lower(), zone=d.zone,
                confidence=d.confidence, anchors=frozenset(seen_anchor[i]),
            )
            for i, d in dets.items()
        ]
        mapping = assign(knowns, seens)

        # view overlap: how much of what is known (and present) HERE is in this frame. A frame of another wall, or of one corner,
        # says nothing about objects that are simply out of view, so it must not count as evidence of absence.
        matched_ids = {v for v in mapping.values() if v}
        present_known = [r for r in rows if r["status"] != "possibly_gone"]
        overlap = (sum(1 for r in present_known if r["id"] in matched_ids) / len(present_known)) if present_known else 1.0
        # Partial view (half the room or less): an absence only counts for objects whose spatial ANCHOR is in view. A laptop that
        # was ON the table is missed if the table is in the frame; a stove out of frame is simply out of frame.
        partial = overlap <= self.s.min_overlap
        seen_labels = {d.label.lower() for d in dets.values()}
        eligible = [r["id"] for r in rows if any(lab in seen_labels for _t, lab in r["anchors"] if lab)] if partial else []
        if partial and not eligible:
            count_misses = False

        # 2) resolve ids (unmatched detections become new objects)
        obj_id: dict[int, str] = {}
        new_ids: set[str] = set()
        for i in dets:
            if mapping[i]:
                obj_id[i] = mapping[i]  # type: ignore[assignment]
            else:
                obj_id[i] = f"obj-{uuid.uuid4().hex[:8]}"
                new_ids.add(obj_id[i])

        det_rows = [
            {
                "id": obj_id[i], "label": d.label.lower(), "color": d.color.lower(),
                "description": d.description, "zone": d.zone, "conf": d.confidence,
                "new": obj_id[i] in new_ids,
            }
            for i, d in dets.items()
        ]

        # 3) observation + objects + evidence edges
        self._write(
            """
            MERGE (l:Location {name:$loc})
              ON CREATE SET l.id = 'plc-' + substring(randomUUID(), 0, 8), l.room_type = coalesce($room, 'unknown'), l.provisional = false,
                            l.created_at = datetime($ts), l.visits = 0
            MERGE (r:Robot {name:$robot})
            MERGE (s:Session {id:$sid}) ON CREATE SET s.started_at = datetime($ts)
            CREATE (ob:Sighting {id:$obs, ts:datetime($ts), summary:$summary, source:$source,
                                    image_ref:$img, n_objects:$n, room_type:$room, overlap:$overlap,
                                    place_confidence:$pc, heading:$heading})
            MERGE (ob)-[:CAPTURED_BY]->(r)
            MERGE (ob)-[:IN_SESSION]->(s)
            MERGE (ob)-[:AT]->(l)
            WITH ob, l, r
            UNWIND $dets AS d
            MERGE (o:Object {id:d.id})
              ON CREATE SET o.label=d.label, o.first_seen=datetime($ts), o.seen_count=0, o.miss_count=0
            SET o.color = CASE WHEN d.color IN ['unknown',''] THEN coalesce(o.color,'unknown') ELSE d.color END,
                o.description = CASE WHEN d.description='' THEN o.description ELSE d.description END,
                o.last_seen=datetime($ts), o.last_zone=d.zone, o.last_confidence=d.conf,
                o.seen_count=o.seen_count+1, o.miss_count=0, o.status='present'
            MERGE (ob)-[:DETECTED {confidence:d.conf, zone:d.zone}]->(o)
            MERGE (r)-[ro:OBSERVED]->(o) ON CREATE SET ro.first=datetime($ts)
            SET ro.last=datetime($ts)
            MERGE (o)-[la:LOCATED_AT]->(l)
              ON CREATE SET la.first_seen=datetime($ts), la.count=0
            SET la.last_seen=datetime($ts), la.count=la.count+1
            """,
            loc=location, robot=self.s.robot_name, sid=session_id, ts=ts, obs=obs_id,
            summary=scene.summary, source=source, img=image_ref, n=len(det_rows), dets=det_rows,
            room=room_type, overlap=round(overlap, 3), pc=place_confidence, heading=heading,
        )

        # 4) spatial relations (one query per predicate: Cypher can't parametrise rel types;
        #    the predicate is whitelisted by the Literal type + PREDICATES filter above)
        by_pred: dict[str, list[dict]] = {}
        for r in rels:
            by_pred.setdefault(r.predicate, []).append(
                {"a": obj_id[r.subject], "b": obj_id[r.object], "conf": r.confidence}
            )
        for pred, items in by_pred.items():
            self._write(
                f"""
                UNWIND $items AS it
                MATCH (a:Object {{id:it.a}}), (b:Object {{id:it.b}})
                MERGE (a)-[r:{pred}]->(b)
                  ON CREATE SET r.since=datetime($ts), r.count=0
                SET r.active=true, r.until=null, r.ended_by=null, r.last_confirmed=datetime($ts),
                    r.count=r.count+1, r.confidence=it.conf, r.last_obs=$obs
                """,
                items=items, ts=ts, obs=obs_id,
            )
        visible = list(obj_id.values())
        # containment is exclusive: a newly confirmed ON/INSIDE supersedes older ones of that subject
        for pred in EXCLUSIVE_PREDICATES:
            self._write(
                f"""
                MATCH (a:Object)-[old:{pred}]->(:Object)
                WHERE a.id IN $vis AND old.active AND old.last_obs <> $obs
                  AND EXISTS {{ (a)-[cur:{pred}]->(:Object) WHERE cur.last_obs = $obs }}
                SET old.active=false, old.until=datetime($ts), old.ended_by='superseded'
                """,
                vis=visible, ts=ts, obs=obs_id,
            )
        # a relation between two visible objects that was not re-seen has ended
        self._write(
            f"""
            MATCH (a:Object)-[r:{REL_ALT}]->(b:Object)
            WHERE a.id IN $vis AND b.id IN $vis AND r.active AND r.last_obs <> $obs
            SET r.active=false, r.until=datetime($ts), r.ended_by='not_reseen'
            """,
            vis=visible, ts=ts, obs=obs_id,
        )

        # 5) misses: expected here but not detected. One miss never means "gone".
        # (skipped right after the robot moved: absent objects are probably just out of the new view)
        missed = [] if not count_misses else self._write(
            """
            MATCH (ob:Sighting {id:$obs})
            MATCH (o:Object)-[:LOCATED_AT]->(:Location {name:$loc})
            WHERE NOT o.id IN $vis AND o.status IN ['present','possibly_gone'] AND (NOT $partial OR o.id IN $elig)
            SET o.miss_count = coalesce(o.miss_count,0)+1
            SET o.status = CASE WHEN o.miss_count >= $thr THEN 'possibly_gone' ELSE o.status END
            MERGE (ob)-[:MISSED]->(o)
            RETURN o.id AS id, o.label AS label, o.miss_count AS misses, o.status AS status
            """,
            loc=location, vis=visible, obs=obs_id, thr=self.s.miss_threshold, partial=partial, elig=eligible,
        )
        # a possibly-gone object no longer "is on" anything: end its relations (history is kept)
        gone_ids = [m["id"] for m in missed if m["status"] == "possibly_gone"]
        if gone_ids:
            self._write(
                f"""
                MATCH (a:Object)-[r:{REL_ALT}]->(:Object)
                WHERE a.id IN $gone AND r.active
                SET r.active=false, r.until=datetime($ts), r.ended_by='possibly_gone'
                """,
                gone=gone_ids, ts=ts,
            )
        moved = self._link_moved(sorted(new_ids), location, ts)
        return {
            "overlap": round(overlap, 2),
            "misses_counted": count_misses,
            "moved_hypotheses": moved,
            "observation_id": obs_id,
            "ts": ts,
            "location": location,
            "new_objects": sorted(f"{d['label']} ({d['id']})" for d in det_rows if d["new"]),
            "matched_objects": sorted(f"{d['label']} ({d['id']})" for d in det_rows if not d["new"]),
            "missed": missed,
            "relations": len(rels),
            "objects": det_rows,
            "relation_list": [
                {"a": obj_id[r.subject], "b": obj_id[r.object], "predicate": r.predicate} for r in rels
            ],
        }

    # ---- read tools (also exposed to the Q&A agent)
    def current_view(self, location: str | None = None) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (ob:Sighting) WHERE $loc IS NULL OR EXISTS { (ob)-[:AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }
            WITH ob ORDER BY ob.ts DESC LIMIT 1
            OPTIONAL MATCH (ob)-[d:DETECTED]->(o:Object)
            OPTIONAL MATCH (o)-[r]->(b:Object) WHERE type(r) IN $preds AND r.last_obs = ob.id
            RETURN ob.id AS observation, toString(ob.ts) AS ts, ob.summary AS summary,
                   [x IN collect(DISTINCT {id:o.id, label:o.label, color:o.color, zone:d.zone,
                                           confidence:d.confidence}) WHERE x.id IS NOT NULL] AS objects,
                   [x IN collect(DISTINCT {subject:o.label+' '+o.id, predicate:type(r),
                                           object:b.label+' '+b.id}) WHERE x.predicate IS NOT NULL] AS relations
            """,
            loc=location, preds=list(PREDICATES),
        )
        return rows[0] if rows and rows[0]["observation"] else {"observation": None}

    def _color_words(self, text: str) -> list[str]:
        have = {r["c"].lower() for r in self._read("MATCH (o:Object) WHERE o.color IS NOT NULL AND o.color <> 'unknown' RETURN DISTINCT o.color AS c")}
        return sorted(w for w in re.findall(r"[a-z]+", (text or "").lower()) if w in have)

    def find_object(self, label: str, color: str | None = None, limit: int = 10) -> list[dict]:
        if color is None:  # 'red shoe' means label shoe AND colour red
            qc = self._color_words(label)
            color = qc[0] if len(qc) == 1 else None
        return self._read(
            """
            MATCH (o:Object) WHERE (toLower(o.label) CONTAINS toLower($label) OR toLower($label) CONTAINS toLower(o.label))
              AND ($color IS NULL OR toLower(coalesce(o.color,'')) = toLower($color))
            CALL (o) {
              OPTIONAL MATCH (o)-[la:LOCATED_AT]->(l:Location)
              RETURN l.name AS location ORDER BY la.last_seen DESC LIMIT 1
            }
            RETURN o.id AS id, o.label AS label, o.color AS color, o.description AS description,
                   o.status AS status, o.miss_count AS consecutive_misses, o.seen_count AS times_seen,
                   toString(o.first_seen) AS first_seen, toString(o.last_seen) AS last_seen,
                   o.last_zone AS last_zone, location,
                   [(o)-[r]->(b:Object) WHERE type(r) IN $preds AND r.active |
                      {predicate: type(r), target: b.label + ' ' + b.id, since: toString(r.since)}] AS active_relations,
                   [(o)-[mv:POSSIBLY_MOVED_FROM]->(c:Object) | {object: coalesce(c.color, 'unknown') + ' ' + c.label, id: c.id,
                      place: head([(c)-[:LOCATED_AT]->(pl:Location) | pl.name]), confidence: mv.confidence}] AS maybe_moved_from,
                   [(n:Object)-[mv2:POSSIBLY_MOVED_FROM]->(o) | {object: coalesce(n.color, 'unknown') + ' ' + n.label, id: n.id,
                      place: head([(n)-[:LOCATED_AT]->(pl2:Location) | pl2.name]), confidence: mv2.confidence}] AS maybe_moved_to,
                   [(o)-[h]->(hb:Object) WHERE type(h) IN $preds |
                      {predicate: type(h), target: hb.label + ' ' + hb.id, currently: h.active,
                       last_seen_so: toString(h.last_confirmed), ended_by: h.ended_by}][..8] AS relation_history
            ORDER BY o.last_seen DESC LIMIT $limit
            """,
            label=label, color=color, preds=list(PREDICATES), limit=limit,
        )

    def related_objects(self, target_label: str, predicate: str = "ANY", include_ended: bool = False) -> list[dict]:
        """Objects related to the target. NEXT_TO and NEAR are symmetric, so both directions count; ON, INSIDE, IN_FRONT_OF and
        BEHIND are directional (the returned object is the one that is ON/INSIDE/... the target). predicate=ANY searches all."""
        predicate = (predicate or "ANY").upper()
        if predicate != "ANY" and predicate not in PREDICATES:
            return [{"error": f"predicate must be ANY or one of {PREDICATES}"}]
        rows: list[dict] = []
        qc = self._color_words(target_label)  # 'yellow bag' = label bag AND colour yellow
        for p in (PREDICATES if predicate == "ANY" else (predicate,)):
            symmetric = p in ("NEXT_TO", "NEAR")
            pattern = f"(a:Object)-[r:{p}]-(b:Object)" if symmetric else f"(a:Object)-[r:{p}]->(b:Object)"
            rows += self._read(
                f"""
                MATCH {pattern}
                WHERE (toLower(b.label) CONTAINS toLower($t) OR toLower($t) CONTAINS toLower(b.label))
                  AND (size($qc) = 0 OR toLower(coalesce(b.color, '')) IN $qc) AND a <> b AND ($ended OR r.active)
                RETURN a.id AS id, a.label AS label, a.color AS color, '{p}' AS relation, b.label+' '+b.id AS target,
                       r.active AS currently, a.status AS status,
                       toString(r.since) AS since, toString(r.last_confirmed) AS last_confirmed,
                       toString(r.until) AS until, r.ended_by AS ended_by, r.count AS times_confirmed
                ORDER BY r.last_confirmed DESC LIMIT 30
                """,
                t=target_label, ended=include_ended, qc=qc,
            )
        return rows[:40]

    def object_history(self, object_id: str, limit: int = 25) -> dict[str, Any]:
        obs = self._read(
            """
            MATCH (o:Object {id:$id})
            OPTIONAL MATCH (ob:Sighting)-[d:DETECTED]->(o)
            OPTIONAL MATCH (ob)-[:AT]->(l:Location)
            WITH o, ob, d, l ORDER BY ob.ts DESC LIMIT $limit
            RETURN o.label AS label, o.color AS color,
                   collect({observation:ob.id, ts:toString(ob.ts), location:l.name,
                            confidence:d.confidence, zone:d.zone}) AS sightings
            """,
            id=object_id, limit=limit,
        )
        rels = self._read(
            f"""
            MATCH (o:Object {{id:$id}})-[r:{REL_ALT}]-(x:Object)
            RETURN type(r) AS predicate, startNode(r).id = $id AS outgoing, x.label+' '+x.id AS other,
                   r.active AS currently, toString(r.since) AS since, toString(r.last_confirmed) AS last_seen_so,
                   toString(r.until) AS until, r.ended_by AS ended_by
            ORDER BY r.since DESC LIMIT 30
            """,
            id=object_id,
        )
        return {"object": obs[0] if obs else None, "relations": rels}

    def first_observation(self, location: str | None = None) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (ob:Sighting) WHERE $loc IS NULL OR EXISTS { (ob)-[:AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }
            WITH ob ORDER BY ob.ts ASC LIMIT 1
            OPTIONAL MATCH (ob)-[d:DETECTED]->(o:Object)
            OPTIONAL MATCH (o)-[r]->(b:Object) WHERE type(r) IN $preds AND r.last_obs = ob.id
            RETURN ob.id AS observation, toString(ob.ts) AS ts, ob.summary AS summary,
                   [x IN collect(DISTINCT {label:o.label, color:o.color, id:o.id}) WHERE x.id IS NOT NULL] AS objects,
                   [x IN collect(DISTINCT {subject:o.label+' '+o.id, predicate:type(r),
                                           object:b.label+' '+b.id}) WHERE x.predicate IS NOT NULL] AS relations
            """,
            loc=location, preds=list(PREDICATES),
        )
        return rows[0] if rows and rows[0]["observation"] else {"observation": None}

    def changes_since(self, since_iso: str | None = None, location: str | None = None) -> dict[str, Any]:
        """What differs from a baseline. Baseline = `since_iso`, or by default the FIRST observation (at `location`).

        Returns structured lists plus `summary`: ready-to-narrate plain sentences, so the answer does not depend on the
        caller picking a good timestamp.
        """
        first = self.first_observation(location)
        base = since_iso or first.get("ts")
        if not base:
            return {"since": None, "summary": ["No observations yet."], "appeared": [], "possibly_gone": [], "relation_changes": []}
        baseline = [o["label"] for o in first.get("objects", [])] if first.get("ts") == base else []
        appeared = self._read(
            """
            MATCH (o:Object) WHERE o.first_seen > datetime($since)
              AND ($loc IS NULL OR EXISTS { (o)-[:LOCATED_AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) })
            RETURN o.id AS id, o.label AS label, o.color AS color, toString(o.first_seen) AS first_seen, o.status AS status
            ORDER BY o.first_seen
            """,
            since=base, loc=location,
        )
        gone = self._read(
            """
            MATCH (o:Object) WHERE o.status = 'possibly_gone'
              AND ($loc IS NULL OR EXISTS { (o)-[:LOCATED_AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) })
            RETURN o.id AS id, o.label AS label, o.color AS color, toString(o.last_seen) AS last_seen,
                   o.miss_count AS consecutive_misses
            """,
            loc=location,
        )
        rels = self._read(
            f"""
            MATCH (a:Object)-[r:{REL_ALT}]->(b:Object)
            WHERE (r.since > datetime($since) OR r.until > datetime($since))
              AND ($loc IS NULL OR EXISTS {{ (a)-[:LOCATED_AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }})
            RETURN a.label + ' ' + a.id AS subject, a.label AS subject_label, a.color AS subject_color,
                   type(r) AS predicate, b.label AS object_label, b.color AS object_color,
                   r.active AS currently, toString(r.since) AS since, toString(r.last_confirmed) AS last_seen_so,
                   toString(r.until) AS until, r.ended_by AS ended_by
            ORDER BY coalesce(r.until, r.since) DESC LIMIT 40
            """,
            since=base, loc=location,
        )

        def name(color: str | None, label: str) -> str:
            return f"{color + ' ' if color and color != 'unknown' else ''}{label}"

        lines: list[str] = []
        if baseline:
            lines.append(f"Baseline (first observation at {base}): " + ", ".join(sorted(baseline)) + ".")
        for o in appeared:
            lines.append(f"NEW: a {name(o['color'], o['label'])} appeared, first seen at {o['first_seen']}.")
        for o in gone:
            lines.append(f"POSSIBLY GONE: the {name(o['color'], o['label'])} was last seen at {o['last_seen']} and has been "
                         f"missed in {o['consecutive_misses']} consecutive observations (a missed detection is not proof it was removed).")
        for r in rels:
            s, ob = name(r["subject_color"], r["subject_label"]), name(r["object_color"], r["object_label"])
            if r["currently"]:
                lines.append(f"RELATION: the {s} is {r['predicate'].replace('_', ' ').lower()} the {ob} (since {r['since']}).")
            else:
                why = {"possibly_gone": "because the object stopped being detected", "superseded": "because it moved elsewhere",
                       "not_reseen": "because it was no longer seen that way"}.get(r["ended_by"] or "", "")
                lines.append(f"RELATION ENDED: the {s} was {r['predicate'].replace('_', ' ').lower()} the {ob}; last seen like that at "
                             f"{r['last_seen_so']}; memory ended it at {r['until']} {why}.".replace("  ", " "))
        return {"since": base, "summary": lines, "appeared": appeared, "possibly_gone": gone, "relation_changes": rels}

    def location_summary(self, location: str | None = None) -> dict[str, Any]:
        """Everything remembered about a place: each object with first/last seen, status and relations."""
        objs = self._read(
            """
            MATCH (o:Object)-[:LOCATED_AT]->(l:Location)
            WHERE $loc IS NULL OR toLower(l.name) = toLower($loc)
            RETURN o.id AS id, o.label AS label, o.color AS color, o.status AS status,
                   toString(o.first_seen) AS first_seen, toString(o.last_seen) AS last_seen,
                   o.seen_count AS times_seen, o.miss_count AS consecutive_misses, l.name AS location,
                   [(o)-[r]->(b:Object) WHERE type(r) IN $preds |
                      {predicate: type(r), target: b.label + ' ' + b.id, currently: r.active,
                       last_seen_so: toString(r.last_confirmed), ended_by: r.ended_by}] AS relations
            ORDER BY o.first_seen
            """,
            loc=location, preds=list(PREDICATES),
        )
        span = self._read(
            """
            MATCH (ob:Sighting)
            WHERE $loc IS NULL OR EXISTS { (ob)-[:AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }
            RETURN count(ob) AS observations, toString(min(ob.ts)) AS first_observation, toString(max(ob.ts)) AS last_observation
            """,
            loc=location,
        )
        return {"observations": span[0] if span else None, "objects": objs}

    def timeline(self, limit: int = 20) -> list[dict]:
        return self._read(
            """
            MATCH (ob:Sighting)
            OPTIONAL MATCH (ob)-[:AT]->(l:Location)
            RETURN ob.id AS id, toString(ob.ts) AS ts, l.name AS location, ob.summary AS summary,
                   ob.n_objects AS n
            ORDER BY ob.ts DESC LIMIT $limit
            """,
            limit=limit,
        )

    def read_cypher(self, query: str) -> list[dict]:
        """Agent escape hatch: read-only (enforced by routing AND a keyword guard)."""
        if _WRITE_WORDS.search(query):
            return [{"error": "write operations are not allowed"}]
        if not re.search(r"\blimit\b", query, re.I):
            query += "\nLIMIT 50"
        try:
            return self._read(query)
        except Exception as e:  # surface syntax errors so the model can retry
            return [{"error": str(e)[:300]}]

    # ------------------------------------------------------------------ spatial memory: places
    def list_places(self) -> list[dict[str, Any]]:
        """Every place with what is known to live there: input for the place resolver."""
        return self._read(
            """
            MATCH (l:Location)
            CALL (l) {
              OPTIONAL MATCH (o:Object)-[:LOCATED_AT]->(l)
              RETURN collect(CASE WHEN o IS NULL THEN NULL ELSE
                {label: o.label, color: coalesce(o.color, 'unknown'), seen: o.seen_count, status: o.status} END) AS objects
            }
            RETURN l.name AS name, coalesce(l.room_type, 'unknown') AS room_type, coalesce(l.provisional, false) AS provisional,
                   coalesce(l.visits, 0) AS visits, objects
            ORDER BY l.name
            """
        )

    def ensure_place(self, name: str, room_type: str = "unknown", provisional: bool = False) -> str:
        rows = self._write(
            """
            MERGE (l:Location {name: $name})
              ON CREATE SET l.id = 'plc-' + substring(randomUUID(), 0, 8), l.room_type = $room, l.provisional = $prov,
                            l.created_at = datetime($ts), l.visits = 0
            RETURN l.id AS id
            """,
            name=name, room=room_type or "unknown", prov=provisional, ts=now_iso(),
        )
        return rows[0]["id"]

    def touch_place(self, name: str, room_type: str | None = None) -> None:
        """The robot is (now) in this place: count the visit; learn the room type if we did not know it."""
        self._write(
            """
            MATCH (l:Location {name: $name})
            SET l.visits = coalesce(l.visits, 0) + 1, l.last_visit = datetime($ts),
                l.room_type = CASE WHEN coalesce(l.room_type, 'unknown') IN ['unknown', 'other'] AND NOT coalesce($room, 'unknown') IN ['unknown', 'other', '']
                                   THEN $room ELSE coalesce(l.room_type, 'unknown') END
            """,
            name=name, room=room_type, ts=now_iso(),
        )

    def note_transition(self, a: str, b: str) -> None:
        """The robot went from a to b: they are adjacent (stored once per pair; the direction is just alphabetical)."""
        x, y = sorted((a, b))
        self._write(
            """
            MATCH (x:Location {name: $x}), (y:Location {name: $y}) WHERE x <> y
            MERGE (x)-[r:ADJACENT_TO]->(y)
              ON CREATE SET r.first_seen = datetime($ts), r.count = 0
            SET r.count = r.count + 1, r.last_seen = datetime($ts)
            """,
            x=x, y=y, ts=now_iso(),
        )

    def rename_place(self, old: str, new: str) -> bool:
        if old == new or self._read("MATCH (l:Location {name: $n}) RETURN count(l) AS c", n=new)[0]["c"]:
            return False
        rows = self._write("MATCH (l:Location {name: $old}) SET l.name = $new, l.provisional = false RETURN count(l) AS c", old=old, new=new)
        return bool(rows and rows[0]["c"])

    def merge_places(self, src: str, dst: str) -> bool:
        """Fold place src into dst (e.g. two provisional views of the same kitchen). Objects keep their ids; look-alikes may remain
        as duplicates and are tidied by normal matching over time."""
        ok = self._read("MATCH (s:Location {name: $s}), (d:Location {name: $d}) WHERE s <> d RETURN count(*) AS c", s=src, d=dst)[0]["c"]
        if not ok:
            return False
        self._write(
            """
            MATCH (o:Object)-[r:LOCATED_AT]->(s:Location {name: $s}), (d:Location {name: $d})
            MERGE (o)-[r2:LOCATED_AT]->(d)
              ON CREATE SET r2.first_seen = r.first_seen, r2.last_seen = r.last_seen, r2.count = r.count
            DELETE r
            """, s=src, d=dst)
        self._write("MATCH (x:Sighting)-[r:AT]->(s:Location {name: $s}), (d:Location {name: $d}) MERGE (x)-[:AT]->(d) DELETE r", s=src, d=dst)
        self._write(
            """
            MATCH (s:Location {name: $s})-[r:ADJACENT_TO]-(n:Location), (d:Location {name: $d}) WHERE n <> d AND n <> s
            WITH d, n, sum(r.count) AS c
            WITH d, n, c, CASE WHEN d.name < n.name THEN [d, n] ELSE [n, d] END AS ordered
            WITH ordered[0] AS x, ordered[1] AS y, c
            MERGE (x)-[a:ADJACENT_TO]->(y) ON CREATE SET a.first_seen = datetime(), a.count = 0
            SET a.count = a.count + c
            """, s=src, d=dst)
        self._write("MATCH (s:Location {name: $s}) DETACH DELETE s", s=src)
        return True

    def places_overview(self) -> list[dict[str, Any]]:
        return self._read(
            """
            MATCH (l:Location)
            CALL (l) {
              OPTIONAL MATCH (o:Object)-[:LOCATED_AT]->(l)
              RETURN count(o) AS objects, sum(CASE WHEN o.status = 'present' THEN 1 ELSE 0 END) AS present,
                     sum(CASE WHEN o.status = 'possibly_gone' THEN 1 ELSE 0 END) AS possibly_gone
            }
            CALL (l) {
              OPTIONAL MATCH (s:Sighting)-[:AT]->(l)
              RETURN count(s) AS sightings, toString(max(s.ts)) AS last_seen
            }
            CALL (l) {
              OPTIONAL MATCH (l)-[r:ADJACENT_TO]-(n:Location)
              RETURN collect(CASE WHEN n IS NULL THEN NULL ELSE {place: n.name, times: r.count} END) AS adjacent
            }
            RETURN l.id AS id, l.name AS name, coalesce(l.room_type, 'unknown') AS room_type, coalesce(l.provisional, false) AS provisional,
                   coalesce(l.visits, 0) AS visits, objects, present, possibly_gone, sightings, last_seen, adjacent
            ORDER BY sightings DESC, name
            """
        )

    def _link_moved(self, new_ids: list[str], place: str, ts: str) -> list[dict]:
        """HYPOTHESES, never facts: a new object here that looks exactly like an object that stopped being seen in ANOTHER place
        may be the same thing, moved. Recorded as (new)-[:POSSIBLY_MOVED_FROM {confidence}]->(old)."""
        if not new_ids:
            return []
        return self._write(
            """
            UNWIND $ids AS nid
            MATCH (n:Object {id: nid})
            MATCH (c:Object) WHERE c.id <> n.id AND c.label = n.label AND c.status = 'possibly_gone'
              AND coalesce(c.color, 'unknown') = coalesce(n.color, 'unknown')
              AND NOT EXISTS { (c)-[:LOCATED_AT]->(l:Location {name: $place}) }
            WITH n, collect(c) AS cands
            UNWIND cands AS c
            WITH n, c, size(cands) AS k, CASE WHEN coalesce(n.color, 'unknown') = 'unknown' THEN 0.5 ELSE 1.0 END AS f
            MERGE (n)-[r:POSSIBLY_MOVED_FROM]->(c)
              ON CREATE SET r.since = datetime($ts)
            SET r.confidence = round(100.0 * f * (CASE WHEN k = 1 THEN 0.6 ELSE 0.6 / k END)) / 100.0,
                r.basis = 'same kind and colour, and the original stopped being seen elsewhere'
            RETURN n.id AS new_id, c.id AS old_id, n.label AS label, r.confidence AS confidence
            """,
            ids=new_ids, place=place, ts=ts,
        )

    def clear_nams_links(self) -> int:
        """Forget which NAMS entity mirrors each object (used before a rebuild). Touches only our Object nodes."""
        rows = self._write("MATCH (o:Object) WHERE o.nams_id IS NOT NULL REMOVE o.nams_id, o.nams_name RETURN count(o) AS n")
        return rows[0]["n"] if rows else 0

    def mirror_links(self) -> int:
        from .nams_graph import link_entities

        return link_entities(self)

    def link_reasoning(self, step_id: str, object_ids: list[str]) -> int:
        from .nams_graph import link_step

        return link_step(self, step_id, object_ids)

    def nams_links(self) -> dict[str, dict]:
        """object id -> {id, name} of the NAMS entity mirroring it (so a restart keeps identities)."""
        rows = self._read("MATCH (o:Object) WHERE o.nams_id IS NOT NULL RETURN o.id AS id, o.nams_id AS nams_id, o.nams_name AS name")
        return {r["id"]: {"id": r["nams_id"], "name": r["name"]} for r in rows}

    def set_nams_links(self, links: list[dict]) -> None:
        self._write("UNWIND $l AS x MATCH (o:Object {id:x.id}) SET o.nams_id = x.nams_id, o.nams_name = x.name", l=links)

    def export_for_nams(self, location: str | None = None) -> dict[str, Any]:
        objs = self._read(
            """
            MATCH (o:Object)-[:LOCATED_AT]->(l:Location)
            WHERE $loc IS NULL OR toLower(l.name) = toLower($loc)
            RETURN o.id AS id, o.label AS label, o.color AS color, o.status AS status,
                   toString(o.last_seen) AS last_seen, l.name AS location
            ORDER BY o.first_seen
            """,
            loc=location,
        )
        rels = self._read(
            f"""
            MATCH (a:Object)-[r:{REL_ALT}]->(b:Object) WHERE r.active
            RETURN a.id AS a, type(r) AS predicate, b.id AS b, toString(r.since) AS since
            """
        )
        return {"objects": objs, "relations": rels}

    def stats(self) -> dict[str, Any]:
        return self._read(
            """
            RETURN count { (:Object) } AS objects, count { (:Sighting) } AS observations,
                   count { (:Location) } AS locations, count { (:Session) } AS sessions,
                   count { (:Object {status:'present'}) } AS present,
                   count { (:Object {status:'possibly_gone'}) } AS possibly_gone
            """
        )[0]

    # ------------------------------------------------------------------ conversation + reasoning memory (stored in Aura)
    def record_exchange(self, session_id: str, question: str, answer: str, trace: list[dict],
                        object_ids: list[str], ts: str | None = None) -> str:
        """Persist one Q&A as short-term (Dialogue/Utterance) and reasoning (ReasoningStep/Lookup) memory.

        Labels mirror NAMS's graph schema: Dialogue-HAS_UTTERANCE->Utterance, Dialogue-HAS_REASONING->ReasoningStep,
        ReasoningStep-MADE_LOOKUP->Lookup, ReasoningStep-RELIED_ON->Object, Utterance-ABOUT->Object, Utterance-FOLLOWED_BY->Utterance.
        """
        ts = ts or now_iso()
        uid = uuid.uuid4().hex[:10]
        qid, aid, stid = f"msg-{uid}-q", f"msg-{uid}-a", f"step-{uid}"
        tcs = [{"id": f"tc-{uid}-{i}", "name": t["tool"], "seq": i,
                "input": json.dumps(t["args"], default=str)[:1000], "output": json.dumps(t["result"], default=str)[:1500]}
               for i, t in enumerate(trace)]

        def work(tx) -> None:
            tx.run(
                """
                MERGE (c:Dialogue {id:$sid}) ON CREATE SET c.started_at = datetime($ts)
                WITH c
                CALL (c) {
                  OPTIONAL MATCH (c)-[:HAS_UTTERANCE]->(l:Utterance) WHERE NOT (l)-[:FOLLOWED_BY]->()
                  RETURN l AS last ORDER BY l.ts DESC LIMIT 1
                }
                CREATE (q:Utterance {id:$qid, role:'user', content:$q, ts:datetime($ts)})
                CREATE (a:Utterance {id:$aid, role:'assistant', content:$a, ts:datetime($ts)})
                CREATE (c)-[:HAS_UTTERANCE]->(q), (c)-[:HAS_UTTERANCE]->(a), (q)-[:FOLLOWED_BY]->(a)
                FOREACH (_ IN CASE WHEN last IS NULL THEN [] ELSE [1] END | CREATE (last)-[:FOLLOWED_BY]->(q))
                """,
                sid=session_id, ts=ts, qid=qid, aid=aid, q=question[:2000], a=answer[:4000],
            )
            tx.run(
                """
                MATCH (c:Dialogue {id:$sid}), (a:Utterance {id:$aid})
                CREATE (st:ReasoningStep {id:$stid, action:$action, reasoning:$reasoning, result:$result, ts:datetime($ts)})
                CREATE (c)-[:HAS_REASONING]->(st), (st)-[:PRODUCED]->(a)
                WITH c
                OPTIONAL MATCH (s:Session {id:$sid})
                FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END | MERGE (c)-[:IN_SESSION]->(s))
                """,
                sid=session_id, aid=aid, stid=stid, ts=ts, action=f"answer: {question[:200]}",
                reasoning="Queried memory with tools: " + (", ".join(t["name"] for t in tcs) or "none"), result=answer[:600],
            )
            tx.run(
                """
                MATCH (st:ReasoningStep {id:$stid})
                UNWIND $tcs AS t
                CREATE (tc:Lookup {id:t.id, name:t.name, input:t.input, output:t.output, seq:t.seq, ts:datetime($ts)})
                CREATE (st)-[:MADE_LOOKUP]->(tc)
                """,
                stid=stid, tcs=tcs, ts=ts,
            )
            tx.run(
                """
                MATCH (st:ReasoningStep {id:$stid}), (a:Utterance {id:$aid})
                UNWIND $ids AS oid
                MATCH (o:Object {id:oid})
                MERGE (st)-[:RELIED_ON]->(o)
                MERGE (a)-[:ABOUT]->(o)
                """,
                stid=stid, aid=aid, ids=object_ids,
            )

        with self.driver.session(database=self.db) as s:
            s.execute_write(work)
        return stid

    def label_counts(self) -> dict[str, int]:
        return self._read(
            """
            RETURN count { (:Object) } AS Object, count { (:Location) } AS Location, count { (:Robot) } AS Robot,
                   count { (:Sighting) } AS Sighting, count { (:Session) } AS Session,
                   count { (:Dialogue) } AS Dialogue, count { (:Utterance) } AS Utterance,
                   count { (:ReasoningStep) } AS ReasoningStep, count { (:Lookup) } AS Lookup
            """
        )[0]

    def rel_counts(self) -> dict[str, int]:
        rows = self._read("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS n")
        return {r["t"]: r["n"] for r in rows}

    def memory_summary(self) -> dict[str, Any]:
        r = self._read(
            f"""
            RETURN count {{ (:Sighting) }} AS observations, count {{ (:Utterance) }} AS messages,
                   count {{ (:Dialogue) }} AS conversations, count {{ (:Object) }} AS objects,
                   count {{ (:Location) }} AS places, count {{ ()-[:{REL_ALT}]->() }} AS spatial,
                   count {{ (:ReasoningStep) }} AS steps, count {{ (:Lookup) }} AS tool_calls
            """
        )[0]
        out = {"short": {"observations": r["observations"], "messages": r["messages"], "conversations": r["conversations"]},
               "long": {"objects": r["objects"], "places": r["places"], "relationships": r["spatial"], "nams_entities": 0},
               "reasoning": {"steps": r["steps"], "tool_calls": r["tool_calls"]}}
        try:  # add what NAMS stored in the same database
            from .nams_graph import summary as nams_summary

            n = nams_summary(self)
            out["short"]["messages"] += n.get("messages", 0)
            out["short"]["conversations"] += n.get("conversations", 0)
            out["long"]["nams_entities"] = n.get("entities", 0)
            out["reasoning"]["steps"] += n.get("steps", 0)
            out["reasoning"]["tool_calls"] += n.get("tool_calls", 0)
        except Exception:
            pass
        return out

    def purge_demo(self, dry_run: bool = True) -> dict[str, int]:
        """Remove scripted/demo data only: observations with source 'scenario'/'demo', objects ONLY ever detected by them,
        places left empty, and sessions left without observations. Real observations are never touched."""
        fake = ["scenario", "demo"]
        counts = self._read(
            """
            RETURN count { (ob:Sighting) WHERE ob.source IN $fake } AS observations,
                   count { (o:Object) WHERE EXISTS { (x:Sighting)-[:DETECTED]->(o) WHERE x.source IN $fake }
                                         AND NOT EXISTS { (y:Sighting)-[:DETECTED]->(o) WHERE NOT y.source IN $fake } } AS objects
            """,
            fake=fake,
        )[0]
        if dry_run:
            return counts
        self._write(
            """
            MATCH (o:Object)
            WHERE EXISTS { (x:Sighting)-[:DETECTED]->(o) WHERE x.source IN $fake }
              AND NOT EXISTS { (y:Sighting)-[:DETECTED]->(o) WHERE NOT y.source IN $fake }
            DETACH DELETE o
            """,
            fake=fake,
        )
        self._write("MATCH (ob:Sighting) WHERE ob.source IN $fake DETACH DELETE ob", fake=fake)
        gone = self._write(
            "MATCH (l:Location) WHERE NOT EXISTS { (:Object)-[:LOCATED_AT]->(l) } AND NOT EXISTS { (:Sighting)-[:AT]->(l) } "
            "DETACH DELETE l RETURN count(l) AS n"
        )
        sess = self._write(
            "MATCH (s:Session) WHERE NOT EXISTS { (:Sighting)-[:IN_SESSION]->(s) } AND NOT EXISTS { (:Dialogue)-[:IN_SESSION]->(s) } "
            "DETACH DELETE s RETURN count(s) AS n"
        )
        counts["places"] = gone[0]["n"] if gone else 0
        counts["sessions"] = sess[0]["n"] if sess else 0
        return counts

    def graph_snapshot(self, limit: int = 300, lens: str = "long") -> dict[str, Any]:
        """Nodes/edges for one memory lens: long (entities+facts), short (observation + conversation stream),
        reasoning (agent steps+tool calls), or all. Each node carries `memory` for colouring; stable ids for diffing."""
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        def node(id_: str, kind: str, label: str, memory: str, **extra) -> None:
            if id_ not in nodes:
                nodes[id_] = {"id": id_, "kind": kind, "label": label, "memory": memory, **extra}

        def edge(a: str, b: str, type_: str, memory: str) -> None:
            if a in nodes and b in nodes:
                edges.append({"source": a, "target": b, "type": type_, "memory": memory})

        def entity_nodes(ids: list[str] | None = None) -> None:
            for n in self._read(
                """
                MATCH (n) WHERE (n:Object OR n:Location) AND ($ids IS NULL OR n.id IN $ids)
                RETURN CASE WHEN n:Object THEN n.id ELSE 'loc:' + n.name END AS id, labels(n)[0] AS kind,
                       coalesce(n.label, n.name) AS label, n.color AS color, n.status AS status LIMIT $limit
                """, ids=ids, limit=limit):
                node(n["id"], n["kind"], n["label"], "long", color=n["color"], status=n["status"])

        if lens in ("long", "all"):
            entity_nodes()
            for e in self._read(
                """
                MATCH (a:Object)-[r]->(b) WHERE (b:Object OR b:Location) AND type(r) IN $types AND coalesce(r.active, true)
                RETURN a.id AS source, CASE WHEN b:Object THEN b.id ELSE 'loc:' + b.name END AS target, type(r) AS type LIMIT $limit
                """, types=list(PREDICATES) + ["LOCATED_AT"], limit=limit * 3):
                edge(e["source"], e["target"], e["type"], "long")

        if lens in ("short", "all"):
            obs = self._read(
                """
                MATCH (ob:Sighting) WITH ob ORDER BY ob.ts DESC LIMIT 12
                OPTIONAL MATCH (ob)-[:AT]->(l:Location)
                RETURN ob.id AS id, toString(ob.ts) AS ts, ob.n_objects AS n, ob.summary AS summary, l.name AS location
                ORDER BY ob.ts
                """)
            for o in obs:
                node(o["id"], "Sighting", o["ts"][11:19] + f" · {o['n']} seen", "short", detail=o["summary"], location=o["location"])
            for a, b in zip(obs, obs[1:]):
                edge(a["id"], b["id"], "THEN", "short")
            ids = [o["id"] for o in obs]
            det = self._read("MATCH (ob:Sighting)-[:DETECTED]->(o:Object) WHERE ob.id IN $ids RETURN ob.id AS a, o.id AS b", ids=ids)
            entity_nodes(sorted({d["b"] for d in det}))
            for d in det:
                edge(d["a"], d["b"], "DETECTED", "short")
            msgs = self._read(
                """
                MATCH (m:Utterance) WITH m ORDER BY m.ts DESC LIMIT 12
                OPTIONAL MATCH (c:Dialogue)-[:HAS_UTTERANCE]->(m)
                RETURN m.id AS id, m.role AS role, left(m.content, 70) AS text, c.id AS conv ORDER BY m.ts
                """)
            for m in msgs:
                node(m["id"], "Utterance", ("Q: " if m["role"] == "user" else "A: ") + m["text"], "short", detail=m["text"])
                if m["conv"]:
                    node("conv:" + m["conv"], "Dialogue", "conversation", "short")
                    edge("conv:" + m["conv"], m["id"], "HAS_UTTERANCE", "short")
            mids = [m["id"] for m in msgs]
            for n in self._read("MATCH (a:Utterance)-[:FOLLOWED_BY]->(b:Utterance) WHERE a.id IN $ids AND b.id IN $ids RETURN a.id AS a, b.id AS b", ids=mids):
                edge(n["a"], n["b"], "FOLLOWED_BY", "short")

        if lens in ("reasoning", "all"):
            steps = self._read(
                """
                MATCH (st:ReasoningStep) WITH st ORDER BY st.ts DESC LIMIT 8
                RETURN st.id AS id, left(st.action, 60) AS action, st.reasoning AS reasoning ORDER BY st.ts
                """)
            for st in steps:
                node(st["id"], "ReasoningStep", st["action"], "reasoning", detail=st["reasoning"])
            sids = [st["id"] for st in steps]
            for t in self._read("MATCH (st:ReasoningStep)-[:MADE_LOOKUP]->(tc:Lookup) WHERE st.id IN $ids RETURN st.id AS a, tc.id AS id, tc.name AS name, tc.seq AS seq ORDER BY tc.seq", ids=sids):
                node(t["id"], "Lookup", t["name"], "reasoning")
                edge(t["a"], t["id"], "MADE_LOOKUP", "reasoning")
            infl = self._read("MATCH (st:ReasoningStep)-[:RELIED_ON]->(o:Object) WHERE st.id IN $ids RETURN st.id AS a, o.id AS b", ids=sids)
            entity_nodes(sorted({d["b"] for d in infl}))
            for d in infl:
                edge(d["a"], d["b"], "RELIED_ON", "reasoning")
        from .nams_graph import add_to_snapshot

        add_to_snapshot(self, lens, node, edge)  # NAMS-stored nodes from the shared database (read-only)
        return {"nodes": list(nodes.values())[: limit * 2], "edges": edges, "lens": lens}
