"""Neo4j (Aura) contextual memory: ingestion with entity matching + temporal queries.

Model
  (Robot)-[:OBSERVED]->(Object)
  (Observation)-[:CAPTURED_BY]->(Robot), -[:IN_SESSION]->(Session), -[:AT]->(Location)
  (Observation)-[:DETECTED {confidence, zone}]->(Object)
  (Observation)-[:MISSED]->(Object)           evidence of absence (not proof of removal)
  (Object)-[:LOCATED_AT {first_seen,last_seen,count}]->(Location)
  (Object)-[:ON|INSIDE|NEXT_TO|NEAR|IN_FRONT_OF|BEHIND {active,since,until,last_confirmed}]->(Object)

Object.status: present | possibly_gone (>= MISS_THRESHOLD consecutive misses at the same location)
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase, RoutingControl

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
    def init_schema(self) -> None:
        for q in [
            "CREATE CONSTRAINT object_id IF NOT EXISTS FOR (n:Object) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT obs_id IF NOT EXISTS FOR (n:Observation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (n:Location) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT robot_name IF NOT EXISTS FOR (n:Robot) REQUIRE n.name IS UNIQUE",
            "CREATE INDEX object_label IF NOT EXISTS FOR (n:Object) ON (n.label)",
            "CREATE INDEX obs_ts IF NOT EXISTS FOR (n:Observation) ON (n.ts)",
        ]:
            self._write(q)
        self._write("MERGE (:Robot {name:$n})", n=self.s.robot_name)

    def reset(self) -> None:
        """Delete all memory this app owns. Never touches other labels."""
        for lbl in ("Observation", "Object", "Session", "Location", "Robot"):
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
                   coalesce(o.last_zone,'unknown') AS zone,
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
            MERGE (r:Robot {name:$robot})
            MERGE (s:Session {id:$sid}) ON CREATE SET s.started_at = datetime($ts)
            CREATE (ob:Observation {id:$obs, ts:datetime($ts), summary:$summary, source:$source,
                                    image_ref:$img, n_objects:$n})
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
            MATCH (ob:Observation {id:$obs})
            MATCH (o:Object)-[:LOCATED_AT]->(:Location {name:$loc})
            WHERE NOT o.id IN $vis AND o.status IN ['present','possibly_gone']
            SET o.miss_count = coalesce(o.miss_count,0)+1
            SET o.status = CASE WHEN o.miss_count >= $thr THEN 'possibly_gone' ELSE o.status END
            MERGE (ob)-[:MISSED]->(o)
            RETURN o.id AS id, o.label AS label, o.miss_count AS misses, o.status AS status
            """,
            loc=location, vis=visible, obs=obs_id, thr=self.s.miss_threshold,
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
        return {
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
            MATCH (ob:Observation) WHERE $loc IS NULL OR EXISTS { (ob)-[:AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }
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

    def find_object(self, label: str, color: str | None = None, limit: int = 10) -> list[dict]:
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
                   [(o)-[h]->(hb:Object) WHERE type(h) IN $preds |
                      {predicate: type(h), target: hb.label + ' ' + hb.id, currently: h.active,
                       last_seen_so: toString(h.last_confirmed), ended_by: h.ended_by}][..8] AS relation_history
            ORDER BY o.last_seen DESC LIMIT $limit
            """,
            label=label, color=color, preds=list(PREDICATES), limit=limit,
        )

    def related_objects(self, target_label: str, predicate: str = "ON", include_ended: bool = False) -> list[dict]:
        predicate = predicate.upper()
        if predicate not in PREDICATES:
            return [{"error": f"predicate must be one of {PREDICATES}"}]
        return self._read(
            f"""
            MATCH (a:Object)-[r:{predicate}]->(b:Object)
            WHERE toLower(b.label) CONTAINS toLower($t) AND ($ended OR r.active)
            RETURN a.id AS id, a.label AS label, a.color AS color, b.label+' '+b.id AS target,
                   r.active AS currently, a.status AS status,
                   toString(r.since) AS since, toString(r.last_confirmed) AS last_confirmed,
                   toString(r.until) AS until, r.ended_by AS ended_by, r.count AS times_confirmed
            ORDER BY r.last_confirmed DESC LIMIT 30
            """,
            t=target_label, ended=include_ended,
        )

    def object_history(self, object_id: str, limit: int = 25) -> dict[str, Any]:
        obs = self._read(
            """
            MATCH (o:Object {id:$id})
            OPTIONAL MATCH (ob:Observation)-[d:DETECTED]->(o)
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
            MATCH (ob:Observation) WHERE $loc IS NULL OR EXISTS { (ob)-[:AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }
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
            MATCH (ob:Observation)
            WHERE $loc IS NULL OR EXISTS { (ob)-[:AT]->(l:Location) WHERE toLower(l.name) = toLower($loc) }
            RETURN count(ob) AS observations, toString(min(ob.ts)) AS first_observation, toString(max(ob.ts)) AS last_observation
            """,
            loc=location,
        )
        return {"observations": span[0] if span else None, "objects": objs}

    def timeline(self, limit: int = 20) -> list[dict]:
        return self._read(
            """
            MATCH (ob:Observation)
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
            RETURN count { (:Object) } AS objects, count { (:Observation) } AS observations,
                   count { (:Location) } AS locations, count { (:Session) } AS sessions,
                   count { (:Object {status:'present'}) } AS present,
                   count { (:Object {status:'possibly_gone'}) } AS possibly_gone
            """
        )[0]

    def graph_snapshot(self, limit: int = 300) -> dict[str, Any]:
        """Nodes/edges with stable ids (Object.id, 'loc:<name>') so the UI can diff and animate."""
        nodes = self._read(
            """
            MATCH (n) WHERE n:Object OR n:Location
            RETURN CASE WHEN n:Object THEN n.id ELSE 'loc:' + n.name END AS id,
                   labels(n)[0] AS kind, coalesce(n.label, n.name) AS label,
                   n.color AS color, n.status AS status
            LIMIT $limit
            """,
            limit=limit,
        )
        edges = self._read(
            """
            MATCH (a:Object)-[r]->(b) WHERE (b:Object OR b:Location)
              AND type(r) IN $types AND coalesce(r.active, true)
            RETURN a.id AS source,
                   CASE WHEN b:Object THEN b.id ELSE 'loc:' + b.name END AS target, type(r) AS type
            LIMIT $limit
            """,
            types=list(PREDICATES) + ["LOCATED_AT"], limit=limit * 3,
        )
        return {"nodes": nodes, "edges": edges}
