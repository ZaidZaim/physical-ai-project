"""People memory in the graph: who the robot talked to, what they do, what they care about, and how warm a lead they are.

Model (camelCase keys, every MERGE key has a uniqueness constraint; written by people.py ONLY after a spoken yes)
  (Person {personId,name,nameKey,role,function,company,email,useCase,familiarity,wantsFollowUp,wantsDemo,followedUp,notes,
           leadScore,leadStage,consentAt,firstMet,lastMet,meetCount})
  (Person)-[:WORKS_AT]->(Company {name})
  (Person)-[:INTERESTED_IN]->(Theme {name})       what they want to learn about / build with
  (Person)-[:ENJOYED]->(Theme)                    what they liked most at the event
  (Person)-[:PARTICIPATED_IN]->(Encounter {id,ts,summary})-[:HELD_AT]->(Place)

`PeopleGraph` is the Aura implementation (mixed into graph.Memory); `FakePeople` is the in-RAM twin for PUPPY_FAKE_MEMORY / tests.
Both expose the same methods and return the same shapes (snake_case).
"""
from __future__ import annotations

from typing import Any

PERSON_CONSTRAINTS = [
    "CREATE CONSTRAINT person_person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.personId IS UNIQUE",
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (n:Company) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (n:Theme) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (n:Encounter) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX person_name_key IF NOT EXISTS FOR (n:Person) ON (n.nameKey)",
    "CREATE INDEX person_lead_stage IF NOT EXISTS FOR (n:Person) ON (n.leadStage)",
]
PEOPLE_LABELS = ("Encounter", "Theme", "Company", "Person")  # deletion order for reset()

_SCALARS = {  # python key -> graph key
    "name": "name", "name_key": "nameKey", "role": "role", "function": "function", "company": "company", "email": "email",
    "use_case": "useCase", "familiarity": "familiarity", "wants_follow_up": "wantsFollowUp", "wants_demo": "wantsDemo",
    "notes": "notes", "lead_score": "leadScore", "lead_stage": "leadStage",
    "next_step": "nextStep", "next_step_by": "nextStepBy",  # the follow-up / call to action for the Neo4j team; by "ai" or "team"
}

_PERSON_RETURN = """
    p.personId AS person_id, p.name AS name, p.nameKey AS name_key, p.role AS role, p.function AS function, p.company AS company,
    p.email AS email, p.useCase AS use_case, p.familiarity AS familiarity, coalesce(p.wantsFollowUp, false) AS wants_follow_up,
    coalesce(p.wantsDemo, false) AS wants_demo, coalesce(p.followedUp, false) AS followed_up, p.notes AS notes,
    coalesce(p.leadScore, 0) AS lead_score, coalesce(p.leadStage, 'cold') AS lead_stage, toString(p.firstMet) AS first_met,
    toString(p.lastMet) AS last_met, coalesce(p.meetCount, 0) AS meet_count,
    [(p)-[:INTERESTED_IN]->(t:Theme) | t.name] AS interests, [(p)-[:ENJOYED]->(t:Theme) | t.name] AS enjoyed,
    p.nextStep AS next_step, coalesce(p.nextStepBy, 'ai') AS next_step_by
"""


class PeopleGraph:
    """Mixin for graph.Memory (uses its _read/_write)."""

    def init_people_schema(self) -> None:
        from neo4j.exceptions import ClientError

        for q in PERSON_CONSTRAINTS:
            try:
                self._write(q)
            except ClientError as e:
                if "already exist" not in (e.message or "").lower():
                    raise

    def find_people(self, name_key: str) -> list[dict[str, Any]]:
        return self._read(f"MATCH (p:Person {{nameKey:$k}}) RETURN {_PERSON_RETURN}", k=name_key)

    def save_person(self, person: dict[str, Any], *, encounter_id: str, place: str | None, summary: str, ts: str) -> dict[str, Any]:
        """One statement, one round trip (from the booth each Aura round trip took ~1 s, and a save used to be 4-5 of them plus
        two reads back). Returns the record as given, with the counters the database keeps."""
        props = {_SCALARS[k]: v for k, v in person.items() if k in _SCALARS and v is not None}
        rows = self._write(
            """
            MERGE (p:Person {personId:$pid})
              ON CREATE SET p.firstMet = datetime($ts), p.meetCount = 0, p.followedUp = false
            SET p += $props, p.lastMet = datetime($ts), p.consentAt = coalesce(p.consentAt, datetime($ts))
            MERGE (e:Encounter {id:$eid})
              ON CREATE SET e.ts = datetime($ts), p.meetCount = p.meetCount + 1
            SET e.summary = $summary
            MERGE (p)-[:PARTICIPATED_IN]->(e)
            WITH p, e
            CALL (e) {
              OPTIONAL MATCH (pl:Place {name:$place})
              WITH e, pl WHERE pl IS NOT NULL
              MERGE (e)-[:HELD_AT]->(pl)
            }
            CALL (p) {
              WITH p WHERE $company <> ''
              MERGE (c:Company {name:$company})
              MERGE (p)-[:WORKS_AT]->(c)
              WITH p, c
              OPTIONAL MATCH (p)-[w:WORKS_AT]->(old:Company) WHERE old <> c
              DELETE w
            }
            CALL (p) {
              UNWIND $interests AS t
              MERGE (tp:Theme {name:t})
              MERGE (p)-[r:INTERESTED_IN]->(tp) ON CREATE SET r.since = datetime($ts)
            }
            CALL (p) {
              UNWIND $enjoyed AS t
              MERGE (tp:Theme {name:t})
              MERGE (p)-[r:ENJOYED]->(tp) ON CREATE SET r.since = datetime($ts)
            }
            RETURN p.meetCount AS meet_count, toString(p.firstMet) AS first_met, toString(p.lastMet) AS last_met,
                   coalesce(p.followedUp, false) AS followed_up
            """,
            pid=person["person_id"], eid=encounter_id, ts=ts, props=props, summary=(summary or "")[:500], place=place,
            company=(person.get("company") or "").strip(), interests=[t for t in person.get("interests", []) if t],
            enjoyed=[t for t in person.get("enjoyed", []) if t],
        )
        return {**person, **(rows[0] if rows else {})}

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        rows = self._read(f"MATCH (p:Person {{personId:$pid}}) RETURN {_PERSON_RETURN}", pid=person_id)
        if not rows:
            return None
        p = rows[0]
        p["encounters"] = self._read(
            """
            MATCH (:Person {personId:$pid})-[:PARTICIPATED_IN]->(e:Encounter)
            RETURN e.id AS id, toString(e.ts) AS ts, e.summary AS summary, head([(e)-[:HELD_AT]->(pl:Place) | pl.name]) AS place
            ORDER BY e.ts DESC LIMIT 20
            """, pid=person_id)
        return p

    def list_people(self, stage: str | None = None, q: str | None = None) -> list[dict[str, Any]]:
        return self._read(
            f"""
            MATCH (p:Person)
            WHERE p.personId IS NOT NULL AND ($stage IS NULL OR p.leadStage = $stage)
              AND ($search IS NULL OR toLower(coalesce(p.name,'') + ' ' + coalesce(p.company,'') + ' ' + coalesce(p.role,'')) CONTAINS $search
                   OR any(t IN [(p)-[:INTERESTED_IN|ENJOYED]->(t:Theme) | t.name] WHERE t CONTAINS $search))
            RETURN {_PERSON_RETURN} ORDER BY p.leadScore DESC, p.lastMet DESC LIMIT 500
            """, stage=stage, search=(q or "").strip().lower() or None)

    def forget_person(self, person_id: str) -> bool:
        """Erase a person and their encounters (the right to be forgotten). Themes and companies stay: they are not personal."""
        rows = self._write(
            """
            MATCH (p:Person {personId:$pid})
            OPTIONAL MATCH (p)-[:PARTICIPATED_IN]->(e:Encounter)
            WITH p, collect(e) AS es
            FOREACH (e IN es | DETACH DELETE e)
            DETACH DELETE p
            RETURN 1 AS done
            """, pid=person_id)
        self._write("MATCH (c:Company) WHERE NOT (c)<-[:WORKS_AT]-() DELETE c")
        self._write("MATCH (t:Theme) WHERE NOT (t)<-[:INTERESTED_IN|ENJOYED]-() DELETE t")
        return bool(rows)

    def set_followed_up(self, person_id: str, done: bool) -> bool:
        return bool(self._write("MATCH (p:Person {personId:$pid}) SET p.followedUp = $d RETURN 1 AS ok", pid=person_id, d=done))

    def set_next_step(self, person_id: str, text: str) -> bool:
        """The team wrote the next step themselves: from now on the AI suggestion never overwrites it."""
        return bool(self._write("MATCH (p:Person {personId:$pid}) SET p.nextStep = $t, p.nextStepBy = 'team' RETURN 1 AS ok",
                                pid=person_id, t=text.strip()[:500]))

    def people_graph(self) -> dict[str, Any]:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for r in self._read(
            """
            MATCH (p:Person) WHERE p.personId IS NOT NULL
            OPTIONAL MATCH (p)-[r:WORKS_AT|INTERESTED_IN|ENJOYED]->(x)
            RETURN p.personId AS pid, p.name AS name, coalesce(p.leadStage,'cold') AS stage, coalesce(p.leadScore, 0) AS score,
                   type(r) AS type,
                   CASE WHEN x:Company THEN 'Company' WHEN x:Theme THEN 'Theme' END AS kind, x.name AS xname LIMIT 1500
            """):
            nodes.setdefault(r["pid"], {"id": r["pid"], "kind": "Person", "label": r["name"], "stage": r["stage"], "score": r["score"]})
            if r["xname"]:
                xid = f"{r['kind']}:{r['xname']}"
                nodes.setdefault(xid, {"id": xid, "kind": r["kind"], "label": r["xname"]})
                edges.append({"source": r["pid"], "target": xid, "type": r["type"]})
        return {"nodes": list(nodes.values()), "edges": edges}

    def people_counts(self) -> dict[str, int]:
        return self._read(
            """
            RETURN count { (p:Person) WHERE p.personId IS NOT NULL } AS people, count { (:Person {leadStage:'hot'}) } AS hot,
                   count { (:Person {leadStage:'warm'}) } AS warm, count { (:Encounter) } AS encounters
            """)[0]


class FakePeople:
    """In-RAM twin of PeopleGraph (dev + tests)."""

    def _people(self) -> dict[str, dict]:
        if not hasattr(self, "_ppl"):
            self._ppl: dict[str, dict] = {}
        return self._ppl

    def init_people_schema(self) -> None: ...

    def find_people(self, name_key: str) -> list[dict[str, Any]]:
        return [self._view(p) for p in self._people().values() if p["name_key"] == name_key]

    def _view(self, p: dict) -> dict:
        return {**p, "interests": list(p["interests"]), "enjoyed": list(p["enjoyed"]), "encounters": [dict(e) for e in p["encounters"]]}

    def save_person(self, person: dict[str, Any], *, encounter_id: str, place: str | None, summary: str, ts: str) -> dict[str, Any]:
        p = self._people().setdefault(person["person_id"], {
            "person_id": person["person_id"], "first_met": ts, "meet_count": 0, "followed_up": False, "encounters": [],
            "interests": [], "enjoyed": [], "wants_follow_up": False, "wants_demo": False, "lead_score": 0, "lead_stage": "cold"})
        for k, v in person.items():
            if k in _SCALARS and v is not None:
                p[k] = v
        for key in ("interests", "enjoyed"):
            for t in person.get(key, []):
                if t and t not in p[key]:
                    p[key].append(t)
        p["last_met"] = ts
        enc = next((e for e in p["encounters"] if e["id"] == encounter_id), None)
        if enc is None:
            p["meet_count"] += 1
            p["encounters"].insert(0, {"id": encounter_id, "ts": ts, "summary": "", "place": place})
            enc = p["encounters"][0]
        enc["summary"] = (summary or "")[:500]
        return self._view(p)

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        p = self._people().get(person_id)
        return self._view(p) if p else None

    def list_people(self, stage: str | None = None, q: str | None = None) -> list[dict[str, Any]]:
        q = (q or "").strip().lower()
        out = []
        for p in self._people().values():
            hay = " ".join([p.get("name") or "", p.get("company") or "", p.get("role") or ""]).lower()
            if stage and p["lead_stage"] != stage:
                continue
            if q and q not in hay and not any(q in t for t in p["interests"] + p["enjoyed"]):
                continue
            out.append(self._view(p))
        return sorted(out, key=lambda p: (p["lead_score"], p["last_met"]), reverse=True)

    def forget_person(self, person_id: str) -> bool:
        return self._people().pop(person_id, None) is not None

    def set_next_step(self, person_id: str, text: str) -> bool:
        p = self._people().get(person_id)
        if p:
            p["next_step"], p["next_step_by"] = text.strip()[:500], "team"
        return p is not None

    def set_followed_up(self, person_id: str, done: bool) -> bool:
        p = self._people().get(person_id)
        if p:
            p["followed_up"] = done
        return p is not None

    def people_graph(self) -> dict[str, Any]:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for p in self._people().values():
            nodes[p["person_id"]] = {"id": p["person_id"], "kind": "Person", "label": p.get("name"), "stage": p["lead_stage"],
                                     "score": p["lead_score"]}
            targets = [("WORKS_AT", "Company", p["company"])] if p.get("company") else []
            targets += [("INTERESTED_IN", "Theme", t) for t in p["interests"]] + [("ENJOYED", "Theme", t) for t in p["enjoyed"]]
            for rel, kind, name in targets:
                xid = f"{kind}:{name}"
                nodes.setdefault(xid, {"id": xid, "kind": kind, "label": name})
                edges.append({"source": p["person_id"], "target": xid, "type": rel})
        return {"nodes": list(nodes.values()), "edges": edges}

    def people_counts(self) -> dict[str, int]:
        ps = list(self._people().values())
        return {"people": len(ps), "hot": sum(p["lead_stage"] == "hot" for p in ps), "warm": sum(p["lead_stage"] == "warm" for p in ps),
                "encounters": sum(len(p["encounters"]) for p in ps)}
