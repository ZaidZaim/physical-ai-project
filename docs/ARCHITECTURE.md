# Architecture notes

## Graph schema
```
(Robot)-[:RAN]->(Session)<-[:IN_SESSION]-(Observation)-[:CAPTURED_BY]->(Robot)
(Observation)-[:AT]->(Location)
(Observation)-[:DETECTED {confidence, zone}]->(Object)     positive evidence
(Observation)-[:MISSED]->(Object)                          expected-but-absent evidence
(Object)-[:LOCATED_AT {first_seen,last_seen,count}]->(Location)
(Object)-[:ON|INSIDE|NEXT_TO|NEAR|IN_FRONT_OF|BEHIND {active,since,until,last_confirmed,last_obs}]->(Object)
Object {id,label,color,description,status,first_seen,last_seen,seen_count,miss_count,last_zone}
```
Model, scenes, and PDF mapping: PDF's `OBSERVED`, `LOCATED_AT`, `NEXT_TO`, `CONTAINS`(= inverse of `INSIDE`),
`PREVIOUSLY_SEEN_AT` (= older `LOCATED_AT` edges / `DETECTED` history) are all represented.

## Ingestion pipeline (graph.Memory.ingest)
1. Filter detections/relations below `MIN_CONFIDENCE`.
2. Load known objects at the same Location with their active relations (anchors).
3. `matching.assign`: greedy one-to-one scoring (label gate, colour, zone, anchors); below threshold → new object.
4. Write observation, objects, evidence edges; upsert relations; supersede exclusive ON/INSIDE; end relations between
   two visible objects that were not re-seen.
5. Objects at this location not detected: `miss_count++`, `MISSED` edge; `possibly_gone` at `MISS_THRESHOLD`.

## Agent
OpenAI tool-calling over fixed graph tools (+ guarded read-only Cypher). The model cannot see the camera or any
memory except through tool results; the tool trace is returned with each answer and recorded in NAMS.

## Known weak points (by design, tracked)
* Identity of look-alike objects relies on colour/zone/relation anchors; the same chair seen from another angle may split.
* "Zone" is image-relative, not world-relative, so it is a weak cue after the robot moves.
* Consecutive-miss logic assumes the camera keeps covering the object; a robot looking elsewhere in the same named
  location can create false `possibly_gone`. Mitigation: set finer Location names per viewpoint.
* NAMS endpoints were checked against its OpenAPI spec but not exercised without an API key.

## graph-explorer-kit
`ZaidZaim/graph-explorer-kit` is a document-GraphRAG scaffold (ProjectSpec, `owns_graph=False` adoption mode, HTTP API
under `/api/*`). It is not wired in: its schema/retrieval model targets Document→Chunk→Entity graphs, whereas this
world model is Observation/Object. Practical use today: point Neo4j Explore/Browser at the same Aura instance. A
ProjectSpec adapter is a reasonable next step once the schema stabilises.
