# Ontology review (Neo4j modeling best practices)

Reviewed 2026-10-22 against the `neo4j-modeling-skill` checklist. Scope: the live Aura model written by `graph.py`, `ontology/puppy_pi.json`
and `ontology.py`.

## Compliant
- Domain labels only (`Item`, `Place`, `Robot`, `Sighting`, `Session`, `Dialogue`, ...); no `:Entity`/`:Node`. App labels never clash with NAMS (tested).
- Every MERGE target has a uniqueness constraint (`Item.itemId`, `Sighting.id`, `Session.id`, `Place.name`/`placeId`, `Robot.name`, `Dialogue`, `Utterance`, `ReasoningStep`, `Lookup`).
- Semantic, SCREAMING_SNAKE_CASE relationship types with meaningful direction (`DETECTED`, `MISSED`, `LOCATED_AT`, `ON`, ...).
- Facts with their own properties are modelled as nodes (`Sighting` for a frame, `Lookup` for a tool call); relation history is kept on edges (`active/since/until`).
- Real temporal types (`datetime()`), not strings. No embeddings on business nodes. All DDL uses `IF NOT EXISTS`.

## Issues found
### `Robot` supernode via `CAPTURED_BY` - WARNING (fixed)
- **Current**: every `Sighting` had `-[:CAPTURED_BY]->(Robot)`; one edge per camera frame, unbounded.
- **Problem**: `Robot` degree grows forever; traversals through it (e.g. `OBSERVED`, `CURRENTLY_IN`) slow down.
- **Fix**: redundant with `(Robot)-[:RAN]->(Session)<-[:IN_SESSION]-(Sighting)`. Ingest now writes `RAN` and no `CAPTURED_BY`;
  `init_schema` migrates old data (adds missing `RAN`, deletes `CAPTURED_BY`).

### Missing mandatory-property / type constraints - INFO (fixed, best-effort)
- Added `Item.label`, `Sighting.ts` NOT NULL; `Sighting.ts` ZONED DATETIME, `Item.status` STRING, `Item.seenCount` INTEGER. These are
  Enterprise/Aura features, so failures are ignored (Community keeps working).

### No index on `Item.status` - INFO (fixed)
- `status = 'possibly_gone'` is filtered in several lenses/queries; added `item_status` index.

### Ontology picture omitted `RAN` and `CURRENTLY_IN` - INFO (fixed)
- Both edges are written but were not in `ontology.MEMORY_EDGES`, so the Ontology tab under-reported the schema.

### Docs used pre-migration labels - INFO (fixed)
- `ARCHITECTURE.md` showed `Observation/Location/Object`; now `Sighting/Place/Item` with the ontology-name mapping.

### Naming conventions - INFO (fixed in a follow-up migration)
- **Property keys** were snake_case (`first_seen`, `seen_count`, `room_type`...); Neo4j's convention is camelCase. All keys this app stores
  on its nodes and relationships are now camelCase (`firstSeen`, `seenCount`, `roomType`, `endedBy`, `lastConfirmed`...). The JSON API and
  Python names stay snake_case: queries alias back (`o.firstSeen AS first_seen`), so the UI and the agent's tools are unchanged.
- **Generic `id` key** on the two entity types is now qualified: `Item.itemId`, `Place.placeId` (constraints `item_item_id`,
  `place_place_id`). Event-like nodes (`Sighting`, `Session`, `Dialogue`, ...) keep `id`.
- **`AT`** is now **`TAKEN_AT`** (`(Sighting)-[:TAKEN_AT]->(Place)`).
- `Memory.init_schema` migrates existing data on startup (`_migrate_keys`, idempotent): it renames the stored keys with dynamic property
  access, swaps the constraints, and rewrites `AT` edges. Nodes that also carry NAMS's `:Entity` label are never touched.
- Found while doing this: NAMS gives an extracted entity its type as a label, so the old "Agent Memory" ontology had written a
  `:Session:Entity` node into the shared database. `reset`, `reset-spatial` and `purge-demo` would have deleted it; they now skip `:Entity`.
- NAMS-side entity properties (`object_id`, `first_seen` on NAMS entities, sent by `nams.py`) are NAMS's data and were left as they are.

## Deferred (not changed, deliberate)
- **`Item.status` as a property** (`present`/`possibly_gone`) - fine: it changes often; a label would churn.
- **Sessions and Places** are bounded hubs (frames per session / place) but will grow in long deployments: if `Place` passes ~100K
  sightings, bucket `Sighting` by day (`(:Place)-[:HAS_DAY]->(:Day)<-[:ON_DAY]-(:Sighting)`).
- `ontology/puppy_pi.json` lists the domain edges only; `MEMORY_EDGES` in `ontology.py` carries the memory edges. Intentional split.
