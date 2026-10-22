# Ontology review (Neo4j modeling best practices)

Reviewed 2026-10-22 against the `neo4j-modeling-skill` checklist. Scope: the live Aura model written by `graph.py`, `ontology/puppy_pi.json`
and `ontology.py`.

## Compliant
- Domain labels only (`Item`, `Place`, `Robot`, `Sighting`, `Session`, `Dialogue`, ...); no `:Entity`/`:Node`. App labels never clash with NAMS (tested).
- Every MERGE target has a uniqueness constraint (`Item.id`, `Sighting.id`, `Session.id`, `Place.name`/`id`, `Robot.name`, `Dialogue`, `Utterance`, `ReasoningStep`, `Lookup`).
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
- Added `Item.label`, `Sighting.ts` NOT NULL; `Sighting.ts` ZONED DATETIME, `Item.status` STRING, `Item.seen_count` INTEGER. These are
  Enterprise/Aura features, so failures are ignored (Community keeps working).

### No index on `Item.status` - INFO (fixed)
- `status = 'possibly_gone'` is filtered in several lenses/queries; added `item_status` index.

### Ontology picture omitted `RAN` and `CURRENTLY_IN` - INFO (fixed)
- Both edges are written but were not in `ontology.MEMORY_EDGES`, so the Ontology tab under-reported the schema.

### Docs used pre-migration labels - INFO (fixed)
- `ARCHITECTURE.md` showed `Observation/Location/Object`; now `Sighting/Place/Item` with the ontology-name mapping.

## Deferred (not changed, deliberate)
- **snake_case properties** (`first_seen`, `seen_count`, `room_type`, `image_ref`) vs the camelCase convention - INFO. Matches NAMS's property style
  and the JSON API; renaming needs a data migration plus every query and the UI. Do it as its own change if wanted.
- **Generic `id` key** - INFO. Prefixed values (`obj-`, `plc-`) keep it unambiguous; `itemId`/`placeId` would be clearer.
- **`Item.status` as a property** (`present`/`possibly_gone`) - fine: it changes often; a label would churn.
- **`AT` relationship name** is terse; `TAKEN_AT` would read better (rename = migration).
- **Sessions and Places** are bounded hubs (frames per session / place) but will grow in long deployments: if `Place` passes ~100K
  sightings, bucket `Sighting` by day (`(:Place)-[:HAS_DAY]->(:Day)<-[:ON_DAY]-(:Sighting)`).
- `ontology/puppy_pi.json` lists the domain edges only; `MEMORY_EDGES` in `ontology.py` carries the memory edges. Intentional split.
