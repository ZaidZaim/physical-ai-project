# Memory design: Aura world model + NAMS agent memory

## Memory layers and types, applied to a robot that lives in space

The umbrella stays the Neo4j agent-memory model: **short-term**, **long-term** and **reasoning** memory. Inside it sit the memory
types of cognitive science, and the usual split (conversational, semantic, episodic, procedural) does not get an extra box called
"spatial": for a robot, *where* runs through every type, the way the hippocampus both holds the map and stamps every episode with a
place. So each type here is the familiar one with a place attached (`ontology.py` files every label and relationship under a layer,
`memory`, and a type, `memory_type`):

| Layer | Memory type | Without space | With space (this robot) | Stored as |
|---|---|---|---|---|
| Short-term | working | the current turn | where am I, which way am I facing | `(Robot)-[:CURRENTLY_IN {heading}]->(Place)`, kept in RAM too |
| Short-term | episodic | the conversation so far | "at 10:42 I saw the mug on the kitchen table" | `Sighting -[:TAKEN_AT]-> Place`, `Dialogue`/`Utterance` |
| Long-term | semantic | "Zaid likes graphs" | "the mug is in the kitchen, on the table" (the map) | `Item -[:LOCATED_AT]-> Place`, spatial relations, `ADJACENT_TO` |
| Long-term | procedural | "how I answer FAQs" | "how to get from the hall to the kitchen", "no push-ups on this carpet" | `Route`, `Trick -[:PRACTISED_AT]-> Place` |
| Long-term | episodic | a summary of a past chat | the anonymous note of one booth conversation, and where it was held | `Encounter -[:HELD_AT]-> Place` |
| Reasoning | | | how an answer was reached, and which things it relied on | `ReasoningStep`, `Lookup`, `RELIED_ON` |

Conversational memory is the episodic memory of talk: `Dialogue`/`Utterance` here, `Conversation`/`Message` in NAMS. What a person
*tells* the robot about the world ("the keys are in the drawer") does not stay a sentence: it becomes a semantic fact marked `told`.

**Provenance and forgetting** (`knowledge.py`). Every semantic fact records how it was learned, as `source` on `LOCATED_AT`, the
spatial relations and `POSSIBLY_MOVED_FROM`:

| `source` | Starting confidence | Example |
|---|---|---|
| `seen` | the detector's confidence (about 0.9) | the camera saw the mug on the table |
| `told` | 0.6 | "the keys are in the kitchen drawer" (`remember_this` tool); a told-only object has status `told` and is never missed |
| `inferred` | 0.4 | a look-alike appeared in another room after the original stopped being seen |

A fact's confidence is the best of its evidence, halved every `FACT_HALF_LIFE_DAYS` (default 7) since that evidence; answers say
"sure" (0.7 and up), "probably" (0.4 to 0.7) or "not sure any more". Nothing is deleted by fading: episodes stay as the record of what
was seen, and a new sighting restores full confidence.

**Procedural memory** is learned by doing:
* `drive.py` logs every move as a segment (forward / back / turn, speed level, seconds). When the robot changes room directly (the same
  evidence that teaches `ADJACENT_TO`), the moves since it last settled in the old room become a `Route` (`STARTS_AT` the old room,
  `ENDS_AT` the new one): one per direction, keeping the quickest walk and the latest, with a walk count. `go_to_object` across rooms
  returns these moves per hop (`route_how`; a reversed route is marked as never walked that way). The robot never replays a route by
  itself: the gyro reports false turns while walking.
* Every trick played is an attempt at that place; one that went wrong (a person says so: `trick_feedback`, or `trick_went_wrong` at the
  booth) counts as failed, source `told`. A trick that failed `TRICK_AVOID_AFTER_FAILS` times (default 2) in a place, and more often
  than it worked, is skipped there unless someone presses its button.
* NAMS owns the label `Skill` (its own procedural memory for agents, e.g. reusable tool sequences), so these are `Route` and `Trick`.

Two graphs, each doing what it is best at, cross-linked.

| | Aura (`graph.py`) | NAMS (`nams.py`) |
|---|---|---|
| Role | Authoritative **physical world model** | **Semantic + episodic** memory around it |
| Holds | Object / Location / Observation, `ON`/`NEXT_TO`... with `since`/`until`/`endedBy`, `possibly_gone`, miss counters | Entities (Robot, Place, PhysicalObject; anonymous booth conversations and topics), asserted relationships, a **narrative log**, Q&A, reasoning traces |
| Good at | Exact, temporal questions: *what is on the table now? what changed? where was X last seen?* | Meaning and recall: *what did a visitor ask earlier? what story did the day tell?* (NAMS builds Observations and Reflections from the narrative messages) |
| Cannot | Fuzzy semantic recall | Update/delete relationships (create-only), update entity properties (name/description only), so it cannot be the time-aware truth |

## Shared database: labels this app owns
Your Aura instance also holds NAMS's own data (bring-your-own-database): it has `idx_entity_*`, `idx_message_*`, `idx_agentstep_*`,
`idx_reflection_*`, `idx_skill_*`, vector indexes and a `PipelineMeta` label. NAMS owns `Entity`, `Conversation`, `Message`, `AgentStep`,
`ToolCall`, `Observation`, `Reflection`, `Skill*`, `Alias`. To never collide, this app uses its own labels (a test enforces it):

| Concept | Memory type | This app (Aura) | NAMS equivalent |
|---|---|---|---|
| Camera frame analysed | short-term (episodic) | `Sighting` (was `Observation`; migrated 2026-10-14) | `Observation` (message summaries) |
| One chat | short-term (episodic) | `Dialogue`-[`HAS_UTTERANCE`]->`Utterance`-[`FOLLOWED_BY`]->`Utterance` | `Conversation`/`Message` |
| Entities and facts | long-term (semantic) | `Item`, `Place`, `Robot` + `LOCATED_AT`, `ON`/`NEXT_TO`/... with `source` | `Entity` + relationships |
| Know-how | long-term (procedural) | `Route`, `Trick`-[`PRACTISED_AT`]->`Place` | `Skill` (not used: a different idea, and its label) |
| Why an answer | reasoning | `ReasoningStep`-[`MADE_LOOKUP`]->`Lookup`, `-[RELIED_ON]->Object`, `Utterance-[ABOUT]->Object` | `AgentStep`/`ToolCall`/`INFLUENCED` |

`reset` only deletes this app's labels. Because both live in one database, linking our `Object` to NAMS's `Entity` can later be a plain
Cypher relationship instead of an API round trip; reading NAMS's own conversations/reasoning into the lenses is a next step once the
NAMS key works (not implemented: its exact property names are unverified).

## Viewing memory by type
The graph panel has one lens per layer: **Long-term** (entities and facts, plus routes and tricks beside the places they belong to),
**Short-term** (recent sightings and the conversation), **Reasoning** (steps, lookups and the objects they relied on), and **All**.
Colour = layer (orange / green / purple); shape = kind of node. `/api/graph?lens=semantic` (or `procedural`, `episodic`, `working`)
shows the layer that type belongs to. Each answer is stored (in the background) as Dialogue/Utterance + ReasoningStep/Lookup so the
short-term and reasoning lenses fill up.
The **Ontology** tab draws the schema itself: domain types from `ontology/puppy_pi.json` plus the memory types, sized by live counts.

## What flows where
Every observation (after Aura ingestion succeeds), on a background thread:
1. **Entities** (bulk; NAMS resolves duplicates by name+type): `Robot` (PuppyPi), `Place` (e.g. Booth), `PhysicalObject` per object instance.
   Look-alikes get distinct names (`black chair`, `black chair #2`). The NAMS id is stored back on the Aura node (`namsId`, `namsName`).
2. **Relationships**: `OBSERVED` (robot to object, confidence), `LOCATED_AT` (new objects), and the spatial ones
   (`ON`, `INSIDE`, `NEXT_TO`, `NEAR`, `IN_FRONT_OF`, `BEHIND`) with `observed_at`, `observation_id`, `location`.
3. **Narrative message** to an *observations* conversation: one block per observation (seen, new, spatial, possibly gone).
   NAMS extracts mentions and summarises windows into Observations/Reflections on its side.
4. **Status** changes go in the entity description (`possibly gone since ...` / `back in view ...`) because that is what NAMS lets you update.

Questions and answers go to a *chat* conversation per session, with a reasoning step and tool calls per answer (provenance).
The agent has a `recall_memory` tool (entity search + message search). Rule: **graph tools decide facts about the world; recall adds
history and context; if they disagree, the graph tools win.**

## Safety properties
* Asynchronous and best-effort: the robot never waits on NAMS; errors are counted and surfaced (`/api/state` -> `nams_status`, page header).
* A 401 switches the layer off (no retry storm) and says how to fix it.
* Scripted/test activity (`seed`, `eval`) does **not** write to NAMS unless you pass `--nams`.
* People: only anonymous presence is stored by default; a `Person` entity requires `PEOPLE_CONSENT=true` (not implemented as an automatic write).

## Setup
`.env`: `NAMS_API_KEY`, `NAMS_WORKSPACE_ID`, `NAMS_BASE_URL` (the workspace is sent as `X-Workspace-Id`).
```
uv run puppy-mind nams-check              # key, workspace, active ontology, entity count
uv run puppy-mind nams-setup              # create the ontology version (ontology/puppy_pi.json); does NOT activate
uv run puppy-mind nams-setup --activate   # switch the workspace to it (changes how NAMS extracts entities)
uv run puppy-mind nams-sync               # backfill the current Aura world model into NAMS
```
`validation_mode` is `permissive`, so entities of these types are accepted even before activation (unverified; see below).

## Verified live against NAMS (2026-10-14)
Run with a real key against your workspace; every test object was deleted afterwards. What was measured, and what the code now does about it:

| Finding | Consequence in the code |
|---|---|
| The key accepts memory read/write but **not** workspace/ontology admin (`/v1/workspace` and ontology management return 403) | The app uses the types of the workspace's **active** ontology (`Agent`, `Location`, `Object`, `Person`); `nams-setup` explains the 403 |
| NAMS writes into the **same Aura database** (`:Entity`, `:Conversation`/`:Message`, `:AgentStep`/`:ToolCall`, `RELATED_TO`) | `nams_graph.py` reads them for the lenses and adds `MIRRORED_BY` and `RELIED_ON` links from our nodes; it never creates or deletes NAMS-labelled nodes (tested) |
| Relationships are stored as generic `RELATED_TO` with the real name in `predicate` (`next_to`, `observed`, `located_at`) | Lenses read `predicate`; spatial duplicates of our own facts are not redrawn |
| Duplicate resolution: auto-merge at similarity >= 0.92, review flag from 0.78 (`/v1/workspace/resolution`, readable) | `black chair` / `black chair #2` was **merged**, as were `mug (red)`/`mug (blue)` and any name+short-id without colour. Names are `label colour short-id` (`black chair 3f2a9c`), which stayed separate; a `merged` result is never mirrored 1:1 (retried with a more distinct name), and review flags between our own objects are rejected via the reviews API |
| Different `userId`s keep entities apart, but relationships across users are refused (`cross_user_relationship`) | One shared scope; `userId` is not used for objects |
| `Agent` requires property `agent_id` | Set to the robot name; per-item errors are now surfaced instead of swallowed |
| `messages/search` returned nothing right after writing (text search, possibly indexed asynchronously) | `recall_memory` also uses entity search; message recall unverified |

Real data in NAMS now: `nams-sync --rebuild` mirrored the robot's 11 real objects, the `Demo Area` location and the `PuppyPi` agent (13 entities, 32 relationships).
Not verified: what NAMS's Observations/Reflections look like (they appear only after message windows are processed), and the effect of confirming a
review pair in the NAMS dashboard (never do this for two of our objects: they are different instances).
