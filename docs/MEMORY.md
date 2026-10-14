# Memory design: Aura world model + NAMS agent memory

Two graphs, each doing what it is best at, cross-linked.

| | Aura (`graph.py`) | NAMS (`nams.py`) |
|---|---|---|
| Role | Authoritative **physical world model** | **Semantic + episodic** memory around it |
| Holds | Object / Location / Observation, `ON`/`NEXT_TO`... with `since`/`until`/`ended_by`, `possibly_gone`, miss counters | Entities (Robot, Place, PhysicalObject, Person if consented), asserted relationships, a **narrative log**, Q&A, reasoning traces |
| Good at | Exact, temporal questions: *what is on the table now? what changed? where was X last seen?* | Meaning and recall: *what did a visitor ask earlier? what story did the day tell?* (NAMS builds Observations and Reflections from the narrative messages) |
| Cannot | Fuzzy semantic recall | Update/delete relationships (create-only), update entity properties (name/description only), so it cannot be the time-aware truth |

## Shared database: labels this app owns
Your Aura instance also holds NAMS's own data (bring-your-own-database): it has `idx_entity_*`, `idx_message_*`, `idx_agentstep_*`,
`idx_reflection_*`, `idx_skill_*`, vector indexes and a `PipelineMeta` label. NAMS owns `Entity`, `Conversation`, `Message`, `AgentStep`,
`ToolCall`, `Observation`, `Reflection`, `Skill*`, `Alias`. To never collide, this app uses its own labels (a test enforces it):

| Concept | Memory type | This app (Aura) | NAMS equivalent |
|---|---|---|---|
| Camera frame analysed | short-term | `Sighting` (was `Observation`; migrated 2026-10-14) | `Observation` (message summaries) |
| One chat | short-term | `Dialogue`-[`HAS_UTTERANCE`]->`Utterance`-[`FOLLOWED_BY`]->`Utterance` | `Conversation`/`Message` |
| Entities and facts | long-term | `Object`, `Location`, `Robot` + `ON`/`NEXT_TO`/... | `Entity` + relationships |
| Why an answer | reasoning | `ReasoningStep`-[`MADE_LOOKUP`]->`Lookup`, `-[RELIED_ON]->Object`, `Utterance-[ABOUT]->Object` | `AgentStep`/`ToolCall`/`INFLUENCED` |

`reset` only deletes this app's labels. Because both live in one database, linking our `Object` to NAMS's `Entity` can later be a plain
Cypher relationship instead of an API round trip; reading NAMS's own conversations/reasoning into the lenses is a next step once the
NAMS key works (not implemented: its exact property names are unverified).

## Viewing memory by type
The graph panel has lenses: **Long-term** (entities and facts), **Short-term** (recent sightings and the conversation), **Reasoning**
(steps, lookups and the objects they relied on), and **All**. Colour = memory type (orange / green / purple); shape = kind of node.
Each answer is stored (in the background) as Dialogue/Utterance + ReasoningStep/Lookup so the short-term and reasoning lenses fill up.
The **Ontology** tab draws the schema itself: domain types from `ontology/puppy_pi.json` plus the memory types, sized by live counts.

## What flows where
Every observation (after Aura ingestion succeeds), on a background thread:
1. **Entities** (bulk; NAMS resolves duplicates by name+type): `Robot` (PuppyPi), `Place` (e.g. Booth), `PhysicalObject` per object instance.
   Look-alikes get distinct names (`black chair`, `black chair #2`). The NAMS id is stored back on the Aura node (`nams_id`, `nams_name`).
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
