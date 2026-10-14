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

## Status of verification
Unit/integration tests run against a fake NAMS built from the public OpenAPI spec (request/response shapes, headers, dedupe results,
401/500 handling). **Not yet verified against the live service**: the API key supplied on 2026-10-14 was rejected (401). Things to confirm
once a valid key exists: custom entity types accepted under the cloned `agent-memory` ontology, `X-Workspace-Id` handling, how NAMS
merges same-name entities, and what Observations/Reflections it generates from the narrative.
