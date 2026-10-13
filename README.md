# puppy-mind: talk with your environment

Physical AI for the Hiwonder **PuppyPi**: the robot's camera becomes a stream of structured observations,
**Neo4j Aura** is its persistent world model, and an OpenAI-backed agent answers questions grounded in that graph.
*Document Intelligence → "talk to your PDF". Physical Intelligence → "talk with your environment".*

```
 PuppyPi (camera)                      Laptop / server ("brain")                         Cloud
 ┌──────────────┐  JPEG frames   ┌─────────────────────────────────────────┐   ┌───────────────────────┐
 │ puppy_streamer│ ────────────► │ perception (OpenAI vision, structured)  │   │ Neo4j Aura            │
 │ (or MJPEG pull)│              │   ↓ Scene{objects, relations}           │──►│ world model graph     │
 └──────────────┘                │ entity matching + temporal reconciliation│   │ Object/Observation/…  │
                                 │   ↓                                      │   └───────────────────────┘
   Phase 5 (later):  ◄────────── │ agent (OpenAI tools over graph queries) │   ┌───────────────────────┐
   navigation commands           │ web UI + REST (FastAPI)                 │──►│ NAMS (hosted)         │
                                 └─────────────────────────────────────────┘   │ chat + reasoning mem. │
                                                                               └───────────────────────┘
```

No Docker anywhere: Aura hosts the graph, NAMS hosts agent memory, the Pi only streams frames.

## Why two memories
| | Holds | Where |
|---|---|---|
| World model | objects, places, spatial relations, observation history, change over time | **Aura** (`graph.py`) |
| Agent memory | who asked what, reasoning traces / tool calls, mirrored object entities | **NAMS** (`nams.py`, optional) |

## Phase map (user goals → what is built)
| Phase | User goal | Where |
|---|---|---|
| 1 Perceive | "What do you see?" | `perception.py` → `Scene`; UI "Right now" |
| 2 Lookup | "What's on the table?" | `related_objects`, `find_object` tools |
| 3 Analyze | "What changed?" | entity matching (`matching.py`), relation `since/until`, `changes_since` |
| 4 Understand | "Where did you last see the red backpack?" | `object_history`, `first_observation`, evidence trace in every answer |
| 5 Interact | "Go where you last saw it" | **not built**; the graph already stores `LOCATED_AT`/zones. Needs localisation + safety (see docs/ARCHITECTURE.md) |

## Quick start
```bash
uv sync --extra dev
cp .env.example .env          # fill NEO4J_*, OPENAI_API_KEY (NAMS_API_KEY optional)
uv run pytest                 # matching logic, no network
uv run puppy-mind init        # constraints + indexes in Aura
uv run puppy-mind seed        # replay a scripted "booth day" (no camera, no OpenAI)
uv run puppy-mind ask "What changed since the beginning?" --evidence
uv run puppy-mind eval        # the 20-question MVP check (needs OPENAI_API_KEY)
uv run puppy-mind serve       # UI at http://localhost:8000
```
Without the robot: `CAMERA_SOURCE=webcam:0 uv run puppy-mind watch`, or `folder:./frames` to replay photos.
With the robot: see [docs/ROBOT_SETUP.md](docs/ROBOT_SETUP.md).

## Memory rules that matter
* **One missed detection never deletes anything.** Objects carry `miss_count`; only after `MISS_THRESHOLD`
  consecutive misses *at the same location* do they become `possibly_gone`, and answers say "not seen", never "removed".
* **Identity is conservative.** A detection matches a known object only on same label + compatible colour
  (+ zone/relation anchors). Doubt → new object. Wrong merges corrupt history; duplicates can be merged later.
* **Relations are temporal edges** (`ON {since, until, active}`); ON/INSIDE are exclusive per subject.
* **Every answer carries its tool trace** (UI: "Evidence"); NAMS stores the same trace as reasoning steps.
* **Privacy by default:** people are stored as anonymous `person` presence only (no description/colour/images) unless
  `PEOPLE_CONSENT=true`; frames are not saved unless `STORE_IMAGES=true`.

## Layout
`src/puppy_mind/` brain (perception, graph, matching, agent, nams, server, cli) · `robot/` on-robot streamer ·
`eval/` scenarios + 20 questions · `tests/` · `docs/`.

## Status / honest limits
* Graph logic and matching are unit-tested (matching) and designed against Aura; the Aura-backed ingestion/queries have NOT yet been executed against a live instance (connection was failing when this was written).
* Single camera, no pose: spatial relations are qualitative and "location" is a name you set (`/api/location`).
* Eval questions use substring checks, a floor not a proof; extend with real captured scenes.
