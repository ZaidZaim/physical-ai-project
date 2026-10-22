<p align="center"><img src="docs/logo.svg" width="96" alt="Puppy Mind logo"></p>

<h1 align="center">puppy-mind</h1>
<p align="center"><b>Talk with your environment.</b> Spatial memory and physical AI for the Hiwonder PuppyPi, on Neo4j.</p>

A small quadruped robot walks through your home and keeps looking. Every camera frame becomes structured facts (objects, where
they are, what sits on what), stored in a **Neo4j Aura** graph that remembers how the world changes over time. You can then ask it
questions in plain language, and an agent answers **only from that memory**, with timestamps and evidence. It can also turn to face
things it remembers.

*Document intelligence is "talk to your PDF". Physical intelligence is "talk with your environment".*

> "What's on the table?" · "What changed since this morning?" · "Where did you last see the red backpack?" · "Turn to the cat."

---

## What it can do

| | You ask | What happens |
|---|---|---|
| **Perceive** | "What do you see?" | OpenAI vision turns a frame into objects, colours, zones and spatial relations |
| **Remember** | "What's on the table?" | Objects keep a stable identity across frames; relations (`ON`, `NEXT_TO`, ...) are temporal edges |
| **Notice change** | "What changed?" | New objects, things not seen any more (`possibly_gone`, never "deleted"), relations that ended |
| **Know where** | "Which rooms do you know?" | Rooms are recognised from what is in them; a room map is learned as the robot moves |
| **Explain** | "Where did you last see it?" | Every answer comes with the tool calls and graph facts it used |
| **Act** | "Turn to the door." | Turns in place to face a remembered object, or looks around for it. Between rooms it gives you the route. It never walks on its own |

## How it works

```
 PuppyPi                         Laptop: the brain (this repo)                          Cloud
 ┌────────────────┐              ┌──────────────────────────────────────────────┐       ┌────────────────────────┐
 │camera (MJPEG)  │  frames ──►  │ perception   OpenAI vision -> Scene          │       │ Neo4j Aura             │
 │gyro, battery   │  gyro   ──►  │ matching     which known object is this?     │  ──►  │ world model graph      │
 │rosbridge       │  turn   ◄──  │ spatial      which room, which way it faces  │       │ (Item, Place, ...)     │
 └────────────────┘              │ agent        OpenAI tools over graph queries │       ├────────────────────────┤
                                 │ navigate     turn to / look for an object    │  ──►  │ NAMS agent memory      │
                                 │ web UI + API FastAPI, live graph, drive      │       │ chat, reasoning, recall│
                                 └──────────────────────────────────────────────┘       └────────────────────────┘
```

No Docker: Aura hosts the graph, NAMS (the Neo4j Agent Memory Service) hosts agent memory, and the robot only streams frames and
takes velocity commands over its stock rosbridge. Nothing has to be installed on the robot.

### Two memories, one database

| | Holds | Code |
|---|---|---|
| **World model** (authoritative) | objects, places, spatial relations, every sighting and miss, change over time | Aura, `graph.py` |
| **Agent memory** (optional) | conversations, reasoning steps and tool calls, mirrored entities, semantic recall | NAMS, `nams.py` |

NAMS writes into the **same** Aura database under its own labels (`Entity`, `Message`, `AgentStep`, ...). The app never writes those
labels and never deletes NAMS's nodes. `MIRRORED_BY` and `RELIED_ON` link the two.

### The graph

```
(Robot)-[:RAN]->(Session)<-[:IN_SESSION]-(Sighting)-[:TAKEN_AT]->(Place)
(Robot)-[:CURRENTLY_IN {heading}]->(Place)
(Sighting)-[:DETECTED {confidence, zone, bearing}]->(Item)       evidence it was there
(Sighting)-[:MISSED]->(Item)                                     expected, not seen (not proof it is gone)
(Item)-[:LOCATED_AT {firstSeen, lastSeen, count}]->(Place)
(Item)-[:ON|INSIDE|NEXT_TO|NEAR|IN_FRONT_OF|BEHIND {active, since, until, lastConfirmed}]->(Item)
(Place)-[:ADJACENT_TO {count}]-(Place)                           learned from moving between rooms
(Item)-[:POSSIBLY_MOVED_FROM {confidence}]->(Item)               a hypothesis, never asserted as fact
Item  {itemId, label, color, status, firstSeen, lastSeen, seenCount, missCount, lastBearing}
Place {placeId, name, roomType, visits}
```

The model follows Neo4j modeling practice: camelCase keys, a uniqueness constraint on every `MERGE` key, existence and type
constraints, no write-only edges and no supernodes ([review](docs/ONTOLOGY_REVIEW.md)). Startup migrates older data automatically.
To look at it in Aura, use the one-question queries in [docs/AURA_VIEWS.md](docs/AURA_VIEWS.md). `MATCH (n)-[r]->(m)` draws
everything at once, NAMS included, and is unreadable.

---

## Quick start

**You need:** Python 3.10+, [uv](https://docs.astral.sh/uv/), a Neo4j AuraDB instance and an OpenAI API key. A NAMS key is optional.

```bash
uv sync --extra dev
cp .env.example .env              # fill in NEO4J_*, OPENAI_API_KEY (NAMS_* optional)
uv run pytest                     # unit tests: no network needed
uv run puppy-mind init            # constraints, indexes and migrations in Aura
```

**Try it without a robot:**

```bash
PUPPY_FAKE_MEMORY=1 uv run puppy-mind serve                        # UI on in-RAM memory: nothing persists
CAMERA_SOURCE=webcam:0 uv run puppy-mind watch                     # your laptop camera, into Aura
CAMERA_SOURCE=folder:./frames uv run puppy-mind watch              # replay photos
```

**With the robot** (on your Wi-Fi, or join its `HW...` hotspot):

```bash
uv run puppy-mind robot-test --real-memory                         # finds the robot, starts UI + camera + Aura memory
```

Open http://localhost:8000, click **Start camera**, and ask away. Robot networking and the camera URL:
[docs/ROBOT_SETUP.md](docs/ROBOT_SETUP.md). Without `--real-memory`, `robot-test` is a driving test that never sends frames to OpenAI.

## Using it

### The web UI

- **Live camera** with the detected labels drawn over the left, centre and right thirds of the frame. One frame is analysed per
  `WATCH_INTERVAL_S`, and a scan line shows while that happens.
- **Live graph** over Server-Sent Events. New objects glow green, re-seen ones pulse blue, misses pulse amber, `possibly_gone` objects
  turn grey, and the nodes the agent used for your last answer glow purple. Lenses: long-term, short-term, reasoning, all.
- **Places:** the learned room map. Rename or merge rooms the robot discovered on its own.
- **Ontology:** the schema as a picture, with live counts. **Timeline:** every sighting.
- **Ask:** chat with the robot. Every answer has an **Evidence** section with the exact tool calls.

### Asking questions

The agent can only learn about the world through graph tools, so it cannot make things up from general knowledge. It cites
timestamps, tells "currently visible" apart from "remembered" and "possibly gone", and treats `POSSIBLY_MOVED_FROM` as a hypothesis.

```bash
uv run puppy-mind ask "What changed since the beginning?" --evidence
uv run puppy-mind ask                                    # interactive chat
```

### Driving

Click **Enable keyboard drive**; nothing moves until you arm it. Hold a key to move, and release it to stop:
`W/S` forward/back · `A/D` turn · `Space` stop · `H` stand · `1 2 3` speed · `Esc` disarm (the arrow keys and on-screen pad work too).

### "Go where you last saw it"

Ask *"turn to the cat"* in the chat, or call `POST /api/nav/go {"label": "cat"}` (add `"execute": false` for the plan only). The robot
has a gyro heading and the direction of every object it has seen, but **no obstacle sensing and no position in the room**, so it
does only what is safe:

| Situation | It will |
|---|---|
| Same room, object seen during this visit | turn in place to face it (closed loop on the gyro) |
| Same room, direction unknown | look around: turn in steps, let the camera check each view, stop and centre when it is seen |
| Another room | give the route over the learned room map, e.g. `Bathroom -> Hallway -> Dining Room`, for **you** to drive |
| Room not recognised yet / no gyro | say so, and not move |

## Safety and privacy

- **Nothing moves unless you arm it** in the UI. Commands expire after 0.4 s of silence (a dead-man). Closing the tab, losing focus,
  Esc or a network drop stops and disarms it. The turn rate is capped at 25 deg/s, the limit Hiwonder's own teleop uses.
- **You always win over the navigator:** any key, Space or Cancel takes over at once. A lost gyro, a stalled turn, a timeout or a change
  of room aborts it.
- With rosbridge driving the dead-man runs on the laptop, so if the laptop crashes mid-walk the robot is not told to stop. Keep the
  speed low and a hand near the power switch, or run `robot/puppy_drive.py` on the robot, which stops on its own.
- Driving from another machine needs `BRAIN_TOKEN`.
- **People:** stored only as anonymous `person` presence (no description, colour or image) unless `PEOPLE_CONSENT=true`.
- **Frames** are sent to OpenAI only while frame analysis is on, and saved to disk only if `STORE_IMAGES=true`.

## Memory rules that matter

- **One miss never deletes anything.** Only `MISS_THRESHOLD` consecutive misses, at the same place and while the object should have
  been in view (by bearing, view overlap or a visible anchor), make it `possibly_gone`. Answers then say "not seen", never "removed".
- **Identity is conservative.** A detection matches a known object only on the same label, a compatible colour and its
  zone/relation anchors. When in doubt it becomes a new object, because a wrong merge corrupts history and duplicates can be merged later.
- **Relations are temporal** (`since`, `until`, `active`). `ON` and `INSIDE` are exclusive per object.
- **Object identity is per room:** a chair in the kitchen and one in the living room are different objects.
- **Moving never looks like loss:** perception pauses while the robot moves, and the first frame afterwards does not count misses.

## Command line

| Command | What it does |
|---|---|
| `init` | constraints, indexes and migrations in Aura; checks connectivity |
| `serve` · `robot-test [--real-memory]` | web UI + API · the same plus finding the robot, its camera and rosbridge |
| `watch` · `observe <image>` | analyse `CAMERA_SOURCE` continuously · one image file |
| `ask ["question"] [--evidence]` | ask once, or chat |
| `stats` | counts in memory |
| `consolidate [--place P] [--yes]` · `split-place <place> [--yes]` | merge look-alike duplicates · split a place that mixed rooms (both dry runs by default) |
| `imu-check --robot-ip <ip>` | turns the robot left for 2 s and checks that the gyro heading rises |
| `robot-check <ip\|auto>` | probes the robot and finds a working camera URL |
| `nams-check` · `nams-setup [--activate]` · `nams-sync [--rebuild]` | NAMS key/workspace · install the PuppyPi ontology · backfill entities |
| `seed --yes` · `eval` | write a scripted "booth day" · the 20-question check (**disposable database only**, see below) |
| `purge-demo [--yes]` · `reset-spatial [--yes]` · `reset --yes` | remove demo data only · all places/objects/sightings · everything this app stores |

## Configuration

Set in `.env` (see [.env.example](.env.example)). `.env` overrides variables already set in your shell.

| Variable | Default | Meaning |
|---|---|---|
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | - , `neo4j`, - , `neo4j` | the Aura database |
| `OPENAI_API_KEY`, `OPENAI_VISION_MODEL`, `OPENAI_CHAT_MODEL` | - , `gpt-4o`, `gpt-4o` | perception and the agent |
| `NAMS_API_KEY`, `NAMS_WORKSPACE_ID`, `NAMS_BASE_URL` | off | optional agent memory (`https://memory.neo4jlabs.com`) |
| `CAMERA_SOURCE` | - | `webcam:0`, `folder:./frames`, or the robot's MJPEG URL |
| `WATCH_INTERVAL_S` | `5` | seconds between analysed frames |
| `ROBOT_NAME`, `DEFAULT_LOCATION` | `PuppyPi`, `Demo Area` | |
| `AUTO_PLACE`, `PLACE_CONFIRM_FRAMES` | `true`, `2` | recognise rooms automatically; frames needed to confirm a change of room |
| `MIN_CONFIDENCE`, `MISS_THRESHOLD`, `MIN_OVERLAP` | `0.5`, `3`, `0.5` | detection floor; misses before `possibly_gone`; view overlap that counts as evidence |
| `CAMERA_FOV_DEG` | `70` | horizontal field of view, used for bearings |
| `OBJECT_VOCAB`, `STRICT_VOCAB` | a common list, `false` | the labels to prefer (or allow only) |
| `PEOPLE_CONSENT`, `STORE_IMAGES` | `false`, `false` | privacy switches |
| `BRAIN_TOKEN` | - | required to drive or push frames from another machine |

## Testing

```bash
uv run pytest                          # 117 unit tests: matching, places, heading, navigation (simulated robot), NAMS (fake server)
uv run python eval/verify_spatial.py   # live Cypher checks: safe on your REAL database (isolated zz- namespace, cleaned up)
uv run python eval/check_views.py      # runs every query in docs/AURA_VIEWS.md, read-only
uv run puppy-mind imu-check --robot-ip <ip>   # on the robot: the gyro sign the navigator relies on
```

Unit tests run against an in-memory stand-in, so they do not exercise the Cypher: run `verify_spatial.py` after changing any query.
`seed`, `eval` and `eval/verify_graph.py` expect a database holding **only** the scripted scene. Run them on an empty, throwaway
database, never on the one with real robot memory. If demo data did get in, `purge-demo` removes just that. What has been verified
on real hardware, and when: [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Project layout

```
src/puppy_mind/   brain.py (orchestration) · perception.py · graph.py (Aura) · matching.py · spatial.py (rooms) · pose.py (heading)
                  navigate.py (phase 5) · agent.py · nams.py / nams_graph.py · drive.py / rosbridge.py · server.py · cli.py
                  static/index.html (the UI)
robot/            puppy_drive.py, puppy_streamer.py: optional scripts for running on the robot
ontology/         puppy_pi.json: the domain ontology (also installed into NAMS)
eval/             scenarios, the 20 questions, live checks
tests/            unit tests
docs/             design notes (below)
```

## Docs

| | |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | ingestion pipeline, schema, known weak points |
| [SPATIAL.md](docs/SPATIAL.md) | room recognition, room map, view-aware misses, gyro heading |
| [MEMORY.md](docs/MEMORY.md) | Aura + NAMS together, measured NAMS behaviour, setup |
| [ONTOLOGY_REVIEW.md](docs/ONTOLOGY_REVIEW.md) | the model checked against Neo4j modeling practice |
| [AURA_VIEWS.md](docs/AURA_VIEWS.md) | clean, one-question queries for Aura |
| [ROBOT_SETUP.md](docs/ROBOT_SETUP.md) | getting the robot on the network, camera, drive |
| [VERIFICATION.md](docs/VERIFICATION.md) | what was tested on real hardware and services, and when |

## Honest limits

- **One camera, no position:** spatial relations are qualitative, and "where" means a room, not coordinates. The heading is relative
  to where the robot entered the room and drifts slowly; it is reset on every change of room.
- **No obstacle sensing,** so the robot never drives itself anywhere. Phase 5 turns in place only.
- **Room recognition can be fooled** by a single unusual frame (once, the Hallway stairs became a "Garage"). Merge such places in the
  Places tab.
- **Look-alike objects** seen from different angles can split into two. `consolidate` merges ones that never appear in the same frame.
- `eval` uses a scripted scene and substring checks: a floor, not a proof. The model is non-deterministic, so re-run it after prompt changes.
