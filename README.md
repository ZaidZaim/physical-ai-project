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
| 5 Interact | "Go where you last saw it" | `navigate.py`: turns in place to face it, or looks around for it; between rooms it gives the route for you to drive. Never walks on its own (see below) |

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

Looking at the graph in Aura itself: use the per-question queries in [docs/AURA_VIEWS.md](docs/AURA_VIEWS.md), not `MATCH (n)-[r]->(m)`.

## Live view (camera + animated graph)
`uv run puppy-mind serve`, open http://localhost:8000.
* **Live camera**: MJPEG at `/api/stream`. Frames come from the robot (`robot/puppy_streamer.py` pushes ~3 fps to `/api/frame`)
  or from the laptop (`Start camera` pulls `CAMERA_SOURCE`). Video is shown at full rate; OpenAI perception runs on at most
  one frame per `WATCH_INTERVAL_S`, and a scan line shows while a frame is being analysed.
* **Graph in real time**: every observation is pushed over Server-Sent Events (`/api/events`). New objects grow in and glow
  green, re-seen objects pulse blue, misses pulse amber, objects that become `possibly_gone` turn grey, ended relations flash
  red and drop out, and the nodes the agent consulted for your question glow purple. Detected labels also appear over the
  left/centre/right third of the video they were seen in.
* **`Demo replay`** plays a scripted booth day (`eval/scenarios/booth.json`) so you can watch the graph build without a robot.
* **Dev only:** `PUPPY_FAKE_MEMORY=1` swaps Aura for an in-memory stand-in (nothing persists) to try the UI offline.

## Go where you last saw it (phase 5)
Ask *"turn to the cat"* / *"go to where you last saw my backpack"* in the chat (or `POST /api/nav/go {"label": "cat"}`).
The robot has a gyro heading and the direction of each object it has seen, but **no obstacle sensing and no position in the room**, so
it only does what is safe:
* **same room, seen during this visit** → turns in place to face it (closed loop on the gyro; bearings from an earlier visit are in another
  frame and are not trusted);
* **same room, direction unknown** → looks around: turns in steps, lets the camera check each view, stops and centres when it sees it
  (needs the camera running);
* **another room** → the route over the learned room map (`Bathroom -> Hallway -> Dining Room`) for **you** to drive; once it recognises
  the room, ask again.

It only moves while you have armed **Enable keyboard drive**. Any key, Space, Esc, losing focus, a lost gyro, a stalled turn or a timeout
stops it at once, and the 0.4 s dead-man still applies. The drive panel shows what it is doing, with a Cancel button.

## Keyboard drive
Click **Enable keyboard drive** (nothing moves until you arm it). Hold keys, release to stop:
`W/S` forward/back · `A/D` turn · `Q/E` strafe · `Space` stop · `H` stand/home · `1 2 3` speed · `Esc` disarm
(arrow keys and the on-screen pad work too). On the robot run `python3 robot/puppy_drive.py --brain http://<laptop-ip>:8000`;
add `--dry-run` first to see the commands without any movement.

Safety: the robot *pulls* commands (the brain never connects in); commands expire after 0.4 s of silence; closing the tab,
losing focus or a network drop stops and disarms; speed caps live on the robot; driving from another machine requires
`BRAIN_TOKEN`. While the robot moves, perception pauses and the first observation afterwards does **not** count misses, so
driving away never makes objects look "possibly gone".

## Memory rules that matter
* **One missed detection never deletes anything.** Objects carry `missCount`; only after `MISS_THRESHOLD`
  consecutive misses *at the same location* do they become `possibly_gone`, and answers say "not seen", never "removed".
* **Identity is conservative.** A detection matches a known object only on same label + compatible colour
  (+ zone/relation anchors). Doubt → new object. Wrong merges corrupt history; duplicates can be merged later.
* **Relations are temporal edges** (`ON {since, until, active}`); ON/INSIDE are exclusive per subject.
* **Every answer carries its tool trace** (UI: "Evidence"); NAMS stores the same trace as reasoning steps.
* **Privacy by default:** people are stored as anonymous `person` presence only (no description/colour/images) unless
  `PEOPLE_CONSENT=true`; frames are not saved unless `STORE_IMAGES=true`.

## Spatial memory

Place recognition, adjacency map, view-aware change detection, moved-object hypotheses and gyro heading: see [docs/SPATIAL.md](docs/SPATIAL.md).

## Layout
`src/puppy_mind/` brain (perception, graph, matching, agent, nams, server, cli) · `robot/` on-robot streamer ·
`eval/` scenarios + 20 questions · `tests/` · `docs/`.

## Milestone: robot test stage (2026-10-13)
Verified on the real PuppyPi by the owner: live camera over the robot's Wi-Fi, keyboard drive through the robot's rosbridge
(forward/back/turn, arm switch, speed levels). Known-good settings: stock gait restored on connect, turn capped at 25 deg/s,
commands sent on change only, no forward+turn mixing. Not yet verified on hardware: perception/memory/Q&A (needs a working
Aura login and an OpenAI key), `robot/puppy_drive.py` on the robot, the lighter 320x240 camera stream.
Run it: `uv run puppy-mind robot-test` while joined to the robot's HW... Wi-Fi.

## Aura verification (use a DISPOSABLE database only)
`seed` writes FAKE scripted data and `verify_graph.py` / `eval` expect a database containing only that data, so run them on an empty
throwaway database, never on the one holding real robot memory: `uv run puppy-mind reset --yes` erases everything this app stores.
If fake data did get into the real graph, remove just that with `uv run puppy-mind purge-demo` (dry run) then `--yes`: it deletes only
observations whose source is `scenario`/`demo` and objects seen only by them; real observations are kept.
Passed against a live AuraDB Professional instance on 2026-10-14 (17+3 graph checks, eval 20/20). The Demo replay button exists only on
throwaway in-RAM memory (`PUPPY_FAKE_MEMORY=1`); on the real graph the server refuses it (403).
Note: `.env` overrides shell/Windows `NEO4J_*` variables (a stale `NEO4J_URI` from another project had silently redirected the app).

## Question answering: verified on live Aura + OpenAI (2026-10-14)
`puppy-mind eval` (20 predefined questions over the scripted booth scene, each asked independently) scored 20/20 on two
consecutive runs after fixes (first run 15/20). Failures found by reading answers, not just scores: case-sensitive place names,
'until' misread as 'last seen', find_object not understanding 'red backpack', and an agent that picked inconsistent baselines for
'what changed' (now a deterministic default plus ready-made summary lines). Caveats: scripted scene, not the real camera; the
checker is substring-based; the model is non-deterministic, so re-run `eval` after any prompt/tool change.
Privacy: camera frames go to OpenAI only when frame analysis is on (`--real-memory` with Aura reachable, or `serve`). Driving
tests (`robot-test` without `--real-memory`) set `PERCEPTION=off`, so frames stay on the computer.

## Memory: Aura + NAMS
Aura is the authoritative, time-aware world model; NAMS holds entities/relationships, a narrative log, Q&A with reasoning traces and
semantic recall, and **writes into the same Aura database**, so the graph viewer shows both (dashed ring = stored by NAMS). Design,
measured NAMS behaviour (merge thresholds, key scope), setup commands: [docs/MEMORY.md](docs/MEMORY.md). Verified live on 2026-10-14.

## Status / honest limits
* Graph logic and matching are unit-tested (matching) and designed against Aura; the Aura-backed ingestion/queries have NOT yet been executed against a live instance (connection was failing when this was written).
* Single camera, no pose: spatial relations are qualitative and "location" is a name you set (`/api/location`).
* Eval questions use substring checks, a floor not a proof; extend with real captured scenes.
