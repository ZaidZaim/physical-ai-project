# Spatial memory: orientation first, tasks follow

The robot answers "where am I?" before "what is here?". The same chair in the kitchen and in the living room are two objects.

## Places
- Each frame the vision model also reports a room type. `PlaceResolver` compares the frame's objects (idf-weighted overlap) with each known place, with a room-type bonus and veto.
- Hysteresis: a switch needs `PLACE_CONFIRM_FRAMES` agreeing frames. Uncertain frames are held back and only written once the place is settled, so they never pollute another room.
- A room never seen before needs `NEW_PLACE_CONFIRM_FRAMES` agreeing frames, and they must look like the **same** place (same room type, or
  shared objects): two unrelated odd frames do not make a room. Right after a start, a known room is recognised at once, but a new one
  still has to be confirmed (only an empty memory turns its first view into a room). Measured on the real tour: this removed a false
  "Garage" (one frame of the stairs right after a restart); 3 frames instead of 2 lost a short bathroom visit, so the default is 2.
- New places are created automatically (`Kitchen`, `Living Room`, ...). Rename, merge or pin one in the Places tab (or `/api/places/*`).
- Moving A to B records an undirected `ADJACENT_TO` edge with a count: a learned topological map.

## Objects and change
- Object identity is scoped to a place, so look-alikes in different rooms stay separate.
- A miss counts only for objects that could have been in view: heading known gives the view cone; otherwise high place overlap (`MIN_OVERLAP`); otherwise only objects whose anchor (e.g. the table a laptop is on) is visible.
- Three real misses make an object "possibly gone". One miss never does.
- A look-alike appearing elsewhere is stored as a `POSSIBLY_MOVED_FROM` hypothesis, never as a fact; the original stays "possibly gone".

## Orientation
- The IMU gyro yaw rate (`/ros_robot_controller/imu_raw`) is integrated into a heading relative to entering the place, reset on every place change.
- Left/centre/right thirds of the frame become bearings via `CAMERA_FOV_DEG`; the agent can say "to your left".

## Limits (honest)
- Heading drifts and is relative; no metric map, SLAM or LiDAR.
- The IMU topic and the 70 degree FOV are not yet verified on the real robot.
- Places mixed in older data (e.g. "Demo Area") have no split tool yet.

Verification: `uv run python eval/verify_spatial.py` (isolated namespace on Aura, cleans up), `uv run pytest`, and on real frames
`uv run puppy-mind export-tour` then `uv run puppy-mind replay-tour tours/<file>.jsonl -v`.

## Schema labels (shared database)
NAMS stores the entities it extracts under labels such as `Object`, `Location`, `Agent` and `Entity`. To keep the two apart, this app
owns `Item` (a thing in a place), `Place`, `Robot`, `Sighting`, `Session`, `Dialogue`, `Utterance`, `ReasoningStep` and `Lookup`.
Older databases are migrated automatically on start (nodes with our `obj-` / `plc-` ids move from `:Object` / `:Location`).
`(Robot)-[:CURRENTLY_IN {heading}]->(Place)` is the robot's position; `(Place)-[:ADJACENT_TO {count}]-(Place)` is the learned map.

## Recognition rules worth knowing
- One place per kind of room: a new view of a room type the robot already knows is filed under that room (a home has one kitchen).
  Use Merge in the Places tab if two rooms were created anyway.
- People and pets are never used to recognise a place and never count as "missing" (that includes "animal": the model sometimes calls
  the cat that). Vague labels (`object`, `unknown`, `item`, `thing`) are not evidence of a place either.
- Colour words from the vision model are treated as families (white/cream/beige, beige/brown, black/dark/grey) when matching objects.

## Two rooms of the same kind (a second bedroom)
Object lists cannot tell two bedrooms apart (both have a bed and a nightstand). When the resolver doubts (few known objects in a known
kind of room) the app compares the picture itself: it asks the vision model whether the current view is the SAME physical room as up to
three views it holds of that room (furniture, colours, walls, floor). A different room becomes its own place ("Bedroom 2") without any
button. At most 3 checks per visit, one model call each. The reference views are kept in RAM only and are lost on restart; until the
robot has seen a room again, the NEW ROOM button and the orange hint are the fallback.
