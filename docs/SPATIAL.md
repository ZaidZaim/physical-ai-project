# Spatial memory: orientation first, tasks follow

The robot answers "where am I?" before "what is here?". The same chair in the kitchen and in the living room are two objects.

## Places
- Each frame the vision model also reports a room type. `PlaceResolver` compares the frame's objects (idf-weighted overlap) with each known place, with a room-type bonus and veto.
- Hysteresis: a switch needs `PLACE_CONFIRM_FRAMES` agreeing frames. Uncertain frames are held back and only written once the place is settled, so they never pollute another room.
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
- After upgrading, `nams-sync --rebuild` once so NAMS place entities use the new `name shortid` naming.

Verification: `uv run python eval/verify_spatial.py` (isolated namespace on Aura, cleans up) and `uv run pytest`.
