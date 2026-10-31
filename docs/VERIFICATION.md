# Verification log

What has been checked on real hardware and live services, and when. Newest first. The README says how to run the checks.

## 2026-10-31: real-scene evaluation, room recognition, NAMS backfill

- **Real tour:** 76 analysed frames from 10 robot sessions, exported from Aura and replayed through room recognition on throwaway
  memory. Room recognition as on `main` made a false "Garage" (88.5% of frames in the room recorded at the time). The new rules
  (a new room needs agreeing frames that look like the same place, also right after a start; "animal" counts as moving; vague
  labels are not evidence) made no false room and found every real one, 93%. Requiring 3 frames for a new room lost a short
  bathroom visit, so the default stays 2. ("Bedroom 2" needs the picture check, which a replay without images cannot run.)
- **Real-house Q&A:** 12 questions about the home against the live memory: 12/12, including "have you ever seen a car?" and "is there
  a fridge in the kitchen?" (neither seen). The node and relationship counts were identical before and after: evaluations no
  longer store their Q&A. One answer surfaced a data question: the white bed was sighted in both "Bedroom" and "Bedroom 2".
- **NAMS backfill:** `nams-sync` named the place entity "Hallway" instead of "Hallway af09af", which NAMS merged into an extracted
  entity, so no object got its place link. Fixed; Hallway backfilled: 35/35 objects linked (was 1/35).

## 2026-10-22: graph best practices, naming migration, phase 5 on the robot

- **Schema review** against Neo4j modeling best practices ([ONTOLOGY_REVIEW.md](ONTOLOGY_REVIEW.md)). Applied live on AuraDB
  (Enterprise 5.27): the Robot supernode edge `CAPTURED_BY` was removed (59 edges, robot links 168 -> 110), and existence/type
  constraints plus the `item_status` index were added. Existing data already satisfied them.
- **Naming migration** (camelCase keys, `itemId`/`placeId`, `TAKEN_AT`) ran on the live database: 72 items, 7 places, 62 sightings
  all kept, and no snake_case keys left. `eval/verify_spatial.py`: 30/30, later 32/32. It caught three regressions the unit tests
  could not see, plus one found on the robot (new sightings still written with old keys).
- **Declutter:** the write-only `(Robot)-[:OBSERVED]->(Item)` edges and 32 empty sessions were removed (robot links 122 -> 11). The nine
  views in [AURA_VIEWS.md](AURA_VIEWS.md) were checked live by `eval/check_views.py` (7-71 nodes each, instead of 675).
- **Phase 5 on the PuppyPi:**
  - `imu-check`: PASS (+34 deg for a left turn).
  - Turn to a remembered object: 131 deg left to the brown door, ending 5.3 deg from its bearing (tolerance 6 deg), with the door
    centred in the next frame.
  - 360 deg look-around for a moved shoe: completed and reported it unseen.
  - Found on the robot and fixed: a gyro bias of about -0.42 deg/s (now learned while still, about -0.01 deg/s), and the scan
    waiting for stored frames while a still robot's frames are skipped as near-duplicates.
- **NAMS:** the PuppyPi ontology is active ("PuppyPi Physical World"; it had been rejected before because `domain.id`/`name` were missing).
  Place merges now reach NAMS. After a restart the robot/place entities are reused (before, NAMS's "merged" answer left objects
  unlinked).
- A false "Garage" place, from a frame of the Hallway stairs, was merged back into Hallway in Aura and removed from NAMS.

## 2026-10-14: memory and question answering on live Aura + OpenAI

- `verify_graph.py` and `eval` against a live AuraDB Professional instance: 17+3 graph checks, and eval 20/20 on two consecutive
  runs after fixes (first run 15/20). Failures were found by reading the answers, not just the scores: case-sensitive place names,
  `until` misread as "last seen", `find_object` not understanding "red backpack", and inconsistent baselines for "what changed"
  (now a deterministic default plus ready-made summary lines). Caveats: a scripted scene, not the real camera; a substring-based
  checker; a non-deterministic model, so re-run `eval` after any prompt or tool change.
- NAMS live: the real robot memory was mirrored (13 entities), with measured behaviour documented in [MEMORY.md](MEMORY.md).
- `.env` overrides shell/Windows `NEO4J_*` variables: a stale `NEO4J_URI` from another project had silently redirected the app.

## 2026-10-13: robot test stage

Verified on the real PuppyPi by the owner: live camera over the robot's Wi-Fi, and keyboard drive through the robot's rosbridge
(forward/back/turn, arm switch, speed levels). Known-good settings: the stock gait restored on connect, turns capped at 25 deg/s,
commands sent on change only, and no forward+turn mixing.
