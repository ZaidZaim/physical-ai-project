# Robot walk-through: rooms, "turn to it", routes

A checklist for testing room recognition, the room map and Phase 5 across several rooms. About 20 minutes. Charge the robot first.

## 0. Start, recording a tour

```bash
uv run puppy-mind imu-check --robot-ip <robot-ip>         # PASS: the heading rises on a left turn (the robot turns ~2 s)
uv run puppy-mind robot-test --real-memory --record       # finds the robot; saves every analysed frame to tours/tour-<time>/
```

Open http://localhost:8000 and click **Start camera**. The drive panel should say *ROBOT CONNECTED* with a battery level.

## 1. Start-up: no false room

Point the robot at something odd first (the stairs, a wall). **Expect:** the place panel says *waiting* / *no place recognised yet*, and
no new room appears in the Places tab. When it sees a familiar view, it should recognise that room (e.g. **Hallway**) within a frame
or two.

## 2. Walk through 2-3 rooms

Drive (or carry) it Hallway → Dining Room → Living Room, pausing ~15 s in each, facing into the room.
**Expect:** the room name changes after about 2 frames in each room. No "Hallway 2", "Unnamed area" or "Garage". In the Places tab
the rooms are connected in the order you walked.

Hold the camera on the cat (or a person) for a few frames. **Expect:** no new room.

## 3. "Turn to it" in each room

In each room, after the robot has looked around for a bit, click **Enable keyboard drive** (keep a hand near it) and ask in the chat:

- *"turn to the sofa"* (Living Room), *"turn to the table"* (Dining Room), *"turn to the door"* (Hallway)

**Expect:** *Turning N degrees left/right to face the ...*, the turn stops within a few degrees, and the next frame shows it. Press
**Space** during a turn: it must stop at once.

Ask for something in the room that it has **not** seen this visit. **Expect:** *I'll look around for it*, a step-wise 360° look, and
then *Found it* or *didn't see it*.

## 4. "Go to it" across rooms

From the Living Room ask *"go to the cat"* or *"go to the toilet"*. **Expect:** a route such as *Living Room -> Dining Room -> Hallway
-> Bathroom* for **you** to drive, and **no movement**. Drive it there; once the room is recognised, ask again: it should turn.

## 5. Things only you can answer

- Are **Bedroom** and **Bedroom 2** really two rooms? The white bed was seen under both names. If they are one room, merge them in the
  Places tab.
- Edit `tours/real-questions.json` so every expected answer is true for your home, then:

```bash
uv run puppy-mind eval tours/real-questions.json          # writes nothing to memory
```

## 6. Afterwards: replay what it saw

```bash
uv run puppy-mind replay-tour tours/tour-<time> -v        # the recorded walk, through room recognition again, offline
```

Note anything that surprised you (a wrong room, a missed turn) with the frame number from `-v`. That replay is how a fix gets tested
against your real home before it goes back on the robot.
