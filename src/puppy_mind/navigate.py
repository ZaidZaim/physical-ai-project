"""Phase 5, "go where you last saw it", within what this robot can do safely.

What the robot has: a gyro heading relative to where it entered the current place (drifts; reset on every change of place), the
bearing of each object in that same frame, the learned map of which places connect (ADJACENT_TO), and turn-in-place control. What it
does NOT have: obstacle sensing, odometry or any position inside a room. So:

  * same place, bearing recorded during THIS visit  -> turn in place to face it (closed loop on the heading)
  * same place, no usable bearing                   -> look around: turn in steps, let the camera check each view, stop when it is seen
  * another place                                   -> plan the route over the place map and ask the person to drive; never walk alone
  * not known / no heading                          -> say so

Safety: nothing moves unless the person has armed driving (DriveState.begin_auto refuses otherwise). Any key they press, a stop,
disarming, a lost IMU, a stalled turn or a timeout ends the motion at once, and every command still expires after DEADMAN_S, so if this
thread dies the robot stops.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .drive import DriveState
from .pose import wrap180, wrap360

ALREADY_FACING_DEG = 12.0
# Spoken moves ("walk left", "come here", "turn around"): short, bounded, and only while the person has armed driving.
MOVE_SPEEDS = {"slow": 1, "normal": 2, "fast": 3, "sprint": 4}
MOVE_DIRECTIONS = ("forward", "back", "left", "right", "turn_left", "turn_right", "turn_around", "spin", "come_back")
OPPOSITE = {"forward": "back", "back": "forward", "left": "right", "right": "left"}
MAX_WALK_S = 4.0       # one spoken walk never lasts longer (at sprint, ~80 cm): say it again to go further
DEFAULT_WALK_S = 1.5
MAX_TURN_DEG = 360.0
TURN_DEG_PER_S = 25.0  # the robot's turn rate at speed level 3 (YAW_LIMIT): times a turn when there is no gyro heading
TURN_SCALE = {1: 0.35, 2: 0.65, 3: 1.0, 4: 1.0}  # the robot's LEVEL_SCALE for turning (a sprint does not turn faster)


@dataclass
class Move:
    """One spoken move. `seconds` for walks, `degrees` for turns (counter-clockwise positive, like the heading)."""
    direction: str
    seconds: float = DEFAULT_WALK_S
    degrees: float = 0.0
    speed: int = 2
    sidestep: str = "strafe"  # strafe: walk sideways; turn: turn 90 degrees, walk, turn back (firmware without lateral steps)

    @property
    def message(self) -> str:
        if self.direction in ("forward", "back", "left", "right"):
            how = "" if self.direction in ("forward", "back") or self.sidestep == "strafe" else " (turning, stepping, turning back)"
            return f"Walking {self.direction} for {self.seconds:.1f} s{how}."
        return f"Turning {abs(self.degrees):.0f} degrees {'left' if self.degrees > 0 else 'right'}."


def plan_move(direction: str, seconds: float | None = None, degrees: float | None = None, speed: str | int | None = None,
              sidestep: str = "strafe") -> Move | str:
    """A spoken request -> a bounded Move, or the reason it cannot be one."""
    d = (direction or "").strip().lower().replace(" ", "_")
    d = {"backward": "back", "backwards": "back", "around": "turn_around"}.get(d, d)
    if d not in MOVE_DIRECTIONS or d == "come_back":
        return f"unknown direction {direction!r}: use one of {', '.join(MOVE_DIRECTIONS)}"
    level = speed if isinstance(speed, int) else MOVE_SPEEDS.get(str(speed or "normal").lower(), 2)
    level = max(1, min(4, int(level)))
    if d in ("forward", "back", "left", "right"):
        secs = DEFAULT_WALK_S if not seconds or seconds <= 0 else min(MAX_WALK_S, max(0.3, float(seconds)))
        if d in ("left", "right") and sidestep == "strafe":
            level = min(level, 3)  # sprint is for walking straight only
        return Move(d, seconds=round(secs, 2), speed=level, sidestep=sidestep)
    deg = {"turn_around": 180.0, "spin": 360.0}.get(d) or (abs(float(degrees)) if degrees else 90.0)
    deg = min(MAX_TURN_DEG, max(5.0, deg))
    sign = -1.0 if d == "turn_right" else 1.0
    return Move(d, seconds=0.0, degrees=sign * deg, speed=min(level, 3))
CHECK_OFFSETS = (0.0, 40.0, -80.0)  # after a turn: where we turned, 40 degrees left of it, 40 degrees right of it


@dataclass
class Plan:
    kind: str  # turn | scan | here | route | unknown_route | no_heading | no_place | not_found
    message: str
    target: dict[str, Any] | None = None
    turn_deg: float | None = None  # counter-clockwise positive (left), like the heading
    route: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def pick_target(candidates: list[dict], here: str | None) -> dict | None:
    """Several objects can match ('the chair'): prefer one in this place, then one still present, then the most recently seen."""
    if not candidates:
        return None
    newest_first = sorted(candidates, key=lambda c: c.get("last_seen") or "", reverse=True)
    return sorted(newest_first, key=lambda c: (c.get("location") != here, c.get("status") == "possibly_gone"))[0]  # stable sort


def plan_go_to(target: dict | None, here: str | None, heading: float | None, bearing: float | None,
               route: list[str] | None, weak_route: bool = False) -> Plan:
    """Decide what 'go to it' means right now. `bearing` must come from THIS visit to `here` (the heading frame resets on entry)."""
    if target is None:
        return Plan("not_found", "I have no memory of anything like that.")
    name = f"{target.get('color') or ''} {target.get('label')}".strip()
    caveat = " (it may have been moved: it was not seen the last few times I looked)" if target.get("status") == "possibly_gone" else ""
    where = target.get("location")
    if where and where != here:
        if not route:
            return Plan("unknown_route", f"The {name} was last seen in {where}{caveat}. I don't know how {here or 'here'} connects to "
                                         f"{where} yet: drive me there and I can look for it.", target)
        hops = " -> ".join(route)
        doubt = " Part of this route was only seen once, so it may be wrong." if weak_route else ""
        return Plan("route", f"The {name} was last seen in {where}{caveat}. I can't walk between rooms on my own (I have no obstacle "
                             f"sensing), so please drive me: {hops}.{doubt} Once I recognise {where}, ask again and I'll turn to it.",
                    target, route=route)
    if heading is None:
        return Plan("no_heading", f"The {name} is here in {where}{caveat}, but I have no gyro heading right now, so I can't turn to it "
                                  "accurately. Turn me by hand and I'll tell you when it's in view.", target)
    if bearing is None:
        return Plan("scan", f"The {name} is here in {where}{caveat}, but I haven't seen it since I came in, so I'll look around for it.",
                    target)
    turn = round(wrap180(bearing - heading), 1)
    if abs(turn) <= ALREADY_FACING_DEG:
        return Plan("here", f"The {name} should be straight ahead of me{caveat}.", target, turn_deg=turn)
    side = "left" if turn > 0 else "right"
    return Plan("turn", f"Turning {abs(turn):.0f} degrees {side} to face the {name}{caveat}.", target, turn_deg=turn)


class Navigator:
    """Runs one turn/scan at a time in a background thread. Every step re-checks that it may still move."""

    def __init__(self, drive: DriveState, heading_fn: Callable[[], float | None], *, publish: Callable[[str, dict], None],
                 obs_seq_fn: Callable[[], int] | None = None, seen_fn: Callable[[dict, int], dict | None] | None = None,
                 fov_deg: float = 70.0, tick_s: float = 0.1, tolerance_deg: float = 6.0, settle_s: float = 2.0,
                 obs_wait_s: float = 15.0, stall_s: float = 4.0):
        self.drive, self.heading_fn, self.publish = drive, heading_fn, publish
        self.obs_seq_fn, self.seen_fn = obs_seq_fn, seen_fn
        self.fov, self.tick, self.tol = fov_deg, tick_s, tolerance_deg
        self.settle_s, self.obs_wait_s, self.stall_s = settle_s, obs_wait_s, stall_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.state: dict[str, Any] = {"active": False, "phase": "idle", "message": ""}
        self.last_walk: Move | None = None  # the last walk that finished: "come back" walks it the other way

    # ---- control
    def start(self, plan: Plan, verify: bool = True) -> dict[str, Any]:
        """verify: after a turn, check with the camera that the object is in view (and look a little either side if not)."""
        if plan.kind not in ("turn", "scan"):
            return {"started": False, "reason": plan.message}
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"started": False, "reason": "already moving: cancel first"}
            if not self.drive.begin_auto():
                return {"started": False, "reason": "driving is not armed. Click 'Enable keyboard drive' first (keep a hand near the "
                                                    "robot), then ask again."}
            self._cancel.clear()
            self._set(active=True, phase=plan.kind, message=plan.message, target=plan.target)
            self._thread = threading.Thread(target=self._run, args=(plan, verify), daemon=True, name="navigator")
            self._thread.start()
        return {"started": True, "message": plan.message}

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancel.set()
        self.drive.end_auto()
        if self.state.get("active"):
            self._set(active=False, phase="cancelled", message=reason)

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t:
            t.join(timeout)

    # ---- internals
    def _set(self, **kw) -> None:
        self.state = {**self.state, **kw}
        self.publish("nav", dict(self.state))

    def _abort_reason(self, need_heading: bool = True) -> str | None:
        if self._cancel.is_set():
            return "cancelled"
        if not self.drive.armed:
            return "driving was disarmed"
        if not self.drive.auto:
            return "you took over the controls"
        if need_heading and self.heading_fn() is None:
            return "lost the gyro heading"
        return None

    # ---- spoken moves
    def start_move(self, move: Move) -> dict[str, Any]:
        """Run one Move in the background. A new move replaces one still running ("no, go left"), never a turn toward an object."""
        with self._lock:
            busy = bool(self._thread and self._thread.is_alive())
        if busy:
            if self.state.get("phase") != "move":
                return {"started": False, "reason": "I'm busy turning toward something: say stop first"}
            self.cancel("replaced by a new move")
            self.wait(1.0)
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"started": False, "reason": "still stopping the last move: ask again"}
            if not self.drive.begin_auto():
                return {"started": False, "reason": "my legs are not enabled: someone has to click 'Enable keys' in the app first "
                                                    "(and keep a hand near me)"}
            self._cancel.clear()
            self._set(active=True, phase="move", message=move.message, target=None)
            self._thread = threading.Thread(target=self._run_move, args=(move,), daemon=True, name="mover")
            self._thread.start()
        return {"started": True, "message": move.message}

    def _run_move(self, move: Move) -> None:
        old_speed = self.drive.speed
        try:
            self.drive.set_speed(move.speed)
            d = move.direction
            if d in ("forward", "back"):
                ok, why = self._walk(1.0 if d == "forward" else -1.0, 0.0, move.seconds)
            elif d in ("left", "right") and move.sidestep == "strafe":
                ok, why = self._walk(0.0, 1.0 if d == "left" else -1.0, move.seconds)
            elif d in ("left", "right"):
                side = 90.0 if d == "left" else -90.0
                ok, why = self._turn_any(side, move.speed)
                if ok:
                    ok, why = self._walk(1.0, 0.0, move.seconds)
                if ok:
                    ok, why = self._turn_any(-side, move.speed)
            else:
                ok, why = self._turn_any(move.degrees, move.speed)
            if ok and d in OPPOSITE:
                self.last_walk = move
            self._finish(ok, move.message.replace("Walking", "Walked", 1).replace("Turning", "Turned", 1) if ok else why)
        except Exception as e:  # never leave the robot walking because of a bug
            self._finish(False, f"stopped: {type(e).__name__}: {e}"[:200])
        finally:
            self.drive.end_auto()
            self.drive.set_speed(old_speed)

    def _walk(self, x: float, y: float, seconds: float) -> tuple[bool, str]:
        """Timed walk: there is no odometry, so distance is speed x time. Every tick re-checks that it may still move."""
        end = time.monotonic() + min(seconds, MAX_WALK_S)
        while time.monotonic() < end:
            why = self._abort_reason(need_heading=False)
            if why:
                return False, why
            self.drive.command(x, y, 0.0, source="auto")
            time.sleep(self.tick)
        self.drive.command(0.0, 0.0, 0.0, source="auto")
        return True, ""

    def _turn_any(self, delta: float, level: int) -> tuple[bool, str]:
        """Turn on the gyro when there is a heading, else by time (rough: about TURN_DEG_PER_S at full turn speed)."""
        if self.heading_fn() is not None:
            return self._turn_by(delta)
        rate = TURN_DEG_PER_S * TURN_SCALE.get(level, 1.0)
        end = time.monotonic() + abs(delta) / rate
        while time.monotonic() < end:
            why = self._abort_reason(need_heading=False)
            if why:
                return False, why
            self.drive.command(0.0, 0.0, 1.0 if delta > 0 else -1.0, source="auto")
            time.sleep(self.tick)
        self.drive.command(0.0, 0.0, 0.0, source="auto")
        return True, ""

    def _run(self, plan: Plan, verify: bool = True) -> None:
        try:
            if plan.kind == "turn":
                ok, why = self._turn_by(plan.turn_deg or 0.0)
                if not ok:
                    self._finish(False, why)
                elif verify and self.obs_seq_fn and self.seen_fn:
                    self._check(plan.target or {})
                else:
                    self._finish(True, plan.message.replace("Turning", "Turned", 1))
            else:
                self._scan(plan.target or {})
        except Exception as e:  # never leave the robot turning because of a bug
            self._finish(False, f"stopped: {type(e).__name__}: {e}"[:200])
        finally:
            self.drive.end_auto()

    def _finish(self, ok: bool, message: str) -> None:
        self.drive.end_auto()
        if self.state.get("phase") != "cancelled":
            self._set(active=False, phase="done" if ok else "stopped", message=message)

    def _turn_by(self, delta: float) -> tuple[bool, str]:
        """Closed-loop turn on the gyro heading. Positive = left (counter-clockwise), like the heading."""
        h0 = self.heading_fn()
        if h0 is None:
            return False, "lost the gyro heading"
        target = wrap360(h0 + delta)
        deadline = time.monotonic() + 4.0 + abs(delta) / 5.0  # 5 deg/s is slower than the slowest speed level
        last_h, last_move = h0, time.monotonic()
        while True:
            why = self._abort_reason()
            if why:
                return False, why
            h = self.heading_fn()
            err = wrap180(target - h)
            if abs(err) <= self.tol:
                self.drive.command(0.0, 0.0, 0.0, source="auto")
                return True, ""
            now = time.monotonic()
            if now > deadline:
                return False, "the turn took too long"
            if abs(wrap180(h - last_h)) > 2.0:
                last_h, last_move = h, now
            elif now - last_move > self.stall_s:
                return False, "I'm not turning (stuck, or the floor is too soft?)"
            mag = min(1.0, max(0.35, abs(err) / 40.0))  # slow down near the target
            self.drive.command(0.0, 0.0, mag if err > 0 else -mag, source="auto")
            self._set(remaining_deg=round(err, 1))
            time.sleep(self.tick)

    def _wait_observation(self, after_seq: int) -> int | None:
        """Hold still until the camera has analysed a NEW frame (perception pauses while moving and SETTLE_S after)."""
        end = time.monotonic() + self.settle_s + self.obs_wait_s
        while time.monotonic() < end:
            if self._abort_reason():
                return None
            self.drive.command(0.0, 0.0, 0.0, source="auto")  # keep ownership while standing still
            seq = self.obs_seq_fn() if self.obs_seq_fn else after_seq
            if seq > after_seq:
                return seq
            time.sleep(self.tick)
        return None

    def _check(self, target: dict) -> None:
        """After a turn: the gyro says we face it; the camera decides. Seen on the robot: after ~500 degrees of fast turns the gyro had
        slipped ~50 degrees, so a bearing from several turns ago could be off. Look where we turned, then a little left, then right."""
        name = f"{target.get('color') or ''} {target.get('label')}".strip()
        seq = self.obs_seq_fn()
        for offset in CHECK_OFFSETS:
            if offset:
                self._set(phase="check", message=f"I don't see the {name} yet: looking a little {'left' if offset > 0 else 'right'}...")
                ok, why = self._turn_by(offset)
                if not ok:
                    return self._finish(False, why)
            else:
                self._set(phase="check", message=f"Checking with the camera that the {name} is in front of me...")
            new = self._wait_observation(seq)
            if new is None:
                why = self._abort_reason()
                if why:
                    return self._finish(False, why)
                return self._finish(True, f"Turned to the {name}, but the camera did not confirm it (no new frame).")
            seq = new
            hit = self.seen_fn(target, seq)
            if hit:
                nudge = {"left": self.fov / 3.0, "right": -self.fov / 3.0}.get(hit.get("zone", "center"), 0.0)
                if nudge:
                    ok, why = self._turn_by(nudge)
                    if not ok:
                        return self._finish(False, f"I saw the {name} but stopped while centring it: {why}")
                return self._finish(True, f"Facing the {name}: I can see it.")
        self._finish(False, f"I turned to where I last saw the {name}, but I can't see it now (it may have moved, or my sense of "
                            f"direction slipped). Ask me to look around for it.")

    def _scan(self, target: dict) -> None:
        if not (self.obs_seq_fn and self.seen_fn):
            return self._finish(False, "looking around needs the camera")
        views = max(4, -(-360 // round(self.fov * 0.75)))  # overlapping views all the way round
        step = 360.0 / views
        name = f"{target.get('color') or ''} {target.get('label')}".strip()
        seq = self.obs_seq_fn()
        for i in range(views):
            self._set(phase="scan", message=f"Looking for the {name} ({i * step:.0f} of 360 degrees)...")
            new = self._wait_observation(seq)
            if new is None:
                why = self._abort_reason()
                return self._finish(False, why or "the camera isn't sending frames: start the camera and ask again")
            seq = new
            hit = self.seen_fn(target, seq)
            if hit:
                nudge = {"left": self.fov / 3.0, "right": -self.fov / 3.0}.get(hit.get("zone", "center"), 0.0)
                if nudge:
                    ok, why = self._turn_by(nudge)
                    if not ok:
                        return self._finish(False, f"I saw the {name} but stopped while centring it: {why}")
                return self._finish(True, f"Found the {name}: it's in front of me now.")
            if i == views - 1:
                break
            ok, why = self._turn_by(step)
            if not ok:
                return self._finish(False, why)
        self._finish(False, f"I looked all the way round and didn't see the {name}. It may have been moved.")
