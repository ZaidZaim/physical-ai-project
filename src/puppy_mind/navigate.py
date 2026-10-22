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
               route: list[str] | None) -> Plan:
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
        return Plan("route", f"The {name} was last seen in {where}{caveat}. I can't walk between rooms on my own (I have no obstacle "
                             f"sensing), so please drive me: {hops}. Once I recognise {where}, ask again and I'll turn to it.",
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

    # ---- control
    def start(self, plan: Plan) -> dict[str, Any]:
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
            self._thread = threading.Thread(target=self._run, args=(plan,), daemon=True, name="navigator")
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

    def _abort_reason(self) -> str | None:
        if self._cancel.is_set():
            return "cancelled"
        if not self.drive.armed:
            return "driving was disarmed"
        if not self.drive.auto:
            return "you took over the controls"
        if self.heading_fn() is None:
            return "lost the gyro heading"
        return None

    def _run(self, plan: Plan) -> None:
        try:
            if plan.kind == "turn":
                ok, why = self._turn_by(plan.turn_deg or 0.0)
                self._finish(ok, plan.message.replace("Turning", "Turned", 1) if ok else why)
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
