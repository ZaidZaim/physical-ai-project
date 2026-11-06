"""Drive state: the single source of truth for "what should the robot be doing right now".

Safety model (every layer fails toward stopping):
  * Disarmed by default; the UI must explicitly arm. Disconnecting the controller disarms.
  * Commands are normalised (-1..1), clamped here, and EXPIRE after DEADMAN_S: a controller that
    stops talking (tab closed, network drop, key stuck) yields zeros.
  * The robot polls this state; if it cannot reach the brain it stops on its own (puppy_drive.py).
  * Absolute speed limits live on the robot, not here.
  * Automatic motion (navigate.py) only ever happens while the UI has armed, and the person always wins: any key they press or a
    stop hands control straight back to them. Their 10 Hz zero heartbeat does not count as a key, so it cannot clobber the turn.
  * Booth mode (opt-in, separate from arming) lets the robot look slightly up at people and play a short trick when it greets
    someone. A trick never starts while the robot is moving or navigating, is rate-limited (servo heat, battery), and blocks all
    velocity commands while it runs, so a key press cannot make a dog that is up on its hind legs walk.
"""
from __future__ import annotations

import math
import threading
import time

DEADMAN_S = 0.4
SETTLE_S = 2.0  # after motion stops, the camera is still settling / viewpoint just changed
# Hiwonder's stock action groups (ros2 branch, large_models/action.py), run by /puppy_control/runActionGroup as "<name>.d6ac".
# The friendly, on-the-spot ones (a visitor may ask for these too). All are action groups on the robot.
TRICKS = ("boxing", "wave", "shake_hands", "bow", "nod", "sit", "stand", "lie_down", "stretch", "shake_head")
# These need room around the robot: only from the Booth tab's buttons (the person pressing sees the space), never on request.
BIG_TRICKS = ("push-up", "moonwalk")
# Postures that HOLD after their action group: the dog stays sitting / lying ("take a nap") until it is told to stand, does another
# trick, or is asked to walk (then it first stands up, RISE_S, and walks after). Every other trick ends back in the standing pose.
HOLD_POSTURES = ("sit", "lie_down")
RISE_S = 1.0
# Booth mode tilts the body nose-up to look at people. At the robot it could not walk properly tilted (the weight sits on the back
# legs), so a walk or turn first levels the body (LEVEL_S, the pose takes 0.5 s) and the tilt comes back RETILT_S after it stops.
LEVEL_S = 0.6
RETILT_S = 1.5
# Tricks (Hiwonder's recorded action groups) start from its level standing pose. At the robot, shake_hands from the 25 degree
# conversation tilt, right after walking, tipped the dog over. So a trick first goes IDLE: stop, stand level, settle IDLE_S; the
# tilt comes back RETILT_S after it ends.
IDLE_S = 1.0
MANUAL_GAP_S = 2.0  # a trick from the buttons waits only for the previous one to finish (plus this), not GESTURE_COOLDOWN_S
MAX_SPEED_LEVEL = 4  # 4 = sprint: forward/back only goes faster (DRIVE_SPRINT_X); turning stays at level 3
MAX_LOOK_UP_DEG = 30.0  # Hiwonder's pose node rejects |pitch| > 31 deg; its own "look down" pose is -15 deg


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


class DriveState:
    def __init__(self, allow_combine: bool = False) -> None:
        # Hiwonder's own teleop never mixes forward and turn, and mixing made the dog flex and nearly flip,
        # so by default turning takes priority over walking (turn in place, then resume).
        self.allow_combine = allow_combine
        self._lock = threading.Lock()
        self.armed = False
        self.speed = 2  # 1 slow, 2 medium, 3 fast, 4 sprint (scaled on the robot)
        self._x = self._y = self._yaw = 0.0
        self._stamp = 0.0
        self.home_seq = 0
        self.robot_seen = 0.0
        self.robot_applied: dict[str, float] = {}
        self._moving_until = 0.0
        self._auto = False  # the navigator is driving (only while armed; any user input ends it)
        # what the robot has done, so the brain can tell "only turned on the spot" from "walked somewhere"
        self.translations = 0  # commands that moved it forward/back/sideways
        self.rotations = 0     # commands that turned it
        self.last_rotation = 0.0
        # booth mode: look up at people, greet with a trick
        self.booth = False
        self.look_up_deg = 10.0
        self.action = ""
        self.action_seq = 0
        self._action_until = 0.0
        self._action_times: list[float] = []
        self.action_busy_s = 8.0  # how long a trick keeps the robot (and the camera) busy; no completion signal on every path
        self.gesture_cooldown_s = 60.0
        self.max_gestures_per_hour = 30
        self.posture = "stand"  # stand | sit | lie_down (a holding trick sets it; standing up or walking clears it)
        self._rise_until = 0.0
        self._walk_until = 0.0  # walking or turning (or just stopped): hold the body level, not tilted

    # ---- booth mode
    def set_booth(self, on: bool, look_up_deg: float | None = None) -> None:
        with self._lock:
            self.booth = bool(on)
            if look_up_deg is not None:
                self.look_up_deg = max(0.0, min(MAX_LOOK_UP_DEG, float(look_up_deg)))

    @property
    def pitch_rad(self) -> float:
        """Body pitch the robot should hold: slightly nose-up in booth mode (to see faces and badges), level otherwise."""
        return math.radians(self.look_up_deg) if self.booth else 0.0

    def request_action(self, name: str, manual: bool = False, asked: bool = False) -> dict:
        """Ask the robot to play a trick. Returns {ok, reason}. Never while moving, navigating or within the rate limits.
        `manual`: pressed in the Booth tab, so the big tricks are allowed and the gap is MANUAL_GAP_S instead of the cooldown that
        spaces out automatic tricks (the hourly limit still holds: servo heat and battery).
        `asked`: a visitor asked for it in the conversation: the same short gap (at the booth the 60 s cooldown refused the second
        trick a visitor asked for), and the big tricks too (push-up, moonwalk: wanted at the booth; keep space around the robot)."""
        quick = manual or asked
        with self._lock:
            now = time.monotonic()
            self._action_times = [t for t in self._action_times if now - t < 3600]
            reason = (
                "unknown trick" if name not in TRICKS + (BIG_TRICKS if quick else ()) else
                "booth mode is off" if not self.booth else
                "a trick is already running" if now < self._action_until else
                "the last trick is still settling" if quick and now < self._action_until + MANUAL_GAP_S else
                "the robot is moving" if now < self._moving_until or self._auto else
                "tricks are rate-limited" if not quick and self._action_times and now - self._action_times[-1] < self.gesture_cooldown_s else
                "hourly trick limit reached" if len(self._action_times) >= self.max_gestures_per_hour else ""
            )
            if reason:
                return {"ok": False, "reason": reason}
            self.action, self.action_seq = name, self.action_seq + 1
            self.posture = name if name in HOLD_POSTURES else "stand"
            self._action_until = now + self.action_busy_s + IDLE_S
            self._moving_until = self._action_until + SETTLE_S  # perception waits: the view is bouncing
            self._walk_until = self._action_until + RETILT_S  # level for the trick and a moment after it
            self._x = self._y = self._yaw = 0.0
            self._action_times.append(now)
            return {"ok": True, "reason": "", "action": name}

    def action_done(self) -> None:
        """The robot reported the trick finished (rosbridge waits for the service reply)."""
        with self._lock:
            now = time.monotonic()
            if now < self._action_until:
                self._action_until = now
                self._moving_until = now + SETTLE_S
                self._walk_until = now + RETILT_S

    @property
    def in_action(self) -> bool:
        with self._lock:
            return time.monotonic() < self._action_until

    # ---- controller side
    def arm(self, on: bool) -> None:
        with self._lock:
            self.armed = bool(on)
            if not on:
                self._x = self._y = self._yaw = 0.0
                self._auto = False

    def set_speed(self, level: int) -> None:
        with self._lock:
            self.speed = int(max(1, min(MAX_SPEED_LEVEL, level)))

    def command(self, x: float, y: float, yaw: float, source: str = "user") -> None:
        with self._lock:
            if not self.armed:
                return
            if time.monotonic() < self._action_until:  # mid-trick (maybe up on its hind legs): no walking
                return
            if source == "auto" and not self._auto:  # the navigator was cancelled: its late commands must not move the robot
                return
            if source == "user" and self._auto:
                if not (x or y or yaw):  # the UI's heartbeat while no key is held: let the navigator keep turning
                    return
                self._auto = False  # a key press takes over
            now = time.monotonic()
            if self.posture != "stand" and (x or y or yaw):  # sitting or napping: stand up first, then walk
                self.posture = "stand"
                self._rise_until = now + RISE_S
            if (x or y or yaw) and self.booth and self.look_up_deg > 0 and now >= self._walk_until:  # tilted: level first
                self._rise_until = max(self._rise_until, now + LEVEL_S)
            if x or y or yaw or self._auto:
                self._walk_until = now + DEADMAN_S + RETILT_S
            self._x, self._y, self._yaw = _clamp(x), _clamp(y), _clamp(yaw)
            self._stamp = time.monotonic()
            if self._x or self._y or self._yaw:
                self._moving_until = self._stamp + DEADMAN_S + SETTLE_S
            if self._x or self._y:
                self.translations += 1
            if self._yaw:
                self.rotations += 1
                self.last_rotation = self._stamp

    def stop(self) -> None:
        with self._lock:
            self._x = self._y = self._yaw = 0.0
            self._stamp = time.monotonic()
            self._auto = False

    def begin_auto(self) -> bool:
        """The navigator asks to drive. Only possible while armed (the person has said motion is OK)."""
        with self._lock:
            if not self.armed:
                return False
            self._auto = True
            return True

    def end_auto(self) -> None:
        with self._lock:
            if self._auto:
                self._x = self._y = self._yaw = 0.0
                self._stamp = time.monotonic()
            self._auto = False

    @property
    def auto(self) -> bool:
        with self._lock:
            return self._auto

    def go_home(self) -> None:
        with self._lock:
            if self.armed:
                self.home_seq += 1
                self.posture = "stand"
                self._moving_until = time.monotonic() + SETTLE_S + 2.0

    # ---- robot side
    def current(self) -> dict:
        with self._lock:
            busy = time.monotonic() < self._action_until
            rising = time.monotonic() < self._rise_until  # getting up from sitting / lying: no walking yet
            fresh = self.armed and not busy and not rising and (time.monotonic() - self._stamp) <= DEADMAN_S
            x, y, yaw = (self._x, self._y, self._yaw) if fresh else (0.0, 0.0, 0.0)
            if not self.allow_combine and x and yaw:
                x = 0.0
            return {"x": x, "y": y, "yaw": yaw, "speed": self.speed, "armed": self.armed,
                    "home_seq": self.home_seq, "action": self.action, "action_seq": self.action_seq,
                    "pitch": self._pitch_now(), "busy": busy, "posture": self.posture}

    def _pitch_now(self) -> float:
        """The tilt to hold right now (lock held): booth mode's nose-up look, except while walking or turning (level)."""
        if not self.booth or time.monotonic() < self._walk_until or self._auto:
            return 0.0
        return math.radians(self.look_up_deg)

    def robot_report(self, applied: dict[str, float]) -> None:
        with self._lock:
            self.robot_seen = time.monotonic()
            self.robot_applied = applied

    # ---- queries
    def turned_in_place_since(self, snapshot: tuple[int, int] | None, within_s: float = 60.0) -> bool:
        """Since `snapshot` (translations, rotations): the robot was turned, recently, and never driven anywhere."""
        if snapshot is None:
            return False
        with self._lock:
            return (self.rotations > snapshot[1] and self.translations == snapshot[0]
                    and time.monotonic() - self.last_rotation <= within_s)

    def motion_snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.translations, self.rotations

    @property
    def moving(self) -> bool:
        """True while commanded to move and for SETTLE_S afterwards (perception should wait)."""
        with self._lock:
            return time.monotonic() < self._moving_until

    def status(self) -> dict:
        c = self.current()
        with self._lock:
            online = (time.monotonic() - self.robot_seen) < 2.0 if self.robot_seen else False
            return {"armed": c["armed"], "auto": self._auto, "speed": self.speed, "cmd": {k: c[k] for k in ("x", "y", "yaw")},
                    "moving": time.monotonic() < self._moving_until, "robot_online": online,
                    "robot_applied": self.robot_applied, "booth": self.booth, "look_up_deg": self.look_up_deg,
                    "in_action": time.monotonic() < self._action_until, "action": self.action, "posture": self.posture}
