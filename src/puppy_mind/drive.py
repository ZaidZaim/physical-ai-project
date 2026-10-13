"""Drive state: the single source of truth for "what should the robot be doing right now".

Safety model (every layer fails toward stopping):
  * Disarmed by default; the UI must explicitly arm. Disconnecting the controller disarms.
  * Commands are normalised (-1..1), clamped here, and EXPIRE after DEADMAN_S: a controller that
    stops talking (tab closed, network drop, key stuck) yields zeros.
  * The robot polls this state; if it cannot reach the brain it stops on its own (puppy_drive.py).
  * Absolute speed limits live on the robot, not here.
"""
from __future__ import annotations

import threading
import time

DEADMAN_S = 0.4
SETTLE_S = 2.0  # after motion stops, the camera is still settling / viewpoint just changed


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


class DriveState:
    def __init__(self, allow_combine: bool = False) -> None:
        # Hiwonder's own teleop never mixes forward and turn, and mixing made the dog flex and nearly flip,
        # so by default turning takes priority over walking (turn in place, then resume).
        self.allow_combine = allow_combine
        self._lock = threading.Lock()
        self.armed = False
        self.speed = 2  # 1 slow, 2 medium, 3 fast (scaled on the robot)
        self._x = self._y = self._yaw = 0.0
        self._stamp = 0.0
        self.home_seq = 0
        self.robot_seen = 0.0
        self.robot_applied: dict[str, float] = {}
        self._moving_until = 0.0

    # ---- controller side
    def arm(self, on: bool) -> None:
        with self._lock:
            self.armed = bool(on)
            if not on:
                self._x = self._y = self._yaw = 0.0

    def set_speed(self, level: int) -> None:
        with self._lock:
            self.speed = int(max(1, min(3, level)))

    def command(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            if not self.armed:
                return
            self._x, self._y, self._yaw = _clamp(x), _clamp(y), _clamp(yaw)
            self._stamp = time.monotonic()
            if self._x or self._y or self._yaw:
                self._moving_until = self._stamp + DEADMAN_S + SETTLE_S

    def stop(self) -> None:
        with self._lock:
            self._x = self._y = self._yaw = 0.0
            self._stamp = time.monotonic()

    def go_home(self) -> None:
        with self._lock:
            if self.armed:
                self.home_seq += 1
                self._moving_until = time.monotonic() + SETTLE_S + 2.0

    # ---- robot side
    def current(self) -> dict:
        with self._lock:
            fresh = self.armed and (time.monotonic() - self._stamp) <= DEADMAN_S
            x, y, yaw = (self._x, self._y, self._yaw) if fresh else (0.0, 0.0, 0.0)
            if not self.allow_combine and x and yaw:
                x = 0.0
            return {"x": x, "y": y, "yaw": yaw, "speed": self.speed, "armed": self.armed,
                    "home_seq": self.home_seq}

    def robot_report(self, applied: dict[str, float]) -> None:
        with self._lock:
            self.robot_seen = time.monotonic()
            self.robot_applied = applied

    # ---- queries
    @property
    def moving(self) -> bool:
        """True while commanded to move and for SETTLE_S afterwards (perception should wait)."""
        with self._lock:
            return time.monotonic() < self._moving_until

    def status(self) -> dict:
        c = self.current()
        with self._lock:
            online = (time.monotonic() - self.robot_seen) < 2.0 if self.robot_seen else False
            return {"armed": c["armed"], "speed": self.speed, "cmd": {k: c[k] for k in ("x", "y", "yaw")},
                    "moving": time.monotonic() < self._moving_until, "robot_online": online,
                    "robot_applied": self.robot_applied}
