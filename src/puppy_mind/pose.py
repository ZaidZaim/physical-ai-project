"""Orientation: which way is the robot facing, relative to where it entered the current place?

PuppyPi has an IMU but no wheel odometry. We integrate the gyro's yaw rate (topic /ros_robot_controller/imu_raw, field
angular_velocity.z, verified in Hiwonder's source) into a heading in degrees, counter-clockwise positive like ROS. It is RELATIVE
(0 = the way the robot faced when it entered the place) and it DRIFTS, so it is reset on every change of place and treated as
"good for the length of a look around a room", never as a map. When no IMU data arrives it reports None and everything that
uses it falls back to the view-overlap rules.

Bearings: an object seen in the left/centre/right third of the frame is at heading + (+fov/3, 0, -fov/3). Left is positive
because turning left raises the heading.
"""
from __future__ import annotations

import math
import threading
import time


def wrap180(a: float) -> float:
    """Angle in (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def wrap360(a: float) -> float:
    return a % 360.0


def zone_offset(zone: str, fov_deg: float) -> float:
    return {"left": fov_deg / 3.0, "right": -fov_deg / 3.0, "center": 0.0}.get(zone, 0.0)


def bearing_of(zone: str, heading: float | None, fov_deg: float) -> float | None:
    return None if heading is None else round(wrap360(heading + zone_offset(zone, fov_deg)), 1)


def in_view(bearing: float | None, heading: float | None, fov_deg: float, margin_deg: float = 10.0) -> bool | None:
    """Is a remembered bearing inside the current view cone? None when we cannot tell."""
    if bearing is None or heading is None:
        return None
    return abs(wrap180(bearing - heading)) <= fov_deg / 2.0 + margin_deg


def direction_words(rel_deg: float) -> str:
    """Relative angle (object bearing minus our heading) as words, for answers like 'to your left'."""
    r = wrap180(rel_deg)
    if abs(r) <= 25:
        return "straight ahead"
    if abs(r) >= 155:
        return "directly behind you"
    side = "left" if r > 0 else "right"
    return f"to your {side}" if abs(r) <= 110 else f"behind you to the {side}"


class HeadingTracker:
    def __init__(self, stale_s: float = 3.0, deadband_rad_s: float = 0.004):
        self.stale_s, self.deadband = stale_s, deadband_rad_s
        self._yaw = 0.0  # radians since reset
        self._t: float | None = None
        self._last_msg: float | None = None
        self._lock = threading.Lock()

    def feed(self, wz: float, t: float | None = None) -> None:
        """One gyro sample: yaw rate in rad/s. t is a monotonic timestamp (defaults to now)."""
        now = time.monotonic() if t is None else t
        with self._lock:
            if self._t is not None:
                dt = now - self._t
                if 0.0 < dt < 0.5 and abs(wz) > self.deadband:  # the deadband stops a still gyro from creeping
                    self._yaw += wz * dt
            self._t = now
            self._last_msg = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._yaw = 0.0

    def value(self) -> float | None:
        """Heading in degrees [0, 360), or None when there is no recent IMU data."""
        with self._lock:
            if self._last_msg is None or time.monotonic() - self._last_msg > self.stale_s:
                return None
            return round(wrap360(math.degrees(self._yaw)), 1)
