"""Drive the PuppyPi from the laptop through the robot's rosbridge websocket (default ws://<robot>:9090).

No software needs to be installed or run on the robot: the stock image already serves rosbridge for the
phone app. This client speaks the standard rosbridge v2 JSON protocol and the same Hiwonder topics as
robot/puppy_drive.py (/puppy_control/velocity, pose, gait, set_mark_time, go_home).

SAFETY - read this: unlike robot/puppy_drive.py, the dead-man here lives on the LAPTOP. This driver sends
zero velocity whenever the command expires, the controller disarms/disconnects, or the process exits. But if
the laptop crashes or Wi-Fi drops mid-walk, nothing on the robot knows to stop. Keep the speed low and a hand
on the robot / power switch. Prefer robot/puppy_drive.py when you can run it.
"""
from __future__ import annotations

import atexit
import json
import threading
import time
import uuid

from .drive import DriveState

LEVEL_SCALE = {1: 0.35, 2: 0.65, 3: 1.0}
# Hiwonder's STOCK walking style (puppy.py GaitConfigFast + Stand pose). Fast, short steps are stable; the slower long-stride
# "Trot" from their demo tipped the robot over when turning. Sending these also undoes any gait set earlier.
GAIT = dict(overlap_time=0.1, swing_time=0.15, clearance_time=0.0, z_clearance=5.0, x_shift=-0.5)
YAW_LIMIT = 0.436  # rad/s = 25 deg/s: the fastest turn Hiwonder's own teleop uses; above it the robot can tip over
TYPE_CANDIDATES = ("puppy_control/Velocity", "puppy_control_msgs/msg/Velocity")  # ROS1, ROS2
ZERO = {"x": 0.0, "y": 0.0, "yaw": 0.0}


class RosbridgeDriver:
    def __init__(self, url: str, drive: DriveState, *, max_x=12.0, max_y=0.0, max_yaw=YAW_LIMIT, invert_yaw=False):
        self.url, self.drive = url, drive
        self.max_x, self.max_y, self.invert_yaw = max_x, max_y, invert_yaw
        self.max_yaw = min(abs(max_yaw), YAW_LIMIT)
        self.connected = False
        self.error = ""
        self.vel_type = ""
        self._stop = threading.Event()
        self._ws = None
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True, name="rosbridge")
        atexit.register(self._send_zero_quiet)

    # ---- lifecycle
    def start(self) -> "RosbridgeDriver":
        self._t.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._send_zero_quiet()
        self._t.join(timeout=2)

    def status(self) -> dict:
        return {"url": self.url, "connected": self.connected, "error": self.error, "velocity_type": self.vel_type}

    # ---- protocol helpers
    def _send(self, obj: dict) -> None:
        with self._lock:
            if self._ws is not None:
                self._ws.send(json.dumps(obj))

    def _publish(self, topic: str, msg: dict) -> None:
        self._send({"op": "publish", "topic": topic, "msg": msg})

    def _call(self, service: str, args: dict | None = None, wait: float = 0.0) -> dict | None:
        cid = "c" + uuid.uuid4().hex[:8]
        self._send({"op": "call_service", "id": cid, "service": service, "args": args or {}})
        if not wait:
            return None
        end = time.monotonic() + wait
        while time.monotonic() < end:
            try:
                m = json.loads(self._ws.recv(timeout=max(0.01, end - time.monotonic())))
            except TimeoutError:
                break
            if m.get("op") == "service_response" and m.get("id") == cid:
                return m
        return None

    def _send_zero_quiet(self) -> None:
        try:
            self._publish("/puppy_control/velocity", {"x": 0.0, "y": 0.0, "yaw_rate": 0.0})
        except Exception:
            pass

    # ---- main loop
    def _run(self) -> None:
        from websockets.sync.client import connect

        while not self._stop.is_set():
            try:
                with connect(self.url, open_timeout=4, close_timeout=1, max_size=2**20) as ws:
                    self._ws = ws
                    self._setup()
                    self.connected, self.error = True, ""
                    self._loop()
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"[:200]
            finally:
                self.connected, self._ws = False, None
            self._stop.wait(1.5)

    def _setup(self) -> None:
        # Ask rosapi (shipped with rosbridge) for the exact message type; fall back to ROS1 then ROS2 names.
        r = self._call("/rosapi/topic_type", {"topic": "/puppy_control/velocity"}, wait=2.0)
        t = ((r or {}).get("values") or {}).get("type") or ""
        self.vel_type = t if t.endswith("Velocity") else TYPE_CANDIDATES[0]
        prefix = self.vel_type[: -len("Velocity")]
        for topic, typ in (("/puppy_control/velocity", "Velocity"), ("/puppy_control/pose", "Pose"),
                           ("/puppy_control/gait", "Gait")):
            self._send({"op": "advertise", "topic": topic, "type": prefix + typ})
        time.sleep(0.2)
        self._publish("/puppy_control/pose", dict(stance_x=0.0, stance_y=0.0, x_shift=GAIT["x_shift"], height=-10.0,
                                                  roll=0.0, pitch=0.0, yaw=0.0, run_time=500))
        time.sleep(0.2)
        self._publish("/puppy_control/gait", {k: v for k, v in GAIT.items() if k != "x_shift"})
        self._call("/puppy_control/set_mark_time", {"data": False})  # leave mark-time so it can walk

    def _loop(self) -> None:
        last, last_pub, last_home = None, 0.0, self.drive.current()["home_seq"]
        zero_again = False
        while not self._stop.is_set():
            t0 = time.monotonic()
            c = self.drive.current()
            s = LEVEL_SCALE.get(int(c["speed"]), LEVEL_SCALE[1])
            want = {
                "x": c["x"] * self.max_x * s,
                "y": c["y"] * self.max_y * s,
                "yaw": c["yaw"] * self.max_yaw * s * (-1 if self.invert_yaw else 1),
            }
            if not c["armed"]:
                want = dict(ZERO)
            if c["home_seq"] != last_home:
                last_home = c["home_seq"]
                self._publish("/puppy_control/velocity", {"x": 0.0, "y": 0.0, "yaw_rate": 0.0})
                self._call("/puppy_control/go_home")
                want = dict(ZERO)
            moving = any(want.values())
            # Publish ONLY on change: the robot re-applies its stance on every velocity message, so a keep-alive
            # stream makes the dog flex. A stop is sent twice (now and 0.3 s later) in case one is lost.
            if want != last:
                self._publish("/puppy_control/velocity", {"x": want["x"], "y": want["y"], "yaw_rate": want["yaw"]})
                last, last_pub, zero_again = dict(want), t0, not moving
            elif zero_again and t0 - last_pub > 0.3:
                self._publish("/puppy_control/velocity", {"x": 0.0, "y": 0.0, "yaw_rate": 0.0})
                zero_again = False
            self.drive.robot_report(want)  # makes the UI show "robot online"
            try:  # drain replies/errors so a closed socket is noticed
                while True:
                    m = json.loads(self._ws.recv(timeout=0))
                    if m.get("op") == "status" and m.get("level") == "error":
                        self.error = str(m.get("msg"))[:200]
            except TimeoutError:
                pass
            time.sleep(max(0.0, 0.1 - (time.monotonic() - t0)))
