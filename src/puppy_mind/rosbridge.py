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
import collections
import json
import math
import os
import threading
import time
import uuid

from .drive import HOLD_POSTURES, IDLE_S, DriveState

LEVEL_SCALE = {1: 0.35, 2: 0.65, 3: 1.0, 4: 1.0}
# Speeds in cm/s. Level 3 walks at DRIVE_MAX_X; level 4 ("sprint") walks forward/back at DRIVE_SPRINT_X; turning stays at level 3
# (a fast turn tips the dog over). Read from the robot's own puppy.py (ROS1 image), VelocityFun:
#   * it ACCEPTS a command only if |x| <= 35 cm/s, y == 0 exactly, and |yaw_rate| <= 51 deg/s; anything else is DROPPED WHOLE, and the
#     robot carries on with the previous command. So we never send y and never exceed FIRMWARE_MAX_X.
#   * the PuppyPi has 8 servos (dof='8': hip swing and knee per leg, no sideways hip joint): it cannot step sideways at all.
FIRMWARE_MAX_X = 35.0

# Hiwonder's STOCK walking style (puppy.py GaitConfigFast + Stand pose). Fast, short steps are stable; the slower long-stride
# "Trot" from their demo tipped the robot over when turning. Sending these also undoes any gait set earlier.
GAIT = dict(overlap_time=0.1, swing_time=0.15, clearance_time=0.0, z_clearance=5.0, x_shift=-0.5)
YAW_LIMIT = 0.436  # rad/s = 25 deg/s: the fastest turn Hiwonder's own teleop uses; above it the robot can tip over
TYPE_CANDIDATES = ("puppy_control/Velocity", "puppy_control_msgs/msg/Velocity")  # ROS1, ROS2
ZERO = {"x": 0.0, "y": 0.0, "yaw": 0.0}


def _env(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


def velocity(c: dict, max_x: float, sprint_x: float, max_yaw: float, invert_yaw: bool = False) -> dict:
    """Normalised drive command -> what the robot is sent (cm/s, rad/s). y is never sent (see FIRMWARE_MAX_X)."""
    level = int(c.get("speed", 1))
    s = LEVEL_SCALE.get(level, LEVEL_SCALE[1])
    sign = -1.0 if invert_yaw else 1.0
    x = c["x"] * (sprint_x if level >= 4 else max_x * s)
    yaw = c["yaw"] * max_yaw * s * sign
    return {"x": max(-FIRMWARE_MAX_X, min(FIRMWARE_MAX_X, x)), "y": 0.0, "yaw": max(-max_yaw, min(max_yaw, yaw))}


class HeadingHold:
    """Keeps a straight walk straight. Seen at the robot: at sprint speed the dog veers off course (long strides, the legs never push
    exactly evenly), the same way every time at a given speed.

    The first version steered on the raw gyro heading and made it worse: at sprint the body rocks about +-12 degrees with every
    stride, and the hold chased that rocking (full left, full right, twice a second), and it measured "off course" from wherever the
    body happened to be mid-rock when the walk began (a random offset of up to 12 degrees on every walk). So now:
      * the heading is averaged over exactly one stride cycle (WINDOW_S), which cancels the rocking, and the course to hold is that
        average over the first cycle of the walk;
      * the steady pull is learned as a TRIM per speed and direction (an integral term) and kept between walks, so the next sprint
        starts already corrected; a smaller proportional term (KP) handles what is left;
      * the steering changes at most once per stride cycle (EVERY_S), in STEP increments: every velocity message makes the robot
        re-apply its stance, so a stream of changes would make it flex."""

    # Tuned in a simulation of 22 drift cases (steady pull, +-12 deg stride rocking, 0.15-0.35 s command delay): after the first
    # walk at a speed, a 4 s sprint ends ~3 deg off course on average (worst 9) against 9 (worst 25) for the first version.
    KP = 0.3          # rad/s of steering per rad of (stride-averaged) heading error
    KI = 0.3          # rad/s of trim learned per rad of error per second
    MAX = 0.3         # rad/s (~17 deg/s): below the 25 deg/s turn limit
    STEP = 0.02       # rad/s: steering is rounded to this
    DEADBAND = 0.026  # rad (~1.5 deg) of averaged error: left alone
    WINDOW_S = 0.5    # one stride cycle (2 x (swing 0.15 + overlap 0.1)): averaging over it cancels the rocking
    EVERY_S = 0.5     # steer at most once per stride cycle

    def __init__(self, path: str | None = None, steer: bool = True) -> None:
        self.path = path  # where learned trims are kept across restarts (None: memory only)
        self.steer = steer  # False: only MEASURE each straight walk's drift (logged), never steer
        self.active = False
        self.err = 0.0       # rad, counter-clockwise positive, since the straight walk began
        self.avg = 0.0       # err averaged over the last stride cycle: the course, without the rocking
        self.ref = None      # the course to hold: avg over the walk's first stride cycle
        self._win: collections.deque = collections.deque()  # (time, err) over the last WINDOW_S
        self._t_start = 0.0
        self.bias = 0.0      # rad/s the gyro reads while standing still (learned)
        self.corr = 0.0
        self.trim: dict = self._load()  # (speed level, forward?) -> learned steady steering, rad/s
        self.key = None
        self._t_imu = None
        self._t_corr = 0.0

    def feed(self, wz: float, still: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._t_imu is None else min(0.2, max(0.0, now - self._t_imu))
        self._t_imu = now
        if still:  # standing: learn the gyro's offset, so it is not mistaken for a turn
            self.bias = 0.98 * self.bias + 0.02 * wz
        elif self.active:
            self.err += (wz - self.bias) * dt
            self._win.append((now, self.err))
            while self._win and now - self._win[0][0] > self.WINDOW_S:
                self._win.popleft()
            self.avg = sum(e for _, e in self._win) / len(self._win)
            if self.ref is None and now - self._t_start >= self.WINDOW_S:
                self.ref = self.avg

    def correction(self, straight: bool, key=None, now: float | None = None) -> float:
        """Call every control tick. straight: walking forward/back with no turn asked for; key: (speed level, forward?), whose trim
        is learned and reused. Returns the yaw rate to add (rad/s)."""
        now = time.monotonic() if now is None else now
        if not straight:
            if self.active and self.key is not None:
                walked = now - self._t_start
                if walked >= 1.0 and self.ref is not None:
                    drift = math.degrees(self.avg - self.ref)
                    print("walk drift (gyro, unreliable while walking): speed %s %s, %.1f s, turned %+.1f deg (%s), %.1f deg/s%s" % (
                        self.key[0], "forward" if self.key[1] else "back", walked, drift, "left" if drift > 0 else "right",
                        drift / walked, "" if self.steer else ", not steered"), flush=True)
                if self.steer:
                    print("heading hold: walk ended, trim for speed %s %s is %+.2f rad/s" % (
                        self.key[0], "forward" if self.key[1] else "back", self.trim.get(self.key, 0.0)), flush=True)
                    self._save()
            self.active, self.err, self.avg, self.corr, self.ref = False, 0.0, 0.0, 0.0, None
            self._win.clear()
            return 0.0
        if not self.active or key != self.key:
            self.active, self.key, self.err, self.avg, self.ref, self._t_corr = True, key, 0.0, 0.0, None, now
            self._t_start = self._t_imu if self._t_imu is not None else now
            self._win.clear()
            self.corr = self._round(self.trim.get(key, 0.0)) if self.steer else 0.0  # start with what the last walk learned
            return self.corr
        if not self.steer:
            return 0.0
        if now - self._t_corr >= self.EVERY_S:
            dt, self._t_corr = now - self._t_corr, now
            if self.ref is None:  # the first stride cycle: the course is not known yet, keep the learned trim
                return self.corr
            e = self.avg - self.ref
            e = e if abs(e) >= self.DEADBAND else 0.0
            trim = max(-self.MAX, min(self.MAX, self.trim.get(key, 0.0) - self.KI * e * dt))
            self.trim[key] = trim
            corr = self._round(trim - self.KP * e)
            if corr != self.corr:
                print("heading hold: off course %+.1f deg -> steer %+.2f rad/s (trim %+.2f)"
                      % (math.degrees(self.avg - self.ref), corr, trim), flush=True)
            self.corr = corr
        return self.corr

    def _load(self) -> dict:
        try:
            raw = json.loads(open(self.path, encoding="utf-8").read()) if self.path else {}
            return {(int(k.split(":")[0]), k.split(":")[1] == "fwd"): max(-self.MAX, min(self.MAX, float(v))) for k, v in raw.items()}
        except (OSError, ValueError, IndexError):
            return {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({f"{lv}:{'fwd' if fw else 'back'}": round(v, 3) for (lv, fw), v in self.trim.items()}, f)
        except OSError:
            pass

    def _round(self, v: float) -> float:
        return max(-self.MAX, min(self.MAX, round(v / self.STEP) * self.STEP))


def battery_status(mv: int | None, t: float) -> dict:
    """Voltage and a rough charge estimate for the 2S LiPo pack (8.4 V full, ~6.6 V is where the robot should be charged).
    The percentage is a straight-line guess from voltage: it sags under load, so read it as a trend, not a fuel gauge."""
    if mv is None or time.monotonic() - t > 15:
        return {"battery_v": None, "battery_pct": None, "battery_level": None}
    v = mv / 1000.0
    pct = max(0, min(100, round((v - 6.6) / (8.4 - 6.6) * 100)))
    return {"battery_v": round(v, 2), "battery_pct": pct, "battery_level": "low" if pct < 20 else "ok"}


class RosbridgeDriver:
    BATTERY_TOPIC = "/ros_robot_controller/battery"  # std_msgs/UInt16, millivolts of the 2S LiPo (Hiwonder ros_robot_controller)
    IMU_TOPIC = "/ros_robot_controller/imu_raw"  # sensor_msgs/Imu, from Hiwonder's puppy_control source

    def __init__(self, url: str, drive: DriveState, *, max_x=None, sprint_x=None, max_yaw=YAW_LIMIT, invert_yaw=False, on_imu=None,
                 hold_heading: bool | None = None):
        self.url, self.drive = url, drive
        self.on_imu = on_imu  # callback(yaw_rate_rad_s) for each IMU sample
        # Steering is OFF by default. At the robot it made fast walks worse and nearly flipped the dog at 30 cm/s, and the reason
        # showed up later: WHILE THE DOG WALKS, its gyro reports a false turn of about 13-20 deg/s (a 15 cm/s sprint that went
        # visibly straight "turned" 60-130 degrees), so the hold steered against turns that were not happening. Turning in place
        # (the navigator) is not affected. Opt in with DRIVE_HOLD_HEADING=1 only with a better yaw source. Off, each straight walk's
        # gyro reading is still logged ("walk drift (gyro, unreliable while walking): ..."): read it as a symptom, not a measurement.
        # hold_heading=False: no gyro at all.
        if hold_heading is None or hold_heading:
            steer = bool(hold_heading) or os.getenv("DRIVE_HOLD_HEADING", "0") == "1"
            self.hold = HeadingHold(os.getenv("DRIVE_TRIM_FILE", "logs/drive_trim.json"), steer=steer)
        else:
            self.hold = None

        self.imu_samples = 0
        self.battery_mv: int | None = None
        self.battery_t = 0.0
        self.max_x = min(FIRMWARE_MAX_X, _env("DRIVE_MAX_X", 12.0) if max_x is None else max_x)
        # 15: the fastest speed Hiwonder's own controller adapts its gait to (puppy.py, VelocityAutogaitFun). Measured at the robot,
        # forward sprints at 20-30 cm/s visibly drifted off course, and at 15 they went straight (watched, 2026-11-05; the gyro's
        # numbers while walking are not usable, see hold_heading). Faster than 15 is possible (up to 35) but does not walk straight.
        self.sprint_x = min(FIRMWARE_MAX_X, max(self.max_x, _env("DRIVE_SPRINT_X", 15.0) if sprint_x is None else sprint_x))
        self.invert_yaw = invert_yaw
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
        return {"url": self.url, "connected": self.connected, "error": self.error, "velocity_type": self.vel_type, "imu_samples": self.imu_samples, **battery_status(self.battery_mv, self.battery_t)}

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

    def _pose(self, pitch: float) -> None:
        """Hiwonder's stock standing pose, with the body pitch (radians, nose-up positive) booth mode asks for."""
        self._publish("/puppy_control/pose", dict(stance_x=0.0, stance_y=0.0, x_shift=GAIT["x_shift"], height=-10.0,
                                                  roll=0.0, pitch=float(pitch), yaw=0.0, run_time=500))

    def _trick(self, name: str) -> None:
        """Run one of Hiwonder's action groups and wait for it (the service replies when the action group has finished),
        then stand back in our pose: an action group leaves the legs wherever its last frame put them.
        It first goes IDLE: stop, stand LEVEL in the stock pose and let it settle. The action groups start from that pose; at the
        robot, shake_hands from the 25 degree conversation tilt right after a walk tipped the dog over."""
        self._publish("/puppy_control/velocity", {"x": 0.0, "y": 0.0, "yaw_rate": 0.0})
        self._pose(0.0)
        time.sleep(IDLE_S)
        r = self._call("/puppy_control/runActionGroup", {"name": f"{name}.d6ac", "wait": True}, wait=self.drive.action_busy_s + 4)
        if r is not None and r.get("result") is False:
            self.error = f"trick {name} failed: {r.get('values')}"[:200]
        if name not in HOLD_POSTURES:  # sitting and napping stay down until it is told to stand or walk
            self._pose(0.0)  # level: the conversation tilt comes back a moment later (drive.RETILT_S)
        self.drive.action_done()

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
        self._pose(self.drive.pitch_rad)
        time.sleep(0.2)
        self._publish("/puppy_control/gait", {k: v for k, v in GAIT.items() if k != "x_shift"})
        self._call("/puppy_control/set_mark_time", {"data": False})  # leave mark-time so it can walk
        if self.on_imu or self.hold:  # orientation: the gyro's yaw rate
            # 10 ms: at sprint the body rocks twice a second, and a 50 ms sample of that rocking added up to a false drift
            self._send({"op": "subscribe", "topic": self.IMU_TOPIC, "type": "sensor_msgs/Imu", "throttle_rate": 10, "queue_length": 1})
        self._send({"op": "subscribe", "topic": self.BATTERY_TOPIC, "type": "std_msgs/UInt16", "throttle_rate": 2000, "queue_length": 1})

    def _loop(self) -> None:
        c0 = self.drive.current()
        last, last_pub, last_home = None, 0.0, c0["home_seq"]
        last_action, last_pitch = c0["action_seq"], c0["pitch"]  # a trick asked for before we connected is stale: skip it
        last_posture = c0.get("posture", "stand")
        zero_again = False
        while not self._stop.is_set():
            t0 = time.monotonic()
            c = self.drive.current()
            want = velocity(c, self.max_x, self.sprint_x, self.max_yaw, self.invert_yaw)
            if not c["armed"]:
                want = dict(ZERO)
            if self.hold:  # walking straight: steer back onto the course the walk started on
                corr = self.hold.correction(bool(want["x"]) and not c["yaw"], key=(int(c["speed"]), want["x"] > 0))
                if corr:
                    want["yaw"] = corr * (-1 if self.invert_yaw else 1)
            if c["home_seq"] != last_home:
                last_home = c["home_seq"]
                self._publish("/puppy_control/velocity", {"x": 0.0, "y": 0.0, "yaw_rate": 0.0})
                self._call("/puppy_control/go_home")
                want = dict(ZERO)
            posture = c.get("posture", "stand")
            if c["action_seq"] != last_action:
                last_action = c["action_seq"]
                self._trick(c["action"])
                last, want = dict(ZERO), dict(ZERO)
            elif posture == "stand" and last_posture != "stand":  # up from sitting / a nap, to walk or to look at someone
                self._pose(c["pitch"])
            elif c["pitch"] != last_pitch and not c["busy"] and posture == "stand":  # a nose-up tilt would stand a napping dog up
                self._pose(c["pitch"])
            last_pitch, last_posture = c["pitch"], posture
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
                    elif m.get("op") == "publish" and m.get("topic") == self.BATTERY_TOPIC:
                        mv = (m.get("msg") or {}).get("data")
                        if isinstance(mv, (int, float)) and 3000 < mv < 20000:
                            self.battery_mv, self.battery_t = int(mv), time.monotonic()
                    elif m.get("op") == "publish" and m.get("topic") == self.IMU_TOPIC:
                        wz = ((m.get("msg") or {}).get("angular_velocity") or {}).get("z")
                        if isinstance(wz, (int, float)):
                            self.imu_samples += 1
                            if self.on_imu:
                                self.on_imu(float(wz))
                            if self.hold:
                                self.hold.feed(float(wz), still=not any(last.values()) if last else True)
            except TimeoutError:
                pass
            time.sleep(max(0.0, 0.1 - (time.monotonic() - t0)))
