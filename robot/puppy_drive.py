#!/usr/bin/env python3
"""Runs ON the PuppyPi. Polls the brain for the keyboard drive command and moves the robot.

Verified against Hiwonder/PuppyPi (ros2 branch, ROS2 Humble): `puppy_control` subscribes
  /puppy_control/velocity  puppy_control_msgs/Velocity {float32 x (cm/s), y (cm/s), yaw_rate (rad/s)}
  /puppy_control/pose, /puppy_control/gait, and offers services set_mark_time (SetBool), go_home (Empty).
The ROS1 image uses a similar package but this script targets ROS2; check with `ros2 topic list`.

Safety (each layer fails toward stopping):
  * The robot PULLS commands; the brain never connects in. If the brain is unreachable, replies
    are stale/garbled, or the brain says disarmed, the robot stops.
  * Absolute speed caps are enforced here (--max-x/--max-y/--max-yaw), whatever the brain sends.
  * Ctrl+C / any exception publishes a stop.

Try it first with NO robot motion:   python3 puppy_drive.py --brain http://<laptop-ip>:8000 --dry-run
Then with the dog on the floor in open space (or held off the ground for the first test).
If a direction is reversed on your unit use --invert-y / --invert-yaw.
"""
import argparse
import json
import math
import time
import urllib.request

LEVEL_SCALE = {1: 0.35, 2: 0.65, 3: 1.0}
GAITS = {  # from Hiwonder's puppy_demo.py
    "trot": dict(overlap_time=0.2, swing_time=0.3, clearance_time=0.0, z_clearance=5.0, x_shift=-0.6),
    "walk": dict(overlap_time=0.1, swing_time=0.2, clearance_time=0.3, z_clearance=5.0, x_shift=-0.65),
    "amble": dict(overlap_time=0.1, swing_time=0.2, clearance_time=0.1, z_clearance=5.0, x_shift=-0.9),
}


def fetch(brain, token, applied, timeout=0.6):
    q = "?ax=%.3f&ay=%.3f&ayaw=%.3f" % (applied["x"], applied["y"], applied["yaw"])
    req = urllib.request.Request(brain + "/api/drive/cmd" + q)
    if token:
        req.add_header("X-Brain-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Driver:
    def __init__(self, dry, gait):
        self.dry = dry
        if dry:
            return
        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import Empty, SetBool
        from puppy_control_msgs.msg import Gait, Pose, Velocity

        self.rclpy, self.Velocity, self.Empty, self.SetBool = rclpy, Velocity, Empty, SetBool
        rclpy.init()
        self.node = Node("puppy_keyboard_drive")
        self.vel = self.node.create_publisher(Velocity, "/puppy_control/velocity", 10)
        pose = self.node.create_publisher(Pose, "/puppy_control/pose", 10)
        gait_pub = self.node.create_publisher(Gait, "/puppy_control/gait", 10)
        self.home_cli = self.node.create_client(Empty, "/puppy_control/go_home")
        mark = self.node.create_client(SetBool, "/puppy_control/set_mark_time")
        g = GAITS[gait]
        time.sleep(0.5)  # let publishers connect
        pose.publish(Pose(stance_x=0.0, stance_y=0.0, x_shift=g["x_shift"], height=-10.0,
                          roll=0.0, pitch=0.0, yaw=0.0, run_time=500))
        gait_pub.publish(Gait(overlap_time=g["overlap_time"], swing_time=g["swing_time"],
                              clearance_time=g["clearance_time"], z_clearance=g["z_clearance"]))
        if mark.wait_for_service(timeout_sec=2.0):  # leave "mark time" mode so it can walk
            mark.call_async(SetBool.Request(data=False))

    def move(self, x, y, yaw):
        if self.dry:
            print("DRY-RUN move x=%.1f cm/s y=%.1f cm/s yaw=%.2f rad/s" % (x, y, yaw))
            return
        self.vel.publish(self.Velocity(x=float(x), y=float(y), yaw_rate=float(yaw)))
        self.rclpy.spin_once(self.node, timeout_sec=0)

    def home(self):
        if self.dry:
            print("DRY-RUN go_home")
            return
        if self.home_cli.wait_for_service(timeout_sec=1.0):
            self.home_cli.call_async(self.Empty.Request())

    def close(self):
        if not self.dry:
            self.rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True, help="e.g. http://192.168.1.20:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--gait", choices=list(GAITS), default="trot")
    ap.add_argument("--max-x", type=float, default=12.0, help="cm/s at speed level 3")
    ap.add_argument("--max-y", type=float, default=6.0, help="cm/s at speed level 3")
    ap.add_argument("--max-yaw", type=float, default=0.6, help="rad/s at speed level 3")
    ap.add_argument("--invert-y", action="store_true")
    ap.add_argument("--invert-yaw", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print commands, publish nothing")
    a = ap.parse_args()

    drv = Driver(a.dry_run, a.gait)
    applied = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    last_home = None
    last_sent = None
    try:
        while True:
            t0 = time.monotonic()
            try:
                c = fetch(a.brain, a.token, applied)
                armed = bool(c.get("armed"))
                s = LEVEL_SCALE.get(int(c.get("speed", 1)), LEVEL_SCALE[1])
                want = {
                    "x": max(-1.0, min(1.0, float(c["x"]))) * a.max_x * s,
                    "y": max(-1.0, min(1.0, float(c["y"]))) * a.max_y * s * (-1 if a.invert_y else 1),
                    "yaw": max(-1.0, min(1.0, float(c["yaw"]))) * a.max_yaw * s * (-1 if a.invert_yaw else 1),
                }
                if not armed or not all(math.isfinite(v) for v in want.values()):
                    want = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                hs = c.get("home_seq", 0)
                if last_home is None:
                    last_home = hs  # ignore stale requests from before we connected
                elif hs != last_home:
                    last_home = hs
                    want = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                    drv.move(0, 0, 0)
                    drv.home()
            except Exception as e:  # unreachable / bad reply -> stop
                if applied != {"x": 0.0, "y": 0.0, "yaw": 0.0}:
                    print("brain unreachable (%s): stopping" % e)
                want = {"x": 0.0, "y": 0.0, "yaw": 0.0}
            if want != last_sent:
                drv.move(want["x"], want["y"], want["yaw"])
                last_sent = dict(want)
            applied = want
            time.sleep(max(0.0, 1.0 / a.hz - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        drv.move(0, 0, 0)
        drv.close()


if __name__ == "__main__":
    main()
