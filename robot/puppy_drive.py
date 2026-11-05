#!/usr/bin/env python3
"""Runs ON the PuppyPi, INSIDE its Docker container (that is where ROS lives). Polls the brain for the
keyboard drive command and moves the robot.

Works on both Hiwonder images (verified against github.com/Hiwonder/PuppyPi, branches ros1 and ros2):
  ROS1:  from puppy_control.msg import Velocity, Pose, Gait
  ROS2:  from puppy_control_msgs.msg import Velocity, Pose, Gait
  topics /puppy_control/velocity (x cm/s, y cm/s, yaw_rate rad/s; yaw CCW positive), /puppy_control/pose,
  /puppy_control/gait; services /puppy_control/go_home (Empty), /puppy_control/set_mark_time (SetBool),
  /puppy_control/runActionGroup (SetRunActionName{name, wait}: "boxing.d6ac", "wave.d6ac", ... for booth-mode tricks).
NOTE: Hiwonder's own demos say lateral (y) movement is "currently not available", so --max-y defaults to 0.

Safety (each layer fails toward stopping):
  * The robot PULLS commands; the brain never connects in. If the brain is unreachable, replies are
    garbled, or the brain says disarmed, the robot stops.
  * Absolute speed caps are enforced here (--max-x/--max-y/--max-yaw), whatever the brain sends.
  * Ctrl+C / any exception publishes a stop.

Try it first with NO robot motion:   python3 puppy_drive.py --brain http://<laptop-ip>:8000 --dry-run
"""
import argparse
import json
import math
import os
import time
import urllib.request

LEVEL_SCALE = {1: 0.35, 2: 0.65, 3: 1.0}
YAW_LIMIT = 0.436  # rad/s = 25 deg/s, the fastest turn Hiwonder's own teleop uses; above it the robot can tip over
GAITS = {  # "stock" = the robot's own default (puppy.py GaitConfigFast + Stand); the others are from puppy_demo.py
    "stock": dict(overlap_time=0.1, swing_time=0.15, clearance_time=0.0, z_clearance=5.0, x_shift=-0.5),
    "trot": dict(overlap_time=0.2, swing_time=0.3, clearance_time=0.0, z_clearance=5.0, x_shift=-0.6),
    "walk": dict(overlap_time=0.1, swing_time=0.2, clearance_time=0.3, z_clearance=5.0, x_shift=-0.65),
    "amble": dict(overlap_time=0.1, swing_time=0.2, clearance_time=0.1, z_clearance=5.0, x_shift=-0.9),
}
ZERO = {"x": 0.0, "y": 0.0, "yaw": 0.0}
TRICKS = ("boxing", "wave", "shake_hands", "bow", "nod")  # only these action groups are ever run, whatever the brain sends
MAX_PITCH = math.radians(30)  # body pitch cap, inside the 31 deg the pose node accepts


def fetch(brain, token, applied, timeout=0.6):
    q = "?ax=%.3f&ay=%.3f&ayaw=%.3f" % (applied["x"], applied["y"], applied["yaw"])
    req = urllib.request.Request(brain + "/api/drive/cmd" + q)
    if token:
        req.add_header("X-Brain-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Driver:
    """Thin ROS1/ROS2 shim. dry=True needs no ROS at all."""

    def __init__(self, dry, gait):
        self.dry, self.ros = dry, None
        if dry:
            return
        g = GAITS[gait]
        want = os.environ.get("ROS_VERSION", "")
        try:
            if want == "1":
                raise ImportError
            self._init_ros2(g)
        except ImportError:
            self._init_ros1(g)

    def _init_ros2(self, g):
        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import Empty, SetBool
        from puppy_control_msgs.msg import Gait, Pose, Velocity

        self.ros = 2
        self.rclpy, self.Velocity, self.Empty = rclpy, Velocity, Empty
        rclpy.init()
        self.node = Node("puppy_keyboard_drive")
        self.vel = self.node.create_publisher(Velocity, "/puppy_control/velocity", 10)
        pose = self.node.create_publisher(Pose, "/puppy_control/pose", 10)
        gait_pub = self.node.create_publisher(Gait, "/puppy_control/gait", 10)
        self.home_cli = self.node.create_client(Empty, "/puppy_control/go_home")
        self.Pose, self.pose_pub, self.x_shift = Pose, pose, g["x_shift"]
        try:  # tricks (booth mode): Hiwonder's action groups
            from puppy_control_msgs.srv import SetRunActionName

            self.SetRunActionName = SetRunActionName
            self.action_cli = self.node.create_client(SetRunActionName, "/puppy_control/runActionGroup")
        except ImportError:
            self.action_cli = None
        mark = self.node.create_client(SetBool, "/puppy_control/set_mark_time")
        time.sleep(0.5)
        pose.publish(Pose(stance_x=0.0, stance_y=0.0, x_shift=g["x_shift"], height=-10.0,
                          roll=0.0, pitch=0.0, yaw=0.0, run_time=500))
        gait_pub.publish(Gait(overlap_time=g["overlap_time"], swing_time=g["swing_time"],
                              clearance_time=g["clearance_time"], z_clearance=g["z_clearance"]))
        if mark.wait_for_service(timeout_sec=2.0):  # leave "mark time" mode so it can walk
            mark.call_async(SetBool.Request(data=False))

    def _init_ros1(self, g):
        import rospy
        from std_srvs.srv import Empty, SetBool
        from puppy_control.msg import Gait, Pose, Velocity

        self.ros = 1
        self.rospy = rospy
        rospy.init_node("puppy_keyboard_drive", anonymous=True, disable_signals=True)
        self.vel = rospy.Publisher("/puppy_control/velocity", Velocity, queue_size=1)
        pose = rospy.Publisher("/puppy_control/pose", Pose, queue_size=1)
        gait_pub = rospy.Publisher("/puppy_control/gait", Gait, queue_size=1)
        self.home_srv = rospy.ServiceProxy("/puppy_control/go_home", Empty)
        self.pose_pub, self.x_shift = pose, g["x_shift"]
        try:  # tricks (booth mode). The ROS1 service type name is assumed to mirror ROS2's: unverified
            from puppy_control.srv import SetRunActionName

            self.action_srv = rospy.ServiceProxy("/puppy_control/runActionGroup", SetRunActionName)
        except ImportError:
            self.action_srv = None
        rospy.sleep(0.5)
        pose.publish(stance_x=0.0, stance_y=0.0, x_shift=g["x_shift"], height=-10.0,
                     roll=0.0, pitch=0.0, yaw=0.0, run_time=500)
        rospy.sleep(0.2)
        gait_pub.publish(overlap_time=g["overlap_time"], swing_time=g["swing_time"],
                         clearance_time=g["clearance_time"], z_clearance=g["z_clearance"])
        rospy.sleep(0.2)
        try:  # leave "mark time" mode so it can walk
            rospy.wait_for_service("/puppy_control/set_mark_time", timeout=2.0)
            rospy.ServiceProxy("/puppy_control/set_mark_time", SetBool)(False)
        except Exception as e:
            print("set_mark_time skipped:", e)

    def move(self, x, y, yaw):
        if self.dry:
            print("DRY-RUN move x=%.1f cm/s y=%.1f cm/s yaw=%.2f rad/s" % (x, y, yaw))
        elif self.ros == 2:
            self.vel.publish(self.Velocity(x=float(x), y=float(y), yaw_rate=float(yaw)))
            self.rclpy.spin_once(self.node, timeout_sec=0)
        else:
            self.vel.publish(x=float(x), y=float(y), yaw_rate=float(yaw))

    def home(self):
        if self.dry:
            print("DRY-RUN go_home")
        elif self.ros == 2:
            if self.home_cli.wait_for_service(timeout_sec=1.0):
                self.home_cli.call_async(self.Empty.Request())
        else:
            self.home_srv()

    def pose(self, pitch):
        """Stock standing pose with a body pitch in radians (nose-up positive), as booth mode asks."""
        if self.dry:
            print("DRY-RUN pose pitch=%.1f deg" % math.degrees(pitch))
            return
        kw = dict(stance_x=0.0, stance_y=0.0, x_shift=self.x_shift, height=-10.0, roll=0.0, pitch=float(pitch), yaw=0.0, run_time=500)
        if self.ros == 2:
            self.pose_pub.publish(self.Pose(**kw))
        else:
            self.pose_pub.publish(**kw)

    def trick(self, name, timeout=12.0):
        """Run one of Hiwonder's action groups ("<name>.d6ac") and wait until it has finished."""
        if self.dry:
            print("DRY-RUN trick", name)
            return
        if self.ros == 2:
            if self.action_cli is None or not self.action_cli.wait_for_service(timeout_sec=1.0):
                print("trick skipped: /puppy_control/runActionGroup not available")
                return
            fut = self.action_cli.call_async(self.SetRunActionName.Request(name=name + ".d6ac", wait=True))
            self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        elif self.action_srv is not None:
            self.action_srv(name + ".d6ac", True)

    def close(self):
        if not self.dry and self.ros == 2:
            self.rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True, help="e.g. http://192.168.149.2:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--gait", choices=list(GAITS), default="stock")
    ap.add_argument("--max-x", type=float, default=12.0, help="cm/s at speed level 3")
    ap.add_argument("--max-y", type=float, default=0.0, help="cm/s at speed level 3 (stock firmware has no lateral move)")
    ap.add_argument("--max-yaw", type=float, default=YAW_LIMIT, help="rad/s at speed level 3 (capped at 0.436)")
    ap.add_argument("--invert-y", action="store_true")
    ap.add_argument("--invert-yaw", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print commands, publish nothing")
    a = ap.parse_args()

    a.max_yaw = min(abs(a.max_yaw), YAW_LIMIT)
    drv = Driver(a.dry_run, a.gait)
    print("ROS%s driver ready%s; polling %s" % (drv.ros or "-", " (dry run)" if a.dry_run else "", a.brain))
    applied = dict(ZERO)
    last_home, last_sent, last_action, last_pitch = None, None, None, 0.0
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
                    want = dict(ZERO)
                hs = c.get("home_seq", 0)
                if last_home is None:
                    last_home = hs  # ignore stale requests from before we connected
                elif hs != last_home:
                    last_home = hs
                    want = dict(ZERO)
                    drv.move(0, 0, 0)
                    drv.home()
                aseq = c.get("action_seq", 0)
                if last_action is None:
                    last_action = aseq  # likewise: a trick asked for before we connected is stale
                elif aseq != last_action and c.get("action") in TRICKS:
                    last_action = aseq
                    drv.move(0, 0, 0)
                    drv.trick(c["action"])
                    drv.pose(float(c.get("pitch", 0.0)))  # stand back up in our pose after the action group
                    last_pitch, want = float(c.get("pitch", 0.0)), dict(ZERO)
                pitch = max(-MAX_PITCH, min(MAX_PITCH, float(c.get("pitch", 0.0))))
                if pitch != last_pitch and not c.get("busy"):
                    drv.pose(pitch)
                    last_pitch = pitch
            except Exception as e:  # unreachable / bad reply -> stop
                if applied != ZERO:
                    print("brain unreachable (%s): stopping" % e)
                want = dict(ZERO)
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
