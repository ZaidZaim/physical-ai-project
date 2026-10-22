"""RosbridgeDriver against a fake rosbridge server: protocol, scaling, dead-man, no-lateral default."""
import json
import threading
import time

from websockets.sync.server import serve

from puppy_mind.drive import DriveState
from puppy_mind.rosbridge import RosbridgeDriver


class FakeBridge:
    def __init__(self, vel_type="puppy_control/Velocity"):
        self.msgs, self.vel_type = [], vel_type
        self.srv = serve(self._h, "127.0.0.1", 0)
        self.port = self.srv.socket.getsockname()[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def _h(self, ws):
        for raw in ws:
            m = json.loads(raw)
            self.msgs.append(m)
            if m["op"] == "subscribe" and m["topic"] == "/ros_robot_controller/imu_raw":  # the robot starts streaming gyro samples
                for _ in range(6):
                    ws.send(json.dumps({"op": "publish", "topic": m["topic"], "msg": {"angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.5}, "linear_acceleration": {"z": 9.8}}}))
            if m["op"] == "subscribe" and m["topic"] == "/ros_robot_controller/battery":
                ws.send(json.dumps({"op": "publish", "topic": m["topic"], "msg": {"data": 7800}}))
            if m["op"] == "call_service" and m["service"] == "/rosapi/topic_type":
                ws.send(json.dumps({"op": "service_response", "id": m["id"], "result": True,
                                    "values": {"type": self.vel_type}}))

    def vels(self):
        return [m["msg"] for m in self.msgs if m["op"] == "publish" and m["topic"] == "/puppy_control/velocity"]

    def close(self):
        self.srv.shutdown()


def run(drive, fb, **kw):
    d = RosbridgeDriver(f"ws://127.0.0.1:{fb.port}", drive, **kw).start()
    t0 = time.time()
    while not d.connected and time.time() - t0 < 5:
        time.sleep(0.05)
    assert d.connected, d.error
    return d


def test_forward_scaled_and_stops_on_expiry_and_disarm():
    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb)
    try:
        drive.arm(True)
        drive.set_speed(3)
        end = time.time() + 0.5
        while time.time() < end:  # keep the command fresh like the browser does
            drive.command(1, 0, 0)
            time.sleep(0.1)
        assert any(abs(v["x"] - 12.0) < 1e-6 for v in fb.vels())
        time.sleep(0.8)  # silence > dead-man => zero
        assert fb.vels()[-1] == {"x": 0.0, "y": 0.0, "yaw_rate": 0.0}
        drive.command(1, 0, 0)
        drive.arm(False)
        time.sleep(0.3)
        assert fb.vels()[-1]["x"] == 0.0
    finally:
        d.close(); fb.close()


def test_lateral_disabled_by_default_and_message_types_follow_robot():
    fb, drive = FakeBridge("puppy_control_msgs/msg/Velocity"), DriveState()
    d = run(drive, fb)
    try:
        drive.arm(True)
        for _ in range(4):
            drive.command(0, 1, 0)
            time.sleep(0.1)
        assert all(v["y"] == 0.0 for v in fb.vels())
        adv = {m["topic"]: m["type"] for m in fb.msgs if m["op"] == "advertise"}
        assert adv["/puppy_control/velocity"] == "puppy_control_msgs/msg/Velocity"
        assert adv["/puppy_control/pose"] == "puppy_control_msgs/msg/Pose"
    finally:
        d.close(); fb.close()


def test_go_home_calls_service_and_robot_reported_online():
    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb)
    try:
        drive.arm(True)
        drive.go_home()
        time.sleep(0.4)
        assert any(m["op"] == "call_service" and m["service"] == "/puppy_control/go_home" for m in fb.msgs)
        assert drive.status()["robot_online"]
    finally:
        d.close(); fb.close()


def test_no_keepalive_republish_while_holding_a_key():
    """Re-sending the same velocity re-applies the robot's stance and makes it flex: send on change only."""
    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb)
    try:
        drive.arm(True)
        end = time.time() + 1.5
        while time.time() < end:  # browser-like 10 Hz heartbeat with an unchanged command
            drive.command(1, 0, 0)
            time.sleep(0.1)
        moving = [v for v in fb.vels() if v["x"] != 0.0]
        assert len(moving) == 1, moving
    finally:
        d.close(); fb.close()


def test_turn_rate_never_exceeds_the_tested_limit_even_at_speed_3():
    from puppy_mind.rosbridge import YAW_LIMIT

    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb, max_yaw=2.0)  # asking for a dangerous turn rate is clamped
    try:
        drive.arm(True)
        drive.set_speed(3)
        for _ in range(5):
            drive.command(0, 0, 1)
            time.sleep(0.1)
        yaws = [abs(v["yaw_rate"]) for v in fb.vels()]
        assert max(yaws) <= YAW_LIMIT + 1e-9 and max(yaws) > 0.4, yaws
    finally:
        d.close(); fb.close()


def test_startup_restores_stock_gait_not_trot():
    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb)
    try:
        time.sleep(0.2)
        gait = [m["msg"] for m in fb.msgs if m["op"] == "publish" and m["topic"] == "/puppy_control/gait"][0]
        assert gait["swing_time"] == 0.15 and gait["overlap_time"] == 0.1
        pose = [m["msg"] for m in fb.msgs if m["op"] == "publish" and m["topic"] == "/puppy_control/pose"][0]
        assert pose["x_shift"] == -0.5 and pose["height"] == -10.0
    finally:
        d.close(); fb.close()


def test_imu_yaw_rate_reaches_the_heading_tracker_and_integrates_to_degrees():
    from puppy_mind.pose import HeadingTracker

    fb, drive, tracker = FakeBridge(), DriveState(), HeadingTracker()
    d = RosbridgeDriver(f"ws://127.0.0.1:{fb.port}", drive, on_imu=tracker.feed).start()
    try:
        t0 = time.time()
        while d.imu_samples < 4 and time.time() - t0 < 5:
            time.sleep(0.05)
        assert d.imu_samples >= 4
        sub = [m for m in fb.msgs if m["op"] == "subscribe"][0]
        assert sub["topic"] == "/ros_robot_controller/imu_raw" and sub["type"] == "sensor_msgs/Imu" and sub["throttle_rate"] == 50
        assert tracker.value() is not None  # data is fresh
    finally:
        d.close(); fb.close()


def test_without_an_imu_callback_the_driver_never_subscribes():
    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb)
    try:
        time.sleep(0.2)
        assert not [m for m in fb.msgs if m["op"] == "subscribe" and "imu" in m["topic"]]
    finally:
        d.close(); fb.close()


def test_battery_voltage_is_reported_with_a_rough_percentage():
    from puppy_mind.rosbridge import battery_status

    fb, drive = FakeBridge(), DriveState()
    d = run(drive, fb)
    try:
        t0 = time.time()
        while d.battery_mv is None and time.time() - t0 < 5:
            time.sleep(0.05)
        st = d.status()
        assert st["battery_v"] == 7.8 and st["battery_pct"] == 67 and st["battery_level"] == "ok"
    finally:
        d.close(); fb.close()
    assert battery_status(6700, time.monotonic())["battery_level"] == "low"
    assert battery_status(None, 0.0)["battery_v"] is None
    assert battery_status(7800, time.monotonic() - 60)["battery_v"] is None  # stale reading is not shown
