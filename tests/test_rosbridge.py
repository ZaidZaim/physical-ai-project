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
