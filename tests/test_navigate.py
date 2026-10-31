"""Phase 5: go where you last saw it. Planning is pure; the turn/scan controller runs against a simulated robot."""
import dataclasses
import threading
import time

import pytest

from puppy_mind.brain import Brain
from puppy_mind.config import get_settings
from puppy_mind.drive import DriveState
from puppy_mind.models import Detection, Scene
from puppy_mind.navigate import Navigator, pick_target, plan_go_to
from puppy_mind.pose import wrap180, wrap360

CAT = {"id": "obj-1", "label": "cat", "color": "gray", "location": "Hallway", "status": "present", "last_seen": "2026-10-22T16:00:00Z"}


# ---------------------------------------------------------------- planning
def test_plan_turns_the_short_way_and_says_which_side():
    left = plan_go_to(CAT, "Hallway", heading=0.0, bearing=90.0, route=None)
    assert (left.kind, left.turn_deg) == ("turn", 90.0) and "left" in left.message
    right = plan_go_to(CAT, "Hallway", heading=10.0, bearing=300.0, route=None)
    assert (right.kind, right.turn_deg) == ("turn", -70.0) and "right" in right.message


def test_plan_already_facing_it_does_not_move():
    assert plan_go_to(CAT, "Hallway", heading=355.0, bearing=3.0, route=None).kind == "here"


def test_plan_without_a_bearing_from_this_visit_looks_around():
    assert plan_go_to(CAT, "Hallway", heading=0.0, bearing=None, route=None).kind == "scan"


def test_plan_without_a_gyro_heading_does_not_turn_blind():
    assert plan_go_to(CAT, "Hallway", heading=None, bearing=90.0, route=None).kind == "no_heading"


def test_plan_in_another_room_gives_a_route_and_never_walks():
    p = plan_go_to(CAT, "Kitchen", heading=0.0, bearing=None, route=["Kitchen", "Living Room", "Hallway"])
    assert p.kind == "route" and p.route[-1] == "Hallway" and "drive me" in p.message and p.turn_deg is None
    assert plan_go_to(CAT, "Kitchen", 0.0, None, route=None).kind == "unknown_route"


def test_plan_mentions_when_it_may_have_been_moved_and_handles_unknown():
    assert "may have been moved" in plan_go_to({**CAT, "status": "possibly_gone"}, "Hallway", 0.0, 90.0, None).message
    assert plan_go_to(None, "Hallway", 0.0, None, None).kind == "not_found"


def test_pick_target_prefers_here_then_present_then_newest():
    a = {**CAT, "id": "a", "location": "Kitchen", "last_seen": "2026-10-22T18:00:00Z"}
    b = {**CAT, "id": "b", "status": "possibly_gone", "last_seen": "2026-10-22T17:00:00Z"}
    c = {**CAT, "id": "c", "last_seen": "2026-10-22T09:00:00Z"}
    d = {**CAT, "id": "d", "last_seen": "2026-10-22T10:00:00Z"}
    assert pick_target([a, b, c, d], "Hallway")["id"] == "d"
    assert pick_target([a, b], "Hallway")["id"] == "b"
    assert pick_target([], "Hallway") is None


# ---------------------------------------------------------------- simulated robot
class SimRobot:
    """Integrates the commanded yaw into a heading, like the real gyro would (degrees per second per unit command)."""

    def __init__(self, drive: DriveState, deg_per_s: float = 400.0, start: float = 0.0):
        self.drive, self.rate, self.heading = drive, deg_per_s, start
        self.imu_ok, self.stuck = True, False
        self.commands: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            time.sleep(0.002)
            now = time.monotonic()
            yaw = self.drive.current()["yaw"]
            self.commands.append(yaw)
            if not self.stuck:
                self.heading = wrap360(self.heading + yaw * self.rate * (now - last))
            last = now

    def value(self):
        return self.heading if self.imu_ok else None

    def close(self):
        self._stop.set()
        self._t.join(1)


@pytest.fixture
def rig():
    d = DriveState()
    d.arm(True)
    sim = SimRobot(d)
    events = []
    nav = Navigator(d, sim.value, publish=lambda t, e: events.append(e), tick_s=0.005, tolerance_deg=6.0, settle_s=0.0,
                    obs_wait_s=0.5, stall_s=0.3)
    yield d, sim, nav, events
    nav.cancel()
    sim.close()


def turn_plan(deg):
    return plan_go_to(CAT, "Hallway", heading=0.0, bearing=wrap360(deg), route=None)


def test_turn_reaches_the_target_heading_then_stops_and_hands_back(rig):
    d, sim, nav, _ = rig
    assert nav.start(turn_plan(120))["started"]
    nav.wait(5)
    assert nav.state["phase"] == "done", nav.state
    assert abs(wrap180(sim.heading - 120)) <= 8
    assert not d.auto and d.current()["yaw"] == 0.0


def test_turn_right_is_negative_yaw(rig):
    d, sim, nav, _ = rig
    nav.start(turn_plan(-80))
    nav.wait(5)
    assert nav.state["phase"] == "done" and abs(wrap180(sim.heading + 80)) <= 8
    assert min(sim.commands) < 0 and max(sim.commands) <= 0


def test_nothing_moves_unless_driving_is_armed():
    d = DriveState()
    nav = Navigator(d, lambda: 0.0, publish=lambda t, e: None)
    r = nav.start(turn_plan(90))
    assert not r["started"] and "not armed" in r["reason"] and d.current()["yaw"] == 0.0


def test_a_key_press_takes_over_but_the_ui_heartbeat_does_not():
    d = DriveState()
    d.arm(True)
    sim = SimRobot(d, deg_per_s=20.0)  # slow, so the turn is still running when we interfere
    nav = Navigator(d, sim.value, publish=lambda t, e: None, tick_s=0.005, stall_s=5)
    try:
        nav.start(turn_plan(170))
        for _ in range(10):
            d.command(0.0, 0.0, 0.0)  # the UI's 10 Hz zero heartbeat while no key is held
            time.sleep(0.01)
        assert d.auto and nav.state["active"]
        d.command(0.0, 0.0, -1.0)  # the person presses a turn key
        nav.wait(2)
        assert nav.state["phase"] == "stopped" and "took over" in nav.state["message"]
        assert d.current()["yaw"] == -1.0  # the person's command stands, not the navigator's
    finally:
        sim.close()


def test_disarm_space_and_cancel_all_stop_it():
    for interrupt in ("disarm", "stop", "cancel"):
        d = DriveState()
        d.arm(True)
        sim = SimRobot(d, deg_per_s=20.0)
        nav = Navigator(d, sim.value, publish=lambda t, e: None, tick_s=0.005, stall_s=5)
        try:
            nav.start(turn_plan(170))
            time.sleep(0.05)
            {"disarm": lambda: d.arm(False), "stop": d.stop, "cancel": nav.cancel}[interrupt]()
            nav.wait(2)
            assert not nav.state["active"] and not d.auto, interrupt
            time.sleep(0.03)
            assert d.current()["yaw"] == 0.0, interrupt
        finally:
            sim.close()


def test_losing_the_gyro_or_not_turning_stops_it(rig):
    d, sim, nav, _ = rig
    sim.stuck = True
    nav.start(turn_plan(90))
    nav.wait(3)
    assert nav.state["phase"] == "stopped" and "not turning" in nav.state["message"] and not d.auto
    sim.stuck = False
    d.arm(True)
    nav2 = Navigator(d, sim.value, publish=lambda t, e: None, tick_s=0.005)
    sim.rate = 20.0
    nav2.start(turn_plan(170))
    time.sleep(0.05)
    sim.imu_ok = False
    nav2.wait(2)
    assert nav2.state["phase"] == "stopped" and "gyro" in nav2.state["message"] and d.current()["yaw"] == 0.0


def test_scan_turns_until_the_camera_sees_it_then_centres_it(rig):
    d, sim, nav, _ = rig
    seq = {"n": 0}
    target_heading = 200.0  # the cat is behind the robot, a little to the left

    def camera():  # a new analysed frame whenever the robot is standing still
        while not stop.is_set():
            time.sleep(0.02)
            if d.current()["yaw"] == 0.0:
                seq["n"] += 1

    def seen(target, s):
        rel = wrap180(target_heading - sim.heading)
        if abs(rel) > 35:
            return None
        return {"zone": "left" if rel > 12 else "right" if rel < -12 else "center"}

    stop = threading.Event()
    nav.obs_seq_fn, nav.seen_fn = (lambda: seq["n"]), seen
    threading.Thread(target=camera, daemon=True).start()
    nav.start(plan_go_to(CAT, "Hallway", heading=0.0, bearing=None, route=None))
    nav.wait(10)
    stop.set()
    assert nav.state["phase"] == "done" and "Found" in nav.state["message"], nav.state
    assert abs(wrap180(sim.heading - target_heading)) <= 30


def test_scan_gives_up_after_a_full_circle(rig):
    d, sim, nav, _ = rig
    seq = {"n": 0}
    stop = threading.Event()

    def camera():
        while not stop.is_set():
            time.sleep(0.02)
            if d.current()["yaw"] == 0.0:
                seq["n"] += 1

    nav.obs_seq_fn, nav.seen_fn = (lambda: seq["n"]), (lambda t, s: None)
    threading.Thread(target=camera, daemon=True).start()
    nav.start(plan_go_to(CAT, "Hallway", heading=0.0, bearing=None, route=None))
    nav.wait(10)
    stop.set()
    assert nav.state["phase"] == "stopped" and "all the way round" in nav.state["message"]


def test_scan_without_camera_frames_stops_and_says_why(rig):
    d, sim, nav, _ = rig
    nav.obs_seq_fn, nav.seen_fn = (lambda: 0), (lambda t, s: None)
    nav.start(plan_go_to(CAT, "Hallway", heading=0.0, bearing=None, route=None))
    nav.wait(3)
    assert nav.state["phase"] == "stopped" and "camera" in nav.state["message"] and not d.auto


# ---------------------------------------------------------------- brain: memory -> plan
class Heading:
    def __init__(self, deg):
        self.deg = deg

    def value(self):
        return self.deg

    def reset(self):
        self.deg = 0.0

    def feed(self, wz):
        pass


def scene(*objs, room="hallway"):
    return Scene(summary="s", room_type=room, room_confidence=0.9,
                 objects=[Detection(idx=i, label=l, color=c, description="", zone=z, confidence=0.9) for i, (l, c, z) in enumerate(objs)], relations=[])


@pytest.fixture
def brain(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="", camera_fov_deg=60.0, default_location="Hallway")
    b = Brain(s, need_openai=False)
    b.heading = Heading(0.0)
    yield b
    b.close()


def test_brain_turns_to_a_bearing_seen_during_this_visit(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")  # cat at +20 deg
    brain.heading.deg = 300.0
    out = brain.go_to("cat", execute=False)
    assert out["plan"]["kind"] == "turn" and out["plan"]["turn_deg"] == 80.0  # 20 - 300 -> +80, left


def test_brain_does_not_trust_a_bearing_from_an_earlier_visit(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")
    time.sleep(0.01)
    brain._new_heading_frame()  # left and came back: a new heading frame
    assert brain.go_to("cat", execute=False)["plan"]["kind"] == "scan"


def test_brain_gives_a_route_to_another_room(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")
    brain.mem.ensure_place("Living Room", "living room")
    brain.mem.ensure_place("Kitchen", "kitchen")
    brain.mem.note_transition("Hallway", "Living Room")
    brain.mem.note_transition("Living Room", "Kitchen")
    brain.location = "Kitchen"
    plan = brain.go_to("cat", execute=False)["plan"]
    assert plan["kind"] == "route" and plan["route"] == ["Kitchen", "Living Room", "Hallway"]


def test_brain_will_not_scan_without_the_camera(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")
    brain._new_heading_frame()
    brain.drive.arm(True)
    out = brain.go_to("cat")
    assert out["run"]["started"] is False and "camera" in out["run"]["reason"] and not brain.drive.auto


def test_brain_seen_now_matches_label_and_colour(brain):
    brain.last_objects = [{"id": "x", "label": "cat", "color": "black", "zone": "left"}]
    assert brain._seen_now({"label": "cat", "color": "gray"}, 1) is None
    assert brain._seen_now({"label": "cat", "color": "unknown"}, 1)["zone"] == "left"
    assert brain._seen_now({"label": "cat", "color": "gray", "id": "x"}, 1) is not None


def test_brain_says_it_does_not_know_the_room_before_recognising_one(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")  # pinned: no recognition
    brain.place_mode, brain.tracker.current, brain.location = "auto", None, "Demo Area"  # a fresh start in auto mode
    brain.drive.arm(True)
    out = brain.go_to("cat")
    assert out["plan"]["kind"] == "no_place" and "Hallway" in out["plan"]["message"] and "Demo Area" not in out["plan"]["message"]
    assert "run" not in out and not brain.drive.auto



def test_routes_prefer_links_seen_more_than_once(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")
    for p in ("Living Room", "Dining Room", "Bedroom 2", "Bathroom"):
        brain.mem.ensure_place(p, "unknown")
    for _ in range(3):
        brain.mem.note_transition("Living Room", "Dining Room")
        brain.mem.note_transition("Dining Room", "Hallway")
    brain.mem.note_transition("Living Room", "Bedroom 2")  # once, on a long drive: probably wrong
    brain.mem.note_transition("Bedroom 2", "Hallway")
    brain.location = "Living Room"
    plan = brain.go_to("cat", execute=False)["plan"]
    assert plan["route"] == ["Living Room", "Dining Room", "Hallway"] and "only seen once" not in plan["message"]


def test_a_route_that_needs_a_one_off_link_says_it_may_be_wrong(brain):
    brain.observe_scene(scene(("cat", "gray", "left"), ("door", "white", "center")), location="Hallway")
    brain.mem.ensure_place("Bathroom", "bathroom")
    brain.mem.note_transition("Bathroom", "Hallway")
    brain.location = "Bathroom"
    plan = brain.go_to("cat", execute=False)["plan"]
    assert plan["route"] == ["Bathroom", "Hallway"] and "only seen once" in plan["message"]
