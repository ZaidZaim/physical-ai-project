import time

from puppy_mind import drive
from puppy_mind.drive import DriveState


def test_disarmed_ignores_commands():
    d = DriveState()
    d.command(1, 0, 0)
    assert d.current()["x"] == 0.0 and not d.moving


def test_armed_command_is_clamped():
    d = DriveState()
    d.arm(True)
    d.command(5, -9, 0)
    c = d.current()
    assert (c["x"], c["y"], c["yaw"]) == (1.0, -1.0, 0.0)


def test_command_expires_when_controller_goes_quiet(monkeypatch):
    d = DriveState()
    d.arm(True)
    d.command(1, 0, 0)
    assert d.current()["x"] == 1.0
    monkeypatch.setattr(drive, "DEADMAN_S", 0.0)
    time.sleep(0.01)
    assert d.current()["x"] == 0.0


def test_disarm_zeroes_and_blocks():
    d = DriveState()
    d.arm(True)
    d.command(1, 1, 1)
    d.arm(False)
    d.command(1, 1, 1)
    assert d.current()["x"] == 0.0 and d.current()["armed"] is False


def test_moving_flag_and_home_requires_armed():
    d = DriveState()
    d.go_home()
    assert d.home_seq == 0
    d.arm(True)
    d.command(0, 0, 1)
    assert d.moving
    d.go_home()
    assert d.home_seq == 1


def test_speed_level_clamped():
    d = DriveState()
    d.set_speed(9)
    assert d.speed == 3
    d.set_speed(-1)
    assert d.speed == 1


def test_forward_plus_turn_is_not_combined_by_default():
    d = DriveState()
    d.arm(True)
    d.command(1, 0, 1)
    c = d.current()
    assert c["x"] == 0.0 and c["yaw"] == 1.0  # turn wins, walking pauses


def test_combining_can_be_enabled_explicitly():
    d = DriveState(allow_combine=True)
    d.arm(True)
    d.command(1, 0, 1)
    c = d.current()
    assert c["x"] == 1.0 and c["yaw"] == 1.0


def test_tricks_need_booth_mode_never_run_while_moving_block_driving_and_are_rate_limited():
    import math

    from puppy_mind.drive import DriveState

    d = DriveState()
    assert d.request_action("boxing") == {"ok": False, "reason": "booth mode is off"}
    assert d.current()["pitch"] == 0.0
    d.set_booth(True, look_up_deg=45)  # clamped: Hiwonder's pose node accepts at most 31 deg
    assert math.isclose(d.current()["pitch"], math.radians(20))
    assert d.request_action("push-up01")["reason"] == "unknown trick"
    d.arm(True)
    d.command(1, 0, 0)
    assert d.request_action("wave")["reason"] == "the robot is moving"
    d.stop()
    d._moving_until = 0  # settled
    r = d.request_action("boxing")
    assert r["ok"] and d.current()["action"] == "boxing" and d.current()["action_seq"] == 1 and d.in_action
    d.command(1, 0, 0)  # a key press mid-trick is ignored: the dog may be up on its hind legs
    assert d.current()["x"] == 0.0 and d.current()["busy"]
    assert d.request_action("wave")["reason"] == "a trick is already running"
    d.action_done()
    assert not d.in_action and d.moving  # the camera still settles after a trick
    d._moving_until = 0
    assert d.request_action("wave")["reason"] == "tricks are rate-limited"
    d.gesture_cooldown_s, d.max_gestures_per_hour = 0, 1
    assert d.request_action("wave")["reason"] == "hourly trick limit reached"
