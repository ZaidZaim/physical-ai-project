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
