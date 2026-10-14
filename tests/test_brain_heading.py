"""Orientation: an object only counts as missing if the robot is actually looking where it was."""
import dataclasses

import pytest

from puppy_mind.brain import Brain
from puppy_mind.config import get_settings
from puppy_mind.models import Detection, Scene


class Heading:
    """Stands in for the IMU heading tracker."""
    def __init__(self, deg):
        self.deg = deg

    def value(self):
        return self.deg

    def reset(self):
        pass

    def feed(self, wz):
        pass


def scene(*objs, room="kitchen"):
    return Scene(summary="s", room_type=room, room_confidence=0.9,
                 objects=[Detection(idx=i, label=l, color=c, description="", zone=z, confidence=0.9) for i, (l, c, z) in enumerate(objs)], relations=[])


@pytest.fixture
def brain(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="", miss_threshold=2, camera_fov_deg=60.0, default_location="Start")
    b = Brain(s, need_openai=False)
    b.heading = Heading(0.0)
    yield b
    b.close()


def by_label(b):
    return {o["label"]: o for o in b.mem.objs.values()}


def test_detections_get_bearings_from_heading_and_position_in_frame(brain):
    brain.heading.deg = 90.0
    brain.observe_scene(scene(("fridge", "white", "left"), ("table", "brown", "center"), ("stove", "black", "right")))
    o = by_label(brain)
    assert o["fridge"]["bearing"] == 110.0 and o["table"]["bearing"] == 90.0 and o["stove"]["bearing"] == 70.0  # left is counter-clockwise


def test_things_behind_the_robot_are_not_missing(brain):
    brain.observe_scene(scene(("fridge", "white", "left"), ("table", "brown", "center"), ("stove", "black", "right")))
    brain.heading.deg = 180.0  # turned around; looking at the opposite wall
    for _ in range(4):
        brain.observe_scene(scene(("window", "white", "center"), ("sofa", "grey", "left")))  # same kind of room: another wall
    assert brain.location == "Kitchen"
    o = by_label(brain)
    assert all(o[k]["status"] == "present" and o[k]["miss"] == 0 for k in ("fridge", "table", "stove"))


def test_but_something_missing_from_where_the_robot_is_looking_does_count(brain):
    brain.observe_scene(scene(("fridge", "white", "left"), ("table", "brown", "center"), ("stove", "black", "right")))
    brain.heading.deg = 0.0
    for _ in range(3):  # looking straight at the old spot of the stove (bearing -20): it is not there
        brain.observe_scene(scene(("fridge", "white", "left"), ("table", "brown", "center")))
    o = by_label(brain)
    assert o["stove"]["status"] == "possibly_gone"


def test_an_object_outside_a_narrow_view_cone_is_never_charged_with_a_miss(brain):
    brain.observe_scene(scene(("table", "brown", "center"), ("sofa", "grey", "left"), ("lamp", "black", "right")))
    brain.heading.deg = 90.0  # looking 90 degrees to the left of the table; the table, sofa and lamp are all out of the cone
    for _ in range(4):
        brain.observe_scene(scene(("window", "white", "center"), ("cabinet", "white", "left")))
    o = by_label(brain)
    assert all(o[k]["miss"] == 0 for k in ("table", "sofa", "lamp"))


def test_without_a_heading_it_falls_back_to_overlap_and_anchors(brain):
    brain.heading.deg = None
    brain.observe_scene(scene(("fridge", "white", "left"), ("table", "brown", "center"), ("stove", "black", "right"), ("chair", "brown", "center")))
    for _ in range(4):
        r = brain.observe_scene(scene(("fridge", "white", "left")))  # a quarter of the room, no anchors: just a corner
    assert r["misses_counted"] is False
    assert by_label(brain)["stove"]["miss"] == 0
