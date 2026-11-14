"""Memory types applied to a robot in space: facts say how they were learned and fade (semantic), routes and tricks are learned by
doing (procedural). The Cypher versions in graph.py mirror the FakeMemory used here; knowledge.py holds the shared logic."""
import dataclasses
import time
from datetime import datetime, timedelta, timezone

import pytest

from puppy_mind import knowledge as k
from puppy_mind.brain import Brain
from puppy_mind.config import get_settings
from puppy_mind.drive import DriveState
from tests.test_brain_places import KITCHEN, LIVING, D

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
ago = lambda days: (NOW - timedelta(days=days)).isoformat()


# ---- semantic memory: how sure, and fading
def test_a_fact_halves_in_confidence_every_half_life_and_told_starts_weaker_than_seen():
    assert k.belief("seen", ago(0), 7, NOW)["confidence"] == 1.0
    assert k.belief("seen", ago(7), 7, NOW)["confidence"] == 0.5 and k.belief("seen", ago(14), 7, NOW)["confidence"] == 0.25
    told = k.belief("told", ago(0), 7, NOW)
    assert told["confidence"] == 0.6 and told["learned_from"] == "told" and told["how"] == "someone told me"
    assert k.belief("inferred", ago(0), 7, NOW)["confidence"] == 0.4
    assert [k.belief("seen", ago(d), 7, NOW)["certainty"] for d in (0, 7, 30)] == ["sure", "probably", "not sure any more"]
    assert k.belief(None, ago(0), 7, NOW)["learned_from"] == "seen"  # facts from before provenance existed came from the camera
    assert k.belief("seen", ago(0), 7, NOW, base=0.8)["confidence"] == 0.8  # the detector's own confidence


def test_the_best_evidence_wins_a_sighting_last_month_is_weaker_than_being_told_today():
    b = k.best_belief([("seen", ago(30), None), ("told", ago(0), None)], 7, NOW)
    assert b["learned_from"] == "told" and b["confidence"] == 0.6
    assert k.best_belief([("seen", ago(1), None), ("told", ago(0), None)], 7, NOW)["learned_from"] == "seen"
    assert k.best_belief([("seen", None, None)], 7, NOW)["learned_from"] is None  # no evidence at all


# ---- procedural memory: routes
def test_a_journey_becomes_route_steps_and_the_way_back_is_the_mirror():
    segs = [{"kind": "forward", "speed": 2, "seconds": 1.5}, {"kind": "forward", "speed": 2, "seconds": 1.0},
            {"kind": "idle", "speed": 2, "seconds": 3}, {"kind": "turn_right", "speed": 2, "seconds": 0.1},  # a blip
            {"kind": "turn_right", "speed": 3, "seconds": 1.2}, {"kind": "forward", "speed": 2, "seconds": 2}]
    steps = k.compress_journey(segs)
    assert steps == [{"kind": "forward", "speed": 2, "seconds": 2.5}, {"kind": "turn_right", "speed": 3, "seconds": 1.2},
                     {"kind": "forward", "speed": 2, "seconds": 2.0}]
    assert k.route_seconds(steps) == 5.7
    assert k.describe_steps(steps) == "walk forward ~3 s, turn right ~1 s, walk forward ~2 s"
    back = k.reverse_steps(steps)
    assert k.describe_steps(back) == "turn round, walk forward ~2 s, turn left ~1 s, walk forward ~3 s"
    assert k.motion_kind(0.5, 0.3) == "forward" and k.motion_kind(0, -1) == "turn_right" and k.motion_kind(0, 0) == "idle"


def test_the_drive_logs_its_moves_as_segments(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("puppy_mind.drive.time.monotonic", lambda: clock[0])
    d = DriveState()
    d.arm(True)
    t0 = clock[0]
    for _ in range(10):  # holding W: a command every 0.2 s for 2 s
        d.command(1, 0, 0)
        clock[0] += 0.2
    for _ in range(5):  # then A for 1 s
        d.command(0, 0, 1)
        clock[0] += 0.2
    d.stop()
    j = d.journey_since(t0)
    assert [s["kind"] for s in j] == ["forward", "turn_left"]
    assert abs(j[0]["seconds"] - 2.0) < 1e-6 and abs(j[1]["seconds"] - 1.0) < 1e-6
    assert d.journey_since(clock[0]) == []  # nothing since


@pytest.fixture
def brain(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="", miss_threshold=2, place_confirm_frames=2, new_place_confirm_frames=2,
                            default_location="Start", trick_avoid_after_fails=2, booth_mode=True)
    b = Brain(s, need_openai=False)
    yield b
    b.close()


def test_walking_from_one_room_to_the_next_teaches_the_route_and_go_to_says_how(brain):
    for _ in range(2):
        brain.observe_scene(KITCHEN)
    brain.drive.arm(True)
    brain.drive.command(1, 0, 0)  # walk out of the kitchen
    time.sleep(0.4)
    brain.drive.stop()
    for _ in range(2):
        brain.observe_scene(LIVING)
    assert brain.location == "Living Room"
    route = brain.mem.routes[("Kitchen", "Living Room")]
    assert route["walks"] == 1 and route["steps"][0]["kind"] == "forward" and 0.3 < route["steps"][0]["seconds"] < 2.0  # (slack for a busy machine)
    how = brain.know_how()
    assert how["routes"][0]["from"] == "Kitchen" and how["routes"][0]["how"].startswith("walk forward")
    # the stove is in the kitchen: going there gives the person the moves, from the way back of the route the robot walked
    r = brain.go_to("stove")
    hop = r["route_how"][0]
    assert (hop["from"], hop["to"]) == ("Living Room", "Kitchen") and hop["reversed"] and hop["how"].startswith("turn round")


def test_turning_on_the_spot_is_no_route(brain):
    brain.drive.arm(True)
    brain.drive.command(0, 0, 1)
    time.sleep(0.4)
    brain.drive.stop()
    brain._learn_route("Kitchen", "Living Room")
    assert brain.mem.routes == {}


# ---- procedural memory: tricks per place
def test_a_trick_that_went_wrong_twice_here_is_skipped_here_but_not_elsewhere(brain):
    brain.observe_scene(KITCHEN)
    brain.drive.arm(True)
    brain.drive.action_busy_s, brain.drive.gesture_cooldown_s = 0.0, 0.0
    def play(**kw):  # each try as if the last trick had long finished
        brain.drive._action_until, brain.drive._moving_until = -100.0, 0.0
        brain.drive._action_times.clear()
        return brain.gesture("push-up", asked=True, **kw)
    assert play()["ok"]
    assert brain.trick_feedback()["recorded"]  # "you fell over": the last trick, here
    assert play()["ok"]
    fb = brain.trick_feedback("push-up")
    assert fb["failed"] == 2 and "skip" in fb["from_now_on"]
    r = play()
    assert not r["ok"] and r["avoided"] and "push-up here" in r["reason"]
    assert play(manual=True)["ok"]  # a person pressing the button still gets it
    assert brain.trick_advice("push-up", "Living Room")["avoid"] is False  # a fact about the kitchen floor only
    t = next(x for x in brain.know_how()["tricks"] if x["trick"] == "push-up")
    assert t["place"] == "Kitchen" and t["avoid"] and t["failed"] == 2


# ---- semantic memory a person adds
def test_being_told_where_something_is_is_believed_less_until_the_camera_sees_it(brain):
    brain.observe_scene(KITCHEN)
    r = brain.remember_told("keys", color="silver", predicate="ON", anchor="table")
    assert r["stored"] and r["place"] == "Kitchen" and r["new_object"] and not r["relation"]["new_anchor"]  # the table it has seen
    keys = brain.mem.find_object("keys")[0]
    assert keys["status"] == "told" and keys["belief"]["learned_from"] == "told" and keys["belief"]["confidence"] == 0.6
    assert keys["active_relations"][0]["predicate"] == "ON" and keys["active_relations"][0]["belief"]["learned_from"] == "told"
    for _ in range(3):  # told-only: never counted as missed, the camera has not looked for it yet
        brain.observe_scene(KITCHEN)
    assert brain.mem.find_object("keys")[0]["status"] == "told"
    seen = KITCHEN.model_copy(update={"objects": KITCHEN.objects + [D(len(KITCHEN.objects), "keys", "silver")]})
    brain.observe_scene(seen)
    keys = brain.mem.find_object("keys")
    assert len(keys) == 1 and keys[0]["status"] == "present" and keys[0]["belief"]["learned_from"] == "seen"  # the same keys, now seen
    assert brain.remember_told("keys", predicate="ABOVE", anchor="sofa")["stored"] is False  # only the known relations


def test_the_booth_voice_can_report_a_trick_that_went_wrong(brain):
    brain.observe_scene(KITCHEN)
    brain.drive.action = "shake_hands"
    r = brain.booth._tool(None, "trick_went_wrong", {})
    assert r["recorded"] and r["trick"] == "shake_hands" and r["place"] == "Kitchen" and r["failed"] == 1


def test_a_spin_with_legs_disabled_is_refused_at_once_and_plays_no_music(brain):
    played = []
    brain.sound_fn = lambda name: played.append(name) or __import__("threading").Event()
    r = brain.move("spin")
    assert not r["started"] and "Enable keys" in r["reason"] and played == []
