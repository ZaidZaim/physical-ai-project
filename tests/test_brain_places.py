"""The robot finds its own way around: places are recognised, created and connected from what it sees; nothing is typed in."""
import dataclasses

import pytest

from puppy_mind.brain import Brain
from puppy_mind.config import get_settings
from puppy_mind.models import Detection, Scene


def D(i, label, color, zone="center"):
    return Detection(idx=i, label=label, color=color, description="", zone=zone, confidence=0.9)


def view(room, *objs):
    return Scene(summary=room, room_type=room, room_confidence=0.9 if room != "unknown" else 0.0,
                 objects=[D(i, l, c) for i, (l, c) in enumerate(objs)], relations=[])


KITCHEN = view("kitchen", ("fridge", "white"), ("stove", "black"), ("table", "brown"), ("chair", "brown"))
KITCHEN_CORNER = view("kitchen", ("fridge", "white"), ("chair", "brown"))
LIVING = view("living room", ("sofa", "grey"), ("tv", "black"), ("table", "brown"), ("chair", "brown"))
WALL = view("unknown")


@pytest.fixture
def brain(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="", miss_threshold=2, place_confirm_frames=2, default_location="Start")
    b = Brain(s, need_openai=False)
    yield b
    b.close()


def places(b):
    return {p["name"]: p for p in b.mem.places_overview()}


def test_it_names_and_creates_places_by_itself_and_learns_the_map(brain):
    r1 = brain.observe_scene(KITCHEN)
    assert brain.location == "Kitchen" and r1["location"] == "Kitchen"  # first view: a place is created from what it sees
    brain.observe_scene(KITCHEN)
    brain.observe_scene(LIVING)  # first odd frame: not confirmed yet...
    assert brain.location == "Kitchen" and brain.place_state["uncertain"]
    brain.observe_scene(LIVING)  # ...second one confirms: the robot is somewhere else
    assert brain.location == "Living Room"
    p = places(brain)
    assert set(p) >= {"Kitchen", "Living Room"} and p["Living Room"]["room_type"] == "living room"
    assert p["Kitchen"]["adjacent"] == [{"place": "Living Room", "times": 1}]  # connected because the robot went between them


def test_a_chair_in_the_kitchen_and_a_chair_in_the_living_room_are_different_things(brain):
    for _ in range(2):
        brain.observe_scene(KITCHEN)
    for _ in range(2):
        brain.observe_scene(LIVING)
    chairs = [o for o in brain.mem.objs.values() if o["label"] == "chair"]
    assert len(chairs) == 2 and {c["loc"] for c in chairs} == {"Kitchen", "Living Room"}


def test_returning_to_a_known_room_recognises_it_instead_of_making_a_new_one(brain):
    for scene in (KITCHEN, KITCHEN, LIVING, LIVING, KITCHEN, KITCHEN):
        brain.observe_scene(scene)
    assert brain.location == "Kitchen"
    assert sorted(places(brain)) == ["Kitchen", "Living Room"]
    assert places(brain)["Kitchen"]["adjacent"] == [{"place": "Living Room", "times": 2}]


def test_moving_between_rooms_never_makes_things_look_gone(brain):
    """The bug that marked 18 objects 'possibly gone' when the robot was carried into another room."""
    for _ in range(3):
        brain.observe_scene(KITCHEN)
    for scene in (LIVING, LIVING, LIVING, LIVING):
        brain.observe_scene(scene)
    kitchen = [o for o in brain.mem.objs.values() if o["loc"] == "Kitchen"]
    assert kitchen and all(o["status"] == "present" and o["miss"] == 0 for o in kitchen)


def test_a_blank_wall_changes_nothing(brain):
    brain.observe_scene(KITCHEN)
    brain.observe_scene(KITCHEN)
    r = brain.observe_scene(WALL)
    assert brain.location == "Kitchen" and not brain.place_state["uncertain"] and r["location"] == "Kitchen"


def test_looking_at_one_corner_of_a_room_is_not_evidence_that_the_rest_is_gone(brain):
    brain.observe_scene(KITCHEN)
    for _ in range(4):
        r = brain.observe_scene(KITCHEN_CORNER)  # stove and table are just out of view
    assert r["misses_counted"] is False
    stove = next(o for o in brain.mem.objs.values() if o["label"] == "stove")
    assert stove["status"] == "present" and stove["miss"] == 0


def test_but_a_stove_that_is_missing_from_a_mostly_complete_view_does_count(brain):
    brain.observe_scene(KITCHEN)
    almost_all = view("kitchen", ("fridge", "white"), ("table", "brown"), ("chair", "brown"))  # 3 of 4: the stove is not there
    for _ in range(3):
        brain.observe_scene(almost_all)
    stove = next(o for o in brain.mem.objs.values() if o["label"] == "stove")
    assert stove["status"] == "possibly_gone"


def test_manual_override_pins_the_place_and_auto_resumes(brain):
    brain.observe_scene(KITCHEN)
    brain.set_place_manual("Study")
    brain.observe_scene(LIVING)
    brain.observe_scene(LIVING)
    assert brain.location == "Study" and brain.place_info()["mode"] == "manual"
    brain.set_place_auto()
    assert brain.place_info()["mode"] == "auto"


def test_rename_and_merge(brain):
    for scene in (KITCHEN, KITCHEN, LIVING, LIVING):
        brain.observe_scene(scene)
    assert brain.rename_place("Living Room", "Lounge") and brain.location == "Lounge"
    assert not brain.rename_place("Lounge", "Kitchen")  # name taken
    assert brain.merge_places("Lounge", "Kitchen") and sorted(places(brain)) == ["Kitchen"]


def test_a_missing_object_counts_when_the_thing_it_rested_on_is_in_view():
    from puppy_mind.models import Relation

    b_scene = Scene(summary="desk", room_type="office", room_confidence=0.9,
                    objects=[D(0, "desk", "brown"), D(1, "mug", "white"), D(2, "lamp", "black"), D(3, "monitor", "black")],
                    relations=[Relation(subject=1, predicate="ON", object=0, confidence=0.9)])
    only_desk = view("office", ("desk", "brown"))  # exactly a quarter of what is known: a partial view
    import dataclasses as dc

    s = dc.replace(get_settings(), nams_api_key="", miss_threshold=2, default_location="Start")
    import os
    os.environ["PUPPY_FAKE_MEMORY"] = "1"
    brain = Brain(s, need_openai=False)
    try:
        brain.observe_scene(b_scene)
        for _ in range(2):
            r = brain.observe_scene(only_desk)
        by = {o["label"]: o for o in brain.mem.objs.values()}
        assert by["mug"]["status"] == "possibly_gone"  # it was ON the desk and the desk is in view: it is really not there
        assert by["lamp"]["status"] == "present" and by["lamp"]["miss"] == 0  # no anchor in view: just out of frame
        assert by["monitor"]["status"] == "present"
    finally:
        brain.close()


def test_throwaway_memory_never_talks_to_a_real_nams(monkeypatch):
    """Regression: a dev demo on in-RAM memory leaked 12 fake entities into the real NAMS workspace."""
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    s = dataclasses.replace(get_settings(), nams_api_key="nams_would_be_a_real_key_000000000000_000000000000000000000000000000000000000000000000")
    b = Brain(s, need_openai=False)
    try:
        assert b.nams.enabled is False and b.nams.http is None
    finally:
        b.close()


def test_place_learns_its_room_type_from_its_own_frames():
    from puppy_mind.devfake import FakeMemory
    from puppy_mind.config import Settings

    m = FakeMemory(Settings())
    m.ensure_place("Spot", "unknown", provisional=True)
    sc = view("hallway", ("door", "brown"))
    m.ingest(sc, session_id="s", location="Spot", source="t", room_type="hallway")
    assert m.learn_room_type("Spot") is None  # one frame is not enough
    m.ingest(sc, session_id="s", location="Spot", source="t", room_type="hallway")
    assert m.learn_room_type("Spot") == "hallway"
    assert m.places["Spot"]["room_type"] == "hallway"
