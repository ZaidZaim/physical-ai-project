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
    s = dataclasses.replace(get_settings(), nams_api_key="", miss_threshold=2, place_confirm_frames=2,
                            new_place_confirm_frames=2, default_location="Start")  # these tests are about naming and the map, not confirmation
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
    assert brain.location == "Kitchen" and not brain.place_state["uncertain"] and r["location"] == "Kitchen" and "skipped" in r  # nothing stored


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


def test_weak_frames_do_not_erase_a_suspected_move():
    """Cabinet frames (kitchen) with unknown-room frames between them used to reset the confirmation count, so the robot stayed
    'in the living room' and filed kitchen things there."""
    for _ in range(3):
        brain_scene = LIVING
    from puppy_mind.spatial import PlaceInfo, PlaceTracker

    places = [PlaceInfo("Living Room", "living room", objects=[{"label": "sofa", "color": "grey", "seen": 3, "status": "present"},
                                                             {"label": "cabinet", "color": "white", "seen": 1, "status": "present"}]),
              PlaceInfo("Kitchen", "kitchen", objects=[{"label": "fridge", "color": "white", "seen": 3, "status": "present"}])]
    t = PlaceTracker(confirm_frames=2)
    t.set_current("Living Room")
    cab = [{"label": "cabinet", "color": "white"}]
    r1 = t.update(cab, "kitchen", 0.9, places)            # looks like a kitchen
    r2 = t.update(cab, "unknown", 0.0, places)            # a frame with no room cue: neither confirms nor cancels
    assert r1.uncertain and r2.uncertain and not r2.changed
    r3 = t.update(cab, "kitchen", 0.9, places)
    assert r3.changed


def test_a_known_kind_of_room_is_recognised_not_duplicated():
    """Walking Hallway -> Kitchen -> Hallway used to create 'Hallway 2' because the clutter in the hallway had changed."""
    from puppy_mind.spatial import PlaceInfo, PlaceTracker

    places = [PlaceInfo("Hallway", "hallway", objects=[{"label": "doorway", "color": "unknown", "seen": 2, "status": "present"}]),
              PlaceInfo("Kitchen", "kitchen", objects=[{"label": "cabinet", "color": "white", "seen": 3, "status": "present"}])]
    t = PlaceTracker(confirm_frames=2)
    t.set_current("Kitchen")
    view = [{"label": "bag", "color": "multicolored"}, {"label": "box", "color": "black"}]
    t.update(view, "hallway", 0.9, places)
    r = t.update(view, "hallway", 0.9, places)
    assert r.changed and not r.is_new and r.place == "Hallway"


def test_people_and_pets_do_not_count_for_place_or_missing():
    from puppy_mind.spatial import TRANSIENT

    assert {"person", "cat"} <= TRANSIENT
    brain_view = [o for o in ("person", "cat", "fridge") if o not in TRANSIENT]
    assert brain_view == ["fridge"]


def test_a_barely_known_view_of_a_known_kind_of_room_raises_a_doubt():
    """A second bedroom looks like 'the bedroom': say so instead of silently extending the first."""
    from puppy_mind.spatial import PlaceInfo, PlaceTracker

    places = [PlaceInfo("Bedroom", "bedroom", objects=[{"label": "bed", "color": "white", "seen": 3, "status": "present"}, {"label": "desk", "color": "brown", "seen": 2, "status": "present"}]),
              PlaceInfo("Hallway", "hallway", objects=[{"label": "door", "color": "brown", "seen": 2, "status": "present"}])]
    t = PlaceTracker(confirm_frames=1)
    t.set_current("Hallway")
    view = [{"label": "nightstand", "color": "dark"}, {"label": "lamp", "color": "white"}, {"label": "carpet", "color": "grey"}]
    r = t.update(view, "bedroom", 0.9, places)
    assert r.place == "Bedroom" and "different bedroom" in r.hint


def test_a_doubtful_room_is_checked_against_pictures_and_a_second_bedroom_gets_its_own_place(brain):
    """The resolver doubts (few known objects in a known kind of room): the vision check decides, without any button."""
    class FakePerception:
        calls = 0

        def same_room(self, current, refs):
            FakePerception.calls += 1
            from puppy_mind.perception import RoomMatch
            return RoomMatch(same_room=False, confidence=0.9, reason="different bed and wall colour")

    import io
    from PIL import Image

    def pic(left_bright):
        im = Image.new("L", (32, 32), 40)
        im.paste(220, (0, 0, 16, 32) if left_bright else (16, 0, 32, 32))
        b = io.BytesIO(); im.convert("RGB").save(b, "JPEG"); return b.getvalue()

    brain.perception = FakePerception()
    bed1 = view("bedroom", ("bed", "white"), ("desk", "brown"), ("chair", "black"))
    bed2 = view("bedroom", ("nightstand", "dark"), ("lamp", "white"), ("carpet", "grey"))
    for _ in range(3):
        brain.observe_scene(bed1, jpeg=pic(True))
    assert brain.location == "Bedroom"
    for _ in range(4):
        brain.observe_scene(bed2, jpeg=pic(False))
    assert brain.location == "Bedroom 2" and FakePerception.calls >= 1
    names = {p["name"] for p in brain.mem.list_places()}
    assert {"Bedroom", "Bedroom 2"} <= names


def test_with_three_new_room_frames_a_new_room_waits_but_a_known_one_still_takes_two(monkeypatch):
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    b = Brain(dataclasses.replace(get_settings(), nams_api_key="", place_confirm_frames=2, new_place_confirm_frames=3,
                                  default_location="Start"), need_openai=False)
    try:
        b.observe_scene(KITCHEN)                       # empty memory: the first view is the first room
        for _ in range(2):
            b.observe_scene(LIVING)
        assert b.location == "Kitchen"                 # two odd frames are not enough for a room never seen before
        b.observe_scene(LIVING)
        assert b.location == "Living Room"
        for _ in range(2):
            b.observe_scene(KITCHEN)
        assert b.location == "Kitchen"                 # going back to a known room: two frames
    finally:
        b.close()


def test_a_close_up_of_the_cat_is_not_a_room(brain):
    """Seen on the real tour: two frames of 'animal, bag' (the cat, called 'animal') became a room, later named 'Hallway 2'."""
    brain.observe_scene(KITCHEN)
    brain.observe_scene(KITCHEN)
    cat = view("unknown", ("animal", "grey"), ("bag", "black"))
    vague = view("unknown", ("object", "unknown"), ("unknown", "unknown"))
    for v in (cat, cat, vague, vague):
        brain.observe_scene(v)
    assert brain.location == "Kitchen" and sorted(places(brain)) == ["Kitchen"]
