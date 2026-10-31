from puppy_mind.spatial import PlaceInfo, PlaceResolver, PlaceTracker


def P(name, room, *objs, provisional=False):
    return PlaceInfo(name, room, provisional, 1, [{"label": l, "color": c, "seen": 3, "status": s} for l, c, s in objs])


def V(*items):
    return [{"label": l, "color": c} for l, c in items]


KITCHEN = P("Kitchen", "kitchen", ("fridge", "white", "present"), ("stove", "black", "present"), ("table", "brown", "present"), ("chair", "brown", "present"))
LIVING = P("Living room", "living room", ("sofa", "grey", "present"), ("tv", "black", "present"), ("table", "brown", "present"), ("chair", "brown", "present"))


def decide(view, room="unknown", conf=0.0, places=(KITCHEN, LIVING), current=None):
    return PlaceResolver().decide(V(*view), room, conf, list(places), current)


def test_distinctive_objects_decide_the_place_not_common_ones():
    d = decide([("sofa", "grey"), ("tv", "black")], current="Kitchen")
    assert not d.stay and d.place == "Living room"  # a sofa and tv are not a kitchen
    d = decide([("chair", "brown")], current="Kitchen")
    assert d.stay and d.place == "Kitchen"  # a chair exists in both rooms: it says nothing, so do not move


def test_the_same_chair_label_in_two_rooms_does_not_confuse_recognition():
    d = decide([("fridge", "white"), ("chair", "brown")], current="Living room")
    assert d.place == "Kitchen" and not d.stay


def test_room_type_is_a_strong_veto_against_look_alike_places():
    d = decide([("table", "brown"), ("chair", "brown")], room="kitchen", conf=0.9, places=(LIVING,), current=None)
    assert d.is_new  # looks like a kitchen, the only known place is a living room: do not file it there


def test_room_type_match_helps_and_mismatch_hurts():
    view = [("table", "brown"), ("lamp", "black")]  # only the table is known: partial overlap, so the bonus is visible
    plain = decide(view, places=(KITCHEN, LIVING), current=None)
    kitchen = decide(view, room="kitchen", conf=0.9, places=(KITCHEN, LIVING), current=None)
    assert kitchen.scores["Kitchen"] > plain.scores["Kitchen"]
    assert kitchen.scores["Living room"] < plain.scores["Living room"]


def test_nothing_known_a_view_with_objects_makes_the_first_place_immediately():
    t = PlaceTracker(confirm_frames=3)
    r = t.update(V(("table", "brown"), ("chair", "brown")), "kitchen", 0.8, [])
    assert r.is_new and r.changed and r.suggested_name == "Kitchen"


def test_blank_wall_or_blur_says_nothing_about_where_we_are():
    t = PlaceTracker()
    t.set_current("Kitchen")
    r = t.update([], "unknown", 0.1, [KITCHEN, LIVING])
    assert r.place == "Kitchen" and not r.changed and not r.uncertain


def test_a_change_needs_consecutive_agreeing_frames():
    t = PlaceTracker(confirm_frames=2)
    t.set_current("Kitchen")
    sofa = V(("sofa", "grey"), ("tv", "black"))
    r1 = t.update(sofa, "unknown", 0.0, [KITCHEN, LIVING])
    assert r1.place == "Kitchen" and r1.uncertain and not r1.changed  # one odd frame: stay, but do not count misses
    r2 = t.update(sofa, "unknown", 0.0, [KITCHEN, LIVING])
    assert r2.changed and r2.place == "Living room" and not r2.is_new


def test_a_flip_flop_never_confirms():
    t = PlaceTracker(confirm_frames=2)
    t.set_current("Kitchen")
    sofa, fridge = V(("sofa", "grey"), ("tv", "black")), V(("fridge", "white"), ("stove", "black"))
    for _ in range(3):
        assert not t.update(sofa, "unknown", 0.0, [KITCHEN, LIVING]).changed
        assert not t.update(fridge, "unknown", 0.0, [KITCHEN, LIVING]).changed


def test_unknown_surroundings_become_a_new_place_after_confirmation():
    t = PlaceTracker(confirm_frames=2)
    t.set_current("Kitchen")
    bed = V(("bed", "white"), ("lamp", "black"))
    assert t.update(bed, "bedroom", 0.9, [KITCHEN, LIVING]).uncertain
    r = t.update(bed, "bedroom", 0.9, [KITCHEN, LIVING])
    assert r.changed and r.is_new and r.suggested_name == "Bedroom"


def test_place_names_do_not_collide():
    ps = [P("Kitchen", "kitchen"), P("Unnamed area 1", "unknown"), P("Unnamed area 2", "unknown")]
    assert PlaceResolver.suggest_name("kitchen", 0.9, ps) == "Kitchen 2"
    assert PlaceResolver.suggest_name("unknown", 0.0, ps) == "Unnamed area 3"
    assert PlaceResolver.suggest_name("bedroom", 0.3, ps) == "Unnamed area 3"  # low room confidence: do not name it


def test_colour_is_a_soft_cue_not_a_hard_filter():
    black_chair_in_brown_chair_place = decide([("chair", "black")], places=(KITCHEN,), current="Kitchen")
    assert black_chair_in_brown_chair_place.stay  # lighting can change a colour: still the same room
    assert black_chair_in_brown_chair_place.scores["Kitchen"] < decide([("chair", "brown")], places=(KITCHEN,), current="Kitchen").scores["Kitchen"]


def test_possibly_gone_objects_count_for_less_when_recognising_a_place():
    ghost = P("Ghost room", "unknown", ("sofa", "grey", "possibly_gone"))
    real = P("Real room", "unknown", ("sofa", "grey", "present"))
    r = PlaceResolver()
    idf = r._idf([ghost, real])
    assert r.score(V(("sofa", "grey")), "unknown", 0.0, ghost, idf) < r.score(V(("sofa", "grey")), "unknown", 0.0, real, idf)


def test_another_wall_of_the_same_room_is_not_a_new_room():
    window_and_cabinet = V(("window", "white"), ("cabinet", "white"))  # nothing here is known about the Kitchen yet
    d = decide(window_and_cabinet, room="kitchen", conf=0.9, current="Kitchen")
    assert d.stay and d.place == "Kitchen" and not d.is_new


def test_but_a_different_kind_of_room_still_becomes_a_new_place():
    d = decide(V(("bed", "white"), ("lamp", "black")), room="bedroom", conf=0.9, current="Kitchen")
    assert d.is_new and d.suggested_name == "Bedroom"


def test_low_confidence_room_type_does_not_trigger_the_same_room_shortcut():
    d = decide(V(("window", "white"), ("cabinet", "white")), room="kitchen", conf=0.3, current="Kitchen")
    assert d.is_new  # not sure it is even a kitchen: do not assume it is the same room


HALLWAY = P("Hallway", "hallway", ("door", "brown", "present"), ("stairs", "brown", "present"), ("box", "brown", "present"), ("shoe", "black", "present"))


def test_one_odd_frame_right_after_a_start_does_not_create_a_room():
    """The false 'Garage': the first frame after a restart showed the Hallway stairs, the model called it a garage, and the
    tracker made a new room at once. Now a new room needs confirming even right after a start; the Hallway is recognised instead."""
    t = PlaceTracker(confirm_frames=2, confirm_new_frames=3)
    places = [HALLWAY, KITCHEN, LIVING]
    r = t.update(V(("staircase", "brown")), "garage", 0.8, places)
    assert not r.changed and r.uncertain and t.current is None
    r = t.update(V(("door", "brown"), ("box", "brown"), ("stairs", "brown")), "hallway", 0.9, places)
    assert r.changed and not r.is_new and r.place == "Hallway"  # a KNOWN place is still recognised at once after a start


def test_a_genuinely_new_room_needs_three_agreeing_frames():
    t = PlaceTracker(confirm_frames=2, confirm_new_frames=3)
    t.set_current("Kitchen")
    bed = V(("bed", "white"), ("lamp", "black"))
    assert t.update(bed, "bedroom", 0.9, [KITCHEN, LIVING]).uncertain
    assert t.update(bed, "bedroom", 0.9, [KITCHEN, LIVING]).uncertain
    r = t.update(V(("bed", "white"), ("wardrobe", "white")), "bedroom", 0.9, [KITCHEN, LIVING])
    assert r.changed and r.is_new and r.suggested_name == "Bedroom"


def test_unrelated_odd_frames_do_not_add_up_to_a_new_room():
    t = PlaceTracker(confirm_frames=2, confirm_new_frames=2)
    t.set_current("Kitchen")
    assert not t.update(V(("bed", "white"), ("lamp", "black")), "bedroom", 0.9, [KITCHEN, LIVING]).changed
    r = t.update(V(("car", "red"), ("bicycle", "blue")), "garage", 0.9, [KITCHEN, LIVING])  # a different 'new' place: starts over
    assert not r.changed and r.uncertain
    r = t.update(V(("car", "red"), ("toolbox", "red")), "garage", 0.9, [KITCHEN, LIVING])  # ...and the garage confirms on its own
    assert r.changed and r.is_new and r.suggested_name == "Garage"


def test_without_a_room_type_frames_of_a_new_place_must_share_objects():
    t = PlaceTracker(confirm_frames=2, confirm_new_frames=2)
    t.set_current("Kitchen")
    t.update(V(("bed", "white"), ("lamp", "black")), "unknown", 0.0, [KITCHEN, LIVING])
    assert not t.update(V(("car", "red"), ("bicycle", "blue")), "unknown", 0.0, [KITCHEN, LIVING]).changed
    t2 = PlaceTracker(confirm_frames=2, confirm_new_frames=2)
    t2.set_current("Kitchen")
    t2.update(V(("bed", "white"), ("lamp", "black")), "unknown", 0.0, [KITCHEN, LIVING])
    assert t2.update(V(("bed", "white"), ("nightstand", "brown")), "unknown", 0.0, [KITCHEN, LIVING]).changed
