from puppy_mind.matching import Known, Seen, assign


def K(i, label, color="unknown", zone="center", anchors=()):
    return Known(i, label, color, zone, frozenset(anchors))


def S(idx, label, color="unknown", zone="center", anchors=()):
    return Seen(idx, label, color, zone, 0.9, frozenset(anchors))


def test_same_object_matches():
    assert assign([K("a", "laptop", "silver")], [S(0, "laptop", "silver")]) == {0: "a"}


def test_different_color_is_a_new_object():
    assert assign([K("a", "backpack", "red")], [S(0, "backpack", "blue")]) == {0: None}


def test_unknown_color_still_matches_sole_candidate():
    assert assign([K("a", "laptop", "silver")], [S(0, "laptop", "unknown")]) == {0: "a"}


def test_two_identical_chairs_stay_distinct_by_zone():
    k = [K("L", "chair", "black", "left"), K("R", "chair", "black", "right")]
    s = [S(0, "chair", "black", "right"), S(1, "chair", "black", "left")]
    assert assign(k, s) == {0: "R", 1: "L"}


def test_one_to_one():
    out = assign([K("a", "cup", "white")], [S(0, "cup", "white"), S(1, "cup", "white")])
    assert sorted(v is None for v in out.values()) == [False, True]


def test_anchor_breaks_tie():
    k = [K("a", "bottle", "blue", "center", [("ON", "table")]), K("b", "bottle", "blue", "center", [("ON", "shelf")])]
    assert assign(k, [S(0, "bottle", "blue", "center", [("ON", "shelf")])]) == {0: "b"}


def test_label_mismatch_never_matches():
    assert assign([K("a", "laptop")], [S(0, "monitor")]) == {0: None}
