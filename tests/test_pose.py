import math
import time

from puppy_mind.pose import HeadingTracker, bearing_of, direction_words, in_view, wrap180, wrap360


def test_integrates_turn_rate_into_degrees_counter_clockwise_positive():
    h = HeadingTracker()
    t = 100.0
    for _ in range(20):  # 2 s at +0.5 rad/s = 1 rad = 57.3 deg
        h.feed(0.5, t)
        t += 0.1
    assert abs(h.value() - math.degrees(0.5 * 1.9)) < 1.0  # first sample only sets the clock


def test_turning_right_wraps_below_zero_into_359():
    h = HeadingTracker()
    t = 0.0
    for _ in range(5):
        h.feed(-0.3, t)
        t += 0.1
    v = h.value()
    assert 330 < v < 360


def test_a_still_gyro_does_not_creep_and_reset_zeroes_it():
    h = HeadingTracker()
    t = 0.0
    for _ in range(100):
        h.feed(0.001, t)  # noise below the deadband
        t += 0.1
    assert h.value() == 0.0
    for _ in range(10):
        h.feed(0.5, t)
        t += 0.1
    assert h.value() > 10
    h.reset()
    assert h.value() == 0.0


def test_no_recent_imu_data_means_unknown_not_zero():
    h = HeadingTracker(stale_s=0.05)
    assert h.value() is None  # never fed
    h.feed(0.0)
    assert h.value() == 0.0
    time.sleep(0.12)
    assert h.value() is None  # stale


def test_bearings_use_left_positive_and_wrap():
    assert bearing_of("center", 350.0, 60) == 350.0
    assert bearing_of("left", 350.0, 60) == 10.0      # 350 + 20
    assert bearing_of("right", 5.0, 60) == 345.0      # 5 - 20
    assert bearing_of("left", None, 60) is None
    assert wrap180(190) == -170 and wrap360(-10) == 350


def test_view_cone_membership_with_margin():
    assert in_view(10, 0, 60) is True and in_view(80, 0, 60) is False
    assert in_view(350, 5, 60) is True                 # across the 0/360 seam
    assert in_view(180, 0, 60) is False                # directly behind
    assert in_view(None, 0, 60) is None and in_view(10, None, 60) is None


def test_direction_words():
    assert direction_words(0) == "straight ahead"
    assert direction_words(60) == "to your left" and direction_words(-60) == "to your right"
    assert direction_words(180) == "directly behind you"
    assert direction_words(130) == "behind you to the left"
