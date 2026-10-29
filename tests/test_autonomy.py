"""The autonomy ladder."""

from hypothesis import given
from hypothesis import strategies as st

from cpa.ops.autonomy import Ladder, wilson


@given(st.integers(0, 200), st.integers(1, 200))
def test_wilson_bounds_contain_the_rate(k, n):
    k = min(k, n)
    lo, hi = wilson(k, n)
    assert 0 <= lo <= k / n <= hi <= 1


def test_promotion_needs_enough_good_outcomes():
    lad = Ladder()
    for i in range(14):
        lad.observe("evse_reset", "DC-150", True, i)
    assert lad.level("evse_reset", "DC-150") == "propose"  # 14 < 15 outcomes
    lad.observe("evse_reset", "DC-150", True, 14)
    assert lad.level("evse_reset", "DC-150") == "act"
    assert lad.level("evse_reset", "AC-22") == "propose"  # per hardware model


def test_demotion_when_it_stops_working():
    lad = Ladder()
    for i in range(20):
        lad.observe("unlock", "DC-60", True, i)
    assert lad.level("unlock", "DC-60") == "act"
    for i in range(20):
        lad.observe("unlock", "DC-60", False, 100 + i)
    assert lad.level("unlock", "DC-60") == "propose"


def test_some_actions_are_never_autonomous():
    lad = Ladder()
    for i in range(100):
        lad.observe("firmware", "HPC-350", True, i)
    assert lad.level("firmware", "HPC-350") == "propose"
