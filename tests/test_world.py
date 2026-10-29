"""Common random numbers: every policy faces the same faults and drivers."""

import numpy as np

from cpa.sim import world as W


def test_world_is_reproducible():
    a, b = W.build(), W.build()
    assert a.chargers.equals(b.chargers) and len(a.episodes) == len(b.episodes)
    assert [(e.charger, e.kind, e.start) for e in a.episodes[:50]] == [
        (e.charger, e.kind, e.start) for e in b.episodes[:50]
    ]


def test_session_draws_do_not_depend_on_who_asks_first():
    w = W.build()
    c1, u1 = w.session_draws(500)
    w.session_draws(7)
    c2, u2 = w.session_draws(500)
    assert np.array_equal(c1, c2) and np.array_equal(u1, u2)


def test_every_hazard_is_a_known_fault():
    cfg = W.truth_config()
    assert {e.kind for e in W.build().episodes} <= set(cfg["faults"])
