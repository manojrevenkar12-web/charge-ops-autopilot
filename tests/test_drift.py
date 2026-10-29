"""Population stability index, as the drift monitor uses it."""

import numpy as np

from cpa.mlops.lifecycle import psi


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    assert psi(rng.poisson(2, 5000).astype(float), rng.poisson(2, 3000).astype(float)) < 0.05


def test_a_shift_is_flagged():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 5000), rng.normal(0.8, 1, 3000)) > 0.2


def test_too_few_rows_say_nothing():
    assert psi(np.ones(10), np.zeros(10)) == 0.0
