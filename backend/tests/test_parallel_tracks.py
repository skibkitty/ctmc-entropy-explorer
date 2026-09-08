"""Unit tests for the parallel tracks model (P1-T4)."""

import numpy as np
import pytest

from backend.ctmc_core.estimators import kth_order_estimator
from backend.ctmc_core.models.parallel_tracks import (
    METASTATE_GROUPS,
    generate_rate_matrix_for_parallel_tracks,
    get_true_EPR_for_parallel_tracks,
)
from backend.ctmc_core.simulation import seed_simulation, simulate_single_trajectory


PARAMETER_SETS = [
    (0.5, 0.3, 1.0, 0.2, 0.8, 0.4),
    (0.2, 0.7, 0.5, 0.1, 1.2, 1.0),
]

SEEDS = (3, 7, 42, 123)


def test_rate_matrix_is_6x6_and_normalized():
    matrix = generate_rate_matrix_for_parallel_tracks(0.5, 0.3, 1.0, 0.2, 0.8, 0.4)
    assert matrix.shape == (6, 6)
    np.testing.assert_allclose(matrix.sum(axis=0), np.zeros(6), atol=1e-12)
    assert np.all(np.isfinite(matrix))


def test_rate_matrix_off_diagonal_non_negative():
    matrix = generate_rate_matrix_for_parallel_tracks(0.5, 0.3, 1.0, 0.2, 0.8, 0.4)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.all(off_diagonal >= 0)


def test_metastate_groups_partition_states():
    assert METASTATE_GROUPS == {"track1": [0, 2, 4], "track2": [1, 3, 5]}
    assigned = [s for states in METASTATE_GROUPS.values() for s in states]
    assert sorted(assigned) == [0, 1, 2, 3, 4, 5]


def test_true_epr_is_positive_and_finite():
    for params in PARAMETER_SETS:
        true_epr = get_true_EPR_for_parallel_tracks(*params)
        assert np.isfinite(true_epr)
        assert true_epr > 0


@pytest.mark.parametrize("params", PARAMETER_SETS, ids=["high-epr", "low-epr"])
def test_estimator_converges_to_true_epr_as_trajectory_grows(params):
    """The k=1 estimate converges toward the closed-form true EPR."""
    true_epr = get_true_EPR_for_parallel_tracks(*params)
    matrix = generate_rate_matrix_for_parallel_tracks(*params)

    relative_errors = {length: [] for length in (1000, 20000)}
    for seed in SEEDS:
        for length in (1000, 20000):
            seed_simulation(seed)
            result = simulate_single_trajectory(matrix, METASTATE_GROUPS, max_length=length)
            estimate = kth_order_estimator(
                result.trajectory.tolist(), 1, result.final_time
            )
            relative_errors[length].append(abs(estimate - true_epr) / true_epr)

    mean_short = np.mean(relative_errors[1000])
    mean_long = np.mean(relative_errors[20000])
    assert mean_long < mean_short
    assert mean_long < 0.1
    for error in relative_errors[20000]:
        assert error < 0.2