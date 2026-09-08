"""Unit tests for the simulation core (P1-T1 / P1-T2)."""

import numpy as np
import pytest

from backend.ctmc_core.simulation import (
    seed_simulation,
    simulate_single_trajectory,
    validate_metastate_groups,
    validate_rate_matrix,
)


def two_state_matrix():
    return np.array([[-1.0, 1.0], [1.0, -1.0]])


def test_returns_structured_result():
    """simulate_single_trajectory returns a populated TrajectoryResult."""
    result = simulate_single_trajectory(
        two_state_matrix(), {"a": [0], "b": [1]}, max_length=100, initial_state=0
    )

    assert len(result.trajectory) == 100
    assert len(result.times) == 100
    assert result.trajectory.dtype == np.int32
    assert result.times.dtype == np.float64
    assert list(result.time_in_metastates.keys()) == ["a", "b"]
    assert result.final_time == pytest.approx(result.times[-1])
    assert result.times[0] == 0.0
    assert all(
        result.times[i] <= result.times[i + 1] for i in range(len(result.times) - 1)
    )
    assert result.metastate_transition_count <= 100
    assert isinstance(result.waiting_times_by_metastate, dict)


def test_result_is_reproducible_with_seed():
    """seed_simulation makes trajectories deterministic."""
    matrix = two_state_matrix()
    seed_simulation(99)
    first = simulate_single_trajectory(matrix, {"a": [0], "b": [1]}, max_length=500)
    seed_simulation(99)
    second = simulate_single_trajectory(matrix, {"a": [0], "b": [1]}, max_length=500)

    assert np.array_equal(first.trajectory, second.trajectory)
    assert np.array_equal(first.times, second.times)


def test_rejects_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        simulate_single_trajectory(
            np.ones((2, 3)), {"a": [0], "b": [1]}, max_length=10
        )


def test_rejects_single_state_matrix():
    with pytest.raises(ValueError, match="at least 2 states"):
        validate_rate_matrix(np.array([[-1.0]]))


def test_rejects_nan_entry():
    bad = two_state_matrix()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match=r"entry \(1, 0\) must be finite"):
        validate_rate_matrix(bad)


def test_rejects_infinite_entry():
    bad = two_state_matrix()
    bad[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_rate_matrix(bad)


def test_rejects_negative_off_diagonal():
    bad = np.array([[-1.0, -2.0], [1.0, -1.0]])
    with pytest.raises(
        ValueError, match=r"entry \(0, 1\) must be non-negative.*got -2.0"
    ):
        validate_rate_matrix(bad)


def test_rejects_non_positive_exit_rate():
    # All-zero matrix: every exit rate is zero.
    with pytest.raises(ValueError, match="strictly positive"):
        validate_rate_matrix(np.zeros((2, 2)))

    # Positive diagonal gives a negative exit rate.
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="strictly positive"):
        validate_rate_matrix(bad)


def test_rejects_unnormalized_columns():
    bad = np.array([[-1.0, 1.0], [1.5, -1.0]])
    with pytest.raises(ValueError, match="column 0 must sum to zero, got 0.5"):
        validate_rate_matrix(bad)


def test_rejects_initial_state_out_of_range():
    with pytest.raises(ValueError, match="initial_state"):
        simulate_single_trajectory(
            two_state_matrix(), {"a": [0], "b": [1]}, max_length=10, initial_state=5
        )


def test_rejects_non_positive_max_length():
    with pytest.raises(ValueError, match="max_length"):
        simulate_single_trajectory(
            two_state_matrix(), {"a": [0], "b": [1]}, max_length=0
        )


def test_rejects_unassigned_state():
    with pytest.raises(ValueError, match="not assigned"):
        simulate_single_trajectory(
            two_state_matrix(), {"a": [0]}, max_length=10
        )


def test_rejects_duplicate_state_assignment():
    with pytest.raises(ValueError, match="multiple metastates"):
        simulate_single_trajectory(
            two_state_matrix(), {"a": [0, 1], "b": [1]}, max_length=10
        )


def test_rejects_state_index_out_of_range_in_group():
    with pytest.raises(ValueError, match="exceeds matrix size"):
        simulate_single_trajectory(
            two_state_matrix(), {"a": [0], "b": [7]}, max_length=10
        )


def test_rejects_empty_metastate_groups():
    with pytest.raises(ValueError, match="non-empty"):
        validate_metastate_groups({}, 2)


def test_validate_rate_matrix_returns_normalized_copy():
    matrix = two_state_matrix()
    validated = validate_rate_matrix(matrix)
    np.testing.assert_allclose(validated, matrix)