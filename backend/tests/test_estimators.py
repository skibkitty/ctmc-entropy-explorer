"""Unit tests for the entropy production estimators (P1-T3)."""

import inspect

import numpy as np
import pytest

from backend.ctmc_core.estimators import (
    count_repeated_transition_patterns,
    kth_order_estimator,
    repeated_transitions_estimator,
    thermodynamic_uncertainty_relation_estimator,
)
from backend.ctmc_core.models.parallel_tracks import (
    METASTATE_GROUPS,
    generate_rate_matrix_for_parallel_tracks,
)
from backend.ctmc_core.simulation import seed_simulation, simulate_single_trajectory

ESTIMATOR_ZERO_TOLERANCE = 0.1

def test_kth_order_zero_on_constant_trajectory():
    """A trajectory that never changes state yields EPR 0.0, never an error."""
    trajectory = [0] * 20
    assert kth_order_estimator(trajectory, 1, total_simulation_time=5.0) == 0.0
    assert kth_order_estimator(trajectory, 2, total_simulation_time=5.0) == 0.0


def test_kth_order_zero_on_cherrypicked_reversible_trajectory():
    """A trajectory that's reversible yields EPR 0.0."""
    trajectory = [0, 1] * 20 + [0]
    assert kth_order_estimator(trajectory, 1, total_simulation_time=5.0) == 0.0
    assert kth_order_estimator(trajectory, 2, total_simulation_time=5.0) == 0.0


def test_kth_order_zero_on_short_trajectory():
    """k-th order returns 0.0 when there are fewer than k+1 states."""
    assert kth_order_estimator([0, 1], 2, total_simulation_time=5.0) == 0.0
    assert kth_order_estimator([], 1, total_simulation_time=5.0) == 0.0


def test_kth_order_zero_total_time():
    """A zero total simulation time returns 0.0 rather than dividing by zero."""
    assert kth_order_estimator([0, 1, 0, 1, 0, 1], 1, total_simulation_time=0.0) == 0.0


def test_kth_order_rejects_k_less_than_one():
    with pytest.raises(ValueError, match="k"):
        kth_order_estimator([0, 1, 0], 0, total_simulation_time=1.0)
    with pytest.raises(ValueError, match="k"):
        kth_order_estimator([0, 1, 0], -3, total_simulation_time=1.0)


def test_kth_order_rejects_negative_time():
    with pytest.raises(ValueError, match="total_simulation_time"):
        kth_order_estimator([0, 1, 0], 1, total_simulation_time=-1.0)


def test_kth_order_positive_on_driven_chain():
    """A driven chain (positive true EPR) gives a positive estimate."""
    matrix = generate_rate_matrix_for_parallel_tracks(0.5, 0.3, 1.0, 0.2, 0.8, 0.4)
    seed_simulation(7)
    result = simulate_single_trajectory(matrix, METASTATE_GROUPS, max_length=20000)

    k1 = kth_order_estimator(result.trajectory.tolist(), 1, result.final_time)
    k2 = kth_order_estimator(result.trajectory.tolist(), 2, result.final_time)
    assert k1 > 0.5
    assert k2 > 0.0


def test_count_repeated_transition_patterns():
    trajectory = [0, 0, 1, 0, 0, 1, 1, 0, 1]
    counts = count_repeated_transition_patterns(trajectory, 0, 1)

    assert list(counts.values()) == [0, 2, 0, 2]


def test_repeated_transitions_zero_when_no_crossings():
    """No transitions between the start/end pair gives 0.0 without raising."""
    assert repeated_transitions_estimator([0, 0, 0], 0, 1, total_simulation_time=1.0) == 0.0


def test_repeated_transitions_positive_on_unicyclic_trajectory():
    """A forward-biased unicyclic trajectory gives a positive estimate."""
    trajectory = [
        0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0,
        1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 0,
        2, 1, 0, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1,
    ]
    estimate = repeated_transitions_estimator(
        trajectory, 0, 1, total_simulation_time=60.0
    )
    assert estimate > 0.1


def test_repeated_transitions_state_mapping():
    """Coarse-graining via state_mapping is applied before analysis."""
    trajectory = [
        1, 2, 3, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1,
        2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 3, 1, 2, 1,
        3, 2, 1, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2,
    ]
    # Merge fine state 1 into 'a' (same group as 0) to prove coarse-graining.
    mapping = {0: "a", 1: "a", 2: "b", 3: "c"}
    estimate = repeated_transitions_estimator(
        trajectory, "a", "b", total_simulation_time=60.0, state_mapping=mapping
    )
    assert estimate > 0.1


def test_tur_requires_snippet_time_length():
    with pytest.raises(ValueError, match="snippet_time_length"):
        thermodynamic_uncertainty_relation_estimator([0, 1, 0], [0.0, 1.0, 2.0])


def test_tur_returns_lower_bound_on_driven_chain():
    matrix = generate_rate_matrix_for_parallel_tracks(0.5, 0.3, 1.0, 0.2, 0.8, 0.4)
    seed_simulation(7)
    result = simulate_single_trajectory(matrix, METASTATE_GROUPS, max_length=20000)

    state_mapping = {i: "A" if i % 2 == 0 else "B" for i in range(6)}
    lower_bound = thermodynamic_uncertainty_relation_estimator(
        result.trajectory.tolist(),
        result.times.tolist(),
        snippet_time_length=result.final_time / 20,
        state_mapping=state_mapping,
    )

    assert math_is_finite_or_inf_positive(lower_bound)

# Note: As time goes to infinity, the probability of observing a sequence that violates
# detailed balance goes to zero. However, we can't simulate an infinitely long trajectory 
# here, instead we have a finite length. We can say the length approximates the limit of 
# infinite time, but because it's still finite, there's a non-zero chance that we estimate 
# entropy production > 0. 
def test_tur_zero_on_undriven_chain():
    # k_i->j == k_j->i for all states with these parameters, so detailed balance 
    # (a.k.a. zero entropy production because it's not being driven)
    matrix = generate_rate_matrix_for_parallel_tracks(0.5, 0.5, 3.0, 3.0, 1.0, 1.0)
    seed_simulation(7)
    result = simulate_single_trajectory(matrix, METASTATE_GROUPS, max_length=20000)

    state_mapping = {i: "A" if i % 2 == 0 else "B" for i in range(6)}
    estimated_entropy_production = thermodynamic_uncertainty_relation_estimator(
        result.trajectory.tolist(),
        result.times.tolist(),
        snippet_time_length=result.final_time / 20,
        state_mapping=state_mapping,
    )

    assert estimated_entropy_production < ESTIMATOR_ZERO_TOLERANCE



def test_estimators_are_pure():
    """Estimators return values; they must not print or write to disk."""
    source = inspect.getsource(kth_order_estimator)
    assert "print(" not in source
    source = inspect.getsource(repeated_transitions_estimator)
    assert "print(" not in source


def math_is_finite_or_inf_positive(value):
    return (np.isfinite(value) and value > 0) or value == float("inf")