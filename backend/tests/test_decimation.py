"""Unit tests for trajectory decimation (P1-T5)."""

import numpy as np
import pytest

from backend.ctmc_core.decimation import (
    decimate_lttb,
    decimate_trajectory,
    decimate_uniform,
)


def staircase(n_points=5000):
    """A well-behaved staircase trajectory for downsampling."""
    times = np.cumsum(np.ones(n_points) * 0.5)
    times[0] = 0.0
    states = np.zeros(n_points, dtype=np.int64)
    states[1:] = np.cumsum(np.random.randint(0, 3, size=n_points - 1)) % 6
    return times, states


def test_passthrough_when_under_limit():
    times = np.array([0.0, 1.0, 2.0])
    states = np.array([0, 1, 2])
    out_times, out_states = decimate_trajectory(times, states, max_points=2000)
    assert np.array_equal(out_times, times)
    assert np.array_equal(out_states, states)


def test_reduces_to_at_most_max_points():
    times, states = staircase()
    out_times, out_states = decimate_trajectory(times, states, max_points=2000)
    assert len(out_times) == 2000
    assert len(out_states) == 2000


def test_preserves_endpoints():
    times, states = staircase(1000)
    for max_points in (2, 10, 500):
        out_times, out_states = decimate_trajectory(times, states, max_points=max_points)
        assert out_times[0] == times[0]
        assert out_times[-1] == times[-1]
        assert out_states[-1] == states[-1]


def test_lttb_keeps_time_order_and_range():
    times, states = staircase(1000)
    out_times, out_states = decimate_lttb(times, states, 100)
    assert np.all(np.diff(out_times) >= 0)
    assert out_times[0] == times[0]
    assert out_times[-1] == times[-1]
    assert len(out_times) == 100


def test_uniform_stride_keeps_last_point():
    times = np.arange(10, dtype=np.float64)
    states = np.arange(10)
    out_times, out_states = decimate_uniform(times, states, 4)
    assert out_times[-1] == 9.0
    assert len(out_times) <= 4
    assert np.all(np.diff(out_times) > 0)


def test_lttb_passthrough_for_small_input():
    times = np.array([0.0, 1.0, 2.0])
    states = np.array([0, 1, 2])
    out_times, out_states = decimate_lttb(times, states, 100)
    assert np.array_equal(out_times, times)


def test_rejects_too_small_max_points():
    times, states = staircase(100)
    with pytest.raises(ValueError, match="at least 2"):
        decimate_trajectory(times, states, max_points=1)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        decimate_trajectory(np.arange(5.0), np.arange(4), max_points=2)