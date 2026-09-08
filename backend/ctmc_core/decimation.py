"""
Trajectory decimation for the API's ``trajectory_preview``.

Downsamples a full-resolution trajectory (potentially up to 20,000 points)
to at most ~2,000 points for charting, while the estimators always run on the
full-resolution trajectory.
"""

import numpy as np

DEFAULT_MAX_POINTS = 2000


def decimate_trajectory(times, states, max_points=DEFAULT_MAX_POINTS):
    """
    Downsample a trajectory to at most ``max_points`` points.

    Uses LTTB (Largest-Triangle-Three-Buckets) when the trajectory exceeds
    ``max_points`` to preserve the shape of the staircase; falls back to a
    uniform-stride sample when LTTB doesn't apply.

    Args:
        times: Array of arrival times
        states: Array of visited states
        max_points: Maximum number of output points (>= 2)

    Returns:
        (times, states) tuple of decimated arrays
    """
    times = np.asarray(times, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)

    if times.ndim != 1 or states.ndim != 1:
        raise ValueError("times and states must be 1-D arrays")
    if len(times) != len(states):
        raise ValueError("times and states must have the same length")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    if len(times) <= max_points:
        return times, states

    if max_points == 2:
        return decimate_uniform(times, states, max_points)

    return decimate_lttb(times, states, max_points)


def decimate_uniform(times, states, max_points):
    """
    Uniform-stride decimation: keeps every k-th point plus the last one.
    """
    n = len(times)
    stride = int(np.ceil(n / max_points))
    indices = np.arange(0, n, stride)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)
    return times[indices], states[indices]


def decimate_lttb(times, states, max_points):
    """
    Largest-Triangle-Three-Buckets decimation.

    Resamples ``times``/``states`` down to ``max_points`` points. The first and
    last points are always preserved. Bucket sizes are computed so every input
    point has a fair chance of being selected.
    """
    n = len(times)
    if n <= 3 or max_points >= n:
        return times, states

    bucket_size = (n - 2) / (max_points - 2)
    sampled_times = np.empty(max_points, dtype=np.float64)
    sampled_states = np.empty(max_points, dtype=np.int64)

    sampled_times[0] = times[0]
    sampled_states[0] = states[0]
    last_selected = 0

    for i in range(max_points - 2):
        # Average of the "next" bucket defines the reference point of the triangle.
        avg_range_start = int(np.floor((i + 1) * bucket_size)) + 1
        avg_range_end = min(int(np.floor((i + 2) * bucket_size)) + 1, n - 1)

        # Current bucket of candidates.
        bucket_start = int(np.floor(i * bucket_size)) + 1
        bucket_end = int(np.floor((i + 1) * bucket_size)) + 1

        if avg_range_end <= avg_range_start or bucket_start >= bucket_end:
            # Degenerate buckets (input too small to subdivide cleanly).
            last_selected = min(bucket_start, n - 1)
            sampled_times[i + 1] = times[last_selected]
            sampled_states[i + 1] = states[last_selected]
            continue

        avg_time = times[avg_range_start:avg_range_end].mean()
        avg_state = states[avg_range_start:avg_range_end].mean()

        prev_time = times[last_selected]
        prev_state = states[last_selected]

        times_window = times[bucket_start:min(bucket_end, n - 1)]
        states_window = states[bucket_start:min(bucket_end, n - 1)]

        # Triangle area between the previously selected point, each candidate,
        # and the average of the next bucket.
        areas = np.abs(
            (prev_time - avg_time) * (states_window - prev_state)
            - (prev_time - times_window) * (avg_state - prev_state)
        ) * 0.5

        best_local = int(np.argmax(areas))
        last_selected = bucket_start + best_local

        sampled_times[i + 1] = times[last_selected]
        sampled_states[i + 1] = states[last_selected]

    sampled_times[-1] = times[-1]
    sampled_states[-1] = states[-1]

    return sampled_times, sampled_states