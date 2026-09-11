"""Pure service layer for the simulation API.

Separated from the route handlers so validation, caps, and response shaping can
be unit-tested without an HTTP round-trip. Raises ``ValueError`` for anything
the API surfaces as a 422.
"""

from __future__ import annotations

import time

from ..ctmc_core.decimation import DEFAULT_MAX_POINTS, decimate_trajectory
from ..ctmc_core.estimators import kth_order_estimator
from ..ctmc_core.simulation import simulate_single_trajectory
from .schemas import (
    MetastateSummary,
    SimulateRequest,
    SimulateResponse,
    SimulationMeta,
    TrajectoryPreview,
)


def run_simulation(
    body: SimulateRequest,
    *,
    max_length_cap: int,
    max_n_states_cap: int,
) -> SimulateResponse:
    """Run one simulation with the given caps, returning the API response model.

    Caps and all matrix/group validation raise ``ValueError`` with a
    human-readable message (surfaced as HTTP 422 by the route layer). Estimators
    always run on the full-resolution trajectory; the preview is decimated
    server-side before it leaves the service.
    """
    n_states = len(body.rate_matrix)
    if n_states > max_n_states_cap:
        raise ValueError(
            f"Rate matrix has {n_states} states; the maximum supported is "
            f"{max_n_states_cap}"
        )
    if body.max_length > max_length_cap:
        raise ValueError(
            f"max_length {body.max_length} exceeds the cap of {max_length_cap}"
        )

    started = time.perf_counter()
    result = simulate_single_trajectory(
        body.rate_matrix,
        body.metastate_groups,
        max_length=body.max_length,
        initial_state=body.initial_state,
    )

    epr_estimates = {
        str(k): kth_order_estimator(result.trajectory.tolist(), k, result.final_time)
        for k in body.estimator.k_values
    }

    preview_times, preview_states = decimate_trajectory(
        result.times, result.trajectory, max_points=DEFAULT_MAX_POINTS
    )

    waiting_times_summary = {
        name: _summarize(
            waits=result.waiting_times_by_metastate[name],
            total_time=result.time_in_metastates[name],
        )
        for name in result.waiting_times_by_metastate
    }

    return SimulateResponse(
        trajectory_preview=TrajectoryPreview(
            states=preview_states.tolist(),
            times=preview_times.tolist(),
        ),
        waiting_times_summary=waiting_times_summary,
        epr_estimates=epr_estimates,
        true_epr=None,
        meta=SimulationMeta(
            n_transitions=int(len(result.trajectory)),
            final_time=result.final_time,
            compute_ms=(time.perf_counter() - started) * 1000.0,
        ),
    )


def _summarize(*, waits: list[float], total_time: float) -> MetastateSummary:
    visits = len(waits)
    return MetastateSummary(
        total_time=float(total_time),
        visits=visits,
        avg=float(total_time / visits) if visits else 0.0,
    )