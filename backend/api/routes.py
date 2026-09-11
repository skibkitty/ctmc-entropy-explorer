"""HTTP routes for the simulation API.

Routes are intentionally thin: request parsing, state gating
("waking up"/busy), and ValueError → 422 mapping. All simulation logic lives in
``service.run_simulation``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..ctmc_core.models.parallel_tracks import (
    METASTATE_GROUPS,
    generate_rate_matrix_for_parallel_tracks,
    get_true_EPR_for_parallel_tracks,
)
from . import config
from .schemas import ParallelTracksPreset, SimulateRequest, SimulateResponse
from .service import run_simulation

router = APIRouter(prefix="/api")

# Default slider values for the flagship parallel-tracks demo. The frontend
# seeds its form from these so it never hardcodes simulation parameters.
PARALLEL_TRACKS_DEFAULTS = {
    "alpha": 0.5,
    "beta": 0.3,
    "u_one": 1.0,
    "w_one": 0.2,
    "u_two": 0.8,
    "w_two": 0.4,
}


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    """Return ``ready`` once the Numba warm-up has finished, ``cold`` otherwise."""
    return {
        "status": "ready" if request.app.state.simulator_ready else "cold",
    }


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: Request, body: SimulateRequest) -> SimulateResponse:
    """Run one CTMC trajectory through the k-th order estimator sweep."""
    if not request.app.state.simulator_ready:
        raise HTTPException(
            status_code=503,
            detail="Simulator is waking up (Numba JIT warm-up); retry shortly",
        )

    try:
        return run_simulation(
            body,
            max_length_cap=config.settings.max_length_cap,
            max_n_states_cap=config.settings.max_n_states_cap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/presets/parallel-tracks", response_model=ParallelTracksPreset)
def parallel_tracks_preset() -> ParallelTracksPreset:
    """Return the flagship demo's default parameters, matrix, and true EPR."""
    matrix = generate_rate_matrix_for_parallel_tracks(**PARALLEL_TRACKS_DEFAULTS)
    return ParallelTracksPreset(
        **PARALLEL_TRACKS_DEFAULTS,
        rate_matrix=matrix.tolist(),
        metastate_groups={name: list(states) for name, states in METASTATE_GROUPS.items()},
        true_epr=float(get_true_EPR_for_parallel_tracks(**PARALLEL_TRACKS_DEFAULTS)),
    )