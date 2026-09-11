"""HTTP routes for the simulation API.

Routes are intentionally thin: request parsing, state gating
("waking up"/busy), and ValueError → 422 mapping. All simulation logic lives in
``service.run_simulation``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..ctmc_core.models.parallel_tracks import (
    METASTATE_GROUPS,
    generate_rate_matrix_for_parallel_tracks,
    get_true_EPR_for_parallel_tracks,
)
from . import config
from .schemas import ParallelTracksPreset, SimulateRequest, SimulateResponse
from .service import run_simulation

router = APIRouter(prefix="/api")

# Per-IP rate limit, complemented by the global concurrency semaphore in
# ``main.py`` (a shared link can hit the box from many IPs at once; per-IP
# limits alone cannot stop that from queueing a single 0.1 CPU instance). The
# limit string is resolved per request so tests can adjust ``RATE_LIMIT``.
limiter = Limiter(key_func=get_remote_address)

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
@limiter.limit(lambda: config.settings.rate_limit)
def simulate(request: Request, body: SimulateRequest) -> SimulateResponse:
    """Run one CTMC trajectory through the k-th order estimator sweep."""
    if not request.app.state.simulator_ready:
        raise HTTPException(
            status_code=503,
            detail="Simulator is waking up (Numba JIT warm-up); retry shortly",
        )

    if not request.app.state.sim_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=(
                "Simulators busy: maximum concurrent simulations in flight; "
                "retry shortly"
            ),
        )
    try:
        try:
            return run_simulation(
                body,
                max_length_cap=config.settings.max_length_cap,
                max_n_states_cap=config.settings.max_n_states_cap,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        request.app.state.sim_semaphore.release()


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