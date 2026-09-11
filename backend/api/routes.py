"""HTTP routes for the simulation API.

Routes are intentionally thin: request parsing, state gating
("waking up"/busy), and ValueError → 422 mapping. All simulation logic lives in
``service.run_simulation``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from . import config
from .schemas import SimulateRequest, SimulateResponse
from .service import run_simulation

router = APIRouter(prefix="/api")


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