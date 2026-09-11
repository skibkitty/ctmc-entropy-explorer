"""Numba JIT warm-up run at process startup (P2-T4).

Render's free tier spins the service down after 15 idle minutes and wipes the
ephemeral filesystem with it, so Numba would otherwise pay its lazy compilation
cost on the very first user request. This module runs one tiny simulation
through every JIT-compiled function off the request path and flips the health
endpoint from ``cold`` to ``ready`` once it is safe.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from fastapi import FastAPI

from ..ctmc_core.estimators import thermodynamic_uncertainty_relation_estimator
from ..ctmc_core.simulation import seed_simulation, simulate_single_trajectory

logger = logging.getLogger(__name__)


def warm_up_simulator() -> None:
    """Touch every Numba-JIT function with a tiny valid workload.

    Compiles ``simulate_trajectory_core`` (via ``simulate_single_trajectory``),
    the ``_seed_random_core`` RNG wrapper, and the TUR helpers
    ``find_snippet_boundaries_fast`` / ``count_transitions_fast``.
    """
    rate_matrix = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    metastate_groups = {"A": [0], "B": [1]}

    seed_simulation(0)
    simulate_single_trajectory(rate_matrix, metastate_groups, max_length=8, initial_state=0)

    trajectory = [0, 1, 0, 1]
    times = [0.0, 0.5, 1.0, 1.5]
    thermodynamic_uncertainty_relation_estimator(
        trajectory, times, snippet_time_length=2.0
    )


def _run_warmup(app: FastAPI) -> None:
    """Warm-up body: flip the ready flag only if compilation fully succeeded."""
    try:
        warm_up_simulator()
    except Exception:
        app.state.simulator_ready = False
        logger.exception("Numba warm-up failed; simulator stays cold")
        return
    app.state.simulator_ready = True


def start_background_warmup(app: FastAPI) -> None:
    """Run ``warm_up_simulator`` on a daemon thread, flipping the ready flag on success."""
    threading.Thread(
        target=_run_warmup, args=(app,), name="numba-warmup", daemon=True,
    ).start()