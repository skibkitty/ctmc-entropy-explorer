"""FastAPI application for the CTMC Entropy Production Explorer.

Public API entry-point: ``uvicorn backend.main:app --app-dir /path/to/repo-root``.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import config
from .api.routes import router
from .api.warmup import start_background_warmup

# Re-export for backward compatibility with tests.
parse_cors_origins = config.parse_cors_origins  # noqa: F841


# ---------------------------------------------------------------------------
# Lifespan (Numba warm-up)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Flip /api/health from cold to ready after Numba JIT compilation."""
    app.state.simulator_ready = False
    start_background_warmup(app)
    yield


# ---------------------------------------------------------------------------
# Application factory + singleton
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build a fresh ``FastAPI`` instance wired to all middleware and routes.

    Tests may call this directly (with monkeypatched ``config.settings``)
    to obtain isolated app instances with custom caps, rate limits, or
    concurrency limits.
    """
    application = FastAPI(
        title="CTMC Entropy Production Explorer",
        description=(
            "Interactive entropy production estimation for "
            "continuous-time Markov chains"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Application state shared across requests
    application.state.simulator_ready = False
    application.state.sim_semaphore = threading.BoundedSemaphore(
        config.settings.max_concurrent_sims
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()