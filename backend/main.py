"""FastAPI application for the CTMC Entropy Production Explorer."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CTMC Entropy Production Explorer",
    description="Interactive entropy production estimation for continuous-time Markov chains",
    version="0.1.0",
)

# CORS origins: comma-separated list in CORS_ORIGINS env var, or dev defaults
_cors_origins_str = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173",
)
cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint. Returns 'cold' until Numba JIT is warmed up."""
    return {"status": "cold"}
