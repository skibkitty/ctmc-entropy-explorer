"""FastAPI application for the CTMC Entropy Production Explorer."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CTMC Entropy Production Explorer",
    description="Interactive entropy production estimation for continuous-time Markov chains",
    version="0.1.0",
)

# CORS will be configured in P0-T4
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint. Returns 'cold' until Numba JIT is warmed up."""
    return {"status": "cold"}
