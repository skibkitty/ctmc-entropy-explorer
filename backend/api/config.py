"""Application configuration — read once from environment at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass


def parse_cors_origins(value: str) -> list[str]:
    """Split a comma-separated CORS origin list into trimmed, non-empty entries."""
    return [o.strip() for o in value.split(",") if o.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    cors_origins: tuple[str, ...]
    rate_limit: str
    max_length_cap: int
    max_n_states_cap: int
    max_concurrent_sims: int


def load_settings() -> Settings:
    """Build a ``Settings`` from the current environment."""
    return Settings(
        cors_origins=tuple(
            parse_cors_origins(
                os.environ.get(
                    "CORS_ORIGINS",
                    "http://localhost:5173,http://localhost:4173",
                )
            )
        ),
        rate_limit=os.environ.get("RATE_LIMIT", "20/minute"),
        max_length_cap=_env_int("MAX_LENGTH_CAP", 20_000),
        max_n_states_cap=_env_int("MAX_N_STATES_CAP", 12),
        max_concurrent_sims=_env_int("MAX_CONCURRENT_SIMS", 3),
    )


settings = load_settings()
