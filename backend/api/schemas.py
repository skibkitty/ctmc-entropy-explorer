"""Pydantic request/response models for the public API (BUILD_PLAN §6).

All fields use built-in coercion and standard error messages. Caps
(``max_length``, ``n_states``) are enforced at the service layer using runtime
config, so they produce a clear 422 with a message referencing the cap value
rather than an opaque Pydantic ``less than maximum`` error.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EstimatorSpec(BaseModel):
    """Which estimator to run and at which orders."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["kth_order"] = "kth_order"
    k_values: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])

    @field_validator("k_values")
    @classmethod
    def _k_values_positive(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("k_values must not be empty")
        for k in value:
            if not isinstance(k, int) or k < 1:
                raise ValueError(f"k value {k!r} must be a positive integer")
        return value


class SimulateRequest(BaseModel):
    """Payload for ``POST /api/simulate``."""

    model_config = ConfigDict(extra="forbid")

    rate_matrix: list[list[float]]
    metastate_groups: dict[str, list[int]]
    max_length: int = Field(default=5000)
    initial_state: int = Field(default=0, ge=0)
    estimator: EstimatorSpec = Field(default_factory=EstimatorSpec)

    @field_validator("rate_matrix")
    @classmethod
    def _rate_matrix_nonempty(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("rate_matrix must not be empty")
        return value


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class MetastateSummary(BaseModel):
    """Per-metastate residence-time statistics."""

    total_time: float
    visits: int
    avg: float


class TrajectoryPreview(BaseModel):
    """Decimated trajectory for charting (max ~2 000 points)."""

    states: list[int]
    times: list[float]


class SimulationMeta(BaseModel):
    """Timing and size metadata for the completed simulation run."""

    n_transitions: int
    final_time: float
    compute_ms: float


class SimulateResponse(BaseModel):
    """Full response from ``POST /api/simulate``."""

    trajectory_preview: TrajectoryPreview
    waiting_times_summary: dict[str, MetastateSummary]
    epr_estimates: dict[str, float]
    true_epr: float | None
    meta: SimulationMeta


# ---------------------------------------------------------------------------
# Preset models
# ---------------------------------------------------------------------------

class ParallelTracksParams(BaseModel):
    """Slider values that define a parallel-tracks configuration."""

    alpha: float
    beta: float
    u_one: float
    w_one: float
    u_two: float
    w_two: float


class ParallelTracksPreset(ParallelTracksParams):
    """Pre-built parallel-tracks model: sliders + derived matrix + ground truth."""

    rate_matrix: list[list[float]]
    metastate_groups: dict[str, list[int]]
    true_epr: float
