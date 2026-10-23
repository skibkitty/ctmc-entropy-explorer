"""Integration tests for the public API (P2-T6).

Uses fresh app instances via ``create_app`` (with monkeypatched
``backend.api.config.settings``) so caps, rate limits, and concurrency can be
tuned per test without touching the production singleton. The slowapi limiter
is shared across app instances, so its storage is reset before every test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.api import config, warmup as warmup_module
from backend.api.routes import PARALLEL_TRACKS_DEFAULTS, limiter
from backend.api.warmup import warm_up_simulator
from backend.ctmc_core.models.parallel_tracks import (
    METASTATE_GROUPS,
    generate_rate_matrix_for_parallel_tracks,
    get_true_EPR_for_parallel_tracks,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _build_app(monkeypatch: pytest.MonkeyPatch, **overrides) -> tuple:
    """Create an isolated app with the given settings overrides."""
    base = dict(
        cors_origins=("http://localhost:5173",),
        rate_limit="1000/minute",
        max_length_cap=20_000,
        max_n_states_cap=12,
        max_concurrent_sims=3,
    )
    base.update(overrides)
    monkeypatch.setattr(config, "settings", config.Settings(**base))
    application = main_module.create_app()
    application.state.simulator_ready = True
    return application


@pytest.fixture(autouse=True)
def _reset_rate_limit_storage():
    """The limiter is module-level and shared across app instances."""
    limiter.reset()
    yield


@pytest.fixture
def valid_two_state_matrix() -> list[list[float]]:
    return [[-1.0, 1.0], [1.0, -1.0]]


@pytest.fixture
def valid_two_state_groups() -> dict[str, list[int]]:
    return {"A": [0], "B": [1]}


@pytest.fixture
def valid_parallel_tracks_matrix() -> list[list[float]]:
    return generate_rate_matrix_for_parallel_tracks(**PARALLEL_TRACKS_DEFAULTS).tolist()


def _square_matrix(n: int) -> list[list[float]]:
    """n-state rate matrix with each column summing to zero."""
    matrix = np.ones((n, n))
    np.fill_diagonal(matrix, -(n - 1))
    return matrix.tolist()


# ---------------------------------------------------------------------------
# /api/health and warm-up
# ---------------------------------------------------------------------------

def test_health_is_cold_before_warmup(monkeypatch):
    app = _build_app(monkeypatch)
    app.state.simulator_ready = False
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "cold"}


def test_health_becomes_ready_after_warmup(monkeypatch):
    monkeypatch.setattr(
        main_module, "start_background_warmup",
        lambda application: warmup_module._run_warmup(application),
    )
    app = _build_app(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.json() == {"status": "ready"}


def test_health_stays_cold_if_warmup_fails(monkeypatch):
    def _failing_warmup():
        raise RuntimeError("boom")

    monkeypatch.setattr(warmup_module, "warm_up_simulator", _failing_warmup)
    monkeypatch.setattr(
        main_module, "start_background_warmup",
        lambda application: warmup_module._run_warmup(application),
    )
    app = _build_app(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.json() == {"status": "cold"}


def test_warm_up_simulator_runs_every_jit_function():
    """The real warm-up must not raise: it compiles the sim core and TUR helpers."""
    warm_up_simulator()


# ---------------------------------------------------------------------------
# POST /api/simulate — happy path
# ---------------------------------------------------------------------------

def test_simulate_happy_path_full_response_shape(
    monkeypatch, valid_parallel_tracks_matrix
):
    app = _build_app(monkeypatch)
    client = TestClient(app)

    body = {
        "rate_matrix": valid_parallel_tracks_matrix,
        "metastate_groups": METASTATE_GROUPS,
        "max_length": 2000,
        "initial_state": 0,
        "estimator": {"type": "kth_order", "k_values": [1, 2, 4]},
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200

    data = response.json()
    assert set(data) == {
        "trajectory_preview",
        "waiting_times_summary",
        "epr_estimates",
        "true_epr",
        "meta",
    }

    preview = data["trajectory_preview"]
    assert set(preview) == {"states", "times"}
    assert len(preview["states"]) == len(preview["times"])
    assert len(preview["states"]) <= 2000
    assert set(preview["states"]).issubset(range(6))

    assert set(data["waiting_times_summary"]) == {"track1", "track2"}
    for summary in data["waiting_times_summary"].values():
        assert set(summary) == {"total_time", "visits", "avg"}
        assert summary["visits"] >= 0
        assert summary["total_time"] >= 0

    assert set(data["epr_estimates"]) == {"1", "2", "4"}
    assert all(math.isfinite(v) for v in data["epr_estimates"].values())
    assert data["true_epr"] is None

    meta = data["meta"]
    assert set(meta) == {"n_transitions", "final_time", "compute_ms"}
    assert meta["n_transitions"] == 2000
    assert meta["final_time"] > 0
    assert meta["compute_ms"] > 0


def test_simulate_decimates_long_trajectories(monkeypatch, valid_two_state_matrix):
    app = _build_app(monkeypatch, max_length_cap=50_000)
    client = TestClient(app)

    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": {"A": [0], "B": [1]},
        "max_length": 50_000,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200
    preview_len = len(response.json()["trajectory_preview"]["states"])
    assert 2 <= preview_len <= 2000


def test_simulate_estimator_defaults_to_k_sweep_1_through_4(
    monkeypatch, valid_two_state_matrix, valid_two_state_groups
):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": valid_two_state_groups,
        "max_length": 500,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200
    assert set(response.json()["epr_estimates"]) == {"1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# POST /api/simulate — caps
# ---------------------------------------------------------------------------

def test_simulate_rejects_max_length_over_cap(
    monkeypatch, valid_two_state_matrix, valid_two_state_groups
):
    app = _build_app(monkeypatch, max_length_cap=1000)
    client = TestClient(app)
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": valid_two_state_groups,
        "max_length": 2000,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "exceeds the cap of 1000" in response.json()["detail"]


def test_simulate_rejects_n_states_over_cap(monkeypatch, valid_two_state_groups):
    app = _build_app(monkeypatch, max_n_states_cap=4)
    client = TestClient(app)
    body = {
        "rate_matrix": _square_matrix(5),
        "metastate_groups": {f"g{i}": [i] for i in range(5)},
        "max_length": 100,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "maximum supported is 4" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/simulate — matrix validation (422 paths)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate,message_fragment",
    [
        (lambda m: m.append(m[0]), "must be square"),
        (lambda m: m[0].__setitem__(1, "NaN"), "must be finite"),
        (lambda m: m[0].__setitem__(1, -0.5), "must be non-negative"),
        (lambda m: m[0].__setitem__(0, 0.5), "Exit rate for state 0"),
    ],
)
def test_simulate_rejects_malformed_matrices(monkeypatch, mutate, message_fragment):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    matrix = [[-1.0, 1.0], [1.0, -1.0]]
    mutate(matrix)
    body = {
        "rate_matrix": matrix,
        "metastate_groups": {"A": [0], "B": [1]},
        "max_length": 100,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert message_fragment in response.json()["detail"]


def test_simulate_rejects_column_not_summing_to_zero(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    body = {
        "rate_matrix": [[-1.0, 0.5], [1.0, -1.0]],
        "metastate_groups": {"A": [0], "B": [1]},
        "max_length": 100,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "sum to zero" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/simulate — metastate group validation (422 paths)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "groups,message_fragment",
    [
        ({"A": [0, 1], "B": [1]}, "assigned to multiple metastates"),
        ({"A": [0]}, "not assigned to any metastate"),
        ({"A": [0], "B": [9]}, "exceeds matrix size"),
    ],
)
def test_simulate_rejects_invalid_metastate_groups(
    monkeypatch, valid_two_state_matrix, groups, message_fragment
):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": groups,
        "max_length": 100,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert message_fragment in response.json()["detail"]


def test_simulate_rejects_out_of_range_initial_state(
    monkeypatch, valid_two_state_matrix, valid_two_state_groups
):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": valid_two_state_groups,
        "max_length": 100,
        "initial_state": 5,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "initial_state" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/simulate — schema-level 422 paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "body_mutate",
    [
        lambda b: b.update(estimator={"type": "repeated_transitions", "k_values": [1]}),
        lambda b: b.update(max_length="not-an-int"),
        lambda b: b["metastate_groups"].update(B="not-a-list"),
    ],
)
def test_simulate_rejects_invalid_payload(monkeypatch, body_mutate):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    body = {
        "rate_matrix": [[-1.0, 1.0], [1.0, -1.0]],
        "metastate_groups": {"A": [0], "B": [1]},
        "max_length": 100,
    }
    body_mutate(body)
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/simulate — load protection
# ---------------------------------------------------------------------------

def test_simulate_returns_503_while_waking_up(
    monkeypatch, valid_two_state_matrix, valid_two_state_groups
):
    app = _build_app(monkeypatch)
    app.state.simulator_ready = False
    client = TestClient(app)
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": valid_two_state_groups,
        "max_length": 100,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 503
    assert "waking up" in response.json()["detail"]


def test_simulate_returns_503_when_all_slots_busy(
    monkeypatch, valid_two_state_matrix, valid_two_state_groups
):
    app = _build_app(monkeypatch, max_concurrent_sims=1)
    client = TestClient(app)
    app.state.sim_semaphore.acquire()  # simulate an in-flight run
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": valid_two_state_groups,
        "max_length": 100,
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 503
    assert "busy" in response.json()["detail"]

    app.state.sim_semaphore.release()
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200


def test_simulate_rate_limits_per_ip(
    monkeypatch, valid_two_state_matrix, valid_two_state_groups
):
    app = _build_app(monkeypatch, rate_limit="2/minute")
    client = TestClient(app)
    body = {
        "rate_matrix": valid_two_state_matrix,
        "metastate_groups": valid_two_state_groups,
        "max_length": 100,
    }
    assert client.post("/api/simulate", json=body).status_code == 200
    assert client.post("/api/simulate", json=body).status_code == 200
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]


# ---------------------------------------------------------------------------
# GET /api/presets/parallel-tracks
# ---------------------------------------------------------------------------

def test_parallel_tracks_preset(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)
    response = client.get("/api/presets/parallel-tracks")
    assert response.status_code == 200

    data = response.json()
    for key, value in PARALLEL_TRACKS_DEFAULTS.items():
        assert data[key] == value

    matrix = np.array(data["rate_matrix"])
    assert matrix.shape == (6, 6)
    np.testing.assert_allclose(matrix.sum(axis=0), np.zeros(6), atol=1e-12)
    assert data["metastate_groups"] == METASTATE_GROUPS

    expected_epr = get_true_EPR_for_parallel_tracks(**PARALLEL_TRACKS_DEFAULTS)
    assert data["true_epr"] == pytest.approx(expected_epr)