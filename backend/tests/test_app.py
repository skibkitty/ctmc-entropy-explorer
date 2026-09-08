"""Tests for the FastAPI application."""

import importlib

from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint returns cold status before warm-up."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "cold"}


def test_cors_allows_dev_origin():
    """Requests from a configured dev origin get the CORS header."""
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin():
    """Requests from an origin not in the list get no CORS header."""
    response = client.get(
        "/api/health",
        headers={"Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_env_override(monkeypatch):
    """CORS_ORIGINS env var replaces the dev defaults."""
    monkeypatch.setenv("CORS_ORIGINS", "https://custom.example")
    importlib.reload(main_module)
    reloaded_client = TestClient(main_module.app)

    response = reloaded_client.get(
        "/api/health",
        headers={"Origin": "https://custom.example"},
    )
    assert response.headers.get("access-control-allow-origin") == "https://custom.example"

    monkeypatch.delenv("CORS_ORIGINS")
    importlib.reload(main_module)