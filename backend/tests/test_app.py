"""Tests for the FastAPI application."""

import os

import pytest
from fastapi.testclient import TestClient

from backend.main import app, parse_cors_origins

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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://custom.example", ["https://custom.example"]),
        ("https://a.example,https://b.example", ["https://a.example", "https://b.example"]),
        (" https://a.example , https://b.example ", ["https://a.example", "https://b.example"]),
        ("https://a.example,,https://b.example,", ["https://a.example", "https://b.example"]),
        ("", []),
    ],
)
def test_parse_cors_origins(raw, expected):
    """Comma-separated origins are trimmed; empty entries are dropped."""
    assert parse_cors_origins(raw) == expected


def test_cors_env_override(monkeypatch):
    """CORS_ORIGINS env var replaces the dev defaults (via the pure parser)."""
    monkeypatch.setenv("CORS_ORIGINS", "https://custom.example")
    origins = parse_cors_origins(os.environ["CORS_ORIGINS"])
    assert origins == ["https://custom.example"]