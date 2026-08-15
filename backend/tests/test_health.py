"""Tests for health check endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint_basic(client: TestClient) -> None:
    """Test GET /api/health returns status ok."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "services" not in data or data["services"] is None


def test_health_endpoint_detailed(client: TestClient) -> None:
    """Test GET /api/health?detailed=true includes service diagnostic info."""
    response = client.get("/api/health?detailed=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "services" in data
    assert data["services"] is not None
    assert "sarvam_configured" in data["services"]
    assert "groq_configured" in data["services"]
