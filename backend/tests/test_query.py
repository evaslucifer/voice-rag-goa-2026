"""Tests for text query endpoint and schema compliance."""

from fastapi.testclient import TestClient


def test_post_query_success(client: TestClient) -> None:
    """Test POST /api/query returns frozen contract compliant schema."""
    payload = {"query": "What was the Manhattan Project?"}
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate exact contract keys
    assert "request_id" in data
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0

    assert data["transcript"] is None
    assert data["query"] == "What was the Manhattan Project?"
    assert data["language"] == "en"
    assert "answer" in data
    assert isinstance(data["confidence_score"], (int, float))
    assert isinstance(data["citations"], list)
    assert data["status"] == "SUCCESS"

    # Validate latency breakdown
    assert "latency_breakdown" in data
    breakdown = data["latency_breakdown"]
    assert "stt" in breakdown
    assert "embedding" in breakdown
    assert "retrieval" in breakdown
    assert "guardrail" in breakdown
    assert "llm_ttft" in breakdown
    assert "total" in breakdown
    assert breakdown["total"] >= 0

    # Validate response headers
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == data["request_id"]
    assert "X-Process-Time" in response.headers


def test_post_query_with_custom_request_id(client: TestClient) -> None:
    """Test custom X-Request-ID header is propagated."""
    custom_id = "custom-req-uuid-1234"
    payload = {"query": "Tell me about quantum computing.", "language": "hi"}
    response = client.post("/api/query", json=payload, headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == custom_id
    assert data["language"] == "hi"
    assert response.headers["X-Request-ID"] == custom_id


def test_post_query_validation_error(client: TestClient) -> None:
    """Test invalid or empty query payload returns structured error."""
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "ERROR"
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "request_id" in data
