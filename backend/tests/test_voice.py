"""Tests for voice query endpoint and WebSocket streaming."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.services.stt_service import get_stt_service


def test_post_voice_query_success(client: TestClient) -> None:
    """Test POST /api/voice/query with mocked STT service."""
    mock_stt = AsyncMock()
    mock_stt.transcribe_audio_bytes.return_value = (
        "What is machine learning?",
        45.2,
        {"status": "success"},
    )

    client.app.dependency_overrides[get_stt_service] = lambda: mock_stt

    audio_content = b"RIFF....WAVEfmt ....data...."
    files = {"file": ("test_voice.wav", audio_content, "audio/wav")}
    data = {"language_hint": "en-IN"}

    response = client.post("/api/voice/query", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["transcript"] == "What is machine learning?"
    assert res_data["query"] == "What is machine learning?"
    assert res_data["status"] in ("SUCCESS", "REFUSED")
    assert res_data["latency_breakdown"]["stt"] >= 0

    client.app.dependency_overrides.pop(get_stt_service, None)


def test_post_voice_query_unconfigured_sarvam(client: TestClient) -> None:
    """Test POST /api/voice/query returns 503 if Sarvam API key is missing."""
    with patch("app.config.Settings.has_sarvam_key", False):
        audio_content = b"fake-audio-bytes"
        files = {"file": ("test.wav", audio_content, "audio/wav")}
        response = client.post("/api/voice/query", files=files)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "ERROR"
        assert "Sarvam STT service is not configured" in data["error"]["message"]


def test_voice_websocket_handshake_and_ping(client: TestClient) -> None:
    """Test WebSocket /api/voice/ws connection and ping/pong flow."""
    with client.websocket_connect("/api/voice/ws") as websocket:
        # Receive connection acknowledgement
        ack_message = websocket.receive_json()
        assert ack_message["type"] == "connection_ack"
        assert "session_id" in ack_message

        # Send ping
        websocket.send_json({"type": "ping"})
        pong_message = websocket.receive_json()
        assert pong_message["type"] == "pong"

        # Send audio bytes chunk
        websocket.send_bytes(b"\x00\x01\x02\x03" * 100)
        audio_ack = websocket.receive_json()
        assert audio_ack["type"] == "audio_ack"
        assert audio_ack["bytes_received"] == 400
