"""Tests for Sarvam STT service."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from app.services.stt_service import (
    SarvamSTTService,
    STTConfigurationError,
    STTServiceError,
    STTTimeoutError,
    get_stt_service,
)


def test_stt_service_singleton() -> None:
    """Test get_stt_service returns singleton instance."""
    s1 = get_stt_service()
    s2 = get_stt_service()
    assert s1 is s2


def test_stt_validate_missing_configuration() -> None:
    """Test validate_configuration raises error when key is absent."""
    service = SarvamSTTService(api_key="")
    with pytest.raises(STTConfigurationError):
        service.validate_configuration()

    service_placeholder = SarvamSTTService(api_key="your_sarvam_api_key_here")
    with pytest.raises(STTConfigurationError):
        service_placeholder.validate_configuration()


@pytest.mark.asyncio
async def test_stt_transcribe_audio_empty_bytes() -> None:
    """Test empty audio payload raises STTServiceError."""
    service = SarvamSTTService(api_key="sk_valid_test_key")
    with pytest.raises(STTServiceError):
        await service.transcribe_audio_bytes(b"")


@pytest.mark.asyncio
async def test_stt_transcribe_audio_success() -> None:
    """Test successful audio transcription response."""
    service = SarvamSTTService(api_key="sk_valid_test_key")

    mock_response = httpx.Response(
        status_code=200,
        json={"transcript": "Hello world from Sarvam STT", "language_code": "en-IN"},
        request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        transcript, latency, meta = await service.transcribe_audio_bytes(b"dummy_wav_bytes")

        assert transcript == "Hello world from Sarvam STT"
        assert latency >= 0
        assert meta["language_code"] == "en-IN"


@pytest.mark.asyncio
async def test_stt_transcribe_timeout() -> None:
    """Test timeout during Sarvam request raises STTTimeoutError."""
    service = SarvamSTTService(api_key="sk_valid_test_key")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")
        with pytest.raises(STTTimeoutError):
            await service.transcribe_audio_bytes(b"dummy_wav_bytes")
