"""Sarvam AI Saaras Speech-to-Text (STT) Service with REST & WebSocket Streaming."""

import asyncio
import json
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
import httpx
import websockets
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.services.stt")

SARVAM_STT_REST_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_WS_URL = "wss://api.sarvam.ai/speech-to-text-stream"


class STTServiceError(Exception):
    """Base exception for Speech-to-Text service failures."""
    pass


class STTConfigurationError(STTServiceError):
    """Raised when STT service is called without proper configuration / API key."""
    pass


class STTTimeoutError(STTServiceError):
    """Raised when an STT request times out."""
    pass


class STTStreamError(STTServiceError):
    """Raised when a WebSocket streaming transcription error occurs."""
    pass


class SarvamSTTService:
    """Async service for transcribing audio via Sarvam AI Saaras API (REST & Streaming WebSocket)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.settings = get_settings()
        self.api_key = api_key if api_key is not None else self.settings.SARVAM_API_KEY
        self.language_code = language_code or self.settings.SARVAM_STT_LANGUAGE_CODE
        self.model = model or self.settings.SARVAM_STT_MODEL
        self.timeout = timeout or self.settings.SARVAM_STT_TIMEOUT_SECONDS

    def validate_configuration(self) -> None:
        """Validate that a valid API key is present."""
        if not self.api_key or self.api_key.startswith("your_"):
            raise STTConfigurationError(
                "Sarvam API key is not configured. Please set SARVAM_API_KEY in .env or environment."
            )

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Transcribe raw audio bytes using Sarvam Saaras REST API.

        Target format: 16kHz mono PCM (WAV/raw).
        Returns: Tuple[transcript, latency_ms, raw_metadata]
        """
        self.validate_configuration()

        if not audio_bytes:
            raise STTServiceError("Audio payload cannot be empty.")

        target_lang = language_code or self.language_code
        target_model = model or self.model

        headers = {
            "api-subscription-key": self.api_key,
        }
        content_type = "audio/wav"
        fn_lower = filename.lower()
        if fn_lower.endswith(".webm") or audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
            content_type = "audio/webm"
        elif fn_lower.endswith(".mp3") or audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
            content_type = "audio/mpeg"
        elif fn_lower.endswith(".ogg") or audio_bytes.startswith(b"OggS"):
            content_type = "audio/ogg"

        files = {
            "file": (filename, audio_bytes, content_type),
        }
        data = {
            "language_code": target_lang,
            "model": target_model,
            "with_diarization": "false",
        }

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    SARVAM_STT_REST_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )

                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                if response.status_code in (401, 403):
                    raise STTConfigurationError(f"Sarvam STT authentication failed (HTTP {response.status_code}).")

                if response.status_code != 200:
                    raise STTServiceError(
                        f"Sarvam STT failed with status {response.status_code}: {response.text}"
                    )

                result_json = response.json()
                transcript = str(result_json.get("transcript", "")).strip()
                return transcript, latency_ms, result_json

        except httpx.TimeoutException as e:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.error("Sarvam STT request timed out after %.2f ms", latency_ms)
            raise STTTimeoutError(f"Sarvam STT request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.error("Sarvam STT network error: %s", str(e))
            raise STTServiceError(f"Network error connecting to Sarvam STT: {e}") from e

    async def stream_transcribe_chunks(
        self,
        audio_chunk_stream: AsyncGenerator[bytes, None],
        language_code: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming transcription using Sarvam Saaras WebSocket API."""
        self.validate_configuration()
        target_lang = language_code or self.language_code

        headers = {
            "api-subscription-key": self.api_key,
        }
        ws_url = f"{SARVAM_STT_WS_URL}?language_code={target_lang}&model={self.model}&sample_rate=16000"

        try:
            async with websockets.connect(ws_url, extra_headers=headers) as ws:
                logger.info("Connected to Sarvam Saaras WebSocket streaming server.")

                async def send_audio():
                    async for chunk in audio_chunk_stream:
                        if chunk:
                            await ws.send(chunk)
                    # Send EOF frame
                    await ws.send(json.dumps({"type": "eos"}))

                async def receive_transcripts():
                    async for message in ws:
                        try:
                            data = json.loads(message) if isinstance(message, str) else {}
                            yield data
                        except json.JSONDecodeError:
                            yield {"type": "raw", "data": str(message)}

                sender_task = asyncio.create_task(send_audio())
                async for event in receive_transcripts():
                    yield event
                await sender_task

        except websockets.exceptions.WebSocketException as e:
            logger.error("Sarvam WebSocket streaming error: %s", str(e))
            raise STTStreamError(f"Sarvam WebSocket connection error: {e}") from e


@lru_cache()
def get_stt_service() -> SarvamSTTService:
    """Return singleton instance of SarvamSTTService."""
    return SarvamSTTService()
