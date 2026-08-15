"""Voice query REST and WebSocket streaming endpoints."""

import json
import uuid
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from app.config import Settings, get_settings
from app.schemas.response import QueryResponse
from app.services.rag_service import RAGService, get_rag_service
from app.services.stt_service import (
    SarvamSTTService,
    STTConfigurationError,
    STTServiceError,
    STTTimeoutError,
    get_stt_service,
)
from app.utils.latency import LatencyTracker
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/voice")


@router.post("/query", response_model=QueryResponse)
async def process_voice_query(
    request: Request,
    file: UploadFile = File(..., description="Audio file in WAV, PCM, MP3, or WebM format"),
    language_hint: Optional[str] = Form(default="en-IN", description="BCP-47 language hint (e.g. en-IN, hi-IN)"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
    stt_service: SarvamSTTService = Depends(get_stt_service),
    rag_service: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Accept an uploaded audio file, transcribe via Sarvam STT, and return RAG response."""
    request_id = x_request_id or getattr(request.state, "request_id", str(uuid.uuid4()))
    tracker = LatencyTracker()

    logger.info(
        "Received voice query request",
        extra={
            "request_id": request_id,
            "endpoint": "/api/voice/query",
            "audio_filename": file.filename,
            "content_type": file.content_type,
        },
    )

    # Check Sarvam configuration
    if not settings.has_sarvam_key:
        logger.warning("Sarvam API key is not configured for voice processing", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sarvam STT service is not configured. Please set SARVAM_API_KEY.",
        )

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )

        # Transcribe with Sarvam STT and track real STT latency
        with tracker.measure("stt"):
            transcript, stt_latency, _ = await stt_service.transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                filename=file.filename or "voice_query.wav",
                language_code=language_hint,
            )

        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not detect clear speech in the provided audio.",
            )

        logger.info(
            "Audio transcribed successfully",
            extra={
                "request_id": request_id,
                "transcript": transcript,
                "stt_latency_ms": tracker.get_stage_latency("stt"),
            },
        )

        detected_lang = language_hint.split("-")[0] if language_hint else "en"

        # Execute full RAG pipeline with transcript
        response = await rag_service.execute_rag(
            query=transcript,
            request_id=request_id,
            language=detected_lang,
            transcript=transcript,
            tracker=tracker,
        )

        return response

    except STTConfigurationError as e:
        logger.error("STT Configuration Error: %s", str(e), extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except STTTimeoutError as e:
        logger.error("STT Timeout Error: %s", str(e), extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e))
    except STTServiceError as e:
        logger.error("STT Service Error: %s", str(e), extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.websocket("/ws")
async def voice_streaming_websocket(
    websocket: WebSocket,
    rag_service: RAGService = Depends(get_rag_service),
) -> None:
    """WebSocket endpoint for real-time bidirectional audio streaming and transcription."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket client connected", extra={"request_id": session_id})

    try:
        await websocket.send_text(
            json.dumps({
                "type": "connection_ack",
                "session_id": session_id,
                "message": "WebSocket connected for real-time voice streaming.",
                "target_sample_rate": 16000,
                "channels": 1,
            })
        )

        accumulated_audio = bytearray()

        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_chunk = message["bytes"]
                accumulated_audio.extend(audio_chunk)
                await websocket.send_text(
                    json.dumps({
                        "type": "audio_ack",
                        "session_id": session_id,
                        "bytes_received": len(audio_chunk),
                        "total_bytes": len(accumulated_audio),
                    })
                )
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type", "ping")

                    if msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "session_id": session_id}))

                    elif msg_type == "query_text":
                        # Support direct text query over websocket
                        q_text = payload.get("query", "")
                        q_lang = payload.get("language", "en")
                        rag_res = await rag_service.execute_rag(
                            query=q_text,
                            request_id=session_id,
                            language=q_lang,
                        )
                        await websocket.send_text(
                            json.dumps({
                                "type": "rag_response",
                                "session_id": session_id,
                                "data": rag_res.model_dump(),
                            })
                        )

                    elif msg_type == "end_of_stream":
                        await websocket.send_text(
                            json.dumps({
                                "type": "final_transcript",
                                "session_id": session_id,
                                "status": "COMPLETED",
                                "message": "Stream ended. Audio frames received.",
                                "total_bytes_received": len(accumulated_audio),
                            })
                        )
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid JSON payload."})
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", extra={"request_id": session_id})
    except Exception as e:
        logger.error("WebSocket unexpected error: %s", str(e), extra={"request_id": session_id})
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
