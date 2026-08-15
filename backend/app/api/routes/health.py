"""Health check endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from app.config import Settings, get_settings
from app.schemas.response import HealthResponse
from app.services.qdrant_service import QdrantService, get_qdrant_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    detailed: bool = Query(default=False, description="Whether to include detailed service status"),
    settings: Settings = Depends(get_settings),
    qdrant_service: QdrantService = Depends(get_qdrant_service),
) -> HealthResponse:
    """Check health status of backend and external services."""
    if not detailed:
        return HealthResponse(status="ok")

    services = {
        "sarvam_configured": "configured" if settings.has_sarvam_key else "unconfigured",
        "groq_configured": "configured" if settings.has_groq_key else "unconfigured",
        "gemini_configured": "configured" if settings.has_gemini_key else "unconfigured",
    }

    # Lightweight check for Qdrant
    try:
        is_qdrant_connected = await qdrant_service.check_connection()
        services["qdrant"] = "connected" if is_qdrant_connected else "unreachable"
    except Exception:
        services["qdrant"] = "unreachable"

    return HealthResponse(
        status="ok",
        version="0.1.0",
        services=services,
    )
