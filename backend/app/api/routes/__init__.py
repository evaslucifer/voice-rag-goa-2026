"""API route routers registration."""

from fastapi import APIRouter
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.voice import router as voice_router

api_router = APIRouter(prefix="/api")

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(query_router, tags=["Query"])
api_router.include_router(voice_router, tags=["Voice"])

__all__ = ["api_router"]
