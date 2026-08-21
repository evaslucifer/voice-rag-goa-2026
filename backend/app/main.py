"""FastAPI Application Main Entrypoint."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import api_router
from app.config import get_settings
from app.schemas.response import ErrorDetail, ErrorResponse
from app.services.qdrant_service import get_qdrant_service
from app.services.embedding_service import get_embedding_service
from app.utils.logging import configure_logging, get_logger

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown events."""
    settings = get_settings()
    configure_logging(log_level=settings.LOG_LEVEL, json_format=settings.is_production)

    logger.info(
        "Starting Voice-Enabled Multilingual RAG Backend",
        extra={
            "stage": "startup",
            "environment": settings.ENVIRONMENT,
            "port": settings.PORT,
            "host": settings.HOST,
        },
    )

    # Initialize logging and settings
    logger.info("Voice-Enabled Multilingual RAG Backend started successfully (services ready for lazy-loading)")
    yield

    # Shutdown logic
    logger.info("Initiating application shutdown", extra={"stage": "shutdown"})
    try:
        qdrant_service = get_qdrant_service()
        await qdrant_service.close()
    except Exception as e:
        logger.warning("Error closing Qdrant connection during shutdown: %s", str(e))
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Voice-Enabled Multilingual RAG API",
        description="Sub-200ms Voice-Enabled Multilingual Retrieval-Augmented Generation API using MSMARCO-XI, Sarvam STT, FastEmbed BGE-Small, Qdrant, and Groq.",
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # 1. Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.effective_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # 2. Request ID & Request Logging Middleware
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}",
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "latency_ms": duration_ms,
                    "status": "ERROR",
                },
                exc_info=True,
            )
            # Central error envelope for unhandled exceptions
            error_response = ErrorResponse(
                request_id=request_id,
                status="ERROR",
                error=ErrorDetail(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected internal error occurred. Please try again later.",
                ),
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump(),
                headers={"X-Request-ID": request_id},
            )

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms}ms"

        logger.info(
            f"Completed request: {request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "status_code": response.status_code,
                "latency_ms": duration_ms,
            },
        )
        return response

    # 3. Centralized Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        error_response = ErrorResponse(
            request_id=request_id,
            status="ERROR",
            error=ErrorDetail(
                code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        error_messages = []
        for err in exc.errors():
            loc = " -> ".join([str(l) for l in err.get("loc", [])])
            error_messages.append(f"{loc}: {err.get('msg')}")
        error_summary = "; ".join(error_messages)

        error_response = ErrorResponse(
            request_id=request_id,
            status="ERROR",
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message=f"Request validation failed: {error_summary}",
                details={"errors": exc.errors()},
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    # 4. Register API routes
    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {
            "name": "Voice-Enabled Multilingual RAG API",
            "status": "online",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_application()
