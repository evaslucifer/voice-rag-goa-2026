"""Text query RAG endpoint conforming to frozen API contract."""

import uuid
from fastapi import APIRouter, Depends, Header, Request
from app.schemas.query import TextQueryRequest
from app.schemas.response import QueryResponse
from app.services.rag_service import RAGService, get_rag_service
from app.utils.latency import LatencyTracker
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_text_query(
    payload: TextQueryRequest,
    request: Request,
    x_request_id: str = Header(default=None, alias="X-Request-ID"),
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Process a natural language text query through the real RAG pipeline."""
    request_id = x_request_id or getattr(request.state, "request_id", str(uuid.uuid4()))
    tracker = LatencyTracker()

    logger.info(
        "Received query request",
        extra={
            "request_id": request_id,
            "endpoint": "/api/query",
            "stage": "received",
            "query": payload.query,
            "language": payload.language,
        },
    )

    detected_language = payload.language or "en"

    # Execute full RAG pipeline
    response = await rag_service.execute_rag(
        query=payload.query,
        request_id=request_id,
        language=detected_language,
        transcript=None,
        tracker=tracker,
    )

    return response
