"""Response schemas conforming to the frozen API contract."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.utils.latency import LatencyBreakdown


class HealthResponse(BaseModel):
    """Payload returned by GET /api/health."""

    status: str = Field(default="ok", description="Service health status", examples=["ok"])
    version: Optional[str] = Field(default=None, description="Backend application version")
    services: Optional[Dict[str, str]] = Field(default=None, description="Status of connected external services")


class CitationItem(BaseModel):
    """Citation or retrieved passage metadata."""

    id: str = Field(..., description="Unique document or chunk ID")
    text: str = Field(..., description="Retrieved passage snippet or text")
    score: float = Field(default=0.0, description="Similarity or relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")


class QueryResponse(BaseModel):
    """Frozen API contract response schema for RAG queries."""

    request_id: str = Field(..., description="Unique UUID for tracing and logging")
    transcript: Optional[str] = Field(default=None, description="Speech-to-text transcript if voice query, else null")
    query: str = Field(..., description="Processed user query")
    language: str = Field(default="en", description="Detected or requested language code")
    answer: str = Field(..., description="Synthesized response from LLM / pipeline")
    confidence_score: float = Field(default=0.0, description="Overall confidence or groundedness score (0.0 - 1.0)")
    citations: List[CitationItem] = Field(default_factory=list, description="List of cited supporting passages")
    status: str = Field(default="SUCCESS", description="Execution status: SUCCESS, PARTIAL, or ERROR")
    latency_breakdown: LatencyBreakdown = Field(
        default_factory=LatencyBreakdown,
        description="Measured latency breakdown in milliseconds"
    )


class ErrorDetail(BaseModel):
    """Structured error payload for consistent client error handling."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional diagnostic details")


class ErrorResponse(BaseModel):
    """Consistent error envelope returned when a request fails."""

    request_id: str = Field(..., description="Request ID associated with the failed request")
    status: str = Field(default="ERROR", description="Failure indicator")
    error: ErrorDetail = Field(..., description="Error specifics")
