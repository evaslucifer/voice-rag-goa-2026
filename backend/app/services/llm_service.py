"""LLM Service wrapping the Model Harness with Groq, Gemini fallback, and Tenacity retries."""

from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas.response import CitationItem
from app.services.harness import HarnessInput, HarnessOutput, ModelHarness, get_model_harness
from app.services.retrieval_service import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMServiceError(Exception):
    """Base exception for LLM generation failures."""
    pass


class LLMTimeoutError(LLMServiceError):
    """Raised when LLM API request times out."""
    pass


class LLMResponse(BaseModel):
    """Structured response from LLM synthesis."""

    answer: str = Field(..., description="Synthesized grounded answer")
    confidence_score: float = Field(default=0.9, description="Confidence score between 0.0 and 1.0")
    citations: List[CitationItem] = Field(default_factory=list, description="Citations used in answer")
    ttft_ms: float = Field(default=0.0, description="Time to first token / inference response latency in ms")
    model_name: str = Field(default="openai/gpt-oss-20b", description="Model used for generation")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw LLM provider response")


class LLMService:
    """Async service managing grounded LLM synthesis via ModelHarness."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds:  Optional[float] = None,
        temperature: float = 0.1,
        harness: Optional[ModelHarness] = None,
    ) -> None:
        self.settings = get_settings()
        self.api_key = (
            api_key
            if api_key is not None
            else self.settings.GROQ_API_KEY
        )

        self.model_name = (
            model_name
            if model_name is not None
            else self.settings.LLM_MODEL
        )

        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else self.settings.LLM_TIMEOUT_SECONDS
        )

        self.temperature = temperature

        self.harness = harness or ModelHarness(
            groq_api_key=self.api_key,
            primary_model=self.model_name,
            timeout_seconds=self.timeout_seconds,
        )

    @property
    def is_live_configured(self) -> bool:
        """Check if a valid live Groq API key is present."""
        return self.harness.has_groq_credentials

    async def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        context_text: str,
        language: str = "en",
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """Execute grounded generation through the Model Harness."""
        harness_input = HarnessInput(
            query=query,
            context_text=context_text,
            retrieved_chunks=retrieved_chunks,
            language=language,
            request_id=request_id or "rag-query",
            temperature=self.temperature,
        )

        harness_output: HarnessOutput = await self.harness.execute(harness_input)

        return LLMResponse(
            answer=harness_output.answer,
            confidence_score=harness_output.confidence_score,
            citations=harness_output.citations,
            ttft_ms=harness_output.ttft_ms,
            model_name=harness_output.model_used,
            raw_response={"provider": harness_output.provider_used, "is_fallback": harness_output.is_fallback},
        )


@lru_cache()
def get_llm_service() -> LLMService:
    """Return singleton instance of LLMService."""
    return LLMService()
