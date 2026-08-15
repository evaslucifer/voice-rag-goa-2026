"""End-to-end integration tests for RAG pipeline orchestration."""

from unittest.mock import AsyncMock
import pytest
from app.schemas.response import QueryResponse
from app.services.cache_service import CacheService
from app.services.guardrail_service import GuardrailService
from app.services.llm_service import LLMResponse, LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalResult, RetrievalService, RetrievedChunk
from app.utils.latency import LatencyTracker


@pytest.mark.asyncio
async def test_e2e_rag_pipeline_success() -> None:
    """Test full E2E RAG orchestration with valid retrieval and synthesis."""
    mock_retrieval = AsyncMock(spec=RetrievalService)
    chunk = RetrievedChunk(
        document_id="msmarco_1",
        chunk_id="msmarco_1_chk_0",
        text="The Manhattan Project was established during World War II.",
        score=0.92,
        language="en",
    )
    mock_retrieval.retrieve.return_value = RetrievalResult(
        query="What was the Manhattan Project?",
        chunks=[chunk],
        context_text=chunk.text,
        top_score=0.92,
        is_sufficient=True,
        embedding_latency_ms=8.0,
        retrieval_latency_ms=12.0,
    )

    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.generate_answer.return_value = LLMResponse(
        answer="The Manhattan Project was an undertaking during World War II.",
        confidence_score=0.95,
        citations=[],
        ttft_ms=65.0,
    )

    rag = RAGService(
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        guardrail_service=GuardrailService(score_threshold=0.5),
        cache_service=CacheService(),
    )

    response: QueryResponse = await rag.execute_rag(
        query="What was the Manhattan Project?",
        request_id="req-test-101",
        language="en",
    )

    assert response.status == "SUCCESS"
    assert response.request_id == "req-test-101"
    assert "World War II" in response.answer
    assert response.confidence_score == 0.95
    assert response.latency_breakdown.embedding == 8.0
    assert response.latency_breakdown.retrieval == 12.0
    assert response.latency_breakdown.llm_ttft == 65.0
    assert response.latency_breakdown.total > 0


@pytest.mark.asyncio
async def test_e2e_rag_pipeline_abstain_on_irrelevant() -> None:
    """Test pipeline abstains safely when retrieval score is below threshold."""
    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResult(
        query="Bake chocolate cookies",
        chunks=[],
        context_text="",
        top_score=0.20,
        is_sufficient=False,
        embedding_latency_ms=6.0,
        retrieval_latency_ms=10.0,
    )

    rag = RAGService(
        retrieval_service=mock_retrieval,
        llm_service=AsyncMock(spec=LLMService),
        guardrail_service=GuardrailService(score_threshold=0.60),
        cache_service=CacheService(),
    )

    response: QueryResponse = await rag.execute_rag(
        query="Bake chocolate cookies",
        request_id="req-test-102",
        language="en",
    )

    assert response.status in ("SUCCESS", "REFUSED")
    assert response.confidence_score == 0.0
    assert "No sufficiently relevant information" in response.answer or "could not find sufficient" in response.answer.lower()


@pytest.mark.asyncio
async def test_e2e_rag_caching() -> None:
    """Test subsequent identical queries hit the cache."""
    cache = CacheService()
    cache.set(
        "rag:en:what is fastembed?",
        {"answer": "FastEmbed is a lightweight ONNX library.", "confidence_score": 0.98, "citations": []},
    )

    rag = RAGService(cache_service=cache)
    response = await rag.execute_rag(
        query="What is FastEmbed?",
        request_id="req-cache-1",
        language="en",
    )

    assert response.answer == "FastEmbed is a lightweight ONNX library."
    assert response.confidence_score == 0.98
