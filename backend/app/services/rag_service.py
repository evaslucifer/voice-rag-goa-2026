"""Master RAG orchestrator coordinating embedding, retrieval, guardrails, and LLM synthesis."""

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional
from app.config import get_settings
from app.schemas.response import CitationItem, QueryResponse
from app.services.cache_service import CacheService, get_cache_service
from app.services.guardrail_service import GuardrailService, get_guardrail_service
from app.services.llm_service import LLMService, get_llm_service
from app.services.retrieval_service import RetrievalService, get_retrieval_service
from app.utils.latency import LatencyBreakdown, LatencyTracker
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RAGService:
    """End-to-end RAG orchestrator with sub-200ms P50 latency optimization."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_service: Optional[LLMService] = None,
        guardrail_service: Optional[GuardrailService] = None,
        cache_service: Optional[CacheService] = None,
    ) -> None:
        self.settings = get_settings()
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.llm_service = llm_service or get_llm_service()
        self.guardrail_service = guardrail_service or get_guardrail_service()
        self.cache_service = cache_service or get_cache_service()

    async def execute_rag(
        self,
        query: str,
        request_id: str,
        language: str = "en",
        transcript: Optional[str] = None,
        tracker: Optional[LatencyTracker] = None,
    ) -> QueryResponse:
        """Execute the full RAG pipeline and return a contract-compliant QueryResponse."""
        if tracker is None:
            tracker = LatencyTracker()

        clean_query = query.strip()
        cache_key = f"rag:{language}:{clean_query.lower()}"

        # 1. Cache Check
        cached_data = self.cache_service.get(cache_key)
        if cached_data is not None:
            logger.info("Cache hit for query: '%s'", clean_query[:50], extra={"request_id": request_id})
            return QueryResponse(
                request_id=request_id,
                transcript=transcript,
                query=clean_query,
                language=language,
                answer=cached_data.get("answer", ""),
                confidence_score=cached_data.get("confidence_score", 1.0),
                citations=cached_data.get("citations", []),
                status="SUCCESS",
                latency_breakdown=tracker.to_breakdown(),
            )

        # 2. Pre-Retrieval Guardrail
        with tracker.measure("guardrail"):
            pre_check = self.guardrail_service.check_pre_retrieval(clean_query)

        if not pre_check.passed:
            logger.warning(
                "Pre-retrieval guardrail rejected query: %s (action=%s)",
                pre_check.reason,
                pre_check.action,
                extra={"request_id": request_id},
            )
            return QueryResponse(
                request_id=request_id,
                transcript=transcript,
                query=clean_query,
                language=language,
                answer=pre_check.safe_response or "Invalid query.",
                confidence_score=0.0,
                citations=[],
                status="REFUSED",
                latency_breakdown=tracker.to_breakdown(),
            )

        # 3. Retrieval (Embedding + Vector Search)
        retrieval_res = await self.retrieval_service.retrieve(query=clean_query)
        tracker.record("embedding", retrieval_res.embedding_latency_ms)
        tracker.record("retrieval", retrieval_res.retrieval_latency_ms)

        # 4. Post-Retrieval Relevance Guardrail
        with tracker.measure("guardrail"):
            rel_check = self.guardrail_service.check_retrieval_relevance(
                query=clean_query,
                retrieved_chunks=retrieval_res.chunks,
                top_score=retrieval_res.top_score,
            )

        if not rel_check.passed:
            logger.info("Abstaining due to low retrieval relevance", extra={"request_id": request_id})
            return QueryResponse(
                request_id=request_id,
                transcript=transcript,
                query=clean_query,
                language=language,
                answer=rel_check.safe_response or "I could not find sufficient grounded evidence in MSMARCO-XI to answer accurately.",
                confidence_score=0.0,
                citations=[],
                status="REFUSED",
                latency_breakdown=tracker.to_breakdown(),
            )

        # 5. LLM Answer Generation
        llm_res = await self.llm_service.generate_answer(
            query=clean_query,
            retrieved_chunks=retrieval_res.chunks,
            context_text=retrieval_res.context_text,
            language=language,
            request_id=request_id,
        )
        tracker.record("llm_ttft", llm_res.ttft_ms)

        # 6. Post-Generation Grounding Guardrail
        with tracker.measure("guardrail"):
            grounding_check = self.guardrail_service.check_grounding(
                query=clean_query,
                answer=llm_res.answer,
                context_text=retrieval_res.context_text,
                confidence_score=llm_res.confidence_score,
            )

        final_answer = llm_res.answer
        final_confidence = llm_res.confidence_score
        final_citations = llm_res.citations
        final_status = "SUCCESS"

        if not grounding_check.passed:
            logger.warning(
                "Grounding check failed: %s (action=%s)",
                grounding_check.reason,
                grounding_check.action,
                extra={"request_id": request_id},
            )
            final_answer = grounding_check.safe_response or "I could not find sufficient grounded evidence in MSMARCO-XI to answer accurately."
            final_confidence = 0.0
            final_citations = []
            final_status = "REFUSED"

        # 7. Store in Cache if successful and grounded
        if final_confidence >= 0.65 and final_status == "SUCCESS":
            self.cache_service.set(
                cache_key,
                {
                    "answer": final_answer,
                    "confidence_score": final_confidence,
                    "citations": [c.model_dump() for c in final_citations],
                },
                ttl_seconds=300.0,
            )

        logger.info(
            "RAG pipeline completed successfully in %.2f ms (status=%s)",
            tracker.get_total_latency(),
            final_status,
            extra={
                "request_id": request_id,
                "confidence": final_confidence,
                "citations_count": len(final_citations),
                "status": final_status,
                "latency_breakdown": tracker.to_dict(),
            },
        )

        return QueryResponse(
            request_id=request_id,
            transcript=transcript,
            query=clean_query,
            language=language,
            answer=final_answer,
            confidence_score=final_confidence,
            citations=final_citations,
            status=final_status,
            latency_breakdown=tracker.to_breakdown(),
        )


@lru_cache()
def get_rag_service() -> RAGService:
    """Return singleton instance of RAGService."""
    return RAGService()
