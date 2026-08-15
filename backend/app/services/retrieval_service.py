"""Retrieval service managing query embedding, vector search, and context assembly."""

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from app.config import get_settings
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.qdrant_service import QdrantService, get_qdrant_service
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievedChunk(BaseModel):
    """Structured retrieved passage from vector search."""

    document_id: str = Field(..., description="Parent document ID")
    chunk_id: str = Field(..., description="Unique chunk ID")
    text: str = Field(..., description="Passage text")
    score: float = Field(..., description="Cosine similarity score")
    language: str = Field(default="en", description="Passage language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class RetrievalResult(BaseModel):
    """Container for retrieval pipeline outputs."""

    query: str
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    context_text: str = Field(default="")
    top_score: float = Field(default=0.0)
    is_sufficient: bool = Field(default=False)
    embedding_latency_ms: float = Field(default=0.0)
    retrieval_latency_ms: float = Field(default=0.0)


class RetrievalService:
    """Service orchestrating dense query embedding and Qdrant similarity search."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_service: Optional[QdrantService] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.qdrant_service = qdrant_service or get_qdrant_service()
        self.top_k = top_k or getattr(self.settings, "RETRIEVAL_TOP_K", 5)
        self.score_threshold = score_threshold or getattr(self.settings, "RETRIEVAL_SCORE_THRESHOLD", 0.55)

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> RetrievalResult:
        """Embed the query, search Qdrant, filter results, and assemble context."""
        k = top_k or self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # 1. Measure embedding latency
        t_embed_start = time.perf_counter()
        query_vector = await self.embedding_service.embed_query(query)
        embedding_latency_ms = round((time.perf_counter() - t_embed_start) * 1000.0, 2)

        # 2. Measure vector search latency
        t_search_start = time.perf_counter()
        raw_hits: List[Dict[str, Any]] = []
        try:
            raw_hits = await self.qdrant_service.search_vectors(
                query_vector=query_vector,
                top_k=k,
                collection_name=collection_name,
                score_threshold=threshold,
            )
        except Exception as e:
            logger.warning("Vector search returned empty/error: %s", str(e))
            raw_hits = []

        retrieval_latency_ms = round((time.perf_counter() - t_search_start) * 1000.0, 2)

        retrieved_chunks: List[RetrievedChunk] = []
        context_parts: List[str] = []
        top_score = 0.0

        seen_parent_ids = set()
        for idx, hit in enumerate(raw_hits):
            score = float(hit.get("score", 0.0))
            if idx == 0:
                top_score = score

            payload = hit.get("payload", {})
            chunk_id = str(payload.get("chunk_id", hit.get("id", "")))
            doc_id = str(payload.get("document_id", "doc"))
            text = str(payload.get("text", "")).strip()
            lang = str(payload.get("language", "en"))
            meta = payload.get("metadata", {}) or {}

            # Check for parent-child hierarchical chunking
            parent_id = meta.get("parent_id")
            parent_text = meta.get("parent_text")

            if text:
                chunk_obj = RetrievedChunk(
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    score=round(score, 4),
                    language=lang,
                    metadata=meta,
                )
                retrieved_chunks.append(chunk_obj)

                # If parent_text is available, present richer parent context without duplicating siblings
                if parent_text and parent_id:
                    if parent_id not in seen_parent_ids:
                        seen_parent_ids.add(parent_id)
                        context_parts.append(f"[Passage {idx + 1} | Parent: {parent_id} | Score: {score:.3f}]\n{parent_text}")
                else:
                    context_parts.append(f"[Passage {idx + 1} | ID: {chunk_id} | Score: {score:.3f}]\n{text}")

        is_sufficient = len(retrieved_chunks) > 0 and top_score >= threshold
        context_text = "\n\n".join(context_parts)

        logger.info(
            "Retrieved %d chunks (top_score=%.3f, sufficient=%s, embed_ms=%.1f, search_ms=%.1f)",
            len(retrieved_chunks),
            top_score,
            is_sufficient,
            embedding_latency_ms,
            retrieval_latency_ms,
        )

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            context_text=context_text,
            top_score=top_score,
            is_sufficient=is_sufficient,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
        )


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    """Return singleton instance of RetrievalService."""
    return RetrievalService()
