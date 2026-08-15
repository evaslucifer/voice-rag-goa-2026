"""Tests for RetrievalService."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from app.services.retrieval_service import RetrievalResult, RetrievalService, RetrievedChunk


@pytest.mark.asyncio
async def test_retrieval_service_success() -> None:
    """Test full retrieval flow with mock embedding and search."""
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1] * 384

    mock_qdrant = AsyncMock()
    mock_qdrant.search_vectors.return_value = [
        {
            "id": "point_1",
            "score": 0.88,
            "payload": {
                "document_id": "doc_100",
                "chunk_id": "doc_100_chk_0",
                "text": "The Manhattan Project was led by Oppenheimer.",
                "language": "en",
                "metadata": {"title": "History"},
            },
        }
    ]

    service = RetrievalService(
        embedding_service=mock_embed,
        qdrant_service=mock_qdrant,
        top_k=3,
        score_threshold=0.50,
    )

    res: RetrievalResult = await service.retrieve("Who led the Manhattan Project?")

    assert res.is_sufficient is True
    assert len(res.chunks) == 1
    assert res.top_score == 0.88
    assert res.chunks[0].chunk_id == "doc_100_chk_0"
    assert "Oppenheimer" in res.context_text
    assert res.embedding_latency_ms >= 0


@pytest.mark.asyncio
async def test_retrieval_service_below_threshold() -> None:
    """Test retrieval marks is_sufficient=False when scores are below threshold."""
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1] * 384

    mock_qdrant = AsyncMock()
    mock_qdrant.search_vectors.return_value = [
        {
            "id": "point_2",
            "score": 0.32,  # Below threshold 0.60
            "payload": {"text": "Unrelated topic", "chunk_id": "c2"},
        }
    ]

    service = RetrievalService(
        embedding_service=mock_embed,
        qdrant_service=mock_qdrant,
        top_k=3,
        score_threshold=0.60,
    )

    res: RetrievalResult = await service.retrieve("Query with low similarity")
    assert res.is_sufficient is False
    assert res.top_score == 0.32
