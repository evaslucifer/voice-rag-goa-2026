"""Tests for FastEmbed embedding service."""

from unittest.mock import MagicMock, patch
import pytest
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    get_embedding_service,
)


def test_embedding_service_singleton() -> None:
    """Test get_embedding_service returns singleton instance."""
    s1 = get_embedding_service()
    s2 = get_embedding_service()
    assert s1 is s2


@pytest.mark.asyncio
async def test_embed_query_empty_error() -> None:
    """Test empty query raises EmbeddingServiceError."""
    service = EmbeddingService()
    with pytest.raises(EmbeddingServiceError):
        await service.embed_query("")

    with pytest.raises(EmbeddingServiceError):
        await service.embed_query("   ")


@pytest.mark.asyncio
async def test_embed_query_mocked_model() -> None:
    """Test query embedding with mocked FastEmbed backend."""
    service = EmbeddingService()
    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

    with patch.object(service, "_get_model", return_value=mock_model):
        vector = await service.embed_query("What is MSMARCO-XI?")
        assert isinstance(vector, list)
        assert len(vector) == 4
        assert vector == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_embed_documents_mocked_model() -> None:
    """Test batch document embedding with mocked FastEmbed backend."""
    service = EmbeddingService()
    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

    with patch.object(service, "_get_model", return_value=mock_model):
        vectors = await service.embed_documents(["Doc 1", "Doc 2"])
        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2]
        assert vectors[1] == [0.3, 0.4]
