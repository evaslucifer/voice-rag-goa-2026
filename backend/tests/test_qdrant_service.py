"""Tests for Qdrant service wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.qdrant_service import (
    QdrantConfigurationError,
    QdrantConnectionError,
    QdrantService,
    get_qdrant_service,
)


def test_qdrant_service_singleton() -> None:
    """Test get_qdrant_service returns singleton instance."""
    s1 = get_qdrant_service()
    s2 = get_qdrant_service()
    assert s1 is s2


@pytest.mark.asyncio
async def test_qdrant_check_connection_success() -> None:
    """Test successful connection check to Qdrant."""
    service = QdrantService(url="http://localhost:6333")
    mock_client = AsyncMock()
    mock_client.get_collections.return_value = MagicMock()

    with patch.object(service, "_get_client", return_value=mock_client):
        connected = await service.check_connection()
        assert connected is True


@pytest.mark.asyncio
async def test_qdrant_check_connection_failure() -> None:
    """Test connection check failure gracefully returns False."""
    service = QdrantService(url="http://invalid-host:6333")
    mock_client = AsyncMock()
    mock_client.get_collections.side_effect = Exception("Connection refused")

    with patch.object(service, "_get_client", return_value=mock_client):
        connected = await service.check_connection()
        assert connected is False


@pytest.mark.asyncio
async def test_qdrant_collection_exists() -> None:
    """Test checking collection existence."""
    service = QdrantService(url="http://localhost:6333")
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = True

    with patch.object(service, "_get_client", return_value=mock_client):
        exists = await service.collection_exists("msmarco_xi_bge_small")
        assert exists is True


@pytest.mark.asyncio
async def test_qdrant_search_vectors() -> None:
    """Test vector similarity search."""
    service = QdrantService(url="http://localhost:6333")
    mock_client = AsyncMock()
    mock_hit = MagicMock()
    mock_hit.id = "doc-123"
    mock_hit.score = 0.89
    mock_hit.payload = {"text": "Passage text content"}
    mock_client.query_points.return_value = MagicMock(points=[mock_hit])

    with patch.object(service, "_get_client", return_value=mock_client):
        results = await service.search_vectors(query_vector=[0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc-123"
        assert results[0]["score"] == 0.89
        assert results[0]["payload"]["text"] == "Passage text content"
