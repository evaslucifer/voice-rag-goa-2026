"""Tests for Parent-Child Hierarchical Chunking and Retrieval expansion."""

import pytest
from app.services.retrieval_service import RetrievalService, RetrievedChunk
from ingestion.chunker import ParentChildChunker, get_chunker


def test_parent_child_chunker_basic() -> None:
    """Test ParentChildChunker partitions long document into parent and granular children."""
    chunker = ParentChildChunker(parent_size=100, child_size=30, child_overlap=5)

    # 150-word document
    words = [f"word{i}" for i in range(150)]
    doc_text = " ".join(words)

    chunks = chunker.chunk(doc_text, document_id="doc_101", metadata={"language": "en", "title": "Test Doc"})

    assert len(chunks) > 1
    # Check first child
    first_child = chunks[0]
    assert "doc_101_parent_0_child_0" == first_child.chunk_id
    assert first_child.metadata["parent_id"] == "doc_101_parent_0"
    assert "parent_text" in first_child.metadata
    assert first_child.metadata["chunk_strategy"] == "parent_child"
    assert first_child.word_count <= 30
    assert len(first_child.metadata["parent_text"].split()) <= 100


def test_get_chunker_parent_child_factory() -> None:
    """Test factory resolution for parent_child and hierarchical strings."""
    c1 = get_chunker("parent_child")
    assert isinstance(c1, ParentChildChunker)
    c2 = get_chunker("hierarchical")
    assert isinstance(c2, ParentChildChunker)


@pytest.mark.asyncio
async def test_parent_child_context_expansion_in_retrieval() -> None:
    """Test that RetrievalService extracts parent context when child hits are returned."""
    retrieval_service = RetrievalService()

    # Mock Qdrant returning a child hit with parent_text metadata
    mock_hits = [
        {
            "id": "doc_101_parent_0_child_1",
            "score": 0.88,
            "payload": {
                "document_id": "doc_101",
                "chunk_id": "doc_101_parent_0_child_1",
                "text": "This is the child snippet about quantum physics.",
                "language": "en",
                "metadata": {
                    "parent_id": "doc_101_parent_0",
                    "parent_text": "This is the full comprehensive parent document describing modern quantum physics, quantum entanglement, and computational theory in complete detail.",
                    "chunk_strategy": "parent_child",
                },
            },
        }
    ]

    from unittest.mock import AsyncMock, patch
    with patch.object(retrieval_service.embedding_service, "embed_query", new_callable=AsyncMock) as mock_embed, \
         patch.object(retrieval_service.qdrant_service, "search_vectors", new_callable=AsyncMock) as mock_search:
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = mock_hits

        res = await retrieval_service.retrieve("quantum physics", score_threshold=0.5)

        assert res.is_sufficient is True
        assert len(res.chunks) == 1
        assert res.chunks[0].chunk_id == "doc_101_parent_0_child_1"
        # Context text must contain the expanded parent text
        assert "full comprehensive parent document" in res.context_text
        assert "Score: 0.880" in res.context_text
