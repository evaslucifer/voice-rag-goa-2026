"""Unit tests for FastEmbed embedding and Qdrant vector indexing."""

import pytest

from data_pipeline.chunking_strategies import ChunkRecord
from data_pipeline.vector_indexer import VectorIndexer


def test_fastembed_embedding_generation(in_memory_indexer):
    """Test FastEmbed multilingual model produces 384-dim vectors for English and Indic."""
    texts = [
        "The Manhattan Project developed nuclear weapons.",
        "इसरो की स्थापना 1969 में हुई थी।",
        "मराठा साम्राज्य रायगड किल्ला.",
    ]
    vectors = in_memory_indexer.generate_embeddings(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == in_memory_indexer.dimension == 384
        assert isinstance(vec[0], float)


def test_qdrant_idempotent_collection_creation(in_memory_indexer):
    """Test collection creation is idempotent and does not fail on repeat calls."""
    in_memory_indexer.ensure_collection(recreate=False)
    # Second call should be a no-op
    in_memory_indexer.ensure_collection(recreate=False)

    collections = in_memory_indexer.client.get_collections().collections
    assert any(c.name == in_memory_indexer.collection_name for c in collections)


def test_index_and_search_chunks(in_memory_indexer, sample_chunk_records):
    """Test indexing chunks into in-memory Qdrant and retrieving relevant matches."""
    stats = in_memory_indexer.index_chunks(sample_chunk_records, recreate_collection=True)

    assert stats["status"] == "SUCCESS"
    assert stats["indexed_count"] == 3
    assert stats["dimension"] == 384

    # Search in English
    en_hits = in_memory_indexer.search("nuclear weapons Manhattan Project", top_k=2)
    assert len(en_hits) >= 1
    assert en_hits[0].document_id == "doc_en_01"
    assert en_hits[0].score > 0.30

    # Search in Hindi
    hi_hits = in_memory_indexer.search("इसरो की स्थापना कब हुई", top_k=2)
    assert len(hi_hits) >= 1
    assert hi_hits[0].document_id == "doc_hi_01"
    assert hi_hits[0].score > 0.30


def test_search_with_language_filter(in_memory_indexer, sample_chunk_records):
    """Test vector search with specific language filter."""
    in_memory_indexer.index_chunks(sample_chunk_records, recreate_collection=True)

    hi_hits = in_memory_indexer.search("space research", top_k=5, language="hi")
    for hit in hi_hits:
        assert hit.language == "hi"


def test_index_duplicate_chunks_idempotent(in_memory_indexer, sample_chunk_records):
    """Test that indexing the same chunks twice updates rather than duplicates points."""
    in_memory_indexer.index_chunks(sample_chunk_records, recreate_collection=True)
    in_memory_indexer.index_chunks(sample_chunk_records, recreate_collection=False)

    collection_info = in_memory_indexer.client.get_collection(in_memory_indexer.collection_name)
    assert collection_info.points_count == len(sample_chunk_records)
