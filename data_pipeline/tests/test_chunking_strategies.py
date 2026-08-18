"""Unit tests for Semantic, Hierarchical, and Overlap chunking strategies."""

import pytest

from data_pipeline.chunking_strategies import (
    HierarchicalChunker,
    OverlapChunker,
    SemanticChunker,
    get_chunker,
)
from data_pipeline.dataset_downloader import NormalizedDocument


def test_semantic_chunker_english(sample_normalized_documents):
    """Test semantic sentence chunking on English text."""
    doc = sample_normalized_documents[0]
    chunker = SemanticChunker(target_words=20, max_words=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == doc.document_id
        assert chunk.strategy == "semantic"
        assert chunk.language == "en"
        assert chunk.word_count > 0
        assert chunk.char_count > 0
        assert "manhattan" in chunk.text.lower() or "oppenheimer" in chunk.text.lower()


def test_semantic_chunker_indic_danda(sample_normalized_documents):
    """Test semantic chunking splitting on Hindi/Marathi Purna Viram (।) boundaries."""
    doc = sample_normalized_documents[2]  # Marathi text with danda
    chunker = SemanticChunker(target_words=10, max_words=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    assert any("शिवाजी महाराज" in c.text for c in chunks)
    assert any("रायगड किल्ला" in c.text for c in chunks)


def test_hierarchical_chunker_parent_child_relationships(sample_normalized_documents):
    """Test hierarchical chunking preserves parent-child IDs and parent text."""
    doc = sample_normalized_documents[0]
    chunker = HierarchicalChunker(parent_words=30, child_words=10, child_overlap=2)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.strategy == "hierarchical"
        assert chunk.parent_id is not None
        assert chunk.parent_text is not None
        assert chunk.parent_id in chunk.chunk_id
        assert "parent_index" in chunk.metadata
        assert "child_index" in chunk.metadata
        assert len(chunk.parent_text) >= len(chunk.text)


def test_overlap_chunker_sliding_window(sample_normalized_documents):
    """Test overlap-based chunking with fixed word size and overlap."""
    doc = sample_normalized_documents[0]
    chunker = OverlapChunker(chunk_words=15, overlap_words=5)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].position == 0
    assert chunks[1].position == 1
    assert chunks[0].strategy == "overlap"
    assert "start_word_idx" in chunks[0].metadata


def test_chunker_empty_document():
    """Test that empty or whitespace-only documents produce empty chunk lists safely."""
    empty_doc = NormalizedDocument(document_id="empty", text="", language="en")
    for strategy in ("semantic", "hierarchical", "overlap"):
        chunker = get_chunker(strategy)
        chunks = chunker.chunk_document(empty_doc)
        assert chunks == []


def test_chunker_factory():
    """Test factory resolver for valid and invalid chunking strategies."""
    assert isinstance(get_chunker("semantic"), SemanticChunker)
    assert isinstance(get_chunker("hierarchical"), HierarchicalChunker)
    assert isinstance(get_chunker("overlap"), OverlapChunker)

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("non_existent_strategy")
