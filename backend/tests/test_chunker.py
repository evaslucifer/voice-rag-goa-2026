"""Tests for multiple chunking strategies."""

import pytest
from ingestion.chunker import (
    FixedSizeChunker,
    MetadataAwareChunker,
    SemanticSentenceChunker,
    get_chunker,
)


def test_fixed_size_chunker() -> None:
    """Test Strategy A: Fixed-size chunking with overlap."""
    text = "word " * 300
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text, document_id="doc_1")

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "doc_1_chk_0"
    assert chunks[0].word_count == 100
    assert chunks[1].word_count <= 100


def test_semantic_sentence_chunker_english() -> None:
    """Test Strategy B: Semantic sentence chunking on English text."""
    text = (
        "The Manhattan Project was established in 1942. "
        "It was led by the United States with UK support. "
        "The project produced the first nuclear weapons. "
        "It concluded after World War II."
    )
    chunker = SemanticSentenceChunker(target_words=20, min_words=5)
    chunks = chunker.chunk(text, document_id="doc_2")

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.chunk_id.startswith("doc_2_chk_")


def test_semantic_sentence_chunker_indic_danda() -> None:
    """Test Strategy B handles Indic danda '।' sentence splits."""
    indic_text = "इसरो की स्थापना 1969 में हुई थी। इसके संस्थापक डॉ. विक्रम साराभाई थे। इसरो का मुख्यालय बेंगलुरु में है।"
    chunker = SemanticSentenceChunker(target_words=10, min_words=3)
    chunks = chunker.chunk(indic_text, document_id="indic_doc")

    assert len(chunks) >= 1
    full_reconstructed = " ".join([c.text for c in chunks])
    assert "विक्रम साराभाई" in full_reconstructed


def test_metadata_aware_chunker() -> None:
    """Test Strategy C: Metadata-aware structure chunking."""
    text = "Paragraph one regarding physics.\n\nParagraph two regarding engineering."
    chunker = MetadataAwareChunker(max_words=100)
    chunks = chunker.chunk(text, document_id="meta_doc", metadata={"title": "Science Doc", "language": "en"})

    assert len(chunks) == 2
    assert "[Science Doc]" in chunks[0].text
    assert chunks[0].metadata["title"] == "Science Doc"


def test_get_chunker_factory() -> None:
    """Test factory instantiation for all strategies."""
    c_fixed = get_chunker("fixed")
    assert isinstance(c_fixed, FixedSizeChunker)

    c_semantic = get_chunker("semantic")
    assert isinstance(c_semantic, SemanticSentenceChunker)

    c_meta = get_chunker("metadata")
    assert isinstance(c_meta, MetadataAwareChunker)
