"""Advanced chunking strategies for document ingestion and benchmark comparisons."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Structured representation of a document chunk."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(default=0, description="Sequential index within parent document")
    word_count: int = Field(default=0, description="Number of words in the chunk")
    char_count: int = Field(default=0, description="Number of characters in the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserved from parent document")


class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, document_id: str = "doc", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split document text into a list of chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Strategy A: Fixed size window with configurable word overlap."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, document_id: str = "doc", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        words = clean_text.split()
        if len(words) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=f"{document_id}_chk_0",
                    text=clean_text,
                    chunk_index=0,
                    word_count=len(words),
                    char_count=len(clean_text),
                    metadata=metadata or {},
                )
            ]

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        idx = 0
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chk_{idx}",
                    text=chunk_text,
                    chunk_index=idx,
                    word_count=len(chunk_words),
                    char_count=len(chunk_text),
                    metadata=metadata or {},
                )
            )
            idx += 1
            if i + self.chunk_size >= len(words):
                break

        return chunks


class SemanticSentenceChunker(ChunkingStrategy):
    """Strategy B: Semantic sentence-based chunking with Indic punctuation support.

    Splits on sentence boundaries (including Indic danda '।' and double danda '॥')
    and aggregates sentences up to target word limit while preserving complete thoughts.
    """

    def __init__(self, target_words: int = 150, min_words: int = 30) -> None:
        self.target_words = target_words
        self.min_words = min_words
        # Regex capturing English, Hindi/Indic dandas, newlines, question marks, and exclamation points
        self.sentence_regex = re.compile(r"(?<=[.!?।॥\n])\s+")

    def _split_sentences(self, text: str) -> List[str]:
        raw_sentences = self.sentence_regex.split(text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def chunk(self, text: str, document_id: str = "doc", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        sentences = self._split_sentences(clean_text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_word_count = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_word_count = len(sentence.split())
            if current_word_count + sent_word_count > self.target_words and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_chk_{chunk_idx}",
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        word_count=current_word_count,
                        char_count=len(chunk_text),
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sent_word_count

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            # If last chunk is too small and we already have previous chunks, append to last chunk if possible
            if chunks and current_word_count < self.min_words:
                prev_chunk = chunks[-1]
                combined_text = f"{prev_chunk.text} {chunk_text}"
                combined_words = prev_chunk.word_count + current_word_count
                chunks[-1] = Chunk(
                    chunk_id=prev_chunk.chunk_id,
                    text=combined_text,
                    chunk_index=prev_chunk.chunk_index,
                    word_count=combined_words,
                    char_count=len(combined_text),
                    metadata=prev_chunk.metadata,
                )
            else:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_chk_{chunk_idx}",
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        word_count=current_word_count,
                        char_count=len(chunk_text),
                        metadata=metadata or {},
                    )
                )

        return chunks


class MetadataAwareChunker(ChunkingStrategy):
    """Strategy C: Structure-aware chunking preserving passage/section boundaries and metadata."""

    def __init__(self, max_words: int = 250) -> None:
        self.max_words = max_words
        self.fallback_chunker = SemanticSentenceChunker(target_words=max_words)

    def chunk(self, text: str, document_id: str = "doc", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        meta = metadata or {}
        clean_text = text.strip()
        if not clean_text:
            return []

        title = meta.get("title", "").strip()
        language = meta.get("language", "en")
        source = meta.get("source", "msmarco-xi")

        # If document contains structured passages or paragraphs
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [clean_text]

        chunks: List[Chunk] = []
        chunk_idx = 0

        for para in paragraphs:
            words = para.split()
            if len(words) > self.max_words:
                # Sub-chunk large paragraph with semantic sentence chunker
                sub_chunks = self.fallback_chunker.chunk(para, document_id=f"{document_id}_p{chunk_idx}", metadata=meta)
                for sc in sub_chunks:
                    rich_text = f"[{title}] {sc.text}" if title and not sc.text.startswith(f"[{title}]") else sc.text
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document_id}_chk_{chunk_idx}",
                            text=rich_text,
                            chunk_index=chunk_idx,
                            word_count=len(rich_text.split()),
                            char_count=len(rich_text),
                            metadata={**meta, "title": title, "language": language, "source": source},
                        )
                    )
                    chunk_idx += 1
            else:
                rich_text = f"[{title}] {para}" if title and not para.startswith(f"[{title}]") else para
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_chk_{chunk_idx}",
                        text=rich_text,
                        chunk_index=chunk_idx,
                        word_count=len(rich_text.split()),
                        char_count=len(rich_text),
                        metadata={**meta, "title": title, "language": language, "source": source},
                    )
                )
                chunk_idx += 1

        return chunks


class ParentChildChunker(ChunkingStrategy):
    """Strategy D: Hierarchical Parent-Child Chunking.

    Generates large Parent chunks (~512 words) for rich LLM context synthesis,
    and smaller granular Child chunks (~128 words) for high-precision vector retrieval.
    Each child chunk carries a pointer and content of its parent in metadata.
    """

    def __init__(
        self,
        parent_size: int = 512,
        child_size: int = 128,
        child_overlap: int = 20,
    ) -> None:
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be strictly less than child_size")
        if child_size >= parent_size:
            raise ValueError("child_size must be less than parent_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, text: str, document_id: str = "doc", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        meta = metadata or {}
        clean_text = text.strip()
        if not clean_text:
            return []

        words = clean_text.split()
        if not words:
            return []

        # 1. Partition into Parents (~512 words)
        parent_chunks_text: List[str] = []
        for i in range(0, len(words), self.parent_size):
            p_words = words[i : i + self.parent_size]
            if p_words:
                parent_chunks_text.append(" ".join(p_words))

        # 2. Partition each Parent into Children (~128 words with overlap)
        child_chunks: List[Chunk] = []
        global_child_idx = 0

        for p_idx, parent_text in enumerate(parent_chunks_text):
            p_id = f"{document_id}_parent_{p_idx}"
            p_words = parent_text.split()

            if len(p_words) <= self.child_size:
                child_chunks.append(
                    Chunk(
                        chunk_id=f"{p_id}_child_0",
                        text=parent_text,
                        chunk_index=global_child_idx,
                        word_count=len(p_words),
                        char_count=len(parent_text),
                        metadata={
                            **meta,
                            "parent_id": p_id,
                            "parent_text": parent_text,
                            "chunk_strategy": "parent_child",
                            "parent_index": p_idx,
                            "child_index": 0,
                        },
                    )
                )
                global_child_idx += 1
                continue

            step = self.child_size - self.child_overlap
            c_idx = 0
            for j in range(0, len(p_words), step):
                c_words = p_words[j : j + self.child_size]
                if not c_words:
                    break
                c_text = " ".join(c_words)
                child_chunks.append(
                    Chunk(
                        chunk_id=f"{p_id}_child_{c_idx}",
                        text=c_text,
                        chunk_index=global_child_idx,
                        word_count=len(c_words),
                        char_count=len(c_text),
                        metadata={
                            **meta,
                            "parent_id": p_id,
                            "parent_text": parent_text,
                            "chunk_strategy": "parent_child",
                            "parent_index": p_idx,
                            "child_index": c_idx,
                        },
                    )
                )
                c_idx += 1
                global_child_idx += 1
                if j + self.child_size >= len(p_words):
                    break

        return child_chunks


def get_chunker(strategy: str = "semantic", **kwargs: Any) -> ChunkingStrategy:
    """Factory function for instantiating chunking strategies."""
    normalized = strategy.lower().strip()
    if normalized in ("fixed", "fixed_size", "strategy_a"):
        return FixedSizeChunker(
            chunk_size=kwargs.get("chunk_size", 200),
            chunk_overlap=kwargs.get("chunk_overlap", 40),
        )
    elif normalized in ("semantic", "sentence", "strategy_b"):
        return SemanticSentenceChunker(
            target_words=kwargs.get("target_words", 150),
            min_words=kwargs.get("min_words", 30),
        )
    elif normalized in ("metadata", "metadata_aware", "structure", "strategy_c"):
        return MetadataAwareChunker(
            max_words=kwargs.get("max_words", 250),
        )
    elif normalized in ("parent_child", "hierarchical", "strategy_d"):
        return ParentChildChunker(
            parent_size=kwargs.get("parent_size", 512),
            child_size=kwargs.get("child_size", 128),
            child_overlap=kwargs.get("child_overlap", 20),
        )
    else:
        # Default to SemanticSentenceChunker
        return SemanticSentenceChunker()
