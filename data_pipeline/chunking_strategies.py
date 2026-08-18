"""Multilingual Chunking Strategies Module.

Implements three distinct chunking strategies for English & Indic text corpora:
1. Semantic Chunking (Sentence-boundary aware for English `.!?` and Indic `।॥`)
2. Hierarchical Chunking (Parent-Child multi-granularity with bidirectional metadata)
3. Overlap-based Chunking (Configurable sliding window with word-boundary preservation)
"""

import abc
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional

from data_pipeline.dataset_downloader import NormalizedDocument


# =============================================================================
# Standard Chunk Schema
# =============================================================================
@dataclass
class ChunkRecord:
    """Unified chunk schema with comprehensive metadata preservation."""

    chunk_id: str
    document_id: str
    text: str
    strategy: str
    position: int
    language: str = "en"
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    source: str = "ai4bharat/MSMARCO-XI"
    word_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.char_count:
            self.char_count = len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return asdict(self)


# =============================================================================
# Base Chunker Interface
# =============================================================================
class BaseChunker(abc.ABC):
    """Abstract base class for all chunking strategies."""

    strategy_name: str = "base"

    @abc.abstractmethod
    def chunk_document(self, document: NormalizedDocument) -> List[ChunkRecord]:
        """Split a NormalizedDocument into a list of ChunkRecords."""
        pass

    def chunk_documents(self, documents: List[NormalizedDocument]) -> List[ChunkRecord]:
        """Batch process multiple normalized documents."""
        all_chunks: List[ChunkRecord] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


# =============================================================================
# Strategy A: Semantic Chunking (English + Indic Sentence Boundary Aware)
# =============================================================================
class SemanticChunker(BaseChunker):
    """Chunks text along natural sentence boundaries without cutting mid-sentence.

    Supports English sentence delimiters (. ? !) and Indic sentence terminators
    (। Purna Viram, ॥ Dirgha Viram, newline boundaries).
    """

    strategy_name: str = "semantic"

    # Multilingual sentence splitting regex
    # Matches English (. ? !), Indic (। ॥), or double newline breaks
    SENTENCE_SPLIT_REGEX = re.compile(
        r"(?<=[.?!।॥\n])\s+(?=[^\s])"
    )

    def __init__(self, target_words: int = 180, max_words: int = 240) -> None:
        self.target_words = target_words
        self.max_words = max_words

    def split_into_sentences(self, text: str) -> List[str]:
        """Split multilingual text into individual sentences."""
        if not text or not text.strip():
            return []

        # Split on sentence terminators
        raw_sentences = self.SENTENCE_SPLIT_REGEX.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        # Fallback if no sentence delimiter found
        if not sentences:
            sentences = [text.strip()]
        return sentences

    def chunk_document(self, document: NormalizedDocument) -> List[ChunkRecord]:
        """Group sentences into semantic chunks respecting word count limits."""
        text = document.text.strip()
        if not text:
            return []

        sentences = self.split_into_sentences(text)
        chunks: List[ChunkRecord] = []

        current_sentences: List[str] = []
        current_word_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence exceeds max_words and we already have content, flush
            if current_word_count + sentence_words > self.max_words and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{document.document_id}_sem_{chunk_index}",
                        document_id=document.document_id,
                        text=chunk_text,
                        strategy=self.strategy_name,
                        position=chunk_index,
                        language=document.language,
                        source=document.source,
                        metadata={
                            **document.metadata,
                            "title": document.title,
                            "query": document.query,
                        },
                    )
                )
                chunk_index += 1
                current_sentences = [sentence]
                current_word_count = sentence_words
            else:
                current_sentences.append(sentence)
                current_word_count += sentence_words

                # If we've reached the target word count, prepare to close chunk
                if current_word_count >= self.target_words:
                    chunk_text = " ".join(current_sentences).strip()
                    chunks.append(
                        ChunkRecord(
                            chunk_id=f"{document.document_id}_sem_{chunk_index}",
                            document_id=document.document_id,
                            text=chunk_text,
                            strategy=self.strategy_name,
                            position=chunk_index,
                            language=document.language,
                            source=document.source,
                            metadata={
                                **document.metadata,
                                "title": document.title,
                                "query": document.query,
                            },
                        )
                    )
                    chunk_index += 1
                    current_sentences = []
                    current_word_count = 0

        # Flush any remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}_sem_{chunk_index}",
                    document_id=document.document_id,
                    text=chunk_text,
                    strategy=self.strategy_name,
                    position=chunk_index,
                    language=document.language,
                    source=document.source,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "query": document.query,
                    },
                )
            )

        return chunks


# =============================================================================
# Strategy B: Hierarchical Chunking (Parent-Child Architecture)
# =============================================================================
class HierarchicalChunker(BaseChunker):
    """Multi-granularity chunker generating Parent (large) and Child (granular) chunks.

    Child chunks are embedded for vector retrieval; Parent text is returned for LLM context.
    """

    strategy_name: str = "hierarchical"

    def __init__(
        self,
        parent_words: int = 512,
        child_words: int = 128,
        child_overlap: int = 20,
    ) -> None:
        self.parent_words = parent_words
        self.child_words = child_words
        self.child_overlap = child_overlap

    def _split_into_windows(self, text: str, size: int, overlap: int) -> List[str]:
        """Split words into fixed-size windows with word boundary preservation."""
        words = text.split()
        if not words:
            return []
        if len(words) <= size:
            return [" ".join(words)]

        step = max(1, size - overlap)
        windows: List[str] = []
        for i in range(0, len(words), step):
            window = words[i : i + size]
            if window:
                windows.append(" ".join(window))
            if i + size >= len(words):
                break
        return windows

    def chunk_document(self, document: NormalizedDocument) -> List[ChunkRecord]:
        """Generate hierarchical parent-child chunks with bidirectional metadata links."""
        text = document.text.strip()
        if not text:
            return []

        # 1. Create Parent Windows
        parent_windows = self._split_into_windows(text, self.parent_words, overlap=40)
        chunks: List[ChunkRecord] = []
        global_child_pos = 0

        for p_idx, parent_text in enumerate(parent_windows):
            parent_id = f"{document.document_id}_parent_{p_idx}"

            # 2. Split Parent into granular Child Chunks
            child_windows = self._split_into_windows(
                parent_text, self.child_words, self.child_overlap
            )

            for c_idx, child_text in enumerate(child_windows):
                child_id = f"{parent_id}_child_{c_idx}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=child_id,
                        document_id=document.document_id,
                        text=child_text,
                        strategy=self.strategy_name,
                        position=global_child_pos,
                        language=document.language,
                        parent_id=parent_id,
                        parent_text=parent_text,
                        source=document.source,
                        metadata={
                            **document.metadata,
                            "title": document.title,
                            "query": document.query,
                            "parent_index": p_idx,
                            "child_index": c_idx,
                        },
                    )
                )
                global_child_pos += 1

        return chunks


# =============================================================================
# Strategy C: Overlap-based Chunking (Sliding Window)
# =============================================================================
class OverlapChunker(BaseChunker):
    """Fixed-size chunker with configurable word-based sliding window and overlap."""

    strategy_name: str = "overlap"

    def __init__(self, chunk_words: int = 200, overlap_words: int = 40) -> None:
        self.chunk_words = chunk_words
        self.overlap_words = min(overlap_words, chunk_words - 1)

    def chunk_document(self, document: NormalizedDocument) -> List[ChunkRecord]:
        """Chunk document using sliding word windows."""
        text = document.text.strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks: List[ChunkRecord] = []
        step = max(1, self.chunk_words - self.overlap_words)
        chunk_idx = 0

        for i in range(0, len(words), step):
            window_words = words[i : i + self.chunk_words]
            if not window_words:
                break

            chunk_text = " ".join(window_words).strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}_ovlp_{chunk_idx}",
                    document_id=document.document_id,
                    text=chunk_text,
                    strategy=self.strategy_name,
                    position=chunk_idx,
                    language=document.language,
                    source=document.source,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "query": document.query,
                        "start_word_idx": i,
                        "end_word_idx": i + len(window_words),
                    },
                )
            )
            chunk_idx += 1
            if i + self.chunk_words >= len(words):
                break

        return chunks


# =============================================================================
# Factory Helper
# =============================================================================
def get_chunker(strategy: str = "semantic", **kwargs: Any) -> BaseChunker:
    """Factory resolver for chunking strategies.

    Strategies: 'semantic', 'hierarchical', 'overlap'
    """
    clean_strategy = strategy.lower().strip()
    if clean_strategy in ("semantic", "sentence"):
        target_words = kwargs.get("target_words", 180)
        max_words = kwargs.get("max_words", 240)
        return SemanticChunker(target_words=target_words, max_words=max_words)
    elif clean_strategy in ("hierarchical", "parent_child", "parent-child"):
        parent_words = kwargs.get("parent_words", 512)
        child_words = kwargs.get("child_words", 128)
        child_overlap = kwargs.get("child_overlap", 20)
        return HierarchicalChunker(
            parent_words=parent_words, child_words=child_words, child_overlap=child_overlap
        )
    elif clean_strategy in ("overlap", "fixed", "sliding_window"):
        chunk_words = kwargs.get("chunk_words", 200)
        overlap_words = kwargs.get("overlap_words", 40)
        return OverlapChunker(chunk_words=chunk_words, overlap_words=overlap_words)
    else:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Supported: 'semantic', 'hierarchical', 'overlap'"
        )
