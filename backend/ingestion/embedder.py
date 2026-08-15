"""Batch embedder for dataset chunks using FastEmbed ONNX."""

import asyncio
import time
from typing import List, Optional, Tuple
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.utils.logging import get_logger
from ingestion.chunker import Chunk

logger = get_logger(__name__)


class BatchEmbedder:
    """Batch embedding generator for ingested chunks."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None, batch_size: int = 32) -> None:
        self.embedding_service = embedding_service or get_embedding_service()
        self.batch_size = batch_size

    async def embed_chunks(self, chunks: List[Chunk]) -> List[Tuple[Chunk, List[float]]]:
        """Generate dense embeddings for a list of Chunk objects in batches.

        Returns:
            List of (Chunk, embedding_vector) tuples.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        t0 = time.perf_counter()

        logger.info("Generating embeddings for %d chunks (batch_size=%d)...", len(chunks), self.batch_size)
        embeddings = await self.embedding_service.embed_documents(texts)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        if len(embeddings) != len(chunks):
            raise RuntimeError(f"Mismatch: got {len(embeddings)} embeddings for {len(chunks)} chunks.")

        logger.info(
            "Embedded %d chunks in %.2f ms (%.2f chunks/sec)",
            len(chunks),
            duration_ms,
            (len(chunks) / (duration_ms / 1000.0)) if duration_ms > 0 else 0,
        )

        return list(zip(chunks, embeddings))
