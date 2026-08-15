"""Qdrant indexer for embedding vectors and structured chunk payloads."""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client.models import PointStruct
from app.services.qdrant_service import QdrantService, get_qdrant_service
from app.utils.logging import get_logger
from ingestion.chunker import Chunk

logger = get_logger(__name__)


class QdrantIndexer:
    """Indexer responsible for writing vector chunks and metadata into Qdrant."""

    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        collection_name: Optional[str] = None,
        vector_dim: int = 384,
    ) -> None:
        self.qdrant_service = qdrant_service or get_qdrant_service()
        self.collection_name = collection_name or self.qdrant_service.collection_name
        self.vector_dim = vector_dim

    async def initialize_collection(self, recreate: bool = False) -> bool:
        """Ensure the target Qdrant collection is created with correct dimensions."""
        logger.info("Initializing Qdrant collection '%s' (vector_dim=%d)...", self.collection_name, self.vector_dim)
        return await self.qdrant_service.create_collection(
            collection_name=self.collection_name,
            vector_size=self.vector_dim,
            distance="Cosine",
            recreate=recreate,
        )

    def _chunk_to_point(self, chunk: Chunk, vector: List[float]) -> PointStruct:
        """Convert a chunk and its embedding into a deterministic Qdrant PointStruct."""
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
        meta = chunk.metadata or {}
        language = meta.get("language", "en")
        source = meta.get("source", "ai4bharat/MSMARCO-XI")
        doc_id = meta.get("document_id") or chunk.chunk_id.split("_chk_")[0]

        payload = {
            "document_id": doc_id,
            "query_id": str(meta.get("query_id") or meta.get("id") or ""),
            "passage_id": str(meta.get("passage_id") or doc_id),
            "parent_id": str(meta.get("parent_id") or ""),
            "child_id": str(chunk.chunk_id if "child" in chunk.chunk_id else ""),
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "chunk_strategy": str(meta.get("chunk_strategy") or "semantic"),
            "text": chunk.text,
            "parent_text": meta.get("parent_text"),
            "word_count": chunk.word_count,
            "char_count": chunk.char_count,
            "language": language,
            "source": source,
            "title": meta.get("title", ""),
            "is_selected": bool(meta.get("is_selected", True)),
            "data_mode": str(meta.get("data_mode") or ""),
            "metadata": {k: v for k, v in meta.items() if k not in ("document_id", "title", "language", "source")},
        }

        vec_list = [float(x) for x in vector] if hasattr(vector, "__iter__") else list(vector)
        return PointStruct(
            id=point_uuid,
            vector=vec_list,
            payload=payload,
        )

    async def index_chunks(
        self,
        chunk_vector_pairs: List[Tuple[Chunk, List[float]]],
        batch_size: int = 100,
    ) -> int:
        """Batch index chunks into Qdrant.

        Returns:
            Number of points indexed.
        """
        if not chunk_vector_pairs:
            return 0

        points: List[PointStruct] = [
            self._chunk_to_point(chunk, vector)
            for chunk, vector in chunk_vector_pairs
        ]

        t0 = time.perf_counter()
        count = await self.qdrant_service.upsert_vectors(
            points=points,
            collection_name=self.collection_name,
            batch_size=batch_size,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "Indexed %d points into '%s' in %.2f ms (%.2f points/sec)",
            count,
            self.collection_name,
            duration_ms,
            (count / (duration_ms / 1000.0)) if duration_ms > 0 else 0,
        )
        return count
