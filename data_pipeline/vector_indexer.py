"""Vector Indexer Pipeline using FastEmbed and Qdrant.

Handles dense multilingual vector embeddings via FastEmbed ONNX runtime
and idempotent collection indexing with rich payload metadata in Qdrant.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from data_pipeline.chunking_strategies import ChunkRecord
from data_pipeline.config import DataPipelineConfig, get_pipeline_config

logger = logging.getLogger("data_pipeline.vector_indexer")


# =============================================================================
# Search Result Schema
# =============================================================================
@dataclass
class SearchResult:
    """Structured vector retrieval search hit."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    language: str = "en"
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    strategy: str = "semantic"
    source: str = "ai4bharat/MSMARCO-XI"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Vector Indexer
# =============================================================================
class VectorIndexer:
    """Orchestrates FastEmbed multilingual embedding generation and Qdrant storage."""

    def __init__(
        self,
        config: Optional[DataPipelineConfig] = None,
        model_name: Optional[str] = None,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.config = config or get_pipeline_config()
        self.model_name = model_name or self.config.EMBEDDING_MODEL_NAME
        self.collection_name = collection_name or self.config.QDRANT_COLLECTION_NAME
        self.mode = mode or self.config.QDRANT_MODE

        # 1. Initialize FastEmbed multilingual model
        logger.info("Initializing FastEmbed model: '%s' (384-dim, Multilingual)", self.model_name)
        self.embedder = TextEmbedding(model_name=self.model_name)
        self.dimension = self.config.EMBEDDING_DIMENSION

        # 2. Initialize Qdrant Client
        if qdrant_client is not None:
            self.client = qdrant_client
        else:
            self.client = self._init_qdrant_client()

    def _init_qdrant_client(self) -> QdrantClient:
        """Initialize Qdrant client based on configured mode."""
        if self.mode == "memory":
            logger.info("Connecting to in-memory Qdrant instance (:memory:)")
            return QdrantClient(":memory:")
        elif self.config.QDRANT_URL:
            logger.info("Connecting to remote Qdrant server at: %s", self.config.QDRANT_URL)
            return QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY,
                timeout=self.config.QDRANT_TIMEOUT_SECONDS,
            )
        else:
            local_path = self.config.QDRANT_PATH
            logger.info("Connecting to local embedded Qdrant at path: %s", local_path)
            return QdrantClient(path=local_path)

    def ensure_collection(self, recreate: bool = False) -> None:
        """Ensure the target collection exists idempotently with Cosine metric."""
        collections_response = self.client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]

        if self.collection_name in existing_collections:
            if recreate:
                logger.info("Recreating collection: '%s'", self.collection_name)
                self.client.delete_collection(collection_name=self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created fresh collection '%s'", self.collection_name)
            else:
                logger.info("Collection '%s' already exists (Idempotent OK).", self.collection_name)
        else:
            logger.info("Creating new Qdrant collection: '%s' (dim=%d, metric=Cosine)", self.collection_name, self.dimension)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Successfully created collection '%s'", self.collection_name)

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute multilingual dense embeddings for a batch of strings."""
        if not texts:
            return []

        embeddings_generator = self.embedder.embed(texts, batch_size=batch_size)
        return [list(vec) for vec in embeddings_generator]

    def index_chunks(
        self,
        chunks: List[ChunkRecord],
        batch_size: Optional[int] = None,
        recreate_collection: bool = False,
    ) -> Dict[str, Any]:
        """Generate embeddings and index chunks into Qdrant in batches.

        Returns execution statistics dictionary.
        """
        if not chunks:
            logger.warning("No chunks provided for indexing.")
            return {"indexed_count": 0, "latency_ms": 0.0, "status": "EMPTY"}

        self.ensure_collection(recreate=recreate_collection)
        batch_sz = batch_size or self.config.QDRANT_BATCH_SIZE

        t0 = time.perf_counter()
        total_chunks = len(chunks)
        indexed_count = 0

        logger.info("Starting indexing of %d chunks into '%s' (batch_size=%d)...", total_chunks, self.collection_name, batch_sz)

        for i in range(0, total_chunks, batch_sz):
            batch = chunks[i : i + batch_sz]
            batch_texts = [c.text for c in batch]

            # 1. Generate vectors in batch
            batch_vectors = self.generate_embeddings(batch_texts, batch_size=len(batch_texts))

            # 2. Build PointStructs with deterministic UUIDs and rich payloads
            points: List[PointStruct] = []
            for chunk, vector in zip(batch, batch_vectors):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "strategy": chunk.strategy,
                    "position": chunk.position,
                    "language": chunk.language,
                    "parent_id": chunk.parent_id,
                    "parent_text": chunk.parent_text,
                    "source": chunk.source,
                    "word_count": chunk.word_count,
                    "char_count": chunk.char_count,
                    "metadata": chunk.metadata,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # 3. Upsert points into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            indexed_count += len(points)
            logger.debug("Indexed batch %d/%d (total points: %d)", i + len(points), total_chunks, indexed_count)

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info("Successfully indexed %d chunks in %.2f ms (Avg: %.2f ms/chunk)", indexed_count, total_latency_ms, total_latency_ms / max(1, indexed_count))

        return {
            "indexed_count": indexed_count,
            "total_chunks": total_chunks,
            "latency_ms": total_latency_ms,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "status": "SUCCESS",
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        language: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform dense multilingual similarity search on the indexed collection."""
        if not query or not query.strip():
            return []

        # 1. Embed query
        query_vector = list(self.embedder.embed([query.strip()]))[0]

        # 2. Build filter if language specified
        query_filter = None
        if language:
            query_filter = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="language",
                        match=rest_models.MatchValue(value=language),
                    )
                ]
            )

        # 3. Query Qdrant (supports both modern query_points and legacy search)
        raw_points = []
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=query_filter,
                with_payload=True,
            )
            raw_points = response.points
        elif hasattr(self.client, "search"):
            raw_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=list(query_vector),
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=query_filter,
                with_payload=True,
            )

        results: List[SearchResult] = []
        for hit in raw_points:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    document_id=str(payload.get("document_id", "")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                    language=str(payload.get("language", "en")),
                    parent_id=payload.get("parent_id"),
                    parent_text=payload.get("parent_text"),
                    strategy=str(payload.get("strategy", "semantic")),
                    source=str(payload.get("source", "ai4bharat/MSMARCO-XI")),
                    metadata=payload.get("metadata", {}),
                )
            )

        return results
