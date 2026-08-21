"""Embedding service using local FastEmbed ONNX model."""

import asyncio
from functools import lru_cache
from typing import List, Optional

from fastembed import TextEmbedding

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when an embedding operation fails."""
    pass


class EmbeddingService:
    """Singleton service for generating fast, local embeddings via ONNX Runtime."""

    _instance: Optional["EmbeddingService"] = None
    _model: Optional[TextEmbedding] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.EMBEDDING_MODEL_NAME

    def initialize(self) -> None:
        """Initialize the embedding model during application startup."""
        if self._model is not None:
            return

        logger.info("Initializing FastEmbed model: %s", self.model_name)

        try:
            self._model = TextEmbedding(model_name=self.model_name, threads=1)
            logger.info("FastEmbed model initialized successfully (single-threaded ONNX mode)")
        except Exception as e:
            logger.error(
                "Failed to load FastEmbed model: %s",
                str(e),
                exc_info=True,
            )
            raise EmbeddingServiceError(
                f"Failed to initialize embedding model "
                f"'{self.model_name}': {e}"
            ) from e

    def _get_model(self) -> TextEmbedding:
        """Return the initialized FastEmbed model."""
        if self._model is None:
            self.initialize()

        return self._model

    def is_initialized(self) -> bool:
        """Check if the embedding model is already loaded in memory."""
        return self._model is not None

    def _sync_embed_query(self, query: str) -> List[float]:
        """Synchronous query embedding computation."""
        model = self._get_model()

        embeddings = list(model.embed([query]))

        if not embeddings:
            raise EmbeddingServiceError(
                "No embedding was generated for the query."
            )

        res = (
            embeddings[0].tolist()
            if hasattr(embeddings[0], "tolist")
            else list(embeddings[0])
        )
        import gc
        gc.collect()
        return res

    def _sync_embed_documents(
        self,
        documents: List[str],
    ) -> List[List[float]]:
        """Synchronous batch document embedding computation."""

        if not documents:
            return []

        model = self._get_model()

        embeddings = list(
            model.embed(
                documents,
                batch_size=self.settings.EMBEDDING_BATCH_SIZE,
            )
        )

        return [
            emb.tolist() if hasattr(emb, "tolist") else list(emb)
            for emb in embeddings
        ]

    async def embed_query(self, query: str) -> List[float]:
        """Asynchronously compute query embedding offloaded to threadpool."""

        if not query or not query.strip():
            raise EmbeddingServiceError("Query cannot be empty.")

        try:
            return await asyncio.to_thread(
                self._sync_embed_query,
                query,
            )
        except Exception as e:
            if not isinstance(e, EmbeddingServiceError):
                raise EmbeddingServiceError(
                    f"Embedding computation failed: {e}"
                ) from e
            raise

    async def embed_documents(
        self,
        documents: List[str],
    ) -> List[List[float]]:
        """Asynchronously compute document embeddings offloaded to threadpool."""

        try:
            return await asyncio.to_thread(
                self._sync_embed_documents,
                documents,
            )
        except Exception as e:
            if not isinstance(e, EmbeddingServiceError):
                raise EmbeddingServiceError(
                    f"Document embedding computation failed: {e}"
                ) from e
            raise


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Return the singleton instance of EmbeddingService."""
    return EmbeddingService()