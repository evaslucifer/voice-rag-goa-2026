"""Qdrant vector database service wrapper with full collection and search operations."""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QdrantServiceError(Exception):
    """Base exception for Qdrant service failures."""
    pass


class QdrantConnectionError(QdrantServiceError):
    """Raised when connection to Qdrant fails or times out."""
    pass


class QdrantConfigurationError(QdrantServiceError):
    """Raised when Qdrant configuration is invalid or missing."""
    pass


class QdrantService:
    """Async client wrapper for interacting with Qdrant vector database."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        timeout: Optional[float] = None,
        location: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.settings = get_settings()
        self.url = url or self.settings.QDRANT_URL
        self.api_key = api_key if api_key is not None else self.settings.QDRANT_API_KEY
        self.collection_name = collection_name or self.settings.QDRANT_COLLECTION
        self.timeout = timeout or self.settings.QDRANT_TIMEOUT_SECONDS
        self.location = location  # e.g., ":memory:"
        self.path = path or getattr(self.settings, "QDRANT_PATH", "./qdrant_storage")
        self._client: Optional[AsyncQdrantClient] = None

    def _get_client(self) -> AsyncQdrantClient:
        """Get or initialize the AsyncQdrantClient."""
        if self._client is None:
            if self.location:
                logger.info("Initializing AsyncQdrantClient with location: %s", self.location)
                self._client = AsyncQdrantClient(location=self.location)
            elif self.url:
                logger.info("Initializing AsyncQdrantClient at URL: %s", self.url)
                self._client = AsyncQdrantClient(
                    url=self.url,
                    api_key=self.api_key or None,
                    timeout=self.timeout,
                )
            else:
                logger.info("Initializing embedded AsyncQdrantClient at local path: %s", self.path)
                self._client = AsyncQdrantClient(path=self.path)
        return self._client

    async def close(self) -> None:
        """Close the Qdrant async client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("AsyncQdrantClient connection closed.")

    async def check_connection(self) -> bool:
        """Verify connectivity to the Qdrant instance."""
        try:
            client = self._get_client()
            collections_response = await client.get_collections()
            return collections_response is not None
        except Exception as e:
            logger.warning("Qdrant connection check failed: %s", str(e))
            return False

    async def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check whether the specified collection exists."""
        target_collection = collection_name or self.collection_name
        try:
            client = self._get_client()
            return await client.collection_exists(collection_name=target_collection)
        except Exception as e:
            logger.error("Failed checking existence of collection '%s': %s", target_collection, str(e))
            raise QdrantConnectionError(f"Failed to query collection '{target_collection}': {e}") from e

    async def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "Cosine",
        recreate: bool = False,
    ) -> bool:
        """Create a new collection if it does not exist (or recreate if requested)."""
        target_collection = collection_name or self.collection_name
        client = self._get_client()

        dist_enum = getattr(Distance, distance.upper(), Distance.COSINE)

        try:
            exists = await self.collection_exists(target_collection)
            if exists:
                if recreate:
                    logger.warning("Recreating Qdrant collection: %s", target_collection)
                    await client.delete_collection(collection_name=target_collection)
                else:
                    logger.info("Collection '%s' already exists.", target_collection)
                    return True

            logger.info("Creating Qdrant collection '%s' (dim=%d, distance=%s)...", target_collection, vector_size, distance)
            await client.create_collection(
                collection_name=target_collection,
                vectors_config=VectorParams(size=vector_size, distance=dist_enum),
            )
            return True
        except Exception as e:
            logger.error("Failed creating collection '%s': %s", target_collection, str(e))
            raise QdrantServiceError(f"Could not create collection '{target_collection}': {e}") from e

    async def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete an existing collection."""
        target_collection = collection_name or self.collection_name
        try:
            client = self._get_client()
            return await client.delete_collection(collection_name=target_collection)
        except Exception as e:
            logger.error("Failed deleting collection '%s': %s", target_collection, str(e))
            raise QdrantServiceError(f"Could not delete collection '{target_collection}': {e}") from e

    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Fetch metadata, vector count, and status for a collection."""
        target_collection = collection_name or self.collection_name
        try:
            client = self._get_client()
            info = await client.get_collection(collection_name=target_collection)
            return {
                "name": target_collection,
                "status": str(info.status),
                "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
            }
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return {
                    "name": target_collection,
                    "status": "NOT_FOUND",
                    "vectors_count": 0,
                    "indexed_vectors_count": 0,
                }
            raise QdrantServiceError(f"Qdrant API error: {e}") from e
        except Exception as e:
            logger.error("Failed fetching info for collection '%s': %s", target_collection, str(e))
            raise QdrantConnectionError(f"Could not connect to Qdrant: {e}") from e

    async def upsert_vectors(
        self,
        points: List[PointStruct],
        collection_name: Optional[str] = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert a list of PointStruct vectors with metadata payloads into Qdrant."""
        if not points:
            return 0

        target_collection = collection_name or self.collection_name
        client = self._get_client()

        try:
            total_upserted = 0
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await client.upsert(
                    collection_name=target_collection,
                    points=batch,
                )
                total_upserted += len(batch)

            logger.info("Upserted %d vectors into collection '%s'.", total_upserted, target_collection)
            return total_upserted
        except Exception as e:
            logger.error("Failed upserting vectors into collection '%s': %s", target_collection, str(e))
            raise QdrantServiceError(f"Upsert failed: {e}") from e

    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search returning top-k matching documents with scores and payloads."""
        target_collection = collection_name or self.collection_name
        if not query_vector:
            raise QdrantServiceError("Query vector cannot be empty.")

        try:
            client = self._get_client()
            search_results = await client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            results: List[Dict[str, Any]] = []
            for hit in getattr(search_results, "points", []):
                results.append({
                    "id": str(hit.id),
                    "score": float(hit.score) if hit.score is not None else 0.0,
                    "payload": hit.payload or {},
                })
            return results
        except Exception as e:
            logger.error("Vector search failed on collection '%s': %s", target_collection, str(e))
            raise QdrantServiceError(f"Vector search failed: {e}") from e


@lru_cache()
def get_qdrant_service() -> QdrantService:
    """Return singleton instance of QdrantService."""
    return QdrantService()
