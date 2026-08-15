"""In-memory caching service for query results and embeddings."""

import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """Lightweight in-memory cache with TTL support."""

    def __init__(self, default_ttl_seconds: float = 300.0) -> None:
        self.default_ttl = default_ttl_seconds
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if present and not expired."""
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Store a key-value pair with a TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all keys from the cache."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of stored keys (including potentially expired ones)."""
        return len(self._store)


@lru_cache()
def get_cache_service() -> CacheService:
    """Return singleton instance of CacheService."""
    return CacheService()
