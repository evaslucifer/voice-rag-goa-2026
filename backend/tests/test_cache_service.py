"""Tests for CacheService in-memory caching."""

import time
from app.services.cache_service import CacheService, get_cache_service


def test_cache_service_singleton() -> None:
    """Test get_cache_service returns singleton instance."""
    c1 = get_cache_service()
    c2 = get_cache_service()
    assert c1 is c2


def test_cache_set_and_get() -> None:
    """Test basic cache set and get functionality."""
    cache = CacheService(default_ttl_seconds=60)
    cache.set("query:1", {"result": "sample answer"})

    cached = cache.get("query:1")
    assert cached is not None
    assert cached["result"] == "sample answer"


def test_cache_miss_and_expiration() -> None:
    """Test cache miss and expiration behavior."""
    cache = CacheService(default_ttl_seconds=0.05)  # 50ms TTL
    cache.set("temp_key", "temp_value")

    assert cache.get("temp_key") == "temp_value"
    time.sleep(0.06)
    assert cache.get("temp_key") is None
    assert cache.get("non_existent_key") is None


def test_cache_delete_and_clear() -> None:
    """Test cache key deletion and clearing."""
    cache = CacheService()
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    assert cache.size() == 2

    assert cache.delete("k1") is True
    assert cache.get("k1") is None
    assert cache.get("k2") == "v2"

    cache.clear()
    assert cache.size() == 0
