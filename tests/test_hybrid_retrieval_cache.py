# =============================================================================
# Query Caching Tests (New from Task 6)
# =============================================================================


def test_lru_cache_basic_operations():
    """Test LRUCache basic get/put operations"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=3)

    # Test put and get
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test non-existent key
    assert cache.get("key2") is None

    # Test update existing key
    cache.put("key1", "value1_updated")
    assert cache.get("key1") == "value1_updated"


def test_lru_cache_eviction():
    """Test LRUCache eviction when capacity is exceeded"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1 (least recently used)

    # key1 should be evicted
    assert cache.get("key1") is None
    # key2 and key3 should still exist
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_lru_cache_access_order():
    """Test that accessing item updates its LRU order"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Access key1 to make it more recently used
    cache.get("key1")

    # Add key3 - should evict key2 (now least recently used)
    cache.put("key3", "value3")

    assert cache.get("key1") == "value1"  # Should still exist
    assert cache.get("key2") is None  # Should be evicted
    assert cache.get("key3") == "value3"


def test_lru_cache_clear():
    """Test LRUCache clear operation"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=5)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert len(cache._cache) == 0


def test_lru_cache_capacity_property():
    """Test LRUCache capacity property"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=100)
    assert cache.capacity == 100


def test_query_caching_enabled():
    """Test that cache stores and returns results when enabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, LRUCache
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Create retriever with cache enabled
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    # Verify cache is initialized
    assert retriever._cache is not None
    assert isinstance(retriever._cache, LRUCache)
    assert retriever._cache.capacity == 100


def test_query_caching_disabled():
    """Test that cache is None when disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Create retriever with cache disabled (default)
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    # Verify cache is None
    assert retriever._cache is None


def test_query_caching_default_disabled():
    """Test that caching is disabled by default"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    # Cache should be None by default
    assert retriever._cache is None


def test_clear_cache_method():
    """Test clear_cache method clears the cache"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    # Add something to cache directly
    retriever._cache.put("test_key", "test_value")
    assert retriever._cache.get("test_key") == "test_value"

    # Clear cache
    retriever.clear_cache()

    # Cache should be empty
    assert retriever._cache.get("test_key") is None
    assert len(retriever._cache._cache) == 0


def test_clear_cache_when_disabled():
    """Test clear_cache doesn't error when cache is disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    # Should not raise error
    retriever.clear_cache()


def test_get_cache_stats_enabled():
    """Test get_cache_stats when cache is enabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    stats = retriever.get_cache_stats()

    assert stats["enabled"] is True
    assert stats["size"] == 0
    assert stats["capacity"] == 100

    # Add something to cache
    retriever._cache.put("key", "value")
    stats = retriever.get_cache_stats()
    assert stats["size"] == 1


def test_get_cache_stats_disabled():
    """Test get_cache_stats when cache is disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    stats = retriever.get_cache_stats()

    assert stats["enabled"] is False
    assert stats["size"] == 0
    assert stats["capacity"] == 0


def test_cache_key_generation():
    """Test cache key generation is deterministic"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )

    # Same inputs should produce same key
    key1 = retriever._generate_cache_key("test query", ["kw1", "kw2"])
    key2 = retriever._generate_cache_key("test query", ["kw1", "kw2"])
    assert key1 == key2

    # Different inputs should produce different keys
    key3 = retriever._generate_cache_key("test query", ["kw1"])
    assert key1 != key3

    key4 = retriever._generate_cache_key("different query", ["kw1", "kw2"])
    assert key1 != key4

    # Different version_id should produce different key
    retriever2 = HybridRetriever(
        version_id="v2",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )
    key5 = retriever2._generate_cache_key("test query", ["kw1", "kw2"])
    assert key1 != key5

    # Different top_k should produce different key
    retriever3 = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=5,
        enable_cache=True,
    )
    key6 = retriever3._generate_cache_key("test query", ["kw1", "kw2"])
    assert key1 != key6


def test_cache_key_with_none_keywords():
    """Test cache key generation with None keywords"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )

    # Should work with None keywords
    key = retriever._generate_cache_key("test query", None)
    assert isinstance(key, str)
    assert len(key) == 64  # SHA256 hex digest length


def test_cache_key_is_sha256():
    """Test that cache key is a valid SHA256 hash"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )

    key = retriever._generate_cache_key("test query", ["kw1"])

    # SHA256 hex digest should be 64 characters
    assert len(key) == 64
    # Should only contain hex characters
    assert all(c in "0123456789abcdef" for c in key)
