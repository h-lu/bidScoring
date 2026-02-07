# =============================================================================
# Connection Pool Tests (New from Task 3)
# =============================================================================


def test_connection_pool_initialization():
    """Test connection pool initialization with default settings"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, HAS_CONNECTION_POOL
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=True
    )

    if HAS_CONNECTION_POOL:
        # Should have pool if psycopg-pool is installed
        assert retriever._pool is not None
    else:
        # Should be None if psycopg-pool not installed
        assert retriever._pool is None


def test_connection_pool_disabled():
    """Test that connection pool can be disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=False
    )

    # Pool should be None when disabled
    assert retriever._pool is None


def test_get_connection_method_exists():
    """Test that _get_connection method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    assert hasattr(retriever, "_get_connection")
    assert callable(getattr(retriever, "_get_connection"))


def test_close_method_exists():
    """Test that close method exists and works"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=True
    )

    assert hasattr(retriever, "close")

    # Close should not raise error
    retriever.close()

    # Pool should be None after close
    assert retriever._pool is None


def test_context_manager():
    """Test that HybridRetriever works as context manager"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Test context manager
    with HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=True
    ) as retriever:
        assert retriever.version_id == "test"
        # Pool should exist inside context

    # Pool should be closed after exiting context


def test_custom_pool_sizes():
    """Test custom pool size configuration"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Should accept custom pool sizes
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        use_connection_pool=True,
        pool_min_size=1,
        pool_max_size=5,
    )

    # Just verify initialization doesn't fail
    assert retriever.version_id == "test"


# =============================================================================
# HNSW Index Parameter Tests (New from Task 4)
# =============================================================================


def test_hnsw_ef_search_configuration():
    """Test that hnsw_ef_search parameter is properly configured"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Test with default value (100)
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)
    assert retriever._hnsw_ef_search == 100

    # Test with custom value
    retriever_custom = HybridRetriever(
        version_id="test", settings=settings, top_k=5, hnsw_ef_search=200
    )
    assert retriever_custom._hnsw_ef_search == 200


def test_vector_search_sets_hnsw_ef_search_without_bind_params(monkeypatch):
    """SET hnsw.ef_search should not use bind params in psycopg3."""
    import bid_scoring.hybrid_retrieval as hr
    import bid_scoring.retrieval.search_vector as sv

    retriever = hr.HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    monkeypatch.setattr(sv, "embed_single_text", lambda _: [0.1, 0.2, 0.3])

    executed_sql: list[tuple[str, object]] = []

    class FakeCursor:
        def execute(self, sql, params=None):
            executed_sql.append((str(sql), params))

        def fetchall(self):
            return [("chunk-1", 0.99)]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    retriever._get_connection = lambda: FakeConnection()

    results = retriever._vector_search("售后响应时间")
    assert results == [("chunk-1", 0.99)]

    hnsw_set_calls = [call for call in executed_sql if "SET hnsw.ef_search" in call[0]]
    assert len(hnsw_set_calls) == 1
    set_sql, set_params = hnsw_set_calls[0]
    assert "SET hnsw.ef_search = %s" not in set_sql
    assert set_params is None
