# =============================================================================
# Fulltext Search Tests (New from Task 2)
# =============================================================================


def test_fulltext_search_method_exists():
    """Test that _keyword_search_fulltext method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Verify method exists
    assert hasattr(retriever, "_keyword_search_fulltext")
    assert hasattr(retriever, "_keyword_search_legacy")


def test_fulltext_search_empty_keywords():
    """Test fulltext search with empty keywords returns empty list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Empty keywords should return empty list
    result = retriever._keyword_search_fulltext([])
    assert result == []


def test_legacy_keyword_search_empty_keywords():
    """Test legacy keyword search with empty keywords returns empty list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Empty keywords should return empty list
    result = retriever._keyword_search_legacy([])
    assert result == []


def test_keyword_extraction_for_fulltext():
    """Test keyword extraction produces valid input for fulltext search"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Test that keywords are extracted correctly
    keywords = retriever.extract_keywords_from_query("培训时长和服务响应")

    # Should contain field keywords
    assert "培训" in keywords
    assert "时长" in keywords
    assert "服务" in keywords
    assert "响应" in keywords


def test_fulltext_search_integration():
    """Integration test for fulltext search with real database"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    # Extract keywords
    keywords = retriever.extract_keywords_from_query("培训")
    assert "培训" in keywords

    # Fulltext search should return results
    results = retriever._keyword_search_fulltext(["培训"])

    # Verify results format
    for chunk_id, score in results:
        assert isinstance(chunk_id, str)
        assert isinstance(score, float)
        assert score >= 0  # ts_rank_cd returns non-negative values


def test_fulltext_search_uses_ilikewith_pg_trgm():
    """Fulltext search now uses ILIKE with pg_trgm GIN index for Chinese text.

    Note: The 'simple' text search configuration does not properly tokenize Chinese
    text, so we use ILIKE pattern matching with pg_trgm GIN index which works
    reliably and efficiently for Chinese keywords.
    """
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    executed_sql = []

    class FakeCursor:
        def execute(self, sql, params=None):
            executed_sql.append((sql, params))

        def fetchall(self):
            return [("chunk-1", 1.0)]

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

    results = retriever._keyword_search_fulltext(["培训", "时长"], use_or_semantic=True)

    assert results == [("chunk-1", 1.0)]
    full_sql = " ".join(sql for sql, _ in executed_sql)
    # Should use ILIKE for Chinese keyword matching with pg_trgm GIN index
    assert "text_raw ILIKE" in full_sql
    assert "FROM chunks" in full_sql


def test_fulltext_search_handles_stopwords_gracefully():
    """Keyword search should handle stopwords gracefully with ILIKE matching."""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    executed_sql = []

    class FakeCursor:
        def execute(self, sql, params=None):
            executed_sql.append((sql, params))

        def fetchall(self):
            # Return empty results for stopword-only queries
            return []

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

    # Even with stopwords, the search should execute (ILIKE doesn't have querytree limitation)
    results = retriever._keyword_search_fulltext(["的", "了"], use_or_semantic=True)

    assert results == []
    # Should still execute the query
    assert sum("FROM chunks" in sql for sql, _ in executed_sql) >= 1
