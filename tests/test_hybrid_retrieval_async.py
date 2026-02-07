import pytest

# =============================================================================
# Async Retrieval Tests (New from Task 7)
# =============================================================================


@pytest.mark.asyncio
async def test_retrieve_async_method_exists():
    """Test that retrieve_async method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    assert hasattr(retriever, "retrieve_async")
    assert callable(getattr(retriever, "retrieve_async"))


@pytest.mark.asyncio
async def test_close_async_method_exists():
    """Test that close_async method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    assert hasattr(retriever, "close_async")
    assert callable(getattr(retriever, "close_async"))


@pytest.mark.asyncio
async def test_close_async_closes_resources():
    """Test that close_async properly closes resources"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        use_connection_pool=True,
    )

    # Close should not raise error
    await retriever.close_async()

    # Pool should be None after close
    assert retriever._pool is None


@pytest.mark.asyncio
async def test_retrieve_async_basic():
    """Test basic async retrieval with real database"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训时长")
        assert isinstance(results, list)
        # Results should be RetrievalResult objects
        for result in results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "text")
            assert hasattr(result, "score")
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_with_keywords():
    """Test async retrieval with explicit keywords"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训时长", keywords=["培训", "时长"])
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_returns_list():
    """Test that retrieve_async returns a list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训")
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_with_cache():
    """Test async retrieval with caching enabled"""
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

    try:
        # First call
        results1 = await retriever.retrieve_async("培训时长", use_cache=True)

        # Second call should hit cache
        results2 = await retriever.retrieve_async("培训时长", use_cache=True)

        # Results should be identical (from cache)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.chunk_id == r2.chunk_id
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_without_cache():
    """Test async retrieval with cache disabled"""
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

    try:
        # Call with use_cache=False
        results = await retriever.retrieve_async("培训时长", use_cache=False)
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_cache_disabled_retriever():
    """Test async retrieval when retriever has cache disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    try:
        results = await retriever.retrieve_async("培训时长", use_cache=True)
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_same_params_as_sync():
    """Test that async version accepts same parameters as sync"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        # Test with all parameters
        results = await retriever.retrieve_async(
            query="培训时长",
            keywords=["培训", "时长"],
            use_cache=False,
        )
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_concurrent_calls():
    """Test that multiple async calls can run concurrently"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings
    import asyncio

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        # Run multiple queries concurrently
        queries = ["培训", "服务", "响应"]
        tasks = [retriever.retrieve_async(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # All should return lists
        for result_list in results:
            assert isinstance(result_list, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_result_format():
    """Test that async results have correct format"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, RetrievalResult
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训时长")

        for result in results:
            assert isinstance(result, RetrievalResult)
            assert isinstance(result.chunk_id, str)
            assert isinstance(result.text, str)
            assert isinstance(result.score, float)
            assert isinstance(result.page_idx, int)
            assert result.source in ["vector", "keyword", "hybrid"]
    finally:
        await retriever.close_async()


def test_retrieve_vector_only_rrf_denominator_matches_rrf_formula():
    """Vector-only fallback should use 1/(k+rank+1), same as RRF fusion."""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=2
    )

    retriever._vector_search = lambda query: [("chunk-1", 0.9), ("chunk-2", 0.8)]
    retriever._keyword_search_fulltext = lambda keywords, use_or_semantic: []

    captured = {}

    def fake_fetch_chunks(merged_results):
        captured["merged_results"] = merged_results
        return []

    retriever._fetch_chunks = fake_fetch_chunks

    retriever.retrieve("任意查询", keywords=["任意"])
    merged = captured["merged_results"]

    assert merged[0][1] == pytest.approx(1 / (retriever.rrf.k + 1), rel=1e-6)
    assert merged[1][1] == pytest.approx(1 / (retriever.rrf.k + 2), rel=1e-6)


def test_retrieve_applies_reranker_when_enabled():
    """retrieve() should apply reranker output when reranking is enabled."""
    from bid_scoring.hybrid_retrieval import HybridRetriever, RetrievalResult

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=3
    )

    retriever._vector_search = lambda query: [("chunk-1", 0.9), ("chunk-2", 0.8)]
    retriever._keyword_search_fulltext = lambda keywords, use_or_semantic: []

    base_results = [
        RetrievalResult("chunk-1", "text-1", 1, 0.2, "vector"),
        RetrievalResult("chunk-2", "text-2", 2, 0.1, "vector"),
    ]
    retriever._fetch_chunks = lambda merged_results: base_results

    class FakeReranker:
        def __init__(self):
            self.called = False
            self.query = None
            self.top_n = None

        def rerank(self, query, results, top_n):
            self.called = True
            self.query = query
            self.top_n = top_n
            return [results[1], results[0]]

    fake_reranker = FakeReranker()
    retriever._enable_rerank = True
    retriever._reranker = fake_reranker
    retriever._rerank_top_n = 2

    results = retriever.retrieve("培训时长", keywords=["培训"])

    assert fake_reranker.called is True
    assert fake_reranker.query == "培训时长"
    assert fake_reranker.top_n == 2
    assert [r.chunk_id for r in results] == ["chunk-2", "chunk-1"]


@pytest.mark.asyncio
async def test_retrieve_async_applies_reranker_when_enabled():
    """retrieve_async() should apply reranker output when reranking is enabled."""
    from bid_scoring.hybrid_retrieval import HybridRetriever, RetrievalResult

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=3
    )

    retriever._vector_search = lambda query: [("chunk-1", 0.9), ("chunk-2", 0.8)]
    retriever._keyword_search_fulltext = lambda keywords, use_or_semantic: []

    base_results = [
        RetrievalResult("chunk-1", "text-1", 1, 0.2, "vector"),
        RetrievalResult("chunk-2", "text-2", 2, 0.1, "vector"),
    ]
    retriever._fetch_chunks = lambda merged_results: base_results

    class FakeReranker:
        def __init__(self):
            self.called = False
            self.query = None
            self.top_n = None

        def rerank(self, query, results, top_n):
            self.called = True
            self.query = query
            self.top_n = top_n
            return [results[1], results[0]]

    fake_reranker = FakeReranker()
    retriever._enable_rerank = True
    retriever._reranker = fake_reranker
    retriever._rerank_top_n = 2

    results = await retriever.retrieve_async("培训时长", keywords=["培训"])

    assert fake_reranker.called is True
    assert fake_reranker.query == "培训时长"
    assert fake_reranker.top_n == 2
    assert [r.chunk_id for r in results] == ["chunk-2", "chunk-1"]


def test_colbert_backend_selected_when_enabled(monkeypatch):
    """HybridRetriever should build ColBERT reranker when backend=colbert."""
    import bid_scoring.hybrid_retrieval as hr
    import bid_scoring.retrieval.rerankers as rr

    class FakeColBERTReranker:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.model_name = model_name
            self.device = device

        def rerank(self, query, results, top_n):
            return results[:top_n]

    monkeypatch.setattr(rr, "HAS_COLBERT_RERANKER", True, raising=False)
    monkeypatch.setattr(rr, "ColBERTReranker", FakeColBERTReranker, raising=False)

    retriever = hr.HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5,
        enable_rerank=True,
        rerank_backend="colbert",
        rerank_model="colbert-ir/colbertv2.0",
    )

    assert isinstance(retriever._reranker, FakeColBERTReranker)
    assert retriever._enable_rerank is True


def test_retrieve_applies_colbert_reranker_when_selected(monkeypatch):
    """retrieve() should call ColBERT reranker when backend=colbert."""
    import bid_scoring.hybrid_retrieval as hr
    import bid_scoring.retrieval.rerankers as rr

    class FakeColBERTReranker:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.called = False
            self.query = None
            self.top_n = None

        def rerank(self, query, results, top_n):
            self.called = True
            self.query = query
            self.top_n = top_n
            return [results[1], results[0]]

    monkeypatch.setattr(rr, "HAS_COLBERT_RERANKER", True, raising=False)
    monkeypatch.setattr(rr, "ColBERTReranker", FakeColBERTReranker, raising=False)

    retriever = hr.HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=3,
        enable_rerank=True,
        rerank_backend="colbert",
        rerank_model="colbert-ir/colbertv2.0",
        rerank_top_n=2,
    )

    retriever._vector_search = lambda query: [("chunk-1", 0.9), ("chunk-2", 0.8)]
    retriever._keyword_search_fulltext = lambda keywords, use_or_semantic: []
    retriever._fetch_chunks = lambda merged_results: [
        hr.RetrievalResult("chunk-1", "text-1", 1, 0.2, "vector"),
        hr.RetrievalResult("chunk-2", "text-2", 2, 0.1, "vector"),
    ]

    results = retriever.retrieve("培训时长", keywords=["培训"])

    assert retriever._reranker.called is True
    assert retriever._reranker.query == "培训时长"
    assert retriever._reranker.top_n == 2
    assert [r.chunk_id for r in results] == ["chunk-2", "chunk-1"]


@pytest.mark.asyncio
async def test_retrieve_async_applies_colbert_reranker_when_selected(monkeypatch):
    """retrieve_async() should call ColBERT reranker when backend=colbert."""
    import bid_scoring.hybrid_retrieval as hr
    import bid_scoring.retrieval.rerankers as rr

    class FakeColBERTReranker:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.called = False
            self.query = None
            self.top_n = None

        def rerank(self, query, results, top_n):
            self.called = True
            self.query = query
            self.top_n = top_n
            return [results[1], results[0]]

    monkeypatch.setattr(rr, "HAS_COLBERT_RERANKER", True, raising=False)
    monkeypatch.setattr(rr, "ColBERTReranker", FakeColBERTReranker, raising=False)

    retriever = hr.HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=3,
        enable_rerank=True,
        rerank_backend="colbert",
        rerank_model="colbert-ir/colbertv2.0",
        rerank_top_n=2,
    )

    retriever._vector_search = lambda query: [("chunk-1", 0.9), ("chunk-2", 0.8)]
    retriever._keyword_search_fulltext = lambda keywords, use_or_semantic: []
    retriever._fetch_chunks = lambda merged_results: [
        hr.RetrievalResult("chunk-1", "text-1", 1, 0.2, "vector"),
        hr.RetrievalResult("chunk-2", "text-2", 2, 0.1, "vector"),
    ]

    results = await retriever.retrieve_async("培训时长", keywords=["培训"])

    assert retriever._reranker.called is True
    assert retriever._reranker.query == "培训时长"
    assert retriever._reranker.top_n == 2
    assert [r.chunk_id for r in results] == ["chunk-2", "chunk-1"]
