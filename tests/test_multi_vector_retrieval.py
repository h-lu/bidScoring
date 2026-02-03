"""Tests for multi-vector retrieval functionality."""

import pytest
import pytest_asyncio
import psycopg

from bid_scoring.config import load_settings
from bid_scoring.multi_vector_retrieval import (
    FallbackRetriever,
    MultiVectorRetriever,
    retrieve_with_fallback,
)
from bid_scoring.search import rrf_fuse


class TestMultiVectorRetriever:
    """Test MultiVectorRetriever class."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_init_default(self, retriever):
        """Test retriever initialization with defaults."""
        assert retriever.dsn is not None
        assert retriever.embedding_dim == 1536
        assert retriever.bm25_weight == 0.4
        assert retriever.vector_weight == 0.6
        assert retriever.rrf_k == 60

    def test_init_custom(self):
        """Test retriever initialization with custom values."""
        retriever = MultiVectorRetriever(
            dsn="postgresql://test",
            embedding_dim=3072,
            bm25_weight=0.5,
            vector_weight=0.5,
            rrf_k=40,
        )
        assert retriever.dsn == "postgresql://test"
        assert retriever.embedding_dim == 3072
        assert retriever.bm25_weight == 0.5
        assert retriever.vector_weight == 0.5
        assert retriever.rrf_k == 40

    def test_retrieval_modes(self, retriever):
        """Test supported retrieval modes."""
        expected_modes = ["child", "parent", "hierarchical", "hybrid"]
        assert retriever.RETRIEVAL_MODES == expected_modes

    @pytest.mark.asyncio
    async def test_retrieve_invalid_mode(self, retriever):
        """Test that invalid retrieval mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid retrieval_mode"):
            await retriever.retrieve("test query", retrieval_mode="invalid")

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_mode_empty_db(self, retriever):
        """Test hybrid retrieval with empty database returns empty list."""
        # Use a non-existent version_id to simulate empty results
        results = await retriever.retrieve(
            "nonexistent query xyz123",
            retrieval_mode="hybrid",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)
        # Should return empty list or fallback

    @pytest.mark.asyncio
    async def test_retrieve_child_mode(self, retriever):
        """Test child mode retrieval."""
        results = await retriever.retrieve(
            "test query",
            retrieval_mode="child",
            top_k=3,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_parent_mode(self, retriever):
        """Test parent mode retrieval."""
        results = await retriever.retrieve(
            "test query",
            retrieval_mode="parent",
            top_k=3,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_hierarchical_mode(self, retriever):
        """Test hierarchical mode retrieval."""
        results = await retriever.retrieve(
            "test query",
            retrieval_mode="hierarchical",
            top_k=3,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_without_rerank(self, retriever):
        """Test retrieval without reranking."""
        results = await retriever.retrieve(
            "test query",
            retrieval_mode="hybrid",
            top_k=5,
            rerank=False,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)


class TestMultiVectorRetrieverBM25Search:
    """Test BM25 search functionality."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_bm25_search_child_chunks(self, retriever, conn):
        """Test BM25 search on child chunks."""
        results = retriever._bm25_search(
            conn,
            query="test",
            version_id="00000000-0000-0000-0000-000000000000",
            top_k=5,
            search_parents=False,
        )
        assert isinstance(results, list)
        # Results should be (chunk_id, score) tuples
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_bm25_search_parent_chunks(self, retriever, conn):
        """Test BM25 search on parent chunks."""
        results = retriever._bm25_search(
            conn,
            query="test",
            version_id="00000000-0000-0000-0000-000000000000",
            top_k=5,
            search_parents=True,
        )
        assert isinstance(results, list)

    def test_bm25_search_no_version(self, retriever, conn):
        """Test BM25 search without version filter."""
        results = retriever._bm25_search(
            conn,
            query="test",
            version_id=None,
            top_k=5,
            search_parents=False,
        )
        assert isinstance(results, list)


class TestMultiVectorRetrieverVectorSearch:
    """Test vector search functionality."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_vector_search_child_chunks(self, retriever, conn):
        """Test vector search on child chunks."""
        # Create a dummy embedding
        query_embedding = [0.0] * 1536

        results = retriever._vector_search(
            conn,
            query_embedding=query_embedding,
            version_id="00000000-0000-0000-0000-000000000000",
            top_k=5,
            search_parents=False,
        )
        assert isinstance(results, list)

    def test_vector_search_parent_chunks(self, retriever, conn):
        """Test vector search on parent chunks."""
        query_embedding = [0.0] * 1536

        results = retriever._vector_search(
            conn,
            query_embedding=query_embedding,
            version_id="00000000-0000-0000-0000-000000000000",
            top_k=5,
            search_parents=True,
        )
        assert isinstance(results, list)


class TestMultiVectorRetrieverParentChunks:
    """Test parent chunk retrieval."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_get_parent_chunks_empty_list(self, retriever, conn):
        """Test getting parent chunks with empty list."""
        results = retriever._get_parent_chunks(conn, [])
        assert results == []

    def test_get_parent_chunks_invalid_ids(self, retriever, conn):
        """Test getting parent chunks with invalid IDs."""
        results = retriever._get_parent_chunks(
            conn, ["00000000-0000-0000-0000-000000000000"]
        )
        assert isinstance(results, list)


class TestMultiVectorRetrieverChunkDetails:
    """Test chunk detail retrieval."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_get_chunk_details_empty_list(self, retriever, conn):
        """Test getting chunk details with empty list."""
        results = retriever._get_chunk_details(conn, [])
        assert results == []

    def test_get_chunk_details_invalid_ids(self, retriever, conn):
        """Test getting chunk details with invalid IDs."""
        results = retriever._get_chunk_details(
            conn, ["00000000-0000-0000-0000-000000000000"]
        )
        assert isinstance(results, list)


class TestMultiVectorRetrieverReranking:
    """Test reranking functionality."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    def test_rerank_results_empty(self, retriever):
        """Test reranking with empty results."""
        results = retriever._rerank_results("query", [], top_k=5)
        assert results == []

    def test_rerank_results_basic(self, retriever):
        """Test basic reranking."""
        results = [
            {"chunk_id": "1", "text": "This is a test about python programming"},
            {"chunk_id": "2", "text": "Another document about java coding"},
            {"chunk_id": "3", "text": "Python is great for data science"},
        ]
        reranked = retriever._rerank_results("python programming", results, top_k=2)
        assert len(reranked) <= 2
        # The result with "python programming" should be first
        assert reranked[0]["chunk_id"] == "1"

    def test_rerank_results_with_empty_text(self, retriever):
        """Test reranking with empty text fields."""
        results = [
            {"chunk_id": "1", "text": ""},
            {"chunk_id": "2", "text": "Some content here"},
            {"chunk_id": "3", "text": None},
        ]
        reranked = retriever._rerank_results("query", results, top_k=3)
        assert len(reranked) <= 3

    def test_rerank_results_preserves_order_for_equal_scores(self, retriever):
        """Test that reranking preserves some order for equal scores."""
        results = [
            {"chunk_id": "1", "text": "Document one"},
            {"chunk_id": "2", "text": "Document two"},
        ]
        reranked = retriever._rerank_results("xyz123nonexistent", results, top_k=2)
        # Should still return results even with no matching terms
        assert len(reranked) == 2


class TestMultiVectorRetrieverHierarchicalSearch:
    """Test hierarchical search functionality."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_hierarchical_search_empty(self, retriever, conn):
        """Test hierarchical search with no matching results."""
        results = retriever._hierarchical_search(
            conn,
            query="xyz123nonexistent",
            version_id="00000000-0000-0000-0000-000000000000",
            top_k=5,
        )
        assert isinstance(results, list)

    def test_hierarchical_search_no_version(self, retriever, conn):
        """Test hierarchical search without version filter."""
        results = retriever._hierarchical_search(
            conn, query="test", version_id=None, top_k=5
        )
        assert isinstance(results, list)


class TestFallbackRetriever:
    """Test FallbackRetriever class."""

    @pytest.fixture
    def fallback(self):
        """Provide a FallbackRetriever instance."""
        return FallbackRetriever()

    @pytest.mark.asyncio
    async def test_fallback_retrieve(self, fallback):
        """Test basic fallback retrieval."""
        results = await fallback.retrieve(
            "test query",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_fallback_retrieve_no_version(self, fallback):
        """Test fallback retrieval without version filter."""
        results = await fallback.retrieve("test query", top_k=5, version_id=None)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_fallback_result_format(self, fallback):
        """Test fallback result format."""
        results = await fallback.retrieve(
            "xyz123nonexistent",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        # Results should have expected fields
        for result in results:
            assert "chunk_id" in result
            assert "text" in result
            assert "page_idx" in result
            assert "element_type" in result
            assert "score" in result
            assert "is_fallback" in result
            assert result["is_fallback"] is True


class TestRetrieveWithFallback:
    """Test the retrieve_with_fallback convenience function."""

    @pytest.mark.asyncio
    async def test_retrieve_with_fallback_basic(self):
        """Test retrieve_with_fallback function."""
        results = await retrieve_with_fallback(
            query="test query",
            retrieval_mode="hybrid",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_with_fallback_all_modes(self):
        """Test retrieve_with_fallback with all modes."""
        for mode in ["hybrid", "child", "parent", "hierarchical"]:
            results = await retrieve_with_fallback(
                query="test query",
                retrieval_mode=mode,
                top_k=3,
                version_id="00000000-0000-0000-0000-000000000000",
            )
            assert isinstance(results, list), f"Mode {mode} should return a list"


class TestHybridSearchIntegration:
    """Test hybrid search (BM25 + Vector) integration."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, retriever):
        """Test that hybrid search returns results."""
        # Use a non-existent query to test empty handling
        results = await retriever.retrieve(
            query="xyz123nonexistent",
            retrieval_mode="hybrid",
            top_k=5,
            rerank=False,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_rerank(self, retriever):
        """Test hybrid search with reranking."""
        results = await retriever.retrieve(
            query="test",
            retrieval_mode="hybrid",
            top_k=3,
            rerank=True,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)
        assert len(results) <= 3


class TestRerankingStrategies:
    """Test different reranking strategies."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    def test_rerank_preserves_top_k(self, retriever):
        """Test that reranking respects top_k limit."""
        results = [
            {"chunk_id": str(i), "text": f"Document {i} about python"}
            for i in range(20)
        ]
        reranked = retriever._rerank_results("python", results, top_k=5)
        assert len(reranked) == 5

    def test_rerank_boosts_exact_match(self, retriever):
        """Test that exact query match gets boosted."""
        results = [
            {"chunk_id": "1", "text": "Some document"},
            {"chunk_id": "2", "text": "machine learning is great"},
            {"chunk_id": "3", "text": "machine learning techniques"},
        ]
        reranked = retriever._rerank_results("machine learning", results, top_k=2)
        # Results with "machine learning" should be in top 2
        assert reranked[0]["chunk_id"] in ["2", "3"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def retriever(self):
        """Provide a MultiVectorRetriever instance."""
        return MultiVectorRetriever()

    @pytest.mark.asyncio
    async def test_empty_query(self, retriever):
        """Test retrieval with empty query."""
        results = await retriever.retrieve(
            "",
            retrieval_mode="hybrid",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_very_long_query(self, retriever):
        """Test retrieval with very long query."""
        long_query = "word " * 1000
        results = await retriever.retrieve(
            long_query,
            retrieval_mode="hybrid",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_special_characters_query(self, retriever):
        """Test retrieval with special characters."""
        special_query = "test!@#$%^&*()_+{}|:<>?[];',./\\"
        results = await retriever.retrieve(
            special_query,
            retrieval_mode="hybrid",
            top_k=5,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_top_k_zero(self, retriever):
        """Test retrieval with top_k=0."""
        results = await retriever.retrieve(
            "test",
            retrieval_mode="hybrid",
            top_k=0,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_large_top_k(self, retriever):
        """Test retrieval with large top_k."""
        results = await retriever.retrieve(
            "xyz123nonexistent",
            retrieval_mode="hybrid",
            top_k=1000,
            version_id="00000000-0000-0000-0000-000000000000",
        )
        assert isinstance(results, list)
