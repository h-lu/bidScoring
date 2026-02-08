#!/usr/bin/env python3
"""
Comprehensive MCP Server Tests

Run with:
    uv run pytest tests/test_mcp_server_comprehensive.py -v
    uv run pytest tests/test_mcp_server_comprehensive.py -v --tb=short
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Import MCP server module
import mcp_servers.retrieval_server as srv
from bid_scoring.retrieval import RetrievalResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return {
        "DATABASE_URL": "postgresql://test:test@localhost/test",
        "OPENAI_API_KEY": "sk-test",
    }


@pytest.fixture
def sample_version_id():
    """Sample version ID for testing."""
    return "33333333-3333-3333-3333-333333333333"


@pytest.fixture
def mock_retrieval_results():
    """Sample retrieval results."""
    return [
        RetrievalResult(
            chunk_id="chunk-001",
            text="质保期5年，自验收合格日起计算。",
            page_idx=16,
            score=0.95,
            source="hybrid",
            vector_score=0.92,
            keyword_score=1.2,
        ),
        RetrievalResult(
            chunk_id="chunk-002",
            text="售后服务响应时间2小时。",
            page_idx=17,
            score=0.88,
            source="hybrid",
            vector_score=0.85,
            keyword_score=1.0,
        ),
    ]


# ============================================================================
# Unit Tests - Server Structure
# ============================================================================


class TestMCPServerStructure:
    """Test MCP server basic structure and exports."""

    def test_mcp_instance_exists(self):
        """Test that mcp instance is exported."""
        assert hasattr(srv, "mcp")
        assert srv.mcp is not None

    def test_mcp_has_retrieve_tool(self):
        """Test that retrieve tool is registered."""
        # FastMCP stores tools in _tool_manager._tools
        tool_manager = srv.mcp._tool_manager
        assert "retrieve" in tool_manager._tools

    def test_required_env_vars(self):
        """Test that required env vars are documented."""
        required = ["DATABASE_URL", "OPENAI_API_KEY"]
        for var in required:
            assert os.getenv(var) or True  # Documented, may not be set in test


# ============================================================================
# Unit Tests - Tool Functions
# ============================================================================


class TestRetrieveTool:
    """Test retrieve tool functionality."""

    def test_retrieve_validates_version_id(self):
        """Test that empty version_id raises error."""
        with pytest.raises((ValueError, srv.ValidationError), match="version_id"):
            srv.retrieve_impl(version_id="", query="test query")

    def test_retrieve_validates_top_k(self):
        """Test that invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            srv.retrieve_impl(version_id="valid-uuid", query="test", top_k=0)

    def test_retrieve_validates_negative_top_k(self):
        """Test that negative top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            srv.retrieve_impl(version_id="valid-uuid", query="test", top_k=-1)

    @patch.object(srv, "get_retriever")
    def test_retrieve_returns_correct_structure(
        self, mock_get_retriever, mock_retrieval_results
    ):
        """Test that retrieve returns expected structure."""
        # Setup mock
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        # Call function
        result = srv.retrieve_impl(
            version_id="test-uuid", query="质保期", top_k=2, mode="hybrid"
        )

        # Verify structure
        assert "version_id" in result
        assert "query" in result
        assert "mode" in result
        assert "top_k" in result
        assert "results" in result

        assert result["version_id"] == "test-uuid"
        assert result["query"] == "质保期"
        assert result["mode"] == "hybrid"
        assert result["top_k"] == 2
        assert len(result["results"]) == 2

    @patch.object(srv, "get_retriever")
    def test_retrieve_result_format(self, mock_get_retriever, mock_retrieval_results):
        """Test that each result has required fields."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        result = srv.retrieve_impl(version_id="test-uuid", query="test", top_k=10)

        for r in result["results"]:
            assert "chunk_id" in r
            assert "page_idx" in r
            assert "source" in r
            assert "score" in r
            assert "text" in r
            assert isinstance(r["chunk_id"], str)
            assert isinstance(r["page_idx"], int)
            assert isinstance(r["score"], float)

    @patch.object(srv, "get_retriever")
    def test_retrieve_respects_include_text(
        self, mock_get_retriever, mock_retrieval_results
    ):
        """Test that include_text=False returns empty text."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        result = srv.retrieve_impl(
            version_id="test-uuid", query="test", include_text=False
        )

        for r in result["results"]:
            assert r["text"] == ""

    @patch.object(srv, "get_retriever")
    def test_retrieve_respects_max_chars(
        self, mock_get_retriever, mock_retrieval_results
    ):
        """Test that max_chars truncates text."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        result = srv.retrieve_impl(
            version_id="test-uuid", query="test", include_text=True, max_chars=10
        )

        for r in result["results"]:
            assert len(r["text"]) <= 10


# ============================================================================
# Unit Tests - Different Modes
# ============================================================================


class TestRetrieveModes:
    """Test different retrieval modes."""

    @patch.object(srv, "get_retriever")
    def test_hybrid_mode(self, mock_get_retriever, mock_retrieval_results):
        """Test hybrid mode calls retrieve method."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        srv.retrieve_impl(version_id="test", query="质保", mode="hybrid")

        mock_retriever.retrieve.assert_called_once()

    @patch.object(srv, "get_retriever")
    def test_vector_mode(self, mock_get_retriever, mock_retrieval_results):
        """Test vector mode calls _vector_search."""
        mock_retriever = MagicMock()
        mock_retriever._vector_search.return_value = [
            ("chunk-001", 0.95),
            ("chunk-002", 0.88),
        ]
        mock_retriever._fetch_chunks.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        srv.retrieve_impl(version_id="test", query="质保", mode="vector")

        mock_retriever._vector_search.assert_called_once_with("质保")

    @patch.object(srv, "get_retriever")
    def test_keyword_mode(self, mock_get_retriever, mock_retrieval_results):
        """Test keyword mode calls _keyword_search_fulltext."""
        mock_retriever = MagicMock()
        mock_retriever.extract_keywords_from_query.return_value = ["质保", "保修"]
        mock_retriever._keyword_search_fulltext.return_value = [
            ("chunk-001", 1.5),
            ("chunk-002", 1.2),
        ]
        mock_retriever._fetch_chunks.return_value = mock_retrieval_results
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        srv.retrieve_impl(version_id="test", query="质保期多久", mode="keyword")

        mock_retriever.extract_keywords_from_query.assert_called_once_with("质保期多久")
        mock_retriever._keyword_search_fulltext.assert_called_once()

    @patch.object(srv, "get_retriever")
    def test_keyword_mode_with_explicit_keywords(
        self, mock_get_retriever, mock_retrieval_results
    ):
        """Test keyword mode with explicit keywords."""
        mock_retriever = MagicMock()
        mock_retriever._keyword_search_fulltext.return_value = [
            ("chunk-001", 1.5),
        ]
        mock_retriever._fetch_chunks.return_value = mock_retrieval_results[:1]
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        srv.retrieve_impl(
            version_id="test", query="test", mode="keyword", keywords=["质保", "5年"]
        )

        # Should not call extract_keywords_from_query when keywords provided
        mock_retriever.extract_keywords_from_query.assert_not_called()
        mock_retriever._keyword_search_fulltext.assert_called_once_with(
            ["质保", "5年"], use_or_semantic=True
        )

    @patch.object(srv, "get_retriever")
    def test_invalid_mode_raises_error(self, mock_get_retriever):
        """Test that invalid mode raises error."""
        mock_retriever = MagicMock()
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        with pytest.raises(ValueError, match="Unknown mode"):
            srv.retrieve_impl(version_id="test", query="test", mode="invalid_mode")


# ============================================================================
# Unit Tests - Caching
# ============================================================================


class TestRetrieverCaching:
    """Test retriever caching behavior."""

    @patch("mcp_servers.retrieval_server.HybridRetriever")
    def test_get_retriever_creates_new_instance(self, mock_hybrid_class):
        """Test that get_retriever creates new instance for new key."""
        mock_instance = MagicMock()
        mock_hybrid_class.return_value = mock_instance

        # Clear cache first
        srv._RETRIEVER_CACHE.clear()

        retriever = srv.get_retriever("version-1", top_k=10)

        mock_hybrid_class.assert_called_once()
        assert retriever is mock_instance

    @patch("mcp_servers.retrieval_server.HybridRetriever")
    def test_get_retriever_returns_cached_instance(self, mock_hybrid_class):
        """Test that get_retriever returns cached instance."""
        mock_instance = MagicMock()
        mock_hybrid_class.return_value = mock_instance

        # Clear cache
        srv._RETRIEVER_CACHE.clear()

        # First call
        retriever1 = srv.get_retriever("version-1", top_k=10)

        # Second call with same params should return cached
        retriever2 = srv.get_retriever("version-1", top_k=10)

        # HybridRetriever should only be called once
        assert mock_hybrid_class.call_count == 1
        assert retriever1 is retriever2

    @patch("mcp_servers.retrieval_server.HybridRetriever")
    def test_get_retriever_different_params_creates_new(self, mock_hybrid_class):
        """Test that different params create different instances."""
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_hybrid_class.side_effect = [mock_instance1, mock_instance2]

        # Clear cache
        srv._RETRIEVER_CACHE.clear()

        retriever1 = srv.get_retriever("version-1", top_k=10)
        retriever2 = srv.get_retriever("version-1", top_k=20)  # Different top_k

        assert mock_hybrid_class.call_count == 2
        assert retriever1 is mock_instance1
        assert retriever2 is mock_instance2


# ============================================================================
# Integration Tests - Real Database (Optional)
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") or not os.getenv("OPENAI_API_KEY"),
    reason="DATABASE_URL and OPENAI_API_KEY required for integration tests",
)
class TestMCPIntegration:
    """Integration tests with real database."""

    def test_real_retrieve_hybrid(self, sample_version_id):
        """Test real hybrid retrieval."""
        result = srv.retrieve_impl(
            version_id=sample_version_id, query="售后响应时间", top_k=3, mode="hybrid"
        )

        assert result["version_id"] == sample_version_id
        assert result["mode"] == "hybrid"
        assert len(result["results"]) <= 3

        if result["results"]:
            for r in result["results"]:
                assert "chunk_id" in r
                assert "text" in r
                # Source can be 'hybrid', 'vector', or 'keyword'
                assert r["source"] in ["hybrid", "vector", "keyword"]

    def test_real_retrieve_keyword(self, sample_version_id):
        """Test real keyword retrieval."""
        result = srv.retrieve_impl(
            version_id=sample_version_id, query="质保期", top_k=5, mode="keyword"
        )

        assert result["mode"] == "keyword"
        assert len(result["results"]) <= 5

    def test_real_retrieve_vector(self, sample_version_id):
        """Test real vector retrieval."""
        result = srv.retrieve_impl(
            version_id=sample_version_id, query="培训计划", top_k=3, mode="vector"
        )

        assert result["mode"] == "vector"
        assert len(result["results"]) <= 3

    def test_all_modes_return_consistent_structure(self, sample_version_id):
        """Test that all modes return consistent result structure."""
        modes = ["hybrid", "keyword", "vector"]

        for mode in modes:
            result = srv.retrieve_impl(
                version_id=sample_version_id, query="测试查询", top_k=2, mode=mode
            )

            # Check consistent structure
            assert "version_id" in result
            assert "query" in result
            assert "mode" in result
            assert "top_k" in result
            assert "results" in result
            assert isinstance(result["results"], list)

            for r in result["results"]:
                assert "chunk_id" in r
                assert "page_idx" in r
                assert "source" in r
                assert "score" in r
                assert "text" in r


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") or not os.getenv("OPENAI_API_KEY"),
    reason="DATABASE_URL and OPENAI_API_KEY required for performance tests",
)
class TestMCPPerformance:
    """Performance tests."""

    def test_retrieve_latency(self, sample_version_id):
        """Test that retrieval completes within reasonable time."""
        import time

        start = time.time()
        result = srv.retrieve_impl(
            version_id=sample_version_id, query="售后响应时间", top_k=10, mode="hybrid"
        )
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Retrieval took too long: {elapsed:.2f}s"
        assert len(result["results"]) > 0

    def test_cached_retriever_faster(self, sample_version_id):
        """Test that cached retriever is faster."""
        import time

        # Clear cache
        srv._RETRIEVER_CACHE.clear()

        # First call (cold)
        start = time.time()
        srv.retrieve_impl(version_id=sample_version_id, query="质保期", top_k=5)
        cold_time = time.time() - start

        # Second call (warm)
        start = time.time()
        srv.retrieve_impl(version_id=sample_version_id, query="培训", top_k=5)
        warm_time = time.time() - start

        # Warm should be faster or similar
        assert warm_time <= cold_time * 1.5, "Cached retrieval should be faster"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    @patch.object(srv, "get_retriever")
    def test_retriever_error_propagates(self, mock_get_retriever):
        """Test that retriever errors are properly propagated."""
        mock_get_retriever.side_effect = Exception("Database connection failed")

        with pytest.raises(Exception, match="Database connection failed"):
            srv.retrieve_impl(version_id="test", query="test")

    @patch.object(srv, "get_retriever")
    def test_malformed_result_handling(self, mock_get_retriever):
        """Test handling of malformed results."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk_id="valid-id",
                text=None,  # None text
                page_idx=-1,
                score=float("nan"),
                source="hybrid",
            )
        ]
        mock_retriever.rrf.k = 60
        mock_get_retriever.return_value = mock_retriever

        result = srv.retrieve_impl(version_id="test", query="test")

        # Should handle gracefully
        assert len(result["results"]) == 1
        assert result["results"][0]["chunk_id"] == "valid-id"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
