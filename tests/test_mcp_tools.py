# tests/test_mcp_tools.py
from unittest.mock import patch, MagicMock
import pytest

from mcp_servers.bid_documents.server import (
    search_chunks,
    get_document_info,
    get_page_content,
)


def test_search_chunks_is_tool():
    """Verify search_chunks is registered as a FunctionTool."""
    assert search_chunks.name == "search_chunks"
    assert callable(search_chunks.fn)


def test_get_document_info_is_tool():
    """Verify get_document_info is registered as a FunctionTool."""
    assert get_document_info.name == "get_document_info"
    assert callable(get_document_info.fn)


def test_get_page_content_is_tool():
    """Verify get_page_content is registered as a FunctionTool."""
    assert get_page_content.name == "get_page_content"
    assert callable(get_page_content.fn)


def test_search_chunks_default_params():
    """Verify search_chunks has correct default parameters."""
    import inspect
    sig = inspect.signature(search_chunks.fn)
    params = sig.parameters

    # Check retrieval_mode defaults to 'standard'
    assert params["retrieval_mode"].default == "standard"
    # Check top_k defaults to 10
    assert params["top_k"].default == 10


def test_search_chunks_accepts_cpc_mode():
    """Verify search_chunks accepts 'cpc' retrieval_mode."""
    import inspect
    from typing import get_origin, get_args

    sig = inspect.signature(search_chunks.fn)
    retrieval_mode_param = sig.parameters["retrieval_mode"]

    # Check it's a Literal type with 'standard' and 'cpc' options
    annotation = retrieval_mode_param.annotation
    if hasattr(annotation, "__args__"):
        assert "standard" in annotation.__args__
        assert "cpc" in annotation.__args__


@patch("mcp_servers.bid_documents.server.register_vector")
@patch("mcp_servers.bid_documents.server.psycopg.connect")
@patch("mcp_servers.bid_documents.server.embed_texts")
@patch("mcp_servers.bid_documents.server.rrf_fuse")
@patch("mcp_servers.bid_documents.server.load_settings")
def test_search_chunks_standard_mode(
    mock_load_settings, mock_rrf_fuse, mock_embed_texts, mock_connect, mock_register_vector
):
    """Test search_chunks in standard mode."""
    # Setup mocks
    mock_load_settings.return_value = {"DATABASE_URL": "postgresql://test"}
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    mock_rrf_fuse.return_value = ["chunk-1", "chunk-2"]

    # Mock database connection
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=None)
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=None)

    # Mock version_id query
    mock_cur.fetchone.return_value = ("version-123",)

    # Call search_chunks with explicit standard mode
    result = search_chunks.fn(
        query="test query",
        document_id="doc-123",
        retrieval_mode="standard"
    )

    # Verify result structure
    assert "query" in result
    assert "results" in result
    assert result["query"] == "test query"
    assert result["results"] == ["chunk-1", "chunk-2"]


import asyncio


@patch("mcp_servers.bid_documents.server.MultiVectorRetriever")
@patch("mcp_servers.bid_documents.server.psycopg.connect")
@patch("mcp_servers.bid_documents.server.load_settings")
def test_search_chunks_cpc_mode(mock_load_settings, mock_connect, mock_retriever_class):
    """Test search_chunks in cpc mode uses MultiVectorRetriever."""
    # Setup mocks
    mock_load_settings.return_value = {"DATABASE_URL": "postgresql://test"}

    # Mock database connection for version lookup
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=None)
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=None)

    # Mock version_id query
    mock_cur.fetchone.return_value = ("version-123",)

    # Mock MultiVectorRetriever - create an async mock for retrieve
    async def mock_retrieve(*args, **kwargs):
        return [
            {"chunk_id": "parent-1", "text": "Parent chunk content", "page_idx": 1}
        ]

    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever
    mock_retriever.retrieve = mock_retrieve

    # Call search_chunks with cpc mode
    result = search_chunks.fn(
        query="test query",
        document_id="doc-123",
        retrieval_mode="cpc",
        top_k=5
    )

    # Verify MultiVectorRetriever was instantiated
    mock_retriever_class.assert_called_once()

    # Verify result structure
    assert "query" in result
    assert "results" in result
    assert result["query"] == "test query"
    assert len(result["results"]) == 1
    assert result["results"][0]["chunk_id"] == "parent-1"
