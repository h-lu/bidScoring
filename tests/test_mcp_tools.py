# tests/test_mcp_tools.py
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
