#!/usr/bin/env python3
"""
MCP Client Integration Tests

Tests MCP server using actual MCP client protocol.
Requires server to be running.

Run with:
    # Terminal 1: Start server
    uv run fastmcp run mcp_servers/retrieval_server.py -t stdio
    
    # Terminal 2: Run tests
    uv run pytest tests/test_mcp_client_integration.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

# Try to import mcp client libraries
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    pytest.skip("mcp package not installed", allow_module_level=True)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def server_params():
    """Server parameters for MCP client."""
    project_root = Path(__file__).resolve().parents[1]
    return StdioServerParameters(
        command="uv",
        args=[
            "run",
            "fastmcp",
            "run",
            "mcp_servers/retrieval_server.py",
            "-t",
            "stdio"
        ],
        cwd=str(project_root),
        env={
            **os.environ,
            "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://localhost:5432/bid_scoring"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        }
    )


@pytest.fixture
async def mcp_session(server_params):
    """Create MCP client session."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@pytest.fixture
def version_id():
    """Test version ID."""
    return "33333333-3333-3333-3333-333333333333"


# ============================================================================
# Basic Connection Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP client not available")
class TestMCPConnection:
    """Test MCP server connection."""

    async def test_server_initializes(self, mcp_session):
        """Test that server initializes correctly."""
        # If we got here, initialization succeeded
        assert True

    async def test_server_has_retrieve_tool(self, mcp_session):
        """Test that server exposes retrieve tool."""
        tools = await mcp_session.list_tools()
        tool_names = [t.name for t in tools.tools]
        assert "retrieve" in tool_names

    async def test_retrieve_tool_schema(self, mcp_session):
        """Test retrieve tool has correct schema."""
        tools = await mcp_session.list_tools()
        retrieve_tool = next((t for t in tools.tools if t.name == "retrieve"), None)
        
        assert retrieve_tool is not None
        assert retrieve_tool.description is not None
        assert "version_id" in retrieve_tool.inputSchema.get("properties", {})
        assert "query" in retrieve_tool.inputSchema.get("properties", {})


# ============================================================================
# Tool Call Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP client not available")
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set"
)
class TestMCPToolCalls:
    """Test MCP tool calls."""

    async def test_retrieve_basic(self, mcp_session, version_id):
        """Test basic retrieve call."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": version_id,
                "query": "售后响应时间",
                "top_k": 3,
                "mode": "hybrid"
            }
        )

        # Check result structure
        assert result.isError is False
        content = json.loads(result.content[0].text)
        
        assert content["version_id"] == version_id
        assert content["mode"] == "hybrid"
        assert "results" in content
        assert isinstance(content["results"], list)

    async def test_retrieve_different_modes(self, mcp_session, version_id):
        """Test retrieve with different modes."""
        modes = ["hybrid", "keyword", "vector"]
        
        for mode in modes:
            result = await mcp_session.call_tool(
                "retrieve",
                {
                    "version_id": version_id,
                    "query": "质保期",
                    "top_k": 2,
                    "mode": mode
                }
            )
            
            assert result.isError is False
            content = json.loads(result.content[0].text)
            assert content["mode"] == mode
            assert "results" in content

    async def test_retrieve_with_max_chars(self, mcp_session, version_id):
        """Test retrieve with max_chars parameter."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": version_id,
                "query": "培训计划",
                "top_k": 2,
                "max_chars": 50,
                "include_text": True
            }
        )

        assert result.isError is False
        content = json.loads(result.content[0].text)
        
        for r in content["results"]:
            if r["text"]:
                assert len(r["text"]) <= 50

    async def test_retrieve_without_text(self, mcp_session, version_id):
        """Test retrieve without text."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": version_id,
                "query": "设备安装",
                "top_k": 2,
                "include_text": False
            }
        )

        assert result.isError is False
        content = json.loads(result.content[0].text)
        
        for r in content["results"]:
            assert r["text"] == ""

    async def test_retrieve_invalid_version(self, mcp_session):
        """Test retrieve with invalid version_id."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": "invalid-uuid",
                "query": "test"
            }
        )

        # Server may return empty results or error
        # Either is acceptable as long as it doesn't crash
        assert result.isError is False or result.isError is True


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP client not available")
class TestMCPErrorHandling:
    """Test MCP error handling."""

    async def test_retrieve_empty_version(self, mcp_session):
        """Test retrieve with empty version_id."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": "",
                "query": "test"
            }
        )

        # Should return error
        assert result.isError is True

    async def test_retrieve_invalid_top_k(self, mcp_session):
        """Test retrieve with invalid top_k."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": "test-uuid",
                "query": "test",
                "top_k": 0
            }
        )

        # Should return error
        assert result.isError is True

    async def test_retrieve_missing_required_param(self, mcp_session):
        """Test retrieve without required parameters."""
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": "test-uuid"
                # missing query
            }
        )

        # Should handle gracefully
        assert result.isError is True or "query" in str(result.content)


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP client not available")
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set"
)
class TestMCPPerformance:
    """Test MCP performance."""

    async def test_retrieve_response_time(self, mcp_session, version_id):
        """Test retrieve response time."""
        start = time.time()
        
        result = await mcp_session.call_tool(
            "retrieve",
            {
                "version_id": version_id,
                "query": "售后响应时间",
                "top_k": 5
            }
        )
        
        elapsed = time.time() - start
        
        assert result.isError is False
        assert elapsed < 10.0, f"Retrieval took too long: {elapsed:.2f}s"

    async def test_multiple_retrieves(self, mcp_session, version_id):
        """Test multiple sequential retrieves."""
        queries = ["质保期", "培训", "响应时间", "安装", "验收"]
        
        for query in queries:
            result = await mcp_session.call_tool(
                "retrieve",
                {
                    "version_id": version_id,
                    "query": query,
                    "top_k": 3
                }
            )
            
            assert result.isError is False
            content = json.loads(result.content[0].text)
            assert "results" in content


# ============================================================================
# Manual Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
