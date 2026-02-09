#!/usr/bin/env python3
"""
Test script for new MCP tools

Usage:
    uv run python scripts/test_mcp_new_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_all_tools():
    """Test all MCP tools."""

    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "python",
            "-c",
            "from mcp_servers.retrieval_server import mcp; mcp.run(transport='stdio')",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    print("🚀 Connecting to MCP server...")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"📋 Available tools: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  • {tool.name}")
            print()

            version_id = "33333333-3333-3333-3333-333333333333"

            # Test 1: list_available_versions
            print("🔍 Test 1: list_available_versions")
            result = await session.call_tool(
                "list_available_versions", {"include_stats": True}
            )
            content = result.content[0].text
            import json

            data = json.loads(content)
            print(f"   ✅ Found {data.get('count', 0)} versions")
            print()

            # Test 2: get_document_outline
            print("🔍 Test 2: get_document_outline")
            result = await session.call_tool(
                "get_document_outline", {"version_id": version_id, "max_depth": 2}
            )
            data = json.loads(result.content[0].text)
            outline_count = len(data.get("outline", []))
            print(f"   ✅ Found {outline_count} outline items")
            print()

            # Test 3: search_chunks
            print("🔍 Test 3: search_chunks")
            result = await session.call_tool(
                "search_chunks",
                {
                    "version_id": version_id,
                    "query": "售后响应时间",
                    "top_k": 2,
                    "mode": "hybrid",
                },
            )
            data = json.loads(result.content[0].text)
            results_count = len(data.get("results", []))
            print(f"   ✅ Found {results_count} results")
            if data.get("results"):
                print(f"   📝 Top result: {data['results'][0]['text'][:80]}...")
            print()

            # Test 4: batch_search
            print("🔍 Test 4: batch_search")
            result = await session.call_tool(
                "batch_search",
                {
                    "version_id": version_id,
                    "queries": ["质保期", "响应时间", "培训天数"],
                    "top_k_per_query": 2,
                    "aggregate_by": "query",
                },
            )
            data = json.loads(result.content[0].text)
            print("   ✅ Batch search complete")
            print(f"   📊 Total results: {data.get('total_results', 0)}")
            print()

            # Test 5: compare_across_versions
            print("🔍 Test 5: compare_across_versions")
            result = await session.call_tool(
                "compare_across_versions",
                {
                    "version_ids": [
                        "33333333-3333-3333-3333-333333333333",
                        "44444444-4444-4444-4444-444444444444",
                    ],
                    "query": "售后响应时间",
                    "top_k_per_version": 2,
                },
            )
            data = json.loads(result.content[0].text)
            print(f"   ✅ Compared across {data.get('version_count', 0)} versions")
            print()

            # Test 6: extract_key_value
            print("🔍 Test 6: extract_key_value")
            result = await session.call_tool(
                "extract_key_value",
                {
                    "version_id": version_id,
                    "key_patterns": ["质保期", "保修期"],
                    "value_patterns": ["年", "月"],
                    "context_window": 30,
                },
            )
            data = json.loads(result.content[0].text)
            if isinstance(data, list):
                print(f"   ✅ Extracted {len(data)} key-value pairs")
                for item in data[:2]:
                    print(f"   📝 {item.get('key')}: {item.get('value', 'N/A')}")
            print()

            print("=" * 60)
            print("✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_all_tools())
