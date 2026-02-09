#!/usr/bin/env python3
"""
Quick MCP Server Test Script

Run without pytest for quick validation:
    uv run python scripts/test_mcp_quick.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_servers.retrieval_server import retrieve_impl


def test_basic_retrieval():
    """Test basic retrieval functionality."""
    print("=" * 60)
    print("MCP Server Quick Test")
    print("=" * 60)

    version_id = "33333333-3333-3333-3333-333333333333"
    test_queries = [
        ("售后响应时间", "hybrid"),
        ("质保期", "keyword"),
        ("培训计划", "vector"),
    ]

    all_passed = True

    for query, mode in test_queries:
        print(f"\n📝 Query: '{query}' (Mode: {mode})")

        try:
            start = time.time()
            result = retrieve_impl(
                version_id=version_id,
                query=query,
                top_k=3,
                mode=mode,
                include_text=True,
                max_chars=100,
            )
            elapsed = time.time() - start

            # Validate result structure
            assert "version_id" in result, "Missing version_id"
            assert "query" in result, "Missing query"
            assert "mode" in result, "Missing mode"
            assert "results" in result, "Missing results"
            assert isinstance(result["results"], list), "Results should be a list"

            print(f"   ✅ Success ({elapsed:.2f}s)")
            print(f"   📊 Found {len(result['results'])} results")

            if result["results"]:
                for i, r in enumerate(result["results"][:2], 1):
                    print(f"   {i}. Page {r['page_idx']}: {r['text'][:60]}...")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return all_passed


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Error Handling Test")
    print("=" * 60)

    test_cases = [
        ("Empty version_id", {"version_id": "", "query": "test"}, ValueError),
        ("Zero top_k", {"version_id": "test", "query": "test", "top_k": 0}, ValueError),
        (
            "Negative top_k",
            {"version_id": "test", "query": "test", "top_k": -1},
            ValueError,
        ),
    ]

    all_passed = True

    for name, params, expected_error in test_cases:
        print(f"\n🧪 {name}")
        try:
            retrieve_impl(**params)
            print(f"   ❌ Should have raised {expected_error.__name__}")
            all_passed = False
        except expected_error as e:
            print(f"   ✅ Correctly raised {expected_error.__name__}: {e}")
        except Exception as e:
            print(f"   ❌ Wrong exception type: {type(e).__name__}: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All error handling tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return all_passed


def main():
    """Run all quick tests."""
    print("\n🚀 Starting MCP Server Quick Tests\n")

    test1 = test_basic_retrieval()
    test2 = test_error_handling()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
