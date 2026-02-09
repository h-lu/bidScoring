#!/usr/bin/env python3
"""
Interactive MCP Retrieval Demo

Usage:
    uv run python scripts/mcp_demo.py --version-id 33333333-3333-3333-3333-333333333333
    uv run python scripts/mcp_demo.py --scenario A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_servers.retrieval_server import retrieve_impl


def print_result(result: dict, show_full_text: bool = False) -> None:
    """Pretty print retrieval results."""
    print(f"\n{'=' * 60}")
    print(f"Query: {result['query']}")
    print(f"Mode: {result['mode']} | Top-K: {result['top_k']}")
    print(f"Version ID: {result['version_id']}")
    print(f"{'=' * 60}")

    for i, r in enumerate(result["results"], 1):
        print(f"\n--- Result {i} ---")
        print(f"  Chunk ID: {r['chunk_id']}")
        print(f"  Page: {r['page_idx']}")
        print(f"  Source: {r['source']}")
        print(f"  Score: {r['score']:.4f}")
        if r["vector_score"] is not None:
            print(f"  Vector Score: {r['vector_score']:.4f}")
        if r["keyword_score"] is not None:
            print(f"  Keyword Score: {r['keyword_score']:.4f}")

        text = r["text"]
        if text:
            if show_full_text:
                print(f"  Text:\n    {text}")
            else:
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"  Text Preview:\n    {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP Retrieval Demo")
    parser.add_argument("--version-id", type=str, help="Document version UUID")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "C"],
        help="Use predefined scenario (A, B, or C)",
    )
    parser.add_argument(
        "--query", type=str, default="售后响应时间", help="Search query"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "vector", "keyword"],
        help="Search mode",
    )
    parser.add_argument(
        "--max-chars", type=int, default=300, help="Max characters to show"
    )
    parser.add_argument(
        "--full-text", action="store_true", help="Show full text (not truncated)"
    )

    args = parser.parse_args()

    # Determine version_id
    version_id = args.version_id
    if args.scenario:
        version_map = {
            "A": "33333333-3333-3333-3333-333333333333",
            "B": "44444444-4444-4444-4444-444444444444",
            "C": "55555555-5555-5555-5555-555555555555",
        }
        version_id = version_map[args.scenario]
        print(f"Using Scenario {args.scenario}: {version_id}")

    if not version_id:
        print("Error: Either --version-id or --scenario must be provided")
        return 1

    print(f"\n🔍 Retrieving: '{args.query}'")
    print(f"   Mode: {args.mode} | Top-K: {args.top_k}")

    try:
        result = retrieve_impl(
            version_id=version_id,
            query=args.query,
            top_k=args.top_k,
            mode=args.mode,
            include_text=True,
            max_chars=None if args.full_text else args.max_chars,
        )

        print_result(result, show_full_text=args.full_text)

        # Save to file option
        output_file = Path("mcp_demo_result.json")
        output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n💾 Full result saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
