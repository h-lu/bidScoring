#!/usr/bin/env python3
"""方案 C: 全量向量化 - 多粒度 Embedding 生成.

This script is intentionally thin; heavy logic lives in `_build_all_embeddings_lib.py`
to keep files under the 500 LOC limit.
"""

from __future__ import annotations

import sys
import time

import psycopg
from pgvector.psycopg import register_vector

from bid_scoring.config import load_settings
from bid_scoring.embeddings import get_embedding_client, get_embedding_config

from _build_all_embeddings_lib import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIMIT,
    build_contextual_chunks,
    format_duration,
    get_chunks_stats,
    get_hierarchical_stats,
    process_chunks_embeddings,
    process_contextual_embeddings,
    process_hierarchical_embeddings,
)


def main() -> None:
    start_time = time.time()

    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    import argparse

    parser = argparse.ArgumentParser(description="方案 C: 全量向量化")
    parser.add_argument("--version-id", help="指定版本 ID")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--skip-chunks", action="store_true", help="跳过 chunks 表")
    parser.add_argument(
        "--skip-contextual", action="store_true", help="跳过 contextual_chunks 表"
    )
    parser.add_argument(
        "--skip-hierarchical", action="store_true", help="跳过 hierarchical_nodes 表"
    )
    parser.add_argument("--show-detail", action="store_true", help="显示详细进度")
    args = parser.parse_args()

    if not settings.get("OPENAI_API_KEY"):
        print("❌ 错误: OPENAI_API_KEY 未设置")
        sys.exit(1)

    config = get_embedding_config()
    client = get_embedding_client()

    print("=" * 80)
    print("🚀 方案 C: 全量向量化")
    print("=" * 80)
    print(f"模型: {config['model']}")
    print(f"维度: {config['dim']}")
    print(f"批次大小: {args.batch_size}")
    if args.version_id:
        print(f"版本过滤: {args.version_id}")
    print()

    total_success = 0
    total_fail = 0

    with psycopg.connect(dsn) as conn:
        register_vector(conn)

        if not args.skip_chunks:
            print("=" * 80)
            print("📦 Step 1: chunks 表基础向量化")
            print("=" * 80)

            stats = get_chunks_stats(conn, args.version_id)
            print(f"  总 chunks: {stats['total']}")
            print(f"  已有向量: {stats['has_count']}")
            print(f"  待处理: {stats['to_process']}")

            if stats["to_process"] > 0:
                success, fail = process_chunks_embeddings(
                    conn,
                    args.version_id,
                    args.batch_size,
                    args.limit,
                    client,
                    config["model"],
                    args.show_detail,
                )
                total_success += success
                total_fail += fail
            else:
                print("  ✅ 无需处理")

        if not args.skip_contextual:
            print("\n" + "=" * 80)
            print("📝 Step 2: contextual_chunks 表上下文增强向量化")
            print("=" * 80)

            print("  构建 contextual_chunks 记录...")
            created = build_contextual_chunks(conn, args.version_id)
            print(f"  创建/更新: {created} 条")

            success, fail = process_contextual_embeddings(
                conn,
                args.version_id,
                args.batch_size,
                args.limit,
                client,
                config["model"],
                args.show_detail,
            )
            total_success += success
            total_fail += fail

        if not args.skip_hierarchical:
            print("\n" + "=" * 80)
            print("🌲 Step 3: hierarchical_nodes 表层次节点向量化")
            print("=" * 80)

            stats = get_hierarchical_stats(conn, args.version_id)
            for level, stat in sorted(stats.items()):
                level_names = {0: "sentence", 1: "paragraph", 2: "section", 3: "document"}
                name = level_names.get(level, f"level_{level}")
                print(
                    f"  Level {level} ({name}): 待处理 {stat['null_count']}/{stat['total']}"
                )

            results = process_hierarchical_embeddings(
                conn,
                args.version_id,
                [1, 2],
                args.batch_size,
                args.limit,
                client,
                config["model"],
                args.show_detail,
            )

            for _level, (success, fail) in results.items():
                total_success += success
                total_fail += fail

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ 处理完成")
    print("=" * 80)
    print(f"总成功: {total_success:,}")
    print(f"总失败: {total_fail:,}")
    print(f"用时:   {format_duration(elapsed)}")
    if total_success > 0:
        print(f"平均:   {total_success / elapsed:.1f} 条/秒")


if __name__ == "__main__":
    main()

