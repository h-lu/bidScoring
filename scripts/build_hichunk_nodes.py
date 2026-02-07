#!/usr/bin/env python3
"""批量生成层次化文档节点 (HiChunk Nodes).

This script is intentionally thin; most heavy logic is in `_hichunk_nodes_lib.py`
to keep files under the 500 LOC limit.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from bid_scoring.config import load_settings
from bid_scoring.hichunk import HiChunkBuilder

from _hichunk_nodes_lib import (
    fetch_pending_versions,
    format_duration,
    get_stats,
    insert_hierarchical_nodes,
    reset_hierarchical_nodes,
)


DEFAULT_BATCH_SIZE = 10  # 每批处理的文档数量
DEFAULT_LIMIT = 100  # 每次运行最大处理文档数
DEFAULT_MAX_NODES_PER_DOC = 10000  # 单个文档最大节点数


def get_chunk_mapping(conn, version_id: str) -> dict[int, str]:
    """chunk_index -> chunk_id mapping for a version."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, chunk_index 
            FROM chunks 
            WHERE version_id = %s 
            ORDER BY chunk_index
            """,
            (version_id,),
        )
        rows = cur.fetchall()
        return {row[1]: str(row[0]) for row in rows}


def process_version(
    conn,
    version_data: dict[str, Any],
    builder: HiChunkBuilder,
    show_detail: bool = False,
) -> tuple[int, int]:
    """处理单个文档版本."""
    version_id = version_data["version_id"]
    document_title = version_data["document_title"]
    content_list = version_data["content_list"]

    if show_detail:
        print(f"\n📄 处理版本: {version_id[:8]}...")
        print(f"   文档标题: {document_title}")
        print(f"   content_list 长度: {len(content_list)}")

    try:
        if show_detail:
            print("  🏗️  构建层次结构...", end=" ", flush=True)

        nodes = builder.build_hierarchy(content_list, document_title)

        if show_detail:
            print(f"✅ ({len(nodes)} 个节点)")

        if len(nodes) > DEFAULT_MAX_NODES_PER_DOC:
            print(
                f"  ⚠️  节点数量 ({len(nodes)}) 超过限制 ({DEFAULT_MAX_NODES_PER_DOC})，跳过"
            )
            return 0, len(nodes)

        if show_detail:
            print("  🔗 获取 chunk 映射...", end=" ", flush=True)

        chunk_mapping = get_chunk_mapping(conn, version_id)

        if show_detail:
            print(f"✅ ({len(chunk_mapping)} 个 chunks)")

        if show_detail:
            print("  💾 插入节点...", end=" ", flush=True)

        return insert_hierarchical_nodes(
            conn, version_id, nodes, chunk_mapping, show_detail
        )

    except Exception as e:
        conn.rollback()
        print(f"  ❌ 处理失败: {e}")
        return 0, 0


def main() -> None:
    start_time = time.time()

    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    import argparse

    parser = argparse.ArgumentParser(description="批量生成层次化文档节点 (HiChunk)")
    parser.add_argument("--version-id", help="指定版本 ID")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"批次大小（默认 {DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"最大处理数量（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES_PER_DOC,
        help=f"单个文档最大节点数（默认 {DEFAULT_MAX_NODES_PER_DOC}）",
    )
    parser.add_argument("--show-detail", action="store_true", help="显示详细进度")
    parser.add_argument(
        "--dry-run", action="store_true", help="干运行模式（不写入数据库）"
    )
    parser.add_argument(
        "--reset",
        "-r",
        action="store_true",
        help="重置/清空 hierarchical_nodes 表后重新生成",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重置，跳过确认提示（配合 --reset 使用）",
    )
    # Kept for CLI backward compatibility; not implemented yet.
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="(暂未实现) 为非叶子节点生成 embeddings",
    )
    parser.add_argument(
        "--embedding-model", default=None, help="(暂未实现) embedding 模型名称"
    )
    args = parser.parse_args()

    if args.with_embeddings:
        print("⚠️  --with-embeddings 当前脚本未实现，将忽略该选项。")

    with psycopg.connect(dsn) as conn:
        register_vector(conn)

        if args.reset:
            if not reset_hierarchical_nodes(conn, args.version_id, args.force):
                sys.exit(0)
            print()

        stats = get_stats(conn, args.version_id)

        print("=" * 80)
        print("🚀 开始生成层次化文档节点 (HiChunk)")
        print("=" * 80)
        print(f"批次大小: {args.batch_size}")
        print(f"最大节点数/文档: {args.max_nodes:,}")
        if args.version_id:
            print(f"版本过滤: {args.version_id}")
        if args.dry_run:
            print("⚠️ 干运行模式: 不会写入数据库")
        print()

        print("=" * 80)
        print("📊 初始统计")
        print("=" * 80)
        print(f"  总版本数:     {stats['total_versions']:,}")
        print(f"  已处理:       {stats['processed']:,}")
        print(f"  待处理:       {stats['to_process']:,}")
        print(f"  现有节点数:   {stats['total_nodes']:,}")
        print()

        if stats["to_process"] == 0:
            print("✅ 所有版本都已处理，无需处理")
            return

        to_process = min(stats["to_process"], args.limit)
        print(f"将处理 {to_process} 个版本（限制: {args.limit}）")
        print()

        builder = HiChunkBuilder()

        total_success = 0
        total_fail = 0
        batch_num = 0
        processed = 0

        print("=" * 80)
        print("🔄 开始处理")
        print("=" * 80)

        while processed < to_process:
            remaining = to_process - processed
            batch_size = min(args.batch_size, remaining)

            versions = fetch_pending_versions(
                conn,
                batch_size=batch_size,
                version_id=args.version_id,
            )

            if not versions:
                print("没有更多数据需要处理")
                break

            batch_num += 1
            batch_len = len(versions)

            print(f"\n批次 {batch_num:>3}: {batch_len:>3} 个版本...")

            if args.dry_run:
                for v in versions:
                    print(
                        f"  ⏭️  {v['version_id'][:8]}... ({len(v['content_list'])} items)"
                    )
                processed += batch_len
                continue

            batch_success = 0
            batch_fail = 0

            for version_data in versions:
                success, fail = process_version(
                    conn,
                    version_data,
                    builder,
                    show_detail=args.show_detail,
                )
                batch_success += success
                batch_fail += fail

            total_success += batch_success
            total_fail += batch_fail
            processed += batch_len

            print(f"  ✅ 完成: {batch_success} 个节点")
            if batch_fail > 0:
                print(f"  ❌ 失败: {batch_fail} 个节点")

            if batch_num % 10 == 0:
                progress = 100 * processed / to_process
                elapsed = time.time() - start_time
                eta = (
                    (elapsed / processed) * (to_process - processed)
                    if processed > 0
                    else 0
                )
                print(
                    f"\n  📈 进度: {processed}/{to_process} ({progress:.1f}%) | 已用: {format_duration(elapsed)} | 预计剩余: {format_duration(eta)}"
                )

        elapsed = time.time() - start_time

        print()
        print("=" * 80)
        print("✅ 处理完成")
        print("=" * 80)
        print(f"成功节点:     {total_success:,}")
        print(f"失败节点:     {total_fail:,}")
        print(f"处理版本数:   {processed:,}")
        print(f"用时:         {format_duration(elapsed)}")
        if processed > 0:
            print(f"平均速度:     {processed / elapsed:.2f} 版本/秒")

        final_stats = get_stats(conn, args.version_id)
        print()
        print("📊 最终状态")
        if final_stats["total_versions"] > 0:
            pct = 100 * final_stats["processed"] / final_stats["total_versions"]
        else:
            pct = 0.0
        print(
            f"  已处理版本: {final_stats['processed']:,} / {final_stats['total_versions']:,} ({pct:.1f}%)"
        )
        print(f"  总节点数:   {final_stats['total_nodes']:,}")
        print(f"  待处理:     {final_stats['to_process']:,}")


if __name__ == "__main__":
    main()
