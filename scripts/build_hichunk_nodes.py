#!/usr/bin/env python3
"""批量生成层次化文档节点 (HiChunk Nodes)

Features:
- 从 document_versions 读取 content_list
- 使用 HiChunkBuilder 构建4层文档树结构
- 将节点插入 hierarchical_nodes 表
- 自动关联 chunks 表（叶子节点）
- 支持 embeddings 生成（非叶子节点，可选）
- 进度显示和恢复机制
- 错误处理和事务回滚

Tree Structure:
- Level 0 (sentence): 叶子节点，对应 content_list 元素
- Level 1 (paragraph): 段落节点，合并相邻句子
- Level 2 (section): 章节节点，按标题分组
- Level 3 (document): 文档根节点
"""

import sys
import time
from functools import lru_cache
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

from bid_scoring.config import load_settings
from bid_scoring.hichunk import HiChunkBuilder
from bid_scoring.embeddings import get_embedding_client, get_embedding_config


# 配置参数
DEFAULT_BATCH_SIZE = 10  # 每批处理的文档数量
DEFAULT_LIMIT = 100  # 每次运行最大处理文档数
DEFAULT_MAX_NODES_PER_DOC = 10000  # 单个文档最大节点数


def reset_hierarchical_nodes(
    conn, version_id: str | None = None, force: bool = False
) -> bool:
    """重置/清空 hierarchical_nodes 表

    Args:
        conn: 数据库连接
        version_id: 版本 ID，如果指定则只删除该版本的记录，否则删除所有
        force: 是否跳过确认提示

    Returns:
        True 表示已重置，False 表示用户取消
    """
    with conn.cursor() as cur:
        # 先查询将要删除的记录数
        if version_id:
            cur.execute(
                "SELECT COUNT(*) FROM hierarchical_nodes WHERE version_id = %s",
                (version_id,),
            )
        else:
            cur.execute("SELECT COUNT(*) FROM hierarchical_nodes")

        count = cur.fetchone()[0]

        if count == 0:
            print("ℹ️  hierarchical_nodes 表为空，无需重置")
            return True

        # 确认提示
        if not force:
            scope = f"版本 '{version_id}'" if version_id else "所有版本"
            print(
                f"\n⚠️  警告: 这将删除 {scope} 的 {count} 条 hierarchical_nodes 记录！"
            )
            response = input("确认重置? 输入 'yes' 继续: ")
            if response.lower() != "yes":
                print("❌ 操作已取消")
                return False

        # 执行删除
        if version_id:
            cur.execute(
                "DELETE FROM hierarchical_nodes WHERE version_id = %s", (version_id,)
            )
        else:
            cur.execute("DELETE FROM hierarchical_nodes")

        conn.commit()

        scope = f"版本 '{version_id}'" if version_id else "所有版本"
        print(f"✅ 已重置 {scope} 的 {count} 条记录")
        return True


def get_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    """获取处理统计信息

    Returns:
        {
            'total_versions': 总版本数,
            'processed_versions': 已处理版本数,
            'to_process': 待处理版本数,
            'total_nodes': 总节点数,
        }
    """
    with conn.cursor() as cur:
        # 基础查询：统计有 chunks 的版本
        base_query = """
            SELECT 
                COUNT(DISTINCT c.version_id) as total_versions,
                COUNT(DISTINCT hn.version_id) as processed_versions,
                COUNT(DISTINCT c.version_id) FILTER (WHERE hn.version_id IS NULL) as to_process
            FROM chunks c
            LEFT JOIN hierarchical_nodes hn ON c.version_id = hn.version_id
        """

        params = []
        if version_id:
            base_query += " WHERE c.version_id = %s"
            params.append(version_id)

        cur.execute(base_query, params)
        row = cur.fetchone()

        # 统计总节点数
        nodes_query = "SELECT COUNT(*) FROM hierarchical_nodes"
        nodes_params = []
        if version_id:
            nodes_query += " WHERE version_id = %s"
            nodes_params.append(version_id)

        cur.execute(nodes_query, nodes_params)
        total_nodes = cur.fetchone()[0]

        return {
            "total_versions": row[0] or 0,
            "processed": row[1] or 0,
            "to_process": row[2] or 0,
            "total_nodes": total_nodes,
        }


def fetch_pending_versions(
    conn,
    batch_size: int = DEFAULT_BATCH_SIZE,
    version_id: str | None = None,
) -> list[dict[str, Any]]:
    """获取待处理的文档版本

    策略:
    1. 选择有 chunks 且未处理层次化节点的版本
    2. 从 chunks 表重建 content_list

    Returns:
        [{
            'version_id': 版本ID,
            'doc_id': 文档ID,
            'document_title': 文档标题,
            'content_list': content_list 列表,
        }, ...]
    """
    with conn.cursor() as cur:
        # 查找有待处理 chunks 的版本
        query = """
            SELECT DISTINCT
                dv.version_id,
                dv.doc_id,
                d.title as document_title
            FROM document_versions dv
            JOIN documents d ON dv.doc_id = d.doc_id
            JOIN chunks c ON dv.version_id = c.version_id
            LEFT JOIN hierarchical_nodes hn ON dv.version_id = hn.version_id
            WHERE hn.version_id IS NULL
        """

        params = []
        if version_id:
            query += " AND dv.version_id = %s"
            params.append(version_id)

        query += " ORDER BY dv.version_id LIMIT %s"
        params.append(batch_size)

        cur.execute(query, params)
        rows = cur.fetchall()

        result = []
        for row in rows:
            version_id_str = str(row[0])

            # 从 chunks 表获取该版本的所有 chunks
            cur.execute(
                """
                SELECT 
                    chunk_id,
                    chunk_index,
                    page_idx,
                    bbox,
                    element_type,
                    text_raw,
                    text_level,
                    img_path,
                    image_caption,
                    image_footnote,
                    table_body,
                    table_caption,
                    table_footnote,
                    list_items,
                    sub_type
                FROM chunks 
                WHERE version_id = %s 
                ORDER BY chunk_index
                """,
                (version_id_str,),
            )

            chunks_rows = cur.fetchall()
            content_list = []
            for cr in chunks_rows:
                item = {
                    "chunk_id": str(cr[0]),
                    "chunk_index": cr[1],
                    "page_idx": cr[2] or 0,
                    "bbox": cr[3],
                    "type": cr[4] or "text",
                    "text": cr[5] or "",
                    "text_level": cr[6] or 0,
                    "img_path": cr[7],
                    "image_caption": cr[8],
                    "image_footnote": cr[9],
                    "table_body": cr[10],
                    "table_caption": cr[11],
                    "table_footnote": cr[12],
                    "list_items": cr[13],
                    "sub_type": cr[14],
                }
                content_list.append(item)

            result.append(
                {
                    "version_id": version_id_str,
                    "doc_id": str(row[1]),
                    "document_title": row[2] or "untitled",
                    "content_list": content_list,
                }
            )

        return result


def get_chunk_mapping(conn, version_id: str) -> dict[int, str]:
    """获取版本下的 chunk 映射

    通过 chunk_index 映射到 chunk_id，用于关联叶子节点

    Returns:
        {chunk_index: chunk_id, ...}
    """
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


def insert_hierarchical_nodes(
    conn,
    version_id: str,
    nodes: list,
    chunk_mapping: dict[int, str],
    show_detail: bool = False,
) -> tuple[int, int]:
    """插入层次化节点到数据库

    按层级从高到低插入（document -> section -> paragraph -> sentence），
    确保父节点先于子节点插入，避免外键约束错误。

    Args:
        conn: 数据库连接
        version_id: 版本 ID
        nodes: HiChunkNode 列表
        chunk_mapping: chunk_index 到 chunk_id 的映射
        show_detail: 是否显示详细信息

    Returns:
        (成功数量, 失败数量)
    """
    if not nodes:
        return 0, 0

    success_count = 0
    fail_count = 0

    # 按层级分组节点（从高到低：3=document, 2=section, 1=paragraph, 0=sentence）
    nodes_by_level = {3: [], 2: [], 1: [], 0: []}
    leaf_nodes = []

    for node in nodes:
        nodes_by_level[node.level].append(node)
        if node.level == 0:
            leaf_nodes.append(node)

    nodes_by_id = {n.node_id: n for n in nodes if n.node_id}
    leaf_by_id = {n.node_id: n for n in leaf_nodes if n.node_id}

    @lru_cache(maxsize=None)
    def _covered_unit_range(node_id: str) -> tuple[int, int] | None:
        n = nodes_by_id.get(node_id)
        if not n:
            return None
        if n.level == 0:
            source_idx = n.metadata.get("source_index")
            if source_idx is None:
                return None
            i = int(source_idx)
            return (i, i)

        ranges: list[tuple[int, int]] = []
        for child_id in n.children_ids or []:
            r = _covered_unit_range(child_id)
            if r is not None:
                ranges.append(r)
        if not ranges:
            return None
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    # 准备所有层级的插入数据
    def prepare_insert_data(node_list):
        data = []
        for node in node_list:
            cov = _covered_unit_range(node.node_id)
            if cov is not None:
                node.metadata["covered_unit_range"] = {"start": cov[0], "end": cov[1]}

            # 对于叶子节点，尝试关联 chunks
            start_chunk_id = None
            end_chunk_id = None

            if node.level == 0:
                source_idx = node.metadata.get("source_index")
                if source_idx is not None and source_idx in chunk_mapping:
                    start_chunk_id = chunk_mapping[source_idx]
                    end_chunk_id = chunk_mapping[source_idx]
            elif node.level == 1:  # paragraph
                # 对于段落，关联其包含的叶子节点的 chunks
                child_source_indices = []
                for child_id in node.children_ids:
                    child_node = leaf_by_id.get(child_id)
                    if child_node:
                        source_idx = child_node.metadata.get("source_index")
                        if source_idx is not None:
                            child_source_indices.append(source_idx)

                if child_source_indices:
                    min_idx = min(child_source_indices)
                    max_idx = max(child_source_indices)
                    if min_idx in chunk_mapping:
                        start_chunk_id = chunk_mapping[min_idx]
                    if max_idx in chunk_mapping:
                        end_chunk_id = chunk_mapping[max_idx]

            data.append(
                (
                    node.node_id,
                    version_id,
                    node.parent_id,
                    node.level,
                    node.node_type,
                    node.content,
                    node.children_ids,
                    start_chunk_id,
                    end_chunk_id,
                    Jsonb(node.metadata),
                )
            )
        return data

    # 按层级顺序插入：3 -> 2 -> 1 -> 0
    try:
        with conn.cursor() as cur:
            for level in [3, 2, 1, 0]:
                level_nodes = nodes_by_level[level]
                if not level_nodes:
                    continue

                insert_data = prepare_insert_data(level_nodes)
                cur.executemany(
                    """
                    INSERT INTO hierarchical_nodes (
                        node_id, version_id, parent_id, level, node_type,
                        content, children_ids, start_chunk_id, end_chunk_id, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO NOTHING
                    """,
                    insert_data,
                )
                success_count += len(level_nodes)

        conn.commit()

        if show_detail:
            print(f"  ✅ 已插入 {success_count} 个节点")
            # 显示层级统计
            for level in range(4):
                count = len(nodes_by_level[level])
                level_name = ["sentence", "paragraph", "section", "document"][level]
                print(f"    - Level {level} ({level_name}): {count}")

    except Exception as e:
        conn.rollback()
        fail_count = len(nodes) - success_count
        print(f"  ❌ 插入失败: {e}")

    return success_count, fail_count


def process_version(
    conn,
    version_data: dict[str, Any],
    builder: HiChunkBuilder,
    show_detail: bool = False,
) -> tuple[int, int]:
    """处理单个文档版本

    Args:
        conn: 数据库连接
        version_data: 版本数据
        builder: HiChunkBuilder 实例
        show_detail: 是否显示详细信息

    Returns:
        (成功数量, 失败数量)
    """
    version_id = version_data["version_id"]
    document_title = version_data["document_title"]
    content_list = version_data["content_list"]

    if show_detail:
        print(f"\n📄 处理版本: {version_id[:8]}...")
        print(f"   文档标题: {document_title}")
        print(f"   content_list 长度: {len(content_list)}")

    try:
        # 步骤 1: 构建层次结构
        if show_detail:
            print("  🏗️  构建层次结构...", end=" ", flush=True)

        nodes = builder.build_hierarchy(content_list, document_title)

        if show_detail:
            print(f"✅ ({len(nodes)} 个节点)")

        # 检查节点数量限制
        if len(nodes) > DEFAULT_MAX_NODES_PER_DOC:
            print(
                f"  ⚠️  节点数量 ({len(nodes)}) 超过限制 ({DEFAULT_MAX_NODES_PER_DOC})，跳过"
            )
            return 0, len(nodes)

        # 步骤 2: 获取 chunk 映射
        if show_detail:
            print("  🔗 获取 chunk 映射...", end=" ", flush=True)

        chunk_mapping = get_chunk_mapping(conn, version_id)

        if show_detail:
            print(f"✅ ({len(chunk_mapping)} 个 chunks)")

        # 步骤 3: 插入节点
        if show_detail:
            print("  💾 插入节点...", end=" ", flush=True)

        success, fail = insert_hierarchical_nodes(
            conn, version_id, nodes, chunk_mapping, show_detail
        )

        return success, fail

    except Exception as e:
        conn.rollback()
        print(f"  ❌ 处理失败: {e}")
        return 0, 0


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}分钟"
    else:
        return f"{seconds / 3600:.1f}小时"


def main():
    """主函数"""
    start_time = time.time()

    # 加载配置
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    # 获取命令行参数
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
        "--dry-run", action="store_true", help="干运行模式（不实际写入数据库）"
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
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="为非叶子节点生成 embeddings（可选，较慢）",
    )
    parser.add_argument("--embedding-model", default=None, help="embedding 模型名称")
    args = parser.parse_args()

    # 连接数据库
    with psycopg.connect(dsn) as conn:
        register_vector(conn)

        # 处理重置请求
        if args.reset:
            if not reset_hierarchical_nodes(conn, args.version_id, args.force):
                sys.exit(0)  # 用户取消，正常退出
            print()

        # 获取初始统计
        stats = get_stats(conn, args.version_id)

        print("=" * 80)
        print("🚀 开始生成层次化文档节点 (HiChunk)")
        print("=" * 80)
        print(f"批次大小: {args.batch_size}")
        print(f"最大节点数/文档: {args.max_nodes:,}")
        if args.with_embeddings:
            config = get_embedding_config()
            print(f"Embedding 模型: {args.embedding_model or config['model']}")
            print(f"Embedding 维度: {config['dim']}")
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

        # 确认处理
        to_process = min(stats["to_process"], args.limit)
        print(f"将处理 {to_process} 个版本（限制: {args.limit}）")
        print()

        # 初始化 builder
        builder = HiChunkBuilder()

        # 初始化 embedding 客户端（如果需要）
        if args.with_embeddings:
            _embedding_client = get_embedding_client()

        # 主循环
        total_success = 0
        total_fail = 0
        batch_num = 0
        processed = 0

        print("=" * 80)
        print("🔄 开始处理")
        print("=" * 80)

        while processed < to_process:
            # 计算剩余需要处理的数量
            remaining = to_process - processed
            batch_size = min(args.batch_size, remaining)

            # 获取一批待处理版本
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
                # 干运行模式：只打印，不实际处理
                for v in versions:
                    print(
                        f"  ⏭️  {v['version_id'][:8]}... ({len(v['content_list'])} items)"
                    )
                processed += batch_len
                continue

            # 处理每个版本
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

            # 每 10 批次显示进度
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

        # 最终统计
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

        # 获取最终统计
        final_stats = get_stats(conn, args.version_id)
        print()
        print("📊 最终状态")
        print(
            f"  已处理版本: {final_stats['processed']:,} / {final_stats['total_versions']:,} ({100 * final_stats['processed'] / final_stats['total_versions']:.1f}%)"
        )
        print(f"  总节点数:   {final_stats['total_nodes']:,}")
        print(f"  待处理:     {final_stats['to_process']:,}")


if __name__ == "__main__":
    main()
