#!/usr/bin/env python3
"""方案 C: 全量向量化 - 多粒度 Embedding 生成

Features:
1. Level 1: chunks 表基础向量化 (1060条)
2. Level 2: contextual_chunks 表上下文增强向量化 (可选)
3. Level 3: hierarchical_nodes 表层次节点向量化 (Level 1-2, 非叶子节点)

最佳实践:
- chunks: 直接嵌入原始文本，用于精确检索
- contextual_chunks: 添加章节前缀，用于语义检索
- hierarchical_nodes: 段落/章节级嵌入，用于粗粒度检索

Usage:
    python scripts/build_all_embeddings.py --version-id="xxx"
    python scripts/build_all_embeddings.py --version-id="xxx" --skip-contextual
    python scripts/build_all_embeddings.py --version-id="xxx" --skip-hierarchical
"""

import sys
import time
from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_texts, estimate_tokens, get_embedding_client, get_embedding_config


# 配置参数
DEFAULT_BATCH_SIZE = 100
DEFAULT_LIMIT = 10000
DEFAULT_MAX_TOKENS = 100000


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


# ============================================================================
# Level 1: chunks 表基础向量化
# ============================================================================

def get_chunks_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    """获取 chunks 表向量化统计"""
    with conn.cursor() as cur:
        base_query = """
            SELECT 
                COUNT(*) FILTER (WHERE embedding IS NULL) as null_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as has_count,
                COUNT(*) FILTER (WHERE embedding IS NULL AND text_raw IS NOT NULL AND text_raw != '') as to_process,
                COUNT(*) FILTER (WHERE embedding IS NULL AND (text_raw IS NULL OR text_raw = '')) as empty_text
            FROM chunks
        """
        
        if version_id:
            base_query += " WHERE version_id = %s"
            cur.execute(base_query, (version_id,))
        else:
            cur.execute(base_query)
        
        row = cur.fetchone()
        return {
            "null_count": row[0],
            "has_count": row[1],
            "to_process": row[2],
            "empty_text": row[3],
            "total": row[0] + row[1],
        }


def process_chunks_embeddings(
    conn,
    version_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int = DEFAULT_LIMIT,
    client = None,
    model: str | None = None,
    show_detail: bool = False,
) -> tuple[int, int]:
    """处理 chunks 表向量化
    
    Returns:
        (成功数量, 失败数量)
    """
    stats = get_chunks_stats(conn, version_id)
    to_process = min(stats["to_process"], limit)
    
    if to_process == 0:
        print("  ✅ chunks 表无需处理")
        return 0, 0
    
    print(f"  待处理: {to_process} 条")
    
    processed = 0
    success_count = 0
    fail_count = 0
    batch_num = 0
    
    while processed < to_process:
        with conn.cursor() as cur:
            # 获取一批数据
            query = """
                SELECT chunk_id, text_raw
                FROM chunks 
                WHERE embedding IS NULL 
                  AND text_raw IS NOT NULL 
                  AND text_raw != ''
            """
            params = []
            if version_id:
                query += " AND version_id = %s"
                params.append(version_id)
            query += " ORDER BY chunk_id LIMIT %s"
            params.append(batch_size)
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            if not rows:
                break
            
            batch_num += 1
            ids = [str(r[0]) for r in rows]
            texts = [r[1] for r in rows]
            
            try:
                # 生成向量
                vecs = embed_texts(texts, client=client, model=model, show_progress=False)
                
                # 批量更新
                update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
                cur.executemany(
                    "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
                    update_data
                )
                conn.commit()
                
                success_count += len(rows)
                if show_detail:
                    print(f"    批次 {batch_num}: {len(rows)} 条 ✅")
                
            except Exception as e:
                conn.rollback()
                fail_count += len(rows)
                print(f"    批次 {batch_num}: ❌ {e}")
            
            processed += len(rows)
    
    print(f"  完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


# ============================================================================
# Level 2: contextual_chunks 表上下文增强向量化
# ============================================================================

def build_contextual_chunks(conn, version_id: str | None = None) -> int:
    """从 chunks 和 hierarchical_nodes 构建 contextual_chunks
    
    策略: 为每个 chunk 添加上下文前缀（章节标题）
    
    Returns:
        创建的 contextual_chunks 数量
    """
    with conn.cursor() as cur:
        # 先清空该版本的现有数据
        if version_id:
            cur.execute(
                "DELETE FROM contextual_chunks WHERE version_id = %s",
                (version_id,)
            )
        else:
            cur.execute("DELETE FROM contextual_chunks")
        
        # 获取所有需要处理的 chunks
        query = """
            SELECT 
                c.chunk_id,
                c.version_id,
                c.text_raw,
                c.page_idx,
                c.element_type
            FROM chunks c
            WHERE c.text_raw IS NOT NULL 
              AND c.text_raw != ''
        """
        params = []
        if version_id:
            query += " AND c.version_id = %s"
            params.append(version_id)
        
        cur.execute(query, params)
        chunks = cur.fetchall()
        
        if not chunks:
            return 0
        
        # 为每个 chunk 查找所属的 section 标题
        created_count = 0
        for chunk_id, ver_id, text_raw, page_idx, elem_type in chunks:
            # 查找包含此 chunk 的 section
            cur.execute(
                """
                SELECT content, metadata
                FROM hierarchical_nodes
                WHERE version_id = %s
                  AND level = 2
                  AND (metadata->>'page_idx')::int <= %s
                ORDER BY (metadata->>'page_idx')::int DESC
                LIMIT 1
                """,
                (ver_id, page_idx or 0)
            )
            section_row = cur.fetchone()
            
            section_title = ""
            if section_row:
                section_title = section_row[0] or ""
            
            # 构建上下文前缀
            context_prefix = ""
            if section_title:
                context_prefix = f"[{section_title}] "
            
            contextualized_text = context_prefix + text_raw
            
            # 插入 contextual_chunks（embedding 稍后更新）
            cur.execute(
                """
                INSERT INTO contextual_chunks (
                    chunk_id, version_id, original_text, context_prefix,
                    contextualized_text, model_name, embedding_model
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    context_prefix = EXCLUDED.context_prefix,
                    contextualized_text = EXCLUDED.contextualized_text
                """,
                (
                    chunk_id, ver_id, text_raw, context_prefix,
                    contextualized_text, "text-embedding-3-small", "text-embedding-3-small"
                )
            )
            created_count += 1
        
        conn.commit()
        return created_count


def process_contextual_embeddings(
    conn,
    version_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int = DEFAULT_LIMIT,
    client = None,
    model: str | None = None,
    show_detail: bool = False,
) -> tuple[int, int]:
    """处理 contextual_chunks 表向量化"""
    with conn.cursor() as cur:
        # 获取统计
        query = """
            SELECT COUNT(*) 
            FROM contextual_chunks 
            WHERE embedding IS NULL
        """
        params = []
        if version_id:
            query += " AND version_id = %s"
            params.append(version_id)
        
        cur.execute(query, params)
        to_process = cur.fetchone()[0]
        to_process = min(to_process, limit)
        
        if to_process == 0:
            print("  ✅ contextual_chunks 表无需处理")
            return 0, 0
        
        print(f"  待处理: {to_process} 条")
    
    processed = 0
    success_count = 0
    fail_count = 0
    batch_num = 0
    
    while processed < to_process:
        with conn.cursor() as cur:
            query = """
                SELECT contextual_id, contextualized_text
                FROM contextual_chunks 
                WHERE embedding IS NULL
            """
            params = []
            if version_id:
                query += " AND version_id = %s"
                params.append(version_id)
            query += " ORDER BY contextual_id LIMIT %s"
            params.append(batch_size)
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            if not rows:
                break
            
            batch_num += 1
            ids = [str(r[0]) for r in rows]
            texts = [r[1] for r in rows]
            
            try:
                vecs = embed_texts(texts, client=client, model=model, show_progress=False)
                
                update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
                cur.executemany(
                    "UPDATE contextual_chunks SET embedding = %s WHERE contextual_id = %s",
                    update_data
                )
                conn.commit()
                
                success_count += len(rows)
                if show_detail:
                    print(f"    批次 {batch_num}: {len(rows)} 条 ✅")
                
            except Exception as e:
                conn.rollback()
                fail_count += len(rows)
                print(f"    批次 {batch_num}: ❌ {e}")
            
            processed += len(rows)
    
    print(f"  完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


# ============================================================================
# Level 3: hierarchical_nodes 表层次节点向量化
# ============================================================================

def get_hierarchical_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    """获取 hierarchical_nodes 表向量化统计"""
    with conn.cursor() as cur:
        query = """
            SELECT 
                level,
                COUNT(*) FILTER (WHERE embedding IS NULL) as null_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as has_count
            FROM hierarchical_nodes
        """
        params = []
        if version_id:
            query += " WHERE version_id = %s"
            params.append(version_id)
        
        query += " GROUP BY level ORDER BY level"
        cur.execute(query, params)
        
        stats = {}
        for row in cur.fetchall():
            stats[row[0]] = {
                "null_count": row[1],
                "has_count": row[2],
                "total": row[1] + row[2],
            }
        return stats


def process_hierarchical_embeddings(
    conn,
    version_id: str | None = None,
    levels: list[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int = DEFAULT_LIMIT,
    client = None,
    model: str | None = None,
    show_detail: bool = False,
) -> dict[int, tuple[int, int]]:
    """处理 hierarchical_nodes 表向量化
    
    Args:
        levels: 要处理的层级列表（默认 [1, 2]，即 paragraph 和 section）
    
    Returns:
        {level: (成功数量, 失败数量)}
    """
    if levels is None:
        levels = [1, 2]  # 默认只处理 paragraph 和 section
    
    level_names = {0: "sentence", 1: "paragraph", 2: "section", 3: "document"}
    results = {}
    
    for level in levels:
        level_name = level_names.get(level, f"level_{level}")
        print(f"\n  处理 Level {level} ({level_name}):")
        
        processed = 0
        success_count = 0
        fail_count = 0
        batch_num = 0
        
        while processed < limit:
            with conn.cursor() as cur:
                query = """
                    SELECT node_id, content
                    FROM hierarchical_nodes 
                    WHERE embedding IS NULL
                      AND level = %s
                """
                params = [level]
                if version_id:
                    query += " AND version_id = %s"
                    params.append(version_id)
                query += " ORDER BY node_id LIMIT %s"
                params.append(batch_size)
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                if not rows:
                    break
                
                batch_num += 1
                ids = [str(r[0]) for r in rows]
                texts = [r[1] for r in rows]
                
                try:
                    vecs = embed_texts(texts, client=client, model=model, show_progress=False)
                    
                    update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
                    cur.executemany(
                        "UPDATE hierarchical_nodes SET embedding = %s WHERE node_id = %s",
                        update_data
                    )
                    conn.commit()
                    
                    success_count += len(rows)
                    if show_detail:
                        print(f"    批次 {batch_num}: {len(rows)} 条 ✅")
                    
                except Exception as e:
                    conn.rollback()
                    fail_count += len(rows)
                    print(f"    批次 {batch_num}: ❌ {e}")
                
                processed += len(rows)
        
        print(f"  完成: 成功 {success_count}, 失败 {fail_count}")
        results[level] = (success_count, fail_count)
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    start_time = time.time()
    
    # 加载配置
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="方案 C: 全量向量化")
    parser.add_argument("--version-id", help="指定版本 ID")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--skip-chunks", action="store_true", help="跳过 chunks 表")
    parser.add_argument("--skip-contextual", action="store_true", help="跳过 contextual_chunks 表")
    parser.add_argument("--skip-hierarchical", action="store_true", help="跳过 hierarchical_nodes 表")
    parser.add_argument("--show-detail", action="store_true", help="显示详细进度")
    args = parser.parse_args()
    
    # 检查 API Key
    if not settings.get("OPENAI_API_KEY"):
        print("❌ 错误: OPENAI_API_KEY 未设置")
        sys.exit(1)
    
    # 获取 embedding 配置
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
    
    # 连接数据库
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        
        # ========================================================================
        # Step 1: chunks 表基础向量化
        # ========================================================================
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
                    conn, args.version_id, args.batch_size, args.limit,
                    client, config["model"], args.show_detail
                )
                total_success += success
                total_fail += fail
            else:
                print("  ✅ 无需处理")
        
        # ========================================================================
        # Step 2: contextual_chunks 表上下文增强向量化
        # ========================================================================
        if not args.skip_contextual:
            print("\n" + "=" * 80)
            print("📝 Step 2: contextual_chunks 表上下文增强向量化")
            print("=" * 80)
            
            # 先构建 contextual_chunks 记录
            print("  构建 contextual_chunks 记录...")
            created = build_contextual_chunks(conn, args.version_id)
            print(f"  创建/更新: {created} 条")
            
            # 然后生成向量
            success, fail = process_contextual_embeddings(
                conn, args.version_id, args.batch_size, args.limit,
                client, config["model"], args.show_detail
            )
            total_success += success
            total_fail += fail
        
        # ========================================================================
        # Step 3: hierarchical_nodes 表层次节点向量化
        # ========================================================================
        if not args.skip_hierarchical:
            print("\n" + "=" * 80)
            print("🌲 Step 3: hierarchical_nodes 表层次节点向量化")
            print("=" * 80)
            
            stats = get_hierarchical_stats(conn, args.version_id)
            for level, stat in sorted(stats.items()):
                level_names = {0: "sentence", 1: "paragraph", 2: "section", 3: "document"}
                name = level_names.get(level, f"level_{level}")
                print(f"  Level {level} ({name}): 待处理 {stat['null_count']}/{stat['total']}")
            
            results = process_hierarchical_embeddings(
                conn, args.version_id, [1, 2],  # 只处理 paragraph 和 section
                args.batch_size, args.limit,
                client, config["model"], args.show_detail
            )
            
            for level, (success, fail) in results.items():
                total_success += success
                total_fail += fail
    
    # 最终统计
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ 处理完成")
    print("=" * 80)
    print(f"总成功: {total_success:,}")
    print(f"总失败: {total_fail:,}")
    print(f"用时:   {format_duration(elapsed)}")
    if total_success > 0:
        print(f"平均:   {total_success/elapsed:.1f} 条/秒")


if __name__ == "__main__":
    main()
