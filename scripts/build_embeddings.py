#!/usr/bin/env python3
"""批量生成文本向量 - 最佳实践实现

Features:
- 智能分批: 按 token 数量和批次大小双重限制
- 批量处理: 50-100条/批（推荐）
- 进度显示: 实时显示处理进度
- 错误处理: 批次失败自动回滚，支持中断恢复
- 空文本过滤: 自动跳过无文本的 chunks
- Token 估算: 避免超出 OpenAI 限制
- 统计报告: 处理完成后显示详细统计

最佳实践参考:
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Chunking Strategies: https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025
- AWS pgvector: https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing/
"""

import sys
import time
from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_texts, estimate_tokens, get_embedding_client, get_embedding_config


# 配置参数（可根据环境变量覆盖）
DEFAULT_BATCH_SIZE = 100      # 每批处理数量（推荐 50-100）
DEFAULT_LIMIT = 1000          # 每次运行最大处理数量
DEFAULT_MAX_TOKENS = 100000   # 每批最大 token 数（保守设置）


def get_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    """获取向量化统计信息"""
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


def fetch_batch(
    conn, 
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    version_id: str | None = None,
) -> list[tuple[str, str]]:
    """获取一批需要处理的 chunks
    
    策略:
    1. 只选择有文本且没有向量的记录
    2. 按 token 数量分批，避免超出限制
    3. 支持按 version_id 过滤
    
    Returns:
        [(chunk_id, text_raw), ...]
    """
    with conn.cursor() as cur:
        # 获取候选数据（多取一些以便按 token 筛选）
        query = """
            SELECT chunk_id, text_raw, LENGTH(text_raw) as text_len
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
        params.append(batch_size * 3)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        if not rows:
            return []
        
        # 按 token 数量分批
        result = []
        total_tokens = 0
        
        for chunk_id, text_raw, text_len in rows:
            tokens = estimate_tokens(text_raw)
            
            # 检查是否超出限制
            if total_tokens + tokens > max_tokens or len(result) >= batch_size:
                break
            
            result.append((str(chunk_id), text_raw))
            total_tokens += tokens
        
        return result


def process_batch(
    conn, 
    rows: list[tuple[str, str]], 
    client = None,
    model: str | None = None,
    show_detail: bool = False,
) -> tuple[int, int]:
    """处理一批数据
    
    Args:
        conn: 数据库连接
        rows: [(chunk_id, text_raw), ...]
        client: OpenAI 客户端
        model: 模型名称
        show_detail: 是否显示详细信息
    
    Returns:
        (成功数量, 失败数量)
    """
    if not rows:
        return 0, 0
    
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    
    try:
        # 调用 embedding API（使用 embed_texts 的批量逻辑）
        vecs = embed_texts(
            texts,
            client=client,
            model=model,
            batch_size=50,  # 内部再分批
            show_progress=show_detail,
        )
        
        # 批量更新数据库
        with conn.cursor() as cur:
            # 使用 executemany 提高效率
            update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
            cur.executemany(
                "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
                update_data
            )
        
        conn.commit()
        return len(rows), 0
        
    except Exception as e:
        conn.rollback()
        print(f"\n  ❌ 批次处理失败: {e}")
        return 0, len(rows)


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def main():
    """主函数"""
    start_time = time.time()
    
    # 加载配置
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="批量生成文本向量")
    parser.add_argument("--version-id", help="指定版本 ID")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"批次大小（默认 {DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"最大处理数量（默认 {DEFAULT_LIMIT}）")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"每批最大 token 数（默认 {DEFAULT_MAX_TOKENS}）")
    parser.add_argument("--show-detail", action="store_true", help="显示详细进度")
    args = parser.parse_args()
    
    # 检查 API Key
    if not settings.get("OPENAI_API_KEY"):
        print("❌ 错误: OPENAI_API_KEY 未设置")
        print("请设置环境变量: export OPENAI_API_KEY=sk-xxx")
        sys.exit(1)
    
    # 获取 embedding 配置
    config = get_embedding_config()
    
    print("=" * 80)
    print("🚀 开始生成向量")
    print("=" * 80)
    print(f"模型: {config['model']}")
    print(f"维度: {config['dim']}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大 Token: {args.max_tokens:,}")
    if args.version_id:
        print(f"版本过滤: {args.version_id}")
    print()
    
    # 初始化 OpenAI 客户端
    client = get_embedding_client()
    
    # 连接数据库
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        
        # 获取初始统计
        stats = get_stats(conn, args.version_id)
        
        print("=" * 80)
        print("📊 初始统计")
        print("=" * 80)
        print(f"  总 chunks:    {stats['total']:,}")
        print(f"  已有向量:     {stats['has_count']:,}")
        print(f"  待处理:       {stats['to_process']:,}")
        print(f"  无法处理:     {stats['empty_text']:,} (空文本)")
        print()
        
        if stats['to_process'] == 0:
            print("✅ 所有 chunks 都已有向量，无需处理")
            return
        
        # 确认处理
        to_process = min(stats['to_process'], args.limit)
        print(f"将处理 {to_process} 条记录（限制: {args.limit}）")
        print()
        
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
            
            # 获取一批数据
            rows = fetch_batch(
                conn, 
                batch_size=batch_size,
                max_tokens=args.max_tokens,
                version_id=args.version_id,
            )
            
            if not rows:
                print("没有更多数据需要处理")
                break
            
            batch_num += 1
            batch_len = len(rows)
            
            # 估算 token 数
            batch_tokens = sum(estimate_tokens(r[1]) for r in rows)
            
            print(f"批次 {batch_num:>3}: {batch_len:>3} 条 ({batch_tokens:,} tokens)...", end=" ", flush=True)
            
            # 处理批次
            success, fail = process_batch(
                conn, rows, 
                client=client, 
                model=config['model'],
                show_detail=args.show_detail,
            )
            
            total_success += success
            total_fail += fail
            processed += batch_len
            
            if success == batch_len:
                elapsed = time.time() - start_time
                speed = total_success / elapsed if elapsed > 0 else 0
                print(f"✅ ({speed:.1f} 条/秒)")
            else:
                print(f"⚠️ 成功 {success}/{batch_len}")
            
            # 每 10 批次显示进度
            if batch_num % 10 == 0:
                progress = 100 * processed / to_process
                elapsed = time.time() - start_time
                eta = (elapsed / processed) * (to_process - processed) if processed > 0 else 0
                print(f"  📈 进度: {processed}/{to_process} ({progress:.1f}%) | 已用: {format_duration(elapsed)} | 预计剩余: {format_duration(eta)}")
        
        # 最终统计
        elapsed = time.time() - start_time
        
        print()
        print("=" * 80)
        print("✅ 处理完成")
        print("=" * 80)
        print(f"成功:     {total_success:,}")
        print(f"失败:     {total_fail:,}")
        print(f"总计:     {total_success + total_fail:,}")
        print(f"用时:     {format_duration(elapsed)}")
        if total_success > 0:
            print(f"平均速度: {total_success/elapsed:.1f} 条/秒")
        
        # 获取最终统计
        final_stats = get_stats(conn, args.version_id)
        print()
        print("📊 最终状态")
        print(f"  已有向量: {final_stats['has_count']:,} / {final_stats['total']:,} ({100*final_stats['has_count']/final_stats['total']:.1f}%)")
        print(f"  待处理:   {final_stats['to_process']:,}")
        print(f"  无法处理: {final_stats['empty_text']:,}")


if __name__ == "__main__":
    main()
