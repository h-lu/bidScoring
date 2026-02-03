#!/usr/bin/env python3
"""批量生成上下文增强 chunks - Contextual Retrieval 实现

Features:
- 智能分批: 按 chunk 数量分批，平衡 LLM 调用效率
- 批量处理: 5-10条/批（LLM 生成上下文）
- 进度显示: 实时显示处理进度
- 错误处理: 批次失败自动回滚，支持中断恢复
- 空文本过滤: 自动跳过无文本的 chunks
- 恢复机制: 自动跳过已处理的 chunks
- 统计报告: 处理完成后显示详细统计

最佳实践参考:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
"""

import sys
import time
from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_texts, estimate_tokens, get_embedding_client, get_embedding_config
from bid_scoring.contextual_retrieval import ContextualRetrievalGenerator


# 配置参数（可根据环境变量覆盖）
DEFAULT_BATCH_SIZE = 5       # 每批处理数量（LLM 调用，推荐 5-10）
DEFAULT_LIMIT = 500          # 每次运行最大处理数量
DEFAULT_MAX_TOKENS = 50000   # 每批最大 token 数（上下文生成）


def reset_contextual_chunks(conn, version_id: str | None = None, force: bool = False) -> bool:
    """重置/清空 contextual_chunks 表
    
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
                "SELECT COUNT(*) FROM contextual_chunks WHERE version_id = %s",
                (version_id,)
            )
        else:
            cur.execute("SELECT COUNT(*) FROM contextual_chunks")
        
        count = cur.fetchone()[0]
        
        if count == 0:
            print("ℹ️  contextual_chunks 表为空，无需重置")
            return True
        
        # 确认提示
        if not force:
            scope = f"版本 '{version_id}'" if version_id else "所有版本"
            print(f"\n⚠️  警告: 这将删除 {scope} 的 {count} 条 contextual_chunks 记录！")
            response = input("确认重置? 输入 'yes' 继续: ")
            if response.lower() != 'yes':
                print("❌ 操作已取消")
                return False
        
        # 执行删除
        if version_id:
            cur.execute(
                "DELETE FROM contextual_chunks WHERE version_id = %s",
                (version_id,)
            )
        else:
            cur.execute("DELETE FROM contextual_chunks")
        
        conn.commit()
        
        scope = f"版本 '{version_id}'" if version_id else "所有版本"
        print(f"✅ 已重置 {scope} 的 {count} 条记录")
        return True


def get_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    with conn.cursor() as cur:
        # 基础查询：统计 chunks 表
        base_query = """
            SELECT 
                COUNT(*) FILTER (WHERE c.text_raw IS NOT NULL AND c.text_raw != '') as total_chunks,
                COUNT(*) FILTER (WHERE cc.chunk_id IS NOT NULL) as processed_chunks,
                COUNT(*) FILTER (WHERE c.text_raw IS NOT NULL AND c.text_raw != '' AND cc.chunk_id IS NULL) as to_process,
                COUNT(*) FILTER (WHERE c.text_raw IS NULL OR c.text_raw = '') as empty_text
            FROM chunks c
            LEFT JOIN contextual_chunks cc ON c.chunk_id = cc.chunk_id
        """
        
        params = []
        if version_id:
            base_query += " WHERE c.version_id = %s"
            params.append(version_id)
        
        cur.execute(base_query, params)
        row = cur.fetchone()
        
        return {
            "total_chunks": row[0],
            "processed": row[1],
            "to_process": row[2],
            "empty_text": row[3],
        }


def fetch_batch(
    conn, 
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    version_id: str | None = None,
) -> list[dict[str, Any]]:
    """获取一批需要处理的 chunks
    
    策略:
    1. 只选择有文本且没有 contextual_chunks 的记录
    2. 获取文档标题用于上下文生成
    3. 获取前后 chunks 用于周围上下文
    4. 按 token 数量分批，避免超出限制
    
    Returns:
        [{chunk_id, version_id, text_raw, document_title, section_title, surrounding_chunks}, ...]
    """
    with conn.cursor() as cur:
        # 获取候选数据（多取一些以便按 token 筛选）
        query = """
            SELECT 
                c.chunk_id,
                c.version_id,
                c.text_raw,
                c.chunk_index,
                d.title as document_title,
                LENGTH(c.text_raw) as text_len
            FROM chunks c
            JOIN document_versions dv ON c.version_id = dv.version_id
            JOIN documents d ON dv.doc_id = d.doc_id
            LEFT JOIN contextual_chunks cc ON c.chunk_id = cc.chunk_id
            WHERE c.text_raw IS NOT NULL 
              AND c.text_raw != ''
              AND cc.chunk_id IS NULL
        """
        
        params = []
        if version_id:
            query += " AND c.version_id = %s"
            params.append(version_id)
        
        query += " ORDER BY c.version_id, c.chunk_index LIMIT %s"
        params.append(batch_size * 3)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        if not rows:
            return []
        
        # 按 token 数量分批
        result = []
        total_tokens = 0
        
        for row in rows:
            chunk_id, ver_id, text_raw, chunk_index, document_title, text_len = row
            tokens = estimate_tokens(text_raw)
            
            # 检查是否超出限制
            if total_tokens + tokens > max_tokens or len(result) >= batch_size:
                break
            
            # 获取前后 chunks 作为周围上下文
            surrounding_chunks = _get_surrounding_chunks(cur, ver_id, chunk_index)
            
            result.append({
                "chunk_id": str(chunk_id),
                "version_id": str(ver_id),
                "text_raw": text_raw,
                "document_title": document_title or "未命名文档",
                "section_title": None,  # TODO: 从 chunk 元数据中提取
                "surrounding_chunks": surrounding_chunks,
            })
            total_tokens += tokens
        
        return result


def _get_surrounding_chunks(
    cur, 
    version_id: str, 
    chunk_index: int,
    window: int = 1
) -> list[str]:
    """获取指定 chunk 前后的 chunks 文本
    
    Args:
        cur: 数据库 cursor
        version_id: 版本 ID
        chunk_index: 当前 chunk 索引
        window: 前后各取多少个 chunks
    
    Returns:
        周围 chunks 的文本列表
    """
    query = """
        SELECT text_raw
        FROM chunks
        WHERE version_id = %s
          AND chunk_index BETWEEN %s AND %s
          AND text_raw IS NOT NULL
          AND text_raw != ''
        ORDER BY chunk_index
    """
    cur.execute(query, (version_id, chunk_index - window, chunk_index + window))
    rows = cur.fetchall()
    
    # 排除当前 chunk 本身
    surrounding = [r[0] for r in rows if r[0]]
    return surrounding


def process_batch(
    conn, 
    chunks: list[dict[str, Any]], 
    context_generator: ContextualRetrievalGenerator,
    embedding_client = None,
    embedding_model: str | None = None,
    show_detail: bool = False,
) -> tuple[int, int]:
    """处理一批数据
    
    Args:
        conn: 数据库连接
        chunks: chunk 数据列表
        context_generator: 上下文生成器
        embedding_client: OpenAI 客户端（用于 embedding）
        embedding_model: embedding 模型名称
        show_detail: 是否显示详细信息
    
    Returns:
        (成功数量, 失败数量)
    """
    if not chunks:
        return 0, 0
    
    try:
        # 步骤 1: 生成上下文前缀
        if show_detail:
            print("  📝 生成上下文...", end=" ", flush=True)
        
        context_chunks = [
            {
                "chunk_text": c["text_raw"],
                "document_title": c["document_title"],
                "section_title": c.get("section_title"),
                "surrounding_chunks": c.get("surrounding_chunks"),
            }
            for c in chunks
        ]
        
        context_prefixes = context_generator.generate_context_batch(context_chunks)
        
        if show_detail:
            print("✅")
        
        # 步骤 2: 准备 contextualized_text
        contextualized_texts = []
        for i, chunk in enumerate(chunks):
            prefix = context_prefixes[i]
            contextualized = f"{prefix}\n\n{chunk['text_raw']}"
            contextualized_texts.append(contextualized)
        
        # 步骤 3: 生成 embeddings
        if show_detail:
            print("  🔢 生成向量...", end=" ", flush=True)
        
        embeddings = embed_texts(
            contextualized_texts,
            client=embedding_client,
            model=embedding_model,
            batch_size=10,  # 内部再分批
            show_progress=False,
        )
        
        if show_detail:
            print("✅")
        
        # 步骤 4: 批量插入数据库
        if show_detail:
            print("  💾 保存到数据库...", end=" ", flush=True)
        
        with conn.cursor() as cur:
            insert_data = []
            for i, chunk in enumerate(chunks):
                insert_data.append((
                    chunk["chunk_id"],
                    chunk["version_id"],
                    chunk["text_raw"],
                    context_prefixes[i],
                    contextualized_texts[i],
                    embeddings[i],
                    context_generator.model,
                    embedding_model or get_embedding_config()["model"],
                ))
            
            cur.executemany(
                """
                INSERT INTO contextual_chunks (
                    chunk_id, version_id, original_text, context_prefix,
                    contextualized_text, embedding, model_name, embedding_model
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                insert_data
            )
        
        conn.commit()
        
        if show_detail:
            print("✅")
        
        return len(chunks), 0
        
    except Exception as e:
        conn.rollback()
        print(f"\n  ❌ 批次处理失败: {e}")
        return 0, len(chunks)


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
    parser = argparse.ArgumentParser(description="批量生成上下文增强 chunks")
    parser.add_argument("--version-id", help="指定版本 ID")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"批次大小（默认 {DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"最大处理数量（默认 {DEFAULT_LIMIT}）")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"每批最大 token 数（默认 {DEFAULT_MAX_TOKENS}）")
    parser.add_argument("--llm-model", default="gpt-4", help="LLM 模型（默认 gpt-4）")
    parser.add_argument("--show-detail", action="store_true", help="显示详细进度")
    parser.add_argument("--dry-run", action="store_true", help="干运行模式（不实际写入数据库）")
    parser.add_argument("--reset", "-r", action="store_true", help="重置/清空 contextual_chunks 表后重新生成")
    parser.add_argument("--force", action="store_true", help="强制重置，跳过确认提示（配合 --reset 使用）")
    args = parser.parse_args()
    
    # 检查 API Key
    if not settings.get("OPENAI_API_KEY"):
        print("❌ 错误: OPENAI_API_KEY 未设置")
        print("请设置环境变量: export OPENAI_API_KEY=sk-xxx")
        sys.exit(1)
    
    # 获取 embedding 配置
    config = get_embedding_config()
    
    print("=" * 80)
    print("🚀 开始生成上下文增强 chunks")
    print("=" * 80)
    print(f"LLM 模型: {args.llm_model}")
    print(f"Embedding 模型: {config['model']}")
    print(f"Embedding 维度: {config['dim']}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大 Token: {args.max_tokens:,}")
    if args.version_id:
        print(f"版本过滤: {args.version_id}")
    if args.dry_run:
        print("⚠️ 干运行模式: 不会写入数据库")
    print()
    
    # 初始化 OpenAI 客户端
    llm_client = ContextualRetrievalGenerator.get_openai_client() if hasattr(ContextualRetrievalGenerator, 'get_openai_client') else None
    if llm_client is None:
        from openai import OpenAI
        llm_client = OpenAI(
            api_key=settings["OPENAI_API_KEY"],
            base_url=settings.get("OPENAI_BASE_URL"),
        )
    
    embedding_client = get_embedding_client()
    
    # 初始化上下文生成器
    context_generator = ContextualRetrievalGenerator(
        client=llm_client,
        model=args.llm_model,
        temperature=0.0,
        max_tokens=200,
    )
    
    # 连接数据库
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        
        # 处理重置请求
        if args.reset:
            if not reset_contextual_chunks(conn, args.version_id, args.force):
                sys.exit(0)  # 用户取消，正常退出
            print()
        
        # 获取初始统计
        stats = get_stats(conn, args.version_id)
        
        print("=" * 80)
        print("📊 初始统计")
        print("=" * 80)
        print(f"  总 chunks:    {stats['total_chunks']:,}")
        print(f"  已处理:       {stats['processed']:,}")
        print(f"  待处理:       {stats['to_process']:,}")
        print(f"  无法处理:     {stats['empty_text']:,} (空文本)")
        print()
        
        if stats['to_process'] == 0:
            print("✅ 所有 chunks 都已处理，无需处理")
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
            chunks = fetch_batch(
                conn, 
                batch_size=batch_size,
                max_tokens=args.max_tokens,
                version_id=args.version_id,
            )
            
            if not chunks:
                print("没有更多数据需要处理")
                break
            
            batch_num += 1
            batch_len = len(chunks)
            
            # 估算 token 数
            batch_tokens = sum(estimate_tokens(c["text_raw"]) for c in chunks)
            
            print(f"批次 {batch_num:>3}: {batch_len:>3} 条 ({batch_tokens:,} tokens)...", end=" ", flush=True)
            
            if args.dry_run:
                # 干运行模式：只打印，不实际处理
                print("⏭️  跳过（干运行）")
                processed += batch_len
                continue
            
            # 处理批次
            success, fail = process_batch(
                conn, chunks, 
                context_generator=context_generator,
                embedding_client=embedding_client,
                embedding_model=config['model'],
                show_detail=args.show_detail,
            )
            
            total_success += success
            total_fail += fail
            processed += batch_len
            
            if success == batch_len:
                elapsed = time.time() - start_time
                speed = total_success / elapsed if elapsed > 0 else 0
                print(f"✅ ({speed:.2f} 条/秒)")
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
            print(f"平均速度: {total_success/elapsed:.2f} 条/秒")
        
        # 获取最终统计
        final_stats = get_stats(conn, args.version_id)
        print()
        print("📊 最终状态")
        print(f"  已处理: {final_stats['processed']:,} / {final_stats['total_chunks']:,} ({100*final_stats['processed']/final_stats['total_chunks']:.1f}%)")
        print(f"  待处理: {final_stats['to_process']:,}")
        print(f"  无法处理: {final_stats['empty_text']:,}")


if __name__ == "__main__":
    main()
