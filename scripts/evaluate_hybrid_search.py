#!/usr/bin/env python3
"""
混合检索效果评估脚本

用法:
    uv run python scripts/evaluate_hybrid_search.py --version-id <version_id> [--top-k 10]

功能:
1. 对比三种检索方法: Vector-only, Keyword-only, Hybrid
2. 计算 Recall@K, Precision@K, MRR 指标
3. 测量各方法的延迟
4. 生成评估报告

参考:
- https://research.aimultiple.com/hybrid-rag/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Set

import psycopg

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bid_scoring.config import load_settings
from bid_scoring.hybrid_retrieval import HybridRetriever


@dataclass
class TestQuery:
    """测试查询定义"""
    query: str
    keywords: list[str]
    description: str
    query_type: str  # "factual", "semantic", "keyword_critical"


@dataclass
class Metrics:
    """检索指标"""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mrr: float
    latency_ms: float


# 预定义的测试查询集 (基于真实业务场景)
TEST_QUERIES: list[TestQuery] = [
    # 关键词关键型查询 (Keyword Critical)
    TestQuery(
        query="培训时长是多少天",
        keywords=["培训", "时长", "天数", "时间"],
        description="培训时长询问",
        query_type="keyword_critical"
    ),
    TestQuery(
        query="保修期多长时间",
        keywords=["保修", "质保", "保修期", "时间"],
        description="保修期询问",
        query_type="keyword_critical"
    ),
    TestQuery(
        query="配件清单有哪些",
        keywords=["配件", "备件", "清单", "耗材"],
        description="配件清单询问",
        query_type="keyword_critical"
    ),

    # 事实型查询 (Factual)
    TestQuery(
        query="售后服务电话是多少",
        keywords=["售后", "服务", "电话", "400"],
        description="售后服务电话",
        query_type="factual"
    ),
    TestQuery(
        query="安装调试需要多久",
        keywords=["安装", "调试", "时间", "周期"],
        description="安装调试周期",
        query_type="factual"
    ),

    # 语义理解型查询 (Semantic)
    TestQuery(
        query="设备出现问题怎么办",
        keywords=["设备", "问题", "故障", "维修"],
        description="故障处理流程",
        query_type="semantic"
    ),
    TestQuery(
        query="如何使用这个系统",
        keywords=["使用", "系统", "操作", "方法"],
        description="系统使用方法",
        query_type="semantic"
    ),
    TestQuery(
        query="培训内容包括哪些",
        keywords=["培训", "内容", "课程", "大纲"],
        description="培训内容概述",
        query_type="semantic"
    ),

    # 复合型查询
    TestQuery(
        query="响应时间和到场时间要求",
        keywords=["响应", "到场", "时间", "时效"],
        description="服务响应时效",
        query_type="keyword_critical"
    ),
    TestQuery(
        query="质保期满后如何维护",
        keywords=["质保", "维护", "保修期", "服务"],
        description="质保期后维护",
        query_type="semantic"
    ),
]


def get_all_chunks_for_version(conn, version_id: str) -> Set[str]:
    """获取版本下的所有 chunk_id"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id::text FROM chunks WHERE version_id = %s",
            (version_id,)
        )
        return {row[0] for row in cur.fetchall()}


def find_relevant_chunks(
    conn,
    version_id: str,
    query: str,
    keywords: list[str]
) -> Set[str]:
    """
    基于关键词匹配找到相关 chunk_id
    这是一个简单的相关性判断方法
    """
    relevant = set()

    with conn.cursor() as cur:
        # 使用全文搜索找到相关文档
        keyword_pattern = " | ".join(keywords)
        cur.execute(
            """
            SELECT chunk_id::text
            FROM chunks
            WHERE version_id = %s
              AND text_tsv @@ to_tsquery('simple', %s)
            LIMIT 20
            """,
            (version_id, keyword_pattern)
        )
        relevant.update(row[0] for row in cur.fetchall())

        # 同时检查包含所有关键词的文档
        if keywords:
            conditions = " AND ".join(["text_raw ILIKE %s"] * len(keywords))
            patterns = [f"%{k}%" for k in keywords]
            cur.execute(
                f"""
                SELECT chunk_id::text
                FROM chunks
                WHERE version_id = %s
                  AND ({conditions})
                LIMIT 20
                """,
                (version_id,) + tuple(patterns)
            )
            relevant.update(row[0] for row in cur.fetchall())

    return relevant


def calculate_metrics(
    retrieved: list[str],
    relevant: Set[str],
    latency_ms: float
) -> Metrics:
    """计算检索指标"""
    def recall_at_k(k: int) -> float:
        if not relevant:
            return 0.0
        top_k = set(retrieved[:k])
        return len(top_k & relevant) / len(relevant)

    def precision_at_k(k: int) -> float:
        if k == 0:
            return 0.0
        top_k = set(retrieved[:k])
        relevant_in_k = top_k & relevant
        return len(relevant_in_k) / k

    def mrr() -> float:
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                return 1.0 / rank
        return 0.0

    return Metrics(
        recall_at_1=recall_at_k(1),
        recall_at_3=recall_at_k(3),
        recall_at_5=recall_at_k(5),
        recall_at_10=recall_at_k(10),
        precision_at_1=precision_at_k(1),
        precision_at_3=precision_at_k(3),
        precision_at_5=precision_at_k(5),
        mrr=mrr(),
        latency_ms=latency_ms
    )


def evaluate_search_method(
    retriever: HybridRetriever,
    conn,
    test_query: TestQuery,
    method: str,
    top_k: int = 10
) -> tuple[list[str], Metrics]:
    """
    评估单个搜索方法

    Returns:
        (retrieved_ids, metrics)
    """
    # 测量延迟
    start = time.perf_counter()

    if method == "vector":
        results = retriever._vector_search(test_query.query)
        retrieved = [r[0] for r in results[:top_k]]
    elif method == "keyword":
        results = retriever._keyword_search(test_query.keywords)
        retrieved = [r[0] for r in results[:top_k]]
    else:  # hybrid
        results = retriever.retrieve(test_query.query, keywords=test_query.keywords)
        retrieved = [r.chunk_id for r in results[:top_k]]

    latency_ms = (time.perf_counter() - start) * 1000

    # 找到相关文档 (使用关键词作为 ground truth)
    relevant = find_relevant_chunks(conn, retriever.version_id, test_query.query, test_query.keywords)

    metrics = calculate_metrics(retrieved, relevant, latency_ms)
    return retrieved, metrics


def print_report(results: dict[str, list[Metrics]], detailed: bool = False):
    """打印评估报告"""
    print("\n" + "=" * 90)
    print("混合检索效果评估报告")
    print("=" * 90)

    methods = ["vector", "keyword", "hybrid"]

    # 1. 总体指标对比
    print("\n【总体指标对比】")
    print("-" * 90)
    print(f"{'指标':<20} {'Vector Only':>18} {'Keyword Only':>18} {'Hybrid':>18} {'改进':>10}")
    print("-" * 90)

    for metric_name in ["mrr", "recall_at_3", "recall_at_5", "precision_at_3"]:
        row = [metric_name]
        values = []
        for method in methods:
            vals = [getattr(m, metric_name) for m in results[method]]
            mean_val = statistics.mean(vals) if vals else 0.0
            row.append(f"{mean_val:.4f}")
            values.append(mean_val)

        # 计算改进幅度 (Hybrid vs Vector)
        if values[0] > 0:
            improvement = (values[2] - values[0]) / values[0] * 100
            row.append(f"{improvement:+.1f}%")
        else:
            row.append("N/A")

        print("  ".join(f"{v:>18}" if i > 0 else f"{v:<20}" for i, v in enumerate(row)))

    # 延迟对比
    print("-" * 90)
    latency_row = ["Latency (ms)"]
    for method in methods:
        latencies = [m.latency_ms for m in results[method]]
        mean_latency = statistics.mean(latencies) if latencies else 0.0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0
        latency_row.append(f"{mean_latency:.1f} (p95: {p95_latency:.1f})")

    latency_row.append("-")
    print("  ".join(f"{v:>18}" if i > 0 else f"{v:<20}" for i, v in enumerate(latency_row)))

    # 2. 按查询类型分析
    print("\n【按查询类型分析 - Hybrid 方法】")
    print("-" * 70)

    query_types = ["keyword_critical", "factual", "semantic"]
    for qtype in query_types:
        type_metrics = [m for m, q in zip(results["hybrid"], TEST_QUERIES) if q.query_type == qtype]
        if type_metrics:
            mrr_val = statistics.mean([m.mrr for m in type_metrics])
            r5_val = statistics.mean([m.recall_at_5 for m in type_metrics])
            print(f"  {qtype:<20} MRR: {mrr_val:.4f}, Recall@5: {r5_val:.4f}")

    # 3. 详细结果 (可选)
    if detailed:
        print("\n【详细查询结果】")
        print("-" * 90)

        for i, query in enumerate(TEST_QUERIES):
            print(f"\nQuery {i+1}: {query.query}")
            print(f"  Type: {query.query_type}, Keywords: {query.keywords}")

            for method in methods:
                m = results[method][i]
                print(f"  {method:<12} R@5={m.recall_at_5:.2f} P@3={m.precision_at_3:.2f} "
                      f"MRR={m.mrr:.2f} Lat={m.latency_ms:.1f}ms")

    print("\n" + "=" * 90)
    print("说明:")
    print("  - MRR (Mean Reciprocal Rank): 首个相关文档排名的倒数均值，越高越好")
    print("  - Recall@K: 前K个结果中包含相关文档的比例，越高越好")
    print("  - Precision@K: 前K个结果中相关文档的比例，越高越好")
    print("  - 改进: Hybrid 相比纯向量搜索的提升幅度")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="混合检索效果评估")
    parser.add_argument(
        "--version-id",
        type=str,
        help="文档版本 ID",
        default="83420a7c-b27b-480f-9427-565c47d2b53c"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="检索结果数量 (默认: 10)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示详细结果"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出 JSON 文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    settings = load_settings()

    # 创建检索器
    retriever = HybridRetriever(
        version_id=args.version_id,
        settings=settings,
        top_k=args.top_k
    )

    # 运行评估
    print(f"开始评估混合检索效果...")
    print(f"版本 ID: {args.version_id}")
    print(f"Top-K: {args.top_k}")
    print(f"测试查询数: {len(TEST_QUERIES)}")

    results: dict[str, list[Metrics]] = {
        "vector": [],
        "keyword": [],
        "hybrid": []
    }

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        # 验证版本存在
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_versions WHERE version_id = %s",
                (args.version_id,)
            )
            if cur.fetchone()[0] == 0:
                print(f"错误: 版本 {args.version_id} 不存在")
                return 1

        # 评估每个查询
        for i, test_query in enumerate(TEST_QUERIES, 1):
            print(f"  评估查询 {i}/{len(TEST_QUERIES)}: {test_query.query[:30]}...", end=" ")

            for method in ["vector", "keyword", "hybrid"]:
                _, metrics = evaluate_search_method(
                    retriever, conn, test_query, method, args.top_k
                )
                results[method].append(metrics)

            print("✓")

    # 打印报告
    print_report(results, detailed=args.detailed)

    # 保存结果到文件
    if args.output:
        output_data = {
            "version_id": args.version_id,
            "top_k": args.top_k,
            "queries": [
                {
                    "query": q.query,
                    "type": q.query_type,
                    "vector": asdict(v),
                    "keyword": asdict(k),
                    "hybrid": asdict(h)
                }
                for q, v, k, h in zip(
                    TEST_QUERIES,
                    results["vector"],
                    results["keyword"],
                    results["hybrid"]
                )
            ],
            "summary": {
                method: {
                    "mrr": statistics.mean([m.mrr for m in results[method]]),
                    "recall_at_5": statistics.mean([m.recall_at_5 for m in results[method]]),
                    "precision_at_3": statistics.mean([m.precision_at_3 for m in results[method]]),
                    "latency_ms": statistics.mean([m.latency_ms for m in results[method]]),
                }
                for method in ["vector", "keyword", "hybrid"]
            }
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
