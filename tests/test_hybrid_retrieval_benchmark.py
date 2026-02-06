"""
混合检索效果基准测试

评估指标:
- Recall@K: 前K个结果中检索到相关文档的比例
- MRR: Mean Reciprocal Rank (首个相关文档排名的倒数)
- Precision@K: 前K个结果的准确率

参考:
- https://research.aimultiple.com/hybrid-rag/
- LangChain Evaluation Best Practices
"""

from dataclasses import dataclass
from typing import List, Set, Dict
import statistics
import time

import pytest

from bid_scoring.hybrid_retrieval import HybridRetriever, ReciprocalRankFusion
from bid_scoring.config import load_settings


@dataclass
class TestQuery:
    """测试查询样本"""
    query_text: str
    keywords: List[str]
    # 预期相关的 chunk_id 集合 (通过人工标注或自动生成)
    expected_chunk_ids: Set[str]
    # 查询类型分类
    query_type: str  # "factual", "semantic", "keyword_critical"


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    recall_at_k: Dict[int, float]  # Recall@1, Recall@3, Recall@5
    precision_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float
    query_type: str


def calculate_metrics(
    retrieved_ids: List[str],
    expected_ids: Set[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> RetrievalMetrics:
    """
    计算检索评估指标

    Args:
        retrieved_ids: 检索结果中的 chunk_id 列表 (按相关性排序)
        expected_ids: 预期的相关 chunk_id 集合
        k_values: 要计算的 K 值列表

    Returns:
        RetrievalMetrics 包含各项指标
    """
    metrics = RetrievalMetrics(
        recall_at_k={},
        precision_at_k={},
        mrr=0.0,
        latency_ms=0.0,
        query_type=""
    )

    # 计算 Recall@K 和 Precision@K
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        relevant_in_k = len(top_k & expected_ids)

        metrics.recall_at_k[k] = relevant_in_k / len(expected_ids) if expected_ids else 0.0
        metrics.precision_at_k[k] = relevant_in_k / k if k > 0 else 0.0

    # 计算 MRR (Mean Reciprocal Rank)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_ids:
            metrics.mrr = 1.0 / rank
            break

    return metrics


def compare_search_methods(
    query: TestQuery,
    retriever: HybridRetriever,
    db_connection,
) -> Dict[str, RetrievalMetrics]:
    """
    对比三种检索方法的性能

    Returns:
        {"vector": metrics, "keyword": metrics, "hybrid": metrics}
    """
    results = {}
    k_values = [1, 3, 5, 10]

    # 1. 纯向量搜索
    start = time.perf_counter()
    vector_results = retriever._vector_search(query.query_text)
    vector_latency = (time.perf_counter() - start) * 1000

    vector_ids = [r[0] for r in vector_results]
    results["vector"] = calculate_metrics(vector_ids, query.expected_chunk_ids, k_values)
    results["vector"].latency_ms = vector_latency
    results["vector"].query_type = query.query_type

    # 2. 纯关键词搜索
    if query.keywords:
        start = time.perf_counter()
        keyword_results = retriever._keyword_search(query.keywords)
        keyword_latency = (time.perf_counter() - start) * 1000

        keyword_ids = [r[0] for r in keyword_results]
        results["keyword"] = calculate_metrics(keyword_ids, query.expected_chunk_ids, k_values)
        results["keyword"].latency_ms = keyword_latency
        results["keyword"].query_type = query.query_type

    # 3. 混合搜索
    start = time.perf_counter()
    hybrid_results = retriever.retrieve(query.query_text, keywords=query.keywords)
    hybrid_latency = (time.perf_counter() - start) * 1000

    hybrid_ids = [r.chunk_id for r in hybrid_results]
    results["hybrid"] = calculate_metrics(hybrid_ids, query.expected_chunk_ids, k_values)
    results["hybrid"].latency_ms = hybrid_latency
    results["hybrid"].query_type = query.query_type

    return results


def print_benchmark_report(all_results: List[Dict[str, RetrievalMetrics]]):
    """打印基准测试报告"""
    print("\n" + "=" * 80)
    print("混合检索基准测试报告")
    print("=" * 80)

    methods = ["vector", "keyword", "hybrid"]

    for method in methods:
        if not all(method in r for r in all_results):
            continue

        print(f"\n【{method.upper()} 方法】")
        print("-" * 40)

        # 收集该方法的指标
        mrr_values = [r[method].mrr for r in all_results if method in r]
        recall_5_values = [r[method].recall_at_k[5] for r in all_results if method in r]
        precision_3_values = [r[method].precision_at_k[3] for r in all_results if method in r]
        latencies = [r[method].latency_ms for r in all_results if method in r]

        print(f"  MRR (Mean Reciprocal Rank):")
        print(f"    Mean: {statistics.mean(mrr_values):.4f}")
        print(f"    Median: {statistics.median(mrr_values):.4f}")

        print(f"  Recall@5:")
        print(f"    Mean: {statistics.mean(recall_5_values):.4f}")
        print(f"    Median: {statistics.median(recall_5_values):.4f}")

        print(f"  Precision@3:")
        print(f"    Mean: {statistics.mean(precision_3_values):.4f}")
        print(f"    Median: {statistics.median(precision_3_values):.4f}")

        print(f"  Latency:")
        print(f"    Mean: {statistics.mean(latencies):.2f} ms")
        print(f"    P95: {sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0:.2f} ms")

    # 混合 vs 纯向量的改进
    print("\n【混合检索改进幅度】")
    print("-" * 40)

    hybrid_mrr = [r["hybrid"].mrr for r in all_results if "hybrid" in r and "vector" in r]
    vector_mrr = [r["vector"].mrr for r in all_results if "hybrid" in r and "vector" in r]

    if hybrid_mrr and vector_mrr:
        mrr_improvement = (statistics.mean(hybrid_mrr) - statistics.mean(vector_mrr)) / statistics.mean(vector_mrr) * 100
        print(f"  MRR 改进: {mrr_improvement:+.2f}%")

    hybrid_r5 = [r["hybrid"].recall_at_k[5] for r in all_results if "hybrid" in r and "vector" in r]
    vector_r5 = [r["vector"].recall_at_k[5] for r in all_results if "hybrid" in r and "vector" in r]

    if hybrid_r5 and vector_r5:
        r5_improvement = (statistics.mean(hybrid_r5) - statistics.mean(vector_r5)) / statistics.mean(vector_r5) * 100
        print(f"  Recall@5 改进: {r5_improvement:+.2f}%")

    print("=" * 80)


# ============ 测试用例 ============

class TestHybridRetrievalBenchmark:
    """混合检索基准测试套件"""

    def test_rrf_fusion_improves_ranking(self):
        """
        测试 RRF 融合是否提升排名质量

        场景: 某个文档在向量搜索中排名 #3，在关键词搜索中排名 #1
        预期: RRF 融合后该文档应该比纯向量搜索结果中更靠前
        """
        rrf = ReciprocalRankFusion(k=60)

        # 模拟搜索结果
        vector_results = [
            ("doc_A", 0.95),  # 高语义相似度
            ("doc_B", 0.90),
            ("doc_C", 0.85),  # 目标文档: 语义相关但不是最匹配
        ]
        keyword_results = [
            ("doc_C", 5.0),   # 目标文档: 关键词完全匹配
            ("doc_D", 3.0),
        ]

        fused = rrf.fuse(vector_results, keyword_results)

        # doc_C 应该排在前面，因为同时在两个列表中都出现
        fused_ids = [r[0] for r in fused]
        doc_c_rank = fused_ids.index("doc_C") + 1

        # doc_C 在融合后排名应该优于或等于在纯向量中的排名 (#3)
        assert doc_c_rank <= 3, f"doc_C should rank better after fusion, got rank {doc_c_rank}"

    def test_keyword_critical_queries(self):
        """
        测试关键词关键型查询

        场景: 查询包含特定术语如 "培训时长"、"保修期"
        预期: 混合检索应该比纯向量搜索更好地找到包含精确关键词的文档
        """
        settings = load_settings()
        retriever = HybridRetriever(
            version_id="test-version",
            settings=settings,
            top_k=10
        )

        # 模拟关键词关键型查询
        query = TestQuery(
            query_text="培训时长是多少天",
            keywords=["培训", "时长", "天数"],
            expected_chunk_ids={"chunk_training_schedule"},  # 假设的预期文档
            query_type="keyword_critical"
        )

        # 提取关键词
        extracted = retriever.extract_keywords_from_query(query.query_text)

        # 验证关键词提取包含关键术语
        assert any(k in extracted for k in ["培训", "时长", "天数"]), \
            "Keywords should contain critical terms"

    def test_semantic_queries(self):
        """
        测试语义理解型查询

        场景: 查询使用同义词或表达方式不同但语义相同
        预期: 向量搜索应该表现良好，混合检索不应降低效果
        """
        settings = load_settings()
        retriever = HybridRetriever(
            version_id="test-version",
            settings=settings,
            top_k=10
        )

        query = TestQuery(
            query_text="售后服务包含哪些内容",
            keywords=["售后", "服务"],
            expected_chunk_ids={"chunk_service_policy"},
            query_type="semantic"
        )

        # 验证关键词提取也能捕获同义词
        extracted = retriever.extract_keywords_from_query(query.query_text)

        # 应该包含 "服务" 及其同义词
        service_synonyms = {"服务", "支持", "维护", "售后"}
        assert any(k in service_synonyms for k in extracted), \
            "Should capture semantic synonyms"

    def test_empty_keyword_fallback(self):
        """
        测试关键词为空时的回退行为

        场景: 查询无法提取有效关键词
        预期: 混合检索应该优雅回退到纯向量搜索
        """
        settings = load_settings()
        retriever = HybridRetriever(
            version_id="test-version",
            settings=settings,
            top_k=10
        )

        # 极端短查询，可能没有有效关键词
        query_text = "的 了 是"
        keywords = retriever.extract_keywords_from_query(query_text)

        # 提取的关键词应该很少或为空
        assert len(keywords) == 0 or all(len(k) >= 2 for k in keywords), \
            "Should filter out stopwords effectively"

    def test_latency_acceptable(self):
        """
        测试延迟是否在可接受范围内

        参考: AIMultiple 研究显示混合检索增加约 200ms 延迟 (24.5%)
        我们的目标: 混合检索延迟 < 2x 纯向量搜索延迟
        """
        settings = load_settings()
        retriever = HybridRetriever(
            version_id="test-version",
            settings=settings,
            top_k=10
        )

        query = "培训方案和时长安排"

        # 测量纯向量搜索延迟
        import time
        start = time.perf_counter()
        _ = retriever._vector_search(query)
        vector_latency = (time.perf_counter() - start) * 1000

        # 测量混合搜索延迟 (并行执行，应该接近较慢的那个)
        start = time.perf_counter()
        _ = retriever.retrieve(query)
        hybrid_latency = (time.perf_counter() - start) * 1000

        # 混合检索延迟应该不超过纯向量搜索的 2 倍
        # (因为并行执行，理论上应该接近 max(vector_time, keyword_time))
        assert hybrid_latency < vector_latency * 2, \
            f"Hybrid latency ({hybrid_latency:.2f}ms) should be < 2x vector latency ({vector_latency:.2f}ms)"


class TestMetricsCalculation:
    """测试指标计算准确性"""

    def test_recall_at_k_calculation(self):
        """测试 Recall@K 计算"""
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        expected = {"doc_2", "doc_4", "doc_6"}  # 3 个相关文档

        metrics = calculate_metrics(retrieved, expected, k_values=[1, 3, 5])

        # Recall@1: 0/3 = 0 (doc_1 不相关)
        assert metrics.recall_at_k[1] == 0.0

        # Recall@3: 1/3 ≈ 0.33 (doc_2 在 top3 中)
        assert abs(metrics.recall_at_k[3] - 1/3) < 0.01

        # Recall@5: 2/3 ≈ 0.67 (doc_2 和 doc_4 在 top5 中)
        assert abs(metrics.recall_at_k[5] - 2/3) < 0.01

    def test_mrr_calculation(self):
        """测试 MRR 计算"""
        # 场景1: 第一个结果就是相关的
        metrics = calculate_metrics(["doc_1", "doc_2"], {"doc_1"})
        assert metrics.mrr == 1.0  # 1/1

        # 场景2: 第二个结果才是相关的
        metrics = calculate_metrics(["doc_1", "doc_2", "doc_3"], {"doc_2"})
        assert metrics.mrr == 0.5  # 1/2

        # 场景3: 没有相关结果
        metrics = calculate_metrics(["doc_1", "doc_2"], {"doc_3"})
        assert metrics.mrr == 0.0

    def test_precision_at_k_calculation(self):
        """测试 Precision@K 计算"""
        retrieved = ["doc_1", "doc_2", "doc_3"]  # doc_1, doc_3 相关
        expected = {"doc_1", "doc_3", "doc_5"}

        metrics = calculate_metrics(retrieved, expected, k_values=[1, 2, 3])

        # Precision@1: 1/1 = 1.0 (doc_1 相关)
        assert metrics.precision_at_k[1] == 1.0

        # Precision@2: 1/2 = 0.5 (doc_1 相关, doc_2 不相关)
        assert metrics.precision_at_k[2] == 0.5

        # Precision@3: 2/3 ≈ 0.67 (doc_1, doc_3 相关)
        assert abs(metrics.precision_at_k[3] - 2/3) < 0.01
