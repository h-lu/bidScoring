#!/usr/bin/env python3
"""
真实效果测试 - 使用数据库中的实际内容验证 Hybrid Retrieval 优化

测试内容：
1. 全文搜索 vs ILIKE 性能对比
2. 向量搜索召回率测试
3. 混合搜索效果验证
4. 缓存效果测试
5. 异步接口测试
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

from bid_scoring.config import load_settings
from bid_scoring.hybrid_retrieval import HybridRetriever

# 测试查询（来自实际业务场景）
TEST_QUERIES = [
    "培训时长",
    "售后服务",
    "CT设备参数",
    "MRI磁共振",
    "保修期限",
    "响应时间",
    "工程师资质",
    "配件供应",
    "设备安装",
    "验收标准",
]

SETTINGS = load_settings()
VERSION_ID = "83420a7c-b27b-480f-9427-565c47d2b53c"  # 使用实际版本


def test_fulltext_vs_ilike():
    """测试 1: 全文搜索 vs ILIKE 性能对比 (AND vs OR 语义)"""
    print("\n" + "=" * 60)
    print("测试 1: 全文搜索 AND vs OR 语义对比")
    print("=" * 60)
    
    retriever = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=10,
    )
    
    keywords = ["培训", "时长"]
    
    # 测试 OR 语义（默认，提高召回率）
    or_times = []
    for _ in range(5):
        start = time.perf_counter()
        results = retriever._keyword_search_fulltext(keywords, use_or_semantic=True)
        or_times.append(time.perf_counter() - start)
    
    or_avg = statistics.mean(or_times) * 1000
    or_results = len(results)
    
    # 测试 AND 语义（提高精确率）
    and_times = []
    for _ in range(5):
        start = time.perf_counter()
        results = retriever._keyword_search_fulltext(keywords, use_or_semantic=False)
        and_times.append(time.perf_counter() - start)
    
    and_avg = statistics.mean(and_times) * 1000
    and_results = len(results)
    
    print(f"\n查询关键词: {keywords}")
    print(f"  OR 语义 (默认，提高召回率):")
    print(f"    - 平均耗时: {or_avg:.2f} ms")
    print(f"    - 返回结果: {or_results} 条")
    print(f"  AND 语义 (提高精确率):")
    print(f"    - 平均耗时: {and_avg:.2f} ms")
    print(f"    - 返回结果: {and_results} 条")
    
    if and_results > 0:
        recall_boost = or_results / and_results
        print(f"\n  📈 OR 语义召回提升: {recall_boost:.1f}x")
    elif or_results > 0:
        print(f"\n  📈 OR 语义召回提升: 无限 (AND 无结果，OR 有结果)")
    
    # 测试 ILIKE（遗留方法）
    ilike_times = []
    for _ in range(5):
        start = time.perf_counter()
        results = retriever._keyword_search_legacy(keywords)
        ilike_times.append(time.perf_counter() - start)
    
    ilike_avg = statistics.mean(ilike_times) * 1000
    ilike_results = len(results)
    
    print(f"\n  ILIKE (旧方法):")
    print(f"    - 平均耗时: {ilike_avg:.2f} ms")
    print(f"    - 返回结果: {ilike_results} 条")
    
    if or_avg > 0:
        speedup = ilike_avg / or_avg
        print(f"\n  ⚡ 全文搜索性能提升: {speedup:.1f}x")
    
    retriever.close()


def test_vector_recall():
    """测试 2: 向量搜索召回率测试（不同 ef_search）"""
    print("\n" + "=" * 60)
    print("测试 2: HNSW ef_search 参数对召回率的影响")
    print("=" * 60)
    
    query = "CT设备技术参数要求"
    
    for ef in [40, 100, 200]:
        retriever = HybridRetriever(
            version_id=VERSION_ID,
            settings=SETTINGS,
            top_k=10,
            hnsw_ef_search=ef,
        )
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            results = retriever._vector_search(query)
            times.append(time.perf_counter() - start)
        
        avg_time = statistics.mean(times) * 1000
        print(f"\n  ef_search={ef}:")
        print(f"    - 平均耗时: {avg_time:.2f} ms")
        print(f"    - 返回结果: {len(results)} 条")
        if results:
            print(f"    - 最高相似度: {results[0][1]:.4f}")
        
        retriever.close()
    
    print("\n  💡 说明: ef_search=100 是推荐默认值（平衡性能和召回率）")


def test_hybrid_search():
    """测试 3: 混合搜索效果验证"""
    print("\n" + "=" * 60)
    print("测试 3: 混合搜索效果验证")
    print("=" * 60)
    
    queries = [
        "培训时长是多少",
        "CT设备售后服务",
        "MRI核磁共振参数",
    ]
    
    retriever = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=5,
        hnsw_ef_search=100,
        vector_weight=1.0,
        keyword_weight=1.0,
    )
    
    for query in queries:
        print(f"\n  查询: '{query}'")
        
        # 提取关键词
        keywords = retriever.extract_keywords_from_query(query)
        print(f"    扩展关键词: {keywords}")
        
        # 执行混合检索
        start = time.perf_counter()
        results = retriever.retrieve(query)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"    总耗时: {elapsed:.2f} ms")
        print(f"    返回结果: {len(results)} 条")
        
        if results:
            print(f"     top-1 结果:")
            print(f"      - 来源: {results[0].source}")
            print(f"      - RRF分数: {results[0].score:.4f}")
            print(f"      - 向量分数: {results[0].vector_score}")
            print(f"      - 关键词分数: {results[0].keyword_score}")
            print(f"      - 文本片段: {results[0].text[:50]}...")
    
    retriever.close()


def test_rrf_weights():
    """测试 4: RRF 权重效果对比"""
    print("\n" + "=" * 60)
    print("测试 4: RRF 权重效果对比")
    print("=" * 60)
    
    query = "培训时长"
    
    weight_configs = [
        ("平衡", 1.0, 1.0),
        ("向量优先", 2.0, 1.0),
        ("关键词优先", 1.0, 2.0),
    ]
    
    for name, vec_w, key_w in weight_configs:
        retriever = HybridRetriever(
            version_id=VERSION_ID,
            settings=SETTINGS,
            top_k=5,
            vector_weight=vec_w,
            keyword_weight=key_w,
        )
        
        results = retriever.retrieve(query)
        
        print(f"\n  配置: {name} (向量={vec_w}, 关键词={key_w})")
        if results:
            print(f"    top-1 来源: {results[0].source}")
            print(f"    top-1 分数: {results[0].score:.4f}")
        else:
            print(f"    结果: 无匹配")
        
        retriever.close()


def test_cache_performance():
    """测试 5: 缓存效果测试"""
    print("\n" + "=" * 60)
    print("测试 5: 查询缓存效果测试")
    print("=" * 60)
    
    query = "售后服务响应时间"
    
    # 无缓存
    retriever_no_cache = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=10,
        enable_cache=False,
    )
    
    times_no_cache = []
    for _ in range(3):
        start = time.perf_counter()
        retriever_no_cache.retrieve(query)
        times_no_cache.append(time.perf_counter() - start)
    
    avg_no_cache = statistics.mean(times_no_cache) * 1000
    retriever_no_cache.close()
    
    # 有缓存
    retriever_with_cache = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=10,
        enable_cache=True,
        cache_size=100,
    )
    
    # 第一次（冷缓存）
    start = time.perf_counter()
    retriever_with_cache.retrieve(query)
    cold_time = (time.perf_counter() - start) * 1000
    
    # 第二次（热缓存）
    start = time.perf_counter()
    retriever_with_cache.retrieve(query)
    hot_time = (time.perf_counter() - start) * 1000
    
    stats = retriever_with_cache.get_cache_stats()
    retriever_with_cache.close()
    
    print(f"\n  查询: '{query}'")
    print(f"  无缓存模式:")
    print(f"    - 平均耗时: {avg_no_cache:.2f} ms")
    print(f"  有缓存模式:")
    print(f"    - 冷缓存: {cold_time:.2f} ms")
    print(f"    - 热缓存: {hot_time:.2f} ms")
    print(f"    - 缓存状态: {stats}")
    
    if hot_time > 0:
        speedup = avg_no_cache / hot_time
        print(f"\n  ⚡ 缓存加速: {speedup:.1f}x")


@pytest.mark.asyncio
async def test_async_performance():
    """测试 6: 异步接口性能测试"""
    print("\n" + "=" * 60)
    print("测试 6: 异步接口性能测试")
    print("=" * 60)
    
    retriever = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=10,
    )
    
    queries = TEST_QUERIES[:5]
    
    # 同步顺序执行
    print("\n  同步顺序执行 (5 个查询):")
    start = time.perf_counter()
    for query in queries:
        retriever.retrieve(query)
    sync_time = (time.perf_counter() - start) * 1000
    print(f"    - 总耗时: {sync_time:.2f} ms")
    print(f"    - 平均: {sync_time/len(queries):.2f} ms/查询")
    
    # 异步并发执行
    print("\n  异步并发执行 (5 个查询):")
    start = time.perf_counter()
    await asyncio.gather(*[
        retriever.retrieve_async(query)
        for query in queries
    ])
    async_time = (time.perf_counter() - start) * 1000
    print(f"    - 总耗时: {async_time:.2f} ms")
    print(f"    - 平均: {async_time/len(queries):.2f} ms/查询")
    
    if async_time > 0:
        speedup = sync_time / async_time
        print(f"\n  ⚡ 并发加速: {speedup:.1f}x")
    
    await retriever.close_async()


def test_connection_pool():
    """测试 7: 连接池效果测试"""
    print("\n" + "=" * 60)
    print("测试 7: 连接池效果测试")
    print("=" * 60)
    
    query = "培训时长"
    
    # 无连接池
    retriever_no_pool = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=10,
        use_connection_pool=False,
    )
    
    times_no_pool = []
    for _ in range(5):
        start = time.perf_counter()
        retriever_no_pool.retrieve(query)
        times_no_pool.append(time.perf_counter() - start)
    
    avg_no_pool = statistics.mean(times_no_pool) * 1000
    retriever_no_pool.close()
    
    # 有连接池
    retriever_with_pool = HybridRetriever(
        version_id=VERSION_ID,
        settings=SETTINGS,
        top_k=10,
        use_connection_pool=True,
        pool_min_size=2,
        pool_max_size=5,
    )
    
    times_with_pool = []
    for _ in range(5):
        start = time.perf_counter()
        retriever_with_pool.retrieve(query)
        times_with_pool.append(time.perf_counter() - start)
    
    avg_with_pool = statistics.mean(times_with_pool) * 1000
    retriever_with_pool.close()
    
    print(f"\n  查询: '{query}'")
    print(f"  无连接池:")
    print(f"    - 平均耗时: {avg_no_pool:.2f} ms")
    print(f"  有连接池:")
    print(f"    - 平均耗时: {avg_with_pool:.2f} ms")
    
    if avg_with_pool > 0:
        speedup = avg_no_pool / avg_with_pool
        print(f"\n  ⚡ 连接池加速: {speedup:.1f}x")


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Hybrid Retrieval 真实效果测试")
    print("=" * 60)
    print(f"\n数据库: {SETTINGS.get('DATABASE_URL', 'N/A').split('@')[-1]}")
    print(f"测试版本: {VERSION_ID}")
    print(f"测试查询数: {len(TEST_QUERIES)}")
    
    # 执行所有测试
    test_fulltext_vs_ilike()
    test_vector_recall()
    test_hybrid_search()
    test_rrf_weights()
    test_cache_performance()
    await test_async_performance()
    test_connection_pool()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
