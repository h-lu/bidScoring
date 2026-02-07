# Hybrid Retrieval 使用文档

本文档介绍如何使用优化后的 Hybrid Retrieval 模块。

## 快速开始

```python
from bid_scoring.config import load_settings
from bid_scoring.retrieval import HybridRetriever

settings = load_settings()

# 基础用法
retriever = HybridRetriever(
    version_id="your-version-id",
    settings=settings,
)

results = retriever.retrieve("培训时长要求")
for r in results:
    print(f"[{r.source}] {r.text[:100]}...")

retriever.close()
```

> 兼容说明：历史代码仍可使用 `from bid_scoring.hybrid_retrieval import HybridRetriever`（保留兼容层），但推荐迁移到 `bid_scoring.retrieval`。

## MCP Server (FastMCP)

本项目提供只读 MCP Server：`mcp_servers/retrieval_server.py`，暴露工具 `retrieve`（默认 `mode="hybrid"`，默认返回全文，可用 `max_chars` 截断）。

### 运行（stdio，给 MCP Client 用）

```bash
uv run fastmcp run mcp_servers/retrieval_server.py -t stdio
```

### 运行（http，本地调试）

```bash
uv run fastmcp run mcp_servers/retrieval_server.py -t http --host 127.0.0.1 --port 8000
```

### 关键环境变量

- `DATABASE_URL`: Postgres 连接串（需要 pgvector + 全文索引）
- `OPENAI_API_KEY`: query embedding（`mode="vector"/"hybrid"` 必需；`mode="keyword"` 可不需要）
- `BID_SCORING_RETRIEVER_CACHE_SIZE`: 服务端 retriever LRU（默认 32）
- `BID_SCORING_QUERY_CACHE_SIZE`: 每个 retriever 的 query LRU（默认 1024）

## 高级配置

### 1. 连接池配置

使用连接池减少数据库连接开销：

```python
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    use_connection_pool=True,    # 启用连接池
    pool_min_size=2,             # 最小连接数
    pool_max_size=10,            # 最大连接数
)
```

### 2. HNSW 参数优化

调整 HNSW 索引参数平衡召回率和性能：

```python
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    hnsw_ef_search=100,  # 默认 100，增大可提高召回率
    # ef_search 选项:
    # - 40:  更快，召回率较低 (pgvector 默认)
    # - 100: 平衡 (推荐)
    # - 200: 更慢，召回率更高
)
```

### 3. RRF 权重配置

调整向量和关键词搜索的权重：

```python
# 向量优先
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    vector_weight=2.0,    # 向量搜索权重
    keyword_weight=1.0,   # 关键词搜索权重
)

# 关键词优先
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    vector_weight=1.0,
    keyword_weight=2.0,
)
```

### 4. 全文搜索语义

选择 AND 或 OR 语义：

```python
# OR 语义（默认）- 匹配任一关键词，提高召回率
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    use_or_semantic=True,  # 默认
)

# AND 语义 - 匹配所有关键词，提高精确率
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    use_or_semantic=False,
)
```

### 5. 查询缓存

启用查询缓存大幅提升重复查询性能：

```python
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    enable_cache=True,     # 启用缓存
    cache_size=1000,       # 缓存大小
)

# 第一次查询（冷缓存）
results = retriever.retrieve("培训时长")  # ~1.2s

# 第二次查询（热缓存）
results = retriever.retrieve("培训时长")  # ~0.09ms，提升 15000x！

# 查看缓存统计
print(retriever.get_cache_stats())
# {'enabled': True, 'size': 1, 'capacity': 1000}

# 清除缓存
retriever.clear_cache()
```

## 异步使用

### 基础异步检索

```python
import asyncio

async def search():
    retriever = HybridRetriever(
        version_id="xxx",
        settings=settings,
    )
    
    results = await retriever.retrieve_async("培训时长")
    
    await retriever.close_async()
    return results

results = asyncio.run(search())
```

### 并发查询

```python
async def batch_search(queries):
    retriever = HybridRetriever(
        version_id="xxx",
        settings=settings,
    )
    
    # 并发执行多个查询
    results = await asyncio.gather(*[
        retriever.retrieve_async(q)
        for q in queries
    ])
    
    await retriever.close_async()
    return results

# 5 个查询并发执行，速度提升 5x
queries = ["查询1", "查询2", "查询3", "查询4", "查询5"]
results = asyncio.run(batch_search(queries))
```

## 上下文管理器

使用上下文管理器自动管理资源：

```python
# 同步版本
with HybridRetriever(version_id="xxx", settings=settings) as retriever:
    results = retriever.retrieve("培训时长")
# 自动关闭连接池

# 异步版本
async with HybridRetriever(version_id="xxx", settings=settings) as retriever:
    results = await retriever.retrieve_async("培训时长")
# 自动关闭连接池
```

## 完整配置示例

```python
retriever = HybridRetriever(
    version_id="your-version-id",
    settings=settings,
    top_k=10,                    # 返回结果数
    rrf_k=60,                    # RRF 阻尼系数
    hnsw_ef_search=100,          # HNSW 搜索参数
    vector_weight=1.0,           # 向量搜索权重
    keyword_weight=1.5,          # 关键词搜索权重
    use_connection_pool=True,    # 连接池
    pool_min_size=2,
    pool_max_size=10,
    enable_cache=True,           # 查询缓存
    cache_size=1000,
    use_or_semantic=True,        # OR 语义
)
```

## 性能优化建议

### 生产环境推荐配置

```python
retriever = HybridRetriever(
    version_id="xxx",
    settings=settings,
    top_k=10,
    hnsw_ef_search=100,        # 平衡召回和性能
    vector_weight=1.0,
    keyword_weight=1.5,        # 稍微偏重关键词
    use_connection_pool=True,  # 必须启用
    enable_cache=True,         # 生产环境必须启用
    cache_size=1000,
    use_or_semantic=True,      # 提高召回率
)
```

### 性能基准

| 优化项 | 效果 | 建议 |
|--------|------|------|
| 查询缓存 | 15,000x 加速 | 生产环境必须启用 |
| 异步并发 | 5x 吞吐量 | 批量查询时使用 |
| 连接池 | 稳定性提升 | 必须启用 |
| HNSW 优化 | +5% 召回率 | 推荐 ef_search=100 |
| OR 语义 | 提高召回率 | 根据业务选择 |

## 故障排查

### 全文搜索无结果

检查是否使用正确的语义：

```python
# OR 语义更宽松，召回率更高
retriever = HybridRetriever(..., use_or_semantic=True)

# AND 语义更严格，精确率更高
retriever = HybridRetriever(..., use_or_semantic=False)
```

### 缓存未生效

确保启用缓存并使用相同参数：

```python
retriever = HybridRetriever(..., enable_cache=True)

# 缓存 key 基于 version_id + query + keywords + top_k
# 任一参数不同都会生成不同的 key
```

### 向量搜索召回率低

增加 `hnsw_ef_search` 参数：

```python
retriever = HybridRetriever(..., hnsw_ef_search=200)  # 提高召回率
```

## API 参考

### HybridRetriever

主要检索类，支持同步和异步操作。

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| version_id | str | 必填 | 文档版本 ID |
| settings | dict | 必填 | 配置字典，包含 DATABASE_URL |
| top_k | int | 10 | 返回结果数 |
| rrf_k | int | 60 | RRF 阻尼系数 |
| hnsw_ef_search | int | 100 | HNSW 搜索参数 |
| vector_weight | float | 1.0 | 向量搜索权重 |
| keyword_weight | float | 1.0 | 关键词搜索权重 |
| use_connection_pool | bool | True | 是否使用连接池 |
| pool_min_size | int | 2 | 连接池最小连接数 |
| pool_max_size | int | 10 | 连接池最大连接数 |
| enable_cache | bool | False | 是否启用查询缓存 |
| cache_size | int | 1000 | 缓存大小 |
| use_or_semantic | bool | True | 全文搜索使用 OR 语义 |

#### 方法

- `retrieve(query, keywords=None, use_cache=True)` - 同步检索
- `retrieve_async(query, keywords=None, use_cache=True)` - 异步检索
- `clear_cache()` - 清除查询缓存
- `get_cache_stats()` - 获取缓存统计信息
- `close()` - 关闭资源（同步）
- `close_async()` - 关闭资源（异步）

### RetrievalResult

检索结果数据类。

| 属性 | 类型 | 说明 |
|------|------|------|
| chunk_id | str | 块 ID |
| text | str | 文本内容 |
| page_idx | int | 页码 |
| score | float | RRF 融合分数 |
| source | str | 来源 (vector/keyword/hybrid) |
| vector_score | float | 原始向量相似度 |
| keyword_score | float | 原始关键词分数 |
