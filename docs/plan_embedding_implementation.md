# Embedding 实现计划

## 背景知识

### Embedding 模型 vs LLM 的区别

| 特性 | Embedding 模型 | LLM |
|------|----------------|-----|
| **用途** | 文本 → 向量 | 理解/生成文本 |
| **输入限制** | 8191 tokens | 128K-200K tokens |
| **输出** | 1536/3072 维向量 | 文本 |
| **模型** | text-embedding-3 | GPT-4/Claude |

**RAG 流程:**
```
Document → Embedding 模型 → 向量 → pgvector 存储
                                      ↓
Query → Embedding 模型 → 向量 → 相似度搜索 → Top K 结果
                                      ↓
                          Results + Query → LLM → Answer
```

## 2026 年最佳实践

### 1. Token 精确计算
- 使用 `tiktoken` 库精确计算 tokens
- 避免超过 8191 tokens 限制
- 超长文本需要截断或分块

### 2. 批处理优化
- OpenAI 限制: 每批最多 2048 个输入
- 每分钟 1M tokens 限制
- 建议使用 50-100 条/批

### 3. 向量存储
- 使用 pgvector 扩展
- 选择合适维度: 1536 (small) 或 3072 (large)
- 添加 HNSW 索引优化搜索性能

### 4. 缓存策略
- 内存缓存: @lru_cache
- 持久缓存: 数据库缓存避免重复 API 调用
- 哈希 key: 使用文本内容哈希

### 5. 错误处理
- Token 超限处理
- API 重试机制
- 降级策略

## 实施计划

### Phase 1: Token 精确计算
1. 添加 tiktoken 依赖
2. 替换 estimate_tokens() 为精确计算
3. 添加超长文本截断功能
4. 测试各种文本类型的 token 数

### Phase 2: 数据库集成
1. 添加 pgvector 扩展
2. 在 hierarchical_nodes 添加 embedding 列
3. 创建相似度搜索函数
4. 批量生成并存储 embeddings

### Phase 3: 缓存优化
1. 设计缓存表结构
2. 实现数据库缓存逻辑
3. 添加缓存命中统计
4. 测试缓存效果

### Phase 4: 搜索功能
1. 实现向量相似度搜索
2. 添加混合搜索 (向量 + 关键词)
3. 实现重排序 (reranking)
4. 性能优化和索引调优

## 具体任务

### Task 1: 添加 tiktoken 精确计算
**优先级**: P0 (高)
**工作量**: 2-3 小时
**依赖**: tiktoken 库

```python
import tiktoken

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_to_max_tokens(text: str, max_tokens: int = 8191) -> str:
    """截断文本到最大 token 数"""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
```

### Task 2: 数据库 schema 更新
**优先级**: P0 (高)
**工作量**: 3-4 小时
**依赖**: pgvector 扩展

```sql
-- 添加 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 添加 embedding 列
ALTER TABLE hierarchical_nodes 
ADD COLUMN embedding vector(1536);

-- 创建 HNSW 索引
CREATE INDEX idx_hierarchical_nodes_embedding 
ON hierarchical_nodes 
USING hnsw (embedding vector_cosine_ops);
```

### Task 3: 批量生成 embeddings
**优先级**: P0 (高)
**工作量**: 4-5 小时
**依赖**: Task 1, Task 2

```python
def generate_embeddings_for_sections(version_id: str):
    """为所有 sections 生成 embeddings"""
    sections = fetch_sections(version_id)
    texts = [s.content for s in sections]
    embeddings = embed_texts(texts)
    store_embeddings(version_id, sections, embeddings)
```

### Task 4: 向量搜索实现
**优先级**: P1 (中)
**工作量**: 3-4 小时
**依赖**: Task 3

```python
def similarity_search(query: str, top_k: int = 5) -> list:
    query_embedding = embed_single_text(query)
    results = db.query("""
        SELECT heading, content, 
               1 - (embedding <=> %s::vector) as similarity
        FROM hierarchical_nodes
        WHERE node_type = 'section'
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, top_k))
    return results
```

### Task 5: 持久化缓存
**优先级**: P1 (中)
**工作量**: 2-3 小时

```python
@lru_cache(maxsize=10000)
def get_cached_embedding(text_hash: str) -> list[float] | None:
    """从数据库缓存获取 embedding"""
    return db.fetch_one(
        "SELECT embedding FROM embedding_cache WHERE text_hash = %s",
        (text_hash,)
    )
```

## 验收标准

1. ✅ Token 计算误差 < 5%
2. ✅ 支持 8191+ tokens 文本截断
3. ✅ 167 个 sections 全部生成 embeddings
4. ✅ 向量搜索 Top 5 准确率 > 80%
5. ✅ 缓存命中率 > 50%
6. ✅ 搜索延迟 < 100ms

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| API 调用费用高 | 成本 | 添加缓存，批量处理 |
| 处理时间长 | 体验 | 异步处理，进度显示 |
| 向量搜索慢 | 性能 | 添加 HNSW 索引 |
| 精度不够 | 质量 | 调整 top_k，添加重排序 |

