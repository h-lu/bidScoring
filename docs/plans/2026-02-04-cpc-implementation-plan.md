# Contextual Parent-Child Chunking (CPC) 实施计划

> **Goal**: 实现融合 Anthropic Contextual Retrieval + HiChunk + RAPTOR + Late Chunking + Multi-Vector 的完整 CPC 管道，解决 MineRU 短片段检索效果差的问题。

---

## 概述

### 问题陈述
当前 MineRU 解析产生的 content_list 存在大量短文本片段（平均 45 字符），导致：
- 语义不完整，向量检索效果差
- 上下文丢失，无法支持复杂问答
- 检索召回率低（约 30-40%）

### 目标指标
| 指标 | 当前 | 目标 |
|------|------|------|
| 平均 chunk 长度 | 45 字符 | 200-500 字符 |
| 检索召回率 | ~35% | >80% |
| 上下文完整性 | 低 | 高 |
| 多跳推理能力 | 无 | 支持 |

### 技术架构
```
MineRU Content List
       ↓
┌─────────────────────────────────────────────┐
│  Phase 1: Contextual Enhancement            │
│  - Anthropic Contextual Retrieval           │
│  - 为每个 chunk 生成上下文前缀               │
└─────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────┐
│  Phase 2: Hierarchical Chunking (HiChunk)   │
│  - 构建 Sentence → Paragraph → Section 树   │
│  - Auto-Merge 逻辑                           │
└─────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────┐
│  Phase 3: RAPTOR Tree Construction          │
│  - 递归聚类                                  │
│  - 层次化摘要                                │
└─────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────┐
│  Phase 4: Late Chunking Embeddings          │
│  - 长上下文感知向量                          │
│  - Chunked Pooling                          │
└─────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────┐
│  Phase 5: Multi-Vector Retrieval System     │
│  - Parent-Child 关联                         │
│  - 混合检索 (BM25 + Vector)                  │
│  - Reranking                                │
└─────────────────────────────────────────────┘
```

---

## 任务清单

### Phase 1: Contextual Enhancement (Anthropic 风格)

#### Task 1.1: 创建 contextual_chunks 表
**描述**: 创建数据库表存储上下文增强后的 chunks

**文件变更**:
- `migrations/005_cpc_contextual_chunks.sql` (新建)

**自治检查**:
- **输入**: 需要支持 uuid, vector, text 类型的 PostgreSQL
- **输出**: `contextual_chunks` 表可用
- **依赖**: Migration 004 必须已应用
- **失败回退**: `DROP TABLE IF EXISTS contextual_chunks`

**验收标准**:
```sql
-- 验证表结构
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'contextual_chunks';
-- 期望: contextual_id, chunk_id, version_id, original_text, 
--       context_prefix, contextualized_text, embedding, ...
```

**提交信息**: `feat(db): add contextual_chunks table for Anthropic-style context enhancement`

---

#### Task 1.2: 实现上下文生成器
**描述**: 使用 LLM 为每个 chunk 生成上下文前缀

**文件变更**:
- `bid_scoring/contextual_retrieval.py` (新建)

**接口设计**:
```python
class ContextualRetrievalGenerator:
    def generate_context(
        self,
        chunk_text: str,
        document_title: str,
        section_title: Optional[str] = None,
        surrounding_chunks: List[str] = None
    ) -> str:
        """生成上下文前缀"""
        ...
```

**自治检查**:
- **输入**: OPENAI_API_KEY 必须可用
- **输出**: `ContextualRetrievalGenerator` 类可用
- **依赖**: Task 1.1
- **失败回退**: 使用基于规则的上下文生成（document_title + section）

**测试要求**:
```python
def test_contextual_generation():
    generator = ContextualRetrievalGenerator(get_llm_client())
    context = generator.generate_context(
        chunk_text="细胞和组织本身会发出荧光",
        document_title="共聚焦显微镜投标文件",
        section_title="技术规格"
    )
    assert "技术规格" in context
    assert "共聚焦" in context
```

**提交信息**: `feat(contextual): implement Anthropic-style context generation`

---

#### Task 1.3: 批量处理脚本
**描述**: 为现有 chunks 生成上下文并存储

**文件变更**:
- `scripts/build_contextual_chunks.py` (新建)

**自治检查**:
- **输入**: chunks 表必须有数据
- **输出**: contextual_chunks 表填充完成
- **依赖**: Task 1.1, Task 1.2
- **失败回退**: 清空 contextual_chunks 表重新生成

**验收标准**:
```python
# 验证上下文生成
SELECT COUNT(*) FROM contextual_chunks 
WHERE context_prefix IS NOT NULL;
-- 应该 > 0
```

**提交信息**: `feat(contextual): add batch processing script for context generation`

---

### Phase 2: Hierarchical Chunking (HiChunk)

#### Task 2.1: 创建层次化节点表
**描述**: 创建 hierarchical_nodes 表存储树形结构

**文件变更**:
- `migrations/006_cpc_hierarchical_nodes.sql` (新建)

**自治检查**:
- **输入**: PostgreSQL 支持 uuid[] 数组类型
- **输出**: hierarchical_nodes 表可用
- **依赖**: Task 1.3 完成
- **失败回退**: `DROP TABLE IF EXISTS hierarchical_nodes`

**提交信息**: `feat(db): add hierarchical_nodes table for HiChunk`

---

#### Task 2.2: 实现 HiChunk 构建器
**描述**: 实现层次化 chunking 逻辑

**文件变更**:
- `bid_scoring/hichunk.py` (新建)

**核心算法**:
```python
class HiChunkBuilder:
    def build_hierarchy(
        self,
        content_list: List[dict],
        document_title: str
    ) -> List[HiChunkNode]:
        # 1. 构建 Leaf Nodes (Sentences)
        # 2. 合并为 Paragraph Nodes
        # 3. 合并为 Section Nodes
        # 4. 创建 Root Node
```

**自治检查**:
- **输入**: content_list 格式必须符合 MineRU 标准
- **输出**: 返回有效的层次节点列表
- **依赖**: Task 2.1
- **失败回退**: 回退到原始 chunks 表

**测试要求**:
```python
def test_hichunk_building():
    builder = HiChunkBuilder()
    nodes = builder.build_hierarchy(sample_content_list, "Test Doc")
    
    # 验证层次结构
    leaf_nodes = [n for n in nodes if n.level == 0]
    para_nodes = [n for n in nodes if n.level == 1]
    
    assert len(leaf_nodes) > 0
    assert len(para_nodes) > 0
    assert all(len(p.children_ids) > 0 for p in para_nodes)
```

**提交信息**: `feat(hichunk): implement hierarchical chunking builder`

---

#### Task 2.3: 入库脚本
**描述**: 将层次化节点入库

**文件变更**:
- `scripts/build_hichunk_nodes.py` (新建)

**提交信息**: `feat(hichunk): add ingestion script for hierarchical nodes`

---

### Phase 3: RAPTOR 树构建

#### Task 3.1: 实现 RAPTOR 构建器
**描述**: 递归聚类和摘要生成

**文件变更**:
- `bid_scoring/raptor.py` (新建)

**依赖库**:
```txt
scikit-learn>=1.3.0  # 用于 KMeans 聚类
```

**核心算法**:
```python
class RAPTORBuilder:
    def build_tree(self, chunks: List[str]) -> List[RAPTORNode]:
        # Level 0: 叶子节点
        # Level 1+: 递归聚类 + 摘要
```

**自治检查**:
- **输入**: chunks 数量必须 >= 2
- **输出**: 返回 RAPTOR 树节点
- **依赖**: Task 2.3
- **失败回退**: 跳过 RAPTOR，使用 HiChunk 直接检索

**提交信息**: `feat(raptor): implement recursive abstractive tree building`

---

### Phase 4: Late Chunking

#### Task 4.1: 实现 Late Chunking 编码器
**描述**: 长上下文感知向量生成

**文件变更**:
- `bid_scoring/late_chunking.py` (新建)

**核心算法**:
```python
class LateChunkingEncoder:
    def encode_with_late_chunking(
        self,
        full_text: str,
        chunk_boundaries: List[Tuple[int, int]]
    ) -> List[List[float]]:
        # 1. Tokenize full text
        # 2. Get token embeddings
        # 3. Pool per chunk boundaries
```

**自治检查**:
- **输入**: 需要长上下文 embedding 模型（如 jina-embeddings-v2）
- **输出**: 返回每个 chunk 的向量
- **依赖**: Task 3.1
- **失败回退**: 使用标准 embedding

**提交信息**: `feat(late-chunking): implement chunked pooling encoder`

---

### Phase 5: Multi-Vector 检索系统

#### Task 5.1: 创建多向量关联表
**描述**: 创建 multi_vector_mappings 表

**文件变更**:
- `migrations/007_cpc_multi_vector.sql` (新建)

**提交信息**: `feat(db): add multi_vector_mappings table`

---

#### Task 5.2: 实现 Multi-Vector Retriever
**描述**: 实现 Parent-Child 关联检索

**文件变更**:
- `bid_scoring/multi_vector_retrieval.py` (新建)

**接口设计**:
```python
class MultiVectorRetriever:
    async def retrieve(
        self,
        query: str,
        retrieval_mode: str = "hybrid",  # 'child', 'parent', 'hierarchical'
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Dict]:
        ...
```

**自治检查**:
- **输入**: 所有前期表必须已填充
- **输出**: 返回 Parent chunks 列表
- **依赖**: Task 5.1
- **失败回退**: 回退到原始 chunks 检索

**提交信息**: `feat(retrieval): implement multi-vector parent-child retriever`

---

#### Task 5.3: 集成 CPC 管道
**描述**: 整合所有组件为完整管道

**文件变更**:
- `bid_scoring/cpc_pipeline.py` (新建)

**提交信息**: `feat(cpc): integrate complete contextual parent-child pipeline`

---

### Phase 6: MCP Server 更新

#### Task 6.1: 更新 search_chunks 工具
**描述**: MCP Server 支持 CPC 检索

**文件变更**:
- `mcp_servers/bid_documents/server.py` (修改)

**新增参数**:
```python
@mcp.tool()
def search_chunks(
    query: str,
    document_id: str,
    retrieval_mode: str = "cpc",  # 'standard', 'cpc'
    top_k: int = 10
):
```

**提交信息**: `feat(mcp): integrate CPC retrieval into search_chunks tool`

---

## 依赖关系图

```
Phase 1 (Contextual)
├── Task 1.1 [DB Table]
├── Task 1.2 [Generator] ← depends on 1.1
└── Task 1.3 [Batch Script] ← depends on 1.2

Phase 2 (HiChunk)
├── Task 2.1 [DB Table] ← depends on 1.3
├── Task 2.2 [Builder] ← depends on 2.1
└── Task 2.3 [Ingestion] ← depends on 2.2

Phase 3 (RAPTOR)
└── Task 3.1 [Builder] ← depends on 2.3

Phase 4 (Late Chunking)
└── Task 4.1 [Encoder] ← depends on 3.1

Phase 5 (Multi-Vector)
├── Task 5.1 [DB Table] ← depends on 4.1
├── Task 5.2 [Retriever] ← depends on 5.1
└── Task 5.3 [Pipeline] ← depends on 5.2

Phase 6 (Integration)
└── Task 6.1 [MCP Update] ← depends on 5.3
```

---

## 执行顺序

1. **Week 1**: Phase 1 (Contextual Enhancement)
   - Task 1.1, 1.2, 1.3

2. **Week 2**: Phase 2 (HiChunk)
   - Task 2.1, 2.2, 2.3

3. **Week 3**: Phase 3-4 (RAPTOR + Late Chunking)
   - Task 3.1, 4.1

4. **Week 4**: Phase 5-6 (Multi-Vector + Integration)
   - Task 5.1, 5.2, 5.3, 6.1

---

## 验收测试

### 集成测试
```python
# tests/test_cpc_pipeline.py

async def test_cpc_end_to_end():
    pipeline = CPCPipeline()
    
    # 处理文档
    result = await pipeline.process_document(
        content_list=load_test_document(),
        document_title="测试文档"
    )
    
    assert result["contextual_chunks"] > 0
    assert result["hierarchical_nodes"] > 0
    assert result["raptor_nodes"] > 0
    
    # 检索测试
    results = await pipeline.retrieve(
        query="技术规格",
        retrieval_mode="hybrid"
    )
    
    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all(len(r["text"]) > 100 for r in results)  # 完整上下文
```

### 性能基准
```python
# 基准测试
python scripts/benchmark_cpc.py \
    --test-queries queries.json \
    --output results.json
```

期望指标:
- 召回率 > 80%
- 平均响应时间 < 500ms
- 上下文完整性评分 > 4.0/5.0

---

## 回退策略

如果任何 Phase 失败:

1. **Phase 1 失败**: 回退到无上下文的原始 chunks
2. **Phase 2 失败**: 跳过层次化，直接使用 Contextual + 原始 chunks
3. **Phase 3 失败**: 跳过 RAPTOR，使用 HiChunk + Late Chunking
4. **Phase 4 失败**: 使用标准 embedding
5. **Phase 5 失败**: 回退到原始 search.py 实现
6. **Phase 6 失败**: MCP Server 保持原有功能

---

## 文档要求

- [ ] 更新 README.md 添加 CPC 架构说明
- [ ] 创建 docs/cpc_architecture.md 详细文档
- [ ] 添加 API 文档 (docstrings)
- [ ] 创建使用示例 notebooks/

---

## For Claude (执行提示)

**REQUIRED SUB-SKILL**: Use superpowers:subagent-driven-development to implement this plan task-by-task.

每个任务必须遵循:
1. **TDD**: 先写测试，再实现
2. **最小变更**: 每个 task 独立提交
3. **验收检查**: 验证依赖是否满足
4. **失败回退**: 明确回退步骤

**重要**: Phase 之间有依赖关系，必须按顺序执行。
