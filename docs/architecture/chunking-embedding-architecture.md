# Chunking & Embedding 架构设计文档

> **版本**: 1.0  
> **日期**: 2026-02-04  
> **状态**: 设计阶段  
> **作者**: Kimi Code + User  

---

## 1. 问题背景

### 1.1 原始问题

MineRU 解析 PDF 后产生的 `content_list.json` 存在严重的**碎片化问题**：

```json
// 示例：一段完整的话被拆分成多个短片段
[
  {"type": "text", "text": "细胞和组织本身会发出荧光", "page_idx": 91},
  {"type": "text", "text": "这种内源性荧光", "page_idx": 91},
  {"type": "text", "text": "通常被视为共聚焦显微术中需要克服的问题", "page_idx": 91}
]
```

**统计数据**（来自 `83420a7c-b27b-480f-9427-565c47d2b53c_content_list.json`）：
- 总条目数: 1,237
- 文本条目: 844
- 平均文本长度: **45 字符**
- 最短: 1 字符，最长: 816 字符

### 1.2 问题影响

| 问题 | 后果 |
|------|------|
| 语义不完整 | 单个 chunk 无法表达完整含义，向量相似度计算失效 |
| 上下文丢失 | "这种内源性荧光" - 不知道"这种"指代什么 |
| 检索效果差 | 查询"自发荧光问题"无法匹配到上面的短片段 |
| LLM 理解困难 | 即使检索到，也缺乏足够上下文生成准确回答 |

### 1.3 初步尝试与局限

**尝试 1: 简单合并**（Rejected）
- 将连续的短 text 合并为 200-500 字符的 chunk
- 问题：粗暴合并可能跨段落，破坏语义边界

**尝试 2: Parent-Child Chunking**（Base Design）
- Parent Chunk: 存储完整段落/句子组
- Child Chunk: 存储原始短片段，指向 Parent
- 检索 Child → 返回 Parent
- 问题：Child 本身还是短片段，匹配精度有限

---

## 2. 深度调研与方案演进

### 2.1 调研的关键技术

通过 Context7、网络搜索和学术论文，调研了 5 大前沿技术：

#### 2.1.1 Anthropic Contextual Retrieval (2024-09)

**核心洞察**: 为每个 chunk 添加全局上下文前缀再向量化

```
原始: "细胞和组织本身会发出荧光"
增强: "[技术规格-共聚焦显微镜-光学原理] 细胞和组织本身会发出荧光"
```

**效果**: 减少 49-67% 的检索失败

**实现成本**: 需要 LLM 生成上下文描述

#### 2.1.2 HiChunk - Tencent (2025-09)

**核心洞察**: 层次化 chunking + Auto-Merge 检索

```
Level 3: Document
Level 2: Section (章节)
Level 1: Paragraph (段落)
Level 0: Sentence (句子)
```

**创新点**: 
- HiCBench 评测基准
- Auto-Merge: 如果多个子 chunk 来自同一段落，直接返回段落

#### 2.1.3 RAPTOR - Stanford (2024-01)

**核心洞察**: 递归聚类 + 摘要，构建树形检索结构

```
Level 2: [摘要摘要] ← 从多个 Level 1 摘要生成
Level 1: [段落摘要] ← 从 Level 0 聚类生成
Level 0: [原始文本] ← 叶子节点
```

**效果**: 在 Q&A 任务上 SOTA

#### 2.1.4 Late Chunking - Jina AI (2024-08)

**核心洞察**: 先嵌入完整上下文，再分块池化

```
传统: 每个 chunk 独立编码 → 丢失跨 chunk 上下文
Late: 整段编码 → 按边界池化 → 保持上下文感知
```

**优势**: 使用长上下文模型（如 jina-embeddings-v2, 8k）

#### 2.1.5 LangChain Multi-Vector Retrieval

**核心洞察**: 一个 Parent 文档对应多个 Child 向量

```
Parent Doc → 存储完整文本
    ↓
Child 1 → 向量 1 (用于检索)
Child 2 → 向量 2 (用于检索)
```

**检索流程**:
1. 查询向量匹配 Child
2. 获取 Parent 文档
3. 返回完整上下文

### 2.2 方案融合: CPC (Contextual Parent-Child Chunking)

**设计理念**: 不选择单一技术，而是融合各技术优点

```
┌─────────────────────────────────────────────────────────────────┐
│                    CPC 融合架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 3: Root Node (文档级摘要) ← RAPTOR 思想                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  "上海妙生科贸有限公司投标文件-六院共聚焦设备采购..."      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  Level 2: Section Nodes (章节) ← HiChunk 层次化                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ 技术规格      │ │  商务条款     │ │  资质文件     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                              ↓                                   │
│  Level 1: Parent Chunks (上下文增强) ← Anthropic Contextual      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Context: "【技术规格-共聚焦显微镜】"                     │   │
│  │  Content: "细胞和组织本身会发出荧光。这种内源性荧光..."   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  Level 0: Child Chunks (检索单元) ← Late Chunking + Multi-Vector │
│  ┌──────────────────┐ ┌──────────────────┐                     │
│  │ "细胞和组织..."   │ │ "这种内源性..."   │                     │
│  │ (长上下文感知)    │ │ (长上下文感知)    │                     │
│  └──────────────────┘ └──────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据库架构设计

### 3.1 表结构演进

#### V1: 基础表 (Migration 001)

```sql
-- 原始 MineRU 直接入库
CREATE TABLE chunks (
    chunk_id uuid primary key,
    version_id uuid references document_versions(version_id),
    text_raw text,
    text_tsv tsvector,
    embedding vector(1536),
    element_type text,  -- 'text', 'table', 'image'
    page_idx int,
    bbox jsonb
);
```

**问题**: 只有原始片段，无上下文

#### V2: MineRU 字段扩展 (Migration 002)

```sql
-- 支持所有 MineRU 类型
ALTER TABLE chunks ADD COLUMN img_path TEXT;
ALTER TABLE chunks ADD COLUMN image_caption TEXT[];
ALTER TABLE chunks ADD COLUMN table_body TEXT;
ALTER TABLE chunks ADD COLUMN list_items TEXT[];
-- ... 等
```

**改进**: 完整保留 MineRU 解析信息

#### V3: pgvector 优化 (Migration 003)

```sql
-- 优化索引和查询函数
CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
USING hnsw(embedding vector_cosine_ops);

-- 检索函数
CREATE OR REPLACE FUNCTION search_chunks_hybrid(...);
```

**改进**: 添加 HNSW 索引和混合搜索

#### V4: Contextual Parent-Child (Migration 005-007)

```sql
-- 上下文增强表 (Anthropic 风格)
CREATE TABLE contextual_chunks (
    contextual_id uuid PRIMARY KEY,
    chunk_id uuid REFERENCES chunks(chunk_id),
    original_text text,
    context_prefix text,  -- 生成的上下文
    contextualized_text text,  -- 完整文本
    embedding vector(1536)
);

-- 层次化节点表 (HiChunk + RAPTOR)
CREATE TABLE hierarchical_nodes (
    node_id uuid PRIMARY KEY,
    level integer,  -- 0=leaf, 1=para, 2=section, 3=root
    parent_id uuid REFERENCES hierarchical_nodes(node_id),
    text_content text,
    summary text,  -- RAPTOR 摘要
    children_ids uuid[],  -- PostgreSQL 数组
    root_path uuid[],  -- 从根到当前路径
    embedding vector(1536)
);

-- 多向量关联表 (LangChain 风格)
CREATE TABLE multi_vector_mappings (
    parent_id uuid,
    parent_text text,
    child_id uuid REFERENCES chunks(chunk_id),
    child_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7
);

-- Late Chunking 专用表
CREATE TABLE late_chunking_embeddings (
    parent_text text,
    parent_embedding vector(1536),
    child_id uuid,
    child_start_idx integer,  -- token 位置
    child_end_idx integer,
    child_embedding vector(1536)  -- 在父上下文中池化
);
```

### 3.2 表关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据库表关系                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  projects                                                       │
│    ↓                                                            │
│  documents                                                      │
│    ↓                                                            │
│  document_versions                                              │
│    ↓                                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    chunks (原始 MineRU)                   │  │
│  │  - 保留所有原始解析结果                                   │  │
│  │  - text_raw, table_body, img_path 等                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│    ↓                                                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐│
│  │ contextual_      │  │ hierarchical_    │  │ multi_vector_  ││
│  │ chunks           │  │ nodes            │  │ mappings       ││
│  │                  │  │                  │  │                ││
│  │ Anthropic 风格   │  │ HiChunk+RAPTOR   │  │ LangChain 风格 ││
│  │ 上下文增强       │  │ 层次化树形       │  │ Parent-Child   ││
│  └──────────────────┘  └──────────────────┘  └────────────────┘│
│    ↓                       ↓                      ↓             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              late_chunking_embeddings                     │  │
│  │                 (Jina Late Chunking)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Chunking 策略详解

### 4.1 传统方式的问题

```python
# 传统方式: 直接存储 MineRU 输出
for item in content_list:
    if item['type'] == 'text':
        insert_chunk(item['text'])  # 太短！
```

### 4.2 CPC 分层处理

```python
class CPCPipeline:
    def process_document(self, content_list, document_title):
        # Step 1: Contextual Enhancement
        contextual_chunks = self._add_context(
            content_list, 
            document_title
        )
        
        # Step 2: Hierarchical Grouping
        hierarchy = self._build_hierarchy(contextual_chunks)
        
        # Step 3: RAPTOR Tree
        raptor_tree = self._build_raptor_tree(hierarchy)
        
        # Step 4: Late Chunking Embeddings
        embeddings = self._late_chunking(raptor_tree)
        
        # Step 5: Multi-Vector Indexing
        self._index_multi_vector(embeddings)
```

### 4.3 各层详细设计

#### Layer 0: Child Chunks (检索单元)

```python
@dataclass
class ChildChunk:
    """最小检索单元"""
    child_id: uuid.UUID
    text: str  # 原始短文本
    
    # Late Chunking: 在父上下文中编码
    embedding: List[float]  # 长上下文感知向量
    
    # 位置信息
    parent_id: uuid.UUID
    position_in_parent: int  # 第几个子 chunk
    token_start: int  # 在父文本中的 token 位置
    token_end: int
```

**关键**: 使用 Late Chunking，即使短文本也有长上下文感知

#### Layer 1: Parent Chunks (上下文增强)

```python
@dataclass
class ParentChunk:
    """上下文增强的父块"""
    parent_id: uuid.UUID
    
    # Anthropic Contextual Retrieval
    context_prefix: str  # "[技术规格-共聚焦显微镜]"
    original_text: str   # 合并后的原始文本
    contextualized_text: str  # context_prefix + original_text
    
    # HiChunk 层次
    level: int = 1  # Paragraph level
    page_idx: int
    bbox: Optional[dict]
    
    # 子节点
    children: List[ChildChunk]
    
    # 可选：Parent 也向量化
    embedding: Optional[List[float]]
```

#### Layer 2: Section Nodes (章节)

```python
@dataclass
class SectionNode:
    """章节节点 (HiChunk Level 2)"""
    node_id: uuid.UUID
    level: int = 2
    
    # 章节信息
    section_title: Optional[str]  # 自动提取或 LLM 生成
    text_content: str  # 章节完整内容
    summary: str  # RAPTOR: LLM 生成的摘要
    
    # 层次关系
    children: List[ParentChunk]
    parent_id: Optional[uuid.UUID]  # 指向 Root
```

#### Layer 3: Root Node (文档)

```python
@dataclass
class RootNode:
    """文档根节点"""
    node_id: uuid.UUID
    level: int = 3
    
    # 文档摘要 (RAPTOR)
    document_title: str
    text_content: str  # 全文或长摘要
    summary: str  # LLM 生成的文档摘要
    
    # 所有 Section
    children: List[SectionNode]
```

---

## 5. Embedding 策略详解

### 5.1 传统方式 vs CPC

| 方式 | 过程 | 问题 |
|------|------|------|
| 传统 | 短文本 → 独立编码 | 丢失上下文，语义不完整 |
| CPC | 长文本 → 整体编码 → 分块池化 | 保持上下文感知 |

### 5.2 Late Chunking 实现

```python
class LateChunkingEncoder:
    """
    Late Chunking: 长上下文感知向量生成
    
    Reference: https://github.com/jina-ai/late-chunking
    """
    
    def encode(
        self,
        parent_text: str,      # 完整父文本
        child_boundaries: List[Tuple[int, int]]  # 子 chunk 边界
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Returns:
            parent_embedding: 父文本向量
            child_embeddings: 每个子 chunk 的向量（上下文感知）
        """
        # 1. Tokenize 完整文本
        tokens = self.tokenizer(parent_text, max_length=8192)
        
        # 2. 通过模型获取 token embeddings
        # 关键：所有 token 一起编码，保持上下文交互
        token_embeddings = self.model(tokens)
        
        # 3. 按边界池化
        child_embs = []
        for start, end in child_boundaries:
            chunk_embs = token_embeddings[start:end]
            pooled = mean_pooling(chunk_embs)
            child_embs.append(pooled)
        
        return token_embeddings[0], child_embs  # [CLS] 作为 parent
```

### 5.3 Contextual Embedding

```python
class ContextualEmbedding:
    """
    Anthropic Contextual Retrieval 实现
    """
    
    CONTEXT_PROMPT = """Given the following text chunk from a larger document, 
generate a concise context (1-2 sentences) that explains:
1. What document/section this chunk is from
2. What topic it relates to

Text chunk: {chunk_text}

Surrounding context: {surrounding}

Context prefix:"""
    
    def generate_context_prefix(
        self,
        chunk_text: str,
        document_title: str,
        surrounding_chunks: List[str]
    ) -> str:
        """生成上下文前缀"""
        
        # 基于规则的快速上下文
        rule_based = f"[{document_title}]"
        
        # LLM 生成的补充上下文
        surrounding = " ".join(surrounding_chunks[:3])
        prompt = self.CONTEXT_PROMPT.format(
            chunk_text=chunk_text[:500],
            surrounding=surrounding[:1000]
        )
        
        llm_context = self.llm.complete(prompt).strip()
        
        return f"{rule_based} {llm_context}"
    
    def embed_with_context(self, chunk: ChildChunk) -> List[float]:
        """添加上下文后编码"""
        contextualized = f"{chunk.context_prefix} {chunk.text}"
        return self.embedding_model.encode(contextualized)
```

### 5.4 批量处理策略

```python
class CPCEmbeddingPipeline:
    """CPC 向量化管道"""
    
    def __init__(self):
        self.late_chunking = LateChunkingEncoder()
        self.contextual = ContextualEmbedding()
    
    def process_parents(self, parents: List[ParentChunk]):
        """批量处理 Parent Chunks"""
        
        for parent in parents:
            # 1. 准备子 chunk 边界
            boundaries = []
            current_pos = 0
            for child in parent.children:
                tokens = len(child.text) // 2  # 粗略估算
                boundaries.append((current_pos, current_pos + tokens))
                current_pos += tokens
            
            # 2. Late Chunking 编码
            parent_emb, child_embs = self.late_chunking.encode(
                parent.contextualized_text,
                boundaries
            )
            
            # 3. 保存向量
            parent.embedding = parent_emb
            for child, emb in zip(parent.children, child_embs):
                child.embedding = emb
```

---

## 6. 检索策略

### 6.1 Multi-Stage Retrieval

```python
class CPCRetriever:
    """CPC 多阶段检索器"""
    
    async def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        检索流程：
        1. Query Understanding
        2. Child-Level Retrieval (精准匹配)
        3. Parent Retrieval (获取上下文)
        4. Reranking (排序优化)
        5. Context Assembly (组装最终上下文)
        """
        
        # Stage 1: 查询理解
        query_emb = await self.embed_query(query)
        
        # Stage 2: Child 检索
        child_results = await self._search_children(
            query_emb, 
            top_k=top_k * 3  # 多取用于重排
        )
        
        # Stage 3: 获取 Parents
        parent_map = {}
        for child in child_results:
            parent = await self._get_parent(child.parent_id)
            if parent.id not in parent_map:
                parent_map[parent.id] = {
                    'parent': parent,
                    'matched_children': [],
                    'max_score': 0
                }
            parent_map[parent.id]['matched_children'].append(child)
            parent_map[parent.id]['max_score'] = max(
                parent_map[parent.id]['max_score'],
                child.score
            )
        
        # Stage 4: Reranking
        reranked = await self._rerank(query, list(parent_map.values()))
        
        # Stage 5: 组装上下文
        final_results = []
        for item in reranked[:top_k]:
            context = self._assemble_context(item)
            final_results.append(context)
        
        return final_results
```

### 6.2 检索模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `child-only` | 只检索 Child，返回 Parent | 快速检索 |
| `hierarchical` | 可上升到 Section/Root | 复杂问答 |
| `hybrid` | BM25 + Vector + Reranking | 生产环境 |

---

## 7. 性能优化

### 7.1 Token 估算与批处理

```python
def estimate_tokens(text: str) -> int:
    """保守估算：中文约 2 字符/token"""
    return len(text) // 2 + 1

# 批处理策略
BATCH_SIZE = 50       # 每批数量
MAX_TOKENS = 100000   # 每批最大 token
```

### 7.2 HNSW 索引调优

```sql
-- 构建参数
CREATE INDEX ON hierarchical_nodes 
USING hnsw(embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 查询参数
SET hnsw.ef_search = 100;  -- 数据量 < 10万
```

### 7.3 缓存策略

```python
# Parent Chunk 缓存（不常变）
@lru_cache(maxsize=1000)
def get_parent_chunk(parent_id: str) -> ParentChunk:
    ...

# Embedding 缓存（基于文本哈希）
@lru_cache(maxsize=5000)
def get_cached_embedding(text_hash: str) -> List[float]:
    ...
```

---

## 8. 回退策略

### 8.1 分级回退

```
Level 1: CPC Full (Contextual + HiChunk + RAPTOR + Late Chunking)
    ↓ 失败
Level 2: CPC Lite (Contextual + Parent-Child)
    ↓ 失败
Level 3: Simple Merge (合并短片段)
    ↓ 失败
Level 4: Original Chunks (原始 MineRU)
```

### 8.2 降级触发条件

| 条件 | 降级到 | 操作 |
|------|--------|------|
| LLM API 不可用 | Level 2 | 使用规则生成上下文 |
| 内存不足 | Level 3 | 跳过层次化构建 |
| 时间限制 | Level 4 | 使用原始 chunks |

---

## 9. 预期效果

### 9.1 指标对比

| 指标 | 当前 | CPC 目标 | 提升 |
|------|------|----------|------|
| 平均 chunk 长度 | 45 字符 | 300-500 字符 | 6-10x |
| 检索召回率 | ~35% | >80% | 2.3x |
| 上下文完整性 | 低 | 高 | - |
| 多跳推理 | 不支持 | 支持 | - |

### 9.2 检索示例

```
Query: "共聚焦显微镜的自发荧光问题怎么解决？"

传统检索:
- 匹配: "这种内源性荧光" (score: 0.6)
- 返回: 短片段，无法理解

CPC 检索:
- 匹配 Child: "这种内源性荧光" (score: 0.85)
- 获取 Parent: "【技术规格-共聚焦显微镜-光学原理】细胞和组织本身会发出荧光。
  这种内源性荧光（自发荧光）通常被视为共聚焦显微术中需要克服的问题..."
- 返回: 完整段落，包含问题和解决方案
```

---

## 10. 实施路线图

详见: `docs/plans/2026-02-04-cpc-implementation-plan.md`

**Phase 1**: Contextual Enhancement (Week 1)  
**Phase 2**: HiChunk Hierarchical (Week 2)  
**Phase 3**: RAPTOR Tree (Week 3)  
**Phase 4**: Late Chunking (Week 3)  
**Phase 5**: Multi-Vector Retrieval (Week 4)  
**Phase 6**: Integration & Testing (Week 4-5)

---

## 11. 参考资料

### 论文
1. **Anthropic Contextual Retrieval** (2024-09)  
   https://www.anthropic.com/news/contextual-retrieval

2. **HiChunk: Hierarchical Chunking** (2025-09, Tencent)  
   arxiv.org/abs/2509.11552

3. **RAPTOR: Recursive Abstractive Processing** (2024-01, Stanford)  
   arxiv.org/abs/2401.18059

4. **Late Chunking** (2024-08, Jina AI)  
   https://github.com/jina-ai/late-chunking

### 开源实现
1. **LangChain ParentDocumentRetriever**  
   https://python.langchain.com/docs/how_to/parent_document_retriever

2. **RAPTOR Official Implementation**  
   https://github.com/parthsarthi03/raptor

3. **HiRAG (Hierarchical RAG)**  
   https://github.com/hhy-huang/hirag

---

## 12. 设计决策记录

### ADR 1: 为什么不直接用 LangChain ParentDocumentRetriever?

**决策**: 自己实现 CPC，而不是直接用 LangChain

**理由**:
1. LangChain 深度耦合其生态，难以定制
2. MineRU 输出有特殊结构（page_idx, bbox）
3. 需要融合多种技术（不仅是 Parent-Child）
4. 需要与现有 PostgreSQL + pgvector 架构集成

### ADR 2: 为什么使用多张表而不是单一表?

**决策**: contextual_chunks, hierarchical_nodes, multi_vector_mappings 分开存储

**理由**:
1. 关注点分离，便于维护
2. 支持不同阶段独立回退
3. PostgreSQL 支持 JOIN，查询性能可接受
4. 便于 A/B 测试不同策略

### ADR 3: 为什么保留原始 chunks 表?

**决策**: 新增表而不是替换 chunks 表

**理由**:
1. 向后兼容，可随时回退
2. 原始 MineRU 数据可用于调试
3. 支持对比实验（原始 vs CPC）

---

**文档结束**

*最后更新: 2026-02-04*  
*状态: 设计完成，待实施*
