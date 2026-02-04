# RAG Chunking 策略分析与优化方案

## 🔍 问题分析

### 当前策略的问题

当前实现：**Section-level 合并**（所有 chunk 合并为一个）

```
Section A
├── Chunk 1 (100 chars) ──┐
├── Chunk 2 (150 chars) ──┤──> 合并为一个大文本 (8000+ chars)
├── Chunk 3 (200 chars) ──┤
└── Chunk 4 (120 chars) ──┘
```

**问题:**
1. ❌ 文本过长 → 超过 8191 token 限制
2. ❌ 搜索精度低 → 大 chunk 包含噪声信息
3. ❌ 召回率下降 → 相似度计算被无关内容稀释

### 为什么这是问题？

**Embedding 相似度原理:**
- 向量表示的是整段文本的平均语义
- chunk 越大，包含的主题越多，语义越模糊
- 用户查询通常只匹配其中一小部分内容

**例子:**
```
大 Chunk: "公司简介 + 财务数据 + 法律声明 + 联系方式"
查询: "公司营收是多少？"
问题: 财务数据只占 1/4，相似度被其他内容稀释
```

---

## ✅ 最佳实践：Small-to-Big / Parent-Child Retrieval

### 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│  SEARCH PHASE (高精度)                                       │
│  ├── 小 Chunks (256-512 tokens)                             │
│  ├── 更高的语义集中度                                        │
│  └── 更好的匹配精度                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  GENERATION PHASE (完整上下文)                                │
│  ├── 根据匹配的 chunk 找到父 Section                         │
│  ├── 将完整 Section 作为上下文提供给 LLM                     │
│  └── 保证答案的完整性和连贯性                                │
└─────────────────────────────────────────────────────────────┘
```

### 数据模型设计

```python
# 数据库 Schema
hierarchical_nodes:
  - id: UUID
  - parent_id: UUID          # 指向父节点
  - node_type: str           # 'document' | 'section' | 'chunk'
  - heading: str
  - content: str             # 用于 LLM 的完整内容
  - content_for_embedding: str  # 用于 embedding 的内容（小 chunk）
  - embedding: vector        # 小 chunk 的向量
  - metadata: JSON
  - order_index: int
```

### 两种实现策略

#### 策略 A: 双内容存储（推荐 ⭐）

```
Section A (node_type='section')
├── content: "完整 section 内容（用于 LLM）"
├── content_for_embedding: null  # Section 本身不用于 embedding
└── children:
    ├── Chunk 1 (node_type='chunk')
    │   ├── content: "完整内容"
    │   ├── content_for_embedding: "处理后的内容（合并短 chunk）"
    │   └── embedding: [vector]
    ├── Chunk 2 (node_type='chunk')
    └── Chunk 3 (node_type='chunk')
```

**搜索流程:**
1. 查询 → Embedding → 在 `chunk` 节点中搜索
2. 找到匹配的 chunk → 通过 `parent_id` 找到 Section
3. 将 Section 的完整 `content` 提供给 LLM

**优点:**
- ✅ 精确控制 embedding 粒度
- ✅ 支持短 chunk 智能合并
- ✅ Section 完整内容用于生成

#### 策略 B: 分层索引

```python
# 同时存储两种粒度
small_chunks = split_into_small_chunks(text, size=256)   # 用于搜索
large_chunks = split_into_large_chunks(text, size=2048)  # 用于生成

# 建立映射关系
small_chunk.parent_large_chunk = large_chunk.id
```

**优点:**
- ✅ 更灵活的多层检索
- ❌ 复杂度更高

---

## 🎯 推荐方案：策略 A + 智能 Chunk 合并

### 核心逻辑

```python
def process_section_chunks(section, min_chunk_size=200, max_chunk_size=800):
    """
    处理 section 下的 chunks:
    1. 合并过短的 chunks（< min_chunk_size）
    2. 截断过长的 chunks（> max_chunk_size）
    3. 保留 section 完整内容用于 LLM
    """
    raw_chunks = section.original_chunks
    
    # 智能合并
    merged_chunks = []
    current_chunk = ""
    
    for chunk in raw_chunks:
        if len(current_chunk) < min_chunk_size:
            current_chunk += chunk.content
        else:
            merged_chunks.append(current_chunk)
            current_chunk = chunk.content
    
    if current_chunk:
        merged_chunks.append(current_chunk)
    
    # 创建数据库记录
    section_record = create_section_node(
        content=merge_all_chunks(raw_chunks),  # 完整内容
        node_type='section'
    )
    
    for merged in merged_chunks:
        create_chunk_node(
            content=merged,
            content_for_embedding=merged[:max_chunk_size],  # 限制长度
            parent_id=section_record.id,
            embedding=generate_embedding(merged),
            node_type='chunk'
        )
```

### Chunk 合并规则

| 原始 Chunk 长度 | 处理方式 |
|----------------|----------|
| < 100 chars    | 与下一个 chunk 合并 |
| 100-800 chars  | 保持不变（最优范围） |
| > 800 chars    | 按语义边界切分 |

### 搜索与生成流程

```python
def rag_query(query: str, top_k: int = 5):
    # Step 1: 嵌入查询
    query_embedding = embed(query)
    
    # Step 2: 在小 chunks 中搜索
    matched_chunks = search_chunks(
        query_embedding, 
        node_type='chunk',
        top_k=top_k
    )
    
    # Step 3: 找到父 sections（去重）
    parent_sections = []
    for chunk in matched_chunks:
        section = get_section_by_id(chunk.parent_id)
        if section not in parent_sections:
            parent_sections.append(section)
    
    # Step 4: 构建上下文（使用完整 section 内容）
    context = "\n\n".join([
        f"【{s.heading}】\n{s.content}" 
        for s in parent_sections
    ])
    
    # Step 5: 生成答案
    answer = llm.generate(query, context)
    
    return answer, parent_sections
```

---

## 📊 效果对比

| 指标 | 当前策略 (Section 合并) | 新策略 (Small-to-Big) |
|------|------------------------|----------------------|
| **搜索精度** | ⭐⭐ 低 | ⭐⭐⭐⭐⭐ 高 |
| **召回率** | ⭐⭐⭐ 中 | ⭐⭐⭐⭐⭐ 高 |
| **上下文完整性** | ⭐⭐⭐⭐⭐ 完整 | ⭐⭐⭐⭐⭐ 完整 |
| **Token 控制** | ❌ 易超限 | ✅ 精确控制 |
| **实现复杂度** | ⭐ 简单 | ⭐⭐⭐ 中等 |

---

## 🚀 实施计划

### Phase 1: 数据库 Schema 更新
1. 添加 `content_for_embedding` 列
2. 修改 `node_type` 枚举值
3. 添加 `parent_id` 索引

### Phase 2: Chunk 处理器重构
1. 实现智能合并算法
2. 保留原始 MinerU chunks
3. 生成 chunk-level embeddings

### Phase 3: 搜索层更新
1. 修改搜索逻辑，只在 chunks 上搜索
2. 实现 parent section 查找
3. 更新上下文构建逻辑

### Phase 4: 验证与优化
1. 对比搜索精度（A/B test）
2. 调整 chunk 大小参数
3. 性能优化

---

## 💡 关键决策点

1. **是否保留当前 Section 合并表结构？**
   - 建议：保留 `section.content` 用于 LLM，新增 `chunk` 节点用于搜索

2. **Chunk 大小参数？**
   - 建议：min=200, max=800（根据内容类型可调）

3. **是否需要重叠 (overlap)？**
   - 建议：初始不需要，MinerU 已经按语义边界分割

4. **失败回退策略？**
   - 如果 chunk 搜索无结果，回退到 section-level 搜索

