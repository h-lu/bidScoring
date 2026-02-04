# Small-to-Big Chunking 策略实施总结

## ✅ 已完成工作

### Phase 1: 数据库 Schema 更新
- ✅ 添加 `content_for_embedding` 列（用于存储处理后的 embedding 内容）
- ✅ 添加 `char_count` 列（用于快速统计）
- ✅ 添加 `page_range` 列（JSONB 格式存储页面范围）
- ✅ 添加 `order_index` 列（用于排序）
- ✅ 添加 `document_id` 列（关联文档）
- ✅ 更新 `node_type` 约束，支持 'chunk' 类型
- ✅ 创建 `v_chunks_with_sections` 视图（方便 small-to-big 查询）
- ✅ 添加 parent_id 索引优化

### Phase 2: Chunk 处理器重构
- ✅ 创建 `bid_scoring/chunk_processor.py` 模块
- ✅ 实现 `SmartChunkMerger` 类：
  - 合并短 chunks (< 200 chars) 避免碎片化
  - 保持中等 chunks (200-800 chars) 不变
  - 不跨页合并
  - 支持 forward/backward 合并策略
- ✅ 实现 `SectionChunkBuilder` 类：
  - 从 ParagraphMerger 输出构建 sections
  - 每个 section 包含多个处理后的 chunks
- ✅ 实现 `create_small_to_big_sections()` 便捷函数
- ✅ 数据类定义：`ProcessedChunk`, `SectionWithChunks`

### Phase 3: CPC Pipeline 更新
- ✅ 修改 `_process_document_structure_first()` 方法
- ✅ 新增 `_store_small_to_big_structure()` 方法：
  - 存储 section 节点（完整内容，用于 LLM）
  - 存储 chunk 节点（处理后内容，用于 embedding）
  - 建立 parent_id 关联
- ✅ 新增 `_generate_chunk_embeddings()` 方法：
  - 为所有 chunk 节点生成 embeddings
  - 支持批量处理
- ✅ 处理 UUID 转换（source_chunk_ids）

## 📊 Small-to-Big vs Section-Merge 对比

| 特性 | 旧策略 (Section-Merge) | 新策略 (Small-to-Big) |
|------|----------------------|---------------------|
| **搜索粒度** | Section (大段落) | Chunk (小段落) |
| **Embedding 内容** | 完整 section | 处理后 chunk |
| **Chunk 大小** | 无限制 (可达 8000+ chars) | 200-800 chars |
| **搜索精度** | ⭐⭐ 低 | ⭐⭐⭐⭐⭐ 高 |
| **Token 控制** | ❌ 易超限 | ✅ 精确控制 |
| **LLM 上下文** | 完整 section | 完整 section (通过 parent) |
| **实现复杂度** | 简单 | 中等 |

## 🎯 核心设计

```
Small-to-Big Retrieval Flow:

1. Ingest (MinerU) → Raw Chunks
                        ↓
2. ParagraphMerger → Natural Paragraphs
                        ↓
3. SmartChunkMerger → Processed Chunks (200-800 chars)
                        ↓
4. SectionChunkBuilder → Sections + Chunks
                        ↓
5. Database Storage:
   - Section: node_type='section', content=完整内容
   - Chunks: node_type='chunk', content_for_embedding=处理后内容, parent_id=section_id
                        ↓
6. Embedding Generation → 只为 chunks 生成 vectors
                        ↓
7. Search Phase:
   - Query → Embedding
   - Search in chunks (small, precise)
   - Return top-k chunks
                        ↓
8. Generation Phase:
   - Get parent sections of matched chunks
   - Use section.content (full context) for LLM
   - Generate answer
```

## 🔧 关键参数

```python
# Chunk 大小阈值
MIN_CHUNK_SIZE = 200  # 小于此值的 chunks 会被合并
MAX_CHUNK_SIZE = 800  # 用于 embedding 的最大字符数
MAX_EMBEDDING_TOKENS = 8191  # OpenAI 限制

# 数据库列
content: str  # Section 的完整内容（用于 LLM）
content_for_embedding: str  # Chunk 的处理后内容（用于 embedding）
char_count: int  # content_for_embedding 的字符数
parent_id: UUID  # Chunk 指向 Section
```

## 📁 新增/修改文件

```
bid_scoring/
├── chunk_processor.py          # NEW: Small-to-Big 核心逻辑
├── cpc_pipeline.py             # MOD: 集成新策略
└── embeddings.py               # MOD: 添加 tiktoken 精确计算

migrations/
└── 010_small_to_big_chunking.sql  # NEW: Schema 更新

docs/
├── rag_chunking_strategy_analysis.md  # NEW: 策略分析
└── small_to_big_implementation_summary.md  # NEW: 本总结
```

## 🧪 测试验证

```bash
# 测试 chunk processor
python -c "from bid_scoring.chunk_processor import create_small_to_big_sections; ..."

# 测试数据库存储
python -c "from bid_scoring.cpc_pipeline import CPCPipeline; ..."

# 验证数据
psql -c "SELECT node_type, COUNT(*) FROM hierarchical_nodes GROUP BY node_type;"
```

## 🚀 后续工作

### Phase 4: 生成 Chunk-level Embeddings
- 需要配置有效的 OpenAI API key
- 运行 pipeline 生成所有 chunk 的 embeddings

### Phase 5: 更新搜索层（Small-to-Big）
- 实现向量相似度搜索（在 chunks 上）
- 实现 parent section 获取
- 更新 RAG query 流程

### Phase 6: 性能优化
- 添加 HNSW 索引参数调优
- 实现缓存策略
- 批量处理优化

## 📝 注意事项

1. **API Key**: 当前 embedding 生成需要有效的 OpenAI API key
2. **数据迁移**: 需要清空并重新导入现有文档
3. **兼容性**: node_type 约束已更新，支持 'chunk' 类型
4. **视图**: v_chunks_with_sections 方便查询 chunk + parent section

## 💡 使用示例

```python
import asyncio
from bid_scoring.cpc_pipeline import CPCPipeline, CPCPipelineConfig

config = CPCPipelineConfig(
    enable_contextual=False,
    enable_raptor=False,
    use_structure_rebuilder=True,
)

pipeline = CPCPipeline(config=config)
result = asyncio.run(pipeline.process_document(
    content_list=mineru_output,
    document_title='投标文件',
    project_id=project_uuid,
    document_id=document_uuid,
    version_id=version_uuid,
))

print(f'Created {result.nodes_created} nodes')
```

