# 全链路测试与评测报告

> 生成日期: 2026-02-08  
> 执行分支: test/run-2026-02-08

---

## 1. 摘要

### 1.1 测试目标
使用 `data/eval/hybrid_medical_synthetic` 模拟数据，跑通从"数据库初始化→数据导入→索引/向量→检索评测→全量测试"的完整链路。

### 1.2 关键结论

| 检查项 | 状态 | 备注 |
|--------|------|------|
| 数据库初始化 | ✅ 通过 | pgcrypto/vector 扩展，textsearch 列 |
| 三版本数据导入 | ✅ 通过 | A/B/C 各 1007 chunks |
| chunks 向量生成 | ✅ 通过 | 3021/3021 成功，null_emb=0 |
| Golden 检索评测 | ✅ 通过 | hybrid MRR=0.842, R@5=0.879 |
| MCP server 烟测 | ⚠️ 部分通过 | vector 模式 OK，keyword 模式有已知问题 |
| 质量门禁 | ✅ 通过 | ruff + pytest 全绿 |

**总体结论**: 核心检索链路工作正常，vector/hybrid 模式表现优异，keyword 模式需要进一步排查。

---

## 2. 运行元信息

| 项目 | 值 |
|------|-----|
| 日期 | 2026-02-08 |
| Git SHA | c6f9f7833077998ca3f4306db88d0bdfe4d0b7fd |
| Python | 3.14.0 |
| uv | 0.9.5 (d5f39331a 2025-10-21) |
| PostgreSQL | 14.18 (Homebrew) |
| 测试数据库 | `postgresql://localhost:5432/bid_scoring_eval_test` |
| Embedding 模型 | openai/text-embedding-3-small |
| Embedding 维度 | 1536 |
| OPENAI_API_KEY | 已配置 (sk-or-v1-****3a1a) |

---

## 3. 数据库初始化

### 3.1 执行的迁移文件

| 文件 | 状态 |
|------|------|
| `migrations/000_init.sql` | ✅ 成功 |
| `migrations/002_add_fulltext_search.sql` | ✅ 成功 |

### 3.2 扩展验证

```
   Name   | Version |   Schema   | Description
----------+---------+------------+----------------------
 pgcrypto | 1.3     | public     | cryptographic functions
 plpgsql  | 1.0     | pg_catalog | PL/pgSQL procedural language
 vector   | 0.8.1   | public     | vector data type and ivfflat/hnsw
```

### 3.3 chunks 表结构验证

关键列:
- `embedding`: vector(1536) ✅
- `textsearch`: tsvector ✅
- `text_tsv`: tsvector ✅

索引:
- `idx_chunks_embedding_hnsw` (HNSW) ✅
- `idx_chunks_textsearch` (GIN) ✅
- `idx_chunks_text_tsv` (GIN) ✅

---

## 4. 数据导入（A/B/C）

### 4.1 导入命令

```bash
uv run python scripts/ingest_mineru.py \
  --path data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_{A,B,C}.json \
  --project-id 11111111-1111-1111-1111-111111111111 \
  --document-id 22222222-2222-2222-2222-22222222222{1,2,3} \
  --version-id {333...,444...,555...}
```

### 4.2 Chunks 数量统计

| Version | Chunks Count |
|---------|--------------|
| A (3333...) | 1007 |
| B (4444...) | 1007 |
| C (5555...) | 1007 |
| **总计** | **3021** |

---

## 5. HiChunk 节点构建

| 检查项 | 结果 |
|--------|------|
| hierarchical_nodes 数量 | 0 |
| 状态 | ❌ 失败 |
| 错误信息 | `'>' not supported between instances of 'NoneType' and 'int'` |
| 根因分析 | chunks.text_level 为 NULL，导致 HiChunkBuilder 比较失败 |

**影响评估**: HiChunk 节点非评测必需，不影响核心检索功能。

---

## 6. 向量生成（chunks）

### 6.1 生成结果

| Version | Total | Has Embedding | Null Embedding | 用时 |
|---------|-------|---------------|----------------|------|
| A | 1007 | 1007 | 0 | 12.9s |
| B | 1007 | 1007 | 0 | 12.6s |
| C | 1007 | 1007 | 0 | 14.0s |
| **总计** | **3021** | **3021** | **0** | **39.5s** |

### 6.2 平均速度
- ~78 条/秒 (A)
- ~80 条/秒 (B)
- ~72 条/秒 (C)

---

## 7. Golden 评测结果（核心）

### 7.1 评测配置

- **评测脚本**: `scripts/evaluate_hybrid_search_multiversion.py`
- **版本映射**: `data/eval/hybrid_medical_synthetic/version_map.json`
- **查询文件**: `data/eval/hybrid_medical_synthetic/queries.json`
- **相关性标注**: qrels.source_id.{A,B,C}.jsonl

### 7.2 详细指标

#### Scenario A

| Method | MRR | R@5 | nDCG@5 | Latency |
|--------|-----|-----|--------|---------|
| vector | 0.8258 | 0.8864 | 0.7829 | 952.44ms |
| keyword | 0.2273 | 0.1364 | 0.1848 | 2.13ms |
| hybrid | 0.8485 | 0.8864 | 0.7957 | 934.92ms |

#### Scenario B

| Method | MRR | R@5 | nDCG@5 | Latency |
|--------|-----|-----|--------|---------|
| vector | 0.8447 | 0.8864 | 0.7964 | 822.69ms |
| keyword | 0.2273 | 0.1364 | 0.1848 | 2.13ms |
| hybrid | 0.8447 | 0.8864 | 0.7961 | 882.84ms |

#### Scenario C

| Method | MRR | R@5 | nDCG@5 | Latency |
|--------|-----|-----|--------|---------|
| vector | 0.8333 | 0.8636 | 0.7864 | 905.45ms |
| keyword | 0.2273 | 0.1591 | 0.1945 | 1.83ms |
| hybrid | 0.8333 | 0.8636 | 0.7860 | 900.80ms |

### 7.3 Macro-average（综合指标）

| Method | MRR | R@5 | nDCG@5 | Latency |
|--------|-----|-----|--------|---------|
| **vector** | 0.8346 | 0.8788 | 0.7886 | 893.53ms |
| keyword | 0.2273 | 0.1439 | 0.1880 | 2.03ms |
| **hybrid** | 0.8422 | 0.8788 | 0.7926 | 906.19ms |

### 7.4 关键发现

1. **hybrid 模式表现最佳**: MRR=0.842，R@5=0.879，在 vector 基础上略有提升
2. **vector 模式表现优异**: MRR=0.835，延迟约 900ms
3. **keyword 模式异常**: MRR 仅 0.227，需要排查 tsquery 构造或索引问题
4. **延迟合理**: hybrid 约 900ms，keyword 仅 2ms（但效果差）

### 7.5 输出文件

- JSON 报告: `tmp/test-run-2026-02-08/eval/multiversion_gold_metrics.json`

---

## 8. MCP Smoke 测试

### 8.1 FastMCP Inspect

文件: `mcp_servers/retrieval_server.py`

暴露的 Tools:
- `retrieve` - 检索工具 (hybrid/keyword/vector 模式)

### 8.2 Keyword 模式烟测

| 检查项 | 结果 |
|--------|------|
| 状态 | ❌ 失败 |
| 查询 | "培训时长", "设备" |
| 返回结果 | 0 |
| 根因 | 待排查 (tsquery 构造或 textsearch 匹配问题) |

### 8.3 Vector 模式烟测

| 检查项 | 结果 |
|--------|------|
| 状态 | ✅ 通过 |
| 查询 | "设备质量保证" |
| 返回结果 | 5 |
| Top1 结果 | chunk_id=cac260f8..., page_idx=23, score=0.016 |
| 文本预览 | "（合同模板条款）通用设备质量保证期为36个月。" |

---

## 9. 质量门禁

### 9.1 Ruff Lint

```
uv run ruff check bid_scoring/ scripts/ tests/ mcp_servers/
All checks passed!
```

### 9.2 Ruff Format

```
uv run ruff format --check bid_scoring/ scripts/ tests/ mcp_servers/
77 files already formatted
```

### 9.3 Pytest

```
uv run pytest -v --tb=short
======================= 229 passed, 2 skipped in 15.93s ========================
```

**质量门禁状态**: ✅ 全部通过

---

## 10. 风险与建议

### 10.1 API 费用与可重复性

| 项目 | 消耗 |
|------|------|
| Embedding 生成 | 3021 chunks × ~500 tokens ≈ 1.5M tokens |
| 评测 query embedding | 44 queries × 3 modes ≈ 132 calls |
| 总成本 | 约 $0.05-0.10 (text-embedding-3-small) |

**建议**: 大规模测试前考虑使用本地 embedding 模型或缓存。

### 10.2 已知问题

| 优先级 | 问题 | 影响 | 建议修复方案 |
|--------|------|------|--------------|
| P1 | keyword 检索返回 0 结果 | 高 | 排查 tsquery 构造逻辑，检查 textsearch 触发器 |
| P2 | HiChunk 节点生成失败 | 低 | 处理 chunks.text_level 为 NULL 的情况 |
| P3 | 虚拟环境警告 | 低 | 修复 VIRTUAL_ENV 路径匹配问题 |

### 10.3 性能优化建议

1. **HNSW 参数调优**: 当前 ef_search 默认值可考虑根据延迟要求调整
2. **连接池**: 已启用，表现良好
3. **Query Cache**: 已启用，建议监控命中率

---

## 11. 附录：原始日志路径

```
tmp/test-run-2026-02-08/
├── db/
│   ├── migrate_000_init.log
│   ├── migrate_002_fulltext.log
│   ├── extensions.txt
│   └── chunks.columns.txt
├── ingest/
│   ├── ingest_A.log
│   ├── ingest_B.log
│   ├── ingest_C.log
│   ├── chunks.counts.txt
│   ├── hichunk.build.log
│   └── hichunk.counts.txt
├── embeddings/
│   ├── emb_A.log
│   ├── emb_B.log
│   ├── emb_C.log
│   └── chunks.embedding.counts.txt
├── eval/
│   ├── multiversion_gold_metrics.stdout.txt
│   └── multiversion_gold_metrics.json
├── mcp/
│   ├── fastmcp.inspect.txt
│   └── retrieve_impl.vector.smoke.txt
├── lint/
│   ├── ruff.check.txt
│   └── ruff.format.txt
├── pytest/
│   └── pytest.v.txt
├── env.python.txt
├── env.uv.txt
├── env.psql.txt
├── env.gitsha.txt
└── dsn.txt
```

---

## 12. 验收标准检查表

| # | 验收标准 | 状态 |
|---|---------|------|
| 1 | 数据库 bid_scoring_eval_test 从零初始化成功 | ✅ |
| 2 | pgcrypto/vector 扩展存在 | ✅ |
| 3 | chunks.textsearch 存在 | ✅ |
| 4 | A/B/C 三版本数据成功导入 | ✅ |
| 5 | chunks 数量合理（约 1000 级别/版本） | ✅ (1007/版本) |
| 6 | A/B/C 三版本 chunks embedding 全部生成 | ✅ (null_emb=0) |
| 7 | evaluate_hybrid_search_multiversion.py 成功输出 | ✅ |
| 8 | ruff check 通过 | ✅ |
| 9 | ruff format --check 通过 | ✅ |
| 10 | pytest 全绿（0 failures） | ✅ (229 passed, 2 skipped) |
| 11 | 报告文件完整、可复现 | ✅ |

---

*报告结束*
