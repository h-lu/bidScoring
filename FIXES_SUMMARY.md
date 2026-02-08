# 测试发现问题修复总结

> 修复日期: 2026-02-08  
> 执行分支: test/run-2026-02-08

---

## 修复清单

### ✅ 修复 1: Keyword 检索返回 0 结果 (P1)

**问题描述**: 
- `keyword_search_fulltext` 使用 `websearch_to_tsquery('simple', ...)` 进行中文关键词检索
- 由于 `simple` 配置不对中文分词，导致整个句子作为一个 token，无法匹配单个关键词
- 例如：`to_tsvector('simple', '通用设备质量...')` 生成 `'通用设备质量...':1`，查询 `'设备'` 无法匹配

**解决方案** (使用 pg_trgm GIN 索引):
1. 启用 `pg_trgm` 扩展 - 提供 trigram 匹配支持
2. 创建 GIN 索引 `idx_chunks_text_raw_trgm` - 为 ILIKE 查询加速
3. 修改 `keyword_search_fulltext` 使用 ILIKE + pg_trgm 索引

**性能对比**:
- **纯 ILIKE** (无索引): 全表扫描，O(n) 复杂度，大数据集性能差
- **ILIKE + pg_trgm GIN**: 索引扫描，Bitmap Index Scan，大表性能显著提升
- 当前测试: 延迟仅 ~5ms，表现优异

**SQL 迁移**:
```sql
-- migrations/003_add_pg_trgm_index_for_keyword_search.sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_chunks_text_raw_trgm 
ON chunks USING gin(text_raw gin_trgm_ops);
```

**修复前指标**:
```
keyword MRR=0.227 R@5=0.144 nDCG@5=0.188
```

**修复后指标**:
```
keyword MRR=0.939 R@5=0.856 nDCG@5=0.857 Lat=5.36ms
```

**改进**: MRR 从 0.227 提升到 0.939 (+313%)，延迟仅 5ms

---

### ✅ 修复 2: HiChunk 节点生成失败 (P2)

**问题描述**:
- `HiChunkBuilder._is_heading()` 方法中 `item.get("text_level", 0)` 返回 `None`
- 当 `text_level` 存在但为 `None` 时，`None > 0` 抛出 TypeError
- 导致大部分版本无法生成 HiChunk 节点

**修复方案**:
- 修改 `bid_scoring/hichunk_builder.py`
- 显式检查 `text_level is not None` 再进行比较

**修复前**:
```python
def _is_heading(self, item: dict) -> bool:
    return item.get("type") == "text" and item.get("text_level", 0) > 0
```

**修复后**:
```python
def _is_heading(self, item: dict) -> bool:
    text_level = item.get("text_level")
    return item.get("type") == "text" and text_level is not None and text_level > 0
```

**修复前状态**: 0 个节点，全部失败  
**修复后状态**: 9,435 个节点，全部成功

---

### ✅ 修复 3: 测试更新

**问题描述**:
- 两个测试用例依赖于旧的 `websearch_to_tsquery` 实现
- `test_fulltext_search_uses_websearch_to_tsquery_and_normalized_rank`
- `test_fulltext_search_skips_when_querytree_not_indexable`

**修复方案**:
- 更新 `tests/test_hybrid_retrieval_fulltext.py`
- 修改测试以验证新的 ILIKE + pg_trgm 实现
- 重命名测试以反映新的行为

---

## 最终评测指标对比

| Method | 修复前 MRR | 修复后 MRR | 改进 |
|--------|-----------|-----------|------|
| vector | 0.835 | 0.835 | - |
| keyword | **0.227** | **0.939** | **+313%** |
| hybrid | 0.842 | 0.948 | +12.6% |

**延迟表现**:
- vector: ~933ms (embedding 计算)
- keyword: ~5ms (pg_trgm GIN 索引)
- hybrid: ~1139ms (vector + keyword)

---

## 修改的文件

1. `bid_scoring/retrieval/search_keyword.py` - 使用 ILIKE + pg_trgm GIN 索引
2. `bid_scoring/hichunk_builder.py` - 修复 text_level NULL 值处理
3. `tests/test_hybrid_retrieval_fulltext.py` - 更新测试用例
4. `migrations/003_add_pg_trgm_index_for_keyword_search.sql` - 新迁移文件

---

## 验证结果

- ✅ Ruff check passed
- ✅ Ruff format passed  
- ✅ Pytest: 229 passed, 2 skipped
- ✅ Keyword 检索: 3/3 测试通过，MRR=0.939
- ✅ HiChunk 节点生成: 9,435 节点成功
- ✅ pg_trgm GIN 索引: 已创建并生效

---

## 技术说明: pg_trgm vs 纯 ILIKE

### 为什么 pg_trgm + GIN 索引比纯 ILIKE 好？

**纯 ILIKE**:
- 全表扫描 (Sequential Scan)
- 逐行匹配模式
- 时间复杂度: O(n*m) n=行数, m=模式长度
- 大数据集性能极差

**ILIKE + pg_trgm GIN 索引**:
- Bitmap Index Scan
- Trigram (3-gram) 索引快速定位候选行
- 时间复杂度: O(k) k=匹配行数 (通常 k << n)
- 大数据集性能显著提升

### 索引验证
```sql
EXPLAIN ANALYZE SELECT ... WHERE text_raw ILIKE '%设备%';
-- 使用: Bitmap Index Scan on idx_chunks_text_raw_trgm
```

---

## 部署说明

在生产环境部署时，需要运行迁移：
```bash
psql $DATABASE_URL -f migrations/003_add_pg_trgm_index_for_keyword_search.sql
```

或在应用启动时自动执行迁移。
