# HNSW SQL Fix And Multi-Version Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 `hnsw.ef_search` SQL 导致向量检索失效的问题，并将模拟集扩展到 A/B/C 三供应商多版本，输出跨版本黄金评测基线。

**Architecture:** 先通过最小回归测试锁定 SQL 根因，再修复 `HybridRetriever._vector_search` 的 session 参数设置方式。随后将现有单版本生成器升级为可输出三份风格差异化投标文档，但共享查询与 qrels 语义框架，最后新增跨版本评测脚本聚合每个版本的 vector/keyword/hybrid 指标。

**Tech Stack:** Python 3.12, psycopg, PostgreSQL/pgvector, pytest, ruff。

### Task 1: hnsw.ef_search 根因回归测试

**Files:**
- Modify: `tests/test_hybrid_retrieval.py`

**Step 1: Write the failing test**
- 构造 fake cursor，断言 `SET hnsw.ef_search` 不应使用参数绑定 `%s`。

**Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_hybrid_retrieval.py -k hnsw_ef_search -q`
Expected: FAIL（当前实现使用参数化）

### Task 2: 最小修复向量检索 SQL

**Files:**
- Modify: `bid_scoring/hybrid_retrieval.py`

**Step 1: Write minimal implementation**
- 改为安全的 SQL 常量拼接方式（对 `int` 强制转换）执行 `SET hnsw.ef_search = <int>`。

**Step 2: Run tests to verify pass**
Run: `uv run pytest tests/test_hybrid_retrieval.py -k hnsw_ef_search -q`
Expected: PASS

### Task 3: 重跑黄金评测并记录修复结果

**Files:**
- Output: `/tmp/hybrid_gold_eval_report_after_fix.json`

**Step 1: 运行黄金评测脚本**
Run: `uv run python scripts/evaluate_hybrid_search_gold.py --version-id 33333333-3333-3333-3333-333333333333 --output /tmp/hybrid_gold_eval_report_after_fix.json`
Expected: vector/hybrid 指标恢复，不再全零

### Task 4: 扩展模拟集为 A/B/C 多供应商

**Files:**
- Modify: `scripts/generate_synthetic_hybrid_eval_data.py`
- Modify: `tests/test_synthetic_hybrid_eval_assets.py`
- Modify: `docs/synthetic_hybrid_eval_design.md`

**Step 1: 生成器支持 A/B/C 三版本**
- 新增 `--scenario` 或 `--all-scenarios` 输出三份 `content_list`。
- 每份保留相同 query/qrels 语义，但条款措辞、冲突文本、干扰项不同。

**Step 2: 测试更新**
Run: `uv run pytest tests/test_synthetic_hybrid_eval_assets.py -q`
Expected: PASS

### Task 5: 跨版本评测基线

**Files:**
- Create: `scripts/evaluate_hybrid_search_multiversion.py`

**Step 1: 实现跨版本聚合**
- 输入版本映射（A/B/C -> version_id）
- 调用黄金评测逻辑汇总全局和分版本指标

**Step 2: 运行验证**
Run: `uv run python scripts/evaluate_hybrid_search_multiversion.py ...`
Expected: 输出每版本 + macro 平均 baseline
