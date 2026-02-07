# Hybrid Retrieval Synthetic Eval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为投标分析系统构建“拟真医疗器械投标文档模拟集 + 黄金检索标准（qrels）+ 可执行评估脚本”，用于稳定评估混合检索效果。

**Architecture:** 采用“数据生成器 + 标注文件 + 评估器”三件套。生成器输出符合 MineRU `content_list` 结构的模拟投标文档和查询集，标注文件以 `source_id` 为主键保存分级相关性，评估器在运行时映射到数据库 `chunk_id` 并计算 Recall/MRR/nDCG。

**Tech Stack:** Python 3.12, JSON/JSONL, psycopg, bid_scoring.hybrid_retrieval, PostgreSQL/pgvector。

### Task 1: 固化数据规范与标注口径

**Files:**
- Create: `docs/synthetic_hybrid_eval_design.md`

**Step 1: 写设计文档草案**
- 定义数据文件结构（`content_list`/`queries`/`qrels`）
- 定义 relevance 评分（0-3）
- 定义边界样本类型（同义词、冲突条款、否定表达、单位换算、表格证据）

**Step 2: 校验与现有结构一致**
Run: `python - <<'PY' ...` 读取真实样本 key/type 分布对齐检查
Expected: 文档记录字段映射与约束

**Step 3: 提交准备（本次不提交）**
- 记录后续脚本要生成的具体文件名

### Task 2: 先写失败测试，锁定数据质量门禁

**Files:**
- Create: `tests/test_synthetic_hybrid_eval_assets.py`

**Step 1: 写失败测试**
- 断言查询数量、query_type 覆盖
- 断言 qrels 含 0-3 分级
- 断言存在关键边界标签（numeric_conflict/negation/alias/table_evidence）

**Step 2: 运行测试确认失败**
Run: `uv run pytest tests/test_synthetic_hybrid_eval_assets.py -q`
Expected: FAIL（文件尚未生成）

### Task 3: 实现可复现数据生成脚本

**Files:**
- Create: `scripts/generate_synthetic_hybrid_eval_data.py`
- Create: `data/eval/hybrid_medical_synthetic/README.md`
- Create: `data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder.json`
- Create: `data/eval/hybrid_medical_synthetic/queries.json`
- Create: `data/eval/hybrid_medical_synthetic/qrels.source_id.jsonl`

**Step 1: 最小实现生成器**
- 生成拟真章节结构（目录/资格/技术/售后/培训/合同）
- 混入 text/list/table/image/aside_text/page_number
- 注入边界样本和干扰项

**Step 2: 运行生成器**
Run: `uv run python scripts/generate_synthetic_hybrid_eval_data.py --output-dir data/eval/hybrid_medical_synthetic`
Expected: 目标文件全部生成

**Step 3: 自校验**
- 检查 source_id/page_idx 与 qrels 对齐
- 检查至少 20 条查询与 query_type 分布

### Task 4: 实现黄金标准评估脚本

**Files:**
- Create: `scripts/evaluate_hybrid_search_gold.py`

**Step 1: 写评估器**
- 输入：`version_id`、`queries.json`、`qrels.source_id.jsonl`
- 运行 vector/keyword/hybrid 三路检索
- 计算：Recall@{1,3,5,10}、MRR、nDCG@{3,5,10}、延迟

**Step 2: 与 DB 映射逻辑**
- 用 `version_id + source_id -> chunk_id` 构建 qrels
- 缺失映射给出 warning，不中断全局评测

**Step 3: 输出报告**
- 汇总总体指标 + 按 query_type 分组指标
- 支持输出 JSON 文件

### Task 5: 绿测与交付说明

**Files:**
- Modify: `docs/synthetic_hybrid_eval_design.md`

**Step 1: 运行测试**
Run: `uv run pytest tests/test_synthetic_hybrid_eval_assets.py -q`
Expected: PASS

**Step 2: 快速功能验证**
Run: `uv run python scripts/generate_synthetic_hybrid_eval_data.py --output-dir data/eval/hybrid_medical_synthetic --validate-only`
Expected: PASS

**Step 3: 交付说明**
- 提供导入数据库、执行评估命令示例
- 明确可扩展点（新增 query、新增边界标签）
