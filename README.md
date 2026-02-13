# 投标分析系统（Evidence-First）

本项目是一个面向投标/评标场景的文档分析系统，核心目标是：

- 结论必须基于事实证据
- 证据必须能追溯到 PDF 原始位置（页码 + bbox）
- 不可验证证据不阻断流程，但必须显式告警

## 当前状态

- 重构状态：evidence-first 主链路已并入 `main`
- 主链路：`PDF/内容 -> 入库 -> 检索 -> 证据 -> 高亮`
- 核心验证：`ruff`、`pytest`、`pipeline CLI`、`retrieval threshold gate` 已通过

## 你最关心的：已经做完什么

1. 建立了新的 evidence-first pipeline 分层架构：
   - `bid_scoring/pipeline/domain`
   - `bid_scoring/pipeline/application`
   - `bid_scoring/pipeline/infrastructure`
   - `bid_scoring/pipeline/interfaces`
2. 统一入口 CLI：`bid-pipeline ingest-content-list`
3. MCP 检索服务完成模块化拆分，并强化证据字段输出：
   - 检索结果包含 `evidence_status/evidence_units/warnings`
   - 增加检索诊断透传 `include_diagnostics`
4. 新增“可高亮目标门禁”：
   - 工具：`prepare_highlight_targets`
   - 只放行可定位且证据可验证的 `chunk_ids`
5. 新增离线评测阈值门禁：
   - 脚本：`scripts/evaluate_hybrid_search_gold.py`
   - 参数：`--thresholds-file --fail-on-thresholds`
6. 清理旧链路：
   - 删除 `mineru/process_pdfs.py`
   - 删除 `mineru/coordinator.py`
   - 删除 `scripts/ingest_mineru.py`

## 快速开始

### 1) 安装依赖

```bash
uv sync --locked --dev
```

### 2) 配置 `.env`

最小配置：

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（如 OpenRouter: `https://openrouter.ai/api/v1`）
- `OPENAI_EMBEDDING_MODEL`（如 `openai/text-embedding-3-small`）
- `OPENAI_EMBEDDING_DIM=1536`

### 3) 初始化数据库

```bash
uv run python scripts/apply_migrations.py
```

### 4) 入库

```bash
uv run bid-pipeline ingest-content-list --help
```

### 5) 生成向量

```bash
uv run python scripts/build_all_embeddings.py --version-id <VERSION_ID>
```

### 6) 启动 MCP 检索服务

```bash
uv run fastmcp run mcp_servers/retrieval_server.py -t stdio
```

## 常用验证命令

```bash
uv run ruff check .
uv run pytest -q
uv run python -m bid_scoring.pipeline.interfaces.cli --help
```

检索评测门禁：

```bash
uv run python scripts/evaluate_hybrid_search_gold.py \
  --version-id <VERSION_ID> \
  --queries-file data/eval/hybrid_medical_synthetic/queries.json \
  --qrels-file data/eval/hybrid_medical_synthetic/qrels.source_id.A.jsonl \
  --thresholds-file data/eval/hybrid_medical_synthetic/retrieval_baseline.thresholds.json \
  --fail-on-thresholds
```

## 目录说明

- `bid_scoring/pipeline/`: 新主链路（ingest 与 evidence-first 领域规则）
- `bid_scoring/retrieval/`: 混合检索与评测门禁
- `mcp_servers/retrieval/`: MCP 检索工具分层实现
- `docs/usage.md`: 详细使用文档（命令与参数说明）
- `tests/unit/`: 当前有效的单元测试基线

## 接下来建议做什么

1. 合并 PR 后，补齐 A/B/C 三套评测基线报告
2. 把 threshold gate 接入 CI，作为回归门禁
3. 规范密钥管理与轮换策略（避免明文泄漏）
