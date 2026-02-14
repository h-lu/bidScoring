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
2. 统一入口 CLI：
   - `bid-pipeline ingest-content-list`
   - `bid-pipeline run-e2e`
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
   - 删除 `mineru/convert_pdf.py`
   - 删除 `scripts/ingest_mineru.py`
   - 删除 `scripts/evaluate_hybrid_search.py`
   - 删除 `scripts/eval_e2e_full_pipeline.py`
   - 删除 `scripts/mcp_demo.py`
   - 删除 `docs/plans/2026-02-13-e2e-cli-fullflow.md`

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

### 5) 端到端跑通（测试模式，绕过 MinerU）

```bash
uv run bid-pipeline run-e2e \
  --context-list data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_A.json \
  --project-id <PROJECT_ID> \
  --document-id <DOCUMENT_ID> \
  --version-id <VERSION_ID> \
  --document-title "示例投标文件" \
  --bidder-name "投标方A" \
  --project-name "示例项目"
```

说明：

- `--context-list` 与 `--content-list` 等价，都会绕过 MinerU。
- `--scoring-backend` 支持 `analyzer|agent-mcp|hybrid`，当前默认 `analyzer`。
- `agent-mcp` 使用 LLM + 检索 MCP 进行评分，且仅基于可定位证据（不可定位内容会告警且不参与打分）。
- `agent-mcp` 调用失败会自动降级到基线评分，并追加告警：`scoring_backend_agent_mcp_fallback`。
- `hybrid` 会融合 `agent-mcp`（主）与 `analyzer`（辅）结果，输出综合评分与合并告警。
- `--hybrid-primary-weight` 可覆盖 `hybrid` 主后端权重（范围 `[0,1]`）。
- `--pdf-path` 已支持直连 MinerU 并自动读取输出 `content_list.json`。
- `--mineru-parser` 支持 `auto|cli|api`（默认 `auto`）。

评分规则文件：

- 默认读取 `config/scoring_rules.yaml`。
- 可通过 `BID_SCORING_RULES_PATH` 指向自定义规则文件。
- `hybrid` 权重也支持环境变量 `BID_SCORING_HYBRID_PRIMARY_WEIGHT`（默认 `0.7`）。
- `agent-mcp` 模型：`BID_SCORING_AGENT_MCP_MODEL`（默认 `gpt-4o-mini`）。
- `agent-mcp` 检索参数：`BID_SCORING_AGENT_MCP_TOP_K`、`BID_SCORING_AGENT_MCP_MODE`、`BID_SCORING_AGENT_MCP_MAX_CHARS`。
- MinerU CLI 模式可通过 `MINERU_PDF_COMMAND` 自定义命令模板（占位符：`{pdf_path}`、`{output_dir}`）。
- MinerU 通用输出根目录：`MINERU_OUTPUT_ROOT`，CLI 超时：`MINERU_PDF_TIMEOUT_SECONDS`（默认 `1800`）。
- MinerU API 模式使用：`MINERU_API_URL`、`MINERU_API_KEY`。
- MinerU API 超时/轮询：`MINERU_API_REQUEST_TIMEOUT_SECONDS`、`MINERU_API_POLL_TIMEOUT_SECONDS`、`MINERU_API_POLL_INTERVAL_SECONDS`。
- MinerU API 上传重试：`MINERU_API_UPLOAD_MAX_RETRIES`（默认 `3`）。

### 6) 生成向量（独立执行）

```bash
uv run python scripts/build_all_embeddings.py --version-id <VERSION_ID>
```

### 7) 启动 MCP 检索服务

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
