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

### 5) 端到端跑通（生产简化入口）

```bash
uv run bid-pipeline run-prod \
  --context-json data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_A.json \
  --project-id <PROJECT_ID> \
  --document-id <DOCUMENT_ID> \
  --version-id <VERSION_ID> \
  --document-title "示例投标文件" \
  --bidder-name "投标方A" \
  --project-name "示例项目"
```

说明：

- 生产主入口建议使用 `run-prod`，只保留两类输入：`--context-json` 或 `--pdf-path`。
- `run-prod` 默认评分后端为 `hybrid`，默认问题集为 `cn_medical_v1 + strict_traceability`。
- 若需高级调参（例如切换后端、改 hybrid 权重），使用 `run-e2e`。
- `agent-mcp` 使用 LLM + 检索 MCP 进行评分，且仅基于可定位证据（不可定位内容会告警且不参与打分）。
- `agent-mcp` 默认执行模式为 `tool-calling`：模型先调用检索工具逐步探索，再输出评分 JSON。
- 可观测性验证：`observability.agent` 会输出 `execution_mode/turns/tool_call_count/tool_names/exploration_trace/trace_id`。
- 运行时探索日志：`--agent-trace-stdout`（或 `BID_SCORING_AGENT_MCP_TRACE_STDOUT=1`）会将每轮工具调用输出到 `stderr`。
- 轨迹落盘：`--agent-trace-save --agent-trace-dir <DIR>`（或环境变量）会保存完整探索轨迹 JSON，并在结果中返回 `observability.agent.trace_file`。
- `agent-mcp` 调用失败会自动降级到基线评分，并追加告警：`scoring_backend_agent_mcp_fallback`。
- `hybrid` 会融合 `agent-mcp`（主）与 `analyzer`（辅）结果，输出综合评分与合并告警。
- 评分结果新增 `evidence_citations`，按维度输出 `chunk_id/page_idx/bbox`，便于审计与高亮追溯。
- `run-e2e` 输出新增 `traceability` 与 `observability`：
  - `traceability`：证据可追溯覆盖率、可高亮 `chunk_ids`、链路告警
  - `observability.timings_ms`：`load/ingest/embeddings/scoring/total` 阶段耗时
- `--hybrid-primary-weight` 可覆盖 `hybrid` 主后端权重（范围 `[0,1]`）。
- `--pdf-path` 已支持直连 MinerU 并自动读取输出 `content_list.json`。
- `--mineru-parser` 支持 `auto|cli|api`（默认 `auto`）。

评分规则文件：

- 默认读取 `config/scoring_rules.yaml`。
- 可通过 `BID_SCORING_RULES_PATH` 指向自定义规则文件。
- `hybrid` 权重也支持环境变量 `BID_SCORING_HYBRID_PRIMARY_WEIGHT`（默认 `0.7`）。
- `agent-mcp` 模型：`BID_SCORING_AGENT_MCP_MODEL`（默认 `gpt-5-mini`）。
- `agent-mcp` 检索参数：`BID_SCORING_AGENT_MCP_TOP_K`、`BID_SCORING_AGENT_MCP_MODE`、`BID_SCORING_AGENT_MCP_MAX_CHARS`。
- `agent-mcp` 执行模式：`BID_SCORING_AGENT_MCP_EXECUTION_MODE=tool-calling|bulk`（默认 `tool-calling`）。
- `agent-mcp` 最大探索轮次：`BID_SCORING_AGENT_MCP_MAX_TURNS`（默认 `8`）。
- `agent-mcp` 评分策略单源：`BID_SCORING_AGENT_MCP_POLICY_PATH`（默认 `config/agent_scoring_policy.yaml`）。
- `agent-mcp` 运行时 trace：`BID_SCORING_AGENT_MCP_TRACE_STDOUT=1`。
- `agent-mcp` trace 落盘：`BID_SCORING_AGENT_MCP_TRACE_SAVE=1`、`BID_SCORING_AGENT_MCP_TRACE_DIR=<DIR>`。
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

### 8) 在 Claude Code 中安装 MCP

推荐使用 Claude Code 的 `mcp add-json` 命令，这样配置可复用、可审计。

方式 A（推荐，项目级，团队共享）：

```bash
claude mcp add-json bid-scoring '{
  "type":"stdio",
  "command":"uv",
  "args":["run","fastmcp","run","mcp_servers/retrieval_server.py","-t","stdio"],
  "env":{
    "DATABASE_URL":"${DATABASE_URL}",
    "OPENAI_API_KEY":"${OPENAI_API_KEY}"
  }
}' --scope project
```

执行后会在项目根目录生成或更新 `.mcp.json`（建议纳入版本控制）：

```json
{
  "mcpServers": {
    "bid-scoring": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "fastmcp", "run", "mcp_servers/retrieval_server.py", "-t", "stdio"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

方式 B（用户级，跨项目可用）：

```bash
claude mcp add-json bid-scoring '{
  "type":"stdio",
  "command":"uvx",
  "args":["--from","git+https://github.com/h-lu/bidScoring.git","bid-scoring-retrieval"],
  "env":{
    "DATABASE_URL":"${DATABASE_URL}",
    "OPENAI_API_KEY":"${OPENAI_API_KEY}"
  }
}' --scope user
```

安装后验证：

```bash
claude mcp list
claude mcp get bid-scoring
```

常用维护命令：

```bash
# 导入 Claude Desktop 已配置的 MCP（macOS/WSL）
claude mcp add-from-claude-desktop

# 重置项目级 MCP 的授权选择
claude mcp reset-project-choices
```

说明：

- 项目级优先适合团队协作；用户级适合个人全局工具。
- 建议仅使用环境变量注入密钥，不要把明文 key 写入 `.mcp.json`。
- 相关参考：`mcp_servers/README.md` 与 Anthropic 官方文档  
  [Connect Claude Code to tools via MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)

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

评分后端回归门禁（analyzer/agent-mcp/hybrid）：

```bash
uv run python scripts/evaluate_scoring_backends.py \
  --fail-on-thresholds \
  --summary-out data/eval/scoring_compare/summary.json
```

Skill/策略一致性门禁：

```bash
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
```

两次真实评分结果对比（baseline vs candidate）：

```bash
uv run python scripts/compare_scoring_runs.py \
  --baseline data/eval/scoring_compare/runs/<baseline>.json \
  --candidate data/eval/scoring_compare/runs/<candidate>.json \
  --output data/eval/scoring_compare/runs/<compare-report>.json
```

## 目录说明

- `bid_scoring/pipeline/`: 新主链路（ingest 与 evidence-first 领域规则）
- `bid_scoring/retrieval/`: 混合检索与评测门禁
- `mcp_servers/retrieval/`: MCP 检索工具分层实现
- `docs/usage.md`: 详细使用文档（命令与参数说明）
- `docs/guides/dual-track-architecture-and-flow.md`: 双轨架构与全流程说明（CLI 生产轨 / Claude Skill+MCP 轨）
- `docs/guides/claude-code-bid-analyze-playbook.md`: Claude Code 实操手册（MCP 安装 + 可直接粘贴 Prompt）
- `docs/guides/claude-skill-authoring-best-practices.md`: Claude Skill 官方最佳实践与本仓库落地约定
- `docs/guides/claude-scoring-run-comparison-2026-02-16.md`: 两次 Claude 全流程评分结果对比示例
- `docs/guides/claude-agent-team-plugin.md`: 标准 plugin 架构（skills/agents/hooks/commands）与安装使用说明
- `docs/guides/claude-agent-team-collaboration-mode.md`: 官方能力对照与当前协作模式决策（Mode-A）
- `tests/unit/`: 当前有效的单元测试基线

## 接下来建议做什么

1. 合并 PR 后，补齐 A/B/C 三套评测基线报告
2. 把 threshold gate 接入 CI，作为回归门禁
3. 规范密钥管理与轮换策略（避免明文泄漏）
