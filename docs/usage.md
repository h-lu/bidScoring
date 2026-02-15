# 投标分析_kimi 使用文档

本仓库提供一套“投标/招标文档分析”能力，核心是：

- 文档内容入库（Postgres）
- 分块与层级结构（HiChunk）
- 混合检索（pgvector 语义向量 + PostgreSQL 全文检索）
- 通过 FastMCP 暴露 `retrieve` 工具，供外部 LLM 客户端调用

下文以“从零跑起来”为目标，覆盖本地开发、数据准备、检索调用与 MCP Server 使用。

## 1. 环境与依赖

- Python: 3.11+
- Postgres: 需要安装 pgvector 扩展（`CREATE EXTENSION vector;`）
- 包管理: 使用 `uv`

安装依赖（含 dev 依赖）：

```bash
uv sync --locked --dev
```

常用自检：

```bash
uv run ruff check .
uv run pytest -q
```

## 2. 配置（.env / 环境变量）

项目使用 `bid_scoring/config.py` 读取配置，`.env` 优先于系统环境变量。

最小必需项：

- `DATABASE_URL`: 例如 `postgresql://localhost:5432/bid_scoring`
- `OPENAI_API_KEY`: 向量/混合检索与向量化脚本需要
- `OPENAI_EMBEDDING_MODEL`: 例如 `text-embedding-3-small`（按你的账号可用模型选择）
- `OPENAI_EMBEDDING_DIM`: 必须为 `1536`（当前 schema 写死为 1536 维）

可选项：

- `OPENAI_BASE_URL`: 代理/私有网关场景
- `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`
- `BID_SCORING_RETRIEVER_CACHE_SIZE`: MCP 侧 retriever LRU 容量（默认 32）
- `BID_SCORING_QUERY_CACHE_SIZE`: retriever 内部 query cache 容量（默认 1024）

## 3. 数据库初始化与迁移

### 3.1 初始化 schema（必需）

用脚本一键初始化（会执行 `migrations/000_init.sql`）：

```bash
uv run python scripts/apply_migrations.py
```

等价手动执行：

```bash
psql "$DATABASE_URL" -f migrations/000_init.sql
```

### 3.2 启用全文检索列 `chunks.textsearch`（强烈建议）

检索代码的 keyword/fulltext 路径使用 `chunks.textsearch` 列。
如果你的库里只有 `chunks.text_tsv` 而没有 `chunks.textsearch`，keyword/hybrid 可能会在运行时 SQL 报错。

执行全文检索迁移：

```bash
psql "$DATABASE_URL" -f migrations/002_add_fulltext_search.sql
```

该迁移会：

- 添加 `chunks.textsearch` (tsvector)
- 回填存量数据的 tsvector
- 创建 GIN 索引与触发器，保证 `text_raw` 变化时自动更新 `textsearch`

## 4. 数据准备

### 4.1 入库（写入 chunks 等表）

核心入库函数：

- `bid_scoring/ingest.py::ingest_content_list(...)`

如已有 MineRU 的解析结果，可通过统一 CLI 或自行调用入库逻辑：

```bash
uv run bid-pipeline ingest-content-list --help
```

示例：

```bash
uv run bid-pipeline ingest-content-list \
  --content-list data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_A.json \
  --project-id <PROJECT_UUID> \
  --document-id <DOCUMENT_UUID> \
  --version-id <VERSION_UUID> \
  --document-title "示例投标文件"
```

### 4.2 端到端 CLI（入库 + 向量化 + 评分）

端到端命令：

```bash
uv run bid-pipeline run-prod --help
```

生产推荐（PDF 直连）：

```bash
uv run bid-pipeline run-prod \
  --pdf-path /path/to/bid.pdf \
  --project-id <PROJECT_UUID> \
  --document-id <DOCUMENT_UUID> \
  --version-id <VERSION_UUID> \
  --document-title "示例投标文件" \
  --bidder-name "投标方A" \
  --project-name "示例项目"
```

预解析输入模式（`context_json` / `content_list`）：

```bash
uv run bid-pipeline run-prod \
  --context-json /path/to/context_list.json \
  --project-id <PROJECT_UUID> \
  --document-id <DOCUMENT_UUID> \
  --version-id <VERSION_UUID> \
  --document-title "示例投标文件" \
  --bidder-name "投标方A" \
  --project-name "示例项目"
```

说明（生产默认）：

- 生产主入口为 `run-prod`，输入入口固定为两种：`--pdf-path` 或 `--context-json`。
- 默认评分后端是 `hybrid`。
- 默认问题集是 `cn_medical_v1`，默认策略是 `strict_traceability`。
- `analyzer / agent-mcp / hybrid` 会统一使用问题集解析出的维度与关键词；不显式传 `--dimensions` 时使用问题集全部维度。
- `agent-mcp` 使用 LLM + 检索 MCP 评分，且仅基于可定位证据（不可定位内容会告警且不参与打分）。
- `agent-mcp` 默认执行模式是 `tool-calling`：模型先调用检索工具逐步探索证据，再输出评分 JSON。
- `agent-mcp` 执行失败会自动降级到基线评分，并追加告警：`scoring_backend_agent_mcp_fallback`。
- `hybrid` 会融合 `agent-mcp`（主）与 `analyzer`（辅）结果，输出综合评分与合并告警。
- 评分输出包含 `evidence_citations`，按维度提供 `chunk_id/page_idx/bbox`，用于事实追溯与 PDF 高亮。
- `run-e2e` 输出包含：
  - `traceability`：证据可追溯统计（覆盖率、可高亮 `chunk_ids`、告警码）
  - `observability.timings_ms`：`load/ingest/embeddings/scoring/total` 阶段耗时（毫秒）
- `--pdf-path` 已支持直连 MinerU 并自动读取输出 `content_list.json`。
- `--mineru-parser` 支持 `auto|cli|api`（默认 `auto`）。
- 当配置了问题集参数时，结果会在 `observability.question_bank` 输出 `pack_id/overlay/question_count`。

开发高级参数（保留扩展，不建议生产常用）：

- `run-e2e`（高级入口）：支持 `--context-list/--context-json/--content-list/--pdf-path`
- `--scoring-backend`、`--hybrid-primary-weight`
- `--skip-embeddings`
- `--question-pack`、`--question-overlay`
- `--mineru-output-dir`

评分规则配置：

- 默认文件：`config/scoring_rules.yaml`
- 自定义路径：设置环境变量 `BID_SCORING_RULES_PATH=/path/to/scoring_rules.yaml`
- `hybrid` 权重：`BID_SCORING_HYBRID_PRIMARY_WEIGHT=0.7`（可被 CLI 参数覆盖）
- `agent-mcp` 模型：`BID_SCORING_AGENT_MCP_MODEL=gpt-5-mini`
- `agent-mcp` 检索参数：`BID_SCORING_AGENT_MCP_TOP_K=8`、`BID_SCORING_AGENT_MCP_MODE=hybrid`、`BID_SCORING_AGENT_MCP_MAX_CHARS=320`
- `agent-mcp` 执行模式：`BID_SCORING_AGENT_MCP_EXECUTION_MODE=tool-calling|bulk`（默认 `tool-calling`）
- `agent-mcp` 最大探索轮次：`BID_SCORING_AGENT_MCP_MAX_TURNS=8`
- `agent-mcp` 策略文件：`BID_SCORING_AGENT_MCP_POLICY_PATH=config/agent_scoring_policy.yaml`
- MinerU 解析模式：`MINERU_PDF_PARSER=auto|cli|api`
- MinerU 命令模板（CLI 模式）：`MINERU_PDF_COMMAND=\"magic-pdf -p {pdf_path} -o {output_dir}\"`
- MinerU 输出目录：`MINERU_OUTPUT_ROOT=.mineru-output`
- MinerU CLI 超时：`MINERU_PDF_TIMEOUT_SECONDS=1800`
- MinerU API：`MINERU_API_URL`、`MINERU_API_KEY`
- MinerU API 请求/轮询：`MINERU_API_REQUEST_TIMEOUT_SECONDS`、`MINERU_API_POLL_TIMEOUT_SECONDS`、`MINERU_API_POLL_INTERVAL_SECONDS`
- MinerU API 上传重试：`MINERU_API_UPLOAD_MAX_RETRIES=3`

### 4.4 评分后端回归门禁（CI同款）

```bash
uv run python scripts/evaluate_scoring_backends.py \
  --fail-on-thresholds \
  --summary-out data/eval/scoring_compare/summary.json
```

说明：

- 默认使用 `data/eval/scoring_compare/content_list.minimal.json`。
- 默认启用 `BID_SCORING_AGENT_MCP_DISABLE=1` 的稳定模式（agent-mcp 走降级链路），用于 CI 可复现门禁。

### 4.5 运行结果归档与对比

建议把每次真实评分输出保存到 `data/eval/scoring_compare/runs/`，例如：

- `data/eval/scoring_compare/runs/2026-02-14-run-prod-hybrid-synthetic-bidder-A.json`

比较两次输出（基线 vs 候选）：

```bash
uv run python scripts/compare_scoring_runs.py \
  --baseline data/eval/scoring_compare/runs/<baseline>.json \
  --candidate data/eval/scoring_compare/runs/<candidate>.json \
  --output data/eval/scoring_compare/runs/<compare-report>.json
```

对比报告包含：

- 核心指标差值：`overall_score`、`coverage_ratio`、`citation_count_total`、`chunks_analyzed`
- 分维度得分差值：`delta.dimension_scores`
- 告警变化：`warnings_added`、`warnings_removed`

### 4.3 生成向量（vector/hybrid 必需）

向量检索需要 `chunks.embedding` 有值。推荐用全量向量化脚本：

```bash
uv run python scripts/build_all_embeddings.py --help
```

常见用法（按版本生成）：

```bash
uv run python scripts/build_all_embeddings.py --version-id "<VERSION_UUID>"
```

注意：

- 该脚本要求 `OPENAI_API_KEY` 已设置
- 会同时处理 `chunks`、`contextual_chunks`、`hierarchical_nodes`（可用参数跳过）

## 5. Python 直接调用检索

### 5.1 HybridRetriever（混合检索）

```python
from bid_scoring.config import load_settings
from bid_scoring.retrieval import HybridRetriever

settings = load_settings()

retriever = HybridRetriever(
    version_id="<VERSION_UUID>",
    settings=settings,
    top_k=10,
    # enable_cache=True,  # 可选：开启 query cache
)

results = retriever.retrieve("请找出投标保证金相关条款")
for r in results:
    print(r.chunk_id, r.page_idx, r.score)
    print(r.text[:200])
```

参数要点（仅列常用）：

- `top_k`: 返回条数
- `vector_weight` / `keyword_weight`: 混合融合权重
- `hnsw_ef_search`: 向量搜索召回/性能权衡
- `enable_rerank`: 是否二次重排（默认关闭）

### 5.2 仅 keyword / 仅 vector

仓库的 MCP `mode="keyword"` / `mode="vector"` 也是直接复用 `HybridRetriever` 的内部实现。
如需更细粒度控制，可阅读并自行调用：

- `HybridRetriever._vector_search(...)`
- `HybridRetriever._keyword_search_fulltext(...)`

## 6. 启动 MCP Server（FastMCP）

入口文件：

- `mcp_servers/retrieval_server.py`

### 6.1 stdio（推荐，用于本地集成到 MCP Client）

```bash
uv run fastmcp run mcp_servers/retrieval_server.py -t stdio
```

### 6.2 http（可选）

```bash
uv run fastmcp run mcp_servers/retrieval_server.py -t http --host 127.0.0.1 --port 8000
```

### 6.3 Tool: retrieve

MCP tool 名称：`retrieve`

参数（与 `mcp_servers/retrieval_server.py::retrieve_impl` 一致）：

- `version_id`: 文档版本 UUID（在该版本范围内检索）
- `query`: 查询文本
- `top_k`: 返回条数（默认 10）
- `mode`: `hybrid` | `keyword` | `vector`（默认 `hybrid`）
- `keywords`: 可选关键词列表（不传会自动抽取）
- `use_or_semantic`: 仅对 keyword 路径有效（默认 true）
- `include_text`: 是否返回 chunk 文本（默认 true）
- `max_chars`: 返回文本截断长度（可选）
- `include_diagnostics`: 是否返回检索诊断信息（默认 false）

返回：

- `warnings`: 聚合告警码（如 `missing_evidence_chain`）
- `results`: 列表，每项包含：
  - 基础字段：`chunk_id/page_idx/source/score/vector_score/keyword_score/rerank_score/text`
  - 定位字段：`bbox/coord_system`
  - 证据字段：`evidence_status/evidence_units/warnings`
- `diagnostics`（可选）：`result_count/vector_hits/keyword_hits/hybrid_hits/warning_counts`

### 6.4 Tool: prepare_highlight_targets

MCP tool 名称：`prepare_highlight_targets`

用途：对查询结果执行“可高亮门禁”，只输出可定位且证据可验证的 `chunk_ids`，并保留告警。

参数：

- `version_id/query/top_k/mode/keywords/use_or_semantic`：与 `retrieve` 对齐
- `include_diagnostics`: 是否返回 `retrieval + gate` 诊断信息

返回：

- `chunk_ids`: 可用于 `highlight_pdf` 的安全候选集合
- `included_count/excluded_count`: 门禁通过/过滤数量
- `warnings`: 聚合告警码（包含检索告警与门禁告警）
- `diagnostics`（可选）：`retrieval` 与 `gate` 统计

## 7. 常见问题排查

### 7.1 `OPENAI_API_KEY` 未设置 / embedding 生成失败

- vector/hybrid 检索或向量化脚本需要 `OPENAI_API_KEY`
- 请检查 `.env` 是否生效（`bid_scoring/config.py` 使用 `load_dotenv(override=True)`）

### 7.2 Postgres 报 `extension "vector" does not exist`

- 需要安装 pgvector，并在库内执行 `CREATE EXTENSION vector;`

### 7.3 keyword/hybrid SQL 报 `column "textsearch" does not exist`

- 执行 `migrations/002_add_fulltext_search.sql`，创建 `chunks.textsearch` 与索引/触发器

### 7.4 vector 结果为空

- 检查 `chunks.embedding` 是否已填充
- 先运行 `scripts/build_all_embeddings.py`
