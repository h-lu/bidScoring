# 使用手册

本文给出项目的可执行命令清单与推荐流程。

## 1. CLI 命令

### 1.1 生产入口

```bash
uv run bid-pipeline run-prod --help
```

要点：
- 输入二选一：`--context-json` 或 `--pdf-path`
- 默认评分后端：`hybrid`
- 默认质量模式：`--quality-mode fast`（对应 `cn_medical_v1 + fast_eval`）
- 高精度模式：`--quality-mode strict`（对应 `cn_medical_v1 + strict_traceability`）

示例：

```bash
uv run bid-pipeline run-prod \
  --context-json data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_A.json \
  --project-id <PROJECT_UUID> \
  --document-id <DOCUMENT_UUID> \
  --version-id <VERSION_UUID> \
  --document-title "示例投标文件" \
  --bidder-name "投标方A" \
  --project-name "示例项目"
```

### 1.2 高级入口

```bash
uv run bid-pipeline run-e2e --help
```

适用场景：
- 需要切换 `--scoring-backend`
- 需要覆盖 `--hybrid-primary-weight`
- 需要手工指定 `--dimensions`

### 1.3 入库入口

```bash
uv run bid-pipeline ingest-content-list --help
```

## 2. 常用环境变量

### 2.1 OpenAI / Embedding

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可选）
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_EMBEDDING_DIM`

### 2.2 策略相关

- `BID_SCORING_POLICY_PACK=cn_medical_v1`
- `BID_SCORING_POLICY_OVERLAY=strict_traceability`
- `BID_SCORING_POLICY_ARTIFACT=artifacts/policy/<pack>/<overlay>/runtime_policy.json`
- `BID_SCORING_PROD_QUALITY_MODE=fast|strict`

### 2.3 评分相关

- `BID_SCORING_HYBRID_PRIMARY_WEIGHT`
- `BID_SCORING_AGENT_MCP_MODEL`
- `BID_SCORING_AGENT_MCP_TOP_K`
- `BID_SCORING_AGENT_MCP_MODE`
- `BID_SCORING_AGENT_MCP_EXECUTION_MODE=tool-calling|bulk`
- `BID_SCORING_AGENT_MCP_MAX_TURNS`

### 2.4 MinerU 相关

- `MINERU_PDF_COMMAND`
- `MINERU_OUTPUT_ROOT`
- `MINERU_PDF_TIMEOUT_SECONDS`
- `MINERU_API_URL`
- `MINERU_API_KEY`

## 3. 策略编译与同步

### 3.1 编译策略产物

```bash
uv run python scripts/build_policy_artifacts.py \
  --pack cn_medical_v1 \
  --overlay strict_traceability
```

### 3.2 校验 skill/policy 同步

```bash
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
```

## 4. 门禁与评测

### 4.1 评分回归门禁

```bash
uv run python scripts/evaluate_scoring_backends.py \
  --fail-on-thresholds \
  --summary-out data/eval/scoring_compare/summary.json
```

### 4.2 混合检索评测（生成 summary）

```bash
uv run python scripts/evaluate_hybrid_search_gold.py \
  --version-id <VERSION_UUID> \
  --queries-file data/eval/hybrid_medical_synthetic/queries.json \
  --qrels-file data/eval/hybrid_medical_synthetic/qrels.source_id.A.jsonl \
  --thresholds-file data/eval/hybrid_medical_synthetic/retrieval_baseline.thresholds.json \
  --output data/eval/hybrid_medical_synthetic/eval_summary.json
```

### 4.3 策略驱动检索门禁

```bash
uv run python scripts/evaluate_retrieval_policy_gate.py \
  --summary-file data/eval/hybrid_medical_synthetic/eval_summary.json \
  --policy-pack cn_medical_v1 \
  --policy-overlay strict_traceability \
  --fail-on-violations
```

## 5. 常见排查

- `summary file not found`：先运行 `evaluate_hybrid_search_gold.py` 生成 summary。
- `missing_policy_pack_reference`：检查 `.claude/skills/bid-analyze/prompt.md` 是否仍引用策略 pack。
- `scoring_backend_agent_mcp_fallback`：说明 agent 后端降级到基线，需检查模型/网络/权限。
