# 投标分析系统（Evidence-First）

本项目是一个面向投标/评标场景的证据优先分析系统。

核心原则：
- 结论必须绑定证据
- 证据必须可定位（`chunk_id/page_idx/bbox`）
- 策略、评分准则、检索阈值全部可配置

## 1. 当前能力

- 端到端生产链路：`load -> ingest -> embeddings -> scoring -> traceability`
- 三种评分后端：`analyzer`、`agent-mcp`、`hybrid`（生产默认）
- 策略系统：`policy pack + overlay + runtime artifact`
- 门禁体系：
  - 单测
  - skill/policy 一致性
  - 评分回归门禁
  - 检索策略阈值门禁

## 2. 架构总览

- 执行入口：`bid-pipeline run-prod`
- 高级入口：`bid-pipeline run-e2e`
- 策略来源：`config/policy/packs/<pack_id>`
- 评分默认策略：`cn_medical_v1 + strict_traceability`

## 3. 快速开始

### 3.1 安装依赖

```bash
uv sync --locked --dev
```

### 3.2 环境变量（最小集合）

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_EMBEDDING_DIM=1536`

### 3.3 初始化数据库

```bash
uv run python scripts/apply_migrations.py
```

### 3.4 生产入口（推荐）

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

## 4. 策略配置（可调、可扩展）

策略目录：
- `config/policy/packs/cn_medical_v1/manifest.yaml`
- `config/policy/packs/cn_medical_v1/base.yaml`
- `config/policy/packs/cn_medical_v1/overlays/*.yaml`

关键环境变量：
- `BID_SCORING_POLICY_PACK`
- `BID_SCORING_POLICY_OVERLAY`
- `BID_SCORING_POLICY_ARTIFACT`（可选）

编译策略产物：

```bash
uv run python scripts/build_policy_artifacts.py \
  --pack cn_medical_v1 \
  --overlay strict_traceability
```

## 5. 质量门禁命令

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
uv run python scripts/evaluate_scoring_backends.py --fail-on-thresholds
uv run python scripts/evaluate_retrieval_policy_gate.py \
  --summary-file data/eval/hybrid_medical_synthetic/eval_summary.json \
  --policy-pack cn_medical_v1 \
  --policy-overlay strict_traceability \
  --fail-on-violations
```

## 6. 输出结果（核心字段）

- `scoring`：总分、维度分、证据引用
- `traceability`：可追溯覆盖率、可高亮 chunk 列表
- `observability`：后端、耗时、问题集信息

## 7. 文档导航

- `docs/usage.md`：完整命令与参数
- `docs/guides/project-handbook.md`：项目全景与路线图
- `docs/guides/dual-track-architecture-and-flow.md`：双轨架构说明
- `.claude/README.md`：Agent Team/Skill 使用指南

## 8. 当前状态

- 已完成：policy-as-config 重构与检索门禁接线
- 当前默认生产策略：`cn_medical_v1 + strict_traceability`
- 建议下一步：基于真实业务样本扩大评测集并固化 CI 门禁
