# 投标分析问题集设计（可维护/可扩展）

## 1. 目标与约束

- 目标：建立一套可长期维护的问题集，支持快速增删改，不需要改代码即可调整评标关注点。
- 核心约束：评标必须“基于事实”，每个问题答案都必须可追溯到原始 PDF 定位（`chunk_id + page_idx + bbox`）。
- 现状对齐：当前系统已有维度评分（`scoring_rules.yaml`）、关键词配置（`retrieval_config.yaml`）、证据可追溯检查（traceability/warnings）。

## 2. 设计方案对比

### 方案 A（推荐）：分层 YAML 问题库 + JSON Schema 校验

- 优点：
  - 业务同学可读可改（YAML）；
  - 通过 Schema 做结构门禁；
  - 便于按维度拆分文件，冲突低。
- 缺点：
  - 需要新增一层 loader/validator（一次性成本）。

### 方案 B：单一大 JSON 文件

- 优点：实现最快。
- 缺点：可读性差、多人协作冲突大、难做模块化版本管理。

### 方案 C：问题集入库（Postgres）

- 优点：可做后台编辑与审计。
- 缺点：实验阶段过重，变更成本高，不利于快速迭代。

结论：当前阶段采用方案 A，后续再平滑升级到方案 C（保留接口）。

## 3. 推荐目录结构

```text
/Users/wangxq/Documents/投标分析_kimi/config/question_bank/
  schema/
    question_bank.schema.json
  packs/
    cn_medical_v1/
      manifest.yaml
      dimensions/
        warranty.yaml
        delivery.yaml
        training.yaml
        financial.yaml
        technical.yaml
        compliance.yaml
      overlays/
        strict_traceability.yaml
        fast_eval.yaml
```

说明：
- `manifest.yaml`：问题包元数据、版本、默认策略。
- `dimensions/*.yaml`：每个维度独立维护问题。
- `overlays/*.yaml`：同一问题集的运行策略覆盖（比如严格证据、快速回归）。

## 4. 问题模型（最小可用字段）

每个问题建议字段：

- `id`：稳定 ID，例如 `WARRANTY_001`。
- `dimension`：必须映射到 `scoring_rules.yaml` 里的维度键。
- `question`：自然语言问题。
- `intent`：该问题要判断什么。
- `keywords`：检索关键词（可叠加 `retrieval_config.yaml` 同义词扩展）。
- `expected_answer_type`：`number | duration | boolean | enum | text`。
- `scoring_rule`：本问题的评分逻辑（阈值或枚举）。
- `evidence_requirements`：证据门禁。
  - `min_citations`
  - `require_page_idx`
  - `require_bbox`
  - `require_verified_status`
- `warning_policy`：不拒绝，只告警（符合你当前要求）。
- `status`：`active | deprecated`。

## 5. 评分与可追溯策略（事实优先）

每题执行后统一产出：

- `answer.value`：结构化答案。
- `answer.confidence`：置信度。
- `citations[]`：每条至少包含 `chunk_id/page_idx/bbox`。
- `warnings[]`：例如 `missing_bbox`、`no_evidence_citations`。

打分规则：
- 有效分数只基于“可验证证据”计算；
- 无证据或部分不可追溯时不拒绝，降置信并输出警告；
- 维度分 = 题目分按权重聚合；总分继续复用当前维度权重。

## 6. 与现有代码的接口衔接（SOLID）

新增 4 个职责单一组件：

- `QuestionBankRepository`：加载问题包（文件系统实现）。
- `QuestionBankValidator`：Schema + 业务约束校验。
- `QuestionSelector`：按维度/运行策略筛题。
- `QuestionScoringEngine`：执行题目评分并输出结构化结果。

接入点：
- `BidAnalyzer` 仍保留现有维度流程作为基线；
- 新增 question-driven 路径（可切换）；
- `agent-mcp/hybrid` 共用同一问题集输入，避免三套口径。

## 7. 变更与版本策略

- 问题包语义版本：`major.minor.patch`。
  - `major`：字段不兼容或评分口径大变；
  - `minor`：新增问题/新增 overlay；
  - `patch`：文案、关键词微调。
- 每题 ID 永不复用；删除题仅 `deprecated`。
- `manifest.yaml` 记录 `compatible_scoring_rules_version`。

## 8. 测试与门禁

最小测试集合：

- Schema 校验测试：问题包必须结构合法。
- 业务规则测试：ID 唯一、维度合法、权重和可配置。
- 可追溯测试：模拟输出缺 `bbox/page_idx` 时必须触发 warning。
- 回归测试：同一 `content_list` 下三种 backend 至少保证不退化到“无证据引用”。

## 9. 首批问题集建议（先小后大）

先做 12 题（每维度 2 题）：
- `warranty/delivery/training/financial/technical/compliance` 各 2 题。
- 每题都要求最少 1 条可定位 citation。
- 先覆盖高价值事实项：质保时长、响应 SLA、交付周期、付款节点、合规承诺。

## 10. 下一步实施建议

1. 先落地 `config/question_bank/packs/cn_medical_v1`（仅数据与 schema，不改核心算法）。
2. 加载器与校验器接入 CLI（增加 `--question-pack`，默认关闭）。
3. 再把 `agent-mcp/hybrid/analyzer` 统一到 question-driven 输入上。
