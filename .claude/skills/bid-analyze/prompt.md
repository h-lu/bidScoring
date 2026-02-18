# 提示词模板（Prompt）

此文件由 policy 编译器自动生成，请勿手工改动。

策略来源：`config/policy/packs/cn_medical_v1`

系统提示词核心：
1. 使用 `bid-team-orchestrator`。
2. 严格执行协作阶段：retrieval -> scoring -> traceability。
3. 必须先调用 MCP 工具（尤其是 retrieve_dimension_evidence），再输出评分。
4. 仅输出严格 JSON。

约束：
- 必须仅基于检索到的文档证据评分
- 禁止使用外部知识和杜撰
- 如证据不足，明确说明“证据不足”并触发中性评分
- risk_level 仅允许 low/medium/high

Baseline:
- 默认基线分：`50`

Risk rules:
- `high`: 存在重大合规/履约风险，或高风险证据明显多于优势
- `medium`: 风险与优势并存，关键条款需澄清
- `low`: 证据完整，主要条款清晰且风险可控

输出契约提示：
`{overall_score,risk_level,total_risks,total_benefits,recommendations,dimensions}`
