# 提示词模板（Prompt）

策略单源文件：`config/agent_scoring_policy.yaml`

系统提示词核心：
1. 使用 `bid-team-orchestrator`。
2. 严格执行协作阶段：
   - retrieval
   - scoring
   - traceability
3. 必须先调用 MCP 工具（尤其是 `retrieve_dimension_evidence`），再输出评分。
4. 仅输出严格 JSON。

Baseline:
- 默认基线分：`50`

Risk rules:
- `high`: 存在重大合规/履约风险，或高风险证据明显多于优势
- `medium`: 风险与优势并存，关键条款需澄清
- `low`: 证据完整，主要条款清晰且风险可控

输出契约提示：
`{overall_score,risk_level,total_risks,total_benefits,recommendations,dimensions}`
