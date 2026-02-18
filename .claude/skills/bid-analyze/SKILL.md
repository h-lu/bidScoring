# bid-analyze Skill

## 1. Skill 目标

在投标文档分析中执行 evidence-first 流程：
- 先取证
- 再评分
- 最后追溯校验

## 2. 输入

- `version_id`
- 维度列表（可选）
- 项目上下文（`bidder_name`、`project_name`）

## 3. 输出

- `scoring_pack`（总分、维度分、风险、建议）
- `evidence_pack`（按维度证据）
- `traceability_pack`（可定位率、告警、可高亮 chunk）

## 4. 硬约束

- 必须先调用检索工具
- 禁止使用文档外知识
- 证据不足时给中性分并告警
- 风险等级只允许 `low/medium/high`

## 5. 执行流程

1. 按 `workflow.md` 先做 retrieval
2. 按 `rubric.md` 做评分
3. 按 traceability 规则做证据校验
4. 生成结构化 JSON 输出

## 6. 同步校验

```bash
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
```
