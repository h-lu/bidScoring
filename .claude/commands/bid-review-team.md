---
description: "通过强制多代理协作执行单文档投标审核"
argument-hint: "<version_id> [bidder_name] [project_name]"
---

# 投标审核团队（单文档）

本任务使用 `bid-team-orchestrator`：

`$ARGUMENTS`

执行要求：
1. 如果缺少 `version_id`，先向用户索取。
2. 必须执行协作三阶段：
   - 阶段一：`bid-team-evidence`
   - 阶段二：`bid-team-scoring`
   - 阶段三：`bid-team-traceability`
3. 三阶段完成前不得返回最终结果。
4. 最终仅返回单个严格 JSON 对象。

JSON 最低字段要求：
- `version_id`
- `overall_score`
- `risk_level`
- `dimensions`
- `warnings`
