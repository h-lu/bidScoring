---
description: "通过强制多代理协作执行多投标方对比审核"
argument-hint: "<version_id_A> <version_id_B> [version_id_C ...]"
---

# 投标审核团队（多文档对比）

本任务使用 `bid-team-orchestrator`：

`$ARGUMENTS`

执行要求：
1. 至少提供两个 `version_id`。
2. 每个投标方使用同一评分口径完成取证与评分。
3. 排序前必须执行跨版本追溯审计。
4. 最终仅返回单个严格 JSON 对象。

JSON 最低字段要求：
- `ranking`
- `bidders`
- `cross_bid_findings`
- `warnings`
