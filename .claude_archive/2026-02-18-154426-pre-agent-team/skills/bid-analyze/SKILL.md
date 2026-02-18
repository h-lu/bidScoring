---
name: bid-analyze
description: Evaluates bidding documents with evidence-first scoring and PDF-traceable citations. Use when users ask for bidder scoring, comparison, risk review, or recommendation ranking.
---

# Bid Analyze Skill

## Goal
先取证，再评分。所有结论必须可追溯到 PDF 原始位置。

## Use When
- 需要对单个 `version_id` 做投标评分
- 需要比较多个投标方并给出排序建议
- 需要解释“为什么这个分数成立”
- 需要确保结论可定位高亮回原始 PDF

## Non-Negotiable Rules
1. 必须先调用 MCP 工具拿证据，再给分。
2. 每条证据必须包含 `chunk_id/page_idx/bbox`，缺一不可。
3. 证据不足时不拒答：写入 `warnings`，该维度给中性分 `50`。
4. 不得使用文档外知识或杜撰内容。
5. 输出必须是结构化 JSON，方便系统回收和审计。
6. MCP 调用参数必须严格匹配类型（数组不能用字符串包裹）。

## Runbook
1. 先读 `workflow.md`，按固定流程执行工具探索。
2. 再读 `rubric.md`，按统一口径计算维度分与总分。
3. 用 `prompt.md` 作为可粘贴模板（系统提示 + 输出契约）。
4. 输出格式偏离时，对照 `examples.md` 修正后再提交结果。

## Output Contract
顶层必须包含：
- `overall_score`
- `risk_level`
- `total_risks`
- `total_benefits`
- `recommendations`
- `dimensions`
- `warnings`

每个 `dimensions[i]` 必须包含：
- `key`
- `score`
- `risk_level`
- `reasoning`
- `evidence[]`（每条带 `version_id/chunk_id/page_idx/bbox/quote`）
- `warnings[]`

## Maintenance
- 评分策略单源：`config/agent_scoring_policy.yaml`
- 策略模板：`prompt.md`（需与策略单源保持同步）
- 一致性校验：`uv run python scripts/check_skill_policy_sync.py --fail-on-violations`
