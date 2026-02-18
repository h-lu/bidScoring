# Claude Code Bid Analyze Playbook

## 1. 目标

在 Claude Code 中稳定执行“先取证后评分”的协作流程，并保证输出可追溯。

## 2. 前置条件

- MCP 检索服务可用
- `.claude/skills/bid-analyze` 已对齐当前 policy
- 目标 `version_id` 已入库且完成 embeddings

## 3. 推荐执行节奏

1. Orchestrator 明确输入和边界
2. Evidence agent 按维度取证
3. Scoring agent 输出结构化评分
4. Traceability agent 复核证据可定位
5. 汇总最终结果并给告警

## 4. 必查项

- 是否先调用了 `retrieve_dimension_evidence`
- 是否存在“无证据高分”
- `page_idx/bbox/chunk_id` 是否齐全
- 证据不足是否回落中性分并告警

## 5. 与 CLI 关系

- CLI 是稳定生产执行器
- Claude Team 是复杂复核执行器
- 两者应使用同一 policy 语义
