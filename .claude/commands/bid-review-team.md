# /bid-review-team

用途：执行完整团队评审流程（retrieval -> scoring -> traceability）。

执行规范：
1. 必须先触发 evidence 阶段
2. scoring 只能消费 evidence 输出
3. traceability 必须在最终输出前执行

输入建议：
- `version_id`
- `bidder_name`
- `project_name`
- `dimensions`（可选）
