# Claude Agent Team 插件使用说明

## 1. 定位

`.claude/agents/*` 定义了团队分工（orchestrator/evidence/scoring/traceability），用于复杂分析场景。

## 2. 组件

- `bid-team-orchestrator.md`
- `bid-team-evidence.md`
- `bid-team-scoring.md`
- `bid-team-traceability.md`

## 3. 设计原则

- 单角色单职责
- 证据优先于结论
- 输出契约固定，避免自由文本漂移

## 4. 调优入口

- 策略调优：`config/policy/packs/*`
- 工作流调优：`.claude/skills/bid-analyze/workflow.md`
- 评分准则调优：`.claude/skills/bid-analyze/rubric.md`
