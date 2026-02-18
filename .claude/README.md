# .claude 目录说明

本目录用于定义 Agent Team 与 Skill，在 Claude Code 中执行证据优先评标流程。

## 1. 目录结构

- `agents/`：团队角色定义
- `skills/bid-analyze/`：主技能（流程、评分、样例、提示词）
- `commands/`：团队化命令入口

## 2. 设计原则

- 先检索再评分
- 只基于证据输出结论
- 输出必须可回溯到 `chunk_id/page_idx/bbox`

## 3. 与策略配置关系

`.claude/skills/bid-analyze/prompt.md` 必须与策略包保持同步：
- `config/policy/packs/cn_medical_v1/base.yaml`
- `config/policy/packs/cn_medical_v1/overlays/strict_traceability.yaml`

同步检查命令：

```bash
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
```
