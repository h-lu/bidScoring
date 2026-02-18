# Claude Skill 编写最佳实践（官方对齐版）

## 1. 结论先说

可以把 prompt 放在 skill 里，但推荐“分层组织”：

1. `SKILL.md` 只放用途、触发条件、执行约束、文件索引。
2. 复杂提示词放 `prompt.md`。
3. 评分口径放 `rubric.md`。
4. 执行步骤放 `workflow.md`。
5. 示例放 `examples.md`。

这样做的好处是：
- 便于维护：改规则不必改主 skill
- 便于扩展：后续新增 agent/mcp 工具时只改局部
- 便于审计：输出契约和策略更容易对齐门禁

## 2. 官方文档（建议固定收藏）

1. Skills（Claude Code）  
   [https://docs.claude.com/en/docs/claude-code/skills](https://docs.claude.com/en/docs/claude-code/skills)
2. Agent Skills Overview  
   [https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
3. Agent Skills Best Practices  
   [https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices)
4. Claude Code MCP  
   [https://docs.claude.com/en/docs/claude-code/mcp](https://docs.claude.com/en/docs/claude-code/mcp)

## 3. 本项目 skill 结构（bid-analyze）

目录：

```text
.claude/skills/bid-analyze/
  SKILL.md
  workflow.md
  rubric.md
  prompt.md
  examples.md
```

职责分工：
- `SKILL.md`：入口与硬约束
- `workflow.md`：先取证再评分的工具流程
- `rubric.md`：维度权重、加减分、风险分级
- `prompt.md`：可粘贴模板与 JSON 契约
- `examples.md`：典型输入输出示例

## 4. 质量门禁（必须保留）

策略单源：`config/policy/packs/<pack_id>/base.yaml`（按 overlay 叠加）  
一致性校验：

```bash
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
```

该门禁用于防止“策略已改、skill 模板未同步”。

## 5. 面向投标分析场景的额外约束

1. 结论必须绑定可定位证据：`chunk_id/page_idx/bbox`
2. 证据不足不拒答：写 warning，维度中性分 `50`
3. 禁止文档外知识补全事实
4. 输出必须为结构化 JSON，便于系统回收、回归对比、PDF 高亮
