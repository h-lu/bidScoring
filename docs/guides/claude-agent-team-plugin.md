# Claude Agent Team Plugin（投标审核自动化）

## 1. 目标

把本仓库 `.claude` 内容重构为可验证的标准 Claude Code plugin，包含：

1. `skills`：评分口径与证据优先规则
2. `agents`：agent team 分工协作
3. `hooks`：MCP 参数错误防呆与日志
4. `commands`：一键触发单文档/多文档审核

历史配置已归档到：
`/.claude_archive/2026-02-18-154426-pre-agent-team/`

## 2. 目录结构

```text
.claude/
  .claude-plugin/
    plugin.json
  agents/
    bid-team-orchestrator.md
    bid-team-evidence.md
    bid-team-scoring.md
    bid-team-traceability.md
  commands/
    bid-review-team.md
    bid-compare-team.md
  hooks/
    hooks.json
    scripts/
      session-start.sh
      mcp-param-guard.sh
  skills/
    bid-analyze/
      SKILL.md
      workflow.md
      rubric.md
      prompt.md
      examples.md
```

## 3. 安装与校验

在项目根目录执行：

```bash
claude plugin validate .claude/.claude-plugin/plugin.json
```

当前 CLI 主要通过 marketplace 安装插件，不支持直接用本地路径安装插件名。

本仓库采用方式：
1. `.claude` 目录直接用于本地会话（skills/agents/commands/hooks 可直接生效）。
2. `.claude/.claude-plugin/plugin.json` 用于“标准 plugin 清单”和发布前校验。

## 4. 使用方式

### 4.1 单文档自动审核（agent team）

在 Claude Code 中执行：

```text
/bid-review-team <version_id> [bidder_name] [project_name]
```

### 4.2 多投标方自动对比

```text
/bid-compare-team <version_id_A> <version_id_B> [version_id_C]
```

## 5. Hook 防错说明

`PostToolUseFailure` 钩子会捕获：

- `mcp__bid-scoring__get_page_metadata`
- `mcp__bid-scoring__search_chunks`
- `mcp__bid-scoring__get_chunk_with_context`

当检测到典型类型错误（例如把数组写成字符串）时，会输出修复提示并将原始事件写入：

`/.claude/logs/mcp-tool-failures.jsonl`

## 6. 官方参考

1. Subagents（agent team）  
   [https://code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)
2. Hooks  
   [https://code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)
3. Plugins  
   [https://code.claude.com/docs/en/plugins](https://code.claude.com/docs/en/plugins)
4. Plugins Reference  
   [https://code.claude.com/docs/en/plugins-reference/index](https://code.claude.com/docs/en/plugins-reference/index)

另见协作模式决策文档：
`docs/guides/claude-agent-team-collaboration-mode.md`
