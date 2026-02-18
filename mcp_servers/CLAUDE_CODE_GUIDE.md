# Claude Code MCP 接入指南

## 1. 项目级接入（推荐）

```bash
claude mcp add-json bid-scoring '{
  "type":"stdio",
  "command":"uv",
  "args":["run","fastmcp","run","mcp_servers/retrieval_server.py","-t","stdio"],
  "env":{
    "DATABASE_URL":"${DATABASE_URL}",
    "OPENAI_API_KEY":"${OPENAI_API_KEY}"
  }
}' --scope project
```

## 2. 校验

```bash
claude mcp list
claude mcp get bid-scoring
```

## 3. 使用建议

- 将 `.mcp.json` 纳入版本控制
- 仅通过环境变量传递密钥
- 变更 MCP 后执行一次端到端小样本回归
