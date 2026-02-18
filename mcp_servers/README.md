# MCP Servers

本目录提供检索服务 MCP 实现，供 `agent-mcp` 和 Claude Skill 调用。

## 1. 启动方式

```bash
uv run fastmcp run mcp_servers/retrieval_server.py -t stdio
```

## 2. 主要能力

- 文档片段检索（vector/keyword/hybrid）
- 维度证据提取
- 证据定位信息透传

## 3. 与主流程关系

- CLI 生产轨：作为 `agent-mcp` 的检索工具
- Claude 协作轨：作为 evidence agent 的核心工具

## 4. 质量要求

- 检索输出必须包含可追溯字段
- 异常必须返回 machine-readable warning code
