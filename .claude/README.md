# Claude 团队目录说明

该目录是本项目在 Claude Code 中的本地插件工作区（投标审核）。

主要组成：
1. `agents/`：总控与专职子代理。
2. `commands/`：可复用的团队执行命令。
3. `hooks/`：运行期守卫（重点是 MCP 参数类型防错）。
4. `skills/`：评分口径与输出契约。
5. `.claude-plugin/plugin.json`：插件清单，用于校验与发布准备。
