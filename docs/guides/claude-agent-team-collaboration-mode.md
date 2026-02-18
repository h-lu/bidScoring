# Claude Agent Team 协作模式（先定模式）

## 官方能力拆分

1. Subagents（单会话内多代理）
   - 适合：一条审核链路内分工协作
   - 文档：[https://code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)
2. Agent Teams（多会话协作）
   - 适合：多任务并行和长流程项目
   - 文档：[https://code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)

## 本项目确定模式（当前阶段）

采用 **Mode-A: Lead + Subagents（单会话）**。

原因：
1. 与现有 MCP 与评分链路耦合最小。
2. 易于控制证据一致性与输出契约。
3. 对实验阶段最稳，便于快速迭代。

## 协作拓扑

1. `bid-team-orchestrator`（总控）
2. `bid-team-evidence`（取证）
3. `bid-team-scoring`（评分）
4. `bid-team-traceability`（追溯审核）

执行顺序：
1. Retrieval
2. Scoring
3. Traceability
4. Merge

## 升级路径（后续）

若进入批量评审生产阶段，再升级 **Mode-B: Agent Teams（多会话）**：
1. Team Lead 会话：任务分发与汇总
2. Bidder 会话：按投标方并行取证评分
3. Audit 会话：统一追溯审计与排名复核
