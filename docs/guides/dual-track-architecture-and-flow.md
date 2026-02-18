# 双轨架构与全流程

## 1. 双轨定义

- 轨道 A（生产轨）：`CLI run-prod/run-e2e`
- 轨道 B（协作轨）：`Claude Skill + MCP`

两条轨并行，不互相替代。

## 2. 轨道 A：生产轨

### 输入
- `--context-json`
- `--pdf-path`

### 执行
- load
- ingest
- embeddings
- scoring
- traceability

### 输出
- `scoring`
- `traceability`
- `observability`

## 3. 轨道 B：协作轨

- Orchestrator 拆分 retrieval/scoring/traceability 任务
- Evidence agent 先调用检索工具
- Scoring agent 只基于证据评分
- Traceability agent 验证证据可定位

## 4. 为什么不是“全文直喂”

- 容易失去可追溯性与可审计性
- 不利于稳定回归和策略门禁
- 难以做细粒度维度优化

## 5. 实务建议

- 日常生产：`run-prod`（稳定、可回归）
- 复杂争议复核：`skill + MCP`（可探索）
- 两条轨输出统一回到证据引用格式
