# 双轨架构与全流程说明（CLI 生产轨 + Claude Skill/MCP 轨）

## 1. 先给结论

当前系统是“双轨并行”：

1. `CLI 生产轨`：`bid-pipeline run-prod/run-e2e`，用于系统内稳定执行与可回归测试。
2. `Claude Skill + MCP 轨`：让 Claude Code 按 skill 先探索再评分，适合深度评审与策略迭代。

两条轨道都可用，但职责不同：

- 生产稳定性与自动化回归，优先 `CLI 生产轨`（当前默认 `hybrid`）。
- 探索式分析、交互式核验、策略快速迭代，优先 `Claude Skill + MCP 轨`。

---

## 2. 轨道 A：CLI 生产轨（系统内执行）

### 2.1 输入层

`run-prod` 只保留两类输入：

1. `--context-json <content_list.json>`
2. `--pdf-path <xxx.pdf>`

由 `AutoContentSource` 自动分流：

- `ContextListSource`：直接读取 `content_list.json`（跳过 MinerU，附加 `mineru_bypassed` 告警）。
- `PdfMinerUAdapter`：先调 MinerU（`auto|cli|api`），再读取产出的 `content_list.json`。

### 2.2 入库层（证据可定位核心）

`ingest_content_list` 做三件关键事情：

1. 写入 `chunks`（检索索引层，可重建）。
2. 写入 `content_units`（稳定证据层，保存 `anchor_json`）。
3. 建立 `chunk_unit_spans`（chunk 与 unit 的映射）。

为什么能高亮回原 PDF？

- 每条证据保留 `chunk_id + page_idx + bbox`
- `traceability` 会检查这些字段是否完整可定位。

### 2.3 评分层（3后端）

可选后端：

1. `analyzer`：规则/关键词打分（无 LLM）。
2. `agent-mcp`：LLM + 工具调用检索（默认 `tool-calling`）。
3. `hybrid`：融合 `agent-mcp(主)` + `analyzer(辅)`，默认主权重 `0.7`。

`hybrid` 的真实逻辑是：

1. 跑一遍 `agent-mcp` 结果（primary）。
2. 跑一遍 `analyzer` 结果（secondary）。
3. 按权重融合总分与维度分，并合并告警/证据。

它不是“把全部文档一次喂给 LLM 后直接结束”。

### 2.4 agent-mcp 执行模式

默认 `tool-calling`：

1. 先给模型：`version_id + bidder/project + 维度关键词`
2. 模型逐轮调用 `retrieve_dimension_evidence`
3. 系统执行工具、回填 tool message
4. 模型输出最终 JSON（overall + 每维度分）

可切换 `bulk`（一次性给证据再评分），但默认不是。

### 2.5 输出层（你最关心）

最终 JSON 包含：

1. `scoring`：分数、维度、证据引用（`evidence_citations`）
2. `traceability`：可追溯覆盖率、可高亮 chunk 列表
3. `observability`：耗时、后端、agent 工具调用轨迹

关键字段：

- `traceability.citation_coverage_ratio`
- `traceability.highlight_ready_chunk_ids`
- `observability.agent.execution_mode`
- `observability.agent.tool_call_count`
- `observability.agent.exploration_trace`

---

## 3. 轨道 B：Claude Skill + MCP 轨（Agent 交互执行）

### 3.1 目标

让 Claude Code 按 skill 执行“先检索再评分”，而不是直接总结。

对应文件：

- `/.claude/skills/bid-analyze/SKILL.md`
- `/.claude/skills/bid-analyze/prompt.md`

### 3.2 强约束（skill 约定）

1. 必须先调用 MCP 工具拿证据。
2. 每条结论要绑定 `chunk_id/page_idx/bbox`。
3. 证据不足时不拒绝：给 warning，维度中性分 `50`。
4. 不得杜撰，不得引入文档外事实。

### 3.3 典型工具链

常见流程：

1. `list_available_versions` / 使用用户给定 `version_id`
2. `get_document_outline`
3. 每维度循环：
   - `search_chunks`
   - `get_chunk_with_context`
   - `extract_key_value`
4. 多版本时 `compare_across_versions`
5. 输出结构化评分 JSON + warnings + evidence

### 3.4 与 CLI 轨的关系

这条轨和 `run-prod` 是并行的，不是替代关系：

1. `run-prod`：系统内执行器（可回归、可门禁、可批处理）。
2. `skill+mcp`：外部智能体执行器（探索更灵活，便于策略快速迭代）。

建议策略：

1. 日常生产：`run-prod`（默认 `hybrid`）。
2. 争议条款复核/深挖：Claude Skill + MCP。
3. 新策略定稿后，直接更新 `config/policy/packs/<pack_id>/base.yaml` 与 overlay，并重新编译策略产物。

---

## 4. 两轨对比（实践视角）

| 维度 | CLI 生产轨 | Claude Skill + MCP 轨 |
|---|---|---|
| 目标 | 稳定交付、可回归 | 探索分析、交互复核 |
| 执行方式 | 固化 pipeline | Agent 动态工具调用 |
| 可控性 | 高（参数和门禁固定） | 中高（受提示词质量影响） |
| 可解释性 | 高（统一输出结构） | 高（可见探索过程） |
| 成本/时延 | analyzer 快，agent/hybrid 较慢 | 通常较慢（多轮工具调用） |
| 适合场景 | 批量跑分、CI 回归 | 复杂问题深挖、策略迭代 |

---

## 5. 现在推荐怎么用

### 5.1 生产默认

1. 用 `run-prod --context-json` 或 `run-prod --pdf-path`
2. 默认后端 `hybrid`
3. 需要看 agent 探索过程时打开：
   - `--agent-trace-stdout`
   - `--agent-trace-save --agent-trace-dir ...`

### 5.2 深度评审

1. 在 Claude Code 中启用 `bid-scoring` MCP
2. 使用 `bid-analyze` skill 提示词
3. 让 Claude 逐维度取证并给结构化评分

---

## 6. 你怎么判断“真有 agent + MCP”

满足以下任意两条即可确认为真：

1. 结果里有 `observability.agent.execution_mode=tool-calling`
2. `tool_call_count > 0` 且 `tool_names` 包含 `retrieve_dimension_evidence`
3. `exploration_trace` 非空，含每轮工具调用记录
4. 运行时 stderr 出现 `agent_mcp_trace ...`
5. 存在 trace 文件，且 `trace_id` 与输出一致

---

## 7. 当前已知边界

1. `agent-mcp` 在不同样本上仍有分数波动，稳定性需继续优化。
2. `agent-mcp` 当前是“按维度评分”，不是“逐题逐问执行器”。
3. 若需完全 question-level 评分，需要单独落地“题目执行器层”。
