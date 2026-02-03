# Agent-Skill 投标评分系统设计（MCP Server + Postgres）

> 目标：在可追溯、可验证、可复算前提下，落地工程级投标评分系统。

## 0. 适用范围与前提
- 文档解析产物来自 MineRU 的 `*_content_list.json` 且结构稳定。  
- 评分规则由业务方提供并可结构化为“条件 → 分值区间”。  
- 默认落地路径：**MCP Server + Postgres（含 pgvector）**。

## 1. 目标与成功标准
系统必须满足以下成功标准：
- **证据链可追溯**：每条评分可回溯到 `page + bbox + 原文片段`，并入库审计。  
- **评分可复算**：同输入、同规则、同模型可重放并复现结果。  
- **检索稳定可控**：索引持久化，检索参数版本化，结果可复现。  
- **评测闭环**：离线评测集 + 回归测试 + 抽检形成质量闭环。  
- **工程可运营**：清晰目录、稳定部署、可观测与可回滚。  

## 2. 总体架构与数据流
系统按四条主线组织：离线索引、在线检索、评分编排、评测闭环。

**数据流**  
`文档解析 → 离线索引 → MCP 检索 → 评分 Skill → 引用验证 → 评分落库 → 评测回归`

**组件职责**  
- **离线索引**：文本标准化、分块、元数据、向量生成、tsvector 生成。  
- **MCP Server**：统一检索入口，只读、可审计、可复现。  
- **Agent Skills**：证据驱动评分、引用验证、结构化输出。  
- **评测闭环**：离线评测、回归测试、抽检与反馈。  

## 3. 数据模型与索引策略（Postgres + pgvector）
核心实体：`projects / documents / document_versions / chunks / scoring_runs / scoring_results / citations`。

**关键字段**  
- `chunks`: `page_idx`、`bbox`、`text_raw`、`text_hash`、`embedding`、`text_tsv`  
- `scoring_runs`: `rules_version`、`model`、`params_hash`、`started_at`  
- `scoring_results`: `dimension`、`score`、`max_score`、`reasoning`  
- `citations`: `source_id`、`chunk_id`、`cited_text`、`verified`  

**索引策略**  
- `chunks.text_tsv`：GIN  
- `chunks.embedding`：HNSW/IVFFLAT  
- `chunks(project_id, version_id, page_idx)`：复合索引  

**一致性要求**  
`source_hash`、`parser_version`、`rules_version` 必须落库，保证可复算。  

## 4. MCP Server 设计与检索策略
MCP Server 是唯一数据访问入口，强调只读、可复现、可审计。

**核心工具**  
- `search_chunks(query, document_id, top_k, filters)`  
- `get_document_info(document_id)`  
- `get_page_content(document_id, page_idx)`  

**检索策略**  
- 先 `project_id + version_id` 过滤，再做 BM25 + pgvector。  
- RRF/加权融合参数版本化。  
- `top_k`、权重、过滤条件写入 `scoring_runs`。  

## 5. Skill 设计与评分流程
Skill 层将“证据 → 评分 → 验证 → 入库”固化为确定性流程。

**核心技能**  
- `bid-scoring`：解析维度、调用检索、生成评分 JSON。  
- `citation-retriever`：封装 MCP 检索并构建 `[Source N]` 上下文。  
- `evidence-verifier`：引用验证与置信度标注。  

**单维度流程**  
1) 解析维度与版本  
2) 调用检索  
3) 构建引用上下文  
4) 生成评分 JSON  
5) 引用验证  
6) 结果落库  

**关键约束**  
- 评分必须带引用，无证据明确返回“未找到证据”。  
- 输出必须符合 JSON Schema。  
- 规则与模型版本必须记录到 `scoring_runs`。  

## 6. 评测闭环与质量保障
为避免“引用正确但评分偏差”，建立离线评测与回归闭环。

**离线评测集**  
- 标注样例包含：标准答案、参考证据、评分区间。  
- 参考轨迹评测（对照人工最佳检索与评分轨迹）。  

**自动回归**  
- 规则/模型/检索参数变更触发回归。  
- 指标建议：faithfulness、answer relevancy、context precision/recall。  
- 引入 LLM-as-judge 轨迹一致性检查。  

**人工抽检**  
- 抽检低置信度与高影响维度。  
- 抽检结果回流评测集。  

## 7. 评分编排与一致性校验
编排层负责并行评分、统一汇总、一致性校验与异常兜底。

**并行编排**  
- Subagent 并行评分，统一输出 Schema。  

**一致性校验**  
- 校验分值范围、规则版本一致、证据覆盖、引用可信度。  
- `review_required = true` 时总结果标注需复核。  

**异常兜底**  
- 检索失败或证据不足时输出保守分值并标注原因。  

## 8. 规则体系与版本管理
评分规则必须结构化、版本化、可回滚。

**规则结构**  
- `dimensions[].rules[]` 记录条件、分值区间、证据要求。  
- 每条规则带 `rule_id` 方便统计与追溯。  

**版本策略**  
- 规则文件版本化（如 `rules/v1.2.0.yaml`）。  
- `scoring_runs` 记录 `rules_version` 与 `rules_hash`。  

**回滚机制**  
- 允许按规则版本回放评分，保证历史可复现。  

## 9. 部署与运维
默认落地为 **MCP Server + Postgres**。

**部署形态**  
- Postgres（含 pgvector）  
- MCP Server（只读检索服务）  
- Agent 应用侧（技能与编排）  

**可观测性**  
- 监控检索耗时、命中率、评分成功率、验证失败率。  
- 运行日志记录 `run_id + model + rules_version + params_hash`。  

**容错与回滚**  
- MCP 超时降级（BM25-only）。  
- 规则与索引支持回滚。  

## 10. 实施路线与里程碑
**Phase 1：基础闭环（1–2 周）**  
- 落库核心表  
- MCP `search_chunks` + `get_document_info`  
- 单维度评分 + 引用验证 + 入库  

**Phase 2：并行评分（1 周）**  
- Subagent 并行  
- 汇总与一致性校验  

**Phase 3：评测闭环（1 周）**  
- 离线评测集 + 回归测试  
- 抽检机制与反馈回流  

**Phase 4：优化运营（持续）**  
- 缓存与增量索引  
- 异常兜底与成本优化  

## 11. 风险清单与对策
**检索偏差**  
- 对策：权重与参数版本化，离线回归验证。  

**评分漂移**  
- 对策：规则版本化与回滚机制。  

**引用可靠性**  
- 对策：引用验证分级（精确/部分/失败），部分匹配需复核。  

**评测缺失**  
- 对策：离线评测 + 回归 + 抽检闭环。  

**运营成本**  
- 对策：缓存、增量更新、冷热分层。  

## 12. 参考指标与评测工具（选用）
可参考的评测方向与工具：
- RAG 指标：faithfulness、context precision/recall、answer relevancy  
- 轨迹评测：LLM-as-judge 参考轨迹一致性  
- 工具调用：tool correctness/tool use 评估  

