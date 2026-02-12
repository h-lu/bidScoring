# 投标分析系统重构设计（Evidence-First）

日期：2026-02-12  
状态：已评审通过（用户确认）  
范围：PDF -> 解析 -> 入库 -> 检索 -> 引用 -> 评分 -> PDF定位渲染

## 1. 设计目标与边界

本次重构目标是“架构解耦、行为可验证、事实优先”，并且遵循以下约束：

- 当前处于实验阶段，数据库可清空重建。
- 不要求兼容旧脚本、旧接口、旧 schema。
- 一次性切换，不保留双轨运行。
- 评标结论必须基于可回溯证据，且可精确定位到 PDF 原始位置。

非目标：

- 本阶段不做性能极限优化。
- 本阶段不引入复杂分布式调度系统。
- 本阶段不保留历史迁移兼容层。

## 2. 事实优先架构（Evidence Spine）

系统采用“事实层”和“检索层”分离策略，构建不可断裂证据主链：

1. 源文件入库时生成不可变工件记录（`file_sha256`、`parser_version`、页信息）。
2. 每个可引用文本单元写入 `content_units`，并携带 `anchor_json`（至少含 `page_idx+bbox+coord_sys+source_element_id`）。
3. 为每个 unit 计算 `unit_hash`，用于后续引用校验。
4. 检索层仅负责召回，最终引用必须落到 `unit_id`。
5. 评分输出必须附 citations，且在落库前执行 `verify_citation`。
6. 任一证据不可验证（`unverifiable`）时，不阻断评分，但必须输出告警并显式标注证据问题。

核心门禁规则：

- `unverifiable evidence must be warned and explicitly marked`（强制）。

## 3. 数据流与 Schema 重整

新主流程固定为：

`ingest -> normalize -> index -> retrieve -> cite -> score -> render`

核心表职责：

- `source_artifacts`：源 PDF 工件元数据。
- `document_pages`：页面几何元数据。
- `content_units`：稳定证据层（事实主键层）。
- `chunks`：可重建索引层（衍生层）。
- `citations`：评分证据层，强制绑定 `unit_id`。

关键约束：

- 检索 API 禁止仅返回 `chunk_id`，必须返回 `unit_id` 证据链。
- `chunks` 可随时重建，不影响既有引用可验证性。
- 评分与渲染均以 `unit_id + anchor_json + quote span` 为准。
- 对 `unverifiable` 证据：保留评分流程，但在结果中输出 `warnings[]` 与 `evidence_status` 标识。
- 不引入人工复核通道；由系统自动标注并告警证据问题。

## 4. 模块拆分方案

新增目录：`bid_scoring/pipeline/`

- `domain/`：`ContentUnit`、`Anchor`、`Citation`、`EvidenceHash` 等领域模型与规则。
- `application/`：`PipelineService.run(request)`，编排 parse/normalize/persist/index。
- `infrastructure/`：`mineru_adapter`、`minio_store`、`postgres_repository`、`index_builder`。
- `interfaces/cli.py`：统一 CLI 入口（例如 `bid-pipeline ingest`）。

扩展接口预留（本阶段不实现）：

- 新增 `EvidenceLocator` 抽象接口，当前实现 `TextBBoxLocator`。
- 预留 `ImageRegionLocator` 扩展点，用于后续“图像片段证据”定位与渲染。

删除旧链路：

- `mineru/process_pdfs.py`
- `mineru/coordinator.py`
- `scripts/ingest_mineru.py`

测试门禁：

- 单测：hash 与 anchor 生成的确定性、引用校验规则。
- 集成：从 content_list 入库到 unit-level 引用的完整性。
- E2E：从 PDF 到定位渲染可回放；`unverifiable` 证据输出告警且标注状态。

## 5. 里程碑与原子提交计划

### M1: 建立新骨架与领域门禁

- 新建 `pipeline/domain` 与 `pipeline/application`。
- 落地 `CitationVerifier`，实现 `unverifiable` 告警标注（不拦截）。
- 新增对应单测。

建议提交：

- `refactor(pipeline): 引入evidence-first领域模型与编排骨架`
- `test(pipeline): 添加证据校验与门禁规则测试`

### M2: schema 重整与 repository 接入

- 新增破坏性 migration（重建核心表结构）。
- 实现 `postgres_repository`，按 unit-first 写入。
- 更新测试初始化脚本与夹具。

建议提交：

- `refactor(db): 重整evidence-first核心schema`
- `test(db): 更新unit-first入库与引用校验测试`

### M3: 解析/存储适配器与统一入口

- 实现 `mineru_adapter`、`minio_store`。
- 接入统一 CLI 命令 `bid-pipeline ingest`。
- 输出结构化 run summary（含 version_id、units_written、verified_rate）。

建议提交：

- `refactor(mineru): 解析与存储适配器接入新pipeline`
- `feat(cli): 添加统一bid-pipeline入口`

### M4: 检索与引用链切换

- 改造检索返回结构：强制返回 `unit_id` 证据链。
- 评分链路改造为接收全部 citations，但对不可验证项输出告警与状态标记。
- 更新 MCP 相关 tool 输出字段定义。

建议提交：

- `refactor(retrieval): 检索结果强制返回unit级证据链`
- `refactor(scoring): 评分输出不可验证证据告警与状态标记`

### M5: 清理旧链路与文档收敛

- 删除旧脚本/旧协调器实现。
- 更新使用文档与迁移说明。
- 全量跑通 lint + tests + e2e smoke。

建议提交：

- `chore(cleanup): 删除旧mineru入库链路实现`
- `docs(usage): 更新evidence-first新流程文档`

## 6. 技术原则（执行中保持）

- 单文件控制在 500 行以内。
- 数据事实主权在 `content_units`，不在 `chunks`。
- 所有结论必须“可定位、可验证、可复算”。
- 先本地验证再宣称完成。
