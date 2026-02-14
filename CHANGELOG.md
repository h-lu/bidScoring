# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Evidence-first pipeline package:
  - `bid_scoring/pipeline/domain`
  - `bid_scoring/pipeline/application`
  - `bid_scoring/pipeline/infrastructure`
  - `bid_scoring/pipeline/interfaces`
- Unified CLI entrypoint: `bid-pipeline ingest-content-list`.
- Retrieval evaluation threshold gate:
  - `bid_scoring/retrieval/evaluation_gate.py`
  - `scripts/evaluate_hybrid_search_gold.py --thresholds-file --fail-on-thresholds`
- MCP traceability gate to prepare safe highlight targets:
  - `prepare_highlight_targets`
- New unit test baseline under `tests/unit/**`.
- E2E traceability summary contract in CLI output:
  - `traceability.status`
  - `traceability.citation_coverage_ratio`
  - `traceability.highlight_ready_chunk_ids`
- E2E observability timings in CLI output:
  - `observability.timings_ms.load|ingest|embeddings|scoring|total`
- Scoring backend regression gate script:
  - `scripts/evaluate_scoring_backends.py`
  - `data/eval/scoring_compare/content_list.minimal.json`
  - `data/eval/scoring_compare/thresholds.json`
- Simplified production CLI entrypoint:
  - `bid-pipeline run-prod`
  - fixed inputs: `--context-json` or `--pdf-path`
  - fixed defaults: `hybrid` backend + `cn_medical_v1/strict_traceability`
- Scoring run comparison tool:
  - `scripts/compare_scoring_runs.py`
  - compares baseline/candidate outputs and reports metric deltas + warning diffs
- First archived real run output:
  - `data/eval/scoring_compare/runs/2026-02-14-run-prod-hybrid-synthetic-bidder-A.json`
- Agent/MCP tool-calling test coverage:
  - `tests/unit/pipeline/test_scoring_agent_tool_loop.py`

### Changed
- Retrieval MCP server refactored into modular operations:
  - discovery / search / evidence / annotation / resources / analysis
- Retrieval output contract now emphasizes evidence fields:
  - `evidence_status`
  - `evidence_units`
  - `warnings`
- Hybrid retrieval behavior enhanced:
  - dynamic weights by query type
  - keyword fulltext path strategy (with fallback)
- Database schema evolved for evidence-first flow (`migrations/000_init.sql`).
- Documentation updated in `docs/usage.md` and MCP docs.
- `run-e2e` default scoring backend changed to `hybrid`.
- CI workflow now runs scoring backend regression gate.
- `agent-mcp` 默认模型调整为 `gpt-5-mini`（可通过 `BID_SCORING_AGENT_MCP_MODEL` 覆盖）。
- `agent-mcp` 提示词约束增强（仅基于证据、证据不足需明确说明）。
- `agent-mcp` 在维度无可验证证据时改为中性评分并追加告警（warning-first）。
- `agent-mcp` 执行路径升级为默认 `tool-calling`（保留 `bulk` 回退）：
  - 新增 `BID_SCORING_AGENT_MCP_EXECUTION_MODE` 与 `BID_SCORING_AGENT_MCP_MAX_TURNS`
  - 引入 `scoring_agent_support.py` 与 `scoring_agent_tool_loop.py`，拆分证据归一化与工具循环逻辑

### Removed
- Legacy MinerU coordinator chain:
  - `mineru/process_pdfs.py`
  - `mineru/coordinator.py`
  - `scripts/ingest_mineru.py`
- Legacy ad-hoc test scripts:
  - `scripts/test_mcp_new_tools.py`
  - `scripts/test_mcp_quick.py`
  - `scripts/test_real_retrieval.py`
- Deprecated historical docs and reports replaced by root `README.md` + this changelog.

### Notes
- This refactor is designed for the experimental stage and may include breaking changes.
- Compatibility with legacy schema/scripts is intentionally not preserved.
