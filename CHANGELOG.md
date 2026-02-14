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
