# Evidence-First Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver the full evidence-first pipeline refactor with isolated worktree execution, complete test-suite rewrite, and end-to-end verification.

**Architecture:** Replace fragmented ingestion/orchestration with a layered `pipeline` package (`domain`, `application`, `infrastructure`, `interfaces`). Enforce unit-level evidence chain in retrieval and scoring outputs, with warning-only handling for unverifiable evidence. Remove legacy MinerU coordinator scripts and switch to a unified CLI entrypoint.

**Tech Stack:** Python 3.11+, psycopg3, FastMCP, MinIO SDK, pytest, ruff.

---

### Task 1: Rewrite Test Suite Baseline

**Files:**
- Delete: `tests/**` (legacy suite)
- Create: `tests/conftest.py`
- Create: `tests/unit/pipeline/test_citation_verifier.py`
- Create: `tests/unit/pipeline/test_pipeline_service.py`
- Create: `tests/unit/pipeline/test_pipeline_cli.py`
- Create: `tests/unit/retrieval/test_result_format.py`
- Create: `tests/unit/mcp/test_retrieve_contract.py`

**Step 1: Write failing tests first for new contracts**
- warning-only citation policy
- mandatory `unit_id` evidence chain in retrieve output
- unified CLI parameter contract

**Step 2: Run targeted tests to confirm RED**
- `uv run pytest -q tests/unit/pipeline tests/unit/retrieval tests/unit/mcp`

**Step 3: Add minimal scaffolding to satisfy imports**

**Step 4: Re-run tests to confirm controlled failures (behavioral)**

**Step 5: Commit test rewrite baseline**
- `test(tests): 重建evidence-first测试基线`

### Task 2: Pipeline Core Implementation

**Files:**
- Modify: `bid_scoring/pipeline/domain/models.py`
- Modify: `bid_scoring/pipeline/domain/verification.py`
- Modify: `bid_scoring/pipeline/application/service.py`
- Create: `bid_scoring/pipeline/infrastructure/postgres_repository.py`
- Create: `bid_scoring/pipeline/infrastructure/mineru_adapter.py`
- Create: `bid_scoring/pipeline/infrastructure/minio_store.py`
- Create: `bid_scoring/pipeline/infrastructure/index_builder.py`
- Create: `bid_scoring/pipeline/interfaces/cli.py`

**Step 1: Expand failing tests for repository/adapters contracts**

**Step 2: Run RED for expanded tests**

**Step 3: Implement minimal pipeline flow**
- ingest content list
- persist source artifact metadata
- evaluate citations with warning status

**Step 4: Run tests GREEN**

**Step 5: Commit core pipeline**
- `feat(pipeline): 实现evidence-first分层pipeline与统一入口`

### Task 3: Retrieval + Analysis Evidence Contract

**Files:**
- Modify: `mcp_servers/retrieval_server.py`
- Modify: `mcp_servers/bid_analyzer.py`
- Modify: `bid_scoring/retrieval/fetch.py`

**Step 1: Add failing tests for retrieve output contract**
- each result includes `evidence_units`
- no evidence -> warning markers

**Step 2: Run RED**

**Step 3: Implement result formatting and warning propagation**

**Step 4: Run GREEN**

**Step 5: Commit retrieval/scoring contract**
- `refactor(retrieval): 强制unit证据链并输出不可验证告警`

### Task 4: Schema + Entrypoint + Cleanup

**Files:**
- Modify: `migrations/000_init.sql`
- Modify: `pyproject.toml`
- Modify: `docs/usage.md`
- Modify: `mineru/__init__.py`
- Delete: `mineru/process_pdfs.py`
- Delete: `mineru/coordinator.py`
- Delete: `scripts/ingest_mineru.py`

**Step 1: Add/adjust schema for evidence-first**
- `source_artifacts`
- citation evidence status fields

**Step 2: Wire new script entrypoint (`bid-pipeline`)**

**Step 3: Remove legacy MinerU coordinator chain**

**Step 4: Update docs**

**Step 5: Commit cleanup**
- `refactor(mineru): 切换至pipeline入口并移除旧链路`

### Task 5: Verification Gate

**Files:**
- Verify only

**Step 1: Run lint**
- `uv run ruff check .`

**Step 2: Run full new tests**
- `uv run pytest -q`

**Step 3: Run smoke command**
- `uv run python -m bid_scoring.pipeline.interfaces.cli --help`

**Step 4: Record outputs in final report**

**Step 5: Commit verification adjustments (if needed)**
- `chore(test): 收敛重构后验证与修复`

