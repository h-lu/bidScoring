# Refactor/Cleanup Implementation Plan

> **For Codex:** execute in the existing worktree at `.worktrees/codex/v0.2-data-layer`.

**Goal:** Refactor the codebase to remove obvious redundancy and bring all repo files under the 500 LOC limit, without changing runtime behavior.

**Architecture:** Split large modules into small, focused modules; keep backward-compatible import paths via thin compatibility wrappers.

**Tech Stack:** Python 3, pytest, ruff, psycopg.

---

## Scope (what “cleanup” means in this PR)

- Split these oversized files into smaller modules/files while keeping behavior and public APIs stable:
  - `bid_scoring/hybrid_retrieval.py` (1515 LOC)
  - `bid_scoring/hichunk.py` (579 LOC)
  - `bid_scoring/synthetic_eval/labels.py` (509 LOC)
  - `scripts/build_hichunk_nodes.py` (720 LOC)
  - `scripts/build_all_embeddings.py` (636 LOC)
  - `tests/test_hybrid_retrieval.py` (1781 LOC)
  - `tests/test_build_hichunk_nodes.py` (744 LOC)
  - `tests/test_hichunk.py` (634 LOC)
  - `migrations/000_init.sql` (539 LOC)

- Keep import compatibility:
  - Existing imports like `from bid_scoring.hybrid_retrieval import HybridRetriever` must keep working.

- Keep verification green:
  - `uv run ruff check bid_scoring/ scripts/ tests/`
  - `uv run ruff format --check bid_scoring/ scripts/ tests/`
  - `uv run pytest -q`
  - Run `uv run python scripts/demo_v0_2_e2e.py --help` (sanity) and a full demo run at the end.

---

## Task 1: Split `bid_scoring/hybrid_retrieval.py` into a retrieval subpackage

**Files:**
- Create: `bid_scoring/retrieval/__init__.py`
- Create: `bid_scoring/retrieval/cache.py`
- Create: `bid_scoring/retrieval/config.py`
- Create: `bid_scoring/retrieval/types.py`
- Create: `bid_scoring/retrieval/rrf.py`
- Create: `bid_scoring/retrieval/rerankers.py`
- Create: `bid_scoring/retrieval/hybrid.py`
- Modify: `bid_scoring/hybrid_retrieval.py`

**Steps:**
1. Move pure types/constants/helpers first (`types.py`, `cache.py`, `config.py`, `rrf.py`).
2. Move reranker optional-dependency logic into `rerankers.py`.
3. Move `HybridRetriever` into `hybrid.py`, importing helpers from the other modules.
4. Convert `bid_scoring/hybrid_retrieval.py` into a thin re-export wrapper and keep all public names used in tests (`HybridRetriever`, `ReciprocalRankFusion`, `DEFAULT_RRF_K`, `LRUCache`, `HAS_CONNECTION_POOL`, `RetrievalResult`, `load_retrieval_config`, etc.).
5. Run: `uv run pytest -q tests/test_hybrid_retrieval.py::test_reciprocal_rank_fusion_basic`
   - Expected: PASS.
6. Commit: `refactor(retrieval): split hybrid_retrieval into focused modules`

---

## Task 2: Split `tests/test_hybrid_retrieval.py` into smaller test modules

**Files:**
- Modify: `tests/test_hybrid_retrieval.py` (split into multiple files)
- Create: `tests/test_hybrid_retrieval_rrf.py`
- Create: `tests/test_hybrid_retrieval_keywords.py`
- Create: `tests/test_hybrid_retrieval_config.py`
- Create: `tests/test_hybrid_retrieval_cache.py`
- Create: `tests/test_hybrid_retrieval_retriever.py`
- Optional: `tests/conftest.py` for shared fixtures (per pytest best practices)

**Steps:**
1. Split tests by topic; keep names so pytest discovery works (`test_*.py`).
2. Keep all tests identical (pure move) to minimize risk.
3. Run: `uv run pytest -q tests/test_hybrid_retrieval_rrf.py tests/test_hybrid_retrieval_cache.py`
   - Expected: PASS.
4. Commit: `refactor(tests): split hybrid retrieval tests`

---

## Task 3: Split `bid_scoring/hichunk.py` and related big tests/scripts

**Files:**
- Create: `bid_scoring/hichunk/__init__.py`
- Create: `bid_scoring/hichunk/model.py` (types/dataclasses)
- Create: `bid_scoring/hichunk/build.py` (core build logic)
- Modify: `bid_scoring/hichunk.py` (thin wrapper for backwards compatibility)
- Modify/Create test splits for `tests/test_hichunk.py`
- Refactor script: `scripts/build_hichunk_nodes.py` into a thin CLI wrapper that calls library code.

**Steps:**
1. Extract library logic from scripts into `bid_scoring/hichunk/build.py`.
2. Make scripts call the library; keep CLI flags stable.
3. Split `tests/test_hichunk.py` and `tests/test_build_hichunk_nodes.py` by topic.
4. Run: `uv run pytest -q tests/test_hichunk.py`
   - Expected: PASS.
5. Commit: `refactor(hichunk): split module and reuse library from scripts`

---

## Task 4: Reduce LOC for remaining oversized files

**Targets:**
- `scripts/build_all_embeddings.py` (split helpers to `scripts/_embeddings_helpers.py` or move to `bid_scoring/embeddings.py` if reusable)
- `bid_scoring/synthetic_eval/labels.py` (split constants or trim obvious redundancy)
- `migrations/000_init.sql` (trim redundant blank lines/comments to <500 LOC without changing schema)

**Steps:**
1. Make the smallest safe change that brings LOC under 500.
2. Run: `uv run pytest -q`
3. Commit(s): `refactor(...)` or `chore(...)` depending on change nature.

---

## Task 5: Full verification + demo

**Steps:**
1. Run:
   - `uv run ruff check bid_scoring/ scripts/ tests/`
   - `uv run ruff format --check bid_scoring/ scripts/ tests/`
   - `uv run pytest -q`
2. Run demo:
   - `uv run python scripts/demo_v0_2_e2e.py`
3. Commit: `chore: run final verification` (only if needed for small final tweaks)

