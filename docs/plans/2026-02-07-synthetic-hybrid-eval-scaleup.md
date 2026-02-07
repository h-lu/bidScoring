# Synthetic Hybrid Eval (v2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the synthetic medical tender evaluation dataset to ~1000 ingestable chunks per version (A/B/C), keep realistic chunk type distribution, ensure DB import works, and rerun golden hybrid evaluation across versions.

**Architecture:** Move the generator logic into `bid_scoring/synthetic_eval/` (each file < 500 LOC). Keep `scripts/generate_synthetic_hybrid_eval_data.py` as a thin CLI/re-export wrapper used by tests. Generate page-interleaved MineRU-like `content_list` with deterministic fillers plus anchored evidence chunks used by qrels.

**Tech Stack:** Python 3, psycopg3/PostgreSQL, pgvector (existing), pytest, ruff.

---

### Task 1: Reproduce the Size Failure (RED)

**Files:**
- Test: `tests/test_synthetic_hybrid_eval_assets.py`

**Step 1: Run failing test**

Run: `uv run pytest tests/test_synthetic_hybrid_eval_assets.py::test_ingestable_chunk_count_is_close_to_real_docs -q`

Expected: FAIL, ingestable chunk count is `49 < 950`.

---

### Task 2: Implement v2 Content Builder (GREEN)

**Files:**
- Create: `bid_scoring/synthetic_eval/content_builder.py`
- Modify: `bid_scoring/synthetic_eval/__init__.py`

**Step 1: Implement page-interleaved generation**
- `page_count=120` (or more) with `header + page_number` per page
- Deterministic per-page fillers to reach:
  - `len(content_list) >= 1200`
  - `ingestable >= 950` (`type` not in `header/page_number/footer`)
- Preserve all anchors referenced by `bid_scoring/synthetic_eval/labels.py`.

**Step 2: Add minimal validation helpers**
- Count types, ensure required types appear.

**Step 3: Run tests**

Run: `uv run pytest tests/test_synthetic_hybrid_eval_assets.py -q`

Expected: PASS.

---

### Task 3: Refactor Generator Script Under 500 LOC

**Files:**
- Create: `bid_scoring/synthetic_eval/dataset.py`
- Modify: `scripts/generate_synthetic_hybrid_eval_data.py`

**Step 1: Move generation/write/validate to `dataset.py`**
- Implement `generate_all(output_dir)` and `validate(output_dir)` compatible with tests.
- Keep output files unchanged:
  - `content_list.synthetic_bidder_{A|B|C}.json`
  - `queries.json`
  - `qrels.source_id.{A|B|C}.jsonl`

**Step 2: Make `scripts/generate_synthetic_hybrid_eval_data.py` a thin wrapper**
- Re-export `generate_all`/`validate` for tests.
- Keep CLI behavior.

**Step 3: Run focused verification**

Run:
- `uv run ruff check bid_scoring/ tests/`
- `uv run pytest -q`

Expected: all green.

---

### Task 4: Regenerate Assets + Import to DB + Rebuild Embeddings

**Files:**
- Outputs: `data/eval/hybrid_medical_synthetic/`
- DB scripts: `scripts/ingest_mineru.py`, `scripts/build_all_embeddings.py`

**Step 1: Regenerate**

Run: `uv run python scripts/generate_synthetic_hybrid_eval_data.py --output-dir data/eval/hybrid_medical_synthetic --all-scenarios`

**Step 2: Import A/B/C**
- Reuse existing version_ids (if desired) to keep benchmark wiring stable.
- Verify counts:
  - `SELECT COUNT(*) FROM chunks WHERE version_id=...;` should be ~950-1100.

**Step 3: Build embeddings**

Run per version:
- `uv run python scripts/build_all_embeddings.py --version-id=<A> --skip-contextual --skip-hierarchical`

---

### Task 5: Rerun Golden Evaluation (Cross-Version Baseline)

**Files:**
- `scripts/evaluate_hybrid_search_multiversion.py`
- `data/eval/hybrid_medical_synthetic/multi_version_manifest.json`

**Step 1: Evaluate**

Run:
- `uv run python scripts/evaluate_hybrid_search_multiversion.py --version-map <json> --top-k 10 --output /tmp/hybrid_multiversion_baseline_v2.json`

Expected:
- Hybrid >= max(vector, keyword) on macro MRR / nDCG@5 for this dataset (not guaranteed but target).

